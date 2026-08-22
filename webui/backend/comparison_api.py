"""WebUI 03：真实 A/B 策略比较 API。

这个模块不修改旧 schedule_api.py / token_schedule_api.py 的默认 Mapping。
03 页面单独使用 scheduling/ 下已经验证过的 RuntimeIndex + Scheduler：

1. Mapping：同一个真实 Request，在两个 mapping.json 上分别运行；
2. Prefill：同一个真实 Prefill Batch，在同一个 Trace-aware Mapping 上切换调度策略；
3. Decode Optimality：同一个真实 Decode Token 的同一层，Greedy vs CP-SAT Optimal。

返回值同时保留 58 层周期、16-SC 汇总和选中层任务，供下一步 Timeline/SC 联动直接复用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import ExecutionRules
from scheduling.decode_optimal_solver import (
    DecodeOptimalSolverError,
    solve_decode_layer_optimal,
)
from scheduling.layer_scheduler import schedule_layer
from scheduling.prefill_scheduler import schedule_prefill_batch
from scheduling.prefill_scheduling_mode import (
    PREFILL_MODE_AGGRESSIVE_REUSE,
    PREFILL_SCHEDULING_MODES,
    normalize_prefill_scheduling_mode,
)
from scheduling.runtime_index import RuntimeIndex, load_runtime_index
from scheduling.token_scheduler import schedule_token
from webui.backend.request_api import _find_prefill_record, _trace_source
from webui.backend.trace_api import (
    NUM_MOE_LAYERS,
    TRACE_FIRST_MOE_LAYER,
    build_token,
    get_trace_file,
    inspect_segment,
    load_trace_json,
    validate_route,
)


router = APIRouter(
    prefix="/api/comparison",
    tags=["Strategy Comparison"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAPPING_MODES = (
    "round_robin",
    "least_loaded",
    "frequency_aware",
    "trace_aware",
)

MAPPING_LABELS = {
    "round_robin": "Round-Robin",
    "least_loaded": "Least-Loaded",
    "frequency_aware": "Frequency-aware",
    "trace_aware": "Trace-aware",
}

PAIRING_MODES = (
    "sequential",
    "random",
    "frequency_aware",
    "greedy",
    "trace_aware",
    "optimal",
)

PAIRING_LABELS = {
    "sequential": "Sequential",
    "random": "Random",
    "frequency_aware": "Frequency-aware",
    "greedy": "Coactivation Greedy",
    "trace_aware": "Greedy + Local Search",
    "optimal": "Optimal Matching",
}

PAIRING_PATHS = {
    mode: (
        PROJECT_ROOT
        / "results"
        / "experiments"
        / "pairing_baselines"
        / mode
        / "mapping.json"
    )
    for mode in PAIRING_MODES
}

MAPPING_PATHS = {
    mode: (
        PROJECT_ROOT
        / "results"
        / "experiments"
        / "mapping_baselines"
        / mode
        / "mapping.json"
    )
    for mode in MAPPING_MODES
}

# Trace-aware 正式 Mapping 还有一个 canonical 发布路径，作为兼容 fallback。
TRACE_AWARE_CANONICAL = (
    PROJECT_ROOT
    / "results"
    / "mappings"
    / "mapping_baseline_N4_H7168_W4096.json"
)

REFERENCE_PATHS = {
    "mapping": (
        PROJECT_ROOT
        / "results"
        / "experiments"
        / "mapping_baselines"
        / "mapping_comparison_summary.json"
    ),
    "prefill": (
        PROJECT_ROOT
        / "results"
        / "experiments"
        / "prefill_scheduler"
        / "prefill_scheduler_ablation_summary.json"
    ),
    "decode": (
        PROJECT_ROOT
        / "results"
        / "decode"
        / "decode_optimality_probe.json"
    ),
}


class ComparisonError(ValueError):
    pass


class DecodeSource(BaseModel):
    category: str
    filename: str
    segment_index: int = Field(ge=1)
    token_index: int = Field(default=0, ge=0)


class MappingComparisonRequest(BaseModel):
    phase: Literal["decode", "prefill"]
    mapping_a: str
    mapping_b: str
    decode_source: DecodeSource | None = None
    prefill_batch_id: int | None = Field(default=None, ge=0)
    selected_layer: int = Field(default=48, ge=0, le=57)


class PrefillComparisonRequest(BaseModel):
    batch_id: int = Field(ge=0)
    mode_a: str
    mode_b: str
    selected_layer: int = Field(default=48, ge=0, le=57)


class PairingComparisonRequest(BaseModel):
    phase: Literal["decode", "prefill"]
    pairing_a: str
    pairing_b: str
    decode_source: DecodeSource | None = None
    prefill_batch_id: int | None = Field(default=None, ge=0)
    selected_layer: int | None = Field(default=None, ge=0, le=57)


class DecodeOptimalityRequest(BaseModel):
    source: DecodeSource
    layer_id: int = Field(default=48, ge=0, le=57)
    time_limit_seconds: float = Field(default=5.0, gt=0.0, le=30.0)
    solver_workers: int = Field(default=8, ge=1, le=32)


# ============================================================
# JSON / Mapping helpers
# ============================================================


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ComparisonError(f"文件不存在：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"无法读取 JSON：{path}") from exc
    if not isinstance(data, dict):
        raise ComparisonError(f"JSON 最外层必须是 dict：{path}")
    return data


def _mapping_path(mode: str) -> Path:
    if mode not in MAPPING_MODES:
        raise ComparisonError(
            f"未知 Mapping：{mode!r}；可选={MAPPING_MODES}。"
        )

    path = MAPPING_PATHS[mode]
    if path.exists():
        return path

    if mode == "trace_aware" and TRACE_AWARE_CANONICAL.exists():
        return TRACE_AWARE_CANONICAL

    raise ComparisonError(
        "缺少该 Mapping 的正式结果文件："
        f"{path.relative_to(PROJECT_ROOT)}"
    )


_INDEX_CACHE: dict[str, tuple[int, RuntimeIndex]] = {}


def _runtime_index(mode: str) -> RuntimeIndex:
    """按 mapping.json 修改时间缓存 RuntimeIndex。"""
    path = _mapping_path(mode)
    mtime_ns = path.stat().st_mtime_ns
    cached = _INDEX_CACHE.get(mode)

    if cached is not None and cached[0] == mtime_ns:
        return cached[1]

    index = load_runtime_index(path)
    _INDEX_CACHE[mode] = (mtime_ns, index)
    return index


def _mapping_metadata(mode: str) -> dict[str, Any]:
    path = _mapping_path(mode)
    raw = _load_json(path)
    mapper = raw.get("subcube_mapping")
    pairing = raw.get("pairing")
    if not isinstance(mapper, dict):
        mapper = {}
    if not isinstance(pairing, dict):
        pairing = {}

    return {
        "id": mode,
        "label": MAPPING_LABELS[mode],
        "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "mapping_conflict_cost": mapper.get("total_conflict_cost"),
        "pre_conflict_cost": mapper.get("pre_conflict_cost"),
        "down_conflict_cost": mapper.get("down_conflict_cost"),
        "pairing_mode": pairing.get("mode"),
        "pairing_cost": pairing.get("total_routed_up_coactivation_cost"),
    }


# ============================================================
# Pairing baseline helpers
# ============================================================


def _pairing_path(mode: str) -> Path:
    if mode not in PAIRING_MODES:
        raise ComparisonError(
            f"未知 Pairing：{mode!r}；可选={PAIRING_MODES}。"
        )
    path = PAIRING_PATHS[mode]
    if not path.exists():
        raise ComparisonError(
            "缺少该 Pairing 的正式 Mapping："
            f"{path.relative_to(PROJECT_ROOT)}"
        )
    return path


_PAIRING_INDEX_CACHE: dict[str, tuple[int, RuntimeIndex]] = {}
_PAIRING_RAW_CACHE: dict[str, tuple[int, dict[str, Any]]] = {}


def _pairing_runtime_index(mode: str) -> RuntimeIndex:
    path = _pairing_path(mode)
    mtime_ns = path.stat().st_mtime_ns
    cached = _PAIRING_INDEX_CACHE.get(mode)
    if cached is not None and cached[0] == mtime_ns:
        return cached[1]
    index = load_runtime_index(path)
    _PAIRING_INDEX_CACHE[mode] = (mtime_ns, index)
    return index


def _pairing_raw(mode: str) -> dict[str, Any]:
    path = _pairing_path(mode)
    mtime_ns = path.stat().st_mtime_ns
    cached = _PAIRING_RAW_CACHE.get(mode)
    if cached is not None and cached[0] == mtime_ns:
        return cached[1]
    raw = _load_json(path)
    pairing = raw.get("pairing")
    mapper = raw.get("subcube_mapping")
    if not isinstance(pairing, dict) or pairing.get("mode") != mode:
        raise ComparisonError(
            f"{path.relative_to(PROJECT_ROOT)} 的 pairing.mode 不匹配。"
        )
    if not isinstance(mapper, dict) or mapper.get("mode") != "trace_aware":
        raise ComparisonError(
            "Pairing 实时对比要求所有方案固定使用 Trace-aware Mapping。"
        )
    _PAIRING_RAW_CACHE[mode] = (mtime_ns, raw)
    return raw


def _pairing_metadata(mode: str) -> dict[str, Any]:
    raw = _pairing_raw(mode)
    pairing = raw.get("pairing") or {}
    return {
        "id": mode,
        "label": PAIRING_LABELS[mode],
        "file": str(_pairing_path(mode).relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "mapping_mode": "trace_aware",
        "local_search_enabled": bool(pairing.get("local_search_enabled", False)),
        "local_search_rounds": pairing.get("local_search_rounds"),
    }


def _pairs_for_layer(mode: str, layer_id: int) -> tuple[tuple[int, int], ...]:
    raw = _pairing_raw(mode)
    pairing = raw.get("pairing") or {}
    layers = pairing.get("routed_up_pairs_by_layer")
    if not isinstance(layers, list):
        raise ComparisonError(f"Pairing {mode} 缺少 routed_up_pairs_by_layer。")

    record: Any = None
    if 0 <= layer_id < len(layers):
        candidate = layers[layer_id]
        if isinstance(candidate, dict) and int(candidate.get("layer_id", -1)) == layer_id:
            record = candidate
    if record is None:
        for item in layers:
            if isinstance(item, dict) and int(item.get("layer_id", -1)) == layer_id:
                record = item
                break
    if not isinstance(record, dict):
        raise ComparisonError(f"Pairing {mode} 找不到 Layer-{layer_id} 配对。")

    pairs = record.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 128:
        raise ComparisonError(f"Pairing {mode} Layer-{layer_id} 配对数量不是 128。")
    return tuple((int(pair[0]), int(pair[1])) for pair in pairs)


def _pairing_layer_view(
    *,
    mode: str,
    layer_id: int,
    token_routes: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    pairs = _pairs_for_layer(mode, layer_id)
    expert_hits = [0] * 256
    route_sets: list[set[int]] = []
    for route in token_routes:
        route_set = {int(x) for x in route if 0 <= int(x) < 256}
        route_sets.append(route_set)
        for expert_id in route_set:
            expert_hits[expert_id] += 1

    pair_rows: list[dict[str, Any]] = []
    collision_count = 0
    touched_plane_count = 0
    active_expert_count = sum(1 for value in expert_hits if value > 0)

    for expert_a, expert_b in pairs:
        hits_a = expert_hits[expert_a]
        hits_b = expert_hits[expert_b]
        co_hit_tokens = sum(
            1 for route_set in route_sets
            if expert_a in route_set and expert_b in route_set
        )
        touched = hits_a > 0 or hits_b > 0
        if touched:
            touched_plane_count += 1
        collision_count += co_hit_tokens
        if touched:
            pair_rows.append({
                "expert_a": expert_a,
                "expert_b": expert_b,
                "hits_a": hits_a,
                "hits_b": hits_b,
                "co_hit_tokens": co_hit_tokens,
                "both_active": hits_a > 0 and hits_b > 0,
            })

    pair_rows.sort(
        key=lambda row: (
            -int(row["co_hit_tokens"]),
            -(int(row["hits_a"]) + int(row["hits_b"])),
            int(row["expert_a"]),
            int(row["expert_b"]),
        )
    )

    return {
        "layer_id": layer_id,
        "active_expert_count": active_expert_count,
        "pair_collision_count": collision_count,
        "touched_up_planes": touched_plane_count,
        "pairs": pair_rows,
    }


def _pairing_views_all_layers(
    *,
    mode: str,
    routes_by_token: tuple[tuple[tuple[int, ...], ...], ...],
) -> list[dict[str, Any]]:
    return [
        _pairing_layer_view(
            mode=mode,
            layer_id=layer_id,
            token_routes=tuple(token[layer_id] for token in routes_by_token),
        )
        for layer_id in range(NUM_MOE_LAYERS)
    ]


def _suggest_pairing_layer(
    a_layers: list[dict[str, Any]],
    b_layers: list[dict[str, Any]],
) -> int:
    return max(
        range(NUM_MOE_LAYERS),
        key=lambda layer_id: (
            abs(
                int(a_layers[layer_id]["pair_collision_count"])
                - int(b_layers[layer_id]["pair_collision_count"])
            ),
            max(
                int(a_layers[layer_id]["pair_collision_count"]),
                int(b_layers[layer_id]["pair_collision_count"]),
            ),
            -layer_id,
        ),
    )


# ============================================================
# Real request helpers
# ============================================================


def _decode_routes(source: DecodeSource) -> tuple[tuple[int, ...], ...]:
    token = build_token(
        category=source.category,
        filename=source.filename,
        segment_index=source.segment_index,
        token_index=source.token_index,
    )
    layers = token.get("layers")
    if not isinstance(layers, list) or len(layers) != NUM_MOE_LAYERS:
        raise ComparisonError("Decode Token 没有完整 58 层 Route。")

    return tuple(
        tuple(int(x) for x in layer["routed_experts"])
        for layer in layers
    )


def _decode_source_public(source: DecodeSource) -> dict[str, Any]:
    return {
        "dataset": "Chinese-SimpleQA",
        "category": source.category,
        "filename": source.filename,
        "segment_index": source.segment_index,
        "token_index": source.token_index,
    }


def _prefill_routes(
    batch_id: int,
) -> tuple[tuple[tuple[int, ...], ...], dict[str, Any]]:
    record = _find_prefill_record(batch_id)
    category, filename, segment_index = _trace_source(record)

    if segment_index != 0:
        raise ComparisonError(
            f"Prefill Batch-{batch_id} 不是 segment0，当前 segment={segment_index}。"
        )

    path = get_trace_file(category=category, filename=filename)
    data = load_trace_json(path)
    if not 0 <= segment_index < len(data):
        raise ComparisonError("Prefill segment_index 越界。")

    segment = data[segment_index]
    token_count = inspect_segment(segment)
    if token_count is None:
        raise ComparisonError("Prefill segment 不完整。")

    routes_by_token: list[tuple[tuple[int, ...], ...]] = []

    for token_index in range(token_count):
        token_layers: list[tuple[int, ...]] = []
        for layer_id in range(NUM_MOE_LAYERS):
            trace_layer_id = TRACE_FIRST_MOE_LAYER + layer_id
            raw_route = segment[str(trace_layer_id)][token_index]
            token_layers.append(validate_route(raw_route))
        routes_by_token.append(tuple(token_layers))

    return (
        tuple(routes_by_token),
        {
            "dataset": "Chinese-SimpleQA",
            "batch_id": batch_id,
            "category": category,
            "filename": filename,
            "segment_index": segment_index,
            "input_tokens": token_count,
        },
    )


# ============================================================
# Serialization helpers
# ============================================================


def _matrix_short(name: str) -> str:
    value = str(name)
    if value in {"gate", "gate_proj"}:
        return "gate"
    if value in {"up", "up_proj"}:
        return "up"
    if value in {"down", "down_proj"}:
        return "down"
    return value


def _serialize_task(task: Any) -> dict[str, Any]:
    payload = {
        "expert_id": int(task.expert_id),
        "matrix_name": _matrix_short(task.matrix_name),
        "subcube_id": int(task.subcube_id),
        "cube_id": int(task.cube_id),
        "ready_time": int(task.ready_time),
        "start_cycle": int(task.dispatch_time),
        "compute_start_cycle": int(task.compute_start_time),
        "end_cycle": int(task.finish_time),
        "wait_cycles": int(task.wait_cycles),
        "activation_cycles": int(task.activation_cycles),
        "compute_cycles": int(task.compute_cycles),
        "switched": bool(task.switched_from_another_cube),
        "initial_activation": bool(task.is_initial_activation),
    }
    token_index = getattr(task, "token_index", None)
    if token_index is not None:
        payload["token_index"] = int(token_index)
    return payload


def _serialize_layer(layer_result: Any, *, include_tasks: bool = True) -> dict[str, Any]:
    critical = [
        int(stat.subcube_id)
        for stat in layer_result.subcube_stats
        if stat.task_count > 0 and stat.last_finish_time == layer_result.total_cycles
    ]

    payload: dict[str, Any] = {
        "layer_id": int(layer_result.layer_id),
        "total_cycles": int(layer_result.total_cycles),
        "task_count": int(layer_result.task_count),
        "switch_count": int(layer_result.switch_count),
        "initial_activation_count": int(layer_result.initial_activation_count),
        "wait_cycles": int(layer_result.wait_cycles),
        "max_task_wait_cycles": int(layer_result.max_task_wait_cycles),
        "critical_subcubes": critical,
        "subcubes": [
            {
                "subcube_id": int(stat.subcube_id),
                "task_count": int(stat.task_count),
                "busy_cycles": int(stat.busy_cycles),
                "switch_count": int(stat.switch_count),
                "wait_cycles": int(stat.wait_cycles),
                "last_finish_time": int(stat.last_finish_time),
            }
            for stat in layer_result.subcube_stats
        ],
    }

    if include_tasks:
        payload["tasks"] = [_serialize_task(task) for task in layer_result.tasks]

    return payload


def _aggregate_subcubes(layer_results: list[Any], num_subcubes: int) -> list[dict[str, int]]:
    rows = [
        {
            "subcube_id": sc,
            "task_count": 0,
            "busy_cycles": 0,
            "switch_count": 0,
            "wait_cycles": 0,
            "critical_layer_count": 0,
        }
        for sc in range(num_subcubes)
    ]

    for lr in layer_results:
        for stat in lr.subcube_stats:
            sc = int(stat.subcube_id)
            row = rows[sc]
            row["task_count"] += int(stat.task_count)
            row["busy_cycles"] += int(stat.busy_cycles)
            row["switch_count"] += int(stat.switch_count)
            row["wait_cycles"] += int(stat.wait_cycles)
            if stat.task_count > 0 and stat.last_finish_time == lr.total_cycles:
                row["critical_layer_count"] += 1

    return rows


def _serialize_token_result(result: Any, *, selected_layer: int) -> dict[str, Any]:
    layer_results = [execution.layer_result for execution in result.layers]
    return {
        "total_cycles": int(result.total_cycles),
        "total_tasks": int(result.total_tasks),
        "total_switches": int(result.total_switches),
        "total_wait_cycles": int(result.total_wait_cycles),
        "max_layer_cycles": int(result.max_layer_cycles),
        "average_layer_cycles": float(result.average_layer_cycles),
        "layer_cycles": [int(execution.cycles) for execution in result.layers],
        "subcubes": _aggregate_subcubes(
            layer_results,
            len(result.final_active_cube_by_subcube),
        ),
        "selected_layer": _serialize_layer(
            result.layer(selected_layer).layer_result,
            include_tasks=True,
        ),
    }


def _serialize_prefill_result(result: Any, *, selected_layer: int) -> dict[str, Any]:
    layer_results = [execution.layer_result for execution in result.layers]
    return {
        "total_cycles": int(result.total_cycles),
        "input_tokens": int(result.token_count),
        "cycles_per_input_token": float(result.cycles_per_input_token),
        "total_tasks": int(result.total_tasks),
        "total_switches": int(result.total_switches),
        "total_wait_cycles": int(result.total_wait_cycles),
        "max_layer_cycles": int(result.max_layer_cycles),
        "average_layer_cycles": float(result.average_layer_cycles),
        "layer_cycles": [int(execution.cycles) for execution in result.layers],
        "subcubes": _aggregate_subcubes(
            layer_results,
            len(result.final_active_cube_by_subcube),
        ),
        "selected_layer": _serialize_layer(
            result.layer(selected_layer).layer_result,
            include_tasks=True,
        ),
    }


def _serialize_cp_sat(cp: Any) -> dict[str, Any]:
    # CP-SAT task 本身不保存 ready_time；为了让 WebUI 能与 Greedy
    # 使用同一套 Timeline / Waiting 状态逻辑，这里按依赖关系恢复：
    # gate/up 在 t=0 ready；down 在本 Expert 的 gate/up 都完成后 ready。
    finish_by_key = {
        (int(task.expert_id), _matrix_short(task.matrix_name)): int(task.finish_time)
        for task in cp.tasks
    }

    tasks: list[dict[str, Any]] = []
    for task in cp.tasks:
        matrix_name = _matrix_short(task.matrix_name)
        expert_id = int(task.expert_id)
        start_cycle = int(task.start_time)
        end_cycle = int(task.finish_time)

        if matrix_name == "down":
            ready_time = max(
                finish_by_key.get((expert_id, "gate"), 0),
                finish_by_key.get((expert_id, "up"), 0),
            )
        else:
            ready_time = 0

        tasks.append(
            {
                "expert_id": expert_id,
                "matrix_name": matrix_name,
                "subcube_id": int(task.subcube_id),
                "cube_id": int(task.cube_id),
                "ready_time": int(ready_time),
                "start_cycle": start_cycle,
                "compute_start_cycle": start_cycle + 1,
                "end_cycle": end_cycle,
                "wait_cycles": max(0, start_cycle - int(ready_time)),
                "activation_cycles": 1,
                "compute_cycles": 1,
                "switched": True,
                "initial_activation": False,
            }
        )

    return {
        "status": cp.status,
        "proven_optimal": bool(cp.proven_optimal),
        "feasible": bool(cp.feasible),
        "total_cycles": (
            None if cp.objective_cycles is None else int(cp.objective_cycles)
        ),
        "best_bound_cycles": cp.best_bound_cycles,
        "wall_time_seconds": float(cp.wall_time_seconds),
        "branches": int(cp.branches),
        "conflicts": int(cp.conflicts),
        "tasks": tasks,
    }


def _greedy_hint(greedy: Any) -> dict[tuple[int, str], int]:
    return {
        (task.expert_id, task.matrix_name): int(task.dispatch_time)
        for task in greedy.tasks
    }


# ============================================================
# Formal references (small, sanitized)
# ============================================================


def _phase_metrics(path: Path) -> dict[str, Any] | None:
    """读取一个正式 phase summary，只保留 WebUI 需要的指标。"""
    if not path.exists():
        return None
    raw = _load_json(path)
    prefill = raw.get("prefill")
    decode = raw.get("decode")
    if not isinstance(prefill, dict) or not isinstance(decode, dict):
        return None

    latency = prefill.get("latency_cycles") or {}
    cpt = prefill.get("cycles_per_input_token") or {}
    decode_cycles = decode.get("cycles_per_token") or {}

    return {
        "prefill_mean": latency.get("mean"),
        "prefill_p95": latency.get("p95"),
        "prefill_cycles_per_input_token": cpt.get("mean"),
        "decode_mean": decode_cycles.get("mean"),
        "decode_p95": decode_cycles.get("p95"),
    }


def _formal_ablation_payload() -> list[dict[str, Any]]:
    """
    2x2 正式消融从正式实验目录直接拼出，避免读取历史旧版 ablation_summary。

    Pairing Only = Trace-aware+LS Pairing + Round-Robin Mapping
    Mapping Only = Sequential Pairing + Trace-aware Mapping
    Full         = Trace-aware+LS Pairing + Trace-aware Mapping
    """
    exp = PROJECT_ROOT / "results" / "experiments"
    candidates = [
        (
            "Naive",
            "Sequential",
            "Round-Robin",
            exp / "ablation_formal" / "naive" / "phase_evaluation_summary.json",
        ),
        (
            "Pairing Only",
            "Trace-aware + LS",
            "Round-Robin",
            exp / "mapping_baselines" / "round_robin" / "phase_evaluation_summary.json",
        ),
        (
            "Mapping Only",
            "Sequential",
            "Trace-aware",
            exp / "pairing_baselines" / "sequential" / "phase_evaluation_summary.json",
        ),
        (
            "Full",
            "Trace-aware + LS",
            "Trace-aware",
            exp / "mapping_baselines" / "trace_aware" / "phase_evaluation_summary.json",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for name, pairing, mapping, path in candidates:
        metrics = _phase_metrics(path)
        if metrics is None:
            continue
        rows.append(
            {
                "experiment": name,
                "pairing": pairing,
                "mapping": mapping,
                **metrics,
            }
        )
    return rows


def _replication_payload() -> dict[str, Any] | None:
    oracle_path = (
        PROJECT_ROOT
        / "results"
        / "replication"
        / "prefill_replication_oracle_probe.json"
    )
    plan_path = (
        PROJECT_ROOT
        / "results"
        / "replication"
        / "replication_plan.json"
    )
    if not oracle_path.exists():
        return None

    oracle = _load_json(oracle_path)
    comparison = oracle.get("comparison")
    if not isinstance(comparison, dict):
        comparison = {}

    hardware: dict[str, Any] = {}
    replicas: list[dict[str, Any]] = []
    if plan_path.exists():
        plan = _load_json(plan_path)
        if isinstance(plan.get("hardware"), dict):
            hardware = {
                key: plan["hardware"].get(key)
                for key in (
                    "empty_plane_slots_before",
                    "replica_planes_used",
                    "empty_plane_slots_after",
                )
            }
        if isinstance(plan.get("replicas"), list):
            replicas = [
                {
                    "layer_id": row.get("layer_id"),
                    "expert_id": row.get("expert_id"),
                }
                for row in plan["replicas"]
                if isinstance(row, dict)
            ]

    return {
        "diagnostic_only": bool(oracle.get("diagnostic_only", True)),
        "comparison": {
            key: comparison.get(key)
            for key in (
                "baseline_mean",
                "balanced_all_mean",
                "oracle_mean",
                "balanced_all_improvement_percent",
                "oracle_improvement_percent",
                "oracle_improved_batches",
                "oracle_equal_batches",
                "oracle_total_saved_cycles",
                "oracle_max_saved_cycles_per_batch",
            )
        },
        "hardware": hardware,
        "replicas": replicas,
    }


def _reference_payload() -> dict[str, Any]:
    result: dict[str, Any] = {}

    mapping = _load_json(REFERENCE_PATHS["mapping"])
    result["mapping"] = {
        "protocol": mapping.get("protocol", {}),
        "metrics": mapping.get("metrics", []),
        "improvements_vs_round_robin": mapping.get(
            "improvements_vs_round_robin", []
        ),
    }

    prefill = _load_json(REFERENCE_PATHS["prefill"])
    result["prefill"] = {
        "protocol": prefill.get("protocol"),
        "modes": prefill.get("modes", []),
        "results": {
            mode: {
                key: value
                for key, value in metrics.items()
                if key
                in {
                    "prefill_mean_cycles",
                    "prefill_p95_cycles",
                    "mean_cycles_per_input_token",
                    "mean_switches_per_batch",
                    "batch_count",
                    "total_input_tokens",
                    "improvement_vs_no_reuse",
                    "buckets",
                }
            }
            for mode, metrics in (prefill.get("results") or {}).items()
            if isinstance(metrics, dict)
        },
    }

    decode = _load_json(REFERENCE_PATHS["decode"])
    result["decode"] = {
        "sampling": decode.get("sampling", {}),
        "summary": decode.get("summary", {}),
    }

    result["ablation"] = _formal_ablation_payload()
    result["replication"] = _replication_payload()

    result["final_scheme"] = {
        "pairing": "Trace-aware + Local Search",
        "mapping": "Trace-aware",
        "prefill": "Aggressive-Reuse",
        "decode": "Greedy",
        "replication": "Not adopted",
    }

    return result


# ============================================================
# APIs
# ============================================================


@router.get("/health")
def comparison_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mapping_modes": [
            {
                "id": mode,
                "label": MAPPING_LABELS[mode],
                "available": (
                    MAPPING_PATHS[mode].exists()
                    or (mode == "trace_aware" and TRACE_AWARE_CANONICAL.exists())
                ),
            }
            for mode in MAPPING_MODES
        ],
        "prefill_modes": list(PREFILL_SCHEDULING_MODES),
        "pairing_modes": [
            {
                "id": mode,
                "label": PAIRING_LABELS[mode],
                "available": PAIRING_PATHS[mode].exists(),
            }
            for mode in PAIRING_MODES
        ],
    }


@router.get("/reference")
def comparison_reference() -> dict[str, Any]:
    try:
        return _reference_payload()
    except ComparisonError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/pairing")
def compare_pairing(request: PairingComparisonRequest) -> dict[str, Any]:
    """02 页面：同一个真实请求，只改变 UP-UP Pairing，Mapping 算法固定 Trace-aware。"""
    try:
        if request.pairing_a == request.pairing_b:
            raise ComparisonError("方案 A/B 请选择不同 Pairing，便于观察差异。")

        index_a = _pairing_runtime_index(request.pairing_a)
        index_b = _pairing_runtime_index(request.pairing_b)
        rules = ExecutionRules()

        if request.phase == "decode":
            if request.decode_source is None:
                raise ComparisonError("Decode Pairing 对比缺少 decode_source。")
            routes = _decode_routes(request.decode_source)
            routes_by_token = (routes,)
            source = _decode_source_public(request.decode_source)

            result_a = schedule_token(
                index=index_a,
                routed_experts_by_layer=routes,
                rules=rules,
                initial_active_cube_by_subcube=None,
                charge_initial_activation=True,
            )
            result_b = schedule_token(
                index=index_b,
                routed_experts_by_layer=routes,
                rules=rules,
                initial_active_cube_by_subcube=None,
                charge_initial_activation=True,
            )
            serialize = _serialize_token_result
        else:
            if request.prefill_batch_id is None:
                raise ComparisonError("Prefill Pairing 对比缺少 prefill_batch_id。")
            routes_by_token, source = _prefill_routes(request.prefill_batch_id)
            result_a = schedule_prefill_batch(
                index=index_a,
                routed_experts_by_token=routes_by_token,
                rules=rules,
                initial_active_cube_by_subcube=None,
                charge_initial_activation=True,
                scheduling_mode=PREFILL_MODE_AGGRESSIVE_REUSE,
            )
            result_b = schedule_prefill_batch(
                index=index_b,
                routed_experts_by_token=routes_by_token,
                rules=rules,
                initial_active_cube_by_subcube=None,
                charge_initial_activation=True,
                scheduling_mode=PREFILL_MODE_AGGRESSIVE_REUSE,
            )
            serialize = _serialize_prefill_result

        a_views = _pairing_views_all_layers(
            mode=request.pairing_a,
            routes_by_token=routes_by_token,
        )
        b_views = _pairing_views_all_layers(
            mode=request.pairing_b,
            routes_by_token=routes_by_token,
        )
        suggested_layer = _suggest_pairing_layer(a_views, b_views)
        selected_layer = (
            suggested_layer if request.selected_layer is None else int(request.selected_layer)
        )

        return {
            "kind": "pairing",
            "phase": request.phase,
            "scope": (
                "single real Decode token; MoE Expert only"
                if request.phase == "decode"
                else "single real Prefill batch; MoE Expert only"
            ),
            "request": source,
            "fixed": {
                "hardware": "N=4, H=7168, W=4096, D=1398",
                "mapping_algorithm": "Trace-aware",
                "prefill_scheduler": (
                    "Aggressive-Reuse" if request.phase == "prefill" else None
                ),
                "decode_scheduler": (
                    "Greedy" if request.phase == "decode" else None
                ),
            },
            "suggested_layer": suggested_layer,
            "selected_layer": selected_layer,
            "a": {
                "strategy": _pairing_metadata(request.pairing_a),
                "result": serialize(result_a, selected_layer=selected_layer),
                "pairing_layers": a_views,
            },
            "b": {
                "strategy": _pairing_metadata(request.pairing_b),
                "result": serialize(result_b, selected_layer=selected_layer),
                "pairing_layers": b_views,
            },
        }
    except (ComparisonError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/mapping")
def compare_mapping(request: MappingComparisonRequest) -> dict[str, Any]:
    try:
        index_a = _runtime_index(request.mapping_a)
        index_b = _runtime_index(request.mapping_b)
        rules = ExecutionRules()

        if request.phase == "decode":
            if request.decode_source is None:
                raise ComparisonError("Decode Mapping 对比缺少 decode_source。")

            routes = _decode_routes(request.decode_source)
            result_a = schedule_token(
                index=index_a,
                routed_experts_by_layer=routes,
                rules=rules,
                initial_active_cube_by_subcube=None,
                charge_initial_activation=True,
            )
            result_b = schedule_token(
                index=index_b,
                routed_experts_by_layer=routes,
                rules=rules,
                initial_active_cube_by_subcube=None,
                charge_initial_activation=True,
            )

            return {
                "kind": "mapping",
                "phase": "decode",
                "scope": "single real Decode token; MoE Expert only",
                "request": _decode_source_public(request.decode_source),
                "fixed": {
                    "pairing": "Trace-aware + Local Search",
                    "decode_scheduler": "Greedy",
                },
                "a": {
                    "strategy": _mapping_metadata(request.mapping_a),
                    "result": _serialize_token_result(
                        result_a,
                        selected_layer=request.selected_layer,
                    ),
                },
                "b": {
                    "strategy": _mapping_metadata(request.mapping_b),
                    "result": _serialize_token_result(
                        result_b,
                        selected_layer=request.selected_layer,
                    ),
                },
            }

        if request.prefill_batch_id is None:
            raise ComparisonError("Prefill Mapping 对比缺少 prefill_batch_id。")

        routes, source = _prefill_routes(request.prefill_batch_id)
        result_a = schedule_prefill_batch(
            index=index_a,
            routed_experts_by_token=routes,
            rules=rules,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=True,
            scheduling_mode=PREFILL_MODE_AGGRESSIVE_REUSE,
        )
        result_b = schedule_prefill_batch(
            index=index_b,
            routed_experts_by_token=routes,
            rules=rules,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=True,
            scheduling_mode=PREFILL_MODE_AGGRESSIVE_REUSE,
        )

        return {
            "kind": "mapping",
            "phase": "prefill",
            "scope": "single real Prefill batch; MoE Expert only",
            "request": source,
            "fixed": {
                "pairing": "Trace-aware + Local Search",
                "prefill_scheduler": "Aggressive-Reuse",
            },
            "a": {
                "strategy": _mapping_metadata(request.mapping_a),
                "result": _serialize_prefill_result(
                    result_a,
                    selected_layer=request.selected_layer,
                ),
            },
            "b": {
                "strategy": _mapping_metadata(request.mapping_b),
                "result": _serialize_prefill_result(
                    result_b,
                    selected_layer=request.selected_layer,
                ),
            },
        }

    except (ComparisonError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/prefill")
def compare_prefill(request: PrefillComparisonRequest) -> dict[str, Any]:
    try:
        mode_a = normalize_prefill_scheduling_mode(request.mode_a)
        mode_b = normalize_prefill_scheduling_mode(request.mode_b)
        routes, source = _prefill_routes(request.batch_id)
        index = _runtime_index("trace_aware")
        rules = ExecutionRules()

        result_a = schedule_prefill_batch(
            index=index,
            routed_experts_by_token=routes,
            rules=rules,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=True,
            scheduling_mode=mode_a,
        )
        result_b = schedule_prefill_batch(
            index=index,
            routed_experts_by_token=routes,
            rules=rules,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=True,
            scheduling_mode=mode_b,
        )

        return {
            "kind": "prefill_scheduler",
            "phase": "prefill",
            "scope": "single real Prefill batch; MoE Expert only",
            "request": source,
            "fixed": {
                "mapping": "Trace-aware",
                "pairing": "Trace-aware + Local Search",
            },
            "a": {
                "strategy": {"id": mode_a, "label": mode_a},
                "result": _serialize_prefill_result(
                    result_a,
                    selected_layer=request.selected_layer,
                ),
            },
            "b": {
                "strategy": {"id": mode_b, "label": mode_b},
                "result": _serialize_prefill_result(
                    result_b,
                    selected_layer=request.selected_layer,
                ),
            },
        }

    except (ComparisonError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/decode-optimality")
def compare_decode_optimality(request: DecodeOptimalityRequest) -> dict[str, Any]:
    try:
        routes = _decode_routes(request.source)
        route = routes[request.layer_id]
        index = _runtime_index("trace_aware")
        rules = ExecutionRules()

        greedy = schedule_layer(
            index=index,
            layer_id=request.layer_id,
            routed_expert_ids=route,
            rules=rules,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=True,
        )

        cp = solve_decode_layer_optimal(
            index=index,
            layer_id=request.layer_id,
            routed_expert_ids=route,
            rules=rules,
            time_limit_seconds=request.time_limit_seconds,
            num_workers=request.solver_workers,
            greedy_upper_bound_cycles=greedy.total_cycles,
            hint_start_times=_greedy_hint(greedy),
            validate_solution=True,
        )

        return {
            "kind": "decode_optimality",
            "phase": "decode",
            "scope": "single real Decode token × one MoE layer",
            "request": {
                **_decode_source_public(request.source),
                "layer_id": request.layer_id,
                "routed_experts": list(route),
            },
            "fixed": {
                "mapping": "Trace-aware",
                "pairing": "Trace-aware + Local Search",
                "cp_sat_role": "optimal reference only",
            },
            "a": {
                "strategy": {"id": "greedy", "label": "Greedy"},
                "result": _serialize_layer(greedy, include_tasks=True),
            },
            "b": {
                "strategy": {"id": "cp_sat", "label": "CP-SAT Optimal"},
                "result": _serialize_cp_sat(cp),
            },
        }

    except (ComparisonError, DecodeOptimalSolverError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
