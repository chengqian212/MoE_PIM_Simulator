"""
WebUI Request Simulator API.

正式阶段口径：

    Prefill -> 每个 JSON 的 segment0（多 Token）
    Decode  -> 每个 JSON 的 segment1+（singleton）

Prefill 页面优先读取已经由 exact prefill_scheduler 生成的
results/prefill/prefill_evaluation.json，避免网页展示时重复跑完整评估。

同时按需读取原始 Trace，用于展示某个 Prefill Batch 在指定 Layer
中每个 Token 的 Top-8 Routed Experts。
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from webui.backend.trace_api import (
    TRACE_FIRST_MOE_LAYER,
    NUM_MOE_LAYERS,
    build_token,
    discover_categories,
    discover_category_files,
    get_trace_file,
    get_valid_segments,
    inspect_segment,
    load_trace_json,
    validate_route,
)


router = APIRouter(
    prefix="/api/request",
    tags=["Request Simulator"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PREFILL_RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "prefill"
    / "prefill_evaluation.json"
)


class RequestDataError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RequestDataError(f"找不到结果文件：{path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RequestDataError(f"JSON 解析失败：{path.name}") from exc

    if not isinstance(raw, dict):
        raise RequestDataError(f"{path.name} 最外层必须是 dict。")

    return raw


def _prefill_payload() -> dict[str, Any]:
    return _load_json(DEFAULT_PREFILL_RESULT_PATH)


def _prefill_records() -> list[dict[str, Any]]:
    raw = _prefill_payload()
    records = raw.get("records")

    if not isinstance(records, list):
        raise RequestDataError(
            "prefill_evaluation.json 缺少 records；请重新运行正式 Prefill evaluator。"
        )

    return [item for item in records if isinstance(item, dict)]


def _prefill_summary() -> dict[str, Any]:
    raw = _prefill_payload()
    summary = raw.get("summary")

    if not isinstance(summary, dict):
        raise RequestDataError("prefill_evaluation.json 缺少 summary。")

    return summary


def _relative_filename(record: dict[str, Any]) -> str:
    relative = str(record.get("relative_file", ""))
    relative = relative.replace("\\", "/")
    return relative.rsplit("/", 1)[-1]


def _public_prefill_record(record: dict[str, Any]) -> dict[str, Any]:
    layer_cycles = [int(v) for v in (record.get("layer_cycles") or [])]
    sc_busy = [int(v) for v in (record.get("subcube_busy_cycles") or [])]
    sc_switches = [int(v) for v in (record.get("subcube_switches") or [])]

    slowest_layer_id = None
    slowest_layer_cycles = None
    if layer_cycles:
        slowest_layer_id = max(range(len(layer_cycles)), key=layer_cycles.__getitem__)
        slowest_layer_cycles = layer_cycles[slowest_layer_id]

    busiest_sc_id = None
    busiest_sc_cycles = None
    if sc_busy:
        busiest_sc_id = max(range(len(sc_busy)), key=sc_busy.__getitem__)
        busiest_sc_cycles = sc_busy[busiest_sc_id]

    return {
        "batch_id": int(record.get("batch_id", 0)),
        "category": str(record.get("category", "")),
        "relative_file": str(record.get("relative_file", "")),
        "filename": _relative_filename(record),
        "segment_index": int(record.get("segment_index", 0)),
        "input_tokens": int(record.get("input_tokens", 0)),
        "total_cycles": int(record.get("total_cycles", 0)),
        "cycles_per_input_token": float(record.get("cycles_per_input_token", 0.0)),
        "input_tokens_per_cycle": float(record.get("input_tokens_per_cycle", 0.0)),
        "total_tasks": int(record.get("total_tasks", 0)),
        "switches": int(record.get("switches", 0)),
        "initial_activations": int(record.get("initial_activations", 0)),
        "activation_overhead_cycles": int(record.get("activation_overhead_cycles", 0)),
        "compute_work_cycles": int(record.get("compute_work_cycles", 0)),
        "busy_work_cycles": int(record.get("busy_work_cycles", 0)),
        "wait_cycles": int(record.get("wait_cycles", 0)),
        "max_task_wait_cycles": int(record.get("max_task_wait_cycles", 0)),
        "layer_cycles": layer_cycles,
        "subcube_busy_cycles": sc_busy,
        "subcube_switches": sc_switches,
        "slowest_layer_id": slowest_layer_id,
        "slowest_layer_cycles": slowest_layer_cycles,
        "busiest_subcube_id": busiest_sc_id,
        "busiest_subcube_cycles": busiest_sc_cycles,
    }


def _find_prefill_record(batch_id: int) -> dict[str, Any]:
    for record in _prefill_records():
        if int(record.get("batch_id", -1)) == batch_id:
            return record

    raise RequestDataError(f"不存在 Prefill Batch-{batch_id}。")


def _trace_source(record: dict[str, Any]) -> tuple[str, str, int]:
    category = str(record.get("category", ""))
    filename = _relative_filename(record)
    segment_index = int(record.get("segment_index", 0))

    if not category or not filename:
        raise RequestDataError("Prefill record 缺少 Trace 来源信息。")

    return category, filename, segment_index


@router.get("/prefill/meta")
def prefill_meta() -> dict[str, Any]:
    try:
        records = _prefill_records()
        summary = _prefill_summary()

        categories = Counter(str(item.get("category", "")) for item in records)
        categories.pop("", None)

        return {
            "batch_count": len(records),
            "summary": {
                "total_input_tokens": summary.get("total_input_tokens"),
                "prompt_tokens": summary.get("prompt_tokens"),
                "total_cycles": summary.get("total_cycles"),
                "cycles_per_input_token": summary.get("cycles_per_input_token"),
                "global_cycles_per_input_token": summary.get("global_cycles_per_input_token"),
            },
            "categories": [
                {"name": name, "batch_count": count}
                for name, count in sorted(categories.items())
            ],
            "scope": "MoE Expert Prefill only; not full TTFT",
        }
    except RequestDataError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/prefill/random")
def random_prefill_batch(
    category: str | None = None,
) -> dict[str, Any]:
    try:
        records = _prefill_records()

        if category:
            records = [
                item
                for item in records
                if str(item.get("category", "")) == category
            ]

        if not records:
            raise RequestDataError("没有符合条件的 Prefill Batch。")

        return _public_prefill_record(random.choice(records))
    except RequestDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/prefill/batches/{batch_id}")
def get_prefill_batch(batch_id: int) -> dict[str, Any]:
    try:
        return _public_prefill_record(_find_prefill_record(batch_id))
    except RequestDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/prefill/batches/{batch_id}/layers/{layer_id}")
def get_prefill_layer_routes(
    batch_id: int,
    layer_id: int,
) -> dict[str, Any]:
    try:
        if not 0 <= layer_id < NUM_MOE_LAYERS:
            raise RequestDataError(f"layer_id 必须位于 0~{NUM_MOE_LAYERS - 1}。")

        record = _find_prefill_record(batch_id)
        category, filename, segment_index = _trace_source(record)

        path = get_trace_file(category=category, filename=filename)
        data = load_trace_json(path)

        if not 0 <= segment_index < len(data):
            raise RequestDataError("Prefill segment_index 越界。")

        segment = data[segment_index]
        token_count = inspect_segment(segment)
        if token_count is None:
            raise RequestDataError("Prefill segment 不完整。")

        trace_layer_id = TRACE_FIRST_MOE_LAYER + layer_id
        raw_routes = segment[str(trace_layer_id)]

        token_routes: list[list[int]] = []
        expert_frequency: Counter[int] = Counter()

        for token_index, raw_route in enumerate(raw_routes):
            route = list(validate_route(raw_route))
            token_routes.append(route)
            expert_frequency.update(route)

        ranked = sorted(
            expert_frequency.items(),
            key=lambda item: (-item[1], item[0]),
        )

        layer_cycles = record.get("layer_cycles") or []
        cycles = int(layer_cycles[layer_id]) if layer_id < len(layer_cycles) else None

        return {
            "batch_id": batch_id,
            "category": category,
            "filename": filename,
            "segment_index": segment_index,
            "layer_id": layer_id,
            "trace_layer_id": trace_layer_id,
            "token_count": token_count,
            "layer_cycles": cycles,
            "shared_expert_id": 256,
            "token_routes": [
                {"token_index": idx, "routed_experts": route}
                for idx, route in enumerate(token_routes)
            ],
            "expert_frequency": [
                {"expert_id": expert_id, "token_count": count}
                for expert_id, count in ranked
            ],
            "unique_routed_expert_count": len(expert_frequency),
        }
    except (RequestDataError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/decode/random")
def random_decode_token(
    category: str | None = None,
) -> dict[str, Any]:
    """只从 segment1+ 且 token_count==1 的纯 Decode segment 中抽样。"""
    try:
        categories = discover_categories()
        if not categories:
            raise RequestDataError("Trace 中没有类别。")

        if category is None:
            category = random.choice(categories)["name"]

        files = discover_category_files(category)
        if not files:
            raise RequestDataError(f"{category} 中没有 JSON 文件。")

        candidate_files = files.copy()
        random.shuffle(candidate_files)

        for path in candidate_files[:200]:
            valid = get_valid_segments(category=category, filename=path.name)
            decode_segments = [
                item
                for item in valid
                if int(item["segment_index"]) > 0
                and int(item["token_count"]) == 1
            ]

            if not decode_segments:
                continue

            chosen = random.choice(decode_segments)
            return build_token(
                category=category,
                filename=path.name,
                segment_index=int(chosen["segment_index"]),
                token_index=0,
            )

        raise RequestDataError("没有找到合法的纯 Decode singleton segment。")
    except (RequestDataError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
