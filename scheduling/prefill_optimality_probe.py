"""真实 Prefill Trace 的 CP-SAT Optimality Probe。

目的：
    不修改当前 Prefill heuristic scheduler，抽取真实 Batch × Layer 实例，比较：

        Current Prefill Heuristic Layer Cycles
                    vs
        CP-SAT Best / Proven Optimal Cycles

从而回答：
    当前 Prefill 调度策略距离“固定 Mapping + 固定 Batch Route +
    固定该层初始 active WC 状态”下的单层理论最优还有多少空间？

重要口径：
1. 每个实例只优化一个 MoE Layer；
2. 当前层 initial_active_cube_by_subcube 来自 heuristic 完整 58 层执行轨迹；
3. CP-SAT 的 final active state 不回灌到下一层；
4. 因此这里证明的是“固定当前层进入状态”的单层最优，
   不是 58 层联合全局最优；
5. 默认使用 certification：已知当前合法上界 H，直接检查 makespan<=H-1；
   若 CP-SAT 返回 INFEASIBLE，则严格证明 OPT=H。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from config import ExecutionRules
from mapping.trace_profile import DEFAULT_TRACE_ROOT
from scheduling.prefill_optimal_solver import (
    PrefillOptimalLayerResult,
    PrefillOptimalSolverError,
    check_prefill_layer_makespan_feasible,
    solve_prefill_layer_optimal,
)
from scheduling.prefill_scheduler import schedule_prefill_batch
from scheduling.prefill_scheduling_mode import (
    PREFILL_MODE_SWITCH_AWARE,
    PREFILL_SCHEDULING_MODES,
    normalize_prefill_scheduling_mode,
)
from scheduling.prefill_workload import iter_prefill_batches
from scheduling.runtime_index import (
    DEFAULT_MAPPING_PATH,
    RuntimeIndex,
    load_runtime_index,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "prefill"
    / "prefill_optimality_probe.json"
)


class PrefillOptimalityProbeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PrefillOptimalityRecord:
    instance_id: int
    batch_id: int
    category: str
    relative_file: str
    segment_index: int

    token_count: int
    layer_id: int
    scheduling_mode: str

    task_count: int
    heuristic_cycles: int

    cp_sat_status: str
    cp_sat_cycles: int | None
    best_bound_cycles: float | None
    proven_optimal: bool

    proof_mode: str
    tested_target_cycles: int | None
    certification_rounds: int
    solver_found_improvement: bool

    improvement_cycles: int | None
    gap_vs_opt_percent: float | None

    wall_time_seconds: float
    branches: int
    conflicts: int


@dataclass(frozen=True, slots=True)
class PrefillOptimalityBucketSummary:
    bucket: str
    instance_count: int
    optimal_proven_count: int
    optimal_proven_rate: float
    heuristic_already_optimal_count: int
    heuristic_already_optimal_rate: float
    heuristic_mean_cycles: float
    proven_optimal_mean_cycles: float | None
    mean_gap_vs_opt_percent: float | None
    p50_gap_vs_opt_percent: float | None
    p95_gap_vs_opt_percent: float | None
    max_gap_vs_opt_percent: float | None


@dataclass(frozen=True, slots=True)
class PrefillOptimalitySummary:
    scheduling_mode: str
    instance_count: int
    feasible_count: int
    optimal_proven_count: int
    optimal_proven_rate: float

    heuristic_already_optimal_count: int
    heuristic_already_optimal_rate: float

    heuristic_mean_cycles: float
    proven_optimal_mean_cycles: float | None

    mean_gap_vs_opt_percent: float | None
    p50_gap_vs_opt_percent: float | None
    p95_gap_vs_opt_percent: float | None
    max_gap_vs_opt_percent: float | None

    mean_improvement_cycles: float | None
    max_improvement_cycles: int | None

    total_solver_wall_time_seconds: float
    buckets: tuple[PrefillOptimalityBucketSummary, ...]


def _percentile(values: Iterable[float], q: float) -> float | None:
    data = sorted(float(x) for x in values)
    if not data:
        return None
    if len(data) == 1:
        return data[0]

    position = (len(data) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return data[lo]

    frac = position - lo
    return data[lo] * (1.0 - frac) + data[hi] * frac


def _parse_layers(text: str, num_layers: int) -> tuple[int, ...]:
    raw = text.strip().lower()
    if raw in {"all", "*"}:
        return tuple(range(num_layers))

    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            layer_id = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--layers 必须是 all 或逗号分隔的 layer id。"
            ) from exc

        if not 0 <= layer_id < num_layers:
            raise argparse.ArgumentTypeError(
                f"layer_id={layer_id} 超出 [0,{num_layers - 1}]。"
            )
        values.append(layer_id)

    if not values:
        raise argparse.ArgumentTypeError("--layers 不能为空。")

    return tuple(dict.fromkeys(values))


def _batch_bucket(token_count: int) -> str:
    if token_count <= 2:
        return "1-2"
    if token_count <= 4:
        return "3-4"
    if token_count <= 8:
        return "5-8"
    if token_count <= 16:
        return "9-16"
    if token_count <= 32:
        return "17-32"
    if token_count <= 64:
        return "33-64"
    return "65+"


def _heuristic_hint(layer_result) -> dict[tuple[int, int, str], int]:
    """把 Exact Prefill heuristic 的 compute_start_time 作为 CP-SAT hint。"""
    return {
        (
            task.token_index,
            task.expert_id,
            task.matrix_name,
        ): task.compute_start_time
        for task in layer_result.tasks
    }


def _make_record(
    *,
    instance_id: int,
    batch,
    layer_id: int,
    scheduling_mode: str,
    heuristic_layer,
    cp_status: str,
    best_cycles: int | None,
    best_bound_cycles: float | None,
    proven_optimal: bool,
    wall_time_seconds: float,
    branches: int,
    conflicts: int,
    proof_mode: str,
    tested_target_cycles: int | None,
    certification_rounds: int,
) -> PrefillOptimalityRecord:
    improvement: int | None = None
    gap: float | None = None

    if best_cycles is not None:
        improvement = heuristic_layer.total_cycles - best_cycles

    if proven_optimal and best_cycles is not None:
        if best_cycles <= 0:
            raise PrefillOptimalityProbeError(
                "CP-SAT optimal cycles 必须大于 0。"
            )
        assert improvement is not None
        gap = improvement / best_cycles * 100.0

    return PrefillOptimalityRecord(
        instance_id=instance_id,
        batch_id=batch.batch_id,
        category=batch.category,
        relative_file=batch.relative_file,
        segment_index=batch.segment_index,
        token_count=batch.token_count,
        layer_id=layer_id,
        scheduling_mode=scheduling_mode,
        task_count=heuristic_layer.task_count,
        heuristic_cycles=heuristic_layer.total_cycles,
        cp_sat_status=cp_status,
        cp_sat_cycles=best_cycles,
        best_bound_cycles=best_bound_cycles,
        proven_optimal=proven_optimal,
        proof_mode=proof_mode,
        tested_target_cycles=tested_target_cycles,
        certification_rounds=certification_rounds,
        solver_found_improvement=(
            best_cycles is not None
            and best_cycles < heuristic_layer.total_cycles
        ),
        improvement_cycles=improvement,
        gap_vs_opt_percent=gap,
        wall_time_seconds=wall_time_seconds,
        branches=branches,
        conflicts=conflicts,
    )


def _certify_layer(
    *,
    index: RuntimeIndex,
    layer_id: int,
    layer_routes,
    heuristic_layer,
    rules: ExecutionRules,
    time_limit_seconds: float,
    solver_workers: int,
    max_sequence_arcs: int | None,
    max_certification_rounds: int,
) -> tuple[
    str,
    int,
    float | None,
    bool,
    int | None,
    int,
    float,
    int,
    int,
]:
    """用 H-1 feasibility 逐级证明单层最优。

    返回：
        (
            final_status,
            best_cycles,
            best_bound_cycles,
            proven_optimal,
            last_tested_target,
            rounds,
            wall_time,
            branches,
            conflicts,
        )

    逻辑：
    1. heuristic 已经给出合法上界 H；
    2. 检查 makespan <= H-1；
    3. INFEASIBLE => OPT=H；
    4. 若找到更优可行解 H'，继续检查 H'-1；
    5. UNKNOWN / 达到 round limit => 暂未证明。
    """
    if max_certification_rounds <= 0:
        raise PrefillOptimalityProbeError(
            "max_certification_rounds 必须大于 0。"
        )

    best_cycles = heuristic_layer.total_cycles
    last_target: int | None = None
    final_status = "NOT_RUN"
    proven_optimal = False
    rounds = 0
    total_wall = 0.0
    total_branches = 0
    total_conflicts = 0

    while rounds < max_certification_rounds:
        target = best_cycles - 1
        last_target = target

        # 周期为正整数；如果已经有 1-cycle 合法解，就无需再问 <=0。
        if target <= 0:
            proven_optimal = True
            final_status = "TRIVIAL_LOWER_BOUND"
            break

        check = check_prefill_layer_makespan_feasible(
            index=index,
            layer_id=layer_id,
            routed_experts_by_token=layer_routes,
            target_makespan_cycles=target,
            rules=rules,
            initial_active_cube_by_subcube=(
                heuristic_layer.initial_active_cube_by_subcube
            ),
            charge_initial_activation=True,
            time_limit_seconds=time_limit_seconds,
            num_workers=solver_workers,
            validate_solution=True,
            log_search_progress=False,
            max_sequence_arcs=max_sequence_arcs,
        )

        rounds += 1
        total_wall += check.wall_time_seconds
        total_branches += check.branches
        total_conflicts += check.conflicts
        final_status = check.status

        if check.status == "INFEASIBLE":
            # 已知 best_cycles 可行；又证明 <= best_cycles-1 不可行。
            # 因为 cycles 是整数，所以 best_cycles 就是严格全局最优。
            proven_optimal = True
            break

        if check.feasible:
            if check.objective_cycles is None:
                raise PrefillOptimalityProbeError(
                    "Feasibility Solver 找到解但缺少 makespan。"
                )
            if check.objective_cycles > target:
                raise PrefillOptimalityProbeError(
                    "Feasibility Solver 返回结果超过 target。"
                )
            if check.objective_cycles >= best_cycles:
                raise PrefillOptimalityProbeError(
                    "Feasibility Solver 没有真正改善当前上界。"
                )

            best_cycles = check.objective_cycles
            # 继续尝试 best-1。
            continue

        # UNKNOWN / MODEL_INVALID 等：没有证明，也没有更优可行解。
        break

    best_bound = float(best_cycles) if proven_optimal else None
    return (
        final_status,
        best_cycles,
        best_bound,
        proven_optimal,
        last_target,
        rounds,
        total_wall,
        total_branches,
        total_conflicts,
    )


def _build_bucket_summary(
    bucket: str,
    records: list[PrefillOptimalityRecord],
) -> PrefillOptimalityBucketSummary:
    optimal = [
        r
        for r in records
        if r.proven_optimal and r.cp_sat_cycles is not None
    ]
    already = [
        r
        for r in optimal
        if r.heuristic_cycles == r.cp_sat_cycles
    ]
    gaps = [
        r.gap_vs_opt_percent
        for r in optimal
        if r.gap_vs_opt_percent is not None
    ]

    return PrefillOptimalityBucketSummary(
        bucket=bucket,
        instance_count=len(records),
        optimal_proven_count=len(optimal),
        optimal_proven_rate=(
            len(optimal) / len(records)
            if records
            else 0.0
        ),
        heuristic_already_optimal_count=len(already),
        heuristic_already_optimal_rate=(
            len(already) / len(optimal)
            if optimal
            else 0.0
        ),
        heuristic_mean_cycles=float(
            mean(r.heuristic_cycles for r in records)
        ),
        proven_optimal_mean_cycles=(
            float(
                mean(
                    r.cp_sat_cycles
                    for r in optimal
                    if r.cp_sat_cycles is not None
                )
            )
            if optimal
            else None
        ),
        mean_gap_vs_opt_percent=(
            float(mean(gaps))
            if gaps
            else None
        ),
        p50_gap_vs_opt_percent=_percentile(gaps, 0.50),
        p95_gap_vs_opt_percent=_percentile(gaps, 0.95),
        max_gap_vs_opt_percent=(max(gaps) if gaps else None),
    )


def build_summary(
    records: tuple[PrefillOptimalityRecord, ...],
    *,
    scheduling_mode: str,
) -> PrefillOptimalitySummary:
    if not records:
        raise PrefillOptimalityProbeError(
            "没有 Prefill Optimality Probe 记录。"
        )

    feasible = [
        r for r in records
        if r.cp_sat_cycles is not None
    ]
    optimal = [
        r
        for r in records
        if r.proven_optimal and r.cp_sat_cycles is not None
    ]
    already = [
        r
        for r in optimal
        if r.heuristic_cycles == r.cp_sat_cycles
    ]
    gaps = [
        r.gap_vs_opt_percent
        for r in optimal
        if r.gap_vs_opt_percent is not None
    ]
    improvements = [
        r.improvement_cycles
        for r in optimal
        if r.improvement_cycles is not None
    ]

    grouped: dict[str, list[PrefillOptimalityRecord]] = {}
    for record in records:
        grouped.setdefault(
            _batch_bucket(record.token_count),
            [],
        ).append(record)

    bucket_order = (
        "1-2",
        "3-4",
        "5-8",
        "9-16",
        "17-32",
        "33-64",
        "65+",
    )
    bucket_summaries = tuple(
        _build_bucket_summary(bucket, grouped[bucket])
        for bucket in bucket_order
        if bucket in grouped
    )

    return PrefillOptimalitySummary(
        scheduling_mode=scheduling_mode,
        instance_count=len(records),
        feasible_count=len(feasible),
        optimal_proven_count=len(optimal),
        optimal_proven_rate=len(optimal) / len(records),
        heuristic_already_optimal_count=len(already),
        heuristic_already_optimal_rate=(
            len(already) / len(optimal)
            if optimal
            else 0.0
        ),
        heuristic_mean_cycles=float(
            mean(r.heuristic_cycles for r in records)
        ),
        proven_optimal_mean_cycles=(
            float(
                mean(
                    r.cp_sat_cycles
                    for r in optimal
                    if r.cp_sat_cycles is not None
                )
            )
            if optimal
            else None
        ),
        mean_gap_vs_opt_percent=(
            float(mean(gaps)) if gaps else None
        ),
        p50_gap_vs_opt_percent=_percentile(gaps, 0.50),
        p95_gap_vs_opt_percent=_percentile(gaps, 0.95),
        max_gap_vs_opt_percent=(max(gaps) if gaps else None),
        mean_improvement_cycles=(
            float(mean(improvements))
            if improvements
            else None
        ),
        max_improvement_cycles=(
            max(improvements)
            if improvements
            else None
        ),
        total_solver_wall_time_seconds=sum(
            r.wall_time_seconds for r in records
        ),
        buckets=bucket_summaries,
    )


def evaluate_prefill_optimality(
    *,
    index: RuntimeIndex,
    trace_root: Path | str = DEFAULT_TRACE_ROOT,
    scheduling_mode: str = PREFILL_MODE_SWITCH_AWARE,
    max_batches: int = 10,
    layers: tuple[int, ...] | None = None,
    max_instances: int | None = 50,
    min_token_count: int = 2,
    max_token_count: int | None = 8,
    time_limit_seconds: float = 10.0,
    solver_workers: int = 8,
    max_sequence_arcs: int | None = 300_000,
    max_files: int | None = None,
    proof_mode: str = "certify",
    max_certification_rounds: int = 4,
    verbose: bool = True,
) -> tuple[
    PrefillOptimalitySummary,
    tuple[PrefillOptimalityRecord, ...],
]:
    if max_batches <= 0:
        raise PrefillOptimalityProbeError(
            "max_batches 必须大于 0。"
        )
    if max_instances is not None and max_instances <= 0:
        raise PrefillOptimalityProbeError(
            "max_instances 必须大于 0。"
        )
    if min_token_count <= 0:
        raise PrefillOptimalityProbeError(
            "min_token_count 必须大于 0。"
        )
    if (
        max_token_count is not None
        and max_token_count < min_token_count
    ):
        raise PrefillOptimalityProbeError(
            "max_token_count 不能小于 min_token_count。"
        )

    proof_mode = str(proof_mode).strip().lower()
    if proof_mode not in {"certify", "optimize"}:
        raise PrefillOptimalityProbeError(
            "proof_mode 必须是 certify 或 optimize。"
        )
    if max_certification_rounds <= 0:
        raise PrefillOptimalityProbeError(
            "max_certification_rounds 必须大于 0。"
        )

    mode = normalize_prefill_scheduling_mode(scheduling_mode)
    rules = ExecutionRules()

    if layers is None:
        layers = tuple(range(index.num_layers))

    for layer_id in layers:
        if not 0 <= layer_id < index.num_layers:
            raise PrefillOptimalityProbeError(
                f"layer_id={layer_id} 超出范围。"
            )

    records: list[PrefillOptimalityRecord] = []
    instance_id = 0
    evaluated_batches = 0

    for batch in iter_prefill_batches(
        trace_root=trace_root,
        max_files=max_files,
        max_batches=None,
        stats=None,
        verbose=False,
    ):
        if not batch.is_prefill_candidate:
            continue

        if batch.token_count < min_token_count:
            continue
        if (
            max_token_count is not None
            and batch.token_count > max_token_count
        ):
            continue

        # 先运行一次当前 heuristic 的完整 58 层。
        # 这样每个 Layer 都能拿到“真实 heuristic 轨迹下”的 incoming active WC state。
        heuristic_batch = schedule_prefill_batch(
            index=index,
            routed_experts_by_token=batch.routed_experts_by_token,
            rules=rules,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=True,
            scheduling_mode=mode,
        )

        evaluated_batches += 1

        for layer_id in layers:
            heuristic_layer = (
                heuristic_batch
                .layer(layer_id)
                .layer_result
            )
            layer_routes = batch.layer_routes(layer_id)

            if proof_mode == "optimize":
                cp = solve_prefill_layer_optimal(
                    index=index,
                    layer_id=layer_id,
                    routed_experts_by_token=layer_routes,
                    rules=rules,
                    initial_active_cube_by_subcube=(
                        heuristic_layer.initial_active_cube_by_subcube
                    ),
                    charge_initial_activation=True,
                    time_limit_seconds=time_limit_seconds,
                    num_workers=solver_workers,
                    heuristic_upper_bound_cycles=(
                        heuristic_layer.total_cycles
                    ),
                    hint_compute_start_times=(
                        _heuristic_hint(heuristic_layer)
                    ),
                    validate_solution=True,
                    log_search_progress=False,
                    max_sequence_arcs=max_sequence_arcs,
                )

                if (
                    cp.objective_cycles is not None
                    and cp.objective_cycles > heuristic_layer.total_cycles
                ):
                    raise PrefillOptimalityProbeError(
                        "CP-SAT 比 heuristic 更差："
                        f"batch={batch.batch_id}, layer={layer_id}, "
                        f"heuristic={heuristic_layer.total_cycles}, "
                        f"cp={cp.objective_cycles}。"
                    )

                if cp.task_count not in (0, heuristic_layer.task_count):
                    raise PrefillOptimalityProbeError(
                        "CP-SAT task_count 与 heuristic 不一致："
                        f"batch={batch.batch_id}, layer={layer_id}, "
                        f"heuristic={heuristic_layer.task_count}, "
                        f"cp={cp.task_count}。"
                    )

                record = _make_record(
                    instance_id=instance_id,
                    batch=batch,
                    layer_id=layer_id,
                    scheduling_mode=mode,
                    heuristic_layer=heuristic_layer,
                    cp_status=cp.status,
                    best_cycles=cp.objective_cycles,
                    best_bound_cycles=cp.best_bound_cycles,
                    proven_optimal=cp.proven_optimal,
                    wall_time_seconds=cp.wall_time_seconds,
                    branches=cp.branches,
                    conflicts=cp.conflicts,
                    proof_mode="optimize",
                    tested_target_cycles=None,
                    certification_rounds=0,
                )

            else:
                (
                    final_status,
                    best_cycles,
                    best_bound,
                    proven_optimal,
                    tested_target,
                    certification_rounds,
                    total_wall,
                    total_branches,
                    total_conflicts,
                ) = _certify_layer(
                    index=index,
                    layer_id=layer_id,
                    layer_routes=layer_routes,
                    heuristic_layer=heuristic_layer,
                    rules=rules,
                    time_limit_seconds=time_limit_seconds,
                    solver_workers=solver_workers,
                    max_sequence_arcs=max_sequence_arcs,
                    max_certification_rounds=max_certification_rounds,
                )

                record = _make_record(
                    instance_id=instance_id,
                    batch=batch,
                    layer_id=layer_id,
                    scheduling_mode=mode,
                    heuristic_layer=heuristic_layer,
                    cp_status=final_status,
                    best_cycles=best_cycles,
                    best_bound_cycles=best_bound,
                    proven_optimal=proven_optimal,
                    wall_time_seconds=total_wall,
                    branches=total_branches,
                    conflicts=total_conflicts,
                    proof_mode="certify",
                    tested_target_cycles=tested_target,
                    certification_rounds=certification_rounds,
                )
            records.append(record)
            instance_id += 1

            if verbose:
                best_text = (
                    str(record.cp_sat_cycles)
                    if record.cp_sat_cycles is not None
                    else "NA"
                )
                gap_text = (
                    f"{record.gap_vs_opt_percent:.2f}%"
                    if record.gap_vs_opt_percent is not None
                    else "unproven"
                )
                target_text = (
                    str(record.tested_target_cycles)
                    if record.tested_target_cycles is not None
                    else "NA"
                )
                print(
                    f"[PrefillOptimality] #{record.instance_id} "
                    f"batch={record.batch_id}, "
                    f"tokens={record.token_count}, "
                    f"L{record.layer_id}, "
                    f"mode={record.scheduling_mode}, "
                    f"proof={record.proof_mode}, "
                    f"heuristic={record.heuristic_cycles}, "
                    f"best={best_text}, "
                    f"test<={target_text}, "
                    f"status={record.cp_sat_status}, "
                    f"proven={'YES' if record.proven_optimal else 'NO'}, "
                    f"gap={gap_text}, "
                    f"rounds={record.certification_rounds}, "
                    f"time={record.wall_time_seconds:.3f}s"
                )

            if (
                max_instances is not None
                and len(records) >= max_instances
            ):
                summary = build_summary(
                    tuple(records),
                    scheduling_mode=mode,
                )
                return summary, tuple(records)

        if evaluated_batches >= max_batches:
            break

    if not records:
        raise PrefillOptimalityProbeError(
            "没有找到满足 token_count 条件的 Prefill Batch。"
        )

    summary = build_summary(
        tuple(records),
        scheduling_mode=mode,
    )
    return summary, tuple(records)


def print_summary(summary: PrefillOptimalitySummary) -> None:
    def fmt(value: float | None, digits: int = 2) -> str:
        return "NA" if value is None else f"{value:.{digits}f}"

    print("\n========== Prefill CP-SAT Optimality Probe ==========")
    print(f"Scheduling Mode：{summary.scheduling_mode}")
    print(f"Instances：{summary.instance_count}")
    print(f"Feasible：{summary.feasible_count}")
    print(
        f"OPTIMAL Proven：{summary.optimal_proven_count} "
        f"({summary.optimal_proven_rate:.2%})"
    )
    print(
        "Heuristic Mean Layer Cycles："
        f"{summary.heuristic_mean_cycles:.4f}"
    )
    print(
        "Proven OPT Mean Layer Cycles："
        f"{fmt(summary.proven_optimal_mean_cycles, 4)}"
    )
    print(
        "Heuristic Already Optimal："
        f"{summary.heuristic_already_optimal_count} "
        f"({summary.heuristic_already_optimal_rate:.2%} of proven instances)"
    )
    print(
        f"Mean Gap vs OPT："
        f"{fmt(summary.mean_gap_vs_opt_percent)}%"
    )
    print(
        f"P50 Gap vs OPT："
        f"{fmt(summary.p50_gap_vs_opt_percent)}%"
    )
    print(
        f"P95 Gap vs OPT："
        f"{fmt(summary.p95_gap_vs_opt_percent)}%"
    )
    print(
        f"Max Gap vs OPT："
        f"{fmt(summary.max_gap_vs_opt_percent)}%"
    )
    print(
        "Mean Improvement："
        f"{fmt(summary.mean_improvement_cycles, 4)} cycles/layer"
    )
    print(
        f"Max Improvement："
        f"{summary.max_improvement_cycles}"
    )
    print(
        "Solver Total Wall Time："
        f"{summary.total_solver_wall_time_seconds:.3f}s"
    )

    if summary.buckets:
        print("\nBy Prefill Token Count：")
        for item in summary.buckets:
            print(
                f"  {item.bucket:<5} "
                f"instances={item.instance_count:<4} "
                f"proven={item.optimal_proven_rate:.2%} "
                f"already_opt={item.heuristic_already_optimal_rate:.2%} "
                f"mean_gap={fmt(item.mean_gap_vs_opt_percent)}% "
                f"p95_gap={fmt(item.p95_gap_vs_opt_percent)}%"
            )


def save_result(
    *,
    output_path: Path | str,
    summary: PrefillOptimalitySummary,
    records: tuple[PrefillOptimalityRecord, ...],
    mapping_path: Path | str,
    trace_root: Path | str,
    layers: tuple[int, ...],
    min_token_count: int,
    max_token_count: int | None,
    time_limit_seconds: float,
    solver_workers: int,
    max_sequence_arcs: int | None,
    proof_mode: str,
    max_certification_rounds: int,
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "evaluation_version": 1,
        "purpose": "Prefill layer heuristic vs CP-SAT optimality gap",
        "metric_scope": (
            "MoE Expert Prefill Layer only; fixed Mapping, fixed Batch Route, "
            "fixed heuristic incoming active-WC state; not 58-layer joint optimum; "
            "not full TTFT"
        ),
        "mapping": str(Path(mapping_path).resolve()),
        "trace_root": str(Path(trace_root).resolve()),
        "scheduling_mode": summary.scheduling_mode,
        "layers": list(layers),
        "token_filter": {
            "min_token_count": min_token_count,
            "max_token_count": max_token_count,
        },
        "solver": {
            "name": "OR-Tools CP-SAT",
            "time_limit_seconds_per_instance": time_limit_seconds,
            "workers": solver_workers,
            "max_sequence_arcs": max_sequence_arcs,
            "OPTIMAL_means_proven_global_optimum_for_fixed_layer_instance": True,
            "active_idle_allowed": True,
            "proof_mode": proof_mode,
            "certify_semantics": (
                "check makespan <= current_best-1; INFEASIBLE proves current_best optimal"
            ),
            "max_certification_rounds": max_certification_rounds,
        },
        "summary": asdict(summary),
        "records": [asdict(record) for record in records],
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "抽样真实 Prefill Batch×Layer，用 CP-SAT 测当前 heuristic "
            "距离固定层实例理论最优的差距。"
        )
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
    )
    parser.add_argument(
        "--scheduling-mode",
        choices=PREFILL_SCHEDULING_MODES,
        default=PREFILL_MODE_SWITCH_AWARE,
        help=(
            "要与 CP-SAT 对比的当前 Prefill heuristic。"
            "默认使用项目当前默认 switch_aware。"
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=10,
        help="最多使用多少个满足 token 过滤条件的真实 Prefill Batch。",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0,7,20,48,57",
        help="all 或逗号分隔，例如 0,7,20,48,57。",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=50,
        help="最多求多少个 Batch×Layer 实例。",
    )
    parser.add_argument(
        "--min-token-count",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-token-count",
        type=int,
        default=8,
        help=(
            "第一次建议限制在 <=8 Token；"
            "传 0 表示不设上限。"
        ),
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="每个 Layer 实例 CP-SAT 最长求解秒数。",
    )
    parser.add_argument(
        "--solver-workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-sequence-arcs",
        type=int,
        default=300_000,
        help=(
            "Prefill CP-SAT sequence arc 规模保护；"
            "传 0 表示关闭保护。"
        ),
    )
    parser.add_argument(
        "--proof-mode",
        choices=("certify", "optimize"),
        default="certify",
        help=(
            "certify：检查 current_best-1 是否可行，INFEASIBLE 即严格证明最优；"
            "optimize：保留原来的直接 minimize makespan 模式。"
        ),
    )
    parser.add_argument(
        "--max-certification-rounds",
        type=int,
        default=4,
        help=(
            "certify 模式最多连续改进/证明多少轮；每轮最多使用 --time-limit 秒。"
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    index = load_runtime_index(args.mapping)
    layers = _parse_layers(args.layers, index.num_layers)

    max_token_count = (
        None
        if args.max_token_count == 0
        else args.max_token_count
    )
    max_sequence_arcs = (
        None
        if args.max_sequence_arcs == 0
        else args.max_sequence_arcs
    )

    try:
        summary, records = evaluate_prefill_optimality(
            index=index,
            trace_root=args.root,
            scheduling_mode=args.scheduling_mode,
            max_batches=args.max_batches,
            layers=layers,
            max_instances=args.max_instances,
            min_token_count=args.min_token_count,
            max_token_count=max_token_count,
            time_limit_seconds=args.time_limit,
            solver_workers=args.solver_workers,
            max_sequence_arcs=max_sequence_arcs,
            max_files=args.max_files,
            proof_mode=args.proof_mode,
            max_certification_rounds=args.max_certification_rounds,
            verbose=not args.quiet,
        )
    except (PrefillOptimalSolverError, PrefillOptimalityProbeError) as exc:
        raise SystemExit(str(exc)) from exc

    print_summary(summary)

    if not args.no_save:
        saved = save_result(
            output_path=args.output,
            summary=summary,
            records=records,
            mapping_path=args.mapping,
            trace_root=args.root,
            layers=layers,
            min_token_count=args.min_token_count,
            max_token_count=max_token_count,
            time_limit_seconds=args.time_limit,
            solver_workers=args.solver_workers,
            max_sequence_arcs=max_sequence_arcs,
            proof_mode=args.proof_mode,
            max_certification_rounds=args.max_certification_rounds,
        )
        print(f"\nSaved：{saved}")


if __name__ == "__main__":
    main()