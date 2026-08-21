"""
Paired Prefill evaluation for Expert Weight Replication.

The formal base Mapping stays frozen.  This evaluator loads an independent
replication_plan.json and compares, batch by batch:

    Baseline:
        Trace-aware Mapping + Aggressive-Reuse

    Replication:
        same Mapping + same Aggressive-Reuse
        + existing-spare Expert replicas
        + deterministic balanced token-to-copy assignment

Important protocol
------------------
1. Replica selection was produced from PROFILE only.
2. This evaluator forces EVALUATION subset for the formal comparison.
3. Both sides run on the exact same Evaluation batches in the same process.
4. The number of logical MoE computations is unchanged; replication only
   changes which physical copy serves selected token/expert tasks.
5. Metric scope remains MoE Expert Prefill only, not full-model TTFT.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from config import ExecutionRules
from mapping.trace_profile import DEFAULT_TRACE_ROOT
from mapping.trace_split import EVALUATION_SUBSET, TRACE_SUBSETS
from scheduling.prefill_evaluator import (
    PrefillEvaluationRecord,
    PrefillEvaluationSummary,
    build_summary,
    make_record,
    print_prefill_evaluation_summary,
)
from scheduling.prefill_fast_evaluator import (
    FastExpert,
    FastMatrix,
    FastPrefillBatchResult,
    FastPrefillScheduler,
    build_fast_tables,
)
from scheduling.prefill_scheduling_mode import PREFILL_MODE_AGGRESSIVE_REUSE
from scheduling.prefill_workload import (
    PrefillWorkloadStats,
    TraceSegmentBatch,
    iter_prefill_batches,
)
from scheduling.runtime_index import (
    DEFAULT_MAPPING_PATH,
    RuntimeIndex,
    load_runtime_index,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLICATION_PLAN = (
    PROJECT_ROOT / "results" / "replication" / "replication_plan.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "replication"
    / "prefill_replication_evaluation.json"
)

ASSIGNMENT_POLICY = "balanced_original_replica_alternation"


class PrefillReplicationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReplicaRuntimeInfo:
    layer_id: int
    expert_id: int
    virtual_expert_id: int
    base_gate_down_subcube_id: int
    base_up_subcube_id: int
    replica_gate_down_subcube_id: int
    replica_up_subcube_id: int


@dataclass(frozen=True, slots=True)
class BatchReplicationStats:
    batch_id: int
    token_count: int
    duplicated_expert_groups: int
    duplicate_token_opportunities: int
    replica_token_assignments: int
    baseline_cycles: int
    replication_cycles: int
    improvement_cycles: int
    improvement_percent: float


@dataclass(frozen=True, slots=True)
class ReplicationAssignmentTotals:
    duplicated_expert_groups: int
    duplicate_token_opportunities: int
    replica_token_assignments: int
    batches_with_replica_assignment: int


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise PrefillReplicationError(f"文件不存在：{path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise PrefillReplicationError(f"JSON 顶层必须是对象：{path}")
    return payload


def _reduction_percent(baseline: float, current: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - current) / baseline * 100.0


def _validate_plan(
    *,
    plan: dict,
    index: RuntimeIndex,
) -> None:
    if plan.get("replication_version") != 1:
        raise PrefillReplicationError(
            "当前 evaluator 只支持 replication_version=1。"
        )

    policy = plan.get("policy")
    if not isinstance(policy, dict):
        raise PrefillReplicationError("Replication plan 缺少 policy。")

    if policy.get("trace_subset") != "profile":
        raise PrefillReplicationError(
            "Replica 必须由 profile subset 选择；"
            f"plan 中为 {policy.get('trace_subset')!r}。"
        )

    hardware = plan.get("hardware")
    if not isinstance(hardware, dict):
        raise PrefillReplicationError("Replication plan 缺少 hardware。")

    if int(hardware.get("num_subcubes", -1)) != index.num_subcubes:
        raise PrefillReplicationError(
            "Replication plan 与 RuntimeIndex 的 Sub-Cube 数不一致。"
        )

    replicas = plan.get("replicas")
    if not isinstance(replicas, list) or not replicas:
        raise PrefillReplicationError("Replication plan 没有 replicas。")

    seen: set[tuple[int, int]] = set()
    for row in replicas:
        layer_id = int(row["layer_id"])
        expert_id = int(row["expert_id"])
        key = (layer_id, expert_id)

        if key in seen:
            raise PrefillReplicationError(
                f"同一个 Expert 出现多个 replica 定义：{key}。"
            )
        seen.add(key)

        # Ensure this is a valid base routed expert.
        expert = index.expert(layer_id, expert_id)
        if expert.is_shared:
            raise PrefillReplicationError(
                f"当前实验只允许 Routed Expert replica：{key}。"
            )

        copy = row["replica_copy"]
        gate = copy["gate"]
        up = copy["up"]
        down = copy["down"]

        gate_sc = int(gate["subcube_id"])
        up_sc = int(up["subcube_id"])
        down_sc = int(down["subcube_id"])

        if gate_sc != down_sc:
            raise PrefillReplicationError(
                f"L{layer_id}/E{expert_id} replica gate/down 未共址。"
            )
        if gate_sc == up_sc:
            raise PrefillReplicationError(
                f"L{layer_id}/E{expert_id} replica gate/up 未分离。"
            )

        for matrix in (gate, up, down):
            sc = int(matrix["subcube_id"])
            if not 0 <= sc < index.num_subcubes:
                raise PrefillReplicationError(
                    f"L{layer_id}/E{expert_id} replica SC 越界：{sc}。"
                )


def build_replication_scheduler(
    *,
    index: RuntimeIndex,
    plan: dict,
    rules: ExecutionRules,
) -> tuple[
    FastPrefillScheduler,
    dict[tuple[int, int], ReplicaRuntimeInfo],
]:
    """
    Extend FastPrefillScheduler's per-layer lookup tables with virtual Expert IDs.

    The original RuntimeIndex is untouched.

    Example for one layer:
        0..255 routed
        256    shared
        257    replica of selected routed expert A
        258    replica of selected routed expert B
    """
    _validate_plan(plan=plan, index=index)

    scheduler = FastPrefillScheduler(
        index=index,
        rules=rules,
        scheduling_mode=PREFILL_MODE_AGGRESSIVE_REUSE,
    )

    base_tables = build_fast_tables(index)
    extended_tables: list[tuple[FastExpert, ...]] = []
    runtime_info: dict[tuple[int, int], ReplicaRuntimeInfo] = {}

    replicas_by_layer: dict[int, list[dict]] = {}
    for row in plan["replicas"]:
        replicas_by_layer.setdefault(int(row["layer_id"]), []).append(row)

    for layer_id in range(index.num_layers):
        row = list(base_tables[layer_id])

        layer_replicas = sorted(
            replicas_by_layer.get(layer_id, []),
            key=lambda item: int(item["expert_id"]),
        )

        for replica in layer_replicas:
            expert_id = int(replica["expert_id"])
            virtual_id = len(row)

            copy = replica["replica_copy"]
            gate = copy["gate"]
            up = copy["up"]
            down = copy["down"]

            row.append(
                FastExpert(
                    gate=FastMatrix(
                        subcube_id=int(gate["subcube_id"]),
                        cube_id=int(gate["virtual_cube_id"]),
                    ),
                    up=FastMatrix(
                        subcube_id=int(up["subcube_id"]),
                        cube_id=int(up["virtual_cube_id"]),
                    ),
                    down=FastMatrix(
                        subcube_id=int(down["subcube_id"]),
                        cube_id=int(down["virtual_cube_id"]),
                    ),
                )
            )

            base = replica["base_copy"]
            runtime_info[(layer_id, expert_id)] = ReplicaRuntimeInfo(
                layer_id=layer_id,
                expert_id=expert_id,
                virtual_expert_id=virtual_id,
                base_gate_down_subcube_id=int(
                    base["gate_down_subcube_id"]
                ),
                base_up_subcube_id=int(base["up_subcube_id"]),
                replica_gate_down_subcube_id=int(gate["subcube_id"]),
                replica_up_subcube_id=int(up["subcube_id"]),
            )

        extended_tables.append(tuple(row))

    # Intentional overlay: this scheduler alone sees the extra virtual copies.
    scheduler.tables = tuple(extended_tables)

    return scheduler, runtime_info


def transform_routes_balanced(
    *,
    batch: TraceSegmentBatch,
    runtime_info: dict[tuple[int, int], ReplicaRuntimeInfo],
) -> tuple[
    tuple[tuple[tuple[int, ...], ...], ...],
    int,
    int,
    int,
]:
    """
    Deterministic balanced assignment.

    For each replicated (layer, expert), collect all tokens in the current
    Prefill batch that route to it.

    hits <= 1:
        keep the original copy.

    hits >= 2:
        token occurrence 0 -> original
        token occurrence 1 -> replica
        token occurrence 2 -> original
        token occurrence 3 -> replica
        ...

    Thus:
        original_count = ceil(hits / 2)
        replica_count  = floor(hits / 2)

    The same token's gate/up/down always use the same physical copy because
    the routed Expert ID itself is replaced by one virtual Expert ID.
    """
    mutable = [
        [list(route) for route in token_routes]
        for token_routes in batch.routed_experts_by_token
    ]

    duplicated_groups = 0
    duplicate_opportunities = 0
    replica_assignments = 0

    for (layer_id, expert_id), info in sorted(runtime_info.items()):
        hit_positions: list[tuple[int, int]] = []

        for token_index, token_routes in enumerate(mutable):
            route = token_routes[layer_id]
            for route_pos, routed_id in enumerate(route):
                if routed_id == expert_id:
                    hit_positions.append((token_index, route_pos))

        hits = len(hit_positions)
        if hits <= 1:
            continue

        duplicated_groups += 1
        duplicate_opportunities += hits - 1

        # Keep first hit on the original copy; alternate afterwards.
        for occurrence_index, (token_index, route_pos) in enumerate(
            hit_positions
        ):
            if occurrence_index % 2 == 1:
                mutable[token_index][layer_id][route_pos] = (
                    info.virtual_expert_id
                )
                replica_assignments += 1

    transformed = tuple(
        tuple(tuple(route) for route in token_routes)
        for token_routes in mutable
    )

    return (
        transformed,
        duplicated_groups,
        duplicate_opportunities,
        replica_assignments,
    )


def _init_accumulators(
    *,
    index: RuntimeIndex,
) -> dict:
    return {
        "records": [],
        "layer_cycles": [[] for _ in range(index.num_layers)],
        "layer_switches": [[] for _ in range(index.num_layers)],
        "layer_waits": [[] for _ in range(index.num_layers)],
        "sc_task_count": [0] * index.num_subcubes,
        "sc_busy_cycles": [0] * index.num_subcubes,
        "sc_switch_count": [0] * index.num_subcubes,
        "sc_initial_count": [0] * index.num_subcubes,
        "sc_wait_cycles": [0] * index.num_subcubes,
        "sc_critical_layer_count": [0] * index.num_subcubes,
    }


def _accumulate(
    *,
    store: dict,
    batch: TraceSegmentBatch,
    result: FastPrefillBatchResult,
) -> None:
    store["records"].append(make_record(batch=batch, result=result))

    for execution in result.layers:
        layer_id = execution.layer_id
        lr = execution.layer_result

        store["layer_cycles"][layer_id].append(lr.total_cycles)
        store["layer_switches"][layer_id].append(lr.switch_count)
        store["layer_waits"][layer_id].append(lr.wait_cycles)

        for stat in lr.subcube_stats:
            if (
                stat.task_count > 0
                and stat.last_finish_time == lr.total_cycles
            ):
                store["sc_critical_layer_count"][stat.subcube_id] += 1

    for stat in result.subcube_stats:
        sc = stat.subcube_id
        store["sc_task_count"][sc] += stat.task_count
        store["sc_busy_cycles"][sc] += stat.busy_cycles
        store["sc_switch_count"][sc] += stat.switch_count
        store["sc_initial_count"][sc] += stat.initial_activation_count
        store["sc_wait_cycles"][sc] += stat.wait_cycles


def _build_summary_from_store(store: dict) -> PrefillEvaluationSummary:
    return build_summary(
        records=store["records"],
        layer_cycle_values=store["layer_cycles"],
        layer_switch_values=store["layer_switches"],
        layer_wait_values=store["layer_waits"],
        sc_task_count=store["sc_task_count"],
        sc_busy_cycles=store["sc_busy_cycles"],
        sc_switch_count=store["sc_switch_count"],
        sc_initial_count=store["sc_initial_count"],
        sc_wait_cycles=store["sc_wait_cycles"],
        sc_critical_layer_count=store["sc_critical_layer_count"],
    )


def evaluate_paired_replication(
    *,
    index: RuntimeIndex,
    plan: dict,
    trace_root: Path,
    trace_manifest: Path,
    trace_subset: str,
    max_batches: int | None,
    progress_every: int,
    verbose: bool,
) -> tuple[
    PrefillEvaluationSummary,
    tuple[PrefillEvaluationRecord, ...],
    PrefillEvaluationSummary,
    tuple[PrefillEvaluationRecord, ...],
    tuple[BatchReplicationStats, ...],
    ReplicationAssignmentTotals,
]:
    if trace_subset != EVALUATION_SUBSET:
        raise PrefillReplicationError(
            "正式 Replication 对比必须使用 evaluation subset，"
            f"当前收到 {trace_subset!r}。"
        )

    rules = ExecutionRules()

    baseline_scheduler = FastPrefillScheduler(
        index=index,
        rules=rules,
        scheduling_mode=PREFILL_MODE_AGGRESSIVE_REUSE,
    )
    replication_scheduler, runtime_info = build_replication_scheduler(
        index=index,
        plan=plan,
        rules=rules,
    )

    workload_stats = PrefillWorkloadStats()
    iterator = iter_prefill_batches(
        trace_root=trace_root,
        trace_manifest=trace_manifest,
        trace_subset=trace_subset,
        max_files=None,
        max_batches=max_batches,
        stats=workload_stats,
        verbose=False,
    )

    baseline_store = _init_accumulators(index=index)
    replication_store = _init_accumulators(index=index)

    paired_rows: list[BatchReplicationStats] = []

    total_groups = 0
    total_opportunities = 0
    total_replica_assignments = 0
    batches_with_assignment = 0

    for batch_index, batch in enumerate(iterator, start=1):
        baseline_result = baseline_scheduler.schedule_batch(
            batch.routed_experts_by_token,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=True,
            validate_routes=False,
        )

        (
            transformed_routes,
            duplicated_groups,
            duplicate_opportunities,
            replica_assignments,
        ) = transform_routes_balanced(
            batch=batch,
            runtime_info=runtime_info,
        )

        replication_result = replication_scheduler.schedule_batch(
            transformed_routes,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=True,
            validate_routes=False,
        )

        # Replication must not change logical work count.
        if (
            replication_result.total_tasks
            != baseline_result.total_tasks
        ):
            raise PrefillReplicationError(
                f"Batch-{batch.batch_id}: replication 改变了 task 数："
                f"baseline={baseline_result.total_tasks}, "
                f"replication={replication_result.total_tasks}。"
            )

        _accumulate(
            store=baseline_store,
            batch=batch,
            result=baseline_result,
        )
        _accumulate(
            store=replication_store,
            batch=batch,
            result=replication_result,
        )

        improvement_cycles = (
            baseline_result.total_cycles
            - replication_result.total_cycles
        )
        improvement_percent = _reduction_percent(
            baseline_result.total_cycles,
            replication_result.total_cycles,
        )

        paired_rows.append(
            BatchReplicationStats(
                batch_id=batch.batch_id,
                token_count=batch.token_count,
                duplicated_expert_groups=duplicated_groups,
                duplicate_token_opportunities=duplicate_opportunities,
                replica_token_assignments=replica_assignments,
                baseline_cycles=baseline_result.total_cycles,
                replication_cycles=replication_result.total_cycles,
                improvement_cycles=improvement_cycles,
                improvement_percent=improvement_percent,
            )
        )

        total_groups += duplicated_groups
        total_opportunities += duplicate_opportunities
        total_replica_assignments += replica_assignments
        if replica_assignments > 0:
            batches_with_assignment += 1

        if verbose and (
            batch_index == 1
            or batch_index % progress_every == 0
        ):
            current_mean_base = mean(
                row.baseline_cycles for row in paired_rows
            )
            current_mean_rep = mean(
                row.replication_cycles for row in paired_rows
            )
            print(
                "[PrefillReplication] "
                f"batches={batch_index}, "
                f"baseline_mean={current_mean_base:.2f}, "
                f"replication_mean={current_mean_rep:.2f}, "
                f"improvement={_reduction_percent(current_mean_base, current_mean_rep):.3f}%"
            )

    if not paired_rows:
        raise PrefillReplicationError(
            "Evaluation subset 中没有 Prefill Candidate。"
        )

    baseline_summary = _build_summary_from_store(baseline_store)
    replication_summary = _build_summary_from_store(replication_store)

    totals = ReplicationAssignmentTotals(
        duplicated_expert_groups=total_groups,
        duplicate_token_opportunities=total_opportunities,
        replica_token_assignments=total_replica_assignments,
        batches_with_replica_assignment=batches_with_assignment,
    )

    return (
        baseline_summary,
        tuple(baseline_store["records"]),
        replication_summary,
        tuple(replication_store["records"]),
        tuple(paired_rows),
        totals,
    )


def print_comparison(
    *,
    baseline: PrefillEvaluationSummary,
    replication: PrefillEvaluationSummary,
    paired: tuple[BatchReplicationStats, ...],
    totals: ReplicationAssignmentTotals,
    plan: dict,
) -> None:
    b_mean = float(baseline.total_cycles.mean)
    r_mean = float(replication.total_cycles.mean)

    b_cpt = float(baseline.cycles_per_input_token.mean)
    r_cpt = float(replication.cycles_per_input_token.mean)

    b_p95 = float(baseline.total_cycles.p95)
    r_p95 = float(replication.total_cycles.p95)

    b_switch = float(baseline.switches.mean)
    r_switch = float(replication.switches.mean)

    improved = sum(1 for row in paired if row.improvement_cycles > 0)
    equal = sum(1 for row in paired if row.improvement_cycles == 0)
    worse = sum(1 for row in paired if row.improvement_cycles < 0)

    print("\n========== Prefill Weight Replication Comparison ==========")
    print("Scheduler：aggressive_reuse")
    print("Assignment：balanced original/replica alternation")
    print(
        "Replicated Experts："
        + ", ".join(
            f"L{row['layer_id']}/E{row['expert_id']}"
            for row in plan["replicas"]
        )
    )
    print(
        "Existing Spare Planes Used："
        f"{plan['hardware']['replica_planes_used']} / "
        f"{plan['hardware']['empty_plane_slots_before']}"
    )

    print("\nMetric                         Baseline    Replication    Improvement")
    print("-" * 72)
    print(
        f"Mean Prefill Cycles          {b_mean:10.2f} {r_mean:14.2f} "
        f"{_reduction_percent(b_mean, r_mean):11.3f}%"
    )
    print(
        f"Mean Cycles/Input Token      {b_cpt:10.4f} {r_cpt:14.4f} "
        f"{_reduction_percent(b_cpt, r_cpt):11.3f}%"
    )
    print(
        f"P95 Prefill Cycles           {b_p95:10.2f} {r_p95:14.2f} "
        f"{_reduction_percent(b_p95, r_p95):11.3f}%"
    )
    print(
        f"Mean Switches/Batch          {b_switch:10.2f} {r_switch:14.2f} "
        f"{_reduction_percent(b_switch, r_switch):11.3f}%"
    )

    print("\nPaired Batch Outcome：")
    print(f"  Improved：{improved}")
    print(f"  Equal：{equal}")
    print(f"  Worse：{worse}")
    print(f"  Total：{len(paired)}")

    print("\nReplication Assignment：")
    print(
        "  Duplicated Expert Groups："
        f"{totals.duplicated_expert_groups}"
    )
    print(
        "  Duplicate Token Opportunities："
        f"{totals.duplicate_token_opportunities}"
    )
    print(
        "  Replica Token Assignments："
        f"{totals.replica_token_assignments}"
    )
    print(
        "  Batches Using Replica："
        f"{totals.batches_with_replica_assignment}/{len(paired)}"
    )

    if paired:
        best = max(
            paired,
            key=lambda row: row.improvement_cycles,
        )
        worst = min(
            paired,
            key=lambda row: row.improvement_cycles,
        )
        print("\nBest / Worst Batch：")
        print(
            f"  Best: batch={best.batch_id}, tokens={best.token_count}, "
            f"{best.baseline_cycles}->{best.replication_cycles}, "
            f"delta={best.improvement_cycles}"
        )
        print(
            f"  Worst: batch={worst.batch_id}, tokens={worst.token_count}, "
            f"{worst.baseline_cycles}->{worst.replication_cycles}, "
            f"delta={worst.improvement_cycles}"
        )


def save_result(
    *,
    output_path: Path,
    mapping_path: Path,
    replication_plan_path: Path,
    trace_root: Path,
    trace_manifest: Path,
    trace_subset: str,
    plan: dict,
    baseline_summary: PrefillEvaluationSummary,
    baseline_records: tuple[PrefillEvaluationRecord, ...],
    replication_summary: PrefillEvaluationSummary,
    replication_records: tuple[PrefillEvaluationRecord, ...],
    paired: tuple[BatchReplicationStats, ...],
    totals: ReplicationAssignmentTotals,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    b_mean = float(baseline_summary.total_cycles.mean)
    r_mean = float(replication_summary.total_cycles.mean)

    improved = sum(1 for row in paired if row.improvement_cycles > 0)
    equal = sum(1 for row in paired if row.improvement_cycles == 0)
    worse = sum(1 for row in paired if row.improvement_cycles < 0)

    payload = {
        "evaluation_version": 1,
        "experiment": "existing_spare_expert_weight_replication",
        "metric_scope": "MoE Expert Prefill only; not full-model TTFT",
        "protocol": {
            "mapping": str(mapping_path.resolve()),
            "replication_plan": str(replication_plan_path.resolve()),
            "trace_root": str(trace_root.resolve()),
            "trace_manifest": str(trace_manifest.resolve()),
            "trace_subset": trace_subset,
            "scheduler": PREFILL_MODE_AGGRESSIVE_REUSE,
            "assignment_policy": ASSIGNMENT_POLICY,
            "paired_same_batches": True,
            "hardware_capacity_unchanged": True,
            "charge_initial_activation": True,
        },
        "replication_plan_summary": {
            "replicated_expert_count": plan["hardware"][
                "replicated_expert_count"
            ],
            "replica_planes_used": plan["hardware"][
                "replica_planes_used"
            ],
            "empty_plane_slots_before": plan["hardware"][
                "empty_plane_slots_before"
            ],
            "empty_plane_slots_after": plan["hardware"][
                "empty_plane_slots_after"
            ],
            "replicas": [
                {
                    "layer_id": row["layer_id"],
                    "expert_id": row["expert_id"],
                    "selection_stats": row["selection_stats"],
                    "base_copy": row["base_copy"],
                    "replica_copy": row["replica_copy"],
                }
                for row in plan["replicas"]
            ],
        },
        "comparison": {
            "baseline_mean_prefill_cycles": b_mean,
            "replication_mean_prefill_cycles": r_mean,
            "mean_prefill_improvement_percent": _reduction_percent(
                b_mean, r_mean
            ),
            "baseline_mean_cycles_per_input_token": float(
                baseline_summary.cycles_per_input_token.mean
            ),
            "replication_mean_cycles_per_input_token": float(
                replication_summary.cycles_per_input_token.mean
            ),
            "cycles_per_input_token_improvement_percent": (
                _reduction_percent(
                    float(
                        baseline_summary.cycles_per_input_token.mean
                    ),
                    float(
                        replication_summary.cycles_per_input_token.mean
                    ),
                )
            ),
            "baseline_p95_prefill_cycles": float(
                baseline_summary.total_cycles.p95
            ),
            "replication_p95_prefill_cycles": float(
                replication_summary.total_cycles.p95
            ),
            "p95_improvement_percent": _reduction_percent(
                float(baseline_summary.total_cycles.p95),
                float(replication_summary.total_cycles.p95),
            ),
            "baseline_mean_switches_per_batch": float(
                baseline_summary.switches.mean
            ),
            "replication_mean_switches_per_batch": float(
                replication_summary.switches.mean
            ),
            "switch_improvement_percent": _reduction_percent(
                float(baseline_summary.switches.mean),
                float(replication_summary.switches.mean),
            ),
            "improved_batches": improved,
            "equal_batches": equal,
            "worse_batches": worse,
            "total_batches": len(paired),
        },
        "assignment_totals": asdict(totals),
        "baseline_summary": asdict(baseline_summary),
        "replication_summary": asdict(replication_summary),
        "paired_records": [asdict(row) for row in paired],
        "baseline_records": [asdict(row) for row in baseline_records],
        "replication_records": [
            asdict(row) for row in replication_records
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "在 Held-out Evaluation 上，逐 Batch 对比 "
            "Aggressive-Reuse Baseline 与 Existing-Spare Expert Replication。"
        )
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
    )
    parser.add_argument(
        "--replication-plan",
        type=Path,
        default=DEFAULT_REPLICATION_PLAN,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
    )
    parser.add_argument(
        "--trace-manifest",
        type=Path,
        required=True,
        help="正式实验必须传 80/20 split manifest。",
    )
    parser.add_argument(
        "--trace-subset",
        choices=TRACE_SUBSETS,
        default=EVALUATION_SUBSET,
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="仅用于 smoke；正式实验不要设置。",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--top-layers",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--top-subcubes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.trace_subset != EVALUATION_SUBSET:
        parser.error(
            "Replication 正式评估必须使用 --trace-subset evaluation。"
        )

    mapping = args.mapping.resolve()
    plan_path = args.replication_plan.resolve()
    root = args.root.resolve()
    manifest = args.trace_manifest.resolve()
    output = args.output.resolve()

    index = load_runtime_index(mapping)
    plan = _load_json(plan_path)
    _validate_plan(plan=plan, index=index)

    (
        baseline_summary,
        baseline_records,
        replication_summary,
        replication_records,
        paired,
        totals,
    ) = evaluate_paired_replication(
        index=index,
        plan=plan,
        trace_root=root,
        trace_manifest=manifest,
        trace_subset=args.trace_subset,
        max_batches=args.max_batches,
        progress_every=args.progress_every,
        verbose=not args.quiet,
    )

    print("\n========== Baseline Prefill ==========")
    print_prefill_evaluation_summary(
        baseline_summary,
        top_layers=args.top_layers,
        top_subcubes=args.top_subcubes,
    )

    print("\n========== Replication Prefill ==========")
    print_prefill_evaluation_summary(
        replication_summary,
        top_layers=args.top_layers,
        top_subcubes=args.top_subcubes,
    )

    print_comparison(
        baseline=baseline_summary,
        replication=replication_summary,
        paired=paired,
        totals=totals,
        plan=plan,
    )

    saved = save_result(
        output_path=output,
        mapping_path=mapping,
        replication_plan_path=plan_path,
        trace_root=root,
        trace_manifest=manifest,
        trace_subset=args.trace_subset,
        plan=plan,
        baseline_summary=baseline_summary,
        baseline_records=baseline_records,
        replication_summary=replication_summary,
        replication_records=replication_records,
        paired=paired,
        totals=totals,
    )
    print(f"\nSaved：{saved}")


if __name__ == "__main__":
    main()