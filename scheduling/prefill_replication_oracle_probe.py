"""
Replication Oracle Probe.

Diagnostic only -- NOT a production scheduler.

Goal
----
The existing-spare replication experiment showed that blindly splitting every
duplicated Expert 50/50 can hurt latency.  This probe answers the next question:

    "Do the existing 4 Expert replicas have useful latency potential at all,
     if the runtime could selectively decide which replicas to use?"

For each held-out Prefill batch:
1. Run the frozen Baseline:
       Trace-aware Mapping + Aggressive-Reuse
2. Find which of the 4 replicated Experts are actually duplicated in this batch.
3. Enumerate all subsets of only those relevant replicated Experts.
4. For Experts enabled in a subset, use the same deterministic balanced
   original/replica alternation as the first experiment.
5. Pick the lowest-cycle result.

The selected best subset is an ORACLE result because it uses repeated simulator
evaluation to choose the best action for that batch.  It is only an upper-bound /
diagnostic experiment and must not be presented as the final runtime policy.

Interpretation
--------------
- Oracle ~= Baseline:
    existing 9 spare planes / selected replicas have little useful potential.
- Oracle clearly better than Baseline, while balanced-all is worse:
    replication is useful, but the runtime assignment policy needs to be selective.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
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
)
from scheduling.prefill_fast_evaluator import (
    FastPrefillBatchResult,
    FastPrefillScheduler,
)
from scheduling.prefill_replication_evaluator import (
    DEFAULT_REPLICATION_PLAN,
    ReplicaRuntimeInfo,
    _load_json,
    _validate_plan,
    build_replication_scheduler,
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
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "replication"
    / "prefill_replication_oracle_probe.json"
)


class ReplicationOracleError(ValueError):
    pass


ReplicaKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class OracleBatchRecord:
    batch_id: int
    token_count: int
    relevant_replicated_experts: tuple[str, ...]
    tested_subset_count: int

    baseline_cycles: int
    balanced_all_cycles: int
    oracle_cycles: int

    balanced_all_delta_cycles: int
    oracle_improvement_cycles: int
    oracle_improvement_percent: float

    oracle_selected_experts: tuple[str, ...]
    oracle_replica_assignments: int
    oracle_switches: int


def _key_text(key: ReplicaKey) -> str:
    return f"L{key[0]}/E{key[1]}"


def _reduction_percent(baseline: float, current: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - current) / baseline * 100.0


def _replicated_hits_in_batch(
    *,
    batch: TraceSegmentBatch,
    runtime_info: dict[ReplicaKey, ReplicaRuntimeInfo],
) -> dict[ReplicaKey, int]:
    hits: dict[ReplicaKey, int] = {}

    for key in runtime_info:
        layer_id, expert_id = key
        count = 0

        for token_routes in batch.routed_experts_by_token:
            route = token_routes[layer_id]
            count += sum(1 for routed_id in route if routed_id == expert_id)

        hits[key] = count

    return hits


def _transform_selected(
    *,
    batch: TraceSegmentBatch,
    runtime_info: dict[ReplicaKey, ReplicaRuntimeInfo],
    enabled: frozenset[ReplicaKey],
) -> tuple[
    tuple[tuple[tuple[int, ...], ...], ...],
    int,
]:
    """
    Only Experts in `enabled` may use their replica.

    For each enabled Expert with h hits:
        original gets occurrences 0,2,4,...
        replica  gets occurrences 1,3,5,...
    """
    mutable = [
        [list(route) for route in token_routes]
        for token_routes in batch.routed_experts_by_token
    ]

    replica_assignments = 0

    for key in sorted(enabled):
        info = runtime_info[key]
        layer_id, expert_id = key

        positions: list[tuple[int, int]] = []

        for token_index, token_routes in enumerate(mutable):
            route = token_routes[layer_id]
            for route_pos, routed_id in enumerate(route):
                if routed_id == expert_id:
                    positions.append((token_index, route_pos))

        if len(positions) <= 1:
            continue

        for occurrence_index, (token_index, route_pos) in enumerate(positions):
            if occurrence_index % 2 == 1:
                mutable[token_index][layer_id][route_pos] = (
                    info.virtual_expert_id
                )
                replica_assignments += 1

    transformed = tuple(
        tuple(tuple(route) for route in token_routes)
        for token_routes in mutable
    )
    return transformed, replica_assignments


def _all_subsets(
    items: tuple[ReplicaKey, ...],
) -> tuple[frozenset[ReplicaKey], ...]:
    rows: list[frozenset[ReplicaKey]] = []

    for size in range(len(items) + 1):
        for combo in combinations(items, size):
            rows.append(frozenset(combo))

    # Empty first; then smaller subsets; deterministic Expert order.
    rows.sort(
        key=lambda subset: (
            len(subset),
            tuple(sorted(subset)),
        )
    )
    return tuple(rows)


def _init_store(index: RuntimeIndex) -> dict:
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


def _summary(store: dict) -> PrefillEvaluationSummary:
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


def evaluate_oracle(
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
    PrefillEvaluationSummary,
    PrefillEvaluationSummary,
    tuple[OracleBatchRecord, ...],
    dict[str, int],
]:
    if trace_subset != EVALUATION_SUBSET:
        raise ReplicationOracleError(
            "Oracle Probe 正式实验只能使用 evaluation subset。"
        )

    rules = ExecutionRules()

    baseline_scheduler = FastPrefillScheduler(
        index=index,
        rules=rules,
        scheduling_mode=PREFILL_MODE_AGGRESSIVE_REUSE,
    )

    replica_scheduler, runtime_info = build_replication_scheduler(
        index=index,
        plan=plan,
        rules=rules,
    )

    all_replica_keys = tuple(sorted(runtime_info))

    baseline_store = _init_store(index)
    balanced_all_store = _init_store(index)
    oracle_store = _init_store(index)

    rows: list[OracleBatchRecord] = []
    selected_histogram: Counter[str] = Counter()

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

    for batch_number, batch in enumerate(iterator, start=1):
        # ----------------------------------------------------
        # 1. Baseline
        # ----------------------------------------------------
        baseline_result = baseline_scheduler.schedule_batch(
            batch.routed_experts_by_token,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=True,
            validate_routes=False,
        )

        # ----------------------------------------------------
        # 2. All-replicas balanced strategy
        # ----------------------------------------------------
        all_routes, _all_assignments = _transform_selected(
            batch=batch,
            runtime_info=runtime_info,
            enabled=frozenset(all_replica_keys),
        )
        balanced_all_result = replica_scheduler.schedule_batch(
            all_routes,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=True,
            validate_routes=False,
        )

        # ----------------------------------------------------
        # 3. Relevant replicas for this batch.
        #    hits<=1 can never gain from balanced splitting,
        #    so do not include them in subset enumeration.
        # ----------------------------------------------------
        hits = _replicated_hits_in_batch(
            batch=batch,
            runtime_info=runtime_info,
        )
        relevant = tuple(
            key for key in all_replica_keys
            if hits[key] > 1
        )

        subsets = _all_subsets(relevant)

        # Empty subset == Baseline.  Reuse the already computed result.
        best_result = baseline_result
        best_subset: frozenset[ReplicaKey] = frozenset()
        best_assignments = 0

        for subset in subsets:
            if not subset:
                continue

            transformed, replica_assignments = _transform_selected(
                batch=batch,
                runtime_info=runtime_info,
                enabled=subset,
            )

            result = replica_scheduler.schedule_batch(
                transformed,
                initial_active_cube_by_subcube=None,
                charge_initial_activation=True,
                validate_routes=False,
            )

            # Strictly prefer lower cycles.
            # If cycles tie, prefer:
            #   fewer replica assignments,
            #   fewer enabled Experts,
            #   lexicographically smaller subset.
            candidate_key = (
                result.total_cycles,
                replica_assignments,
                len(subset),
                tuple(sorted(subset)),
            )
            best_key = (
                best_result.total_cycles,
                best_assignments,
                len(best_subset),
                tuple(sorted(best_subset)),
            )

            if candidate_key < best_key:
                best_result = result
                best_subset = subset
                best_assignments = replica_assignments

        if best_result.total_cycles > baseline_result.total_cycles:
            raise ReplicationOracleError(
                "Oracle 结果不应劣于 Baseline。"
            )

        # Logical work must remain identical.
        if (
            balanced_all_result.total_tasks != baseline_result.total_tasks
            or best_result.total_tasks != baseline_result.total_tasks
        ):
            raise ReplicationOracleError(
                f"Batch-{batch.batch_id} replication 改变了 task 数。"
            )

        _accumulate(
            store=baseline_store,
            batch=batch,
            result=baseline_result,
        )
        _accumulate(
            store=balanced_all_store,
            batch=batch,
            result=balanced_all_result,
        )
        _accumulate(
            store=oracle_store,
            batch=batch,
            result=best_result,
        )

        selected_names = tuple(
            _key_text(key)
            for key in sorted(best_subset)
        )
        histogram_key = (
            "NONE"
            if not selected_names
            else "+".join(selected_names)
        )
        selected_histogram[histogram_key] += 1

        improvement = (
            baseline_result.total_cycles
            - best_result.total_cycles
        )

        rows.append(
            OracleBatchRecord(
                batch_id=batch.batch_id,
                token_count=batch.token_count,
                relevant_replicated_experts=tuple(
                    _key_text(key)
                    for key in relevant
                ),
                tested_subset_count=len(subsets),
                baseline_cycles=baseline_result.total_cycles,
                balanced_all_cycles=balanced_all_result.total_cycles,
                oracle_cycles=best_result.total_cycles,
                balanced_all_delta_cycles=(
                    baseline_result.total_cycles
                    - balanced_all_result.total_cycles
                ),
                oracle_improvement_cycles=improvement,
                oracle_improvement_percent=_reduction_percent(
                    baseline_result.total_cycles,
                    best_result.total_cycles,
                ),
                oracle_selected_experts=selected_names,
                oracle_replica_assignments=best_assignments,
                oracle_switches=best_result.total_switches,
            )
        )

        if verbose and (
            batch_number == 1
            or batch_number % progress_every == 0
        ):
            base_mean = mean(row.baseline_cycles for row in rows)
            all_mean = mean(row.balanced_all_cycles for row in rows)
            oracle_mean = mean(row.oracle_cycles for row in rows)

            print(
                "[ReplicationOracle] "
                f"batches={batch_number}, "
                f"baseline={base_mean:.2f}, "
                f"all={all_mean:.2f}, "
                f"oracle={oracle_mean:.2f}, "
                f"oracle_gain={_reduction_percent(base_mean, oracle_mean):.3f}%"
            )

    if not rows:
        raise ReplicationOracleError(
            "Evaluation subset 中没有 Prefill Candidate。"
        )

    return (
        _summary(baseline_store),
        _summary(balanced_all_store),
        _summary(oracle_store),
        tuple(rows),
        dict(
            sorted(
                selected_histogram.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    )


def print_probe_summary(
    *,
    baseline: PrefillEvaluationSummary,
    balanced_all: PrefillEvaluationSummary,
    oracle: PrefillEvaluationSummary,
    rows: tuple[OracleBatchRecord, ...],
    histogram: dict[str, int],
) -> None:
    b_mean = float(baseline.total_cycles.mean)
    a_mean = float(balanced_all.total_cycles.mean)
    o_mean = float(oracle.total_cycles.mean)

    b_cpt = float(baseline.cycles_per_input_token.mean)
    a_cpt = float(balanced_all.cycles_per_input_token.mean)
    o_cpt = float(oracle.cycles_per_input_token.mean)

    b_p95 = float(baseline.total_cycles.p95)
    a_p95 = float(balanced_all.total_cycles.p95)
    o_p95 = float(oracle.total_cycles.p95)

    b_switch = float(baseline.switches.mean)
    a_switch = float(balanced_all.switches.mean)
    o_switch = float(oracle.switches.mean)

    improved = sum(1 for row in rows if row.oracle_improvement_cycles > 0)
    equal = sum(1 for row in rows if row.oracle_improvement_cycles == 0)

    total_gain = sum(row.oracle_improvement_cycles for row in rows)
    max_gain = max(row.oracle_improvement_cycles for row in rows)

    print("\n========== Prefill Replication Oracle Probe ==========")
    print("Scheduler：aggressive_reuse")
    print("Oracle meaning：per-batch best subset of existing replicas")
    print("WARNING：Oracle is diagnostic upper bound, not final runtime policy.")

    print(
        "\nMetric                         Baseline    Balanced-All         Oracle"
    )
    print("-" * 76)
    print(
        f"Mean Prefill Cycles          {b_mean:10.2f} {a_mean:15.2f} {o_mean:14.2f}"
    )
    print(
        f"Mean Cycles/Input Token      {b_cpt:10.4f} {a_cpt:15.4f} {o_cpt:14.4f}"
    )
    print(
        f"P95 Prefill Cycles           {b_p95:10.2f} {a_p95:15.2f} {o_p95:14.2f}"
    )
    print(
        f"Mean Switches/Batch          {b_switch:10.2f} {a_switch:15.2f} {o_switch:14.2f}"
    )

    print("\nRelative to Baseline：")
    print(
        f"  Balanced-All Mean："
        f"{_reduction_percent(b_mean, a_mean):.4f}%"
    )
    print(
        f"  Oracle Mean："
        f"{_reduction_percent(b_mean, o_mean):.4f}%"
    )
    print(
        f"  Oracle Cycles/Input Token："
        f"{_reduction_percent(b_cpt, o_cpt):.4f}%"
    )
    print(
        f"  Oracle P95："
        f"{_reduction_percent(b_p95, o_p95):.4f}%"
    )

    print("\nOracle Paired Outcome：")
    print(f"  Improved：{improved}")
    print(f"  Equal：{equal}")
    print(f"  Worse：0")
    print(f"  Total：{len(rows)}")
    print(f"  Total Saved Cycles：{total_gain}")
    print(f"  Max Saved Cycles / Batch：{max_gain}")

    print("\nOracle Selected Subsets：")
    for subset, count in list(histogram.items())[:12]:
        print(f"  {subset}: {count}")


def save_result(
    *,
    output: Path,
    mapping: Path,
    plan_path: Path,
    manifest: Path,
    trace_root: Path,
    baseline: PrefillEvaluationSummary,
    balanced_all: PrefillEvaluationSummary,
    oracle: PrefillEvaluationSummary,
    rows: tuple[OracleBatchRecord, ...],
    histogram: dict[str, int],
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)

    b_mean = float(baseline.total_cycles.mean)
    a_mean = float(balanced_all.total_cycles.mean)
    o_mean = float(oracle.total_cycles.mean)

    payload = {
        "probe_version": 1,
        "probe": "existing_spare_replication_per_batch_subset_oracle",
        "diagnostic_only": True,
        "metric_scope": "MoE Expert Prefill only; not full-model TTFT",
        "protocol": {
            "mapping": str(mapping.resolve()),
            "replication_plan": str(plan_path.resolve()),
            "trace_root": str(trace_root.resolve()),
            "trace_manifest": str(manifest.resolve()),
            "trace_subset": EVALUATION_SUBSET,
            "scheduler": PREFILL_MODE_AGGRESSIVE_REUSE,
            "replica_assignment_when_enabled": (
                "balanced_original_replica_alternation"
            ),
            "oracle_action": (
                "enumerate all subsets of replicated Experts with >1 hit "
                "in each batch and select minimum simulated Prefill cycles"
            ),
        },
        "comparison": {
            "baseline_mean": b_mean,
            "balanced_all_mean": a_mean,
            "oracle_mean": o_mean,
            "balanced_all_improvement_percent": _reduction_percent(
                b_mean, a_mean
            ),
            "oracle_improvement_percent": _reduction_percent(
                b_mean, o_mean
            ),
            "oracle_improved_batches": sum(
                1 for row in rows
                if row.oracle_improvement_cycles > 0
            ),
            "oracle_equal_batches": sum(
                1 for row in rows
                if row.oracle_improvement_cycles == 0
            ),
            "oracle_total_saved_cycles": sum(
                row.oracle_improvement_cycles for row in rows
            ),
            "oracle_max_saved_cycles_per_batch": max(
                row.oracle_improvement_cycles for row in rows
            ),
        },
        "baseline_summary": asdict(baseline),
        "balanced_all_summary": asdict(balanced_all),
        "oracle_summary": asdict(oracle),
        "oracle_selected_subset_histogram": histogram,
        "records": [asdict(row) for row in rows],
    }

    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "现有 4 个 Expert Replica 的 Held-out Prefill "
            "per-batch subset oracle diagnostic。"
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
        help="仅 smoke 使用；正式 probe 不设置。",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.trace_subset != EVALUATION_SUBSET:
        parser.error(
            "Oracle Probe 正式实验必须使用 --trace-subset evaluation。"
        )

    mapping = args.mapping.resolve()
    plan_path = args.replication_plan.resolve()
    trace_root = args.root.resolve()
    manifest = args.trace_manifest.resolve()
    output = args.output.resolve()

    index = load_runtime_index(mapping)
    plan = _load_json(plan_path)
    _validate_plan(plan=plan, index=index)

    (
        baseline,
        balanced_all,
        oracle,
        rows,
        histogram,
    ) = evaluate_oracle(
        index=index,
        plan=plan,
        trace_root=trace_root,
        trace_manifest=manifest,
        trace_subset=args.trace_subset,
        max_batches=args.max_batches,
        progress_every=args.progress_every,
        verbose=not args.quiet,
    )

    print_probe_summary(
        baseline=baseline,
        balanced_all=balanced_all,
        oracle=oracle,
        rows=rows,
        histogram=histogram,
    )

    saved = save_result(
        output=output,
        mapping=mapping,
        plan_path=plan_path,
        manifest=manifest,
        trace_root=trace_root,
        baseline=baseline,
        balanced_all=balanced_all,
        oracle=oracle,
        rows=rows,
        histogram=histogram,
    )
    print(f"\nSaved：{saved}")


if __name__ == "__main__":
    main()