"""
真实 Decode Trace 的 CP-SAT Optimality Probe（Held-out / 分层随机抽样版）。

目的：
    不修改当前 Decode Greedy Scheduler，
    只在正式 Held-out Evaluation 子集上抽取真实 Token×Layer 实例，比较：

        Current Greedy Layer cycles
            vs
        CP-SAT best / proven optimal cycles

抽样协议：
    1. 只从 trace_manifest 指定的 subset（正式实验用 evaluation）读取 Decode Token；
    2. 先统计各类别 Decode Token 数；
    3. 按类别规模做比例分层；
    4. 每个类别内部用固定 random seed 做 reservoir sampling；
    5. 默认对抽到的每个 Token 检查全部 58 个 MoE Layer。

这样避免旧版本 --max-tokens N 按文件顺序截取，导致样本集中在单个 JSON 文件。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from config import ExecutionRules
from mapping.trace_profile import DEFAULT_TRACE_ROOT
from mapping.trace_split import EVALUATION_SUBSET, TRACE_SUBSETS
from scheduling.decode_optimal_solver import (
    DecodeOptimalLayerResult,
    DecodeOptimalSolverError,
    solve_decode_layer_optimal,
)
from scheduling.decode_workload import iter_decode_tokens
from scheduling.layer_scheduler import schedule_layer
from scheduling.runtime_index import (
    DEFAULT_MAPPING_PATH,
    RuntimeIndex,
    load_runtime_index,
)
from scheduling.trace_workload import TraceToken


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "decode"
    / "decode_optimality_probe.json"
)


class DecodeOptimalityProbeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DecodeOptimalityRecord:
    instance_id: int
    token_id: int
    category: str
    relative_file: str
    segment_index: int
    layer_id: int
    routed_experts: tuple[int, ...]
    greedy_cycles: int
    cp_sat_status: str
    cp_sat_cycles: int | None
    best_bound_cycles: float | None
    proven_optimal: bool
    improvement_cycles: int | None
    gap_vs_opt_percent: float | None
    wall_time_seconds: float
    branches: int
    conflicts: int


@dataclass(frozen=True, slots=True)
class DecodeOptimalitySummary:
    instance_count: int
    feasible_count: int
    optimal_proven_count: int
    optimal_proven_rate: float
    greedy_already_optimal_count: int
    greedy_already_optimal_rate: float
    greedy_mean_cycles: float
    proven_optimal_mean_cycles: float | None
    mean_gap_vs_opt_percent: float | None
    p50_gap_vs_opt_percent: float | None
    p95_gap_vs_opt_percent: float | None
    max_gap_vs_opt_percent: float | None
    mean_improvement_cycles: float | None
    max_improvement_cycles: int | None
    total_solver_wall_time_seconds: float


@dataclass(frozen=True, slots=True)
class DecodeSamplingSummary:
    protocol: str
    trace_subset: str
    sample_seed: int
    requested_tokens: int
    source_token_count: int
    source_file_count: int
    sampled_token_count: int
    sampled_file_count: int
    source_category_counts: dict[str, int]
    sampled_category_counts: dict[str, int]


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
                "--layers 必须是 all 或逗号分隔 layer id。"
            ) from exc
        if not 0 <= layer_id < num_layers:
            raise argparse.ArgumentTypeError(
                f"layer_id={layer_id} 超出 [0,{num_layers - 1}]。"
            )
        values.append(layer_id)

    if not values:
        raise argparse.ArgumentTypeError("--layers 不能为空。")
    return tuple(dict.fromkeys(values))


def _greedy_hint(layer_result) -> dict[tuple[int, str], int]:
    return {
        (task.expert_id, task.matrix_name): task.dispatch_time
        for task in layer_result.tasks
    }


def _make_record(
    *,
    instance_id: int,
    token: TraceToken,
    layer_id: int,
    route: tuple[int, ...],
    greedy_cycles: int,
    cp: DecodeOptimalLayerResult,
) -> DecodeOptimalityRecord:
    improvement: int | None = None
    gap: float | None = None

    if cp.objective_cycles is not None:
        improvement = greedy_cycles - cp.objective_cycles

    if cp.proven_optimal and cp.objective_cycles is not None:
        if cp.objective_cycles <= 0:
            raise DecodeOptimalityProbeError("CP-SAT optimal cycles 非正。")
        gap = improvement / cp.objective_cycles * 100.0

    return DecodeOptimalityRecord(
        instance_id=instance_id,
        token_id=token.token_id,
        category=token.category,
        relative_file=token.relative_file,
        segment_index=token.segment_index,
        layer_id=layer_id,
        routed_experts=route,
        greedy_cycles=greedy_cycles,
        cp_sat_status=cp.status,
        cp_sat_cycles=cp.objective_cycles,
        best_bound_cycles=cp.best_bound_cycles,
        proven_optimal=cp.proven_optimal,
        improvement_cycles=improvement,
        gap_vs_opt_percent=gap,
        wall_time_seconds=cp.wall_time_seconds,
        branches=cp.branches,
        conflicts=cp.conflicts,
    )


def build_summary(
    records: tuple[DecodeOptimalityRecord, ...],
) -> DecodeOptimalitySummary:
    if not records:
        raise DecodeOptimalityProbeError("没有 Optimality Probe 记录。")

    feasible = [r for r in records if r.cp_sat_cycles is not None]
    optimal = [r for r in records if r.proven_optimal and r.cp_sat_cycles is not None]
    gaps = [r.gap_vs_opt_percent for r in optimal if r.gap_vs_opt_percent is not None]
    improvements = [
        r.improvement_cycles
        for r in optimal
        if r.improvement_cycles is not None
    ]
    already_optimal = [
        r for r in optimal
        if r.greedy_cycles == r.cp_sat_cycles
    ]

    return DecodeOptimalitySummary(
        instance_count=len(records),
        feasible_count=len(feasible),
        optimal_proven_count=len(optimal),
        optimal_proven_rate=len(optimal) / len(records),
        greedy_already_optimal_count=len(already_optimal),
        greedy_already_optimal_rate=(
            len(already_optimal) / len(optimal) if optimal else 0.0
        ),
        greedy_mean_cycles=float(mean(r.greedy_cycles for r in records)),
        proven_optimal_mean_cycles=(
            float(mean(r.cp_sat_cycles for r in optimal if r.cp_sat_cycles is not None))
            if optimal else None
        ),
        mean_gap_vs_opt_percent=float(mean(gaps)) if gaps else None,
        p50_gap_vs_opt_percent=_percentile(gaps, 0.50),
        p95_gap_vs_opt_percent=_percentile(gaps, 0.95),
        max_gap_vs_opt_percent=max(gaps) if gaps else None,
        mean_improvement_cycles=(float(mean(improvements)) if improvements else None),
        max_improvement_cycles=(max(improvements) if improvements else None),
        total_solver_wall_time_seconds=sum(r.wall_time_seconds for r in records),
    )


def _allocate_proportional_quotas(
    category_counts: dict[str, int],
    sample_tokens: int,
) -> dict[str, int]:
    """按类别规模分层，使用 largest-remainder 分配整数 quota。"""

    if sample_tokens <= 0:
        raise DecodeOptimalityProbeError("sample_tokens 必须大于 0。")

    positive = {
        category: count
        for category, count in category_counts.items()
        if count > 0
    }
    total = sum(positive.values())
    if total <= 0:
        raise DecodeOptimalityProbeError("Held-out Decode 中没有可抽样 Token。")
    if sample_tokens > total:
        raise DecodeOptimalityProbeError(
            f"sample_tokens={sample_tokens} 超过 Held-out Decode 总数 {total}。"
        )

    categories = sorted(positive)

    # 样本量足够时，先保证每个非空类别至少 1 个。
    quotas = {category: 0 for category in categories}
    base_reserved = 0
    if sample_tokens >= len(categories):
        for category in categories:
            quotas[category] = 1
        base_reserved = len(categories)

    remaining = sample_tokens - base_reserved
    if remaining <= 0:
        return quotas

    capacities = {
        category: positive[category] - quotas[category]
        for category in categories
    }
    capacity_total = sum(capacities.values())

    raw_extra: dict[str, float] = {}
    floor_extra: dict[str, int] = {}
    for category in categories:
        if capacity_total == 0:
            raw = 0.0
        else:
            raw = remaining * capacities[category] / capacity_total
        raw_extra[category] = raw
        floor_extra[category] = min(capacities[category], math.floor(raw))
        quotas[category] += floor_extra[category]

    left = sample_tokens - sum(quotas.values())
    ranked = sorted(
        categories,
        key=lambda category: (
            -(raw_extra[category] - floor_extra[category]),
            category,
        ),
    )

    for category in ranked:
        if left <= 0:
            break
        if quotas[category] < positive[category]:
            quotas[category] += 1
            left -= 1

    # 极端容量边界下兜底。
    if left > 0:
        for category in categories:
            while left > 0 and quotas[category] < positive[category]:
                quotas[category] += 1
                left -= 1
            if left <= 0:
                break

    if sum(quotas.values()) != sample_tokens:
        raise DecodeOptimalityProbeError("类别 quota 分配失败。")

    return quotas


def sample_decode_tokens_stratified(
    *,
    trace_root: Path | str,
    trace_manifest: Path | str | None,
    trace_subset: str,
    sample_tokens: int,
    sample_seed: int,
    verbose: bool = True,
) -> tuple[tuple[TraceToken, ...], DecodeSamplingSummary]:
    """
    对 Held-out Decode 做两遍流式扫描：

    Pass-1：统计类别规模和源文件数；
    Pass-2：按类别 quota 做固定种子的 reservoir sampling。

    不把 50k 全量 Token 常驻内存。
    """

    category_counts: Counter[str] = Counter()
    source_files: set[str] = set()

    for token in iter_decode_tokens(
        trace_root=trace_root,
        trace_manifest=trace_manifest,
        trace_subset=trace_subset,
        max_tokens=None,
        strict_singleton=True,
        verbose=False,
    ):
        category_counts[token.category] += 1
        source_files.add(token.relative_file)

    total_tokens = sum(category_counts.values())
    quotas = _allocate_proportional_quotas(dict(category_counts), sample_tokens)

    if verbose:
        print("\n========== Decode Sampling ==========")
        print(f"Subset：{trace_subset}")
        print(f"Source Decode Tokens：{total_tokens}")
        print(f"Source Files：{len(source_files)}")
        print(f"Sample Tokens：{sample_tokens}")
        print(f"Seed：{sample_seed}")
        print("Category Quotas：")
        for category in sorted(quotas):
            print(
                f"  {category}: source={category_counts[category]}, "
                f"sample={quotas[category]}"
            )

    rng = random.Random(sample_seed)
    seen: Counter[str] = Counter()
    reservoirs: dict[str, list[TraceToken]] = {
        category: [] for category in quotas
    }

    for token in iter_decode_tokens(
        trace_root=trace_root,
        trace_manifest=trace_manifest,
        trace_subset=trace_subset,
        max_tokens=None,
        strict_singleton=True,
        verbose=False,
    ):
        category = token.category
        quota = quotas.get(category, 0)
        if quota <= 0:
            continue

        seen[category] += 1
        bucket = reservoirs[category]

        if len(bucket) < quota:
            bucket.append(token)
            continue

        replacement = rng.randrange(seen[category])
        if replacement < quota:
            bucket[replacement] = token

    sampled = tuple(
        sorted(
            (
                token
                for category in sorted(reservoirs)
                for token in reservoirs[category]
            ),
            key=lambda token: token.token_id,
        )
    )

    if len(sampled) != sample_tokens:
        raise DecodeOptimalityProbeError(
            f"实际抽样 {len(sampled)} 个 Token，期望 {sample_tokens}。"
        )

    sampled_category_counts = Counter(token.category for token in sampled)
    sampled_files = {token.relative_file for token in sampled}

    sampling = DecodeSamplingSummary(
        protocol="heldout_proportional_category_stratified_reservoir",
        trace_subset=trace_subset,
        sample_seed=sample_seed,
        requested_tokens=sample_tokens,
        source_token_count=total_tokens,
        source_file_count=len(source_files),
        sampled_token_count=len(sampled),
        sampled_file_count=len(sampled_files),
        source_category_counts={
            category: category_counts[category]
            for category in sorted(category_counts)
        },
        sampled_category_counts={
            category: sampled_category_counts[category]
            for category in sorted(sampled_category_counts)
        },
    )

    return sampled, sampling


def evaluate_decode_optimality(
    *,
    index: RuntimeIndex,
    trace_root: Path | str = DEFAULT_TRACE_ROOT,
    trace_manifest: Path | str | None = None,
    trace_subset: str = EVALUATION_SUBSET,
    sample_tokens: int = 100,
    sample_seed: int = 42,
    layers: tuple[int, ...] | None = None,
    max_instances: int | None = None,
    time_limit_seconds: float = 5.0,
    solver_workers: int = 8,
    verbose: bool = True,
) -> tuple[
    DecodeOptimalitySummary,
    tuple[DecodeOptimalityRecord, ...],
    DecodeSamplingSummary,
]:
    if sample_tokens <= 0:
        raise DecodeOptimalityProbeError("sample_tokens 必须大于 0。")
    if max_instances is not None and max_instances <= 0:
        raise DecodeOptimalityProbeError("max_instances 必须大于 0。")

    if layers is None:
        layers = tuple(range(index.num_layers))

    for layer_id in layers:
        if not 0 <= layer_id < index.num_layers:
            raise DecodeOptimalityProbeError(f"layer_id={layer_id} 超出范围。")

    sampled_tokens, sampling = sample_decode_tokens_stratified(
        trace_root=trace_root,
        trace_manifest=trace_manifest,
        trace_subset=trace_subset,
        sample_tokens=sample_tokens,
        sample_seed=sample_seed,
        verbose=verbose,
    )

    rules = ExecutionRules()
    records: list[DecodeOptimalityRecord] = []
    instance_id = 0

    for sample_index, token in enumerate(sampled_tokens):
        if verbose:
            print(
                f"[DecodeOptimalityToken] {sample_index + 1}/{len(sampled_tokens)} "
                f"token={token.token_id}, category={token.category}, "
                f"file={token.relative_file}, segment={token.segment_index}"
            )

        for layer_id in layers:
            route = token.routed_experts_by_layer[layer_id]

            # 当前 Baseline 中，跨层 active state 不会命中本层 WC，且 startup==switch，
            # 因此 cold-state 单层 latency 与正式 Decode layer latency 口径一致。
            greedy = schedule_layer(
                index=index,
                layer_id=layer_id,
                routed_expert_ids=route,
                rules=rules,
                initial_active_cube_by_subcube=None,
                charge_initial_activation=True,
            )

            cp = solve_decode_layer_optimal(
                index=index,
                layer_id=layer_id,
                routed_expert_ids=route,
                rules=rules,
                time_limit_seconds=time_limit_seconds,
                num_workers=solver_workers,
                greedy_upper_bound_cycles=greedy.total_cycles,
                hint_start_times=_greedy_hint(greedy),
                validate_solution=True,
            )

            if cp.objective_cycles is not None and cp.objective_cycles > greedy.total_cycles:
                raise DecodeOptimalityProbeError(
                    f"CP-SAT 比 Greedy 更差：token={token.token_id}, layer={layer_id}, "
                    f"greedy={greedy.total_cycles}, cp={cp.objective_cycles}。"
                )

            record = _make_record(
                instance_id=instance_id,
                token=token,
                layer_id=layer_id,
                route=route,
                greedy_cycles=greedy.total_cycles,
                cp=cp,
            )
            records.append(record)
            instance_id += 1

            if verbose and (
                not record.proven_optimal
                or (record.improvement_cycles or 0) > 0
            ):
                opt_text = (
                    str(record.cp_sat_cycles)
                    if record.cp_sat_cycles is not None
                    else "NA"
                )
                gap_text = (
                    f"{record.gap_vs_opt_percent:.2f}%"
                    if record.gap_vs_opt_percent is not None
                    else "unproven"
                )
                print(
                    f"  [Interesting] #{record.instance_id} "
                    f"L{record.layer_id}, greedy={record.greedy_cycles}, "
                    f"cp={opt_text}, status={record.cp_sat_status}, "
                    f"gap={gap_text}, time={record.wall_time_seconds:.3f}s"
                )

            if max_instances is not None and len(records) >= max_instances:
                summary = build_summary(tuple(records))
                return summary, tuple(records), sampling

    summary = build_summary(tuple(records))
    return summary, tuple(records), sampling


def print_sampling_summary(sampling: DecodeSamplingSummary) -> None:
    print("\n========== Decode Sampling Summary ==========")
    print(f"Protocol：{sampling.protocol}")
    print(f"Subset：{sampling.trace_subset}")
    print(f"Seed：{sampling.sample_seed}")
    print(f"Source Tokens：{sampling.source_token_count}")
    print(f"Source Files：{sampling.source_file_count}")
    print(f"Sampled Tokens：{sampling.sampled_token_count}")
    print(f"Sampled Files：{sampling.sampled_file_count}")
    print("Sampled Category Counts：")
    for category, count in sampling.sampled_category_counts.items():
        print(f"  {category}: {count}")


def print_summary(summary: DecodeOptimalitySummary) -> None:
    def fmt(value: float | None, digits: int = 2) -> str:
        return "NA" if value is None else f"{value:.{digits}f}"

    print("\n========== Decode CP-SAT Optimality Probe ==========")
    print(f"Instances：{summary.instance_count}")
    print(f"Feasible：{summary.feasible_count}")
    print(
        f"OPTIMAL Proven：{summary.optimal_proven_count} "
        f"({summary.optimal_proven_rate:.2%})"
    )
    print(f"Greedy Mean Layer Cycles：{summary.greedy_mean_cycles:.4f}")
    print(
        "Proven OPT Mean Layer Cycles："
        f"{fmt(summary.proven_optimal_mean_cycles, 4)}"
    )
    print(
        "Greedy Already Optimal："
        f"{summary.greedy_already_optimal_count} "
        f"({summary.greedy_already_optimal_rate:.2%} of proven instances)"
    )
    print(f"Mean Gap vs OPT：{fmt(summary.mean_gap_vs_opt_percent)}%")
    print(f"P50 Gap vs OPT：{fmt(summary.p50_gap_vs_opt_percent)}%")
    print(f"P95 Gap vs OPT：{fmt(summary.p95_gap_vs_opt_percent)}%")
    print(f"Max Gap vs OPT：{fmt(summary.max_gap_vs_opt_percent)}%")
    print(
        "Mean Improvement："
        f"{fmt(summary.mean_improvement_cycles, 4)} cycles/layer"
    )
    print(f"Max Improvement：{summary.max_improvement_cycles}")
    print(
        "Solver Total Wall Time："
        f"{summary.total_solver_wall_time_seconds:.3f}s"
    )


def save_result(
    *,
    output_path: Path | str,
    summary: DecodeOptimalitySummary,
    records: tuple[DecodeOptimalityRecord, ...],
    sampling: DecodeSamplingSummary,
    mapping_path: Path | str,
    trace_root: Path | str,
    trace_manifest: Path | str | None,
    trace_subset: str,
    time_limit_seconds: float,
    solver_workers: int,
    layers: tuple[int, ...],
    max_instances: int | None,
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_version": 2,
        "purpose": "Decode layer Greedy vs CP-SAT optimality gap",
        "metric_scope": "MoE Expert Decode Layer only; not full TPOT",
        "mapping": str(Path(mapping_path).resolve()),
        "trace_root": str(Path(trace_root).resolve()),
        "trace_protocol": {
            "manifest": (
                str(Path(trace_manifest).resolve())
                if trace_manifest is not None
                else None
            ),
            "subset": trace_subset,
        },
        "sampling": asdict(sampling),
        "probe_config": {
            "layers": list(layers),
            "max_instances": max_instances,
        },
        "solver": {
            "name": "OR-Tools CP-SAT",
            "time_limit_seconds_per_instance": time_limit_seconds,
            "workers": solver_workers,
            "OPTIMAL_means_proven_global_optimum": True,
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
            "从 Held-out Decode 中按类别比例分层随机抽 Token×Layer，"
            "用 CP-SAT 测当前 Greedy 距离理论最优的差距。"
        )
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument(
        "--trace-manifest",
        type=Path,
        default=None,
        help="正式实验传 80/20 split manifest。",
    )
    parser.add_argument(
        "--trace-subset",
        choices=TRACE_SUBSETS,
        default=EVALUATION_SUBSET,
    )
    parser.add_argument(
        "--sample-tokens",
        "--max-tokens",
        dest="sample_tokens",
        type=int,
        default=100,
        help=(
            "从指定 subset 中按类别比例分层随机抽多少个 Decode Token；"
            "--max-tokens 作为兼容别名，但不再表示顺序截断。"
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="all 或逗号分隔，例如 0,7,20,48,57。",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help=(
            "可选硬上限。正式 all-layer 实验建议不设置，"
            "避免在最后一个 Token 中途截断。"
        ),
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=5.0,
        help="每个 Layer 实例 CP-SAT 最长求解秒数。",
    )
    parser.add_argument(
        "--solver-workers",
        type=int,
        default=8,
        help="单个 CP-SAT 实例内部搜索线程数。",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    index = load_runtime_index(args.mapping)
    layers = _parse_layers(args.layers, index.num_layers)

    try:
        summary, records, sampling = evaluate_decode_optimality(
            index=index,
            trace_root=args.root,
            trace_manifest=args.trace_manifest,
            trace_subset=args.trace_subset,
            sample_tokens=args.sample_tokens,
            sample_seed=args.sample_seed,
            layers=layers,
            max_instances=args.max_instances,
            time_limit_seconds=args.time_limit,
            solver_workers=args.solver_workers,
            verbose=not args.quiet,
        )
    except DecodeOptimalSolverError as exc:
        raise SystemExit(str(exc)) from exc

    print_sampling_summary(sampling)
    print_summary(summary)

    if not args.no_save:
        saved = save_result(
            output_path=args.output,
            summary=summary,
            records=records,
            sampling=sampling,
            mapping_path=args.mapping,
            trace_root=args.root,
            trace_manifest=args.trace_manifest,
            trace_subset=args.trace_subset,
            time_limit_seconds=args.time_limit,
            solver_workers=args.solver_workers,
            layers=layers,
            max_instances=args.max_instances,
        )
        print(f"\nSaved：{saved}")


if __name__ == "__main__":
    main()