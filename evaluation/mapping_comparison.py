"""
Mapping baseline comparison summary.

Pairing is fixed to trace_aware (Greedy + Local Search).

Deterministic modes:
    round_robin / least_loaded / frequency_aware / trace_aware
run once.

Random:
    if random_seeds/seed_<N>/ exists, aggregate multiple seeds and report Mean ± Std.
    otherwise fall back to the historical single random/ result.

All runs must use the same:
- Pairing cost
- Profile fingerprint
- Held-out Evaluation fingerprint
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "results" / "experiments" / "mapping_baselines"

MAPPING_MODES = (
    "round_robin",
    "random",
    "least_loaded",
    "frequency_aware",
    "trace_aware",
)

DISPLAY_NAMES = {
    "round_robin": "Round-Robin",
    "random": "Random",
    "least_loaded": "Least-Loaded",
    "frequency_aware": "Frequency-aware",
    "trace_aware": "Trace-aware",
}


class MappingComparisonError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MappingComparisonMetrics:
    mode: str
    display_name: str
    seed_count: int
    seeds: tuple[int, ...]

    pairing_cost: float
    pairing_cost_std: float

    mapping_conflict_cost: float
    mapping_conflict_cost_std: float
    pre_conflict_cost: float
    pre_conflict_cost_std: float
    down_conflict_cost: float
    down_conflict_cost_std: float

    prefill_mean_latency: float
    prefill_mean_latency_std: float
    prefill_mean_cycles_per_input_token: float
    prefill_mean_cycles_per_input_token_std: float
    decode_mean_cycles_per_token: float
    decode_mean_cycles_per_token_std: float
    decode_p95_cycles_per_token: float
    decode_p95_cycles_per_token_std: float

    profile_file_count: int
    evaluation_file_count: int
    profile_fingerprint: str | None
    evaluation_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class MappingImprovement:
    mode: str
    conflict_reduction_vs_round_robin_percent: float
    prefill_mean_improvement_vs_round_robin_percent: float
    decode_mean_improvement_vs_round_robin_percent: float
    decode_p95_improvement_vs_round_robin_percent: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MappingComparisonError(f"文件不存在：{path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise MappingComparisonError(f"无法读取 JSON：{path}") from exc
    if not isinstance(data, dict):
        raise MappingComparisonError(f"JSON 最外层必须是 dict：{path}")
    return data


def _number(obj: dict, key: str, context: str) -> float:
    value = obj.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MappingComparisonError(f"{context}.{key} 不是数值")
    return float(value)


def _integer(obj: dict, key: str, context: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise MappingComparisonError(f"{context}.{key} 不是整数")
    return int(value)


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def reduction_percent(baseline: float, current: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - current) / baseline * 100.0


def _load_one_result(
    *,
    mode_dir: Path,
    expected_mode: str,
    expected_seed: int | None,
) -> dict[str, Any]:
    mapping = _load_json(mode_dir / "mapping.json")
    summary = _load_json(mode_dir / "phase_evaluation_summary.json")

    pairing = mapping.get("pairing")
    mapper = mapping.get("subcube_mapping")
    if not isinstance(pairing, dict) or not isinstance(mapper, dict):
        raise MappingComparisonError(f"{mode_dir}: 缺少 pairing/subcube_mapping")

    if pairing.get("mode") != "trace_aware":
        raise MappingComparisonError(
            f"{mode_dir}: Mapping comparison 必须固定 pairing=trace_aware"
        )
    if not pairing.get("local_search_enabled", False):
        raise MappingComparisonError(
            f"{mode_dir}: 固定 Pairing 必须启用 Local Search"
        )
    if mapper.get("mode") != expected_mode:
        raise MappingComparisonError(
            f"{mode_dir}: subcube_mapping.mode={mapper.get('mode')!r}, "
            f"expected={expected_mode!r}"
        )

    if expected_mode == "random" and expected_seed is not None:
        stored_seed = mapper.get("random_seed")
        # 兼容 seed 也可能记录在 mapping_policy / 顶层子字段中；
        # build_output_dict 当前版本应保存 random_seed。
        if stored_seed is not None and int(stored_seed) != int(expected_seed):
            raise MappingComparisonError(
                f"{mode_dir}: random_seed={stored_seed}, expected={expected_seed}"
            )

    pairing_cost = _integer(
        pairing,
        "total_routed_up_coactivation_cost",
        "pairing",
    )
    pre = _integer(mapper, "pre_conflict_cost", "subcube_mapping")
    down = _integer(mapper, "down_conflict_cost", "subcube_mapping")
    total = _integer(mapper, "total_conflict_cost", "subcube_mapping")
    if pre + down != total:
        raise MappingComparisonError(
            f"{mode_dir}: conflict 不一致：pre={pre}, down={down}, total={total}"
        )

    prefill = summary.get("prefill")
    decode = summary.get("decode")
    if not isinstance(prefill, dict) or not isinstance(decode, dict):
        raise MappingComparisonError(f"{mode_dir}: Phase summary 缺少 prefill/decode")

    latency = prefill.get("latency_cycles")
    cpt = prefill.get("cycles_per_input_token")
    decode_cycles = decode.get("cycles_per_token")
    if not all(isinstance(x, dict) for x in (latency, cpt, decode_cycles)):
        raise MappingComparisonError(f"{mode_dir}: Phase summary 指标结构不完整")

    trace = mapping.get("trace", {})
    profile_protocol = trace.get("profile_protocol", {})
    phase_protocol = summary.get("protocol", {})

    if profile_protocol.get("subset") != "profile":
        raise MappingComparisonError(f"{mode_dir}: Mapping 必须只使用 profile subset")
    if phase_protocol.get("evaluation_subset") != "evaluation":
        raise MappingComparisonError(f"{mode_dir}: Evaluation 必须只使用 held-out subset")

    return {
        "pairing_cost": float(pairing_cost),
        "mapping_conflict_cost": float(total),
        "pre_conflict_cost": float(pre),
        "down_conflict_cost": float(down),
        "prefill_mean_latency": _number(
            latency, "mean", "prefill.latency_cycles"
        ),
        "prefill_mean_cycles_per_input_token": _number(
            cpt, "mean", "prefill.cycles_per_input_token"
        ),
        "decode_mean_cycles_per_token": _number(
            decode_cycles, "mean", "decode.cycles_per_token"
        ),
        "decode_p95_cycles_per_token": _number(
            decode_cycles, "p95", "decode.cycles_per_token"
        ),
        "profile_file_count": int(trace.get("file_count", 0)),
        "evaluation_file_count": int(
            phase_protocol.get("evaluation_file_count", 0)
        ),
        "profile_fingerprint": profile_protocol.get("file_fingerprint"),
        "evaluation_fingerprint": phase_protocol.get("evaluation_fingerprint"),
    }


def _aggregate(
    *,
    mode: str,
    seeds: tuple[int, ...],
    rows: list[dict[str, Any]],
) -> MappingComparisonMetrics:
    if not rows:
        raise MappingComparisonError(f"{mode}: 没有可汇总结果")

    # 每个 seed/模式必须使用完全一致的数据集合。
    profile_fp = {row["profile_fingerprint"] for row in rows}
    evaluation_fp = {row["evaluation_fingerprint"] for row in rows}
    profile_counts = {row["profile_file_count"] for row in rows}
    evaluation_counts = {row["evaluation_file_count"] for row in rows}
    pairing_costs = {row["pairing_cost"] for row in rows}

    if len(profile_fp) != 1:
        raise MappingComparisonError(f"{mode}: 不同 seed 使用了不同 Profile")
    if len(evaluation_fp) != 1:
        raise MappingComparisonError(f"{mode}: 不同 seed 使用了不同 Evaluation")
    if len(profile_counts) != 1 or len(evaluation_counts) != 1:
        raise MappingComparisonError(f"{mode}: 不同 seed 文件数量不一致")
    if len(pairing_costs) != 1:
        raise MappingComparisonError(f"{mode}: 不同 seed 的 Pairing Cost 不一致")

    def vals(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    return MappingComparisonMetrics(
        mode=mode,
        display_name=DISPLAY_NAMES[mode],
        seed_count=len(rows),
        seeds=seeds,
        pairing_cost=_mean(vals("pairing_cost")),
        pairing_cost_std=_sample_std(vals("pairing_cost")),
        mapping_conflict_cost=_mean(vals("mapping_conflict_cost")),
        mapping_conflict_cost_std=_sample_std(vals("mapping_conflict_cost")),
        pre_conflict_cost=_mean(vals("pre_conflict_cost")),
        pre_conflict_cost_std=_sample_std(vals("pre_conflict_cost")),
        down_conflict_cost=_mean(vals("down_conflict_cost")),
        down_conflict_cost_std=_sample_std(vals("down_conflict_cost")),
        prefill_mean_latency=_mean(vals("prefill_mean_latency")),
        prefill_mean_latency_std=_sample_std(vals("prefill_mean_latency")),
        prefill_mean_cycles_per_input_token=_mean(
            vals("prefill_mean_cycles_per_input_token")
        ),
        prefill_mean_cycles_per_input_token_std=_sample_std(
            vals("prefill_mean_cycles_per_input_token")
        ),
        decode_mean_cycles_per_token=_mean(
            vals("decode_mean_cycles_per_token")
        ),
        decode_mean_cycles_per_token_std=_sample_std(
            vals("decode_mean_cycles_per_token")
        ),
        decode_p95_cycles_per_token=_mean(
            vals("decode_p95_cycles_per_token")
        ),
        decode_p95_cycles_per_token_std=_sample_std(
            vals("decode_p95_cycles_per_token")
        ),
        profile_file_count=rows[0]["profile_file_count"],
        evaluation_file_count=rows[0]["evaluation_file_count"],
        profile_fingerprint=rows[0]["profile_fingerprint"],
        evaluation_fingerprint=rows[0]["evaluation_fingerprint"],
    )


def load_mode(
    root: Path,
    mode: str,
    random_seeds: tuple[int, ...],
) -> MappingComparisonMetrics:
    if mode != "random":
        row = _load_one_result(
            mode_dir=root / mode,
            expected_mode=mode,
            expected_seed=None,
        )
        return _aggregate(
            mode=mode,
            seeds=tuple(),
            rows=[row],
        )

    # 优先使用多 seed 正式结果。
    random_root = root / "random_seeds"
    seed_rows: list[dict[str, Any]] = []
    found_seeds: list[int] = []

    for seed in random_seeds:
        seed_dir = random_root / f"seed_{seed}"
        if not (
            (seed_dir / "mapping.json").exists()
            and (seed_dir / "phase_evaluation_summary.json").exists()
        ):
            continue

        seed_rows.append(
            _load_one_result(
                mode_dir=seed_dir,
                expected_mode="random",
                expected_seed=seed,
            )
        )
        found_seeds.append(seed)

    if seed_rows:
        if len(seed_rows) != len(random_seeds):
            missing = sorted(set(random_seeds) - set(found_seeds))
            raise MappingComparisonError(
                "Random 多 seed 结果不完整；"
                f"已找到={found_seeds}, 缺少={missing}"
            )

        return _aggregate(
            mode="random",
            seeds=tuple(found_seeds),
            rows=seed_rows,
        )

    # 兼容旧的单 seed random/ 结果。
    row = _load_one_result(
        mode_dir=root / "random",
        expected_mode="random",
        expected_seed=None,
    )
    return _aggregate(
        mode="random",
        seeds=tuple(),
        rows=[row],
    )


def _fmt_mean_std(
    mean_value: float,
    std_value: float,
    decimals: int = 2,
) -> str:
    if std_value == 0.0:
        return f"{mean_value:.{decimals}f}"
    return (
        f"{mean_value:.{decimals}f}"
        f"±{std_value:.{decimals}f}"
    )


def _fmt_int_mean_std(mean_value: float, std_value: float) -> str:
    if std_value == 0.0:
        return f"{int(round(mean_value)):,}"
    return f"{int(round(mean_value)):,}±{int(round(std_value)):,}"


def print_table(metrics: list[MappingComparisonMetrics]) -> None:
    print("\n" + "=" * 122)
    print(
        "Mapping Baseline Comparison "
        "(Pairing fixed to Trace-aware + Local Search)"
    )
    print("=" * 122)
    header = (
        f"{'Mapping':<25}"
        f"{'Map Conflict':>24}"
        f"{'Pre Conflict':>22}"
        f"{'Down Conflict':>22}"
        f"{'Prefill':>18}"
        f"{'Decode':>18}"
        f"{'P95':>16}"
    )
    print(header)
    print("-" * len(header))

    for item in metrics:
        label = item.display_name
        if item.mode == "random" and item.seed_count > 1:
            label += f" ({item.seed_count} seeds)"

        print(
            f"{label:<25}"
            f"{_fmt_int_mean_std(item.mapping_conflict_cost, item.mapping_conflict_cost_std):>24}"
            f"{_fmt_int_mean_std(item.pre_conflict_cost, item.pre_conflict_cost_std):>22}"
            f"{_fmt_int_mean_std(item.down_conflict_cost, item.down_conflict_cost_std):>22}"
            f"{_fmt_mean_std(item.prefill_mean_latency, item.prefill_mean_latency_std):>18}"
            f"{_fmt_mean_std(item.decode_mean_cycles_per_token, item.decode_mean_cycles_per_token_std):>18}"
            f"{_fmt_mean_std(item.decode_p95_cycles_per_token, item.decode_p95_cycles_per_token_std):>16}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="汇总 Mapping baseline 对比结果；Random 支持多 seed Mean ± Std。"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MAPPING_MODES,
        default=list(MAPPING_MODES),
    )
    parser.add_argument(
        "--random-seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if len(set(args.random_seeds)) != len(args.random_seeds):
        parser.error("--random-seeds 不能包含重复值")

    root = args.root.resolve()
    random_seeds = tuple(args.random_seeds)
    metrics = [
        load_mode(root, mode, random_seeds)
        for mode in args.modes
    ]

    if "round_robin" not in args.modes:
        raise MappingComparisonError("必须包含 round_robin 作为共同 baseline")

    # 所有 Mapping 必须真正固定同一个 Pairing。
    pairing_costs = {
        round(item.pairing_cost, 9)
        for item in metrics
    }
    if len(pairing_costs) != 1:
        raise MappingComparisonError(
            "不同 Mapping 模式的 Pairing Cost 不一致，说明没有真正固定 Pairing。"
        )

    profile_fingerprints = {
        item.profile_fingerprint
        for item in metrics
    }
    evaluation_fingerprints = {
        item.evaluation_fingerprint
        for item in metrics
    }
    if len(profile_fingerprints) != 1:
        raise MappingComparisonError(
            "不同 Mapping 使用了不同 Profile 文件集合。"
        )
    if len(evaluation_fingerprints) != 1:
        raise MappingComparisonError(
            "不同 Mapping 使用了不同 Evaluation 文件集合。"
        )

    baseline = next(
        item
        for item in metrics
        if item.mode == "round_robin"
    )

    improvements = [
        MappingImprovement(
            mode=item.mode,
            conflict_reduction_vs_round_robin_percent=reduction_percent(
                baseline.mapping_conflict_cost,
                item.mapping_conflict_cost,
            ),
            prefill_mean_improvement_vs_round_robin_percent=reduction_percent(
                baseline.prefill_mean_latency,
                item.prefill_mean_latency,
            ),
            decode_mean_improvement_vs_round_robin_percent=reduction_percent(
                baseline.decode_mean_cycles_per_token,
                item.decode_mean_cycles_per_token,
            ),
            decode_p95_improvement_vs_round_robin_percent=reduction_percent(
                baseline.decode_p95_cycles_per_token,
                item.decode_p95_cycles_per_token,
            ),
        )
        for item in metrics
    ]

    print_table(metrics)

    print("\nImprovement vs Round-Robin (positive = better; Random uses seed mean)")
    for item in improvements:
        suffix = ""
        metric = next(x for x in metrics if x.mode == item.mode)
        if metric.mode == "random" and metric.seed_count > 1:
            suffix = f"  seeds={list(metric.seeds)}"

        print(
            f"{DISPLAY_NAMES[item.mode]:<24} "
            f"Conflict={item.conflict_reduction_vs_round_robin_percent:>7.2f}%  "
            f"Prefill={item.prefill_mean_improvement_vs_round_robin_percent:>7.2f}%  "
            f"Decode={item.decode_mean_improvement_vs_round_robin_percent:>7.2f}%  "
            f"P95={item.decode_p95_improvement_vs_round_robin_percent:>7.2f}%"
            f"{suffix}"
        )

    output = (
        args.output.resolve()
        if args.output is not None
        else root / "mapping_comparison_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    random_metric = next(
        (x for x in metrics if x.mode == "random"),
        None,
    )

    payload = {
        "comparison_version": 2,
        "scope": (
            "Mapping baseline comparison; Pairing fixed to trace_aware + Local Search; "
            "Profile and held-out Evaluation split; Random reports multi-seed Mean ± Std"
        ),
        "protocol": {
            "pairing_mode": "trace_aware",
            "local_search_enabled": True,
            "pairing_cost": metrics[0].pairing_cost,
            "profile_file_count": metrics[0].profile_file_count,
            "evaluation_file_count": metrics[0].evaluation_file_count,
            "profile_fingerprint": metrics[0].profile_fingerprint,
            "evaluation_fingerprint": metrics[0].evaluation_fingerprint,
            "random_seeds": (
                list(random_metric.seeds)
                if random_metric is not None
                else []
            ),
        },
        "modes": list(args.modes),
        "metrics": [asdict(x) for x in metrics],
        "improvements_vs_round_robin": [
            asdict(x)
            for x in improvements
        ],
    }

    with output.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved：{output}")


if __name__ == "__main__":
    main()
