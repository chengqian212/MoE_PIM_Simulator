"""
Formal 2x2 Pairing/Mapping ablation summary.

The four groups are:
1. Naive        = Sequential Pairing + Round-Robin Mapping
2. Pairing Only = Trace-aware Pairing + Local Search + Round-Robin Mapping
3. Mapping Only = Sequential Pairing + Trace-aware Mapping
4. Full         = Trace-aware Pairing + Local Search + Trace-aware Mapping

This version is strict:
- checks actual pairing/mapping modes from mapping.json
- checks trace_aware groups have Local Search enabled
- checks all groups use the same Profile fingerprint
- checks all groups use the same Held-out Evaluation fingerprint
- checks Profile/Evaluation subsets are correct
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "ablation_formal"
)

DEFAULT_NAIVE_ROOT = DEFAULT_ROOT / "naive"

DEFAULT_PAIRING_ONLY_ROOT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "mapping_baselines"
    / "round_robin"
)

DEFAULT_MAPPING_ONLY_ROOT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "pairing_baselines"
    / "sequential"
)

DEFAULT_FULL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "mapping_baselines"
    / "trace_aware"
)

DEFAULT_OUTPUT = DEFAULT_ROOT / "ablation_summary.json"


class AblationSummaryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AblationMetrics:
    key: str
    experiment: str
    pairing_mode: str
    mapping_mode: str
    local_search_enabled: bool

    pairing_cost: int
    mapping_conflict_cost: int
    pre_conflict_cost: int
    down_conflict_cost: int

    prefill_mean_latency: float
    prefill_p50_latency: float
    prefill_p95_latency: float
    prefill_p99_latency: float
    prefill_max_latency: float
    prefill_mean_cycles_per_input_token: float
    prefill_global_cycles_per_input_token: float

    decode_mean_cycles_per_token: float
    decode_p50_cycles_per_token: float
    decode_p95_cycles_per_token: float
    decode_p99_cycles_per_token: float
    decode_max_cycles_per_token: float

    profile_file_count: int
    evaluation_file_count: int
    profile_fingerprint: str | None
    evaluation_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class ImprovementMetrics:
    key: str
    experiment: str
    prefill_mean_improvement_percent: float
    prefill_cycles_per_input_token_improvement_percent: float
    decode_mean_improvement_percent: float
    decode_p95_improvement_percent: float
    pairing_cost_reduction_percent: float
    mapping_conflict_cost_reduction_percent: float


EXPECTED = {
    "naive": {
        "experiment": "Naive",
        "pairing_mode": "sequential",
        "mapping_mode": "round_robin",
        "local_search_enabled": False,
    },
    "pairing_only": {
        "experiment": "Pairing Only",
        "pairing_mode": "trace_aware",
        "mapping_mode": "round_robin",
        "local_search_enabled": True,
    },
    "mapping_only": {
        "experiment": "Mapping Only",
        "pairing_mode": "sequential",
        "mapping_mode": "trace_aware",
        "local_search_enabled": False,
    },
    "full": {
        "experiment": "Full",
        "pairing_mode": "trace_aware",
        "mapping_mode": "trace_aware",
        "local_search_enabled": True,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AblationSummaryError(f"文件不存在：{path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise AblationSummaryError(f"无法读取 JSON：{path}") from exc
    if not isinstance(data, dict):
        raise AblationSummaryError(f"JSON 最外层必须是 dict：{path}")
    return data


def _dict(obj: dict, key: str, context: str) -> dict:
    value = obj.get(key)
    if not isinstance(value, dict):
        raise AblationSummaryError(f"{context} 缺少 dict 字段 {key}")
    return value


def _number(obj: dict, key: str, context: str) -> float:
    value = obj.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AblationSummaryError(f"{context}.{key} 不是数值")
    return float(value)


def _int(obj: dict, key: str, context: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise AblationSummaryError(f"{context}.{key} 不是整数")
    return int(value)


def _reduction(baseline: float, current: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - current) / baseline * 100.0


def _load_group(key: str, root: Path) -> AblationMetrics:
    expected = EXPECTED[key]
    mapping = _load_json(root / "mapping.json")
    summary = _load_json(root / "phase_evaluation_summary.json")

    pairing = _dict(mapping, "pairing", "mapping")
    mapper = _dict(mapping, "subcube_mapping", "mapping")
    trace = _dict(mapping, "trace", "mapping")
    profile_protocol = _dict(trace, "profile_protocol", "mapping.trace")

    actual_pairing = pairing.get("mode")
    actual_mapping = mapper.get("mode")
    local_search = bool(pairing.get("local_search_enabled", False))

    if actual_pairing != expected["pairing_mode"]:
        raise AblationSummaryError(
            f"{key}: pairing.mode={actual_pairing!r}, "
            f"expected={expected['pairing_mode']!r}"
        )
    if actual_mapping != expected["mapping_mode"]:
        raise AblationSummaryError(
            f"{key}: mapping.mode={actual_mapping!r}, "
            f"expected={expected['mapping_mode']!r}"
        )
    if local_search != expected["local_search_enabled"]:
        raise AblationSummaryError(
            f"{key}: local_search_enabled={local_search}, "
            f"expected={expected['local_search_enabled']}"
        )

    if profile_protocol.get("subset") != "profile":
        raise AblationSummaryError(f"{key}: Mapping 必须只使用 profile subset")

    protocol = _dict(summary, "protocol", "phase summary")
    if protocol.get("evaluation_subset") != "evaluation":
        raise AblationSummaryError(
            f"{key}: Prefill/Decode 必须只使用 evaluation subset"
        )

    prefill = _dict(summary, "prefill", "phase summary")
    decode = _dict(summary, "decode", "phase summary")
    latency = _dict(prefill, "latency_cycles", "prefill")
    cpt = _dict(prefill, "cycles_per_input_token", "prefill")
    decode_cycles = _dict(decode, "cycles_per_token", "decode")

    pre_conflict = _int(mapper, "pre_conflict_cost", "subcube_mapping")
    down_conflict = _int(mapper, "down_conflict_cost", "subcube_mapping")
    total_conflict = _int(mapper, "total_conflict_cost", "subcube_mapping")
    if pre_conflict + down_conflict != total_conflict:
        raise AblationSummaryError(
            f"{key}: conflict 不一致 "
            f"{pre_conflict}+{down_conflict}!={total_conflict}"
        )

    return AblationMetrics(
        key=key,
        experiment=expected["experiment"],
        pairing_mode=actual_pairing,
        mapping_mode=actual_mapping,
        local_search_enabled=local_search,
        pairing_cost=_int(
            pairing,
            "total_routed_up_coactivation_cost",
            "pairing",
        ),
        mapping_conflict_cost=total_conflict,
        pre_conflict_cost=pre_conflict,
        down_conflict_cost=down_conflict,
        prefill_mean_latency=_number(latency, "mean", "prefill.latency"),
        prefill_p50_latency=_number(latency, "p50", "prefill.latency"),
        prefill_p95_latency=_number(latency, "p95", "prefill.latency"),
        prefill_p99_latency=_number(latency, "p99", "prefill.latency"),
        prefill_max_latency=_number(latency, "maximum", "prefill.latency"),
        prefill_mean_cycles_per_input_token=_number(
            cpt, "mean", "prefill.cycles_per_input_token"
        ),
        prefill_global_cycles_per_input_token=_number(
            prefill,
            "global_cycles_per_input_token",
            "prefill",
        ),
        decode_mean_cycles_per_token=_number(
            decode_cycles, "mean", "decode.cycles_per_token"
        ),
        decode_p50_cycles_per_token=_number(
            decode_cycles, "p50", "decode.cycles_per_token"
        ),
        decode_p95_cycles_per_token=_number(
            decode_cycles, "p95", "decode.cycles_per_token"
        ),
        decode_p99_cycles_per_token=_number(
            decode_cycles, "p99", "decode.cycles_per_token"
        ),
        decode_max_cycles_per_token=_number(
            decode_cycles, "maximum", "decode.cycles_per_token"
        ),
        profile_file_count=int(trace.get("file_count", 0)),
        evaluation_file_count=int(protocol.get("evaluation_file_count", 0)),
        profile_fingerprint=profile_protocol.get("file_fingerprint"),
        evaluation_fingerprint=protocol.get("evaluation_fingerprint"),
    )


def _validate_protocol(groups: list[AblationMetrics]) -> None:
    profile_fp = {g.profile_fingerprint for g in groups}
    evaluation_fp = {g.evaluation_fingerprint for g in groups}
    profile_counts = {g.profile_file_count for g in groups}
    evaluation_counts = {g.evaluation_file_count for g in groups}

    if len(profile_fp) != 1:
        raise AblationSummaryError(
            "四组使用了不同 Profile 文件集合，实验不可比较。"
        )
    if len(evaluation_fp) != 1:
        raise AblationSummaryError(
            "四组使用了不同 Held-out Evaluation 文件集合。"
        )
    if len(profile_counts) != 1:
        raise AblationSummaryError("四组 Profile file_count 不一致。")
    if len(evaluation_counts) != 1:
        raise AblationSummaryError("四组 Evaluation file_count 不一致。")


def _improvement(naive: AblationMetrics, item: AblationMetrics) -> ImprovementMetrics:
    return ImprovementMetrics(
        key=item.key,
        experiment=item.experiment,
        prefill_mean_improvement_percent=_reduction(
            naive.prefill_mean_latency,
            item.prefill_mean_latency,
        ),
        prefill_cycles_per_input_token_improvement_percent=_reduction(
            naive.prefill_mean_cycles_per_input_token,
            item.prefill_mean_cycles_per_input_token,
        ),
        decode_mean_improvement_percent=_reduction(
            naive.decode_mean_cycles_per_token,
            item.decode_mean_cycles_per_token,
        ),
        decode_p95_improvement_percent=_reduction(
            naive.decode_p95_cycles_per_token,
            item.decode_p95_cycles_per_token,
        ),
        pairing_cost_reduction_percent=_reduction(
            naive.pairing_cost,
            item.pairing_cost,
        ),
        mapping_conflict_cost_reduction_percent=_reduction(
            naive.mapping_conflict_cost,
            item.mapping_conflict_cost,
        ),
    )


def _print(groups: list[AblationMetrics], improvements: list[ImprovementMetrics]) -> None:
    print("\n" + "=" * 104)
    print("Formal 2x2 Pairing / Mapping Ablation")
    print("=" * 104)
    print(
        f"{'Experiment':<16}"
        f"{'Pairing':<16}"
        f"{'Mapping':<16}"
        f"{'Pair Cost':>14}"
        f"{'Map Conflict':>16}"
        f"{'Prefill':>12}"
        f"{'Decode':>12}"
        f"{'P95':>10}"
    )
    print("-" * 112)
    for g in groups:
        pairing_name = (
            "Trace+LS"
            if g.pairing_mode == "trace_aware"
            else "Sequential"
        )
        mapping_name = (
            "Trace-aware"
            if g.mapping_mode == "trace_aware"
            else "Round-Robin"
        )
        print(
            f"{g.experiment:<16}"
            f"{pairing_name:<16}"
            f"{mapping_name:<16}"
            f"{g.pairing_cost:>14,d}"
            f"{g.mapping_conflict_cost:>16,d}"
            f"{g.prefill_mean_latency:>12.2f}"
            f"{g.decode_mean_cycles_per_token:>12.2f}"
            f"{g.decode_p95_cycles_per_token:>10.2f}"
        )

    print("\nImprovement vs Naive (positive = better)")
    for x in improvements:
        print(
            f"{x.experiment:<16}"
            f"Prefill={x.prefill_mean_improvement_percent:>7.2f}%  "
            f"Decode={x.decode_mean_improvement_percent:>7.2f}%  "
            f"P95={x.decode_p95_improvement_percent:>7.2f}%  "
            f"PairCost={x.pairing_cost_reduction_percent:>7.2f}%  "
            f"MapConflict={x.mapping_conflict_cost_reduction_percent:>7.2f}%"
        )

    naive, pairing_only, mapping_only, full = groups

    print("\nCore Findings")
    print(
        "  Pairing contribution (Naive -> Pairing Only): "
        f"Prefill {_reduction(naive.prefill_mean_latency, pairing_only.prefill_mean_latency):.2f}%, "
        f"Decode {_reduction(naive.decode_mean_cycles_per_token, pairing_only.decode_mean_cycles_per_token):.2f}%"
    )
    print(
        "  Mapping contribution (Naive -> Mapping Only): "
        f"Prefill {_reduction(naive.prefill_mean_latency, mapping_only.prefill_mean_latency):.2f}%, "
        f"Decode {_reduction(naive.decode_mean_cycles_per_token, mapping_only.decode_mean_cycles_per_token):.2f}%"
    )
    print(
        "  Full improvement (Naive -> Full): "
        f"Prefill {_reduction(naive.prefill_mean_latency, full.prefill_mean_latency):.2f}%, "
        f"Decode {_reduction(naive.decode_mean_cycles_per_token, full.decode_mean_cycles_per_token):.2f}%"
    )
    print(
        "  Pairing marginal gain on top of Mapping (Mapping Only -> Full): "
        f"Prefill {_reduction(mapping_only.prefill_mean_latency, full.prefill_mean_latency):.3f}%, "
        f"Decode {_reduction(mapping_only.decode_mean_cycles_per_token, full.decode_mean_cycles_per_token):.3f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="正式 2x2 Pairing/Mapping 消融汇总。"
    )
    parser.add_argument("--naive-root", type=Path, default=DEFAULT_NAIVE_ROOT)
    parser.add_argument(
        "--pairing-only-root",
        type=Path,
        default=DEFAULT_PAIRING_ONLY_ROOT,
    )
    parser.add_argument(
        "--mapping-only-root",
        type=Path,
        default=DEFAULT_MAPPING_ONLY_ROOT,
    )
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    roots = {
        "naive": args.naive_root.resolve(),
        "pairing_only": args.pairing_only_root.resolve(),
        "mapping_only": args.mapping_only_root.resolve(),
        "full": args.full_root.resolve(),
    }

    groups = [
        _load_group("naive", roots["naive"]),
        _load_group("pairing_only", roots["pairing_only"]),
        _load_group("mapping_only", roots["mapping_only"]),
        _load_group("full", roots["full"]),
    ]
    _validate_protocol(groups)

    naive = groups[0]
    improvements = [_improvement(naive, g) for g in groups]
    _print(groups, improvements)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ablation_version": 2,
        "scope": (
            "Formal 2x2 Pairing/Mapping ablation; "
            "Profile and held-out Evaluation split"
        ),
        "protocol": {
            "profile_file_count": groups[0].profile_file_count,
            "evaluation_file_count": groups[0].evaluation_file_count,
            "profile_fingerprint": groups[0].profile_fingerprint,
            "evaluation_fingerprint": groups[0].evaluation_fingerprint,
        },
        "experiments": [asdict(g) for g in groups],
        "improvements_vs_naive": [asdict(x) for x in improvements],
        "sources": {
            key: {
                "mapping": str(root / "mapping.json"),
                "phase_summary": str(root / "phase_evaluation_summary.json"),
            }
            for key, root in roots.items()
        },
    }
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved：{output}")


if __name__ == "__main__":
    main()
