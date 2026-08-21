"""
Pairing baseline comparison summary.

This script is intentionally separate from evaluation.ablation_summary:
- ablation_summary.py keeps the existing 2x2 module ablation unchanged;
- this file compares Pairing algorithms while Mapping is fixed to trace_aware.

Expected directory:
results/experiments/pairing_baselines/
    sequential/
    random/
    frequency_aware/
    greedy/
    trace_aware/
    optimal/

Each mode directory contains:
    mapping.json
    prefill_evaluation.json
    decode_fast_evaluation.json
    phase_evaluation_summary.json
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
    / "pairing_baselines"
)

PAIRING_MODES = (
    "sequential",
    "random",
    "frequency_aware",
    "greedy",
    "trace_aware",
    "optimal",
)

DISPLAY_NAMES = {
    "sequential": "Sequential",
    "random": "Random",
    "frequency_aware": "Frequency-aware",
    "greedy": "Coactivation Greedy",
    "trace_aware": "Greedy + Local Search",
    "optimal": "Optimal Matching",
}


class PairingComparisonError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PairingComparisonMetrics:
    mode: str
    display_name: str
    pairing_cost: int
    mapping_conflict_cost: int
    pre_conflict_cost: int
    down_conflict_cost: int
    prefill_mean_latency: float
    prefill_mean_cycles_per_input_token: float
    decode_mean_cycles_per_token: float
    decode_p95_cycles_per_token: float
    profile_file_count: int
    evaluation_file_count: int
    profile_fingerprint: str | None
    evaluation_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class PairingImprovement:
    mode: str
    pairing_cost_reduction_vs_sequential_percent: float
    prefill_mean_improvement_vs_sequential_percent: float
    decode_mean_improvement_vs_sequential_percent: float
    decode_p95_improvement_vs_sequential_percent: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PairingComparisonError(f"文件不存在：{path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise PairingComparisonError(f"无法读取 JSON：{path}") from exc
    if not isinstance(data, dict):
        raise PairingComparisonError(f"JSON 最外层必须是 dict：{path}")
    return data


def _number(obj: dict, key: str, context: str) -> float:
    value = obj.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PairingComparisonError(f"{context}.{key} 不是数值")
    return float(value)


def _integer(obj: dict, key: str, context: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise PairingComparisonError(f"{context}.{key} 不是整数")
    return int(value)


def _mapping_metrics(mapping: dict) -> tuple[int, int, int, int]:
    pairing = mapping.get("pairing")
    mapper = mapping.get("subcube_mapping")
    if not isinstance(pairing, dict) or not isinstance(mapper, dict):
        raise PairingComparisonError("Mapping JSON 缺少 pairing/subcube_mapping")

    pairing_cost = _integer(
        pairing,
        "total_routed_up_coactivation_cost",
        "pairing",
    )
    pre = _integer(mapper, "pre_conflict_cost", "subcube_mapping")
    down = _integer(mapper, "down_conflict_cost", "subcube_mapping")
    total = _integer(mapper, "total_conflict_cost", "subcube_mapping")
    if pre + down != total:
        raise PairingComparisonError(
            f"Mapping conflict 不一致：pre={pre}, down={down}, total={total}"
        )
    return pairing_cost, total, pre, down


def _phase_metrics(summary: dict) -> tuple[float, float, float, float]:
    prefill = summary.get("prefill")
    decode = summary.get("decode")
    if not isinstance(prefill, dict) or not isinstance(decode, dict):
        raise PairingComparisonError("Phase summary 缺少 prefill/decode")

    latency = prefill.get("latency_cycles")
    cpt = prefill.get("cycles_per_input_token")
    decode_cycles = decode.get("cycles_per_token")
    if not all(isinstance(x, dict) for x in (latency, cpt, decode_cycles)):
        raise PairingComparisonError("Phase summary 指标结构不完整")

    return (
        _number(latency, "mean", "prefill.latency_cycles"),
        _number(cpt, "mean", "prefill.cycles_per_input_token"),
        _number(decode_cycles, "mean", "decode.cycles_per_token"),
        _number(decode_cycles, "p95", "decode.cycles_per_token"),
    )


def reduction_percent(baseline: float, current: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - current) / baseline * 100.0


def load_mode(root: Path, mode: str) -> PairingComparisonMetrics:
    mode_dir = root / mode
    mapping = _load_json(mode_dir / "mapping.json")
    summary = _load_json(mode_dir / "phase_evaluation_summary.json")

    pairing_cost, total_conflict, pre_conflict, down_conflict = (
        _mapping_metrics(mapping)
    )
    prefill_mean, prefill_cpt, decode_mean, decode_p95 = _phase_metrics(summary)

    pairing_section = mapping.get("pairing", {})
    actual_mode = pairing_section.get("mode")
    if actual_mode != mode:
        raise PairingComparisonError(
            f"{mode_dir}: mapping pairing.mode={actual_mode!r}, expected={mode!r}"
        )

    mapping_section = mapping.get("subcube_mapping", {})
    if mapping_section.get("mode") != "trace_aware":
        raise PairingComparisonError(
            f"{mode_dir}: Pairing comparison 必须固定 mapping=trace_aware"
        )

    trace_section = mapping.get("trace", {})
    profile_protocol = trace_section.get("profile_protocol", {})
    phase_protocol = summary.get("protocol", {})
    if profile_protocol.get("subset") != "profile":
        raise PairingComparisonError(
            f"{mode_dir}: Mapping 必须只使用 profile subset"
        )
    if phase_protocol.get("evaluation_subset") != "evaluation":
        raise PairingComparisonError(
            f"{mode_dir}: Prefill/Decode 必须只使用 evaluation subset"
        )

    profile_file_count = int(trace_section.get("file_count", 0))
    evaluation_file_count = int(phase_protocol.get("evaluation_file_count", 0))

    return PairingComparisonMetrics(
        mode=mode,
        display_name=DISPLAY_NAMES[mode],
        pairing_cost=pairing_cost,
        mapping_conflict_cost=total_conflict,
        pre_conflict_cost=pre_conflict,
        down_conflict_cost=down_conflict,
        prefill_mean_latency=prefill_mean,
        prefill_mean_cycles_per_input_token=prefill_cpt,
        decode_mean_cycles_per_token=decode_mean,
        decode_p95_cycles_per_token=decode_p95,
        profile_file_count=profile_file_count,
        evaluation_file_count=evaluation_file_count,
        profile_fingerprint=profile_protocol.get("file_fingerprint"),
        evaluation_fingerprint=phase_protocol.get("evaluation_fingerprint"),
    )


def print_table(metrics: list[PairingComparisonMetrics]) -> None:
    print("\n" + "=" * 102)
    print("Pairing Baseline Comparison (Mapping fixed to Trace-aware)")
    print("=" * 102)
    header = (
        f"{'Pairing':<26}"
        f"{'Pairing Cost':>16}"
        f"{'Map Conflict':>16}"
        f"{'Prefill':>14}"
        f"{'Decode':>14}"
        f"{'P95':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in metrics:
        print(
            f"{item.display_name:<26}"
            f"{item.pairing_cost:>16,d}"
            f"{item.mapping_conflict_cost:>16,d}"
            f"{item.prefill_mean_latency:>14.2f}"
            f"{item.decode_mean_cycles_per_token:>14.2f}"
            f"{item.decode_p95_cycles_per_token:>10.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="汇总 Pairing baseline 对比结果。"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=PAIRING_MODES,
        default=list(PAIRING_MODES),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    metrics = [load_mode(root, mode) for mode in args.modes]

    profile_fingerprints = {item.profile_fingerprint for item in metrics}
    evaluation_fingerprints = {item.evaluation_fingerprint for item in metrics}
    if len(profile_fingerprints) != 1:
        raise PairingComparisonError(
            "不同 Pairing 模式使用了不同 Profile 文件集合，实验不公平。"
        )
    if len(evaluation_fingerprints) != 1:
        raise PairingComparisonError(
            "不同 Pairing 模式使用了不同 Held-out Evaluation 文件集合。"
        )

    if "sequential" not in args.modes:
        raise PairingComparisonError("必须包含 sequential 作为共同 baseline")

    sequential = next(x for x in metrics if x.mode == "sequential")
    improvements = [
        PairingImprovement(
            mode=item.mode,
            pairing_cost_reduction_vs_sequential_percent=reduction_percent(
                sequential.pairing_cost,
                item.pairing_cost,
            ),
            prefill_mean_improvement_vs_sequential_percent=reduction_percent(
                sequential.prefill_mean_latency,
                item.prefill_mean_latency,
            ),
            decode_mean_improvement_vs_sequential_percent=reduction_percent(
                sequential.decode_mean_cycles_per_token,
                item.decode_mean_cycles_per_token,
            ),
            decode_p95_improvement_vs_sequential_percent=reduction_percent(
                sequential.decode_p95_cycles_per_token,
                item.decode_p95_cycles_per_token,
            ),
        )
        for item in metrics
    ]

    print_table(metrics)

    print("\nImprovement vs Sequential (positive = better)")
    for item in improvements:
        print(
            f"{DISPLAY_NAMES[item.mode]:<26} "
            f"PairCost={item.pairing_cost_reduction_vs_sequential_percent:>7.2f}%  "
            f"Prefill={item.prefill_mean_improvement_vs_sequential_percent:>7.2f}%  "
            f"Decode={item.decode_mean_improvement_vs_sequential_percent:>7.2f}%  "
            f"P95={item.decode_p95_improvement_vs_sequential_percent:>7.2f}%"
        )

    output = (
        args.output.resolve()
        if args.output is not None
        else root / "pairing_comparison_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "comparison_version": 2,
        "scope": (
            "Pairing baseline comparison; Trace-aware Mapping fixed; "
            "Profile and held-out Evaluation split"
        ),
        "protocol": {
            "profile_file_count": metrics[0].profile_file_count,
            "evaluation_file_count": metrics[0].evaluation_file_count,
            "profile_fingerprint": metrics[0].profile_fingerprint,
            "evaluation_fingerprint": metrics[0].evaluation_fingerprint,
        },
        "modes": list(args.modes),
        "metrics": [asdict(x) for x in metrics],
        "improvements_vs_sequential": [asdict(x) for x in improvements],
    }
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved：{output}")


if __name__ == "__main__":
    main()
