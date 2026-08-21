"""一键运行 Prefill Scheduler 四策略消融，并按 Prefill Token 数分桶统计。"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean

from mapping.trace_profile import DEFAULT_TRACE_ROOT
from mapping.trace_split import EVALUATION_SUBSET, TRACE_SUBSETS
from scheduling.prefill_scheduling_mode import (
    PREFILL_MODE_AGGRESSIVE_REUSE,
    PREFILL_MODE_LARGEST_BATCH_REUSE,
    PREFILL_MODE_NO_REUSE,
    PREFILL_MODE_SWITCH_AWARE,
    PREFILL_SCHEDULING_MODES,
)
from scheduling.runtime_index import DEFAULT_MAPPING_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "experiments" / "prefill_scheduler"

# 当前 Held-out Evaluation 的 Prefill 长度范围约为 6~39。
# 采用下面四档，避免 33+ 单独成桶后样本过少，同时仍能观察长度趋势。
TOKEN_BUCKETS = (
    ("2-8", 2, 8),
    ("9-16", 9, 16),
    ("17-24", 17, 24),
    ("25+", 25, None),
)


def _reduction(baseline: float, current: float) -> float:
    return 0.0 if baseline == 0 else (baseline - current) / baseline * 100.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return data[lo]
    frac = pos - lo
    return data[lo] * (1.0 - frac) + data[hi] * frac


def _load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_metrics(path: Path) -> dict:
    payload = _load_payload(path)
    s = payload["summary"]
    return {
        "prefill_mean_cycles": float(s["total_cycles"]["mean"]),
        "prefill_p95_cycles": float(s["total_cycles"]["p95"]),
        "mean_cycles_per_input_token": float(s["cycles_per_input_token"]["mean"]),
        "global_cycles_per_input_token": float(s["global_cycles_per_input_token"]),
        "mean_switches_per_batch": float(s["switches"]["mean"]),
        "mean_wait_cycles_per_batch": float(s["wait_cycles"]["mean"]),
        "batch_count": int(s["batch_count"]),
        "total_input_tokens": int(s["total_input_tokens"]),
    }


def _bucket_name(token_count: int) -> str | None:
    for name, lower, upper in TOKEN_BUCKETS:
        if token_count >= lower and (upper is None or token_count <= upper):
            return name
    return None


def _build_bucket_metrics(path: Path) -> dict[str, dict]:
    payload = _load_payload(path)
    records = payload.get("records", [])

    grouped: dict[str, list[dict]] = {name: [] for name, _, _ in TOKEN_BUCKETS}
    for record in records:
        token_count = int(record["input_tokens"])
        name = _bucket_name(token_count)
        if name is not None:
            grouped[name].append(record)

    result: dict[str, dict] = {}
    for name, lower, upper in TOKEN_BUCKETS:
        items = grouped[name]
        if not items:
            result[name] = {
                "range": {"min_tokens": lower, "max_tokens": upper},
                "batch_count": 0,
            }
            continue

        token_counts = [int(x["input_tokens"]) for x in items]
        total_cycles = [float(x["total_cycles"]) for x in items]
        cycles_per_token = [float(x["cycles_per_input_token"]) for x in items]
        switches = [float(x["switches"]) for x in items]
        waits = [float(x["wait_cycles"]) for x in items]

        sum_tokens = sum(token_counts)
        sum_cycles = sum(total_cycles)

        result[name] = {
            "range": {"min_tokens": lower, "max_tokens": upper},
            "batch_count": len(items),
            "total_input_tokens": sum_tokens,
            "mean_input_tokens": float(mean(token_counts)),
            "mean_prefill_cycles": float(mean(total_cycles)),
            "p95_prefill_cycles": float(_percentile(total_cycles, 0.95)),
            "mean_cycles_per_input_token": float(mean(cycles_per_token)),
            "global_cycles_per_input_token": (
                float(sum_cycles / sum_tokens) if sum_tokens > 0 else 0.0
            ),
            "mean_switches_per_batch": float(mean(switches)),
            "mean_wait_cycles_per_batch": float(mean(waits)),
        }

    return result


def _add_bucket_improvements(results: dict[str, dict]) -> None:
    baseline = results.get(PREFILL_MODE_NO_REUSE)
    if baseline is None:
        return

    baseline_buckets = baseline.get("buckets", {})
    for mode, metrics in results.items():
        for bucket_name, bucket in metrics.get("buckets", {}).items():
            base = baseline_buckets.get(bucket_name)
            if not base or not bucket.get("batch_count") or not base.get("batch_count"):
                continue

            bucket["improvement_vs_no_reuse"] = {
                "prefill_mean_percent": _reduction(
                    float(base["mean_prefill_cycles"]),
                    float(bucket["mean_prefill_cycles"]),
                ),
                "cycles_per_input_token_percent": _reduction(
                    float(base["mean_cycles_per_input_token"]),
                    float(bucket["mean_cycles_per_input_token"]),
                ),
                "switches_percent": _reduction(
                    float(base["mean_switches_per_batch"]),
                    float(bucket["mean_switches_per_batch"]),
                ),
            }


def _print_bucket_table(results: dict[str, dict], modes: list[str]) -> None:
    print("\n========== Prefill Batch Size Sensitivity ==========")
    for bucket_name, _, _ in TOKEN_BUCKETS:
        print(f"\n[{bucket_name} tokens]")
        print(
            f"{'Mode':<22} {'N':>5} {'Mean':>10} {'C/Token':>10} "
            f"{'P95':>10} {'Switches':>11} {'Mean↓':>9}"
        )
        print("-" * 85)

        for mode in modes:
            b = results[mode]["buckets"].get(bucket_name, {})
            n = int(b.get("batch_count", 0))
            if n == 0:
                print(f"{mode:<22} {0:>5} {'NA':>10} {'NA':>10} {'NA':>10} {'NA':>11} {'NA':>9}")
                continue

            imp = b.get("improvement_vs_no_reuse", {})
            mean_reduction = imp.get("prefill_mean_percent", 0.0)
            print(
                f"{mode:<22} "
                f"{n:>5d} "
                f"{b['mean_prefill_cycles']:>10.2f} "
                f"{b['mean_cycles_per_input_token']:>10.4f} "
                f"{b['p95_prefill_cycles']:>10.2f} "
                f"{b['mean_switches_per_batch']:>11.2f} "
                f"{mean_reduction:>8.2f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "运行 Prefill Scheduler 消融，并按 2-8 / 9-16 / 17-24 / 25+ "
            "Prefill Token 分桶统计。"
        )
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--exact-check", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="每种策略只跑 10 个 Prefill Batch。")
    parser.add_argument(
        "--trace-manifest",
        type=Path,
        default=None,
        help="Profile/Evaluation split manifest；正式实验必须传入。",
    )
    parser.add_argument(
        "--trace-subset",
        choices=TRACE_SUBSETS,
        default=EVALUATION_SUBSET,
        help="正式 Scheduler 评估应使用 evaluation。",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=PREFILL_SCHEDULING_MODES,
        default=[
            PREFILL_MODE_NO_REUSE,
            PREFILL_MODE_SWITCH_AWARE,
            PREFILL_MODE_AGGRESSIVE_REUSE,
            PREFILL_MODE_LARGEST_BATCH_REUSE,
        ],
    )
    args = parser.parse_args()

    mapping = args.mapping.resolve()
    root = args.root.resolve()
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    trace_manifest = args.trace_manifest.resolve() if args.trace_manifest is not None else None
    if trace_manifest is None and not args.smoke:
        parser.error("正式 Prefill 消融必须传 --trace-manifest，避免误跑全量 Trace。")

    max_batches = args.max_batches
    if max_batches is None and args.smoke:
        max_batches = 10

    results: dict[str, dict] = {}

    for mode in args.modes:
        mode_dir = out_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        output = mode_dir / "prefill_evaluation.json"

        cmd = [
            sys.executable,
            "-m",
            "scheduling.prefill_fast_evaluator",
            "--mapping",
            str(mapping),
            "--root",
            str(root),
            "--output",
            str(output),
            "--scheduling-mode",
            mode,
            "--exact-check",
            str(args.exact_check),
            "--workers",
            str(args.workers),
            "--trace-subset",
            args.trace_subset,
        ]
        if trace_manifest is not None:
            cmd.extend(["--trace-manifest", str(trace_manifest)])
        if max_batches is not None:
            cmd.extend(["--max-batches", str(max_batches)])

        print("\n" + "=" * 72)
        print(f"[Prefill Ablation] mode={mode}")
        print(" ".join(cmd))
        print("=" * 72)

        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Prefill 消融运行失败：mode={mode}, returncode={completed.returncode}"
            )
        if not output.exists():
            raise RuntimeError(f"缺少输出：{output}")

        results[mode] = _load_metrics(output)
        results[mode]["buckets"] = _build_bucket_metrics(output)
        results[mode]["output"] = str(output)

    baseline = results.get(PREFILL_MODE_NO_REUSE)
    if baseline is not None:
        for mode, metrics in results.items():
            metrics["improvement_vs_no_reuse"] = {
                "prefill_mean_percent": _reduction(
                    baseline["prefill_mean_cycles"], metrics["prefill_mean_cycles"]
                ),
                "cycles_per_input_token_percent": _reduction(
                    baseline["mean_cycles_per_input_token"],
                    metrics["mean_cycles_per_input_token"],
                ),
                "p95_percent": _reduction(
                    baseline["prefill_p95_cycles"], metrics["prefill_p95_cycles"]
                ),
                "switches_percent": _reduction(
                    baseline["mean_switches_per_batch"],
                    metrics["mean_switches_per_batch"],
                ),
            }

    _add_bucket_improvements(results)

    summary = {
        "experiment": "prefill_scheduler_ablation_with_batch_size_sensitivity",
        "protocol": "heldout_evaluation",
        "modes": list(args.modes),
        "mapping": str(mapping),
        "trace_root": str(root),
        "trace_manifest": str(trace_manifest) if trace_manifest is not None else None,
        "trace_subset": args.trace_subset,
        "max_batches": max_batches,
        "token_buckets": [
            {"name": name, "min_tokens": lower, "max_tokens": upper}
            for name, lower, upper in TOKEN_BUCKETS
        ],
        "results": results,
    }
    if set(args.modes) == set(PREFILL_SCHEDULING_MODES):
        summary_path = out_root / "prefill_scheduler_ablation_summary.json"
    else:
        mode_suffix = "__".join(args.modes)
        summary_path = out_root / f"prefill_scheduler_ablation_summary__{mode_suffix}.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== Prefill Scheduler Ablation ==========")
    for mode in args.modes:
        m = results[mode]
        print(
            f"{mode:<20} "
            f"mean={m['prefill_mean_cycles']:.2f}, "
            f"cycles/token={m['mean_cycles_per_input_token']:.4f}, "
            f"p95={m['prefill_p95_cycles']:.2f}, "
            f"switches={m['mean_switches_per_batch']:.2f}, "
            f"wait={m['mean_wait_cycles_per_batch']:.2f}"
        )

    _print_bucket_table(results, list(args.modes))
    print(f"\nSaved：{summary_path}")


if __name__ == "__main__":
    main()