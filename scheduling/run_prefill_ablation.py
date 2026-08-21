"""一键运行 Prefill Scheduler 四策略消融。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mapping.trace_profile import DEFAULT_TRACE_ROOT
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


def _reduction(baseline: float, current: float) -> float:
    return 0.0 if baseline == 0 else (baseline - current) / baseline * 100.0


def _load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Prefill Scheduler 消融：no_reuse / switch_aware / aggressive_reuse / largest_batch_reuse。")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--exact-check", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="每种策略只跑 10 个 Prefill Batch。")
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
        ]
        if max_batches is not None:
            cmd.extend(["--max-batches", str(max_batches)])

        print("\n" + "=" * 72)
        print(f"[Prefill Ablation] mode={mode}")
        print(" ".join(cmd))
        print("=" * 72)

        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Prefill 消融运行失败：mode={mode}, returncode={completed.returncode}")
        if not output.exists():
            raise RuntimeError(f"缺少输出：{output}")

        results[mode] = _load_metrics(output)
        results[mode]["output"] = str(output)

    baseline = results.get(PREFILL_MODE_NO_REUSE)
    if baseline is not None:
        for mode, metrics in results.items():
            metrics["improvement_vs_no_reuse"] = {
                "prefill_mean_percent": _reduction(
                    baseline["prefill_mean_cycles"], metrics["prefill_mean_cycles"]
                ),
                "cycles_per_input_token_percent": _reduction(
                    baseline["mean_cycles_per_input_token"], metrics["mean_cycles_per_input_token"]
                ),
                "p95_percent": _reduction(
                    baseline["prefill_p95_cycles"], metrics["prefill_p95_cycles"]
                ),
                "switches_percent": _reduction(
                    baseline["mean_switches_per_batch"], metrics["mean_switches_per_batch"]
                ),
            }

    summary = {
        "experiment": "prefill_scheduler_ablation",
        "modes": list(args.modes),
        "mapping": str(mapping),
        "trace_root": str(root),
        "max_batches": max_batches,
        "results": results,
    }
    summary_path = out_root / "prefill_scheduler_ablation_summary.json"
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
    print(f"\nSaved：{summary_path}")


if __name__ == "__main__":
    main()
