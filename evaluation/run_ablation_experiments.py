"""
One-command formal 2x2 Pairing/Mapping ablation runner.

Formal 2x2:
1. Naive
   Sequential Pairing + Round-Robin Mapping
2. Pairing Only
   Trace-aware Pairing + Local Search + Round-Robin Mapping
3. Mapping Only
   Sequential Pairing + Trace-aware Mapping
4. Full
   Trace-aware Pairing + Local Search + Trace-aware Mapping

Important optimization:
- Pairing Only is REUSED from results/experiments/mapping_baselines/round_robin
- Mapping Only is REUSED from results/experiments/pairing_baselines/sequential
- Full is REUSED from results/experiments/mapping_baselines/trace_aware
- Only Naive is newly generated/evaluated.

All four groups must use the same:
- Profile subset / fingerprint
- Held-out Evaluation subset / fingerprint
- spatial layout / hardware
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from mapping.trace_profile import DEFAULT_TRACE_ROOT
from mapping.trace_split import (
    DEFAULT_PROFILE_CACHE,
    DEFAULT_SPLIT_MANIFEST,
    EVALUATION_SUBSET,
    PROFILE_SUBSET,
    ensure_trace_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "ablation_formal"
)

PAIRING_BASELINE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "pairing_baselines"
)

MAPPING_BASELINE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "mapping_baselines"
)


class FormalAblationRunnerError(RuntimeError):
    pass


def _run(command: list[str]) -> None:
    print("\n" + "=" * 100)
    print(" ".join(command))
    print("=" * 100)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise FormalAblationRunnerError(
            f"命令执行失败，returncode={completed.returncode}: "
            + " ".join(command)
        )


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise FormalAblationRunnerError(
            f"缺少 {label}：{path}\n"
            "请先完成正式 Pairing baseline 和 Mapping baseline。"
        )


def _check_reused_results() -> None:
    required = (
        (
            MAPPING_BASELINE_ROOT / "round_robin" / "mapping.json",
            "Pairing Only mapping",
        ),
        (
            MAPPING_BASELINE_ROOT / "round_robin" / "phase_evaluation_summary.json",
            "Pairing Only phase summary",
        ),
        (
            PAIRING_BASELINE_ROOT / "sequential" / "mapping.json",
            "Mapping Only mapping",
        ),
        (
            PAIRING_BASELINE_ROOT / "sequential" / "phase_evaluation_summary.json",
            "Mapping Only phase summary",
        ),
        (
            MAPPING_BASELINE_ROOT / "trace_aware" / "mapping.json",
            "Full mapping",
        ),
        (
            MAPPING_BASELINE_ROOT / "trace_aware" / "phase_evaluation_summary.json",
            "Full phase summary",
        ),
    )
    for path, label in required:
        _require(path, label)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "正式 2x2 Pairing/Mapping 消融：只新跑 Naive，"
            "其余三组复用已完成的正式 baseline。"
        )
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--profile-workers", type=int, default=0)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
    )
    parser.add_argument(
        "--profile-cache",
        type=Path,
        default=DEFAULT_PROFILE_CACHE,
    )
    parser.add_argument("--profile-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--refresh-split", action="store_true")
    parser.add_argument(
        "--rerun-naive",
        action="store_true",
        help="即使 Naive 已存在也重新生成并评估。",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    args = parser.parse_args()

    if args.workers < 0:
        parser.error("--workers 不能小于 0")
    if args.profile_workers < 0:
        parser.error("--profile-workers 不能小于 0")
    if not 0.0 < args.profile_ratio < 1.0:
        parser.error("--profile-ratio 必须位于 (0,1)")

    start = time.perf_counter()

    trace_root = args.trace_root.resolve()
    manifest, manifest_path, _rebuilt = ensure_trace_split(
        trace_root=trace_root,
        manifest_path=args.split_manifest,
        profile_ratio=args.profile_ratio,
        seed=args.split_seed,
        force=args.refresh_split,
        verbose=True,
    )

    print("\n========== Formal 2x2 Ablation Protocol ==========")
    print(f"Trace Root：{trace_root}")
    print(f"Manifest：{manifest_path}")
    print(f"Profile Files：{manifest['profile']['file_count']}")
    print(f"Held-out Evaluation Files：{manifest['evaluation']['file_count']}")
    print(f"Split Seed：{manifest['seed']}")

    _check_reused_results()

    root = args.root.resolve()
    naive_dir = root / "naive"
    naive_dir.mkdir(parents=True, exist_ok=True)

    naive_mapping = naive_dir / "mapping.json"
    naive_prefill = naive_dir / "prefill_evaluation.json"
    naive_decode = naive_dir / "decode_fast_evaluation.json"
    naive_summary = naive_dir / "phase_evaluation_summary.json"

    python = sys.executable

    if args.rerun_naive or not (
        naive_mapping.exists()
        and naive_prefill.exists()
        and naive_decode.exists()
        and naive_summary.exists()
    ):
        print("\n[Naive] Sequential Pairing + Round-Robin Mapping")

        mapping_cmd = [
            python,
            "run_mapping_baseline.py",
            "--pairing-mode",
            "sequential",
            "--mapping-mode",
            "round_robin",
            "--trace-root",
            str(trace_root),
            "--trace-manifest",
            str(manifest_path),
            "--trace-subset",
            PROFILE_SUBSET,
            "--profile-cache",
            str(args.profile_cache),
            "--profile-workers",
            str(args.profile_workers),
            "--output",
            str(naive_mapping),
            "--quiet",
        ]
        _run(mapping_cmd)

        # FAST evaluator has already been exact-validated in the preceding
        # Pairing/Mapping formal baseline runs. Do not repeat expensive checks.
        phase_cmd = [
            python,
            "-m",
            "scheduling.run_phase_evaluation",
            "--mapping",
            str(naive_mapping),
            "--trace-root",
            str(trace_root),
            "--trace-manifest",
            str(manifest_path),
            "--trace-subset",
            EVALUATION_SUBSET,
            "--prefill-output",
            str(naive_prefill),
            "--decode-output",
            str(naive_decode),
            "--summary-output",
            str(naive_summary),
            "--workers",
            str(args.workers),
            "--prefill-exact-check",
            "0",
            "--exact-check",
            "0",
        ]
        _run(phase_cmd)
    else:
        print("\n[Reuse] Naive formal result already exists.")
        print(f"  {naive_summary}")

    compare_cmd = [
        python,
        "-m",
        "evaluation.ablation_summary",
        "--naive-root",
        str(naive_dir),
        "--pairing-only-root",
        str(MAPPING_BASELINE_ROOT / "round_robin"),
        "--mapping-only-root",
        str(PAIRING_BASELINE_ROOT / "sequential"),
        "--full-root",
        str(MAPPING_BASELINE_ROOT / "trace_aware"),
        "--output",
        str(root / "ablation_summary.json"),
    ]
    _run(compare_cmd)

    elapsed = time.perf_counter() - start
    print("\nFormal 2x2 ablation completed.")
    print(f"Results：{root}")
    print(f"Naive：{naive_dir}")
    print(
        "Pairing Only reused："
        f"{MAPPING_BASELINE_ROOT / 'round_robin'}"
    )
    print(
        "Mapping Only reused："
        f"{PAIRING_BASELINE_ROOT / 'sequential'}"
    )
    print(
        "Full reused："
        f"{MAPPING_BASELINE_ROOT / 'trace_aware'}"
    )
    print(f"Total Elapsed：{elapsed:.2f}s")


if __name__ == "__main__":
    main()
