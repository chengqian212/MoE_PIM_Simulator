"""
One-command formal Pairing baseline experiment runner.

Formal protocol:
1. Chinese-SimpleQA JSON files are stratified by category into
   Profile 80% / Held-out Evaluation 20% (seed=42 by default).
2. frequency/coactivation are built ONLY from Profile files.
3. TraceProfile is cached on disk and reused by every Pairing mode.
4. Mapping is fixed to trace_aware for all Pairing comparisons.
5. Prefill/Decode metrics are computed ONLY on the held-out Evaluation files.
6. The first mode performs FAST==EXACT checks; later modes reuse that validated
   evaluator implementation and skip repeated expensive exact checks by default.
7. Existing outputs are overwritten.

Typical usage:
    python -m evaluation.run_pairing_baselines --smoke --workers 1
    python -m evaluation.run_pairing_baselines --workers 0
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from mapping.trace_profile import DEFAULT_TRACE_ROOT
from mapping.trace_profile_cache import load_or_build_trace_profile
from mapping.trace_split import (
    DEFAULT_PROFILE_CACHE,
    DEFAULT_SPLIT_MANIFEST,
    EVALUATION_SUBSET,
    PROFILE_SUBSET,
    ensure_trace_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAIRING_MODES = (
    "sequential",
    "random",
    "frequency_aware",
    "greedy",
    "trace_aware",
    "optimal",
)

CANONICAL_MAPPING = (
    PROJECT_ROOT
    / "results"
    / "mappings"
    / "mapping_baseline_N4_H7168_W4096.json"
)
CANONICAL_PREFILL = PROJECT_ROOT / "results" / "prefill" / "prefill_evaluation.json"
CANONICAL_DECODE = PROJECT_ROOT / "results" / "decode" / "decode_fast_evaluation.json"
CANONICAL_SUMMARY = PROJECT_ROOT / "results" / "phase_evaluation_summary.json"


class PairingBaselineRunnerError(RuntimeError):
    pass


def run(command: list[str]) -> None:
    print("\n" + "=" * 96)
    print(" ".join(command))
    print("=" * 96)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise PairingBaselineRunnerError(
            f"命令执行失败，returncode={completed.returncode}: "
            + " ".join(command)
        )


def _remove_old_outputs(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _publish_trace_aware(root: Path) -> None:
    source = root / "trace_aware"
    pairs = (
        (source / "mapping.json", CANONICAL_MAPPING),
        (source / "prefill_evaluation.json", CANONICAL_PREFILL),
        (source / "decode_fast_evaluation.json", CANONICAL_DECODE),
        (source / "phase_evaluation_summary.json", CANONICAL_SUMMARY),
    )
    for src, dst in pairs:
        if not src.exists():
            raise PairingBaselineRunnerError(f"无法发布默认结果，缺少：{src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[Publish] {src} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "一次运行全部 Pairing baseline：Profile/Held-out 分离，"
            "固定 Trace-aware Mapping，只改变 Pairing。"
        )
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=PAIRING_MODES,
        default=list(PAIRING_MODES),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Prefill/Decode worker；0=自动，最多8。",
    )
    parser.add_argument(
        "--profile-workers",
        type=int,
        default=0,
        help="首次 TraceProfile 构建 worker；0=自动，最多4。",
    )
    parser.add_argument("--random-seed", type=int, default=42)
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
    parser.add_argument("--refresh-profile-cache", action="store_true")
    parser.add_argument(
        "--validate-each-mode",
        action="store_true",
        help="每种 Pairing 都重复 FAST==EXACT；默认仅第一种执行。",
    )
    args = parser.parse_args()

    if args.workers < 0:
        parser.error("--workers 不能小于 0")
    if args.profile_workers < 0:
        parser.error("--profile-workers 不能小于 0")
    if not 0.0 < args.profile_ratio < 1.0:
        parser.error("--profile-ratio 必须位于 (0,1)")

    trace_root = args.trace_root.resolve()
    manifest, manifest_path, _rebuilt = ensure_trace_split(
        trace_root=trace_root,
        manifest_path=args.split_manifest,
        profile_ratio=args.profile_ratio,
        seed=args.split_seed,
        force=args.refresh_split,
        verbose=True,
    )

    print("\n========== Formal Trace Protocol ==========")
    print(f"Trace Root：{trace_root}")
    print(f"Manifest：{manifest_path}")
    print(f"Profile Files：{manifest['profile']['file_count']}")
    print(f"Held-out Evaluation Files：{manifest['evaluation']['file_count']}")
    print(f"Split Seed：{manifest['seed']}")

    # Build once before spawning six mapping runs.  Every mapping subprocess then
    # gets a cache HIT rather than recomputing frequency/coactivation.
    debug_profile_files = 10 if args.smoke else None
    _profile, profile_info = load_or_build_trace_profile(
        trace_root=trace_root,
        manifest_path=manifest_path,
        subset=PROFILE_SUBSET,
        cache_path=args.profile_cache,
        max_files=debug_profile_files,
        workers=args.profile_workers,
        refresh_cache=args.refresh_profile_cache,
        strict=True,
        verbose=True,
    )
    print(
        "[Profile Ready] "
        f"cache_hit={profile_info.cache_hit}, "
        f"files={profile_info.file_count}, "
        f"cache={profile_info.cache_path}"
    )

    root_name = "pairing_baselines_smoke" if args.smoke else "pairing_baselines"
    root = PROJECT_ROOT / "results" / "experiments" / root_name
    root.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    for mode_index, mode in enumerate(args.modes):
        mode_dir = root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        mapping_path = mode_dir / "mapping.json"
        prefill_path = mode_dir / "prefill_evaluation.json"
        decode_path = mode_dir / "decode_fast_evaluation.json"
        summary_path = mode_dir / "phase_evaluation_summary.json"
        _remove_old_outputs([mapping_path, prefill_path, decode_path, summary_path])

        mapping_cmd = [
            python,
            "run_mapping_baseline.py",
            "--pairing-mode",
            mode,
            "--mapping-mode",
            "trace_aware",
            "--pairing-random-seed",
            str(args.random_seed),
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
            str(mapping_path),
            "--quiet",
        ]
        if args.smoke:
            mapping_cmd.extend(["--max-files", "10"])
        run(mapping_cmd)

        phase_cmd = [
            python,
            "-m",
            "scheduling.run_phase_evaluation",
            "--mapping",
            str(mapping_path),
            "--trace-root",
            str(trace_root),
            "--trace-manifest",
            str(manifest_path),
            "--trace-subset",
            EVALUATION_SUBSET,
            "--prefill-output",
            str(prefill_path),
            "--decode-output",
            str(decode_path),
            "--summary-output",
            str(summary_path),
            "--workers",
            str(args.workers),
        ]
        if args.smoke:
            phase_cmd.append("--smoke")

        # The evaluator implementation is already regression-tested against exact.
        # Keep one end-to-end validation in this run, then avoid repeating the
        # expensive exact path for five more static mappings.
        if mode_index > 0 and not args.validate_each_mode:
            phase_cmd.extend(
                [
                    "--prefill-exact-check",
                    "0",
                    "--exact-check",
                    "0",
                ]
            )
        run(phase_cmd)

    compare_cmd = [
        python,
        "-m",
        "evaluation.pairing_comparison",
        "--root",
        str(root),
        "--modes",
        *args.modes,
    ]
    run(compare_cmd)

    # The complete Trace-aware mode becomes the project's default formal output.
    if not args.smoke and "trace_aware" in args.modes:
        _publish_trace_aware(root)

    print("\nFormal Pairing baseline experiment completed.")
    print(f"Results：{root}")
    print(f"Split：{manifest_path}")
    print(f"Profile Cache：{profile_info.cache_path}")


if __name__ == "__main__":
    main()
