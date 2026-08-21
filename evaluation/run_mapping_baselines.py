"""
One-command formal Mapping baseline experiment runner.

Formal protocol:
1. Chinese-SimpleQA is fixed to the existing stratified Profile 80% /
   Held-out Evaluation 20% split.
2. TraceProfile is built/cached only from Profile files.
3. Pairing is fixed to the project's practical full method:
       trace_aware = Coactivation Greedy + Local Search.
4. The shared LogicalWeightCube / Spatial Layout / TraceProfile / Pairing are
   built ONCE in this process, then reused by all Mapping modes.
5. Mapping modes compared:
       round_robin / random / least_loaded / frequency_aware / trace_aware.
6. Every Mapping mode shares the same gate/up separation, capacity limit and
   up-stage feasibility guard.
7. Prefill/Decode are evaluated only on held-out Evaluation files.
8. FAST==EXACT is performed for the first Mapping mode only by default.
9. Existing formal outputs are overwritten.

Typical usage:
    python -m evaluation.run_mapping_baselines --smoke --workers 1
    python -m evaluation.run_mapping_baselines --workers 0
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from config import ModelConfig
from mapping.logical_weight import build_logical_weight_cubes
from mapping.physical_binder import bind_logical_mapping_to_physical_slots
from mapping.plane_pairer import (
    DEFAULT_PAIRING_RANDOM_SEED,
    PAIRING_MODE_TRACE_AWARE,
    build_logical_planes,
)
from mapping.spatial_layout_loader import (
    DEFAULT_RESULTS_DIR,
    load_spatial_layout_bundle,
)
from mapping.subcube_mapper import (
    DEFAULT_MAPPING_RANDOM_SEED,
    MAPPING_MODE_FREQUENCY_AWARE,
    MAPPING_MODE_LEAST_LOADED,
    MAPPING_MODE_RANDOM,
    MAPPING_MODE_ROUND_ROBIN,
    MAPPING_MODE_TRACE_AWARE,
    map_logical_planes_to_subcubes,
)
from mapping.trace_profile import DEFAULT_TRACE_ROOT
from mapping.trace_profile_cache import load_or_build_trace_profile
from mapping.trace_split import (
    DEFAULT_PROFILE_CACHE,
    DEFAULT_SPLIT_MANIFEST,
    EVALUATION_SUBSET,
    PROFILE_SUBSET,
    ensure_trace_split,
)
from run_mapping_baseline import (
    DEFAULT_H,
    DEFAULT_N,
    DEFAULT_W,
    build_output_dict,
    save_mapping_result,
    validate_complete_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAPPING_MODES = (
    MAPPING_MODE_ROUND_ROBIN,
    MAPPING_MODE_RANDOM,
    MAPPING_MODE_LEAST_LOADED,
    MAPPING_MODE_FREQUENCY_AWARE,
    MAPPING_MODE_TRACE_AWARE,
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


class MappingBaselineRunnerError(RuntimeError):
    pass


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remain = divmod(seconds, 60)
    return f"{int(minutes)}m {remain:.1f}s"


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
        raise MappingBaselineRunnerError(
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
    source = root / MAPPING_MODE_TRACE_AWARE
    pairs = (
        (source / "mapping.json", CANONICAL_MAPPING),
        (source / "prefill_evaluation.json", CANONICAL_PREFILL),
        (source / "decode_fast_evaluation.json", CANONICAL_DECODE),
        (source / "phase_evaluation_summary.json", CANONICAL_SUMMARY),
    )
    for src, dst in pairs:
        if not src.exists():
            raise MappingBaselineRunnerError(f"无法发布默认结果，缺少：{src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[Publish] {src} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "一次运行全部 Mapping baseline：固定 Trace-aware Pairing，"
            "Profile/Held-out 分离，只改变 Mapping。"
        )
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MAPPING_MODES,
        default=list(MAPPING_MODES),
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
    parser.add_argument(
        "--mapping-random-seed",
        type=int,
        default=DEFAULT_MAPPING_RANDOM_SEED,
    )
    parser.add_argument(
        "--pairing-random-seed",
        type=int,
        default=DEFAULT_PAIRING_RANDOM_SEED,
    )
    parser.add_argument("--local-search-rounds", type=int, default=4)
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
        help="每种 Mapping 都重复 FAST==EXACT；默认仅第一种执行。",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--H", type=int, default=DEFAULT_H)
    parser.add_argument("--W", type=int, default=DEFAULT_W)
    parser.add_argument("--spatial-rank", type=int, default=None)
    parser.add_argument("--layout-id", type=str, default=None)
    args = parser.parse_args()

    if args.workers < 0:
        parser.error("--workers 不能小于 0")
    if args.profile_workers < 0:
        parser.error("--profile-workers 不能小于 0")
    if args.local_search_rounds < 0:
        parser.error("--local-search-rounds 不能小于 0")
    if not 0.0 < args.profile_ratio < 1.0:
        parser.error("--profile-ratio 必须位于 (0,1)")

    total_start = time.perf_counter()
    trace_root = args.trace_root.resolve()

    manifest, manifest_path, _rebuilt = ensure_trace_split(
        trace_root=trace_root,
        manifest_path=args.split_manifest,
        profile_ratio=args.profile_ratio,
        seed=args.split_seed,
        force=args.refresh_split,
        verbose=True,
    )

    print("\n========== Formal Mapping Protocol ==========")
    print(f"Trace Root：{trace_root}")
    print(f"Manifest：{manifest_path}")
    print(f"Profile Files：{manifest['profile']['file_count']}")
    print(f"Held-out Evaluation Files：{manifest['evaluation']['file_count']}")
    print(f"Split Seed：{manifest['seed']}")
    print("Pairing Fixed：trace_aware (Greedy + Local Search)")

    # --------------------------------------------------------
    # Shared preparation: only once.
    # --------------------------------------------------------
    print("\n[Shared Stage 1] Build LogicalWeightCube once")
    cubes = build_logical_weight_cubes(
        ModelConfig(include_shared_expert=True)
    )
    print(f"Logical Weight-Cubes：{len(cubes)}")

    print("\n[Shared Stage 2] Load Spatial Layout once")
    spatial = load_spatial_layout_bundle(
        results_dir=args.results_dir,
        N=args.N,
        H=args.H,
        W=args.W,
        spatial_rank=args.spatial_rank,
        layout_id=args.layout_id,
        expected_matrix_count=len(cubes),
        require_single_chunk=True,
    )
    print(
        f"Spatial：N={spatial.hardware.N}, "
        f"SC={spatial.hardware.num_subcubes}, D={spatial.hardware.D}, "
        f"P={spatial.plane_count}, Q={spatial.hardware.total_plane_slots}"
    )

    print("\n[Shared Stage 3] Load TraceProfile once")
    debug_profile_files = 10 if args.smoke else None
    effective_profile_cache = args.profile_cache
    if args.smoke:
        suffix = args.profile_cache.suffix or ".pkl"
        effective_profile_cache = args.profile_cache.with_name(
            args.profile_cache.stem + "_smoke10" + suffix
        )

    profile, profile_info = load_or_build_trace_profile(
        trace_root=trace_root,
        manifest_path=manifest_path,
        subset=PROFILE_SUBSET,
        cache_path=effective_profile_cache,
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

    print("\n[Shared Stage 4] Build Trace-aware Pairing once")
    pairing_start = time.perf_counter()
    pairing = build_logical_planes(
        cubes=cubes,
        profile=profile,
        pairing_mode=PAIRING_MODE_TRACE_AWARE,
        improve_pairs=True,
        local_search_rounds=args.local_search_rounds,
        random_seed=args.pairing_random_seed,
    )
    if len(pairing.planes) != spatial.plane_count:
        raise MappingBaselineRunnerError(
            "固定 Pairing 的 LogicalPlane 数与 Spatial Plane 数不一致："
            f"logical={len(pairing.planes)}, physical={spatial.plane_count}"
        )
    print(
        f"Pairing Planes：{len(pairing.planes)}, "
        f"Pairing Cost：{pairing.total_routed_up_coactivation_cost}, "
        f"elapsed={_format_seconds(time.perf_counter() - pairing_start)}"
    )

    root_name = "mapping_baselines_smoke" if args.smoke else "mapping_baselines"
    root = PROJECT_ROOT / "results" / "experiments" / root_name
    root.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    for mode_index, mode in enumerate(args.modes):
        print("\n" + "#" * 96)
        print(f"# Mapping Mode：{mode}")
        print("#" * 96)

        mode_dir = root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = mode_dir / "mapping.json"
        prefill_path = mode_dir / "prefill_evaluation.json"
        decode_path = mode_dir / "decode_fast_evaluation.json"
        summary_path = mode_dir / "phase_evaluation_summary.json"
        _remove_old_outputs([mapping_path, prefill_path, decode_path, summary_path])

        mapping_start = time.perf_counter()
        subcube_mapping = map_logical_planes_to_subcubes(
            pairing=pairing,
            cubes=cubes,
            profile=profile,
            hardware=spatial.hardware,
            mapping_mode=mode,
            random_seed=args.mapping_random_seed,
        )
        binding = bind_logical_mapping_to_physical_slots(
            cubes=cubes,
            pairing=pairing,
            subcube_mapping=subcube_mapping,
            physical_planes=spatial.physical_planes,
        )
        validate_complete_mapping(
            cubes=cubes,
            spatial=spatial,
            pairing=pairing,
            subcube_mapping=subcube_mapping,
            binding=binding,
        )

        output_data = build_output_dict(
            cubes=cubes,
            profile=profile,
            spatial=spatial,
            pairing=pairing,
            subcube_mapping=subcube_mapping,
            binding=binding,
            pairing_mode=PAIRING_MODE_TRACE_AWARE,
            mapping_mode=mode,
            local_search_enabled=True,
            local_search_rounds=args.local_search_rounds,
            pairing_random_seed=args.pairing_random_seed,
            mapping_random_seed=args.mapping_random_seed,
            trace_manifest=manifest_path,
            trace_subset=PROFILE_SUBSET,
            trace_profile_fingerprint=profile_info.file_fingerprint,
            trace_profile_cache_path=profile_info.cache_path,
        )
        save_mapping_result(
            output_path=mapping_path,
            data=output_data,
        )
        print(
            f"[Mapping Done] mode={mode}, "
            f"conflict={subcube_mapping.total_conflict_cost:,}, "
            f"plane_range={subcube_mapping.min_planes_in_subcube}~"
            f"{subcube_mapping.max_planes_in_subcube}, "
            f"elapsed={_format_seconds(time.perf_counter() - mapping_start)}"
        )
        print(f"Saved：{mapping_path}")

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
        "evaluation.mapping_comparison",
        "--root",
        str(root),
        "--modes",
        *args.modes,
    ]
    run(compare_cmd)

    if not args.smoke and MAPPING_MODE_TRACE_AWARE in args.modes:
        _publish_trace_aware(root)

    print("\nFormal Mapping baseline experiment completed.")
    print(f"Results：{root}")
    print(f"Split：{manifest_path}")
    print(f"Profile Cache：{profile_info.cache_path}")
    print(f"Total Elapsed：{_format_seconds(time.perf_counter() - total_start)}")


if __name__ == "__main__":
    main()
