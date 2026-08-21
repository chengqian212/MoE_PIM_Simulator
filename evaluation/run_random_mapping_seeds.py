"""
Random Mapping 多 seed 正式补充实验。

用途：
- 只重复 Random Mapping；
- 默认 seed = 42,43,44,45,46；
- 固定 Trace-aware Pairing + Local Search；
- 固定同一 Profile 80% / Held-out Evaluation 20%；
- LogicalWeightCube / Spatial Layout / TraceProfile / Pairing 只构造一次；
- 每个 seed 只重新做 Random Mapping + Prefill/Decode；
- 不重复 FAST==EXACT（前面的正式 baseline 已经验证）。

典型命令：
    python -m evaluation.run_random_mapping_seeds --workers 0 --profile-workers 0
"""

from __future__ import annotations

import argparse
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
    MAPPING_MODE_RANDOM,
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
DEFAULT_RANDOM_SEEDS = (42, 43, 44, 45, 46)


class RandomSeedRunnerError(RuntimeError):
    pass


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remain = divmod(seconds, 60)
    return f"{int(minutes)}m {remain:.1f}s"


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
        raise RandomSeedRunnerError(
            f"命令执行失败，returncode={completed.returncode}: "
            + " ".join(command)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random Mapping 多 seed 正式补充实验。"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_RANDOM_SEEDS),
        help="Random Mapping seeds，默认 42 43 44 45 46。",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--profile-workers", type=int, default=0)
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
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--H", type=int, default=DEFAULT_H)
    parser.add_argument("--W", type=int, default=DEFAULT_W)
    parser.add_argument("--spatial-rank", type=int, default=None)
    parser.add_argument("--layout-id", type=str, default=None)
    parser.add_argument(
        "--root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "experiments"
            / "mapping_baselines"
        ),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="若某个 seed 的 4 个输出都存在则直接复用。",
    )
    args = parser.parse_args()

    if not args.seeds:
        parser.error("--seeds 不能为空")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds 不能包含重复值")
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
    root = args.root.resolve()
    random_root = root / "random_seeds"
    random_root.mkdir(parents=True, exist_ok=True)

    manifest, manifest_path, _rebuilt = ensure_trace_split(
        trace_root=trace_root,
        manifest_path=args.split_manifest,
        profile_ratio=args.profile_ratio,
        seed=args.split_seed,
        force=args.refresh_split,
        verbose=True,
    )

    print("\n========== Random Mapping Multi-Seed Protocol ==========")
    print(f"Seeds：{args.seeds}")
    print(f"Profile Files：{manifest['profile']['file_count']}")
    print(f"Held-out Evaluation Files：{manifest['evaluation']['file_count']}")
    print("Pairing Fixed：trace_aware (Greedy + Local Search)")
    print("Mapping：random only")
    print("FAST==EXACT：reuse previous validation")

    # --------------------------------------------------------
    # Shared stages: only once.
    # --------------------------------------------------------
    print("\n[Shared 1] Build LogicalWeightCube")
    cubes = build_logical_weight_cubes(
        ModelConfig(include_shared_expert=True)
    )

    print("\n[Shared 2] Load Spatial Layout")
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

    print("\n[Shared 3] Load Profile Cache")
    profile, profile_info = load_or_build_trace_profile(
        trace_root=trace_root,
        manifest_path=manifest_path,
        subset=PROFILE_SUBSET,
        cache_path=args.profile_cache,
        max_files=None,
        workers=args.profile_workers,
        refresh_cache=args.refresh_profile_cache,
        strict=True,
        verbose=True,
    )
    print(
        f"[Profile Ready] cache_hit={profile_info.cache_hit}, "
        f"files={profile_info.file_count}, cache={profile_info.cache_path}"
    )

    print("\n[Shared 4] Build Trace-aware Pairing once")
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
        raise RandomSeedRunnerError(
            "固定 Pairing Plane 数与 Spatial Plane 数不一致。"
        )
    print(
        f"Pairing Cost：{pairing.total_routed_up_coactivation_cost:,}, "
        f"elapsed={_format_seconds(time.perf_counter() - pairing_start)}"
    )

    python = sys.executable

    for seed in args.seeds:
        seed_dir = random_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        mapping_path = seed_dir / "mapping.json"
        prefill_path = seed_dir / "prefill_evaluation.json"
        decode_path = seed_dir / "decode_fast_evaluation.json"
        summary_path = seed_dir / "phase_evaluation_summary.json"

        complete = all(
            path.exists()
            for path in (
                mapping_path,
                prefill_path,
                decode_path,
                summary_path,
            )
        )
        if complete and args.reuse_existing:
            print(f"\n[Reuse] Random seed={seed} 已存在，跳过。")
            continue

        print("\n" + "#" * 100)
        print(f"# Random Mapping Seed = {seed}")
        print("#" * 100)

        mapping_start = time.perf_counter()
        subcube_mapping = map_logical_planes_to_subcubes(
            pairing=pairing,
            cubes=cubes,
            profile=profile,
            hardware=spatial.hardware,
            mapping_mode=MAPPING_MODE_RANDOM,
            random_seed=seed,
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
            mapping_mode=MAPPING_MODE_RANDOM,
            local_search_enabled=True,
            local_search_rounds=args.local_search_rounds,
            pairing_random_seed=args.pairing_random_seed,
            mapping_random_seed=seed,
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
            f"[Mapping Done] seed={seed}, "
            f"conflict={subcube_mapping.total_conflict_cost:,}, "
            f"plane_range={subcube_mapping.min_planes_in_subcube}~"
            f"{subcube_mapping.max_planes_in_subcube}, "
            f"elapsed={_format_seconds(time.perf_counter() - mapping_start)}"
        )

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
            "--prefill-exact-check",
            "0",
            "--exact-check",
            "0",
        ]
        _run(phase_cmd)

    print("\n[Summary] Rebuild Mapping comparison with Random Mean ± Std")
    compare_cmd = [
        python,
        "-m",
        "evaluation.mapping_comparison",
        "--root",
        str(root),
        "--random-seeds",
        *[str(seed) for seed in args.seeds],
    ]
    _run(compare_cmd)

    print("\nRandom Mapping multi-seed experiment completed.")
    print(f"Random Seed Results：{random_root}")
    print(f"Updated Comparison：{root / 'mapping_comparison_summary.json'}")
    print(f"Total Elapsed：{_format_seconds(time.perf_counter() - total_start)}")


if __name__ == "__main__":
    main()
