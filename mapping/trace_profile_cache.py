"""Cached and optionally parallel Chinese-SimpleQA TraceProfile builder."""

from __future__ import annotations

import os
import pickle
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mapping.trace_profile import (
    NUM_MOE_LAYERS,
    NUM_ROUTED_EXPERTS,
    TraceProfile,
    TraceProfileError,
    process_trace_file,
    validate_profile,
)
from mapping.trace_split import (
    DEFAULT_PROFILE_CACHE,
    PROFILE_SUBSET,
    fingerprint_trace_files,
    resolve_trace_files,
)


PROFILE_CACHE_VERSION = 1


class TraceProfileCacheError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TraceProfileLoadInfo:
    cache_hit: bool
    cache_path: Path
    file_count: int
    file_fingerprint: str
    workers: int


def _resolve_worker_count(workers: int) -> int:
    if workers < 0:
        raise TraceProfileCacheError("profile workers 不能小于 0。")
    if workers == 0:
        cpu = os.cpu_count() or 1
        if cpu <= 2:
            return 1
        # Each worker owns ~30 MiB coactivation arrays, so cap at 4.
        return max(1, min(4, cpu - 1))
    return workers


def _empty_accumulators():
    frequency = [
        [0 for _ in range(NUM_ROUTED_EXPERTS)]
        for _ in range(NUM_MOE_LAYERS)
    ]
    pair_matrix_size = NUM_ROUTED_EXPERTS * NUM_ROUTED_EXPERTS
    coactivation = [
        array("Q", [0]) * pair_matrix_size
        for _ in range(NUM_MOE_LAYERS)
    ]
    token_count_by_layer = [0 for _ in range(NUM_MOE_LAYERS)]
    return frequency, coactivation, token_count_by_layer


def _freeze_profile(
    *,
    file_count: int,
    trace_segment_count: int,
    skipped_segment_count: int,
    category_file_counts: dict[str, int],
    frequency: list[list[int]],
    coactivation: list[array],
    token_count_by_layer: list[int],
    strict: bool,
) -> TraceProfile:
    profile = TraceProfile(
        file_count=file_count,
        trace_segment_count=trace_segment_count,
        skipped_segment_count=skipped_segment_count,
        category_file_counts=dict(sorted(category_file_counts.items())),
        frequency=tuple(tuple(layer) for layer in frequency),
        coactivation=tuple(coactivation),
        token_count_by_layer=tuple(token_count_by_layer),
    )
    validate_profile(profile, strict=strict)
    return profile


def _category(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) >= 2 else "__root__"


def _build_profile_sequential(
    *,
    root: Path,
    files: tuple[Path, ...],
    strict: bool,
    verbose: bool,
) -> TraceProfile:
    frequency, coactivation, token_count = _empty_accumulators()
    category_counts: Counter[str] = Counter()
    trace_segments = 0
    skipped_segments = 0
    total_files = len(files)

    for file_index, path in enumerate(files, start=1):
        category_counts[_category(root, path)] += 1
        segment_count, skipped_count = process_trace_file(
            path=path,
            frequency=frequency,
            coactivation=coactivation,
            token_count_by_layer=token_count,
        )
        trace_segments += segment_count
        skipped_segments += skipped_count
        if verbose and (
            file_index == 1
            or file_index == total_files
            or file_index % 100 == 0
        ):
            print(
                f"[TraceProfile] {file_index}/{total_files} "
                f"{path.relative_to(root)}"
            )

    return _freeze_profile(
        file_count=total_files,
        trace_segment_count=trace_segments,
        skipped_segment_count=skipped_segments,
        category_file_counts=dict(category_counts),
        frequency=frequency,
        coactivation=coactivation,
        token_count_by_layer=token_count,
        strict=strict,
    )


def _profile_worker(
    root_text: str,
    relatives: tuple[str, ...],
) -> tuple:
    root = Path(root_text)
    files = tuple(root / relative for relative in relatives)
    frequency, coactivation, token_count = _empty_accumulators()
    category_counts: Counter[str] = Counter()
    trace_segments = 0
    skipped_segments = 0

    for path in files:
        category_counts[_category(root, path)] += 1
        segment_count, skipped_count = process_trace_file(
            path=path,
            frequency=frequency,
            coactivation=coactivation,
            token_count_by_layer=token_count,
        )
        trace_segments += segment_count
        skipped_segments += skipped_count

    return (
        len(files),
        trace_segments,
        skipped_segments,
        dict(category_counts),
        tuple(tuple(layer) for layer in frequency),
        tuple(coactivation),
        tuple(token_count),
    )


def _make_shards(root: Path, files: tuple[Path, ...], workers: int) -> list[tuple[str, ...]]:
    shard_count = min(len(files), workers)
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    for index, path in enumerate(files):
        shards[index % shard_count].append(str(path.relative_to(root)))
    return [tuple(shard) for shard in shards if shard]


def _merge_partial_profiles(
    *,
    partials: Iterable[tuple],
    strict: bool,
) -> TraceProfile:
    frequency, coactivation, token_count = _empty_accumulators()
    category_counts: Counter[str] = Counter()
    file_count = 0
    trace_segments = 0
    skipped_segments = 0

    for part in partials:
        (
            part_files,
            part_segments,
            part_skipped,
            part_categories,
            part_frequency,
            part_coactivation,
            part_token_count,
        ) = part
        file_count += part_files
        trace_segments += part_segments
        skipped_segments += part_skipped
        category_counts.update(part_categories)

        for layer_id in range(NUM_MOE_LAYERS):
            dst_freq = frequency[layer_id]
            src_freq = part_frequency[layer_id]
            for expert_id, value in enumerate(src_freq):
                dst_freq[expert_id] += value

            dst_pair = coactivation[layer_id]
            src_pair = part_coactivation[layer_id]
            for index, value in enumerate(src_pair):
                if value:
                    dst_pair[index] += value

            token_count[layer_id] += part_token_count[layer_id]

    return _freeze_profile(
        file_count=file_count,
        trace_segment_count=trace_segments,
        skipped_segment_count=skipped_segments,
        category_file_counts=dict(category_counts),
        frequency=frequency,
        coactivation=coactivation,
        token_count_by_layer=token_count,
        strict=strict,
    )


def _build_profile_parallel(
    *,
    root: Path,
    files: tuple[Path, ...],
    workers: int,
    strict: bool,
    verbose: bool,
) -> TraceProfile:
    shards = _make_shards(root, files, workers)
    if verbose:
        print(
            "[TraceProfileParallel] "
            f"workers={workers}, shards={len(shards)}, files={len(files)}"
        )

    partials: list[tuple] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_profile_worker, str(root), shard)
            for shard in shards
        ]
        for index, future in enumerate(futures, start=1):
            partials.append(future.result())
            if verbose:
                print(f"[TraceProfileParallel] shard {index}/{len(futures)} done")

    if verbose:
        print("[TraceProfileParallel] merging worker statistics...")
    return _merge_partial_profiles(partials=partials, strict=strict)


def _load_cache(
    *,
    cache_path: Path,
    expected_fingerprint: str,
    expected_file_count: int,
) -> TraceProfile | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as file:
            payload = pickle.load(file)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("cache_version") != PROFILE_CACHE_VERSION:
        return None
    if payload.get("file_fingerprint") != expected_fingerprint:
        return None
    if payload.get("file_count") != expected_file_count:
        return None
    profile = payload.get("profile")
    if not isinstance(profile, TraceProfile):
        return None
    return profile


def _save_cache(
    *,
    cache_path: Path,
    profile: TraceProfile,
    file_fingerprint: str,
    file_count: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": PROFILE_CACHE_VERSION,
        "file_fingerprint": file_fingerprint,
        "file_count": file_count,
        "profile": profile,
    }
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(cache_path)


def load_or_build_trace_profile(
    *,
    trace_root: Path | str,
    manifest_path: Path | str | None = None,
    subset: str = PROFILE_SUBSET,
    cache_path: Path | str = DEFAULT_PROFILE_CACHE,
    max_files: int | None = None,
    workers: int = 0,
    refresh_cache: bool = False,
    strict: bool = True,
    verbose: bool = True,
) -> tuple[TraceProfile, TraceProfileLoadInfo]:
    root = Path(trace_root).resolve()
    files = list(
        resolve_trace_files(
            trace_root=root,
            manifest_path=manifest_path,
            subset=subset,
        )
    )

    if max_files is not None:
        if max_files <= 0:
            raise TraceProfileCacheError("max_files 必须大于 0。")
        files = files[:max_files]
    if not files:
        raise TraceProfileCacheError("Profile 文件集合为空。")

    selected = tuple(files)
    fingerprint = fingerprint_trace_files(trace_root=root, files=selected)
    base_cache = Path(cache_path).resolve()
    actual_cache = base_cache
    if max_files is not None:
        actual_cache = base_cache.with_name(
            f"{base_cache.stem}_debug_{max_files}files{base_cache.suffix}"
        )

    if not refresh_cache:
        cached = _load_cache(
            cache_path=actual_cache,
            expected_fingerprint=fingerprint,
            expected_file_count=len(selected),
        )
        if cached is not None:
            validate_profile(cached, strict=strict)
            if verbose:
                print(
                    "[TraceProfileCache] HIT "
                    f"files={len(selected)} -> {actual_cache}"
                )
            return cached, TraceProfileLoadInfo(
                cache_hit=True,
                cache_path=actual_cache,
                file_count=len(selected),
                file_fingerprint=fingerprint,
                workers=0,
            )

    resolved_workers = _resolve_worker_count(workers)
    if verbose:
        print(
            "[TraceProfileCache] MISS "
            f"files={len(selected)}, workers={resolved_workers}"
        )

    if resolved_workers <= 1 or len(selected) < 2:
        profile = _build_profile_sequential(
            root=root,
            files=selected,
            strict=strict,
            verbose=verbose,
        )
        resolved_workers = 1
    else:
        profile = _build_profile_parallel(
            root=root,
            files=selected,
            workers=min(resolved_workers, len(selected)),
            strict=strict,
            verbose=verbose,
        )

    _save_cache(
        cache_path=actual_cache,
        profile=profile,
        file_fingerprint=fingerprint,
        file_count=len(selected),
    )
    if verbose:
        print(f"[TraceProfileCache] SAVED -> {actual_cache}")

    return profile, TraceProfileLoadInfo(
        cache_hit=False,
        cache_path=actual_cache,
        file_count=len(selected),
        file_fingerprint=fingerprint,
        workers=resolved_workers,
    )
