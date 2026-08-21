"""Profile/Held-out split, cache, and workload isolation tests."""

from __future__ import annotations

import json
from pathlib import Path

from mapping.trace_profile_cache import load_or_build_trace_profile
from mapping.trace_split import (
    EVALUATION_SUBSET,
    PROFILE_SUBSET,
    ensure_trace_split,
    load_trace_split_manifest,
    resolve_trace_files,
)
from scheduling.decode_workload import DecodeWorkloadStats, iter_decode_tokens
from scheduling.prefill_workload import PrefillWorkloadStats, iter_prefill_batches


def _route(seed: int) -> list[int]:
    base = (seed * 13) % 240
    return [base + i for i in range(8)]


def _segment(token_count: int, seed: int) -> dict[str, list[list[int]]]:
    return {
        str(trace_layer): [
            _route(seed + trace_layer * 17 + token_index)
            for token_index in range(token_count)
        ]
        for trace_layer in range(3, 61)
    }


def _write_trace(root: Path, files_per_category: int = 5) -> None:
    for category_index, category_name in enumerate(("cat_a", "cat_b")):
        category = root / category_name
        category.mkdir(parents=True, exist_ok=True)
        for file_index in range(files_per_category):
            seed = category_index * 10000 + file_index * 1000
            payload = [
                _segment(3, seed),
                _segment(1, seed + 101),
            ]
            (category / f"trace_{file_index}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )


def test_stratified_split_is_deterministic_and_disjoint(tmp_path):
    root = tmp_path / "trace"
    _write_trace(root)
    manifest_path = tmp_path / "split.json"

    first, saved, rebuilt = ensure_trace_split(
        trace_root=root,
        manifest_path=manifest_path,
        profile_ratio=0.8,
        seed=42,
        force=False,
        verbose=False,
    )
    assert rebuilt
    second, saved2, rebuilt2 = ensure_trace_split(
        trace_root=root,
        manifest_path=manifest_path,
        profile_ratio=0.8,
        seed=42,
        force=False,
        verbose=False,
    )
    assert not rebuilt2
    assert saved == saved2
    assert first == second

    profile = set(first[PROFILE_SUBSET]["files"])
    evaluation = set(first[EVALUATION_SUBSET]["files"])
    assert profile.isdisjoint(evaluation)
    assert len(profile | evaluation) == 10

    # 5 files/category -> 4 profile + 1 evaluation per category.
    assert first[PROFILE_SUBSET]["file_count"] == 8
    assert first[EVALUATION_SUBSET]["file_count"] == 2
    for counts in first["category_counts"].values():
        assert counts["profile"] == 4
        assert counts["evaluation"] == 1


def test_profile_cache_hits_on_second_load(tmp_path):
    root = tmp_path / "trace"
    _write_trace(root)
    manifest_path = tmp_path / "split.json"
    cache_path = tmp_path / "profile.pkl"

    manifest, _, _ = ensure_trace_split(
        trace_root=root,
        manifest_path=manifest_path,
        profile_ratio=0.5,
        seed=7,
        verbose=False,
    )

    profile1, info1 = load_or_build_trace_profile(
        trace_root=root,
        manifest_path=manifest_path,
        subset=PROFILE_SUBSET,
        cache_path=cache_path,
        workers=1,
        strict=True,
        verbose=False,
    )
    profile2, info2 = load_or_build_trace_profile(
        trace_root=root,
        manifest_path=manifest_path,
        subset=PROFILE_SUBSET,
        cache_path=cache_path,
        workers=1,
        strict=True,
        verbose=False,
    )

    assert not info1.cache_hit
    assert info2.cache_hit
    assert profile1.frequency == profile2.frequency
    assert profile1.token_count_by_layer == profile2.token_count_by_layer

    # Each selected profile file contains 3 prefill tokens + 1 decode token.
    expected_tokens_per_layer = manifest[PROFILE_SUBSET]["file_count"] * 4
    assert profile1.tokens_per_layer == expected_tokens_per_layer


def test_prefill_decode_iterators_only_use_held_out_files(tmp_path):
    root = tmp_path / "trace"
    _write_trace(root)
    manifest_path = tmp_path / "split.json"

    manifest, _, _ = ensure_trace_split(
        trace_root=root,
        manifest_path=manifest_path,
        profile_ratio=0.8,
        seed=42,
        verbose=False,
    )
    eval_count = manifest[EVALUATION_SUBSET]["file_count"]

    eval_files = resolve_trace_files(
        trace_root=root,
        manifest_path=manifest_path,
        subset=EVALUATION_SUBSET,
    )
    assert len(eval_files) == eval_count

    prefill_stats = PrefillWorkloadStats()
    prefill = list(
        iter_prefill_batches(
            trace_root=root,
            trace_manifest=manifest_path,
            trace_subset=EVALUATION_SUBSET,
            stats=prefill_stats,
            verbose=False,
        )
    )
    assert len(prefill) == eval_count
    assert {batch.relative_file for batch in prefill} == {
        str(path.relative_to(root)) for path in eval_files
    }

    decode_stats = DecodeWorkloadStats()
    decode = list(
        iter_decode_tokens(
            trace_root=root,
            trace_manifest=manifest_path,
            trace_subset=EVALUATION_SUBSET,
            stats=decode_stats,
            verbose=False,
        )
    )
    assert len(decode) == eval_count  # one segment1 per file in this fixture
    assert {token.relative_file for token in decode} == {
        str(path.relative_to(root)) for path in eval_files
    }
