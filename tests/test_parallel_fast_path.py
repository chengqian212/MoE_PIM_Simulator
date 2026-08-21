"""Fast 内核 + Windows spawn 多进程一致性测试。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from scheduling.decode_fast_evaluator import evaluate_decode_fast
from scheduling.prefill_fast_evaluator import evaluate_prefill_fast
from scheduling.prefill_scheduling_mode import PREFILL_SCHEDULING_MODES
from scheduling.runtime_index import DEFAULT_MAPPING_PATH, load_runtime_index


def _route(seed: int) -> list[int]:
    base = (seed * 11) % 240
    return [base + i for i in range(8)]


def _segment(token_count: int, seed: int) -> dict[str, list[list[int]]]:
    segment: dict[str, list[list[int]]] = {}
    for trace_layer in range(3, 61):
        segment[str(trace_layer)] = [
            _route(seed + trace_layer * 17 + token_index)
            for token_index in range(token_count)
        ]
    return segment


def _write_tiny_trace(root: Path) -> None:
    for file_index in range(4):
        category = root / ("cat_a" if file_index % 2 == 0 else "cat_b")
        category.mkdir(parents=True, exist_ok=True)
        payload = [
            _segment(3 + (file_index % 2), seed=1000 * file_index),
            _segment(1, seed=1000 * file_index + 101),
            _segment(1, seed=1000 * file_index + 202),
            _segment(1, seed=1000 * file_index + 303),
        ]
        (category / f"trace_{file_index}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


@pytest.fixture(scope="module")
def runtime_index():
    path = Path(DEFAULT_MAPPING_PATH).resolve()
    if not path.exists():
        pytest.skip(f"Mapping 不存在：{path}")
    return load_runtime_index(path)


@pytest.mark.parametrize("scheduling_mode", PREFILL_SCHEDULING_MODES)
def test_prefill_parallel_matches_single(runtime_index, tmp_path, scheduling_mode):
    root = tmp_path / "trace"
    _write_tiny_trace(root)

    single_summary, single_records = evaluate_prefill_fast(
        index=runtime_index,
        trace_root=root,
        exact_check=0,
        workers=1,
        verbose=False,
        scheduling_mode=scheduling_mode,
    )
    parallel_summary, parallel_records = evaluate_prefill_fast(
        index=runtime_index,
        trace_root=root,
        exact_check=0,
        workers=2,
        verbose=False,
        scheduling_mode=scheduling_mode,
    )

    assert parallel_records == single_records
    assert asdict(parallel_summary) == asdict(single_summary)


def test_decode_parallel_matches_single(runtime_index, tmp_path):
    root = tmp_path / "trace"
    _write_tiny_trace(root)

    single_summary, single_records, single_stats = evaluate_decode_fast(
        index=runtime_index,
        trace_root=root,
        exact_check=0,
        cache_size=4000,
        workers=1,
        verbose=False,
    )
    parallel_summary, parallel_records, parallel_stats = evaluate_decode_fast(
        index=runtime_index,
        trace_root=root,
        exact_check=0,
        cache_size=4000,
        workers=2,
        verbose=False,
    )

    assert parallel_records == single_records
    assert asdict(parallel_stats) == asdict(single_stats)

    single_payload = asdict(single_summary)
    parallel_payload = asdict(parallel_summary)
    for key in ("cache_hits", "cache_misses", "cache_currsize"):
        single_payload.pop(key)
        parallel_payload.pop(key)
    assert parallel_payload == single_payload
