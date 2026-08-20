from __future__ import annotations

import random
from pathlib import Path

import pytest

from config import ExecutionRules
from scheduling.decode_fast_evaluator import FastDecodeScheduler
from scheduling.prefill_fast_evaluator import FastPrefillScheduler
from scheduling.prefill_scheduler import schedule_prefill_batch
from scheduling.runtime_index import DEFAULT_MAPPING_PATH, load_runtime_index
from scheduling.token_scheduler import schedule_token


@pytest.fixture(scope="module")
def runtime_index():
    path = Path(DEFAULT_MAPPING_PATH).resolve()
    if not path.exists():
        pytest.skip(f"Mapping 不存在：{path}")
    return load_runtime_index(path)


def _random_token_routes(seed: int) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    return tuple(
        tuple(rng.sample(range(256), 8))
        for _ in range(58)
    )


def test_optimized_decode_matches_exact_random_routes(runtime_index):
    rules = ExecutionRules()
    fast = FastDecodeScheduler(
        index=runtime_index,
        rules=rules,
        cache_size=10000,
    )

    for seed in range(10):
        routes = _random_token_routes(seed)
        fast_result = fast.schedule_token(routes, validate_routes=False)
        exact = schedule_token(
            index=runtime_index,
            routed_experts_by_layer=routes,
            rules=rules,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=True,
        )
        assert fast_result.total_cycles == exact.total_cycles
        assert fast_result.layer_cycles == tuple(x.cycles for x in exact.layers)


def test_fast_prefill_matches_exact_random_batches(runtime_index):
    rules = ExecutionRules()
    fast = FastPrefillScheduler(index=runtime_index, rules=rules)

    for batch_size, seed in ((1, 101), (4, 102), (16, 103)):
        rng = random.Random(seed)
        batch = tuple(
            tuple(
                tuple(rng.sample(range(256), 8))
                for _ in range(58)
            )
            for _ in range(batch_size)
        )

        fast_result = fast.schedule_batch(
            batch,
            charge_initial_activation=True,
            validate_routes=False,
        )
        exact = schedule_prefill_batch(
            index=runtime_index,
            routed_experts_by_token=batch,
            rules=rules,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=True,
        )

        assert fast_result.total_cycles == exact.total_cycles
        assert tuple(x.cycles for x in fast_result.layers) == tuple(
            x.cycles for x in exact.layers
        )
        assert fast_result.total_tasks == exact.total_tasks
        assert fast_result.total_switches == exact.total_switches
        assert (
            fast_result.total_initial_activations
            == exact.total_initial_activations
        )
        assert (
            fast_result.total_activation_overhead_cycles
            == exact.total_activation_overhead_cycles
        )
        assert (
            fast_result.total_compute_work_cycles
            == exact.total_compute_work_cycles
        )
        assert fast_result.total_busy_cycles == exact.total_busy_cycles
        assert fast_result.total_wait_cycles == exact.total_wait_cycles
        assert fast_result.max_task_wait_cycles == exact.max_task_wait_cycles
        assert (
            fast_result.final_active_cube_by_subcube
            == exact.final_active_cube_by_subcube
        )
