from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import ExecutionRules
from scheduling.decode_fast_evaluator import FastDecodeScheduler
from scheduling.prefill_fast_evaluator import FastPrefillScheduler
from scheduling.prefill_scheduler import schedule_prefill_batch
from scheduling.prefill_layer_scheduler import _PrefillTaskSpec, _select_ready_task
from scheduling.prefill_scheduling_mode import (
    PREFILL_MODE_LARGEST_BATCH_REUSE,
    PREFILL_SCHEDULING_MODES,
)
from scheduling.runtime_index import DEFAULT_MAPPING_PATH, load_runtime_index
from mapping.logical_weight import MATRIX_GATE
from scheduling.token_scheduler import schedule_token


def _fake_prefill_task(*, cube_id: int, ready_time: int, route_rank: int, token_index: int):
    return _PrefillTaskSpec(
        token_index=token_index,
        layer_id=0,
        expert_id=route_rank,
        route_rank=route_rank,
        matrix_name=MATRIX_GATE,
        location=SimpleNamespace(cube_id=cube_id, subcube_id=0),
        ready_time=ready_time,
    )


def test_largest_batch_reuse_priority_uses_only_current_ready_tasks():
    queue = [
        _fake_prefill_task(cube_id=10, ready_time=0, route_rank=0, token_index=0),
        _fake_prefill_task(cube_id=20, ready_time=0, route_rank=5, token_index=0),
        _fake_prefill_task(cube_id=20, ready_time=0, route_rank=6, token_index=1),
        # cube 30 虽然未来会有 3 个任务，但 t=0 时不能偷看。
        _fake_prefill_task(cube_id=30, ready_time=1, route_rank=1, token_index=0),
        _fake_prefill_task(cube_id=30, ready_time=1, route_rank=2, token_index=1),
        _fake_prefill_task(cube_id=30, ready_time=1, route_rank=3, token_index=2),
    ]

    selected = _select_ready_task(
        queue=queue,
        current_time=0,
        active_cube_id=None,
        scheduling_mode=PREFILL_MODE_LARGEST_BATCH_REUSE,
    )
    assert selected is not None
    assert selected.location.cube_id == 20

    # 当前 WC 还有 Ready Task 时，仍然保持 aggressive reuse。
    selected_active = _select_ready_task(
        queue=queue,
        current_time=0,
        active_cube_id=10,
        scheduling_mode=PREFILL_MODE_LARGEST_BATCH_REUSE,
    )
    assert selected_active is not None
    assert selected_active.location.cube_id == 10


@pytest.fixture(scope="module")
def runtime_index():
    path = Path(DEFAULT_MAPPING_PATH).resolve()
    if not path.exists():
        pytest.skip(f"Mapping 不存在：{path}")
    return load_runtime_index(path)


def _random_token_routes(seed: int) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    return tuple(tuple(rng.sample(range(256), 8)) for _ in range(58))


def test_optimized_decode_matches_exact_random_routes(runtime_index):
    rules = ExecutionRules()
    fast = FastDecodeScheduler(index=runtime_index, rules=rules, cache_size=10000)

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


@pytest.mark.parametrize("scheduling_mode", PREFILL_SCHEDULING_MODES)
def test_fast_prefill_matches_exact_random_batches(runtime_index, scheduling_mode):
    rules = ExecutionRules()
    fast = FastPrefillScheduler(
        index=runtime_index,
        rules=rules,
        scheduling_mode=scheduling_mode,
    )

    for batch_size, seed in ((1, 101), (4, 102), (16, 103)):
        rng = random.Random(seed)
        batch = tuple(
            tuple(tuple(rng.sample(range(256), 8)) for _ in range(58))
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
            scheduling_mode=scheduling_mode,
        )

        assert fast_result.total_cycles == exact.total_cycles
        assert tuple(x.cycles for x in fast_result.layers) == tuple(
            x.cycles for x in exact.layers
        )
        assert fast_result.total_tasks == exact.total_tasks
        assert fast_result.total_switches == exact.total_switches
        assert fast_result.total_initial_activations == exact.total_initial_activations
        assert (
            fast_result.total_activation_overhead_cycles
            == exact.total_activation_overhead_cycles
        )
        assert fast_result.total_compute_work_cycles == exact.total_compute_work_cycles
        assert fast_result.total_busy_cycles == exact.total_busy_cycles
        assert fast_result.total_wait_cycles == exact.total_wait_cycles
        assert fast_result.max_task_wait_cycles == exact.max_task_wait_cycles
        assert fast_result.final_active_cube_by_subcube == exact.final_active_cube_by_subcube


def test_default_prefill_mode_is_switch_aware(runtime_index):
    rules = ExecutionRules()
    rng = random.Random(999)
    batch = tuple(
        tuple(tuple(rng.sample(range(256), 8)) for _ in range(58))
        for _ in range(8)
    )

    default_result = schedule_prefill_batch(
        index=runtime_index,
        routed_experts_by_token=batch,
        rules=rules,
    )
    explicit_result = schedule_prefill_batch(
        index=runtime_index,
        routed_experts_by_token=batch,
        rules=rules,
        scheduling_mode="switch_aware",
    )

    assert default_result.total_cycles == explicit_result.total_cycles
    assert default_result.total_switches == explicit_result.total_switches
    assert default_result.total_wait_cycles == explicit_result.total_wait_cycles
