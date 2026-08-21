from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ortools")

from config import ExecutionRules
from scheduling.decode_optimal_solver import solve_decode_layer_optimal
from scheduling.layer_scheduler import schedule_layer
from scheduling.runtime_index import DEFAULT_MAPPING_PATH, load_runtime_index


@pytest.fixture(scope="module")
def runtime_index():
    path = Path(DEFAULT_MAPPING_PATH).resolve()
    if not path.exists():
        pytest.skip(f"Mapping 不存在：{path}")
    return load_runtime_index(path)


def test_cp_sat_decode_layer_not_worse_than_greedy(runtime_index):
    rules = ExecutionRules()
    route = (0, 1, 2, 3, 4, 5, 6, 7)

    greedy = schedule_layer(
        index=runtime_index,
        layer_id=0,
        routed_expert_ids=route,
        rules=rules,
        initial_active_cube_by_subcube=None,
        charge_initial_activation=True,
    )
    hint = {
        (task.expert_id, task.matrix_name): task.dispatch_time
        for task in greedy.tasks
    }

    cp = solve_decode_layer_optimal(
        index=runtime_index,
        layer_id=0,
        routed_expert_ids=route,
        rules=rules,
        time_limit_seconds=10.0,
        num_workers=1,
        greedy_upper_bound_cycles=greedy.total_cycles,
        hint_start_times=hint,
        validate_solution=True,
    )

    assert cp.feasible
    assert cp.objective_cycles is not None
    assert cp.objective_cycles <= greedy.total_cycles
    assert len(cp.tasks) == 27

    if cp.proven_optimal:
        assert cp.optimal_cycles == cp.objective_cycles
