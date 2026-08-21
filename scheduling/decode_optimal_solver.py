"""
Decode 单个 MoE Layer 的 CP-SAT 最优调度器。

用途：
    作为 optimality oracle，判断当前 Greedy Decode Layer Scheduler
    距离理论最优 Layer makespan 还有多少空间。

当前模型严格对应项目 Baseline：
1. 一个 Decode Layer：8 Routed Expert + 1 Shared Expert = 9 Expert；
2. 每个 Expert 有 gate / up / down 三个任务，共 27 tasks；
3. gate / up 无前驱；down 等待本 Expert 的 gate 和 up；
4. 每个任务固定落在 Mapping 给定的 Sub-Cube；
5. 不同 Sub-Cube 完全并行；同一个 Sub-Cube 任务不能重叠；
6. 当前 Decode 单 Token、单 Layer 内 Weight-Cube 不重复，且
   activation/switch=1、compute=1，因此每个任务固定占用目标 SC：
       2 cycles；
7. 目标：minimize Layer makespan。

注意：
    本文件不修改 scheduling/layer_scheduler.py，也不替换正式 Decode 主链。
    它只是一个“理论最优参考尺”。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from config import ExecutionRules
from mapping.logical_weight import MATRIX_DOWN, MATRIX_GATE, MATRIX_UP
from scheduling.runtime_index import RuntimeIndex


class DecodeOptimalSolverError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DecodeOptimalTask:
    expert_id: int
    matrix_name: str
    subcube_id: int
    cube_id: int
    start_time: int
    finish_time: int


@dataclass(frozen=True, slots=True)
class DecodeOptimalLayerResult:
    layer_id: int
    routed_expert_ids: tuple[int, ...]
    active_expert_ids: tuple[int, ...]
    status: str
    proven_optimal: bool
    feasible: bool
    objective_cycles: int | None
    best_bound_cycles: float | None
    wall_time_seconds: float
    branches: int
    conflicts: int
    tasks: tuple[DecodeOptimalTask, ...]

    @property
    def optimal_cycles(self) -> int | None:
        """只有已经证明 OPTIMAL 时才返回真正理论最优值。"""
        if not self.proven_optimal:
            return None
        return self.objective_cycles


_MATRIX_ORDER = {
    MATRIX_GATE: 0,
    MATRIX_UP: 1,
    MATRIX_DOWN: 2,
}


def _import_cp_model():
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise DecodeOptimalSolverError(
            "缺少 OR-Tools。请先执行：pip install ortools"
        ) from exc
    return cp_model


def _validate_baseline_rules(rules: ExecutionRules) -> None:
    if (
        rules.compute_cycles != 1
        or rules.switch_cycles != 1
        or rules.cross_subcube_cycles != 0
        or not rules.unlimited_parallel_subcubes
        or not rules.one_active_weight_cube_per_subcube
    ):
        raise DecodeOptimalSolverError(
            "当前 CP-SAT Oracle 只支持正式 Baseline："
            "compute=1, switch=1, crossSC=0, SC 间并行、SC 内互斥。"
        )


def _task_key(expert_id: int, matrix_name: str) -> tuple[int, str]:
    return (expert_id, matrix_name)


def _normalize_hint(
    hint_start_times: Mapping[tuple[int, str], int] | None,
) -> dict[tuple[int, str], int]:
    if hint_start_times is None:
        return {}
    result: dict[tuple[int, str], int] = {}
    for key, value in hint_start_times.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise DecodeOptimalSolverError("hint key 必须是 (expert_id, matrix_name)。")
        expert_id, matrix_name = key
        if matrix_name not in _MATRIX_ORDER:
            raise DecodeOptimalSolverError(f"非法 hint matrix_name={matrix_name!r}。")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise DecodeOptimalSolverError("hint start_time 必须是非负整数。")
        result[(int(expert_id), matrix_name)] = value
    return result


def validate_cp_sat_schedule(
    *,
    result: DecodeOptimalLayerResult,
    service_cycles: int = 2,
) -> None:
    """对求解器返回的调度再做一次独立 Python 检查。"""
    if not result.feasible:
        return

    lookup = {
        (task.expert_id, task.matrix_name): task
        for task in result.tasks
    }

    expected_count = len(result.active_expert_ids) * 3
    if len(result.tasks) != expected_count or len(lookup) != expected_count:
        raise DecodeOptimalSolverError(
            f"CP-SAT task 数错误：actual={len(result.tasks)}, expected={expected_count}。"
        )

    for task in result.tasks:
        if task.start_time < 0:
            raise DecodeOptimalSolverError("CP-SAT 出现负 start_time。")
        if task.finish_time - task.start_time != service_cycles:
            raise DecodeOptimalSolverError("CP-SAT task duration 与 Baseline 不一致。")

    for expert_id in result.active_expert_ids:
        gate = lookup[(expert_id, MATRIX_GATE)]
        up = lookup[(expert_id, MATRIX_UP)]
        down = lookup[(expert_id, MATRIX_DOWN)]
        if down.start_time < gate.finish_time:
            raise DecodeOptimalSolverError(f"Expert-{expert_id} down 早于 gate 完成。")
        if down.start_time < up.finish_time:
            raise DecodeOptimalSolverError(f"Expert-{expert_id} down 早于 up 完成。")

    by_sc: dict[int, list[DecodeOptimalTask]] = {}
    for task in result.tasks:
        by_sc.setdefault(task.subcube_id, []).append(task)

    for sc, tasks in by_sc.items():
        ordered = sorted(tasks, key=lambda x: (x.start_time, x.finish_time, x.expert_id))
        for left, right in zip(ordered, ordered[1:]):
            if right.start_time < left.finish_time:
                raise DecodeOptimalSolverError(
                    f"SC-{sc} 出现任务重叠："
                    f"{left.expert_id}/{left.matrix_name} 与 "
                    f"{right.expert_id}/{right.matrix_name}。"
                )

    if result.objective_cycles is None:
        raise DecodeOptimalSolverError("可行解缺少 objective_cycles。")

    actual_makespan = max((task.finish_time for task in result.tasks), default=0)
    if actual_makespan != result.objective_cycles:
        raise DecodeOptimalSolverError(
            f"makespan 不一致：tasks={actual_makespan}, objective={result.objective_cycles}。"
        )


def solve_decode_layer_optimal(
    *,
    index: RuntimeIndex,
    layer_id: int,
    routed_expert_ids: Iterable[int],
    rules: ExecutionRules | None = None,
    time_limit_seconds: float = 5.0,
    num_workers: int = 8,
    greedy_upper_bound_cycles: int | None = None,
    hint_start_times: Mapping[tuple[int, str], int] | None = None,
    validate_solution: bool = True,
) -> DecodeOptimalLayerResult:
    """
    求一个真实 Decode Layer 的最小 makespan。

    greedy_upper_bound_cycles：
        可传当前 Greedy Layer cycles。因为 Greedy 本身是可行解，
        加上 makespan <= greedy 可以安全缩小搜索空间，不会排除最优解。

    hint_start_times：
        可传当前 Greedy 的 dispatch_time，给 CP-SAT 一个高质量初始提示。
    """
    cp_model = _import_cp_model()

    if rules is None:
        rules = ExecutionRules()
    _validate_baseline_rules(rules)

    if time_limit_seconds <= 0:
        raise DecodeOptimalSolverError("time_limit_seconds 必须大于 0。")
    if num_workers <= 0:
        raise DecodeOptimalSolverError("num_workers 必须大于 0。")

    routed = tuple(routed_expert_ids)
    active_ids = index.resolve_active_expert_ids(
        layer_id=layer_id,
        routed_expert_ids=routed,
    )

    service_cycles = rules.switch_cycles + rules.compute_cycles
    task_count = len(active_ids) * 3
    horizon = task_count * service_cycles

    model = cp_model.CpModel()

    start_vars: dict[tuple[int, str], object] = {}
    end_vars: dict[tuple[int, str], object] = {}
    interval_vars: dict[tuple[int, str], object] = {}
    task_meta: dict[tuple[int, str], tuple[int, int]] = {}
    intervals_by_sc: list[list[object]] = [
        [] for _ in range(index.num_subcubes)
    ]

    for expert_id in active_ids:
        expert = index.expert(layer_id, expert_id)
        for matrix_name, location in (
            (MATRIX_GATE, expert.gate),
            (MATRIX_UP, expert.up),
            (MATRIX_DOWN, expert.down),
        ):
            key = _task_key(expert_id, matrix_name)
            safe_name = matrix_name.replace("_proj", "")
            start = model.new_int_var(0, horizon, f"s_e{expert_id}_{safe_name}")
            end = model.new_int_var(0, horizon, f"e_e{expert_id}_{safe_name}")
            interval = model.new_interval_var(
                start,
                service_cycles,
                end,
                f"i_e{expert_id}_{safe_name}",
            )
            start_vars[key] = start
            end_vars[key] = end
            interval_vars[key] = interval
            task_meta[key] = (location.subcube_id, location.cube_id)
            intervals_by_sc[location.subcube_id].append(interval)

    # gate/up -> down
    for expert_id in active_ids:
        down_start = start_vars[_task_key(expert_id, MATRIX_DOWN)]
        model.add(
            down_start >= end_vars[_task_key(expert_id, MATRIX_GATE)]
        )
        model.add(
            down_start >= end_vars[_task_key(expert_id, MATRIX_UP)]
        )

    # 同一 SC 串行
    for intervals in intervals_by_sc:
        if len(intervals) >= 2:
            model.add_no_overlap(intervals)

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, list(end_vars.values()))

    if greedy_upper_bound_cycles is not None:
        if greedy_upper_bound_cycles <= 0:
            raise DecodeOptimalSolverError("greedy_upper_bound_cycles 必须大于 0。")
        model.add(makespan <= greedy_upper_bound_cycles)

    hint = _normalize_hint(hint_start_times)
    for key, value in hint.items():
        var = start_vars.get(key)
        if var is not None:
            model.add_hint(var, value)

    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(num_workers)

    status_code = solver.solve(model)
    status_name = solver.status_name(status_code)
    feasible = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    proven_optimal = status_code == cp_model.OPTIMAL

    objective_cycles: int | None = None
    best_bound: float | None = None
    tasks: list[DecodeOptimalTask] = []

    if feasible:
        objective_cycles = int(round(solver.objective_value))
        best_bound = float(solver.best_objective_bound)

        for expert_id in active_ids:
            for matrix_name in (MATRIX_GATE, MATRIX_UP, MATRIX_DOWN):
                key = _task_key(expert_id, matrix_name)
                sc, cube_id = task_meta[key]
                start = int(solver.value(start_vars[key]))
                end = int(solver.value(end_vars[key]))
                tasks.append(
                    DecodeOptimalTask(
                        expert_id=expert_id,
                        matrix_name=matrix_name,
                        subcube_id=sc,
                        cube_id=cube_id,
                        start_time=start,
                        finish_time=end,
                    )
                )
    else:
        # UNKNOWN/INFEASIBLE/MODEL_INVALID 也可能有有效 lower bound，
        # 但这里不把它伪装成已求出的最优周期。
        try:
            best_bound = float(solver.best_objective_bound)
        except Exception:
            best_bound = None

    tasks.sort(
        key=lambda task: (
            task.start_time,
            task.subcube_id,
            task.finish_time,
            task.expert_id,
            _MATRIX_ORDER[task.matrix_name],
        )
    )

    result = DecodeOptimalLayerResult(
        layer_id=layer_id,
        routed_expert_ids=routed,
        active_expert_ids=active_ids,
        status=status_name,
        proven_optimal=proven_optimal,
        feasible=feasible,
        objective_cycles=objective_cycles,
        best_bound_cycles=best_bound,
        wall_time_seconds=float(solver.wall_time),
        branches=int(solver.num_branches),
        conflicts=int(solver.num_conflicts),
        tasks=tuple(tasks),
    )

    if validate_solution and feasible:
        validate_cp_sat_schedule(result=result, service_cycles=service_cycles)

    return result
