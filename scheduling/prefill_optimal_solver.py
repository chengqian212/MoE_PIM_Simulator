"""Prefill 单个 MoE Layer 的 CP-SAT 最优调度器。

用途：
    作为 optimality oracle，判断当前 Prefill heuristic scheduler
    距离“固定 Mapping + 固定 Batch Route + 固定初始 active WC 状态”下的
    单层理论最优 makespan 还有多少空间。

当前模型严格对应项目既定 Prefill Baseline：
1. 一个 Prefill Layer 有 B 个 Token；
2. 每个 Token：8 Routed Expert + 1 Shared Expert = 9 Expert；
3. 每个 Token / Expert 都有 gate / up / down 三个任务；
4. gate / up 无前驱；down 只等待“同 Token、同 Expert”的 gate 和 up；
5. 每个任务固定落在 Mapping 给定的 Sub-Cube / Weight-Cube；
6. 不同 Sub-Cube 完全并行；同一个 Sub-Cube 同一时刻只能服务一个任务；
7. 每次矩阵 compute 固定为 compute_cycles（当前 Baseline=1）；
8. 同一 SC 连续访问相同 Weight-Cube：setup/switch=0；
   连续访问不同 Weight-Cube：setup/switch=switch_cycles（当前=1）；
9. 每个 SC 的初始 active Weight-Cube 可由上一层传入；
10. 允许 Solver 主动 idle；
11. 目标：minimize Layer makespan；
12. 对同一 Expert 的多 Token 等价副本，仅对 gate 任务加入安全的 token-copy symmetry breaking：
    按 token_index 规范化 gate 的执行先后，用于删掉仅 Token 标签不同的重复搜索。

和 Decode CP-SAT 的关键区别：
    Decode 单 Token / 单 Layer 内每个 WC 只访问一次，可以把每个任务近似成固定
    switch+compute=2 cycles；Prefill 会重复访问同一个 WC，因此切换开销与任务顺序有关，
    必须显式建模 sequence-dependent setup time。

注意：
    本文件不替换 scheduling/prefill_layer_scheduler.py。
    它只作为“小/中规模真实 Prefill Layer”的理论最优参考尺。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from config import ExecutionRules
from mapping.logical_weight import MATRIX_DOWN, MATRIX_GATE, MATRIX_UP
from scheduling.runtime_index import RuntimeIndex


class PrefillOptimalSolverError(ValueError):
    pass


TaskKey = tuple[int, int, str]
# (token_index, expert_id, matrix_name)


@dataclass(frozen=True, slots=True)
class PrefillOptimalTask:
    token_index: int
    expert_id: int
    matrix_name: str
    subcube_id: int
    cube_id: int

    # dispatch_time：开始 activation/switch 的时刻；
    # compute_start_time：真正开始 compute 的时刻；
    # finish_time：compute 完成时刻。
    dispatch_time: int
    compute_start_time: int
    finish_time: int

    activation_cycles: int
    previous_active_cube_id: int | None

    @property
    def compute_cycles(self) -> int:
        return self.finish_time - self.compute_start_time

    @property
    def total_service_cycles(self) -> int:
        return self.activation_cycles + self.compute_cycles


@dataclass(frozen=True, slots=True)
class PrefillOptimalLayerResult:
    layer_id: int
    token_count: int
    routed_experts_by_token: tuple[tuple[int, ...], ...]
    active_expert_ids_by_token: tuple[tuple[int, ...], ...]

    status: str
    proven_optimal: bool
    feasible: bool

    objective_cycles: int | None
    best_bound_cycles: float | None

    wall_time_seconds: float
    branches: int
    conflicts: int

    initial_active_cube_by_subcube: tuple[int | None, ...]
    final_active_cube_by_subcube: tuple[int | None, ...]

    tasks: tuple[PrefillOptimalTask, ...]

    @property
    def optimal_cycles(self) -> int | None:
        """只有已经证明 OPTIMAL 时才返回真正理论最优值。"""
        if not self.proven_optimal:
            return None
        return self.objective_cycles

    @property
    def task_count(self) -> int:
        return len(self.tasks)


_MATRIX_ORDER = {
    MATRIX_GATE: 0,
    MATRIX_UP: 1,
    MATRIX_DOWN: 2,
}


def _import_cp_model():
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise PrefillOptimalSolverError(
            "缺少 OR-Tools。请先执行：pip install ortools"
        ) from exc
    return cp_model


def _validate_baseline_rules(rules: ExecutionRules) -> None:
    if (
        rules.compute_cycles <= 0
        or rules.switch_cycles < 0
        or rules.cross_subcube_cycles != 0
        or not rules.unlimited_parallel_subcubes
        or not rules.one_active_weight_cube_per_subcube
    ):
        raise PrefillOptimalSolverError(
            "当前 Prefill CP-SAT Oracle 只支持项目既定 Baseline："
            "compute>0, switch>=0, crossSC=0, SC 间并行、SC 内互斥。"
        )


def _normalize_routes(
    *,
    index: RuntimeIndex,
    layer_id: int,
    routed_experts_by_token: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], ...]:
    index.layer(layer_id)

    routes = tuple(tuple(route) for route in routed_experts_by_token)
    if not routes:
        raise PrefillOptimalSolverError("Prefill Layer 至少需要一个 Token。")

    for token_index, route in enumerate(routes):
        try:
            index.resolve_active_expert_ids(
                layer_id=layer_id,
                routed_expert_ids=route,
            )
        except ValueError as exc:
            raise PrefillOptimalSolverError(
                f"Token-{token_index} Router Route 非法。"
            ) from exc

    return routes


def _normalize_initial_state(
    *,
    index: RuntimeIndex,
    initial_active_cube_by_subcube: Iterable[int | None] | None,
) -> tuple[int | None, ...]:
    if initial_active_cube_by_subcube is None:
        return tuple(None for _ in range(index.num_subcubes))

    state = tuple(initial_active_cube_by_subcube)
    if len(state) != index.num_subcubes:
        raise PrefillOptimalSolverError(
            "initial_active_cube_by_subcube 长度错误："
            f"actual={len(state)}, expected={index.num_subcubes}。"
        )

    for cube_id in state:
        if cube_id is None:
            continue
        if (
            not isinstance(cube_id, int)
            or isinstance(cube_id, bool)
            or cube_id < 0
        ):
            raise PrefillOptimalSolverError(
                "初始 active cube 必须为 None 或非负整数。"
            )

    return state


def _initial_activation_cost(
    *,
    initial_cube_id: int | None,
    next_cube_id: int,
    rules: ExecutionRules,
    charge_initial_activation: bool,
) -> int:
    if initial_cube_id is None:
        return rules.switch_cycles if charge_initial_activation else 0
    if initial_cube_id == next_cube_id:
        return 0
    return rules.switch_cycles


def _transition_cost(
    *,
    previous_cube_id: int,
    next_cube_id: int,
    rules: ExecutionRules,
) -> int:
    if previous_cube_id == next_cube_id:
        return 0
    return rules.switch_cycles


def _normalize_hint(
    hint_compute_start_times: Mapping[TaskKey, int] | None,
) -> dict[TaskKey, int]:
    if hint_compute_start_times is None:
        return {}

    result: dict[TaskKey, int] = {}
    for key, value in hint_compute_start_times.items():
        if not isinstance(key, tuple) or len(key) != 3:
            raise PrefillOptimalSolverError(
                "hint key 必须是 (token_index, expert_id, matrix_name)。"
            )
        token_index, expert_id, matrix_name = key
        if matrix_name not in _MATRIX_ORDER:
            raise PrefillOptimalSolverError(
                f"非法 hint matrix_name={matrix_name!r}。"
            )
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise PrefillOptimalSolverError(
                "hint compute_start_time 必须是非负整数。"
            )
        result[(int(token_index), int(expert_id), matrix_name)] = value

    return result


def _task_sort_key(task: PrefillOptimalTask) -> tuple[int, int, int, int, int]:
    return (
        task.compute_start_time,
        task.subcube_id,
        task.finish_time,
        task.token_index,
        _MATRIX_ORDER[task.matrix_name],
    )


def validate_cp_sat_prefill_schedule(
    *,
    result: PrefillOptimalLayerResult,
    rules: ExecutionRules | None = None,
    charge_initial_activation: bool = True,
) -> None:
    """对 CP-SAT 返回结果做一次独立 Python 检查。"""
    if not result.feasible:
        return

    if rules is None:
        rules = ExecutionRules()
    _validate_baseline_rules(rules)

    expected_count = sum(
        len(active_ids) * 3
        for active_ids in result.active_expert_ids_by_token
    )
    if len(result.tasks) != expected_count:
        raise PrefillOptimalSolverError(
            "CP-SAT task 数错误："
            f"actual={len(result.tasks)}, expected={expected_count}。"
        )

    lookup: dict[TaskKey, PrefillOptimalTask] = {}
    for task in result.tasks:
        key = (task.token_index, task.expert_id, task.matrix_name)
        if key in lookup:
            raise PrefillOptimalSolverError(f"CP-SAT 出现重复 task：{key}。")
        lookup[key] = task

        if task.dispatch_time < 0:
            raise PrefillOptimalSolverError("CP-SAT 出现负 dispatch_time。")
        if task.compute_start_time < task.dispatch_time:
            raise PrefillOptimalSolverError("compute_start_time 早于 dispatch_time。")
        if task.finish_time - task.compute_start_time != rules.compute_cycles:
            raise PrefillOptimalSolverError("CP-SAT compute duration 与 Baseline 不一致。")
        if task.compute_start_time - task.dispatch_time != task.activation_cycles:
            raise PrefillOptimalSolverError("CP-SAT activation timing 不一致。")

    # gate/up -> down
    for token_index, active_ids in enumerate(result.active_expert_ids_by_token):
        for expert_id in active_ids:
            gate = lookup[(token_index, expert_id, MATRIX_GATE)]
            up = lookup[(token_index, expert_id, MATRIX_UP)]
            down = lookup[(token_index, expert_id, MATRIX_DOWN)]
            if down.compute_start_time < gate.finish_time:
                raise PrefillOptimalSolverError(
                    f"Token-{token_index} Expert-{expert_id} down 早于 gate 完成。"
                )
            if down.compute_start_time < up.finish_time:
                raise PrefillOptimalSolverError(
                    f"Token-{token_index} Expert-{expert_id} down 早于 up 完成。"
                )

    # SC 内顺序 + sequence-dependent switch/setup
    by_sc: dict[int, list[PrefillOptimalTask]] = {}
    for task in result.tasks:
        by_sc.setdefault(task.subcube_id, []).append(task)

    final_state = list(result.initial_active_cube_by_subcube)

    for sc in range(len(result.initial_active_cube_by_subcube)):
        tasks = sorted(by_sc.get(sc, []), key=_task_sort_key)
        previous_finish = 0
        previous_cube = result.initial_active_cube_by_subcube[sc]

        for index_in_sc, task in enumerate(tasks):
            if index_in_sc == 0:
                expected_activation = _initial_activation_cost(
                    initial_cube_id=previous_cube,
                    next_cube_id=task.cube_id,
                    rules=rules,
                    charge_initial_activation=charge_initial_activation,
                )
                earliest_dispatch = 0
            else:
                assert previous_cube is not None
                expected_activation = _transition_cost(
                    previous_cube_id=previous_cube,
                    next_cube_id=task.cube_id,
                    rules=rules,
                )
                earliest_dispatch = previous_finish

            if task.previous_active_cube_id != previous_cube:
                raise PrefillOptimalSolverError(
                    f"SC-{sc} previous_active_cube_id 记录错误。"
                )

            if task.activation_cycles != expected_activation:
                raise PrefillOptimalSolverError(
                    f"SC-{sc} activation_cycles 错误："
                    f"actual={task.activation_cycles}, expected={expected_activation}。"
                )

            if task.dispatch_time < earliest_dispatch:
                raise PrefillOptimalSolverError(
                    f"SC-{sc} 出现任务/切换重叠。"
                )

            previous_finish = task.finish_time
            previous_cube = task.cube_id

        if tasks:
            final_state[sc] = tasks[-1].cube_id

    if tuple(final_state) != result.final_active_cube_by_subcube:
        raise PrefillOptimalSolverError("CP-SAT final active state 与任务顺序不一致。")

    if result.objective_cycles is None:
        raise PrefillOptimalSolverError("可行解缺少 objective_cycles。")

    actual_makespan = max((task.finish_time for task in result.tasks), default=0)
    if actual_makespan != result.objective_cycles:
        raise PrefillOptimalSolverError(
            f"makespan 不一致：tasks={actual_makespan}, "
            f"objective={result.objective_cycles}。"
        )


def solve_prefill_layer_optimal(
    *,
    index: RuntimeIndex,
    layer_id: int,
    routed_experts_by_token: Iterable[Iterable[int]],
    rules: ExecutionRules | None = None,
    initial_active_cube_by_subcube: Iterable[int | None] | None = None,
    charge_initial_activation: bool = True,
    time_limit_seconds: float = 10.0,
    num_workers: int = 8,
    heuristic_upper_bound_cycles: int | None = None,
    hint_compute_start_times: Mapping[TaskKey, int] | None = None,
    validate_solution: bool = True,
    log_search_progress: bool = False,
    max_sequence_arcs: int | None = 300_000,
    target_makespan_cycles: int | None = None,
    minimize_makespan: bool = True,
    enable_token_copy_symmetry_breaking: bool = True,
) -> PrefillOptimalLayerResult:
    """求一个真实 Prefill Layer 的最小 makespan。

    这里固定：
        - Mapping；
        - 当前 Layer 的所有 Token Router Route；
        - 初始 active Weight-Cube 状态；
        - 执行规则。

    Solver 只优化：
        每个 Sub-Cube 上任务的执行顺序和必要的 idle。

    heuristic_upper_bound_cycles：
        可传当前 Prefill heuristic 的 Layer cycles。因为 heuristic 本身是合法调度，
        加上 makespan <= heuristic 可以安全缩小搜索空间，不会排除真正最优解。

    hint_compute_start_times：
        建议传当前 heuristic 的 ScheduledPrefillTask.compute_start_time。
        key = (token_index, expert_id, matrix_name)。

    max_sequence_arcs：
        CP-SAT 为每个 SC 建立“谁紧接谁”的 Circuit。SC 上 n 个任务大约需要 n^2+n 条 arc。
        该阈值用于防止误把超大 Batch 直接丢给 Exact Solver。传 None 可关闭保护。

    target_makespan_cycles：
        可选的硬约束 makespan <= target。
        当 minimize_makespan=False 时，Solver 只回答“target 周期内是否存在合法调度”。
        这用于 certification：已知 heuristic=H 时，直接检查 H-1 是否可行。

    minimize_makespan：
        True：原来的优化模式，最小化 makespan；
        False：纯可行性模式，不设置 objective，只检查 target 是否可行。

    enable_token_copy_symmetry_breaking：
        True（默认）：对“同一 Expert 的多 Token 等价副本”只规范 gate 的标签顺序。
        这是严格安全的 symmetry breaking：同一 Expert 内各 Token 的 gate/up/down 三元组
        在物理位置、计算时间和依赖结构上完全相同，而且不同 Expert 之间没有 Token 级
        前驱依赖，因此可以整体重命名这些副本而不改变任何物理调度。
        这里只约束 gate，避免对 gate/up 同时排序而误删真实调度。
    """
    cp_model = _import_cp_model()

    if rules is None:
        rules = ExecutionRules()
    _validate_baseline_rules(rules)

    if time_limit_seconds <= 0:
        raise PrefillOptimalSolverError("time_limit_seconds 必须大于 0。")
    if num_workers <= 0:
        raise PrefillOptimalSolverError("num_workers 必须大于 0。")
    if max_sequence_arcs is not None and max_sequence_arcs <= 0:
        raise PrefillOptimalSolverError("max_sequence_arcs 必须大于 0 或为 None。")
    if target_makespan_cycles is not None and target_makespan_cycles <= 0:
        raise PrefillOptimalSolverError("target_makespan_cycles 必须大于 0 或为 None。")
    if not minimize_makespan and target_makespan_cycles is None:
        raise PrefillOptimalSolverError(
            "纯可行性模式必须提供 target_makespan_cycles。"
        )

    routes = _normalize_routes(
        index=index,
        layer_id=layer_id,
        routed_experts_by_token=routed_experts_by_token,
    )
    initial_state = _normalize_initial_state(
        index=index,
        initial_active_cube_by_subcube=initial_active_cube_by_subcube,
    )
    hint = _normalize_hint(hint_compute_start_times)

    active_ids_by_token = tuple(
        index.resolve_active_expert_ids(
            layer_id=layer_id,
            routed_expert_ids=route,
        )
        for route in routes
    )

    # --------------------------------------------------------
    # Task metadata
    # --------------------------------------------------------
    task_meta: dict[TaskKey, tuple[int, int]] = {}
    # key -> (subcube_id, cube_id)

    keys_by_sc: list[list[TaskKey]] = [
        [] for _ in range(index.num_subcubes)
    ]

    for token_index, active_ids in enumerate(active_ids_by_token):
        for expert_id in active_ids:
            expert = index.expert(layer_id, expert_id)
            for matrix_name, location in (
                (MATRIX_GATE, expert.gate),
                (MATRIX_UP, expert.up),
                (MATRIX_DOWN, expert.down),
            ):
                key = (token_index, expert_id, matrix_name)
                if key in task_meta:
                    raise PrefillOptimalSolverError(f"重复 task key：{key}。")
                task_meta[key] = (location.subcube_id, location.cube_id)
                keys_by_sc[location.subcube_id].append(key)

    task_count = len(task_meta)
    if task_count <= 0:
        raise PrefillOptimalSolverError("Prefill Layer 没有任务。")

    estimated_sequence_arcs = sum(
        len(keys) * len(keys) + len(keys)
        for keys in keys_by_sc
    )
    if (
        max_sequence_arcs is not None
        and estimated_sequence_arcs > max_sequence_arcs
    ):
        raise PrefillOptimalSolverError(
            "当前 Prefill Layer 的 Exact sequence model 过大："
            f"tasks={task_count}, estimated_arcs={estimated_sequence_arcs}, "
            f"limit={max_sequence_arcs}。"
            "建议先减小 Batch / 只抽样小中型真实 Batch；"
            "如确实要尝试，可显式传 max_sequence_arcs=None。"
        )

    # 一个非常保守但有限的时间上界：
    # 极端情况下把全部任务串行，每个任务前都收一次 switch。
    horizon = task_count * (rules.compute_cycles + rules.switch_cycles)
    horizon += rules.switch_cycles
    horizon = max(horizon, rules.compute_cycles)

    model = cp_model.CpModel()

    start_vars: dict[TaskKey, object] = {}
    end_vars: dict[TaskKey, object] = {}
    interval_vars: dict[TaskKey, object] = {}

    intervals_by_sc: list[list[object]] = [
        [] for _ in range(index.num_subcubes)
    ]

    for key, (sc, _cube_id) in task_meta.items():
        token_index, expert_id, matrix_name = key
        safe_name = matrix_name.replace("_proj", "")
        start = model.new_int_var(
            0,
            horizon,
            f"s_t{token_index}_e{expert_id}_{safe_name}",
        )
        end = model.new_int_var(
            0,
            horizon,
            f"e_t{token_index}_e{expert_id}_{safe_name}",
        )
        interval = model.new_interval_var(
            start,
            rules.compute_cycles,
            end,
            f"i_t{token_index}_e{expert_id}_{safe_name}",
        )
        start_vars[key] = start
        end_vars[key] = end
        interval_vars[key] = interval
        intervals_by_sc[sc].append(interval)

    # --------------------------------------------------------
    # Token 级 gate/up -> down
    # --------------------------------------------------------
    for token_index, active_ids in enumerate(active_ids_by_token):
        for expert_id in active_ids:
            gate_key = (token_index, expert_id, MATRIX_GATE)
            up_key = (token_index, expert_id, MATRIX_UP)
            down_key = (token_index, expert_id, MATRIX_DOWN)
            model.add(start_vars[down_key] >= end_vars[gate_key])
            model.add(start_vars[down_key] >= end_vars[up_key])

    # --------------------------------------------------------
    # Token-copy symmetry breaking（严格安全）
    # --------------------------------------------------------
    #
    # 对固定 Expert-e，若多个 Token 都激活了它，那么这些：
    #
    #   (gate_t,e, up_t,e, down_t,e)
    #
    # 是完全等价的任务副本：三种矩阵落在完全相同的 WC/SC，compute/setup 规则相同，
    # 且项目当前没有“Token 内跨 Expert”的依赖；唯一依赖只是每个副本自己的
    # gate/up -> down。因此任意可行调度都可以把这些副本整体重新编号，得到一个
    # gate 开始顺序按 token_index 递增、但 makespan 完全不变的等价调度。
    #
    # 所以这里只保留这个 canonical representative：
    #
    #   gate(token_a,e) 在 gate(token_b,e) 之前，若 token_a < token_b。
    #
    # 注意：故意不再同时强制 up/down 也按同一 token 顺序，否则可能误删
    # 非等价的真实任务排列。
    # --------------------------------------------------------
    if enable_token_copy_symmetry_breaking:
        tokens_by_expert: dict[int, list[int]] = {}
        for token_index, active_ids in enumerate(active_ids_by_token):
            for expert_id in active_ids:
                tokens_by_expert.setdefault(expert_id, []).append(token_index)

        for expert_id, token_indices in tokens_by_expert.items():
            if len(token_indices) < 2:
                continue

            ordered_tokens = sorted(token_indices)
            for left_token, right_token in zip(ordered_tokens, ordered_tokens[1:]):
                left_gate = (left_token, expert_id, MATRIX_GATE)
                right_gate = (right_token, expert_id, MATRIX_GATE)

                # 两个 gate 在同一个 WC/SC，且 compute duration > 0。
                # 用 end(left) <= start(right) 比单纯 start(left) <= start(right)
                # 传播更强，同时仍只是规范等价副本的标签顺序。
                model.add(
                    start_vars[right_gate] >= end_vars[left_gate]
                )

    # --------------------------------------------------------
    # SC 内 compute 不能重叠。
    # Circuit 负责“紧邻任务 + setup”，NoOverlap 作为额外传播约束。
    # --------------------------------------------------------
    for intervals in intervals_by_sc:
        if len(intervals) >= 2:
            model.add_no_overlap(intervals)

    # --------------------------------------------------------
    # 每个 SC：Hamiltonian path（通过 dummy=0 的 Circuit 表示）
    #
    # 0 -> first -> ... -> last -> 0
    #
    # 如果 i 紧接 j：
    #     start_j >= end_i + setup(cube_i, cube_j)
    #
    # 如果 dummy -> first：
    #     start_first >= initial_setup(initial_active, cube_first)
    # --------------------------------------------------------
    arc_vars_by_sc: list[dict[tuple[int, int], object]] = [
        {} for _ in range(index.num_subcubes)
    ]
    local_key_by_node_by_sc: list[dict[int, TaskKey]] = [
        {} for _ in range(index.num_subcubes)
    ]
    local_node_by_key_by_sc: list[dict[TaskKey, int]] = [
        {} for _ in range(index.num_subcubes)
    ]

    for sc, sc_keys in enumerate(keys_by_sc):
        if not sc_keys:
            continue

        # 保持确定性，只影响变量命名与 hint，不影响最优性。
        ordered_keys = sorted(
            sc_keys,
            key=lambda key: (
                key[0],
                key[1],
                _MATRIX_ORDER[key[2]],
                task_meta[key][1],
            ),
        )

        key_by_node: dict[int, TaskKey] = {}
        node_by_key: dict[TaskKey, int] = {}
        for node_id, key in enumerate(ordered_keys, start=1):
            key_by_node[node_id] = key
            node_by_key[key] = node_id

        local_key_by_node_by_sc[sc] = key_by_node
        local_node_by_key_by_sc[sc] = node_by_key

        arcs: list[tuple[int, int, object]] = []
        arc_vars: dict[tuple[int, int], object] = {}

        # dummy -> first / last -> dummy
        for node_id, key in key_by_node.items():
            token_index, expert_id, matrix_name = key
            safe_name = matrix_name.replace("_proj", "")

            first_lit = model.new_bool_var(
                f"arc_sc{sc}_0_to_t{token_index}_e{expert_id}_{safe_name}"
            )
            last_lit = model.new_bool_var(
                f"arc_sc{sc}_t{token_index}_e{expert_id}_{safe_name}_to_0"
            )
            arcs.append((0, node_id, first_lit))
            arcs.append((node_id, 0, last_lit))
            arc_vars[(0, node_id)] = first_lit
            arc_vars[(node_id, 0)] = last_lit

            _sc, cube_id = task_meta[key]
            initial_setup = _initial_activation_cost(
                initial_cube_id=initial_state[sc],
                next_cube_id=cube_id,
                rules=rules,
                charge_initial_activation=charge_initial_activation,
            )
            model.add(start_vars[key] >= initial_setup).only_enforce_if(first_lit)

        # task i -> task j
        node_ids = tuple(key_by_node)
        for left_id in node_ids:
            left_key = key_by_node[left_id]
            left_cube = task_meta[left_key][1]

            for right_id in node_ids:
                if left_id == right_id:
                    continue

                right_key = key_by_node[right_id]
                right_cube = task_meta[right_key][1]

                lit = model.new_bool_var(
                    f"arc_sc{sc}_{left_id}_to_{right_id}"
                )
                arcs.append((left_id, right_id, lit))
                arc_vars[(left_id, right_id)] = lit

                setup = _transition_cost(
                    previous_cube_id=left_cube,
                    next_cube_id=right_cube,
                    rules=rules,
                )
                model.add(
                    start_vars[right_key] >= end_vars[left_key] + setup
                ).only_enforce_if(lit)

        model.add_circuit(arcs)
        arc_vars_by_sc[sc] = arc_vars

        # ----------------------------------------------------
        # Greedy/heuristic hint：
        # 1) start time hint；
        # 2) 如果该 SC 的所有任务都有 hint，再给 Circuit 顺序 hint。
        # ----------------------------------------------------
        for key in ordered_keys:
            value = hint.get(key)
            if value is not None:
                model.add_hint(start_vars[key], value)

        hinted_keys = [key for key in ordered_keys if key in hint]
        if len(hinted_keys) == len(ordered_keys):
            hinted_order = sorted(
                ordered_keys,
                key=lambda key: (
                    hint[key],
                    key[0],
                    key[1],
                    _MATRIX_ORDER[key[2]],
                ),
            )
            hinted_nodes = [node_by_key[key] for key in hinted_order]
            hinted_edges: set[tuple[int, int]] = set()
            if hinted_nodes:
                hinted_edges.add((0, hinted_nodes[0]))
                for left, right in zip(hinted_nodes, hinted_nodes[1:]):
                    hinted_edges.add((left, right))
                hinted_edges.add((hinted_nodes[-1], 0))

            for edge, lit in arc_vars.items():
                model.add_hint(lit, 1 if edge in hinted_edges else 0)

    # --------------------------------------------------------
    # Makespan
    # --------------------------------------------------------
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, list(end_vars.values()))

    if heuristic_upper_bound_cycles is not None:
        if heuristic_upper_bound_cycles <= 0:
            raise PrefillOptimalSolverError(
                "heuristic_upper_bound_cycles 必须大于 0。"
            )
        model.add(makespan <= heuristic_upper_bound_cycles)

    if target_makespan_cycles is not None:
        model.add(makespan <= target_makespan_cycles)

    if minimize_makespan:
        model.minimize(makespan)

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.log_search_progress = bool(log_search_progress)

    status_code = solver.solve(model)
    status_name = solver.status_name(status_code)
    feasible = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    # 纯 feasibility 模式下，即使 CP-SAT 返回 OPTIMAL，
    # 也只代表“找到了满足 target 的解”，不是全局最优证明。
    proven_optimal = (
        minimize_makespan
        and status_code == cp_model.OPTIMAL
    )

    objective_cycles: int | None = None
    best_bound: float | None = None
    tasks: list[PrefillOptimalTask] = []
    final_state = list(initial_state)

    if feasible:
        # 优化模式读 objective；纯 feasibility 模式直接读取 makespan 变量。
        if minimize_makespan:
            objective_cycles = int(round(solver.objective_value))
            best_bound = float(solver.best_objective_bound)
        else:
            objective_cycles = int(solver.value(makespan))
            best_bound = None

        # 先根据 Circuit 恢复每个 SC 的严格前后顺序。
        predecessor_key: dict[TaskKey, TaskKey | None] = {}

        for sc, key_by_node in enumerate(local_key_by_node_by_sc):
            if not key_by_node:
                continue

            arc_vars = arc_vars_by_sc[sc]
            successor: dict[int, int] = {}
            for (tail, head), lit in arc_vars.items():
                if solver.boolean_value(lit):
                    successor[tail] = head

            if 0 not in successor:
                raise PrefillOptimalSolverError(
                    f"SC-{sc} Circuit 缺少 dummy 起点。"
                )

            current = successor[0]
            previous_key: TaskKey | None = None
            visited: set[int] = set()
            last_key: TaskKey | None = None

            while current != 0:
                if current in visited:
                    raise PrefillOptimalSolverError(
                        f"SC-{sc} Circuit 出现非法子环。"
                    )
                visited.add(current)

                key = key_by_node[current]
                predecessor_key[key] = previous_key
                previous_key = key
                last_key = key

                if current not in successor:
                    raise PrefillOptimalSolverError(
                        f"SC-{sc} Circuit 中间节点缺少 successor。"
                    )
                current = successor[current]

            if len(visited) != len(key_by_node):
                raise PrefillOptimalSolverError(
                    f"SC-{sc} Circuit 未覆盖全部任务。"
                )

            if last_key is not None:
                final_state[sc] = task_meta[last_key][1]

        for key, (sc, cube_id) in task_meta.items():
            token_index, expert_id, matrix_name = key
            compute_start = int(solver.value(start_vars[key]))
            finish = int(solver.value(end_vars[key]))

            previous_key = predecessor_key.get(key)
            if previous_key is None:
                previous_cube_id = initial_state[sc]
                activation = _initial_activation_cost(
                    initial_cube_id=previous_cube_id,
                    next_cube_id=cube_id,
                    rules=rules,
                    charge_initial_activation=charge_initial_activation,
                )
            else:
                previous_cube_id = task_meta[previous_key][1]
                activation = _transition_cost(
                    previous_cube_id=previous_cube_id,
                    next_cube_id=cube_id,
                    rules=rules,
                )

            dispatch = compute_start - activation
            if dispatch < 0:
                raise PrefillOptimalSolverError(
                    "CP-SAT 恢复出的 dispatch_time 为负数。"
                )

            tasks.append(
                PrefillOptimalTask(
                    token_index=token_index,
                    expert_id=expert_id,
                    matrix_name=matrix_name,
                    subcube_id=sc,
                    cube_id=cube_id,
                    dispatch_time=dispatch,
                    compute_start_time=compute_start,
                    finish_time=finish,
                    activation_cycles=activation,
                    previous_active_cube_id=previous_cube_id,
                )
            )
    else:
        if minimize_makespan:
            try:
                best_bound = float(solver.best_objective_bound)
            except Exception:
                best_bound = None
        else:
            best_bound = None

    tasks.sort(key=_task_sort_key)

    result = PrefillOptimalLayerResult(
        layer_id=layer_id,
        token_count=len(routes),
        routed_experts_by_token=routes,
        active_expert_ids_by_token=active_ids_by_token,
        status=status_name,
        proven_optimal=proven_optimal,
        feasible=feasible,
        objective_cycles=objective_cycles,
        best_bound_cycles=best_bound,
        wall_time_seconds=float(solver.wall_time),
        branches=int(solver.num_branches),
        conflicts=int(solver.num_conflicts),
        initial_active_cube_by_subcube=initial_state,
        final_active_cube_by_subcube=tuple(final_state),
        tasks=tuple(tasks),
    )

    if validate_solution and feasible:
        validate_cp_sat_prefill_schedule(
            result=result,
            rules=rules,
            charge_initial_activation=charge_initial_activation,
        )

    return result

def check_prefill_layer_makespan_feasible(
    *,
    index: RuntimeIndex,
    layer_id: int,
    routed_experts_by_token: Iterable[Iterable[int]],
    target_makespan_cycles: int,
    rules: ExecutionRules | None = None,
    initial_active_cube_by_subcube: Iterable[int | None] | None = None,
    charge_initial_activation: bool = True,
    time_limit_seconds: float = 10.0,
    num_workers: int = 8,
    validate_solution: bool = True,
    log_search_progress: bool = False,
    max_sequence_arcs: int | None = 300_000,
    enable_token_copy_symmetry_breaking: bool = True,
) -> PrefillOptimalLayerResult:
    """检查是否存在 makespan <= target 的合法 Prefill Layer 调度。

    返回语义：
    - status == INFEASIBLE：已经严格证明 target 及以下不可能；
    - feasible == True：已经找到一个 <= target 的合法调度；
    - status == UNKNOWN：在时间限制内既没找到解，也没证明无解。

    注意：这是 feasibility/certification 接口，不直接声明全局最优。
    """
    return solve_prefill_layer_optimal(
        index=index,
        layer_id=layer_id,
        routed_experts_by_token=routed_experts_by_token,
        rules=rules,
        initial_active_cube_by_subcube=initial_active_cube_by_subcube,
        charge_initial_activation=charge_initial_activation,
        time_limit_seconds=time_limit_seconds,
        num_workers=num_workers,
        heuristic_upper_bound_cycles=None,
        hint_compute_start_times=None,
        validate_solution=validate_solution,
        log_search_progress=log_search_progress,
        max_sequence_arcs=max_sequence_arcs,
        target_makespan_cycles=target_makespan_cycles,
        minimize_makespan=False,
        enable_token_copy_symmetry_breaking=enable_token_copy_symmetry_breaking,
    )