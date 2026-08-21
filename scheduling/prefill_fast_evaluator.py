"""
高性能 Prefill Evaluator（与 exact Prefill Scheduler 语义一致）。

目的：
    只优化 Python 模拟器本身的运行效率，不修改推理周期定义。

关键优化：
1. 只读取每个 JSON 的 segment0，不再为 Prefill 扫描后续 Decode segments；
2. 不创建每个 Token×Expert×Matrix 的 ScheduledPrefillTask Python 对象；
3. gate/up 全部 ready=0，按当前 exact priority 直接分组计算；
4. down 的 ready_time 在 gate/up 完成后已经确定，因此 16 个 SC 可独立调度；
5. 前 exact_check 个 Batch 与原 schedule_prefill_batch() 做逐层和汇总硬校验。

输出指标与 prefill_evaluator.py 保持同一口径：
    MoE Expert Prefill only，不是完整 TTFT。
"""

from __future__ import annotations

import argparse
import heapq
import multiprocessing as mp
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Iterable, Iterator

from config import ExecutionRules
from mapping.trace_profile import DEFAULT_TRACE_ROOT, discover_trace_files
from scheduling.prefill_evaluator import (
    DEFAULT_OUTPUT_PATH,
    PrefillEvaluationRecord,
    PrefillEvaluationSummary,
    build_summary,
    make_record,
    print_prefill_evaluation_summary,
    save_prefill_evaluation,
)
from scheduling.prefill_scheduler import schedule_prefill_batch
from scheduling.prefill_scheduling_mode import (
    PREFILL_MODE_AGGRESSIVE_REUSE,
    PREFILL_MODE_LARGEST_BATCH_REUSE,
    PREFILL_MODE_NO_REUSE,
    PREFILL_MODE_SWITCH_AWARE,
    PREFILL_SCHEDULING_MODES,
    normalize_prefill_scheduling_mode,
)
from scheduling.prefill_workload import (
    STAGE_PREFILL_CANDIDATE,
    PrefillWorkloadStats,
    TraceSegmentBatch,
    _load_json as _load_prefill_json,
    build_segment_batch,
    iter_prefill_batches,
)
from scheduling.runtime_index import (
    DEFAULT_MAPPING_PATH,
    RuntimeIndex,
    load_runtime_index,
)


class PrefillFastEvaluatorError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FastMatrix:
    subcube_id: int
    cube_id: int


@dataclass(frozen=True, slots=True)
class FastExpert:
    gate: FastMatrix
    up: FastMatrix
    down: FastMatrix


@dataclass(frozen=True, slots=True)
class FastPrefillSubcubeLayerStats:
    subcube_id: int
    task_count: int
    compute_cycles: int
    activation_cycles: int
    switch_count: int
    initial_activation_count: int
    busy_cycles: int
    wait_cycles: int
    max_task_wait_cycles: int
    last_finish_time: int
    initial_active_cube_id: int | None
    final_active_cube_id: int | None


@dataclass(frozen=True, slots=True)
class FastPrefillLayerResult:
    layer_id: int
    token_count: int
    total_cycles: int
    subcube_stats: tuple[FastPrefillSubcubeLayerStats, ...]
    initial_active_cube_by_subcube: tuple[int | None, ...]
    final_active_cube_by_subcube: tuple[int | None, ...]
    max_task_wait_cycles: int

    @property
    def task_count(self) -> int:
        return sum(x.task_count for x in self.subcube_stats)

    @property
    def switch_count(self) -> int:
        return sum(x.switch_count for x in self.subcube_stats)

    @property
    def initial_activation_count(self) -> int:
        return sum(x.initial_activation_count for x in self.subcube_stats)

    @property
    def activation_overhead_cycles(self) -> int:
        return sum(x.activation_cycles for x in self.subcube_stats)

    @property
    def compute_cycles(self) -> int:
        return sum(x.compute_cycles for x in self.subcube_stats)

    @property
    def busy_cycles(self) -> int:
        return sum(x.busy_cycles for x in self.subcube_stats)

    @property
    def wait_cycles(self) -> int:
        return sum(x.wait_cycles for x in self.subcube_stats)


@dataclass(frozen=True, slots=True)
class FastPrefillLayerExecution:
    layer_id: int
    global_start_time: int
    global_finish_time: int
    layer_result: FastPrefillLayerResult

    @property
    def cycles(self) -> int:
        return self.layer_result.total_cycles


@dataclass(frozen=True, slots=True)
class FastPrefillSubcubeBatchStats:
    subcube_id: int
    task_count: int
    compute_cycles: int
    activation_cycles: int
    switch_count: int
    initial_activation_count: int
    busy_cycles: int
    wait_cycles: int
    max_task_wait_cycles: int


@dataclass(frozen=True, slots=True)
class FastPrefillBatchResult:
    token_count: int
    layers: tuple[FastPrefillLayerExecution, ...]
    initial_active_cube_by_subcube: tuple[int | None, ...]
    final_active_cube_by_subcube: tuple[int | None, ...]
    subcube_stats: tuple[FastPrefillSubcubeBatchStats, ...]
    total_cycles: int

    @property
    def total_tasks(self) -> int:
        return sum(x.layer_result.task_count for x in self.layers)

    @property
    def total_switches(self) -> int:
        return sum(x.layer_result.switch_count for x in self.layers)

    @property
    def total_initial_activations(self) -> int:
        return sum(x.layer_result.initial_activation_count for x in self.layers)

    @property
    def total_activation_overhead_cycles(self) -> int:
        return sum(x.layer_result.activation_overhead_cycles for x in self.layers)

    @property
    def total_compute_work_cycles(self) -> int:
        return sum(x.layer_result.compute_cycles for x in self.layers)

    @property
    def total_busy_cycles(self) -> int:
        return sum(x.layer_result.busy_cycles for x in self.layers)

    @property
    def total_wait_cycles(self) -> int:
        return sum(x.layer_result.wait_cycles for x in self.layers)

    @property
    def max_task_wait_cycles(self) -> int:
        return max(
            (x.layer_result.max_task_wait_cycles for x in self.layers),
            default=0,
        )

    @property
    def cycles_per_input_token(self) -> float:
        return self.total_cycles / self.token_count if self.token_count else 0.0

    @property
    def input_tokens_per_cycle(self) -> float:
        return self.token_count / self.total_cycles if self.total_cycles else 0.0


def build_fast_tables(index: RuntimeIndex) -> tuple[tuple[FastExpert, ...], ...]:
    layers: list[tuple[FastExpert, ...]] = []
    for layer_id in range(index.num_layers):
        experts: list[FastExpert] = []
        layer = index.layers[layer_id]
        for expert in layer.experts:
            experts.append(
                FastExpert(
                    gate=FastMatrix(expert.gate.subcube_id, expert.gate.cube_id),
                    up=FastMatrix(expert.up.subcube_id, expert.up.cube_id),
                    down=FastMatrix(expert.down.subcube_id, expert.down.cube_id),
                )
            )
        layers.append(tuple(experts))
    return tuple(layers)


def _activation_cost(
    previous_cube_id: int | None,
    next_cube_id: int,
    *,
    switch_cycles: int,
    charge_initial_activation: bool,
) -> tuple[int, bool, bool]:
    if previous_cube_id is None:
        if charge_initial_activation:
            return switch_cycles, False, True
        return 0, False, False
    if previous_cube_id == next_cube_id:
        return 0, False, False
    return switch_cycles, True, False


class FastPrefillScheduler:
    """Compact Prefill scheduler；正式 evaluator 可跳过重复 route validation。"""

    def __init__(
        self,
        *,
        index: RuntimeIndex,
        rules: ExecutionRules,
        scheduling_mode: str = PREFILL_MODE_SWITCH_AWARE,
    ) -> None:
        if (
            rules.cross_subcube_cycles != 0
            or not rules.unlimited_parallel_subcubes
            or not rules.one_active_weight_cube_per_subcube
            or rules.compute_cycles <= 0
            or rules.switch_cycles < 0
        ):
            raise PrefillFastEvaluatorError("当前 Fast Prefill 只支持本项目既定 Baseline 规则。")

        self.index = index
        self.rules = rules
        self.scheduling_mode = normalize_prefill_scheduling_mode(scheduling_mode)
        self.tables = build_fast_tables(index)
        self.shared_expert_id = index.shared_expert_id
        self.num_subcubes = index.num_subcubes

    def _schedule_layer_largest_batch_event(
        self,
        *,
        layer_id: int,
        routed_experts_by_token: tuple[tuple[int, ...], ...],
        initial_state: tuple[int | None, ...],
        charge_initial_activation: bool,
    ) -> FastPrefillLayerResult:
        """Largest-Batch 的紧凑事件式实现。

        这个模式不能使用普通 Fast Prefill 的
        "先 gate/up，后 down" 两阶段重排，因为 down 一旦在运行中 Ready，
        就必须立刻参加当前 SC 的 Ready-WC batch-size 竞争。

        这里仍然不创建 ScheduledPrefillTask 对象，但保持与 Exact 相同的：
        - 全局完成事件顺序；
        - down 动态创建时机；
        - active WC 复用；
        - Largest-Batch Ready 数量选择；
        - wait / switch / final state 统计。
        """

        layer_table = self.tables[layer_id]
        token_count = len(routed_experts_by_token)
        shared = self.shared_expert_id
        nsc = self.num_subcubes
        compute_cost = self.rules.compute_cycles
        switch_cost = self.rules.switch_cycles

        active_cube = list(initial_state)
        running = [False] * nsc
        last_finish = [0] * nsc

        task_count = [0] * nsc
        compute_cycles = [0] * nsc
        activation_cycles = [0] * nsc
        switch_count = [0] * nsc
        initial_count = [0] * nsc
        wait_cycles = [0] * nsc
        max_wait_by_sc = [0] * nsc
        max_wait = 0

        # entry = (ready_time, route_rank, token_index, matrix_code, cube_id, expert_id)
        # matrix_code: gate=0, up=1, down=2
        ready_by_sc: list[dict[int, list[tuple[int, int, int, int, int, int]]]] = [
            {} for _ in range(nsc)
        ]

        active_ids_by_token: list[tuple[int, ...]] = []
        active_count = len(routed_experts_by_token[0]) + (1 if shared is not None else 0)
        gate_finish = [[-1] * active_count for _ in range(token_count)]
        up_finish = [[-1] * active_count for _ in range(token_count)]
        down_created = [[False] * active_count for _ in range(token_count)]

        def push_ready(
            *,
            subcube_id: int,
            ready_time: int,
            route_rank: int,
            token_index: int,
            matrix_code: int,
            cube_id: int,
            expert_id: int,
        ) -> None:
            heap = ready_by_sc[subcube_id].setdefault(cube_id, [])
            heapq.heappush(
                heap,
                (
                    ready_time,
                    route_rank,
                    token_index,
                    matrix_code,
                    cube_id,
                    expert_id,
                ),
            )

        # t=0: 所有 gate / up Ready。
        for token_index, route in enumerate(routed_experts_by_token):
            ids = route if shared is None else route + (shared,)
            active_ids_by_token.append(ids)
            for rank, expert_id in enumerate(ids):
                expert = layer_table[expert_id]
                push_ready(
                    subcube_id=expert.gate.subcube_id,
                    ready_time=0,
                    route_rank=rank,
                    token_index=token_index,
                    matrix_code=0,
                    cube_id=expert.gate.cube_id,
                    expert_id=expert_id,
                )
                push_ready(
                    subcube_id=expert.up.subcube_id,
                    ready_time=0,
                    route_rank=rank,
                    token_index=token_index,
                    matrix_code=1,
                    cube_id=expert.up.cube_id,
                    expert_id=expert_id,
                )

        expected_task_count = token_count * active_count * 3
        completed = 0
        current_time = 0
        serial = 0

        # event = (finish_time, serial, sc, entry)
        running_heap: list[
            tuple[int, int, int, tuple[int, int, int, int, int, int]]
        ] = []

        def pop_next_for_sc(sc: int):
            groups = ready_by_sc[sc]
            if not groups:
                return None

            current_cube = active_cube[sc]
            if current_cube is not None:
                heap = groups.get(current_cube)
                if heap:
                    entry = heapq.heappop(heap)
                    if not heap:
                        del groups[current_cube]
                    return entry

            # 当前 active WC 没有 Ready Task：
            # 选择当前 Ready Task 数最多的 WC；数量相同按该 WC 的
            # 最早任务 base priority，再按 cube_id 确定性 tie-break。
            target_cube = min(
                groups,
                key=lambda cube_id: (
                    -len(groups[cube_id]),
                    groups[cube_id][0][0],  # ready_time
                    groups[cube_id][0][1],  # route_rank
                    groups[cube_id][0][2],  # token_index
                    groups[cube_id][0][3],  # matrix_priority/code
                    cube_id,
                ),
            )
            heap = groups[target_cube]
            entry = heapq.heappop(heap)
            if not heap:
                del groups[target_cube]
            return entry

        def dispatch_idle_scs() -> int:
            nonlocal serial, max_wait
            dispatched = 0
            for sc in range(nsc):
                if running[sc]:
                    continue
                entry = pop_next_for_sc(sc)
                if entry is None:
                    continue

                ready, _rank, _token, _matrix, cube_id, _expert = entry
                if ready > current_time:
                    raise PrefillFastEvaluatorError(
                        "Largest-Batch Fast 发现未来任务被提前放入 Ready 集合。"
                    )

                activation, is_switch, is_initial = _activation_cost(
                    active_cube[sc],
                    cube_id,
                    switch_cycles=switch_cost,
                    charge_initial_activation=charge_initial_activation,
                )
                wait = current_time - ready
                wait_cycles[sc] += wait
                if wait > max_wait_by_sc[sc]:
                    max_wait_by_sc[sc] = wait
                if wait > max_wait:
                    max_wait = wait

                activation_cycles[sc] += activation
                switch_count[sc] += int(is_switch)
                initial_count[sc] += int(is_initial)
                task_count[sc] += 1
                compute_cycles[sc] += compute_cost

                finish = current_time + activation + compute_cost
                active_cube[sc] = cube_id
                running[sc] = True
                heapq.heappush(running_heap, (finish, serial, sc, entry))
                serial += 1
                dispatched += 1
            return dispatched

        while completed < expected_task_count:
            dispatch_idle_scs()

            if not running_heap:
                raise PrefillFastEvaluatorError(
                    "Largest-Batch Fast 调度死锁：没有运行任务，也没有可执行任务。"
                )

            next_finish = running_heap[0][0]
            if next_finish < current_time:
                raise PrefillFastEvaluatorError(
                    "Largest-Batch Fast 内部时间状态错误。"
                )
            current_time = next_finish

            finished_now: list[
                tuple[int, tuple[int, int, int, int, int, int]]
            ] = []
            while running_heap and running_heap[0][0] == current_time:
                _finish, _serial, sc, entry = heapq.heappop(running_heap)
                if not running[sc]:
                    raise PrefillFastEvaluatorError(
                        "Largest-Batch Fast running state 与事件堆不一致。"
                    )
                running[sc] = False
                last_finish[sc] = current_time
                completed += 1
                finished_now.append((sc, entry))

            # Exact 会在同一时刻所有完成事件处理完之后，
            # 再统一创建新 Ready 的 down。
            affected: set[tuple[int, int, int]] = set()
            # (token, rank, expert_id)
            for _sc, entry in finished_now:
                _ready, rank, token_index, matrix_code, _cube, expert_id = entry
                if matrix_code == 0:
                    gate_finish[token_index][rank] = current_time
                    affected.add((token_index, rank, expert_id))
                elif matrix_code == 1:
                    up_finish[token_index][rank] = current_time
                    affected.add((token_index, rank, expert_id))

            for token_index, rank, expert_id in sorted(
                affected,
                key=lambda item: (item[1], item[0], item[2]),
            ):
                if down_created[token_index][rank]:
                    continue
                gate_done = gate_finish[token_index][rank]
                up_done = up_finish[token_index][rank]
                if gate_done < 0 or up_done < 0:
                    continue

                ready_time = max(gate_done, up_done)
                down = layer_table[expert_id].down
                push_ready(
                    subcube_id=down.subcube_id,
                    ready_time=ready_time,
                    route_rank=rank,
                    token_index=token_index,
                    matrix_code=2,
                    cube_id=down.cube_id,
                    expert_id=expert_id,
                )
                down_created[token_index][rank] = True

        if completed != expected_task_count:
            raise PrefillFastEvaluatorError(
                "Largest-Batch Fast 完成任务数错误。"
            )

        total_cycles = max(last_finish, default=0)
        subcube_stats = tuple(
            FastPrefillSubcubeLayerStats(
                subcube_id=sc,
                task_count=task_count[sc],
                compute_cycles=compute_cycles[sc],
                activation_cycles=activation_cycles[sc],
                switch_count=switch_count[sc],
                initial_activation_count=initial_count[sc],
                busy_cycles=compute_cycles[sc] + activation_cycles[sc],
                wait_cycles=wait_cycles[sc],
                max_task_wait_cycles=max_wait_by_sc[sc],
                last_finish_time=last_finish[sc],
                initial_active_cube_id=initial_state[sc],
                final_active_cube_id=active_cube[sc],
            )
            for sc in range(nsc)
        )

        return FastPrefillLayerResult(
            layer_id=layer_id,
            token_count=token_count,
            total_cycles=total_cycles,
            subcube_stats=subcube_stats,
            initial_active_cube_by_subcube=initial_state,
            final_active_cube_by_subcube=tuple(active_cube),
            max_task_wait_cycles=max_wait,
        )

    def schedule_layer(
        self,
        *,
        layer_id: int,
        routed_experts_by_token: tuple[tuple[int, ...], ...],
        initial_active_cube_by_subcube: tuple[int | None, ...] | None,
        charge_initial_activation: bool,
        validate_routes: bool = False,
    ) -> FastPrefillLayerResult:
        if not routed_experts_by_token:
            raise PrefillFastEvaluatorError("Prefill Layer 至少需要一个 Token。")

        if initial_active_cube_by_subcube is None:
            initial_state = (None,) * self.num_subcubes
        else:
            initial_state = tuple(initial_active_cube_by_subcube)
            if len(initial_state) != self.num_subcubes:
                raise PrefillFastEvaluatorError("initial state 长度错误。")

        if validate_routes:
            for route in routed_experts_by_token:
                self.index.resolve_active_expert_ids(
                    layer_id=layer_id,
                    routed_expert_ids=route,
                )

        if self.scheduling_mode == PREFILL_MODE_LARGEST_BATCH_REUSE:
            return self._schedule_layer_largest_batch_event(
                layer_id=layer_id,
                routed_experts_by_token=routed_experts_by_token,
                initial_state=initial_state,
                charge_initial_activation=charge_initial_activation,
            )

        layer_table = self.tables[layer_id]
        active_cube = list(initial_state)
        token_count = len(routed_experts_by_token)
        shared = self.shared_expert_id
        compute_cost = self.rules.compute_cycles
        switch_cost = self.rules.switch_cycles
        nsc = self.num_subcubes

        # 每个 Token 对应 8 routed + shared，rank 直接就是 exact route_rank。
        active_ids_by_token: list[tuple[int, ...]] = []
        active_count = len(routed_experts_by_token[0]) + (1 if shared is not None else 0)
        gate_finish = [[-1] * active_count for _ in range(token_count)]
        up_finish = [[-1] * active_count for _ in range(token_count)]

        # pre_groups[sc][cube] = [(rank, token, matrix_code), ...]
        # matrix_code: gate=0, up=1
        pre_groups: list[dict[int, list[tuple[int, int, int]]]] = [
            {} for _ in range(nsc)
        ]

        for token_index, route in enumerate(routed_experts_by_token):
            ids = route if shared is None else route + (shared,)
            active_ids_by_token.append(ids)

            for rank, expert_id in enumerate(ids):
                expert = layer_table[expert_id]
                for matrix_code, matrix in ((0, expert.gate), (1, expert.up)):
                    groups = pre_groups[matrix.subcube_id]
                    groups.setdefault(matrix.cube_id, []).append(
                        (rank, token_index, matrix_code)
                    )

        task_count = [0] * nsc
        compute_cycles = [0] * nsc
        activation_cycles = [0] * nsc
        switch_count = [0] * nsc
        initial_count = [0] * nsc
        wait_cycles = [0] * nsc
        max_wait_by_sc = [0] * nsc
        last_finish = [0] * nsc
        max_wait = 0

        # ----------------------------------------------------
        # Pre stage：所有 gate/up ready_time 都是 0。
        # no_reuse：按原始确定性顺序逐任务执行；
        # 其余两种模式会把同一 active cube 的 Ready Task 连续执行。
        # ----------------------------------------------------
        for sc in range(nsc):
            groups = pre_groups[sc]
            if not groups:
                continue

            for items in groups.values():
                items.sort()

            current_time = 0

            if self.scheduling_mode == PREFILL_MODE_NO_REUSE:
                flat_tasks: list[tuple[int, int, int, int]] = []
                for cube_id, items in groups.items():
                    for rank, token_index, matrix_code in items:
                        flat_tasks.append((rank, token_index, matrix_code, cube_id))
                flat_tasks.sort()

                for rank, token_index, matrix_code, cube_id in flat_tasks:
                    activation, is_switch, is_initial = _activation_cost(
                        active_cube[sc], cube_id,
                        switch_cycles=switch_cost,
                        charge_initial_activation=charge_initial_activation,
                    )
                    switch_count[sc] += int(is_switch)
                    initial_count[sc] += int(is_initial)
                    wait = current_time
                    wait_cycles[sc] += wait
                    max_wait_by_sc[sc] = max(max_wait_by_sc[sc], wait)
                    max_wait = max(max_wait, wait)
                    activation_cycles[sc] += activation
                    current_time += activation + compute_cost
                    task_count[sc] += 1
                    compute_cycles[sc] += compute_cost
                    active_cube[sc] = cube_id
                    if matrix_code == 0:
                        gate_finish[token_index][rank] = current_time
                    else:
                        up_finish[token_index][rank] = current_time
            else:
                initial_cube = active_cube[sc]
                order: list[int] = []
                if initial_cube in groups:
                    order.append(initial_cube)  # type: ignore[arg-type]
                remaining = [cube for cube in groups if cube != initial_cube]

                if self.scheduling_mode == PREFILL_MODE_LARGEST_BATCH_REUSE:
                    # Pre 阶段所有 gate/up 都在 t=0 Ready。
                    # 因此当前 WC 耗尽后，Exact 会选择 Ready Task 数最多的 WC，
                    # 并把该 WC 连续执行完。
                    remaining.sort(
                        key=lambda cube: (
                            -len(groups[cube]),
                            groups[cube][0][0],
                            groups[cube][0][1],
                            groups[cube][0][2],
                            cube,
                        )
                    )
                else:
                    remaining.sort(
                        key=lambda cube: (
                            groups[cube][0][0],
                            groups[cube][0][1],
                            groups[cube][0][2],
                            cube,
                        )
                    )

                order.extend(remaining)

                for cube_id in order:
                    items = groups[cube_id]
                    activation, is_switch, is_initial = _activation_cost(
                        active_cube[sc], cube_id,
                        switch_cycles=switch_cost,
                        charge_initial_activation=charge_initial_activation,
                    )
                    switch_count[sc] += int(is_switch)
                    initial_count[sc] += int(is_initial)
                    for item_index, (rank, token_index, matrix_code) in enumerate(items):
                        wait = current_time
                        wait_cycles[sc] += wait
                        max_wait_by_sc[sc] = max(max_wait_by_sc[sc], wait)
                        max_wait = max(max_wait, wait)
                        task_activation = activation if item_index == 0 else 0
                        activation_cycles[sc] += task_activation
                        current_time += task_activation + compute_cost
                        task_count[sc] += 1
                        compute_cycles[sc] += compute_cost
                        active_cube[sc] = cube_id
                        if matrix_code == 0:
                            gate_finish[token_index][rank] = current_time
                        else:
                            up_finish[token_index][rank] = current_time

            last_finish[sc] = current_time

        # ----------------------------------------------------
        # Down：ready_time 已全部确定。不同 SC 不再互相产生依赖，
        # 所以可以分别独立调度，不需要全局 event heap。
        # ----------------------------------------------------
        down_by_sc: list[list[tuple[int, int, int, int]]] = [
            [] for _ in range(nsc)
        ]
        for token_index, ids in enumerate(active_ids_by_token):
            for rank, expert_id in enumerate(ids):
                ready_time = max(
                    gate_finish[token_index][rank],
                    up_finish[token_index][rank],
                )
                down = layer_table[expert_id].down
                down_by_sc[down.subcube_id].append(
                    (ready_time, rank, token_index, down.cube_id)
                )

        for sc, down_tasks in enumerate(down_by_sc):
            if not down_tasks:
                continue

            # ------------------------------------------------
            # Largest-Batch Reuse：事件式 Ready 集合。
            #
            # Fast 版不能把未来 down 提前计入 batch size。
            # 因此按 ready_time 把任务逐步移入 Ready 集合；
            # 当前 WC 有 Ready Task 就继续，否则选择 Ready 数最多的 WC。
            # ------------------------------------------------
            if self.scheduling_mode == PREFILL_MODE_LARGEST_BATCH_REUSE:
                # future entry: (ready, rank, token, cube, serial)
                future = [
                    (ready, rank, token_index, cube_id, task_id)
                    for task_id, (ready, rank, token_index, cube_id)
                    in enumerate(down_tasks)
                ]
                heapq.heapify(future)

                # 每个 cube 当前已经 Ready 的任务最小堆。
                ready_heaps: dict[int, list[tuple[int, int, int, int, int]]] = {}
                ready_count: dict[int, int] = {}
                total_ready = 0

                # max-ready heap 使用 lazy version：
                # (-ready_count, best_ready, best_rank, best_token, cube, version)
                batch_heap: list[tuple[int, int, int, int, int, int]] = []
                version: dict[int, int] = {}

                def push_cube_state(cube_id: int) -> None:
                    heap = ready_heaps.get(cube_id)
                    count = ready_count.get(cube_id, 0)
                    version[cube_id] = version.get(cube_id, 0) + 1
                    if not heap or count <= 0:
                        return
                    best = heap[0]
                    heapq.heappush(
                        batch_heap,
                        (
                            -count,
                            best[0],
                            best[1],
                            best[2],
                            cube_id,
                            version[cube_id],
                        ),
                    )

                def release_ready(now: int) -> None:
                    nonlocal total_ready
                    touched: set[int] = set()
                    while future and future[0][0] <= now:
                        entry = heapq.heappop(future)
                        cube_id = entry[3]
                        heapq.heappush(ready_heaps.setdefault(cube_id, []), entry)
                        ready_count[cube_id] = ready_count.get(cube_id, 0) + 1
                        total_ready += 1
                        touched.add(cube_id)
                    for cube_id in touched:
                        push_cube_state(cube_id)

                def pop_from_cube(cube_id: int):
                    nonlocal total_ready
                    heap = ready_heaps[cube_id]
                    entry = heapq.heappop(heap)
                    ready_count[cube_id] -= 1
                    total_ready -= 1
                    push_cube_state(cube_id)
                    return entry

                def pop_largest_cube() -> int:
                    while batch_heap:
                        (
                            neg_count,
                            best_ready,
                            best_rank,
                            best_token,
                            cube_id,
                            entry_version,
                        ) = heapq.heappop(batch_heap)

                        if entry_version != version.get(cube_id):
                            continue
                        heap = ready_heaps.get(cube_id)
                        count = ready_count.get(cube_id, 0)
                        if not heap or count <= 0:
                            continue
                        best = heap[0]
                        current_key = (
                            -count,
                            best[0],
                            best[1],
                            best[2],
                            cube_id,
                            entry_version,
                        )
                        if current_key != (
                            neg_count,
                            best_ready,
                            best_rank,
                            best_token,
                            cube_id,
                            entry_version,
                        ):
                            # 理论上 version 已经足以排除陈旧项；保守重推。
                            push_cube_state(cube_id)
                            continue
                        return cube_id
                    raise PrefillFastEvaluatorError(
                        "Largest-Batch Ready heap 意外为空。"
                    )

                remaining = len(down_tasks)
                current_time = last_finish[sc]

                while remaining:
                    release_ready(current_time)

                    if total_ready <= 0:
                        if not future:
                            raise PrefillFastEvaluatorError(
                                "Largest-Batch Down 调度死锁。"
                            )
                        current_time = max(current_time, future[0][0])
                        release_ready(current_time)

                    current_cube = active_cube[sc]
                    if (
                        current_cube is not None
                        and ready_count.get(current_cube, 0) > 0
                    ):
                        target_cube = current_cube
                    else:
                        target_cube = pop_largest_cube()

                    ready, _rank, _token_index, cube_id, _task_id = pop_from_cube(
                        target_cube
                    )
                    remaining -= 1

                    dispatch_time = current_time
                    wait = dispatch_time - ready
                    wait_cycles[sc] += wait
                    if wait > max_wait_by_sc[sc]:
                        max_wait_by_sc[sc] = wait
                    if wait > max_wait:
                        max_wait = wait

                    activation, is_switch, is_initial = _activation_cost(
                        active_cube[sc],
                        cube_id,
                        switch_cycles=switch_cost,
                        charge_initial_activation=charge_initial_activation,
                    )
                    activation_cycles[sc] += activation
                    if is_switch:
                        switch_count[sc] += 1
                    if is_initial:
                        initial_count[sc] += 1

                    current_time += activation + compute_cost
                    task_count[sc] += 1
                    compute_cycles[sc] += compute_cost
                    active_cube[sc] = cube_id

                last_finish[sc] = current_time
                continue

            # ------------------------------------------------
            # 原三种策略：保留已经验证过的 Fast 路径。
            # ------------------------------------------------
            global_heap: list[tuple[int, int, int, int, int]] = []
            cube_heaps: dict[int, list[tuple[int, int, int, int, int]]] = {}
            alive = [True] * len(down_tasks)

            for task_id, (ready, rank, token_index, cube_id) in enumerate(down_tasks):
                entry = (ready, rank, token_index, cube_id, task_id)
                heapq.heappush(global_heap, entry)
                heapq.heappush(cube_heaps.setdefault(cube_id, []), entry)

            remaining = len(down_tasks)
            current_time = last_finish[sc]

            def clean(heap: list[tuple[int, int, int, int, int]]) -> None:
                while heap and not alive[heap[0][4]]:
                    heapq.heappop(heap)

            while remaining:
                clean(global_heap)
                min_ready = global_heap[0][0]
                if current_time < min_ready:
                    current_time = min_ready

                selected: tuple[int, int, int, int, int] | None = None
                current_cube = active_cube[sc]

                if self.scheduling_mode == PREFILL_MODE_NO_REUSE:
                    clean(global_heap)
                    selected = heapq.heappop(global_heap)
                elif self.scheduling_mode == PREFILL_MODE_SWITCH_AWARE:
                    if current_cube is not None:
                        same_heap = cube_heaps.get(current_cube)
                        if same_heap:
                            clean(same_heap)
                            if same_heap and same_heap[0][0] == min_ready:
                                selected = heapq.heappop(same_heap)
                    if selected is None:
                        clean(global_heap)
                        selected = heapq.heappop(global_heap)
                else:
                    if current_cube is not None:
                        same_heap = cube_heaps.get(current_cube)
                        if same_heap:
                            clean(same_heap)
                            if same_heap and same_heap[0][0] <= current_time:
                                selected = heapq.heappop(same_heap)
                    if selected is None:
                        clean(global_heap)
                        selected = heapq.heappop(global_heap)

                ready, _rank, _token_index, cube_id, task_id = selected
                if not alive[task_id]:
                    raise PrefillFastEvaluatorError("Fast Prefill lazy heap 状态错误。")
                alive[task_id] = False
                remaining -= 1

                dispatch_time = current_time
                wait = dispatch_time - ready
                wait_cycles[sc] += wait
                if wait > max_wait_by_sc[sc]:
                    max_wait_by_sc[sc] = wait
                if wait > max_wait:
                    max_wait = wait

                activation, is_switch, is_initial = _activation_cost(
                    active_cube[sc],
                    cube_id,
                    switch_cycles=switch_cost,
                    charge_initial_activation=charge_initial_activation,
                )
                activation_cycles[sc] += activation
                if is_switch:
                    switch_count[sc] += 1
                if is_initial:
                    initial_count[sc] += 1

                current_time += activation + compute_cost
                task_count[sc] += 1
                compute_cycles[sc] += compute_cost
                active_cube[sc] = cube_id

            last_finish[sc] = current_time

        total_cycles = max(last_finish, default=0)

        subcube_stats = tuple(
            FastPrefillSubcubeLayerStats(
                subcube_id=sc,
                task_count=task_count[sc],
                compute_cycles=compute_cycles[sc],
                activation_cycles=activation_cycles[sc],
                switch_count=switch_count[sc],
                initial_activation_count=initial_count[sc],
                busy_cycles=compute_cycles[sc] + activation_cycles[sc],
                wait_cycles=wait_cycles[sc],
                max_task_wait_cycles=max_wait_by_sc[sc],
                last_finish_time=last_finish[sc],
                initial_active_cube_id=initial_state[sc],
                final_active_cube_id=active_cube[sc],
            )
            for sc in range(nsc)
        )

        return FastPrefillLayerResult(
            layer_id=layer_id,
            token_count=token_count,
            total_cycles=total_cycles,
            subcube_stats=subcube_stats,
            initial_active_cube_by_subcube=initial_state,
            final_active_cube_by_subcube=tuple(active_cube),
            max_task_wait_cycles=max_wait,
        )

    def schedule_batch(
        self,
        routed_experts_by_token: Iterable[Iterable[Iterable[int]]],
        *,
        initial_active_cube_by_subcube: tuple[int | None, ...] | None = None,
        charge_initial_activation: bool = True,
        validate_routes: bool = False,
    ) -> FastPrefillBatchResult:
        if isinstance(routed_experts_by_token, tuple) and all(
            isinstance(token_routes, tuple)
            for token_routes in routed_experts_by_token
        ):
            routes = routed_experts_by_token  # type: ignore[assignment]
        else:
            routes = tuple(
                tuple(tuple(route) for route in token_routes)
                for token_routes in routed_experts_by_token
            )

        if not routes:
            raise PrefillFastEvaluatorError("Prefill Batch 至少需要一个 Token。")

        for token_index, token_routes in enumerate(routes):
            if len(token_routes) != self.index.num_layers:
                raise PrefillFastEvaluatorError(
                    f"Token-{token_index} Layer 数错误："
                    f"actual={len(token_routes)}, expected={self.index.num_layers}。"
                )

        initial_state = (
            (None,) * self.num_subcubes
            if initial_active_cube_by_subcube is None
            else tuple(initial_active_cube_by_subcube)
        )
        state = initial_state
        global_time = 0
        executions: list[FastPrefillLayerExecution] = []

        # aggregate by SC
        sc_task = [0] * self.num_subcubes
        sc_compute = [0] * self.num_subcubes
        sc_activation = [0] * self.num_subcubes
        sc_switch = [0] * self.num_subcubes
        sc_initial = [0] * self.num_subcubes
        sc_busy = [0] * self.num_subcubes
        sc_wait = [0] * self.num_subcubes
        sc_max_wait = [0] * self.num_subcubes

        for layer_id in range(self.index.num_layers):
            layer_routes = tuple(
                token_routes[layer_id]
                for token_routes in routes
            )
            layer_result = self.schedule_layer(
                layer_id=layer_id,
                routed_experts_by_token=layer_routes,
                initial_active_cube_by_subcube=state,
                charge_initial_activation=charge_initial_activation,
                validate_routes=validate_routes,
            )

            start = global_time
            global_time += layer_result.total_cycles
            executions.append(
                FastPrefillLayerExecution(
                    layer_id=layer_id,
                    global_start_time=start,
                    global_finish_time=global_time,
                    layer_result=layer_result,
                )
            )
            state = layer_result.final_active_cube_by_subcube

            for stat in layer_result.subcube_stats:
                sc = stat.subcube_id
                sc_task[sc] += stat.task_count
                sc_compute[sc] += stat.compute_cycles
                sc_activation[sc] += stat.activation_cycles
                sc_switch[sc] += stat.switch_count
                sc_initial[sc] += stat.initial_activation_count
                sc_busy[sc] += stat.busy_cycles
                sc_wait[sc] += stat.wait_cycles
                if stat.max_task_wait_cycles > sc_max_wait[sc]:
                    sc_max_wait[sc] = stat.max_task_wait_cycles

        batch_sc_stats = tuple(
            FastPrefillSubcubeBatchStats(
                subcube_id=sc,
                task_count=sc_task[sc],
                compute_cycles=sc_compute[sc],
                activation_cycles=sc_activation[sc],
                switch_count=sc_switch[sc],
                initial_activation_count=sc_initial[sc],
                busy_cycles=sc_busy[sc],
                wait_cycles=sc_wait[sc],
                max_task_wait_cycles=sc_max_wait[sc],
            )
            for sc in range(self.num_subcubes)
        )

        return FastPrefillBatchResult(
            token_count=len(routes),
            layers=tuple(executions),
            initial_active_cube_by_subcube=initial_state,
            final_active_cube_by_subcube=state,
            subcube_stats=batch_sc_stats,
            total_cycles=global_time,
        )



# ============================================================
# 多进程 Fast Prefill
# ============================================================


@dataclass(frozen=True, slots=True)
class _PrefillParallelChunkResult:
    records: tuple[PrefillEvaluationRecord, ...]
    layer_cycles: tuple[tuple[int, ...], ...]
    layer_switches: tuple[tuple[int, ...], ...]
    layer_waits: tuple[tuple[int, ...], ...]
    sc_task_count: tuple[int, ...]
    sc_busy_cycles: tuple[int, ...]
    sc_switch_count: tuple[int, ...]
    sc_initial_count: tuple[int, ...]
    sc_wait_cycles: tuple[int, ...]
    sc_critical_layer_count: tuple[int, ...]
    total_cycles: int


_PREFILL_WORKER_SCHEDULER: FastPrefillScheduler | None = None
_PREFILL_WORKER_CHARGE_INITIAL = True


def _resolve_worker_count(workers: int) -> int:
    if workers < 0:
        raise PrefillFastEvaluatorError("workers 不能小于 0。")
    if workers == 0:
        cpu = os.cpu_count() or 1
        if cpu <= 2:
            return 1
        return max(1, min(8, cpu - 1))
    return workers


def _iter_prefill_chunks(
    items: Iterable[TraceSegmentBatch],
    chunk_size: int,
) -> Iterator[tuple[TraceSegmentBatch, ...]]:
    if chunk_size <= 0:
        raise PrefillFastEvaluatorError("parallel_chunk_size 必须大于 0。")

    chunk: list[TraceSegmentBatch] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield tuple(chunk)
            chunk.clear()
    if chunk:
        yield tuple(chunk)


def _init_prefill_worker(
    index: RuntimeIndex,
    rules: ExecutionRules,
    charge_initial_activation: bool,
    scheduling_mode: str,
) -> None:
    global _PREFILL_WORKER_SCHEDULER, _PREFILL_WORKER_CHARGE_INITIAL
    _PREFILL_WORKER_SCHEDULER = FastPrefillScheduler(
        index=index, rules=rules, scheduling_mode=scheduling_mode
    )
    _PREFILL_WORKER_CHARGE_INITIAL = charge_initial_activation


def _prefill_worker_chunk(
    batches: tuple[TraceSegmentBatch, ...],
) -> _PrefillParallelChunkResult:
    scheduler = _PREFILL_WORKER_SCHEDULER
    if scheduler is None:
        raise PrefillFastEvaluatorError("Prefill worker 尚未初始化。")

    nl = scheduler.index.num_layers
    ns = scheduler.index.num_subcubes

    records: list[PrefillEvaluationRecord] = []
    layer_cycles: list[list[int]] = [[] for _ in range(nl)]
    layer_switches: list[list[int]] = [[] for _ in range(nl)]
    layer_waits: list[list[int]] = [[] for _ in range(nl)]

    sc_task_count = [0] * ns
    sc_busy_cycles = [0] * ns
    sc_switch_count = [0] * ns
    sc_initial_count = [0] * ns
    sc_wait_cycles = [0] * ns
    sc_critical_layer_count = [0] * ns
    total_cycles = 0

    for batch in batches:
        result = scheduler.schedule_batch(
            batch.routed_experts_by_token,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=_PREFILL_WORKER_CHARGE_INITIAL,
            validate_routes=False,
        )

        records.append(make_record(batch=batch, result=result))  # type: ignore[arg-type]
        total_cycles += result.total_cycles

        for execution in result.layers:
            layer_id = execution.layer_id
            layer_result = execution.layer_result
            layer_cycles[layer_id].append(layer_result.total_cycles)
            layer_switches[layer_id].append(layer_result.switch_count)
            layer_waits[layer_id].append(layer_result.wait_cycles)

            for stat in layer_result.subcube_stats:
                if stat.task_count > 0 and stat.last_finish_time == layer_result.total_cycles:
                    sc_critical_layer_count[stat.subcube_id] += 1

        for stat in result.subcube_stats:
            sc = stat.subcube_id
            sc_task_count[sc] += stat.task_count
            sc_busy_cycles[sc] += stat.busy_cycles
            sc_switch_count[sc] += stat.switch_count
            sc_initial_count[sc] += stat.initial_activation_count
            sc_wait_cycles[sc] += stat.wait_cycles

    return _PrefillParallelChunkResult(
        records=tuple(records),
        layer_cycles=tuple(tuple(x) for x in layer_cycles),
        layer_switches=tuple(tuple(x) for x in layer_switches),
        layer_waits=tuple(tuple(x) for x in layer_waits),
        sc_task_count=tuple(sc_task_count),
        sc_busy_cycles=tuple(sc_busy_cycles),
        sc_switch_count=tuple(sc_switch_count),
        sc_initial_count=tuple(sc_initial_count),
        sc_wait_cycles=tuple(sc_wait_cycles),
        sc_critical_layer_count=tuple(sc_critical_layer_count),
        total_cycles=total_cycles,
    )


def _bounded_parallel_prefill_chunks(
    *,
    executor: ProcessPoolExecutor,
    chunks: Iterable[tuple[TraceSegmentBatch, ...]],
    max_pending: int,
) -> Iterator[_PrefillParallelChunkResult]:
    if max_pending <= 0:
        raise PrefillFastEvaluatorError("max_pending 必须大于 0。")

    iterator = iter(chunks)
    pending = set()

    def submit_one() -> bool:
        try:
            chunk = next(iterator)
        except StopIteration:
            return False
        pending.add(executor.submit(_prefill_worker_chunk, chunk))
        return True

    while len(pending) < max_pending and submit_one():
        pass

    while pending:
        done, not_done = wait(pending, return_when=FIRST_COMPLETED)
        pending = set(not_done)
        for future in done:
            yield future.result()
            submit_one()



@dataclass(frozen=True, slots=True)
class _PrefillFileRecord:
    file_order: int
    record: PrefillEvaluationRecord


@dataclass(frozen=True, slots=True)
class _PrefillFileShardResult:
    records: tuple[_PrefillFileRecord, ...]
    layer_cycles: tuple[tuple[int, ...], ...]
    layer_switches: tuple[tuple[int, ...], ...]
    layer_waits: tuple[tuple[int, ...], ...]
    sc_task_count: tuple[int, ...]
    sc_busy_cycles: tuple[int, ...]
    sc_switch_count: tuple[int, ...]
    sc_initial_count: tuple[int, ...]
    sc_wait_cycles: tuple[int, ...]
    sc_critical_layer_count: tuple[int, ...]
    total_cycles: int


_PREFILL_WORKER_TRACE_ROOT: Path | None = None


def _init_prefill_file_worker(
    index: RuntimeIndex,
    rules: ExecutionRules,
    charge_initial_activation: bool,
    scheduling_mode: str,
    trace_root: str,
) -> None:
    _init_prefill_worker(index, rules, charge_initial_activation, scheduling_mode)
    global _PREFILL_WORKER_TRACE_ROOT
    _PREFILL_WORKER_TRACE_ROOT = Path(trace_root).resolve()


def _prefill_worker_file_shard(
    shard: tuple[tuple[int, str], ...],
) -> _PrefillFileShardResult:
    scheduler = _PREFILL_WORKER_SCHEDULER
    root = _PREFILL_WORKER_TRACE_ROOT
    if scheduler is None or root is None:
        raise PrefillFastEvaluatorError("Prefill file worker 尚未初始化。")

    nl = scheduler.index.num_layers
    ns = scheduler.index.num_subcubes
    rows: list[_PrefillFileRecord] = []
    layer_cycles = [[] for _ in range(nl)]
    layer_switches = [[] for _ in range(nl)]
    layer_waits = [[] for _ in range(nl)]
    sc_task_count = [0] * ns
    sc_busy_cycles = [0] * ns
    sc_switch_count = [0] * ns
    sc_initial_count = [0] * ns
    sc_wait_cycles = [0] * ns
    sc_critical_layer_count = [0] * ns
    total_cycles = 0

    for file_order, relative_file in shard:
        path = root / relative_file
        data = _load_prefill_json(path)
        relative = Path(relative_file)
        category = relative.parts[0] if len(relative.parts) >= 2 else "__root__"

        batch = build_segment_batch(
            path=path,
            relative_file=relative_file,
            category=category,
            batch_id=file_order,
            segment_index=0,
            segment=data[0],
        )
        if batch is None or batch.stage != STAGE_PREFILL_CANDIDATE:
            continue

        result = scheduler.schedule_batch(
            batch.routed_experts_by_token,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=_PREFILL_WORKER_CHARGE_INITIAL,
            validate_routes=False,
        )
        record = make_record(batch=batch, result=result)  # type: ignore[arg-type]
        rows.append(_PrefillFileRecord(file_order=file_order, record=record))
        total_cycles += result.total_cycles

        for execution in result.layers:
            layer_id = execution.layer_id
            lr = execution.layer_result
            layer_cycles[layer_id].append(lr.total_cycles)
            layer_switches[layer_id].append(lr.switch_count)
            layer_waits[layer_id].append(lr.wait_cycles)
            for stat in lr.subcube_stats:
                if stat.task_count > 0 and stat.last_finish_time == lr.total_cycles:
                    sc_critical_layer_count[stat.subcube_id] += 1

        for stat in result.subcube_stats:
            sc = stat.subcube_id
            sc_task_count[sc] += stat.task_count
            sc_busy_cycles[sc] += stat.busy_cycles
            sc_switch_count[sc] += stat.switch_count
            sc_initial_count[sc] += stat.initial_activation_count
            sc_wait_cycles[sc] += stat.wait_cycles

    return _PrefillFileShardResult(
        records=tuple(rows),
        layer_cycles=tuple(tuple(x) for x in layer_cycles),
        layer_switches=tuple(tuple(x) for x in layer_switches),
        layer_waits=tuple(tuple(x) for x in layer_waits),
        sc_task_count=tuple(sc_task_count),
        sc_busy_cycles=tuple(sc_busy_cycles),
        sc_switch_count=tuple(sc_switch_count),
        sc_initial_count=tuple(sc_initial_count),
        sc_wait_cycles=tuple(sc_wait_cycles),
        sc_critical_layer_count=tuple(sc_critical_layer_count),
        total_cycles=total_cycles,
    )


def _make_prefill_file_shards(
    files: list[Path],
    root: Path,
    workers: int,
) -> list[tuple[tuple[int, str], ...]]:
    if not files:
        return []
    shard_count = min(len(files), workers * 4)
    shards: list[list[tuple[int, str]]] = [[] for _ in range(shard_count)]
    for file_order, path in enumerate(files):
        shards[file_order % shard_count].append((file_order, str(path.relative_to(root))))
    return [tuple(x) for x in shards if x]


def validate_fast_against_exact(
    *,
    index: RuntimeIndex,
    rules: ExecutionRules,
    batch: TraceSegmentBatch,
    fast_result: FastPrefillBatchResult,
    charge_initial_activation: bool,
    scheduling_mode: str,
) -> None:
    exact = schedule_prefill_batch(
        index=index,
        routed_experts_by_token=batch.routed_experts_by_token,
        rules=rules,
        initial_active_cube_by_subcube=None,
        charge_initial_activation=charge_initial_activation,
        scheduling_mode=scheduling_mode,
    )

    fast_layer_cycles = tuple(x.cycles for x in fast_result.layers)
    exact_layer_cycles = tuple(x.cycles for x in exact.layers)

    pairs = {
        "total_cycles": (fast_result.total_cycles, exact.total_cycles),
        "layer_cycles": (fast_layer_cycles, exact_layer_cycles),
        "total_tasks": (fast_result.total_tasks, exact.total_tasks),
        "switches": (fast_result.total_switches, exact.total_switches),
        "initial": (fast_result.total_initial_activations, exact.total_initial_activations),
        "activation_cycles": (
            fast_result.total_activation_overhead_cycles,
            exact.total_activation_overhead_cycles,
        ),
        "compute_cycles": (fast_result.total_compute_work_cycles, exact.total_compute_work_cycles),
        "busy_cycles": (fast_result.total_busy_cycles, exact.total_busy_cycles),
        "wait_cycles": (fast_result.total_wait_cycles, exact.total_wait_cycles),
        "max_wait": (fast_result.max_task_wait_cycles, exact.max_task_wait_cycles),
        "final_state": (
            fast_result.final_active_cube_by_subcube,
            exact.final_active_cube_by_subcube,
        ),
    }

    for name, (fast_value, exact_value) in pairs.items():
        if fast_value != exact_value:
            raise PrefillFastEvaluatorError(
                f"FAST Prefill != EXACT：Batch-{batch.batch_id}, {name}, "
                f"fast={fast_value}, exact={exact_value}。"
            )

    for layer_id, (fast_execution, exact_execution) in enumerate(
        zip(fast_result.layers, exact.layers)
    ):
        fast_layer = fast_execution.layer_result
        exact_layer = exact_execution.layer_result
        if (
            fast_layer.switch_count != exact_layer.switch_count
            or fast_layer.wait_cycles != exact_layer.wait_cycles
            or fast_layer.max_task_wait_cycles != exact_layer.max_task_wait_cycles
            or fast_layer.final_active_cube_by_subcube
            != exact_layer.final_active_cube_by_subcube
        ):
            raise PrefillFastEvaluatorError(
                f"FAST Prefill != EXACT：Batch-{batch.batch_id}, Layer-{layer_id} 统计不一致。"
            )



def _evaluate_prefill_fast_file_parallel(
    *,
    index: RuntimeIndex,
    trace_root: Path | str,
    rules: ExecutionRules,
    max_files: int | None,
    charge_initial_activation: bool,
    exact_check: int,
    progress_every: int,
    verbose: bool,
    workers: int,
    scheduling_mode: str,
) -> tuple[PrefillEvaluationSummary, tuple[PrefillEvaluationRecord, ...]]:
    root = Path(trace_root).resolve()
    files = list(discover_trace_files(root))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise PrefillFastEvaluatorError("没有找到 Prefill Trace 文件。")

    # exact-check 只做校验，正式统计由文件级 worker 全量生成。
    checked = 0
    if exact_check > 0:
        scheduler = FastPrefillScheduler(
            index=index,
            rules=rules,
            scheduling_mode=scheduling_mode,
        )
        exact_stats = PrefillWorkloadStats()
        for batch in iter_prefill_batches(
            trace_root=root,
            max_files=max_files,
            max_batches=exact_check,
            stats=exact_stats,
            verbose=False,
        ):
            result = scheduler.schedule_batch(
                batch.routed_experts_by_token,
                initial_active_cube_by_subcube=None,
                charge_initial_activation=charge_initial_activation,
                validate_routes=False,
            )
            validate_fast_against_exact(
                index=index,
                rules=rules,
                batch=batch,
                fast_result=result,
                charge_initial_activation=charge_initial_activation,
                scheduling_mode=scheduling_mode,
            )
            checked += 1
            if verbose:
                print(f"[PrefillFastCheck] {checked}/{exact_check} FAST == EXACT")

    layer_cycle_values = [[] for _ in range(index.num_layers)]
    layer_switch_values = [[] for _ in range(index.num_layers)]
    layer_wait_values = [[] for _ in range(index.num_layers)]
    sc_task_count = [0] * index.num_subcubes
    sc_busy_cycles = [0] * index.num_subcubes
    sc_switch_count = [0] * index.num_subcubes
    sc_initial_count = [0] * index.num_subcubes
    sc_wait_cycles = [0] * index.num_subcubes
    sc_critical_layer_count = [0] * index.num_subcubes
    file_records: list[_PrefillFileRecord] = []
    running_total_cycles = 0

    shards = _make_prefill_file_shards(files, root, workers)
    if verbose:
        print(
            "[PrefillFastFileParallel] "
            f"workers={workers}, shards={len(shards)}, files={len(files)}"
        )

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_prefill_file_worker,
        initargs=(
            index,
            rules,
            charge_initial_activation,
            scheduling_mode,
            str(root),
        ),
    ) as executor:
        futures = [executor.submit(_prefill_worker_file_shard, shard) for shard in shards]
        pending = set(futures)
        last_report = 0
        while pending:
            done, not_done = wait(pending, return_when=FIRST_COMPLETED)
            pending = set(not_done)
            for future in done:
                part = future.result()
                file_records.extend(part.records)
                running_total_cycles += part.total_cycles

                for layer_id in range(index.num_layers):
                    layer_cycle_values[layer_id].extend(part.layer_cycles[layer_id])
                    layer_switch_values[layer_id].extend(part.layer_switches[layer_id])
                    layer_wait_values[layer_id].extend(part.layer_waits[layer_id])
                for sc in range(index.num_subcubes):
                    sc_task_count[sc] += part.sc_task_count[sc]
                    sc_busy_cycles[sc] += part.sc_busy_cycles[sc]
                    sc_switch_count[sc] += part.sc_switch_count[sc]
                    sc_initial_count[sc] += part.sc_initial_count[sc]
                    sc_wait_cycles[sc] += part.sc_wait_cycles[sc]
                    sc_critical_layer_count[sc] += part.sc_critical_layer_count[sc]

                if verbose and len(file_records) - last_report >= progress_every:
                    last_report = len(file_records)
                    print(
                        "[PrefillFastFileParallel] "
                        f"batches={len(file_records)}, "
                        f"mean_cycles={running_total_cycles / len(file_records):.2f}"
                    )

    if not file_records:
        raise PrefillFastEvaluatorError("没有找到 Prefill Candidate。")

    file_records.sort(key=lambda x: x.file_order)
    records = [
        replace(item.record, batch_id=batch_id)
        for batch_id, item in enumerate(file_records)
    ]

    summary = build_summary(
        records=records,
        layer_cycle_values=layer_cycle_values,
        layer_switch_values=layer_switch_values,
        layer_wait_values=layer_wait_values,
        sc_task_count=sc_task_count,
        sc_busy_cycles=sc_busy_cycles,
        sc_switch_count=sc_switch_count,
        sc_initial_count=sc_initial_count,
        sc_wait_cycles=sc_wait_cycles,
        sc_critical_layer_count=sc_critical_layer_count,
    )
    return summary, tuple(records)


def evaluate_prefill_fast(
    *,
    index: RuntimeIndex,
    trace_root: Path | str = DEFAULT_TRACE_ROOT,
    rules: ExecutionRules | None = None,
    max_files: int | None = None,
    max_batches: int | None = None,
    charge_initial_activation: bool = True,
    exact_check: int = 5,
    progress_every: int = 50,
    verbose: bool = True,
    workers: int = 1,
    parallel_chunk_size: int = 8,
    scheduling_mode: str = PREFILL_MODE_SWITCH_AWARE,
) -> tuple[PrefillEvaluationSummary, tuple[PrefillEvaluationRecord, ...]]:
    """
    Fast Prefill evaluator。

    exact-check 前 N 个 Batch 在主进程做 FAST == EXACT；剩余 Batch 可按
    chunk 分发给多个进程。每个请求之间本来就相互独立，因此该并行只
    加速模拟器，不改变请求内部 58 层与 SC 调度语义。
    """

    if rules is None:
        rules = ExecutionRules()
    scheduling_mode = normalize_prefill_scheduling_mode(scheduling_mode)
    if exact_check < 0:
        raise PrefillFastEvaluatorError("exact_check 不能小于 0。")
    if progress_every <= 0:
        raise PrefillFastEvaluatorError("progress_every 必须大于 0。")
    if parallel_chunk_size <= 0:
        raise PrefillFastEvaluatorError("parallel_chunk_size 必须大于 0。")

    resolved_workers = _resolve_worker_count(workers)

    # 全量 Prefill 按文件粗粒度并行；Smoke / max_batches 小实验保持单进程，
    # 避免 spawn 启动成本超过实际调度时间。
    if resolved_workers > 1 and max_batches is None:
        return _evaluate_prefill_fast_file_parallel(
            index=index,
            trace_root=trace_root,
            rules=rules,
            max_files=max_files,
            charge_initial_activation=charge_initial_activation,
            exact_check=exact_check,
            progress_every=progress_every,
            verbose=verbose,
            workers=resolved_workers,
            scheduling_mode=scheduling_mode,
        )

    if resolved_workers > 1 and max_batches is not None and verbose:
        print(
            "[PrefillFast] max_batches 已设置，使用单进程 Fast；"
            "全量评估时才启用文件级多进程。"
        )
    resolved_workers = 1
    scheduler = FastPrefillScheduler(
        index=index,
        rules=rules,
        scheduling_mode=scheduling_mode,
    )
    workload_stats = PrefillWorkloadStats()
    batch_iterator = iter_prefill_batches(
        trace_root=trace_root,
        max_files=max_files,
        max_batches=max_batches,
        stats=workload_stats,
        verbose=False,
    )

    records: list[PrefillEvaluationRecord] = []
    layer_cycle_values = [[] for _ in range(index.num_layers)]
    layer_switch_values = [[] for _ in range(index.num_layers)]
    layer_wait_values = [[] for _ in range(index.num_layers)]

    sc_task_count = [0] * index.num_subcubes
    sc_busy_cycles = [0] * index.num_subcubes
    sc_switch_count = [0] * index.num_subcubes
    sc_initial_count = [0] * index.num_subcubes
    sc_wait_cycles = [0] * index.num_subcubes
    sc_critical_layer_count = [0] * index.num_subcubes

    checked = 0
    running_total_cycles = 0

    def accumulate_batch(
        batch: TraceSegmentBatch,
        result: FastPrefillBatchResult,
    ) -> None:
        nonlocal running_total_cycles

        records.append(make_record(batch=batch, result=result))  # type: ignore[arg-type]
        running_total_cycles += result.total_cycles

        for execution in result.layers:
            layer_id = execution.layer_id
            layer_result = execution.layer_result
            layer_cycle_values[layer_id].append(layer_result.total_cycles)
            layer_switch_values[layer_id].append(layer_result.switch_count)
            layer_wait_values[layer_id].append(layer_result.wait_cycles)

            for stat in layer_result.subcube_stats:
                if stat.task_count > 0 and stat.last_finish_time == layer_result.total_cycles:
                    sc_critical_layer_count[stat.subcube_id] += 1

        for stat in result.subcube_stats:
            sc = stat.subcube_id
            sc_task_count[sc] += stat.task_count
            sc_busy_cycles[sc] += stat.busy_cycles
            sc_switch_count[sc] += stat.switch_count
            sc_initial_count[sc] += stat.initial_activation_count
            sc_wait_cycles[sc] += stat.wait_cycles

    def accumulate_chunk(chunk: _PrefillParallelChunkResult) -> None:
        nonlocal running_total_cycles

        records.extend(chunk.records)
        running_total_cycles += chunk.total_cycles

        for layer_id in range(index.num_layers):
            layer_cycle_values[layer_id].extend(chunk.layer_cycles[layer_id])
            layer_switch_values[layer_id].extend(chunk.layer_switches[layer_id])
            layer_wait_values[layer_id].extend(chunk.layer_waits[layer_id])

        for sc in range(index.num_subcubes):
            sc_task_count[sc] += chunk.sc_task_count[sc]
            sc_busy_cycles[sc] += chunk.sc_busy_cycles[sc]
            sc_switch_count[sc] += chunk.sc_switch_count[sc]
            sc_initial_count[sc] += chunk.sc_initial_count[sc]
            sc_wait_cycles[sc] += chunk.sc_wait_cycles[sc]
            sc_critical_layer_count[sc] += chunk.sc_critical_layer_count[sc]

    # ========================================================
    # exact-check：主进程
    # ========================================================

    while checked < exact_check:
        try:
            batch = next(batch_iterator)
        except StopIteration:
            break

        result = scheduler.schedule_batch(
            batch.routed_experts_by_token,
            initial_active_cube_by_subcube=None,
            charge_initial_activation=charge_initial_activation,
            validate_routes=False,
        )
        validate_fast_against_exact(
            index=index,
            rules=rules,
            batch=batch,
            fast_result=result,
            charge_initial_activation=charge_initial_activation,
            scheduling_mode=scheduling_mode,
        )
        checked += 1
        accumulate_batch(batch, result)

        if verbose:
            print(f"[PrefillFastCheck] {checked}/{exact_check} FAST == EXACT")

    # ========================================================
    # 剩余 Batch
    # ========================================================

    if resolved_workers <= 1:
        for batch in batch_iterator:
            result = scheduler.schedule_batch(
                batch.routed_experts_by_token,
                initial_active_cube_by_subcube=None,
                charge_initial_activation=charge_initial_activation,
                validate_routes=False,
            )
            accumulate_batch(batch, result)

            n = len(records)
            if verbose and (n == 1 or n % progress_every == 0):
                print(
                    "[PrefillFast] "
                    f"batches={n}, last_tokens={batch.token_count}, "
                    f"last_cycles={result.total_cycles}, "
                    f"mean_cycles={running_total_cycles / n:.2f}, workers=1"
                )
    else:
        if verbose:
            print(
                "[PrefillFastParallel] "
                f"workers={resolved_workers}, chunk={parallel_chunk_size}"
            )

        ctx = mp.get_context("spawn")
        chunks = _iter_prefill_chunks(batch_iterator, parallel_chunk_size)
        with ProcessPoolExecutor(
            max_workers=resolved_workers,
            mp_context=ctx,
            initializer=_init_prefill_worker,
            initargs=(
                index,
                rules,
                charge_initial_activation,
                scheduling_mode,
            ),
        ) as executor:
            for chunk_result in _bounded_parallel_prefill_chunks(
                executor=executor,
                chunks=chunks,
                max_pending=max(2, resolved_workers * 2),
            ):
                accumulate_chunk(chunk_result)

                n = len(records)
                if verbose and (n == 1 or n % progress_every < parallel_chunk_size):
                    print(
                        "[PrefillFastParallel] "
                        f"batches={n}, mean_cycles={running_total_cycles / n:.2f}"
                    )

    if not records:
        raise PrefillFastEvaluatorError("没有找到 Prefill Candidate。")

    # worker 完成顺序不固定；batch_id 排序保持 JSON 稳定。
    records.sort(key=lambda record: record.batch_id)

    summary = build_summary(
        records=records,
        layer_cycle_values=layer_cycle_values,
        layer_switch_values=layer_switch_values,
        layer_wait_values=layer_wait_values,
        sc_task_count=sc_task_count,
        sc_busy_cycles=sc_busy_cycles,
        sc_switch_count=sc_switch_count,
        sc_initial_count=sc_initial_count,
        sc_wait_cycles=sc_wait_cycles,
        sc_critical_layer_count=sc_critical_layer_count,
    )
    return summary, tuple(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="高性能 Chinese-SimpleQA segment0 Prefill 评估；前 N Batch 与 EXACT 硬校验。"
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--exact-check", type=int, default=5)
    parser.add_argument(
        "--scheduling-mode",
        choices=PREFILL_SCHEDULING_MODES,
        default=PREFILL_MODE_SWITCH_AWARE,
        help="Prefill 调度：no_reuse / switch_aware / aggressive_reuse / largest_batch_reuse。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="并行 worker 数；0=自动，1=关闭多进程。",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--top-layers", type=int, default=10)
    parser.add_argument("--top-subcubes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-initial-activation-cost", action="store_true")
    args = parser.parse_args()

    index = load_runtime_index(args.mapping)
    summary, records = evaluate_prefill_fast(
        index=index,
        trace_root=args.root,
        rules=ExecutionRules(),
        max_files=args.max_files,
        max_batches=args.max_batches,
        charge_initial_activation=not args.no_initial_activation_cost,
        exact_check=args.exact_check,
        progress_every=args.progress_every,
        verbose=not args.quiet,
        workers=args.workers,
        scheduling_mode=args.scheduling_mode,
    )

    print_prefill_evaluation_summary(
        summary,
        top_layers=args.top_layers,
        top_subcubes=args.top_subcubes,
    )

    if not args.no_save:
        saved = save_prefill_evaluation(
            output_path=args.output,
            summary=summary,
            records=records,
            mapping_path=args.mapping,
            trace_root=args.root,
            charge_initial_activation=not args.no_initial_activation_cost,
        )
        print(f"\nSaved：{saved}")


if __name__ == "__main__":
    main()
