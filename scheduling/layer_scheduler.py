"""
第五步：单个 MoE Layer 的事件调度器。

输入：

    RuntimeIndex

以及某一层 Router 选择出的：

    Routed Top-K Expert

RuntimeIndex 会自动加入：

    Shared Expert

------------------------------------------------------------

当前执行规则：

1. gate / up 初始即可执行；

2. 对每个 Expert：

       gate(e) ----\
                    -> down(e)
       up(e) ------/

   down 只有在该 Expert 自己的 gate 和 up
   都完成后才可以执行；

3. 不同 Sub-Cube 完全并行；

4. 同一个 Sub-Cube 同一时刻只能执行
   一个 Weight-Cube；

5. Weight-Cube depth = 1：

       compute = 1 cycle

6. Weight-Cube 激活 / 切换：

       activation / switch = 1 cycle

7. 跨 Sub-Cube：

       0 cycle

------------------------------------------------------------

注意：

本文件只模拟：

    一个 token
    在
    一个 MoE Layer

中的执行。

后面的 token_scheduler.py 再把：

    58 个 Layer

串起来。
"""

from __future__ import annotations

import argparse
import heapq

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


from config import (
    ExecutionRules,
)

from mapping.logical_weight import (
    MATRIX_DOWN,
    MATRIX_GATE,
    MATRIX_UP,
)

from scheduling.runtime_index import (
    DEFAULT_MAPPING_PATH,
    RuntimeIndex,
    RuntimeMatrixLocation,
    load_runtime_index,
)


# ============================================================
# 异常
# ============================================================


class LayerSchedulerError(ValueError):
    """单层调度失败。"""


# ============================================================
# 尚未执行的任务
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class _TaskSpec:
    """
    一个等待执行的 Weight-Cube。

    ready_time：

        依赖已经满足，
        可以开始竞争 Sub-Cube 的最早时间。

    route_rank：

        Router Top-K 中的原始顺序。

        Shared Expert 放在最后。
    """

    layer_id: int

    expert_id: int

    route_rank: int

    matrix_name: str

    location: RuntimeMatrixLocation

    ready_time: int


# ============================================================
# 一次实际执行记录
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class ScheduledTask:
    """
    一次 Weight-Cube 的完整执行记录。
    """

    # ========================================================
    # Identity
    # ========================================================

    layer_id: int

    expert_id: int

    matrix_name: str

    cube_id: int

    subcube_id: int

    # ========================================================
    # Timing
    # ========================================================

    # 依赖满足时间
    ready_time: int

    # 开始占用 Sub-Cube
    #
    # 如果需要切换，
    # 从这里开始花 switch cycle。
    dispatch_time: int

    # 真正开始矩阵计算
    compute_start_time: int

    # 完成时间
    finish_time: int

    # ========================================================
    # Cost
    # ========================================================

    # 因 Sub-Cube 忙碌造成的等待
    wait_cycles: int

    # 激活 / 切换开销
    activation_cycles: int

    # 矩阵计算开销
    compute_cycles: int

    # 执行前这个 SC 激活的是谁
    previous_active_cube_id: (
        int | None
    )

    # ========================================================
    # Properties
    # ========================================================

    @property
    def total_service_cycles(
        self,
    ) -> int:
        """
        本任务实际占用 SC 的总周期。
        """

        return (
            self.activation_cycles
            + self.compute_cycles
        )

    @property
    def switched_from_another_cube(
        self,
    ) -> bool:
        """
        是否真的发生：

            WeightCube-A
                ->
            WeightCube-B
        """

        return (
            self.previous_active_cube_id
            is not None
            and
            self.previous_active_cube_id
            != self.cube_id
        )

    @property
    def is_initial_activation(
        self,
    ) -> bool:
        """
        是否属于：

            None
              ->
            WeightCube

        的首次启动。
        """

        return (
            self.previous_active_cube_id
            is None
            and
            self.activation_cycles
            > 0
        )


# ============================================================
# 单个 Sub-Cube 的 Layer 统计
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class SubcubeLayerStats:
    """
    一个 Sub-Cube 在当前 Layer 中的统计。
    """

    subcube_id: int

    task_count: int

    compute_cycles: int

    activation_cycles: int

    # 已有 WeightCube -> 另一个 WeightCube
    switch_count: int

    # None -> 第一个 WeightCube
    initial_activation_count: int

    # compute + activation
    busy_cycles: int

    # 所有任务累计等待
    wait_cycles: int

    last_finish_time: int

    initial_active_cube_id: (
        int | None
    )

    final_active_cube_id: (
        int | None
    )

    def utilization(
        self,
        layer_cycles: int,
    ) -> float:
        """
        当前层：

            SC busy cycles
            ----------------
            layer total cycles
        """

        if layer_cycles <= 0:
            return 0.0

        return (
            self.busy_cycles
            / layer_cycles
        )


# ============================================================
# 单层结果
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class LayerScheduleResult:
    """
    一个 token 在一个 MoE Layer
    的完整执行结果。
    """

    layer_id: int

    routed_expert_ids: tuple[
        int,
        ...
    ]

    # Routed Top-K + Shared
    active_expert_ids: tuple[
        int,
        ...
    ]

    total_cycles: int

    tasks: tuple[
        ScheduledTask,
        ...
    ]

    subcube_stats: tuple[
        SubcubeLayerStats,
        ...
    ]

    initial_active_cube_by_subcube: tuple[
        int | None,
        ...
    ]

    final_active_cube_by_subcube: tuple[
        int | None,
        ...
    ]

    # ========================================================
    # Summary
    # ========================================================

    @property
    def task_count(
        self,
    ) -> int:

        return len(
            self.tasks
        )

    @property
    def switch_count(
        self,
    ) -> int:

        return sum(
            stat.switch_count
            for stat
            in self.subcube_stats
        )

    @property
    def initial_activation_count(
        self,
    ) -> int:

        return sum(
            stat.initial_activation_count
            for stat
            in self.subcube_stats
        )

    @property
    def activation_overhead_cycles(
        self,
    ) -> int:

        return sum(
            stat.activation_cycles
            for stat
            in self.subcube_stats
        )

    @property
    def compute_cycles(
        self,
    ) -> int:

        return sum(
            stat.compute_cycles
            for stat
            in self.subcube_stats
        )

    @property
    def busy_cycles(
        self,
    ) -> int:

        return sum(
            stat.busy_cycles
            for stat
            in self.subcube_stats
        )

    @property
    def wait_cycles(
        self,
    ) -> int:

        return sum(
            task.wait_cycles
            for task
            in self.tasks
        )

    @property
    def max_task_wait_cycles(
        self,
    ) -> int:

        return max(
            (
                task.wait_cycles
                for task
                in self.tasks
            ),
            default=0,
        )

    # ========================================================
    # Task Query
    # ========================================================

    def task(
        self,
        expert_id: int,
        matrix_name: str,
    ) -> ScheduledTask:

        for task in self.tasks:

            if (
                task.expert_id
                == expert_id
                and
                task.matrix_name
                == matrix_name
            ):

                return task

        raise LayerSchedulerError(
            "找不到任务："
            f"expert={expert_id}, "
            f"matrix={matrix_name}。"
        )


# ============================================================
# Matrix Priority
# ============================================================


def _matrix_priority(
    matrix_name: str,
) -> int:
    """
    同一个 ready_time 时：

        gate
        up
        down

    只是确定性 tie-break。

    真正并行关系仍由 SC 决定。
    """

    if (
        matrix_name
        == MATRIX_GATE
    ):
        return 0

    if (
        matrix_name
        == MATRIX_UP
    ):
        return 1

    if (
        matrix_name
        == MATRIX_DOWN
    ):
        return 2

    raise LayerSchedulerError(
        "未知 matrix_name："
        f"{matrix_name!r}。"
    )


# ============================================================
# Initial Active State
# ============================================================


def _validate_initial_state(
    *,
    index: RuntimeIndex,

    initial_active_cube_by_subcube: (
        Iterable[int | None]
        | None
    ),
) -> tuple[
    int | None,
    ...
]:
    """
    标准化每个 SC 初始 active Weight-Cube。

    后面 58 层连续模拟时：

        上一层 final state
            ->
        下一层 initial state

    就靠这个接口传递。
    """

    if (
        initial_active_cube_by_subcube
        is None
    ):

        return tuple(
            None
            for _ in range(
                index.num_subcubes
            )
        )

    state = tuple(
        initial_active_cube_by_subcube
    )

    if (
        len(state)
        != index.num_subcubes
    ):

        raise LayerSchedulerError(
            "initial_active_cube_by_subcube "
            "长度错误："
            f"actual={len(state)}, "
            f"expected="
            f"{index.num_subcubes}。"
        )

    for cube_id in state:

        if (
            cube_id is not None
            and
            (
                not isinstance(
                    cube_id,
                    int,
                )
                or
                isinstance(
                    cube_id,
                    bool,
                )
                or
                cube_id < 0
            )
        ):

            raise LayerSchedulerError(
                "初始 active cube "
                "必须为 None 或非负整数。"
            )

    return state


# ============================================================
# Activation Cost
# ============================================================


def _activation_cost(
    *,
    previous_cube_id: int | None,

    next_cube_id: int,

    rules: ExecutionRules,

    charge_initial_activation: bool,
) -> tuple[
    int,
    bool,
    bool,
]:
    """
    返回：

        (
            activation_cycles,
            is_switch,
            is_initial_activation
        )

    --------------------------------------------------------

    previous = None：

        默认：
            1 cycle startup

    previous == next：

        已经激活这个 Weight-Cube，
        不需要切换。

    previous != next：

        Weight-Cube Switching
        = 1 cycle
    """

    # ========================================================
    # First activation
    # ========================================================

    if (
        previous_cube_id
        is None
    ):

        if charge_initial_activation:

            return (
                rules.switch_cycles,
                False,
                True,
            )

        return (
            0,
            False,
            False,
        )

    # ========================================================
    # Same Weight-Cube
    # ========================================================

    if (
        previous_cube_id
        == next_cube_id
    ):

        return (
            0,
            False,
            False,
        )

    # ========================================================
    # Switch
    # ========================================================

    return (
        rules.switch_cycles,
        True,
        False,
    )


# ============================================================
# 同一 SC 的任务选择
# ============================================================


def _select_ready_task(
    *,
    queue: list[
        _TaskSpec
    ],

    current_time: int,

    active_cube_id: (
        int | None
    ),
) -> _TaskSpec | None:
    """
    当前 Baseline 的 SC 内调度顺序：

    1. ready_time 更早；

    2. 如果当前已经激活的 WeightCube
       正好也在等待队列里，
       优先继续它；

    3. Router route_rank；

    4. gate -> up -> down；

    5. cube_id。

    这是一个固定、可复现的简单调度规则。
    """

    candidates = [
        task

        for task
        in queue

        if (
            task.ready_time
            <= current_time
        )
    ]

    if not candidates:

        return None

    def key(
        task: _TaskSpec,
    ) -> tuple[
        int,
        int,
        int,
        int,
        int,
    ]:

        already_active = (
            active_cube_id
            == task.location.cube_id
        )

        return (
            task.ready_time,

            0
            if already_active
            else 1,

            task.route_rank,

            _matrix_priority(
                task.matrix_name
            ),

            task.location.cube_id,
        )

    return min(
        candidates,
        key=key,
    )


# ============================================================
# 主调度函数
# ============================================================


def schedule_layer(
    *,
    index: RuntimeIndex,

    layer_id: int,

    routed_expert_ids: Iterable[
        int
    ],

    rules: ExecutionRules | None = None,

    initial_active_cube_by_subcube: (
        Iterable[int | None]
        | None
    ) = None,

    charge_initial_activation: bool = True,
) -> LayerScheduleResult:
    """
    模拟：

        一个 token
        一个 MoE Layer

    --------------------------------------------------------

    这里不是：

        所有 gate/up 全结束
            ↓
        所有 down 再开始

    而是真正按 Expert 依赖：

        gate(e) 完成
        +
        up(e) 完成
            ↓
        down(e) ready

    down ready 后仍然需要等待其目标 SC 空闲。
    """

    if rules is None:

        rules = (
            ExecutionRules()
        )

    # ========================================================
    # Rules
    # ========================================================

    if (
        rules.compute_cycles
        <= 0
        or
        rules.switch_cycles
        < 0
    ):

        raise LayerSchedulerError(
            "ExecutionRules 周期配置非法。"
        )

    if not (
        rules.unlimited_parallel_subcubes
    ):

        raise LayerSchedulerError(
            "当前 Scheduler 要求 "
            "不同 Sub-Cube 可以完全并行。"
        )

    if not (
        rules
        .one_active_weight_cube_per_subcube
    ):

        raise LayerSchedulerError(
            "当前 Scheduler 基于 "
            "Sub-Cube 内 Weight-Cube 互斥规则。"
        )

    if (
        rules.cross_subcube_cycles
        != 0
    ):

        raise LayerSchedulerError(
            "当前 Baseline "
            "跨 Sub-Cube 开销必须为 0。"
        )

    # ========================================================
    # Active Experts
    # ========================================================

    routed = tuple(
        routed_expert_ids
    )

    active_expert_ids = (
        index.resolve_active_expert_ids(
            layer_id=layer_id,

            routed_expert_ids=(
                routed
            ),
        )
    )

    route_rank = {
        expert_id: rank

        for (
            rank,
            expert_id,
        ) in enumerate(
            active_expert_ids
        )
    }

    active_experts = {
        expert_id:
        index.expert(
            layer_id,
            expert_id,
        )

        for expert_id
        in active_expert_ids
    }

    # ========================================================
    # Initial State
    # ========================================================

    initial_state = (
        _validate_initial_state(
            index=index,

            initial_active_cube_by_subcube=(
                initial_active_cube_by_subcube
            ),
        )
    )

    active_cube_by_sc = list(
        initial_state
    )

    # ========================================================
    # Ready Queues
    # ========================================================

    ready_by_sc: list[
        list[
            _TaskSpec
        ]
    ] = [
        []
        for _ in range(
            index.num_subcubes
        )
    ]

    # ========================================================
    # t=0:
    #
    # 所有 gate / up 都 ready。
    # ========================================================

    for expert_id in (
        active_expert_ids
    ):

        expert = (
            active_experts[
                expert_id
            ]
        )

        for (
            matrix_name,
            location,
        ) in (
            (
                MATRIX_GATE,
                expert.gate,
            ),
            (
                MATRIX_UP,
                expert.up,
            ),
        ):

            ready_by_sc[
                location.subcube_id
            ].append(
                _TaskSpec(
                    layer_id=(
                        layer_id
                    ),

                    expert_id=(
                        expert_id
                    ),

                    route_rank=(
                        route_rank[
                            expert_id
                        ]
                    ),

                    matrix_name=(
                        matrix_name
                    ),

                    location=(
                        location
                    ),

                    ready_time=0,
                )
            )

    # ========================================================
    # Runtime State
    # ========================================================

    current_time = 0

    running_by_sc: list[
        ScheduledTask | None
    ] = [
        None
        for _ in range(
            index.num_subcubes
        )
    ]

    # heap:
    #
    # (
    #     finish_time,
    #     serial,
    #     subcube_id,
    #     ScheduledTask
    # )
    running_heap = []

    serial = 0

    completion_time: dict[
        tuple[
            int,
            str,
        ],
        int,
    ] = {}

    down_created: set[
        int
    ] = set()

    completed_tasks: list[
        ScheduledTask
    ] = []

    # ========================================================
    # Stats
    # ========================================================

    task_count = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    compute_cycles = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    activation_cycles = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    switch_count = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    initial_activation_count = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    wait_cycles = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    last_finish_time = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    expected_task_count = (
        len(
            active_expert_ids
        )
        * 3
    )

    # ========================================================
    # Dispatch
    # ========================================================

    def dispatch_ready_tasks() -> int:

        nonlocal serial

        dispatched = 0

        for sc in range(
            index.num_subcubes
        ):

            # SC 正忙
            if (
                running_by_sc[
                    sc
                ]
                is not None
            ):

                continue

            spec = (
                _select_ready_task(
                    queue=(
                        ready_by_sc[
                            sc
                        ]
                    ),

                    current_time=(
                        current_time
                    ),

                    active_cube_id=(
                        active_cube_by_sc[
                            sc
                        ]
                    ),
                )
            )

            if spec is None:
                continue

            ready_by_sc[
                sc
            ].remove(
                spec
            )

            previous_cube_id = (
                active_cube_by_sc[
                    sc
                ]
            )

            (
                activation_cost,
                is_switch,
                is_initial,
            ) = _activation_cost(
                previous_cube_id=(
                    previous_cube_id
                ),

                next_cube_id=(
                    spec.location.cube_id
                ),

                rules=rules,

                charge_initial_activation=(
                    charge_initial_activation
                ),
            )

            # =================================================
            # Timing
            # =================================================

            dispatch_time = (
                current_time
            )

            compute_start_time = (
                dispatch_time
                + activation_cost
            )

            finish_time = (
                compute_start_time
                + rules.compute_cycles
            )

            wait = (
                dispatch_time
                - spec.ready_time
            )

            if wait < 0:

                raise LayerSchedulerError(
                    "任务在 ready_time "
                    "之前被执行。"
                )

            scheduled = (
                ScheduledTask(
                    layer_id=(
                        layer_id
                    ),

                    expert_id=(
                        spec.expert_id
                    ),

                    matrix_name=(
                        spec.matrix_name
                    ),

                    cube_id=(
                        spec.location
                        .cube_id
                    ),

                    subcube_id=(
                        sc
                    ),

                    ready_time=(
                        spec.ready_time
                    ),

                    dispatch_time=(
                        dispatch_time
                    ),

                    compute_start_time=(
                        compute_start_time
                    ),

                    finish_time=(
                        finish_time
                    ),

                    wait_cycles=(
                        wait
                    ),

                    activation_cycles=(
                        activation_cost
                    ),

                    compute_cycles=(
                        rules.compute_cycles
                    ),

                    previous_active_cube_id=(
                        previous_cube_id
                    ),
                )
            )

            # =================================================
            # SC 开始执行
            # =================================================

            running_by_sc[
                sc
            ] = scheduled

            active_cube_by_sc[
                sc
            ] = (
                spec.location
                .cube_id
            )

            heapq.heappush(
                running_heap,

                (
                    finish_time,
                    serial,
                    sc,
                    scheduled,
                ),
            )

            serial += 1

            dispatched += 1

            # =================================================
            # Stats
            # =================================================

            task_count[
                sc
            ] += 1

            compute_cycles[
                sc
            ] += (
                rules.compute_cycles
            )

            activation_cycles[
                sc
            ] += (
                activation_cost
            )

            if is_switch:

                switch_count[
                    sc
                ] += 1

            if is_initial:

                initial_activation_count[
                    sc
                ] += 1

            wait_cycles[
                sc
            ] += wait

        return dispatched

    # ========================================================
    # Event Loop
    # ========================================================

    while (
        len(
            completed_tasks
        )
        < expected_task_count
    ):

        # ----------------------------------------------------
        # 当前时间：
        # 所有空闲 SC 尽可能各启动一个任务。
        # ----------------------------------------------------

        dispatch_ready_tasks()

        # ----------------------------------------------------
        # 没有任何运行中的任务。
        # ----------------------------------------------------

        if not running_heap:

            future_ready_times = [
                spec.ready_time

                for queue
                in ready_by_sc

                for spec
                in queue

                if (
                    spec.ready_time
                    > current_time
                )
            ]

            if future_ready_times:

                current_time = min(
                    future_ready_times
                )

                continue

            raise LayerSchedulerError(
                "调度死锁："
                "没有运行任务，"
                "也没有未来 Ready Task。"
            )

        # ----------------------------------------------------
        # 推进到下一次完成事件。
        # ----------------------------------------------------

        next_finish_time = (
            running_heap[
                0
            ][
                0
            ]
        )

        if (
            next_finish_time
            < current_time
        ):

            raise LayerSchedulerError(
                "内部时间状态错误。"
            )

        current_time = (
            next_finish_time
        )

        finished_now: list[
            tuple[
                int,
                ScheduledTask,
            ]
        ] = []

        # ----------------------------------------------------
        # 同一时刻可能多个 SC 一起完成。
        # ----------------------------------------------------

        while (
            running_heap
            and
            running_heap[
                0
            ][
                0
            ]
            == current_time
        ):

            (
                _finish_time,
                _serial,
                sc,
                task,
            ) = heapq.heappop(
                running_heap
            )

            if (
                running_by_sc[
                    sc
                ]
                != task
            ):

                raise LayerSchedulerError(
                    "SC running state "
                    "与事件队列不一致。"
                )

            running_by_sc[
                sc
            ] = None

            last_finish_time[
                sc
            ] = (
                current_time
            )

            key = (
                task.expert_id,
                task.matrix_name,
            )

            if (
                key
                in completion_time
            ):

                raise LayerSchedulerError(
                    "同一任务完成了两次。"
                )

            completion_time[
                key
            ] = (
                current_time
            )

            completed_tasks.append(
                task
            )

            finished_now.append(
                (
                    sc,
                    task,
                )
            )

        # ====================================================
        # 检查哪些 Expert 的 down
        # 现在可以进入 Ready Queue
        # ====================================================

        affected_experts = {
            task.expert_id

            for (
                _sc,
                task,
            ) in finished_now

            if (
                task.matrix_name
                in (
                    MATRIX_GATE,
                    MATRIX_UP,
                )
            )
        }

        for expert_id in sorted(
            affected_experts,

            key=lambda value: (
                route_rank[
                    value
                ]
            ),
        ):

            if (
                expert_id
                in down_created
            ):

                continue

            gate_key = (
                expert_id,
                MATRIX_GATE,
            )

            up_key = (
                expert_id,
                MATRIX_UP,
            )

            if (
                gate_key
                not in completion_time
                or
                up_key
                not in completion_time
            ):

                continue

            down_ready_time = max(
                completion_time[
                    gate_key
                ],

                completion_time[
                    up_key
                ],
            )

            expert = (
                active_experts[
                    expert_id
                ]
            )

            down_location = (
                expert.down
            )

            ready_by_sc[
                down_location
                .subcube_id
            ].append(
                _TaskSpec(
                    layer_id=(
                        layer_id
                    ),

                    expert_id=(
                        expert_id
                    ),

                    route_rank=(
                        route_rank[
                            expert_id
                        ]
                    ),

                    matrix_name=(
                        MATRIX_DOWN
                    ),

                    location=(
                        down_location
                    ),

                    ready_time=(
                        down_ready_time
                    ),
                )
            )

            down_created.add(
                expert_id
            )

    # ========================================================
    # Final Validation
    # ========================================================

    if (
        len(
            completion_time
        )
        != expected_task_count
    ):

        raise LayerSchedulerError(
            "完成任务数错误："
            f"actual="
            f"{len(completion_time)}, "
            f"expected="
            f"{expected_task_count}。"
        )

    if (
        len(
            down_created
        )
        != len(
            active_expert_ids
        )
    ):

        raise LayerSchedulerError(
            "不是所有 Expert "
            "都成功创建了 down。"
        )

    # ========================================================
    # 检查每个 Expert：
    #
    # down.ready_time
    # =
    # max(gate.finish, up.finish)
    # ========================================================

    for expert_id in (
        active_expert_ids
    ):

        gate_finish = (
            completion_time[
                (
                    expert_id,
                    MATRIX_GATE,
                )
            ]
        )

        up_finish = (
            completion_time[
                (
                    expert_id,
                    MATRIX_UP,
                )
            ]
        )

        down_task = next(
            task

            for task
            in completed_tasks

            if (
                task.expert_id
                == expert_id
                and
                task.matrix_name
                == MATRIX_DOWN
            )
        )

        expected_ready = max(
            gate_finish,
            up_finish,
        )

        if (
            down_task.ready_time
            != expected_ready
        ):

            raise LayerSchedulerError(
                f"Expert-{expert_id} "
                "down ready_time 错误。"
            )

        if (
            down_task.dispatch_time
            < expected_ready
        ):

            raise LayerSchedulerError(
                f"Expert-{expert_id} "
                "down 在 gate/up 完成前执行。"
            )

    total_cycles = max(
        task.finish_time
        for task
        in completed_tasks
    )

    # ========================================================
    # Sub-Cube Stats
    # ========================================================

    subcube_stats: list[
        SubcubeLayerStats
    ] = []

    for sc in range(
        index.num_subcubes
    ):

        busy = (
            compute_cycles[
                sc
            ]
            +
            activation_cycles[
                sc
            ]
        )

        subcube_stats.append(
            SubcubeLayerStats(
                subcube_id=sc,

                task_count=(
                    task_count[
                        sc
                    ]
                ),

                compute_cycles=(
                    compute_cycles[
                        sc
                    ]
                ),

                activation_cycles=(
                    activation_cycles[
                        sc
                    ]
                ),

                switch_count=(
                    switch_count[
                        sc
                    ]
                ),

                initial_activation_count=(
                    initial_activation_count[
                        sc
                    ]
                ),

                busy_cycles=(
                    busy
                ),

                wait_cycles=(
                    wait_cycles[
                        sc
                    ]
                ),

                last_finish_time=(
                    last_finish_time[
                        sc
                    ]
                ),

                initial_active_cube_id=(
                    initial_state[
                        sc
                    ]
                ),

                final_active_cube_id=(
                    active_cube_by_sc[
                        sc
                    ]
                ),
            )
        )

    # ========================================================
    # 按真实执行顺序输出
    # ========================================================

    completed_tasks.sort(
        key=lambda task: (
            task.dispatch_time,
            task.subcube_id,
            task.finish_time,
            task.expert_id,
            _matrix_priority(
                task.matrix_name
            ),
        )
    )

    return LayerScheduleResult(
        layer_id=(
            layer_id
        ),

        routed_expert_ids=(
            routed
        ),

        active_expert_ids=(
            active_expert_ids
        ),

        total_cycles=(
            total_cycles
        ),

        tasks=tuple(
            completed_tasks
        ),

        subcube_stats=tuple(
            subcube_stats
        ),

        initial_active_cube_by_subcube=(
            initial_state
        ),

        final_active_cube_by_subcube=tuple(
            active_cube_by_sc
        ),
    )


# ============================================================
# Print
# ============================================================


def print_layer_schedule_summary(
    result: LayerScheduleResult,
) -> None:
    """
    打印单 Layer 调度结果。
    """

    print(
        "\n"
        "========== Layer Schedule =========="
    )

    print(
        f"Layer："
        f"{result.layer_id}"
    )

    print(
        "Routed Experts："
        f"{result.routed_expert_ids}"
    )

    print(
        "Active Experts (+Shared)："
        f"{result.active_expert_ids}"
    )

    print(
        f"Tasks："
        f"{result.task_count}"
    )

    print(
        f"Layer Cycles："
        f"{result.total_cycles}"
    )

    print(
        "Compute Cycles "
        "(sum across SCs)："
        f"{result.compute_cycles}"
    )

    print(
        "Activation/Switch "
        "Overhead Cycles："
        f"{result.activation_overhead_cycles}"
    )

    print(
        f"Weight-Cube Switches："
        f"{result.switch_count}"
    )

    print(
        f"Initial Activations："
        f"{result.initial_activation_count}"
    )

    print(
        f"Task Wait Cycles："
        f"{result.wait_cycles}"
    )

    print(
        f"Max Task Wait："
        f"{result.max_task_wait_cycles}"
    )

    print(
        "\nSub-Cube Stats："
    )

    for stat in (
        result.subcube_stats
    ):

        if (
            stat.task_count
            == 0
        ):

            continue

        utilization = (
            stat.utilization(
                result.total_cycles
            )
        )

        print(
            f"  SC-{stat.subcube_id}: "
            f"tasks={stat.task_count}, "
            f"busy={stat.busy_cycles}, "
            f"switch={stat.switch_count}, "
            f"initial="
            f"{stat.initial_activation_count}, "
            f"wait={stat.wait_cycles}, "
            f"util={utilization:.2%}"
        )


# ============================================================
# CLI Route
# ============================================================


def _parse_route(
    text: str,
) -> tuple[
    int,
    ...
]:
    """
    CLI：

        --route 1,2,3,4,5,6,7,8
    """

    parts = [
        part.strip()

        for part
        in text.split(",")

        if part.strip()
    ]

    try:

        return tuple(
            int(part)
            for part
            in parts
        )

    except ValueError as exc:

        raise argparse.ArgumentTypeError(
            "route 必须是逗号分隔的整数。"
        ) from exc


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "模拟一个 token "
                "在一个 MoE Layer 上的 "
                "gate/up/down 调度。"
            )
        )
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=(
            DEFAULT_MAPPING_PATH
        ),
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--route",
        type=_parse_route,

        default=(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ),

        help=(
            "Routed Expert ID，"
            "例如："
            "0,1,2,3,4,5,6,7"
        ),
    )

    parser.add_argument(
        "--no-initial-activation-cost",
        action="store_true",

        help=(
            "仅用于对照实验："
            "None -> WeightCube "
            "不收取首次激活周期。"
        ),
    )

    args = (
        parser.parse_args()
    )

    index = (
        load_runtime_index(
            args.mapping
        )
    )

    result = (
        schedule_layer(
            index=index,

            layer_id=(
                args.layer
            ),

            routed_expert_ids=(
                args.route
            ),

            rules=(
                ExecutionRules()
            ),

            charge_initial_activation=(
                not args
                .no_initial_activation_cost
            ),
        )
    )

    print_layer_schedule_summary(
        result
    )


if __name__ == "__main__":
    main()