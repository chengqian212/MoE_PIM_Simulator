"""
第六步：Prefill 单个 MoE Layer 的多 Token 事件调度器。

输入：

    RuntimeIndex

以及同一个 Prefill Batch 在某一层的：

    Token-0 -> Routed Top-K
    Token-1 -> Routed Top-K
    ...
    Token-(B-1) -> Routed Top-K

RuntimeIndex 会为每个 Token 自动加入：

    Shared Expert

------------------------------------------------------------

当前执行规则：

1. Prefill 同一层内，所有 Token 的 gate / up 初始即可执行；

2. 对每个 Token、每个 Expert：

       gate(token, e) ----\
                          -> down(token, e)
       up(token, e) ------/

   某个 Token 的 down
   只等待该 Token 自己在该 Expert 上的
   gate 和 up 完成；

3. 不同 Sub-Cube 完全并行；

4. 同一个 Sub-Cube 同一时刻只能执行
   一个 Weight-Cube；

5. 每个 Token 访问一个 Weight-Cube：

       compute = 1 cycle

   因此同一个 Expert 被 30 个 Token 选中，
   仍然需要 30 次 compute；

6. Weight-Cube 激活 / 切换：

       activation / switch = 1 cycle

7. 如果同一个 Sub-Cube 连续处理
   同一个 Weight-Cube 的多个 Token：

       Switch 一次
       + 多次 Compute

   中间不重复收取切换周期；

8. 跨 Sub-Cube：

       0 cycle

------------------------------------------------------------

本文件只模拟：

    一个 Prefill Batch
    在
    一个 MoE Layer

中的执行。

后面的：

    scheduling/prefill_scheduler.py

再把：

    58 个 MoE Layer

严格串起来。
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


class PrefillLayerSchedulerError(
    ValueError
):
    """Prefill 单层调度失败。"""


# ============================================================
# 尚未执行的任务
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class _PrefillTaskSpec:
    """
    一个等待执行的：

        Token
        ×
        Expert
        ×
        Weight-Cube

    任务。

    token_index：

        当前 Prefill Batch 内的 Token 下标。

    ready_time：

        当前任务依赖满足后，
        最早可以竞争目标 Sub-Cube 的时间。

    route_rank：

        当前 Token 的 Router Top-K 原始顺序。

        Shared Expert 放最后。
    """

    token_index: int

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
class ScheduledPrefillTask:
    """
    Prefill 中一次真实矩阵计算记录。

    注意：

        两个 Token 即使访问同一个 cube_id，
        也仍然是两个不同任务。

    区别只是：

        如果连续执行同一 cube_id，
        后一个任务 activation_cycles = 0。
    """

    # ========================================================
    # Identity
    # ========================================================

    token_index: int

    layer_id: int

    expert_id: int

    matrix_name: str

    cube_id: int

    subcube_id: int

    # ========================================================
    # Timing
    # ========================================================

    ready_time: int

    dispatch_time: int

    compute_start_time: int

    finish_time: int

    # ========================================================
    # Cost
    # ========================================================

    wait_cycles: int

    activation_cycles: int

    compute_cycles: int

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

        return (
            self.activation_cycles
            + self.compute_cycles
        )

    @property
    def switched_from_another_cube(
        self,
    ) -> bool:

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

        return (
            self.previous_active_cube_id
            is None

            and

            self.activation_cycles
            > 0
        )


# ============================================================
# 单个 Sub-Cube 的 Prefill Layer 统计
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PrefillSubcubeLayerStats:
    """
    一个 Sub-Cube 在当前 Prefill Layer
    中的统计。
    """

    subcube_id: int

    task_count: int

    compute_cycles: int

    activation_cycles: int

    switch_count: int

    initial_activation_count: int

    busy_cycles: int

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

        if layer_cycles <= 0:
            return 0.0

        return (
            self.busy_cycles
            / layer_cycles
        )


# ============================================================
# Prefill 单层结果
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PrefillLayerScheduleResult:
    """
    一个 Prefill Batch 在一个 MoE Layer
    的完整调度结果。
    """

    layer_id: int

    token_count: int

    # 每个 Token 的 Routed Top-K
    routed_experts_by_token: tuple[
        tuple[
            int,
            ...
        ],
        ...
    ]

    # Routed Top-K + Shared
    active_expert_ids_by_token: tuple[
        tuple[
            int,
            ...
        ],
        ...
    ]

    total_cycles: int

    tasks: tuple[
        ScheduledPrefillTask,
        ...
    ]

    subcube_stats: tuple[
        PrefillSubcubeLayerStats,
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

    @property
    def cycles_per_token(
        self,
    ) -> float:
        """
        当前 Layer 的总时间 / 输入 Token 数。

        注意：

        这只是一个归一化指标，
        不是说每个 Token 真正独占这些周期。
        """

        if self.token_count <= 0:
            return 0.0

        return (
            self.total_cycles
            / self.token_count
        )

    @property
    def tokens_per_cycle(
        self,
    ) -> float:
        """
        当前 Layer 的 Prefill 吞吐。
        """

        if self.total_cycles <= 0:
            return 0.0

        return (
            self.token_count
            / self.total_cycles
        )

    # ========================================================
    # Task Query
    # ========================================================

    def task(
        self,
        token_index: int,
        expert_id: int,
        matrix_name: str,
    ) -> ScheduledPrefillTask:

        for task in self.tasks:

            if (
                task.token_index
                == token_index

                and

                task.expert_id
                == expert_id

                and

                task.matrix_name
                == matrix_name
            ):

                return task

        raise PrefillLayerSchedulerError(
            "找不到任务："
            f"token={token_index}, "
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
    仅用于确定性 tie-break：

        gate
        up
        down
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

    raise PrefillLayerSchedulerError(
        "未知 matrix_name："
        f"{matrix_name!r}。"
    )


# ============================================================
# Route 标准化
# ============================================================


def _normalize_prefill_routes(
    *,
    index: RuntimeIndex,

    layer_id: int,

    routed_experts_by_token: Iterable[
        Iterable[int]
    ],
) -> tuple[
    tuple[
        int,
        ...
    ],
    ...
]:
    """
    标准化成：

        token_count × Top-K

    并让 RuntimeIndex 对每个 Token 的
    Top-K 做合法性检查。
    """

    # 顺便检查 layer_id
    index.layer(
        layer_id
    )

    routes = tuple(
        tuple(
            route
        )

        for route
        in routed_experts_by_token
    )

    if not routes:

        raise PrefillLayerSchedulerError(
            "Prefill Layer "
            "至少需要一个 Token。"
        )

    for (
        token_index,
        route,
    ) in enumerate(
        routes
    ):

        try:

            index.resolve_active_expert_ids(
                layer_id=(
                    layer_id
                ),

                routed_expert_ids=(
                    route
                ),
            )

        except ValueError as exc:

            raise PrefillLayerSchedulerError(
                f"Token-{token_index} "
                "Router Route 非法。"
            ) from exc

    return routes


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

    后续 Prefill 58 层模拟：

        Layer-L final state
            ->
        Layer-(L+1) initial state
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

        raise PrefillLayerSchedulerError(
            "initial_active_cube_by_subcube "
            "长度错误："
            f"actual={len(state)}, "
            f"expected="
            f"{index.num_subcubes}。"
        )

    for cube_id in state:

        if cube_id is None:
            continue

        if (
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
        ):

            raise PrefillLayerSchedulerError(
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
    与单 Token layer_scheduler.py
    保持相同语义。

    返回：

        (
            activation_cycles,
            is_switch,
            is_initial_activation
        )

    previous = None：

        默认：
            startup = switch_cycles

    previous == next：

        同一个 Weight-Cube 连续服务下一个 Token，
        不需要再次切换。

    previous != next：

        收取 switch_cycles。
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
        _PrefillTaskSpec
    ],

    current_time: int,

    active_cube_id: (
        int | None
    ),
) -> (
    _PrefillTaskSpec
    | None
):
    """
    Prefill Baseline 的 SC 内调度顺序：

    1. ready_time 更早；

    2. 如果当前已经激活的 Weight-Cube
       还有 Ready Task，
       优先继续当前 Weight-Cube；

       这是 Prefill 批处理中减少重复 Switch
       的关键；

    3. 当前 Token 内的 route_rank；

    4. token_index；

    5. gate -> up -> down；

    6. cube_id。

    --------------------------------------------------------

    例如：

        当前 SC 已激活 E11 gate

        Ready：
            Token-0 E11 gate
            Token-3 E11 gate
            Token-8 E11 gate
            Token-2 E37 up

    那么会优先连续执行
    E11 gate 的 Token 任务，
    中间不重复 Switch。
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
        task: _PrefillTaskSpec,
    ) -> tuple[
        int,
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

            task.token_index,

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


def schedule_prefill_layer(
    *,
    index: RuntimeIndex,

    layer_id: int,

    routed_experts_by_token: Iterable[
        Iterable[int]
    ],

    rules: ExecutionRules | None = None,

    initial_active_cube_by_subcube: (
        Iterable[int | None]
        | None
    ) = None,

    charge_initial_activation: bool = True,
) -> PrefillLayerScheduleResult:
    """
    模拟：

        一个 Prefill Batch
        ×
        一个 MoE Layer

    --------------------------------------------------------

    最关键的区别：

    单 Token：

        completion key
            =
        (expert_id, matrix_name)

    Prefill：

        completion key
            =
        (
            token_index,
            expert_id,
            matrix_name,
        )

    因此：

        Token-A 的 down

    只等待：

        Token-A 自己的 gate/up

    不会错误等待其他 Token。
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

        raise PrefillLayerSchedulerError(
            "ExecutionRules 周期配置非法。"
        )

    if not (
        rules.unlimited_parallel_subcubes
    ):

        raise PrefillLayerSchedulerError(
            "当前 Scheduler 要求 "
            "不同 Sub-Cube 可以完全并行。"
        )

    if not (
        rules
        .one_active_weight_cube_per_subcube
    ):

        raise PrefillLayerSchedulerError(
            "当前 Scheduler 基于 "
            "Sub-Cube 内 Weight-Cube 互斥规则。"
        )

    if (
        rules.cross_subcube_cycles
        != 0
    ):

        raise PrefillLayerSchedulerError(
            "当前 Baseline "
            "跨 Sub-Cube 开销必须为 0。"
        )

    # ========================================================
    # Routes
    # ========================================================

    routes = (
        _normalize_prefill_routes(
            index=index,

            layer_id=(
                layer_id
            ),

            routed_experts_by_token=(
                routed_experts_by_token
            ),
        )
    )

    token_count = len(
        routes
    )

    # ========================================================
    # 每个 Token：
    #
    # Routed Top-K + Shared
    # ========================================================

    active_expert_ids_by_token: list[
        tuple[
            int,
            ...
        ]
    ] = []

    route_rank_by_token: list[
        dict[
            int,
            int,
        ]
    ] = []

    for route in routes:

        active_ids = (
            index.resolve_active_expert_ids(
                layer_id=(
                    layer_id
                ),

                routed_expert_ids=(
                    route
                ),
            )
        )

        active_expert_ids_by_token.append(
            active_ids
        )

        route_rank_by_token.append(
            {
                expert_id: rank

                for (
                    rank,
                    expert_id,
                ) in enumerate(
                    active_ids
                )
            }
        )

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
            _PrefillTaskSpec
        ]
    ] = [
        []
        for _ in range(
            index.num_subcubes
        )
    ]

    # ========================================================
    # t=0：
    #
    # 所有 Token 的 gate / up 都 ready。
    # ========================================================

    for (
        token_index,
        active_ids,
    ) in enumerate(
        active_expert_ids_by_token
    ):

        for expert_id in (
            active_ids
        ):

            expert = (
                index.expert(
                    layer_id,
                    expert_id,
                )
            )

            route_rank = (
                route_rank_by_token[
                    token_index
                ][
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
                    _PrefillTaskSpec(
                        token_index=(
                            token_index
                        ),

                        layer_id=(
                            layer_id
                        ),

                        expert_id=(
                            expert_id
                        ),

                        route_rank=(
                            route_rank
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
        ScheduledPrefillTask
        | None
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
    #     ScheduledPrefillTask,
    # )
    running_heap: list[
        tuple[
            int,
            int,
            int,
            ScheduledPrefillTask,
        ]
    ] = []

    serial = 0

    # ========================================================
    # 完成时间：
    #
    # (
    #     token_index,
    #     expert_id,
    #     matrix_name,
    # )
    # ========================================================

    completion_time: dict[
        tuple[
            int,
            int,
            str,
        ],
        int,
    ] = {}

    # ========================================================
    # 某个 Token / Expert 的 down
    # 是否已经创建。
    # ========================================================

    down_created: set[
        tuple[
            int,
            int,
        ]
    ] = set()

    completed_tasks: list[
        ScheduledPrefillTask
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

    # ========================================================
    # 每个 Token：
    #
    # 8 Routed + Shared
    # × 3 Matrix
    #
    # 一般：
    #
    # 27 tasks / token
    # ========================================================

    expected_task_count = sum(
        len(
            active_ids
        )
        * 3

        for active_ids
        in active_expert_ids_by_token
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

            # =================================================
            # SC 正忙
            # =================================================

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

                raise PrefillLayerSchedulerError(
                    "任务在 ready_time "
                    "之前被执行。"
                )

            scheduled = (
                ScheduledPrefillTask(
                    token_index=(
                        spec.token_index
                    ),

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
        # 当前时刻：
        # 所有空闲 SC 各尽量启动一个任务。
        # ----------------------------------------------------

        dispatch_ready_tasks()

        # ----------------------------------------------------
        # 没有运行任务。
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

            raise PrefillLayerSchedulerError(
                "调度死锁："
                "没有运行任务，"
                "也没有未来 Ready Task。"
            )

        # ----------------------------------------------------
        # 推进到下一次任务完成。
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

            raise PrefillLayerSchedulerError(
                "内部时间状态错误。"
            )

        current_time = (
            next_finish_time
        )

        finished_now: list[
            tuple[
                int,
                ScheduledPrefillTask,
            ]
        ] = []

        # ----------------------------------------------------
        # 同一时刻多个 SC 可能同时完成。
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

                raise PrefillLayerSchedulerError(
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
                task.token_index,
                task.expert_id,
                task.matrix_name,
            )

            if (
                key
                in completion_time
            ):

                raise PrefillLayerSchedulerError(
                    "同一个 Token/Expert/Matrix "
                    "任务完成了两次。"
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
        # 检查哪些：
        #
        #     Token / Expert
        #
        # 的 down 现在可以 Ready。
        # ====================================================

        affected_pairs = {
            (
                task.token_index,
                task.expert_id,
            )

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

        for (
            token_index,
            expert_id,
        ) in sorted(
            affected_pairs,

            key=lambda pair: (
                route_rank_by_token[
                    pair[0]
                ][
                    pair[1]
                ],
                pair[0],
                pair[1],
            ),
        ):

            pair_key = (
                token_index,
                expert_id,
            )

            if (
                pair_key
                in down_created
            ):

                continue

            gate_key = (
                token_index,
                expert_id,
                MATRIX_GATE,
            )

            up_key = (
                token_index,
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
                index.expert(
                    layer_id,
                    expert_id,
                )
            )

            down_location = (
                expert.down
            )

            ready_by_sc[
                down_location
                .subcube_id
            ].append(
                _PrefillTaskSpec(
                    token_index=(
                        token_index
                    ),

                    layer_id=(
                        layer_id
                    ),

                    expert_id=(
                        expert_id
                    ),

                    route_rank=(
                        route_rank_by_token[
                            token_index
                        ][
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
                pair_key
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

        raise PrefillLayerSchedulerError(
            "完成任务数错误："
            f"actual="
            f"{len(completion_time)}, "
            f"expected="
            f"{expected_task_count}。"
        )

    expected_down_count = sum(
        len(
            active_ids
        )

        for active_ids
        in active_expert_ids_by_token
    )

    if (
        len(
            down_created
        )
        != expected_down_count
    ):

        raise PrefillLayerSchedulerError(
            "不是所有 Token/Expert "
            "都成功创建了 down。"
        )

    # ========================================================
    # 检查每个 Token / Expert：
    #
    # down.ready_time
    # =
    # max(
    #     gate.finish,
    #     up.finish,
    # )
    # ========================================================

    task_lookup = {
        (
            task.token_index,
            task.expert_id,
            task.matrix_name,
        ): task

        for task
        in completed_tasks
    }

    if (
        len(
            task_lookup
        )
        != expected_task_count
    ):

        raise PrefillLayerSchedulerError(
            "Task Lookup 数量错误。"
        )

    for (
        token_index,
        active_ids,
    ) in enumerate(
        active_expert_ids_by_token
    ):

        for expert_id in (
            active_ids
        ):

            gate_task = (
                task_lookup[
                    (
                        token_index,
                        expert_id,
                        MATRIX_GATE,
                    )
                ]
            )

            up_task = (
                task_lookup[
                    (
                        token_index,
                        expert_id,
                        MATRIX_UP,
                    )
                ]
            )

            down_task = (
                task_lookup[
                    (
                        token_index,
                        expert_id,
                        MATRIX_DOWN,
                    )
                ]
            )

            expected_ready = max(
                gate_task.finish_time,
                up_task.finish_time,
            )

            if (
                down_task.ready_time
                != expected_ready
            ):

                raise PrefillLayerSchedulerError(
                    f"Token-{token_index} "
                    f"Expert-{expert_id} "
                    "down ready_time 错误。"
                )

            if (
                down_task.dispatch_time
                < expected_ready
            ):

                raise PrefillLayerSchedulerError(
                    f"Token-{token_index} "
                    f"Expert-{expert_id} "
                    "down 在 gate/up "
                    "完成前执行。"
                )

    # ========================================================
    # 一个 Token 的每个 active Expert
    # 必须恰好 gate/up/down 三个任务。
    # ========================================================

    for (
        token_index,
        active_ids,
    ) in enumerate(
        active_expert_ids_by_token
    ):

        for expert_id in (
            active_ids
        ):

            for matrix_name in (
                MATRIX_GATE,
                MATRIX_UP,
                MATRIX_DOWN,
            ):

                key = (
                    token_index,
                    expert_id,
                    matrix_name,
                )

                if key not in task_lookup:

                    raise PrefillLayerSchedulerError(
                        "缺少任务："
                        f"token={token_index}, "
                        f"expert={expert_id}, "
                        f"matrix={matrix_name}。"
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
        PrefillSubcubeLayerStats
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
            PrefillSubcubeLayerStats(
                subcube_id=(
                    sc
                ),

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
            task.token_index,
            task.expert_id,
            _matrix_priority(
                task.matrix_name
            ),
        )
    )

    return (
        PrefillLayerScheduleResult(
            layer_id=(
                layer_id
            ),

            token_count=(
                token_count
            ),

            routed_experts_by_token=(
                routes
            ),

            active_expert_ids_by_token=tuple(
                active_expert_ids_by_token
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
    )


# ============================================================
# 输出
# ============================================================


def print_prefill_layer_schedule_summary(
    result: PrefillLayerScheduleResult,
) -> None:
    """
    打印一个 Prefill Batch
    在单个 Layer 的调度摘要。
    """

    print(
        "\n"
        "========== Prefill Layer Schedule =========="
    )

    print(
        f"Layer："
        f"{result.layer_id}"
    )

    print(
        f"Tokens："
        f"{result.token_count}"
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
        f"Cycles / Token："
        f"{result.cycles_per_token:.4f}"
    )

    print(
        f"Tokens / Cycle："
        f"{result.tokens_per_cycle:.6f}"
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

        --route 0,1,2,3,4,5,6,7

    Smoke Test 时：

        把同一条 Top-8
        临时复制给多个 Token。

    正式 Prefill 实验不会这样做，
    后面直接读取真实 Trace。
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
            "route 必须是逗号分隔整数。"
        ) from exc


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "模拟一个 Prefill Batch "
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
        "--tokens",
        type=int,
        default=17,

        help=(
            "Smoke Test 的 Batch Token 数。"
        ),
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
            "Smoke Test 用 Top-K Route。"
            "会临时复制给全部 Token。"
        ),
    )

    parser.add_argument(
        "--no-initial-activation-cost",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    if (
        args.tokens
        <= 0
    ):

        raise PrefillLayerSchedulerError(
            "--tokens 必须大于 0。"
        )

    index = (
        load_runtime_index(
            args.mapping
        )
    )

    # ========================================================
    # Smoke Test：
    #
    # 临时让所有 Token 使用同一条 Route。
    #
    # 这样非常适合检查：
    #
    # 同一个 Weight-Cube
    # 是否能够：
    #
    #     Switch 一次
    #     连续 Compute 多个 Token。
    # ========================================================

    routes = tuple(
        args.route

        for _ in range(
            args.tokens
        )
    )

    result = (
        schedule_prefill_layer(
            index=index,

            layer_id=(
                args.layer
            ),

            routed_experts_by_token=(
                routes
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

    print_prefill_layer_schedule_summary(
        result
    )


if __name__ == "__main__":
    main()
