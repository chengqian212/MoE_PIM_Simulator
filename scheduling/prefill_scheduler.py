"""
第六步：完整 Prefill Batch 的 58 层 MoE 调度。

上一层：

    scheduling/prefill_layer_scheduler.py

负责：

    一个 Prefill Batch
    ×
    一个 MoE Layer

本文件进一步负责：

    一个 Prefill Batch
    ×
    全部 58 个 MoE Layer

------------------------------------------------------------

输入数据 shape：

    Token
        ->
    Layer
        ->
    Routed Top-K

即：

    token_count × 58 × 8

这与：

    scheduling/prefill_workload.py

中的：

    TraceSegmentBatch.routed_experts_by_token

完全一致。

------------------------------------------------------------

当前 Baseline：

1. Prefill 按 Layer 严格顺序执行：

       整个 Batch 完成 Layer-0
                    ↓
       整个 Batch 完成 Layer-1
                    ↓
                  ...
                    ↓
       整个 Batch 完成 Layer-57

   不是：

       Token-0 跑完 58 层
       再跑 Token-1。

2. 同一层内部由：

       prefill_layer_scheduler.py

   完成：

       所有 Token gate/up 初始 ready

       gate(token,e) || up(token,e)
                    ↓
                down(token,e)

3. 不同 Sub-Cube 并行；

4. 同一 Sub-Cube 串行；

5. 同一个 Weight-Cube 连续服务多个 Token：

       只收一次切换开销
       但每个 Token 都仍然需要一次 compute；

6. Sub-Cube 当前 active Weight-Cube 状态
   在 Layer 之间继承：

       Layer-L final state
            ↓
       Layer-(L+1) initial state

7. 当前不做跨 Layer pipeline。

------------------------------------------------------------

注意：

这里统计的是：

    MoE Expert 部分的 Prefill 周期

不包含：

    Attention
    KV Cache
    Embedding
    LM Head
    其他非 MoE 模块

因此最终应称为：

    MoE Prefill Latency

而不是完整模型 TTFT。
"""

from __future__ import annotations

import argparse

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


from config import (
    ExecutionRules,
)

from scheduling.prefill_layer_scheduler import (
    PrefillLayerScheduleResult,
    schedule_prefill_layer,
)

from scheduling.runtime_index import (
    DEFAULT_MAPPING_PATH,
    RuntimeIndex,
    load_runtime_index,
)


# ============================================================
# 异常
# ============================================================


class PrefillSchedulerError(
    ValueError
):
    """完整 Prefill Batch 调度失败。"""


# ============================================================
# 一层在完整 Prefill 时间轴中的位置
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PrefillLayerExecution:
    """
    一个 Layer 在完整 Prefill Batch
    时间轴中的执行结果。

    layer_result 内部时间从 0 开始。

    这里额外记录：

        global_start_time
        global_finish_time

    因为 58 层当前严格串行。
    """

    layer_id: int

    global_start_time: int

    global_finish_time: int

    layer_result: (
        PrefillLayerScheduleResult
    )

    def __post_init__(
        self,
    ) -> None:

        if self.layer_id < 0:

            raise PrefillSchedulerError(
                "layer_id 不能为负数。"
            )

        if (
            self.global_start_time
            < 0
        ):

            raise PrefillSchedulerError(
                "global_start_time "
                "不能为负数。"
            )

        if (
            self.global_finish_time
            < self.global_start_time
        ):

            raise PrefillSchedulerError(
                "global_finish_time "
                "不能早于 start。"
            )

        if (
            self.layer_result.layer_id
            != self.layer_id
        ):

            raise PrefillSchedulerError(
                "PrefillLayerExecution "
                "与 PrefillLayerScheduleResult "
                "layer_id 不一致。"
            )

        actual_cycles = (
            self.global_finish_time
            - self.global_start_time
        )

        if (
            actual_cycles
            != self.layer_result
            .total_cycles
        ):

            raise PrefillSchedulerError(
                f"Layer-{self.layer_id} "
                "全局时间长度与 "
                "Layer Cycles 不一致。"
            )

    @property
    def cycles(
        self,
    ) -> int:

        return (
            self.layer_result
            .total_cycles
        )


# ============================================================
# 一个 SC 在完整 Prefill 中的汇总
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PrefillSubcubeBatchStats:
    """
    一个 Sub-Cube 跨 58 层的汇总统计。

    由于当前 58 层严格串行，

        busy_cycles / total_prefill_cycles

    可以作为该 SC 在完整 Prefill 时间轴上的
    利用率。
    """

    subcube_id: int

    task_count: int

    compute_cycles: int

    activation_cycles: int

    switch_count: int

    initial_activation_count: int

    busy_cycles: int

    wait_cycles: int

    max_task_wait_cycles: int

    def utilization(
        self,
        total_prefill_cycles: int,
    ) -> float:

        if (
            total_prefill_cycles
            <= 0
        ):

            return 0.0

        return (
            self.busy_cycles
            / total_prefill_cycles
        )


# ============================================================
# 完整 Prefill 结果
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PrefillScheduleResult:
    """
    一个完整 Prefill Batch
    经过全部 MoE Layer 的结果。
    """

    # ========================================================
    # Route
    #
    # shape:
    #
    # token_count × 58 × Top-K
    # ========================================================

    routed_experts_by_token: tuple[
        tuple[
            tuple[
                int,
                ...
            ],
            ...
        ],
        ...
    ]

    # ========================================================
    # Layer Results
    # ========================================================

    layers: tuple[
        PrefillLayerExecution,
        ...
    ]

    # ========================================================
    # Runtime State
    # ========================================================

    initial_active_cube_by_subcube: tuple[
        int | None,
        ...
    ]

    final_active_cube_by_subcube: tuple[
        int | None,
        ...
    ]

    # ========================================================
    # SC Aggregate
    # ========================================================

    subcube_stats: tuple[
        PrefillSubcubeBatchStats,
        ...
    ]

    # ========================================================
    # 总时间
    # ========================================================

    total_cycles: int

    # ========================================================
    # 基础属性
    # ========================================================

    @property
    def token_count(
        self,
    ) -> int:

        return len(
            self.routed_experts_by_token
        )

    @property
    def num_layers(
        self,
    ) -> int:

        return len(
            self.layers
        )

    @property
    def total_tasks(
        self,
    ) -> int:

        return sum(
            execution
            .layer_result
            .task_count

            for execution
            in self.layers
        )

    @property
    def total_switches(
        self,
    ) -> int:

        return sum(
            execution
            .layer_result
            .switch_count

            for execution
            in self.layers
        )

    @property
    def total_initial_activations(
        self,
    ) -> int:

        return sum(
            execution
            .layer_result
            .initial_activation_count

            for execution
            in self.layers
        )

    @property
    def total_activation_overhead_cycles(
        self,
    ) -> int:

        return sum(
            execution
            .layer_result
            .activation_overhead_cycles

            for execution
            in self.layers
        )

    @property
    def total_compute_work_cycles(
        self,
    ) -> int:
        """
        所有 SC 上 compute work 的总和。

        不是 Prefill latency。
        """

        return sum(
            execution
            .layer_result
            .compute_cycles

            for execution
            in self.layers
        )

    @property
    def total_busy_cycles(
        self,
    ) -> int:
        """
        所有 SC 的：

            compute
            +
            activation/switch

        work 总和。

        不是 latency。
        """

        return sum(
            execution
            .layer_result
            .busy_cycles

            for execution
            in self.layers
        )

    @property
    def total_wait_cycles(
        self,
    ) -> int:

        return sum(
            execution
            .layer_result
            .wait_cycles

            for execution
            in self.layers
        )

    @property
    def max_task_wait_cycles(
        self,
    ) -> int:

        return max(
            (
                execution
                .layer_result
                .max_task_wait_cycles

                for execution
                in self.layers
            ),
            default=0,
        )

    # ========================================================
    # Prefill 指标
    # ========================================================

    @property
    def cycles_per_input_token(
        self,
    ) -> float:
        """
        完整 58 层 MoE Prefill latency
        除以输入 Token 数。

        这是归一化指标。
        """

        if self.token_count <= 0:
            return 0.0

        return (
            self.total_cycles
            / self.token_count
        )

    @property
    def input_tokens_per_cycle(
        self,
    ) -> float:
        """
        当前 MoE Expert 部分的
        Prefill 吞吐。
        """

        if self.total_cycles <= 0:
            return 0.0

        return (
            self.token_count
            / self.total_cycles
        )

    # ========================================================
    # Layer 指标
    # ========================================================

    @property
    def max_layer_cycles(
        self,
    ) -> int:

        return max(
            (
                layer.cycles
                for layer
                in self.layers
            ),
            default=0,
        )

    @property
    def min_layer_cycles(
        self,
    ) -> int:

        return min(
            (
                layer.cycles
                for layer
                in self.layers
            ),
            default=0,
        )

    @property
    def average_layer_cycles(
        self,
    ) -> float:

        if not self.layers:
            return 0.0

        return float(
            mean(
                layer.cycles
                for layer
                in self.layers
            )
        )

    # ========================================================
    # Query
    # ========================================================

    def layer(
        self,
        layer_id: int,
    ) -> PrefillLayerExecution:

        if not (
            0
            <= layer_id
            < len(self.layers)
        ):

            raise PrefillSchedulerError(
                f"layer_id={layer_id} "
                "超出范围。"
            )

        execution = (
            self.layers[
                layer_id
            ]
        )

        if (
            execution.layer_id
            != layer_id
        ):

            raise PrefillSchedulerError(
                "layers 没有按照 "
                "layer_id 排序。"
            )

        return execution

    def subcube(
        self,
        subcube_id: int,
    ) -> PrefillSubcubeBatchStats:

        if not (
            0
            <= subcube_id
            < len(self.subcube_stats)
        ):

            raise PrefillSchedulerError(
                f"subcube_id={subcube_id} "
                "超出范围。"
            )

        stat = (
            self.subcube_stats[
                subcube_id
            ]
        )

        if (
            stat.subcube_id
            != subcube_id
        ):

            raise PrefillSchedulerError(
                "subcube_stats 没有按照 "
                "subcube_id 排序。"
            )

        return stat


# ============================================================
# Route 标准化
# ============================================================


def normalize_prefill_routes(
    *,
    index: RuntimeIndex,

    routed_experts_by_token: Iterable[
        Iterable[
            Iterable[int]
        ]
    ],
) -> tuple[
    tuple[
        tuple[
            int,
            ...
        ],
        ...
    ],
    ...
]:
    """
    输入：

        Token
            ->
        Layer
            ->
        Routed Top-K

    shape：

        B × 58 × 8

    并利用 RuntimeIndex
    对每个 Token、每一层的 Route
    做合法性检查。
    """

    routes = tuple(
        tuple(
            tuple(
                layer_route
            )

            for layer_route
            in token_routes
        )

        for token_routes
        in routed_experts_by_token
    )

    # ========================================================
    # 至少一个 Token
    # ========================================================

    if not routes:

        raise PrefillSchedulerError(
            "Prefill Batch "
            "至少需要一个 Token。"
        )

    # ========================================================
    # 每个 Token 必须有完整 58 层
    # ========================================================

    for (
        token_index,
        token_routes,
    ) in enumerate(
        routes
    ):

        if (
            len(token_routes)
            != index.num_layers
        ):

            raise PrefillSchedulerError(
                f"Token-{token_index} "
                "Layer 数错误："
                f"actual="
                f"{len(token_routes)}, "
                f"expected="
                f"{index.num_layers}。"
            )

        # ====================================================
        # 每层 Top-K 合法性
        # ====================================================

        for (
            layer_id,
            route,
        ) in enumerate(
            token_routes
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

                raise PrefillSchedulerError(
                    f"Token-{token_index} "
                    f"Layer-{layer_id} "
                    "Router Route 非法。"
                ) from exc

    return routes


# ============================================================
# Initial State
# ============================================================


def normalize_initial_state(
    *,
    index: RuntimeIndex,

    initial_active_cube_by_subcube: (
        Iterable[
            int | None
        ]
        | None
    ),
) -> tuple[
    int | None,
    ...
]:
    """
    Prefill 开始前：

        每个 SC 当前激活的 Weight-Cube。

    单独评估一个 Prefill：

        一般 None × 16

    如果以后模拟连续请求：

        可以显式传入已有状态。
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

        raise PrefillSchedulerError(
            "initial_active_cube_by_subcube "
            "长度错误："
            f"actual={len(state)}, "
            f"expected="
            f"{index.num_subcubes}。"
        )

    for value in state:

        if value is None:
            continue

        if (
            not isinstance(
                value,
                int,
            )

            or

            isinstance(
                value,
                bool,
            )

            or

            value < 0
        ):

            raise PrefillSchedulerError(
                "Initial active cube "
                "必须为 None 或非负整数。"
            )

    return state


# ============================================================
# SC 汇总
# ============================================================


def _aggregate_subcube_stats(
    *,
    index: RuntimeIndex,

    layers: tuple[
        PrefillLayerExecution,
        ...
    ],
) -> tuple[
    PrefillSubcubeBatchStats,
    ...
]:

    results: list[
        PrefillSubcubeBatchStats
    ] = []

    for sc in range(
        index.num_subcubes
    ):

        task_count = 0
        compute_cycles = 0
        activation_cycles = 0
        switch_count = 0
        initial_activation_count = 0
        busy_cycles = 0
        wait_cycles = 0
        max_task_wait_cycles = 0

        for execution in layers:

            stat = (
                execution
                .layer_result
                .subcube_stats[
                    sc
                ]
            )

            if (
                stat.subcube_id
                != sc
            ):

                raise PrefillSchedulerError(
                    f"Layer-"
                    f"{execution.layer_id} "
                    f"SC-{sc} "
                    "统计顺序错误。"
                )

            task_count += (
                stat.task_count
            )

            compute_cycles += (
                stat.compute_cycles
            )

            activation_cycles += (
                stat.activation_cycles
            )

            switch_count += (
                stat.switch_count
            )

            initial_activation_count += (
                stat.initial_activation_count
            )

            busy_cycles += (
                stat.busy_cycles
            )

            wait_cycles += (
                stat.wait_cycles
            )

            # 当前单层结果的 max wait
            # 是全 SC 的最大值。
            #
            # 这里为了得到某个 SC 自己的 max wait，
            # 直接从该层 task 中筛当前 SC。
            sc_layer_max_wait = max(
                (
                    task.wait_cycles

                    for task
                    in execution
                    .layer_result
                    .tasks

                    if (
                        task.subcube_id
                        == sc
                    )
                ),
                default=0,
            )

            max_task_wait_cycles = max(
                max_task_wait_cycles,
                sc_layer_max_wait,
            )

        results.append(
            PrefillSubcubeBatchStats(
                subcube_id=sc,

                task_count=(
                    task_count
                ),

                compute_cycles=(
                    compute_cycles
                ),

                activation_cycles=(
                    activation_cycles
                ),

                switch_count=(
                    switch_count
                ),

                initial_activation_count=(
                    initial_activation_count
                ),

                busy_cycles=(
                    busy_cycles
                ),

                wait_cycles=(
                    wait_cycles
                ),

                max_task_wait_cycles=(
                    max_task_wait_cycles
                ),
            )
        )

    return tuple(
        results
    )


# ============================================================
# 主函数
# ============================================================


def schedule_prefill_batch(
    *,
    index: RuntimeIndex,

    routed_experts_by_token: Iterable[
        Iterable[
            Iterable[int]
        ]
    ],

    rules: ExecutionRules | None = None,

    initial_active_cube_by_subcube: (
        Iterable[
            int | None
        ]
        | None
    ) = None,

    charge_initial_activation: bool = True,
) -> PrefillScheduleResult:
    """
    模拟一个完整 Prefill Batch
    经过全部 MoE Layer。

    当前严格：

        Batch 全部 Token 完成 Layer-0
                    ↓
        Batch 全部 Token 完成 Layer-1
                    ↓
                  ...
                    ↓
        Batch 全部 Token 完成 Layer-57

    不做跨 Layer pipeline。
    """

    if rules is None:

        rules = (
            ExecutionRules()
        )

    # ========================================================
    # Route
    # ========================================================

    routes = (
        normalize_prefill_routes(
            index=index,

            routed_experts_by_token=(
                routed_experts_by_token
            ),
        )
    )

    token_count = len(
        routes
    )

    # ========================================================
    # Initial State
    # ========================================================

    initial_state = (
        normalize_initial_state(
            index=index,

            initial_active_cube_by_subcube=(
                initial_active_cube_by_subcube
            ),
        )
    )

    current_state = (
        initial_state
    )

    # ========================================================
    # Global Timeline
    # ========================================================

    global_time = 0

    layer_executions: list[
        PrefillLayerExecution
    ] = []

    # ========================================================
    # 58 层严格顺序执行
    # ========================================================

    for layer_id in range(
        index.num_layers
    ):

        # ====================================================
        # 从：
        #
        #     Token × Layer × Top-K
        #
        # 提取当前层：
        #
        #     Token × Top-K
        # ====================================================

        layer_routes = tuple(
            token_routes[
                layer_id
            ]

            for token_routes
            in routes
        )

        # ====================================================
        # 当前 Layer 整个 Batch 调度
        # ====================================================

        layer_result = (
            schedule_prefill_layer(
                index=index,

                layer_id=(
                    layer_id
                ),

                routed_experts_by_token=(
                    layer_routes
                ),

                rules=(
                    rules
                ),

                initial_active_cube_by_subcube=(
                    current_state
                ),

                charge_initial_activation=(
                    charge_initial_activation
                ),
            )
        )

        # ====================================================
        # Layer 必须仍然是同一个 Batch
        # ====================================================

        if (
            layer_result.token_count
            != token_count
        ):

            raise PrefillSchedulerError(
                f"Layer-{layer_id} "
                "token_count 与 Batch 不一致。"
            )

        # ====================================================
        # Global Time
        # ====================================================

        global_start = (
            global_time
        )

        global_finish = (
            global_start
            + layer_result
            .total_cycles
        )

        layer_executions.append(
            PrefillLayerExecution(
                layer_id=(
                    layer_id
                ),

                global_start_time=(
                    global_start
                ),

                global_finish_time=(
                    global_finish
                ),

                layer_result=(
                    layer_result
                ),
            )
        )

        # ====================================================
        # 当前 Layer 的 active WC 状态
        # 传给下一层。
        # ====================================================

        current_state = (
            layer_result
            .final_active_cube_by_subcube
        )

        # ====================================================
        # 必须整层结束后
        # 才进入下一层。
        # ====================================================

        global_time = (
            global_finish
        )

    layer_tuple = tuple(
        layer_executions
    )

    subcube_stats = (
        _aggregate_subcube_stats(
            index=index,

            layers=(
                layer_tuple
            ),
        )
    )

    result = (
        PrefillScheduleResult(
            routed_experts_by_token=(
                routes
            ),

            layers=(
                layer_tuple
            ),

            initial_active_cube_by_subcube=(
                initial_state
            ),

            final_active_cube_by_subcube=(
                current_state
            ),

            subcube_stats=(
                subcube_stats
            ),

            total_cycles=(
                global_time
            ),
        )
    )

    validate_prefill_schedule(
        result=result,
        index=index,
    )

    return result


# ============================================================
# 最终检查
# ============================================================


def validate_prefill_schedule(
    *,
    result: PrefillScheduleResult,

    index: RuntimeIndex,
) -> None:
    """
    检查完整 Prefill：

    1. Token 数；
    2. Layer 数；
    3. Layer 顺序；
    4. Layer 时间连续；
    5. SC active state 连续；
    6. total_cycles；
    7. 每层 task 数；
    8. 总 task 数；
    9. SC 汇总与 Layer 汇总一致。
    """

    # ========================================================
    # Token
    # ========================================================

    if (
        result.token_count
        <= 0
    ):

        raise PrefillSchedulerError(
            "Prefill token_count "
            "必须大于 0。"
        )

    # ========================================================
    # Layer
    # ========================================================

    if (
        result.num_layers
        != index.num_layers
    ):

        raise PrefillSchedulerError(
            "Prefill Layer 数错误。"
        )

    # ========================================================
    # Layer timeline
    # ========================================================

    expected_start = 0

    previous_final_state = (
        result
        .initial_active_cube_by_subcube
    )

    for layer_id in range(
        index.num_layers
    ):

        execution = (
            result.layers[
                layer_id
            ]
        )

        if (
            execution.layer_id
            != layer_id
        ):

            raise PrefillSchedulerError(
                "Layer 顺序错误。"
            )

        if (
            execution.global_start_time
            != expected_start
        ):

            raise PrefillSchedulerError(
                f"Layer-{layer_id} "
                "没有紧接上一层执行。"
            )

        # ====================================================
        # Active state continuity
        # ====================================================

        if (
            execution
            .layer_result
            .initial_active_cube_by_subcube
            != previous_final_state
        ):

            raise PrefillSchedulerError(
                f"Layer-{layer_id} "
                "初始 Weight-Cube 状态 "
                "没有继承上一层。"
            )

        previous_final_state = (
            execution
            .layer_result
            .final_active_cube_by_subcube
        )

        expected_start = (
            execution
            .global_finish_time
        )

    # ========================================================
    # Final state
    # ========================================================

    if (
        previous_final_state
        != result
        .final_active_cube_by_subcube
    ):

        raise PrefillSchedulerError(
            "最终 active state 错误。"
        )

    # ========================================================
    # Total cycles
    # ========================================================

    if (
        expected_start
        != result.total_cycles
    ):

        raise PrefillSchedulerError(
            "Prefill total_cycles "
            "与 Layer 时间轴不一致。"
        )

    expected_cycle_sum = sum(
        execution
        .layer_result
        .total_cycles

        for execution
        in result.layers
    )

    if (
        expected_cycle_sum
        != result.total_cycles
    ):

        raise PrefillSchedulerError(
            "Prefill total_cycles "
            "不等于 58 层周期之和。"
        )

    # ========================================================
    # 每层任务数
    #
    # B Token
    # ×
    # (Top-K + Shared)
    # ×
    # 3 Matrix
    # ========================================================

    active_experts_per_token = (
        index.model_config
        .experts_per_token

        +

        int(
            index.model_config
            .include_shared_expert
        )
    )

    expected_tasks_per_layer = (
        result.token_count
        * active_experts_per_token
        * 3
    )

    for execution in (
        result.layers
    ):

        if (
            execution
            .layer_result
            .task_count
            != expected_tasks_per_layer
        ):

            raise PrefillSchedulerError(
                f"Layer-"
                f"{execution.layer_id} "
                "任务数量错误："
                f"actual="
                f"{execution.layer_result.task_count}, "
                f"expected="
                f"{expected_tasks_per_layer}。"
            )

    expected_total_tasks = (
        expected_tasks_per_layer
        * index.num_layers
    )

    if (
        result.total_tasks
        != expected_total_tasks
    ):

        raise PrefillSchedulerError(
            "完整 Prefill "
            "任务总数错误："
            f"actual="
            f"{result.total_tasks}, "
            f"expected="
            f"{expected_total_tasks}。"
        )

    # ========================================================
    # SC aggregate
    # ========================================================

    if (
        len(
            result.subcube_stats
        )
        != index.num_subcubes
    ):

        raise PrefillSchedulerError(
            "subcube_stats 数量错误。"
        )

    aggregate_tasks = sum(
        stat.task_count
        for stat
        in result.subcube_stats
    )

    if (
        aggregate_tasks
        != result.total_tasks
    ):

        raise PrefillSchedulerError(
            "SC 汇总 task_count "
            "与完整 Prefill 不一致。"
        )

    aggregate_compute = sum(
        stat.compute_cycles
        for stat
        in result.subcube_stats
    )

    if (
        aggregate_compute
        != result.total_compute_work_cycles
    ):

        raise PrefillSchedulerError(
            "SC 汇总 compute_cycles "
            "与 Layer 汇总不一致。"
        )

    aggregate_activation = sum(
        stat.activation_cycles
        for stat
        in result.subcube_stats
    )

    if (
        aggregate_activation
        != result
        .total_activation_overhead_cycles
    ):

        raise PrefillSchedulerError(
            "SC 汇总 activation_cycles "
            "与 Layer 汇总不一致。"
        )

    aggregate_busy = sum(
        stat.busy_cycles
        for stat
        in result.subcube_stats
    )

    if (
        aggregate_busy
        != result.total_busy_cycles
    ):

        raise PrefillSchedulerError(
            "SC 汇总 busy_cycles "
            "与 Layer 汇总不一致。"
        )

    aggregate_wait = sum(
        stat.wait_cycles
        for stat
        in result.subcube_stats
    )

    if (
        aggregate_wait
        != result.total_wait_cycles
    ):

        raise PrefillSchedulerError(
            "SC 汇总 wait_cycles "
            "与 Layer 汇总不一致。"
        )


# ============================================================
# 输出
# ============================================================


def print_prefill_schedule_summary(
    result: PrefillScheduleResult,
    *,
    top_layers: int = 10,
) -> None:
    """
    打印完整 58 层 Prefill 摘要。
    """

    print(
        "\n"
        "========== Prefill Schedule =========="
    )

    print(
        f"Input Tokens："
        f"{result.token_count}"
    )

    print(
        f"MoE Layers："
        f"{result.num_layers}"
    )

    print(
        f"Total Tasks："
        f"{result.total_tasks}"
    )

    print(
        f"MoE Prefill Total Cycles："
        f"{result.total_cycles}"
    )

    print(
        f"Cycles / Input Token："
        f"{result.cycles_per_input_token:.4f}"
    )

    print(
        f"Input Tokens / Cycle："
        f"{result.input_tokens_per_cycle:.6f}"
    )

    print(
        f"Total Weight-Cube Switches："
        f"{result.total_switches}"
    )

    print(
        f"Initial Activations："
        f"{result.total_initial_activations}"
    )

    print(
        "Activation/Switch "
        "Overhead Work："
        f"{result.total_activation_overhead_cycles}"
    )

    print(
        f"Compute Work："
        f"{result.total_compute_work_cycles}"
    )

    print(
        f"Total SC Busy Work："
        f"{result.total_busy_cycles}"
    )

    print(
        f"Total Task Wait Cycles："
        f"{result.total_wait_cycles}"
    )

    print(
        f"Max Task Wait："
        f"{result.max_task_wait_cycles}"
    )

    print(
        f"Layer Cycles Range："
        f"{result.min_layer_cycles}"
        " ~ "
        f"{result.max_layer_cycles}"
    )

    print(
        f"Average Layer Cycles："
        f"{result.average_layer_cycles:.4f}"
    )

    # ========================================================
    # Slow Layers
    # ========================================================

    if top_layers > 0:

        ranked_layers = sorted(
            result.layers,

            key=lambda execution: (
                -execution.cycles,
                execution.layer_id,
            ),
        )

        print(
            "\n"
            f"Top-{min(top_layers, len(ranked_layers))} "
            "Slowest Layers："
        )

        for execution in (
            ranked_layers[
                :top_layers
            ]
        ):

            layer_result = (
                execution
                .layer_result
            )

            print(
                f"  L{execution.layer_id}: "
                f"cycles="
                f"{execution.cycles}, "
                f"switch="
                f"{layer_result.switch_count}, "
                f"wait="
                f"{layer_result.wait_cycles}"
            )

    # ========================================================
    # SC Stats
    # ========================================================

    print(
        "\nSub-Cube Aggregate Stats："
    )

    for stat in (
        result.subcube_stats
    ):

        utilization = (
            stat.utilization(
                result.total_cycles
            )
        )

        print(
            f"  SC-{stat.subcube_id}: "
            f"tasks="
            f"{stat.task_count}, "
            f"busy="
            f"{stat.busy_cycles}, "
            f"switch="
            f"{stat.switch_count}, "
            f"initial="
            f"{stat.initial_activation_count}, "
            f"wait="
            f"{stat.wait_cycles}, "
            f"max_wait="
            f"{stat.max_task_wait_cycles}, "
            f"util="
            f"{utilization:.2%}"
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

        同一条 route
        复制给：

            所有 Token
            ×
            所有 58 Layer

    正式实验不会这样做，
    正式实验直接使用 segment0
    的真实 Route。
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
                "模拟一个完整 Prefill Batch "
                "通过全部 58 个 MoE Layer。"
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
        "--tokens",
        type=int,
        default=17,

        help=(
            "Smoke Test 的 Prefill Token 数。"
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
            "会复制给全部 Token 和全部 Layer。"
        ),
    )

    parser.add_argument(
        "--top-layers",
        type=int,
        default=10,
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

        raise PrefillSchedulerError(
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
    # Token × Layer × Top-K
    #
    # 所有 Token / Layer 暂时使用相同 Route。
    #
    # 仅用于验证 58 层串联是否正确。
    # ========================================================

    routes = tuple(
        tuple(
            args.route

            for _ in range(
                index.num_layers
            )
        )

        for _ in range(
            args.tokens
        )
    )

    result = (
        schedule_prefill_batch(
            index=index,

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

    print_prefill_schedule_summary(
        result,

        top_layers=(
            args.top_layers
        ),
    )


if __name__ == "__main__":
    main()
