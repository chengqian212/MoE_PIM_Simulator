"""
第五步：完整 Token 的 58 层 MoE 调度。

上一层：

    scheduling/layer_scheduler.py

负责模拟：

    一个 token
    ×
    一个 MoE Layer

本文件进一步模拟：

    一个 token
    ×
    全部 58 个 MoE Layer

------------------------------------------------------------

当前 Baseline：

1. Layer 严格顺序执行：

       Layer-0
           ↓
       Layer-1
           ↓
       ...
           ↓
       Layer-57

   当前不考虑跨 Layer pipeline。

2. Sub-Cube 的 active Weight-Cube 状态
   在 Layer 之间保留。

   即：

       Layer-L final state
           ↓
       Layer-(L+1) initial state

3. 每一层内部继续采用：

       gate || up
             ↓
           down

   并按照 Expert 自身依赖动态触发 down。

4. Shared Expert 自动加入。

Router 输入只需要：

    每层 Top-8 Routed Expert。

------------------------------------------------------------

输入形式：

    routed_experts_by_layer = (
        (e0, e1, ..., e7),   # Layer-0
        (e0, e1, ..., e7),   # Layer-1
        ...
        (e0, e1, ..., e7),   # Layer-57
    )

共 58 项。

------------------------------------------------------------

注意：

本文件负责的是：

    一个完整 token

还没有遍历整个 Chinese-SimpleQA。

下一步再写：

    trace_workload.py

把真实 Trace 中所有有效 token
整理成这种 58 层 route。
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

from scheduling.layer_scheduler import (
    LayerScheduleResult,
    schedule_layer,
)

from scheduling.runtime_index import (
    DEFAULT_MAPPING_PATH,
    RuntimeIndex,
    load_runtime_index,
)


# ============================================================
# 异常
# ============================================================


class TokenSchedulerError(ValueError):
    """完整 Token 调度失败。"""


# ============================================================
# 一层在完整 Token 时间轴中的位置
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class TokenLayerExecution:
    """
    一个 Layer 在完整 Token 时间轴中的结果。

    layer_result 中的时间：

        从 0 开始

    而这里额外记录：

        global_start_time
        global_finish_time

    因为当前：

        Layer 严格串行。
    """

    layer_id: int

    global_start_time: int

    global_finish_time: int

    layer_result: (
        LayerScheduleResult
    )

    def __post_init__(
        self,
    ) -> None:

        if self.layer_id < 0:

            raise TokenSchedulerError(
                "layer_id 不能为负数。"
            )

        if (
            self.global_start_time
            < 0
        ):

            raise TokenSchedulerError(
                "global_start_time "
                "不能为负数。"
            )

        if (
            self.global_finish_time
            < self.global_start_time
        ):

            raise TokenSchedulerError(
                "global_finish_time "
                "不能早于 start。"
            )

        if (
            self.layer_result.layer_id
            != self.layer_id
        ):

            raise TokenSchedulerError(
                "TokenLayerExecution "
                "与 LayerScheduleResult "
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

            raise TokenSchedulerError(
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
# 完整 Token 结果
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class TokenScheduleResult:
    """
    一个完整 Token 的 58 层调度结果。
    """

    # ========================================================
    # Route
    # ========================================================

    routed_experts_by_layer: tuple[
        tuple[
            int,
            ...
        ],
        ...
    ]

    # ========================================================
    # Layer Results
    # ========================================================

    layers: tuple[
        TokenLayerExecution,
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
    # 总时间
    # ========================================================

    total_cycles: int

    # ========================================================
    # 基础统计
    # ========================================================

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
        """
        DeepSeek 当前：

            9 Expert
            ×
            3 Matrix
            ×
            58 Layer

        = 1566 tasks / token
        """

        return sum(
            layer.layer_result
            .task_count

            for layer
            in self.layers
        )

    @property
    def total_switches(
        self,
    ) -> int:

        return sum(
            layer.layer_result
            .switch_count

            for layer
            in self.layers
        )

    @property
    def total_initial_activations(
        self,
    ) -> int:

        return sum(
            layer.layer_result
            .initial_activation_count

            for layer
            in self.layers
        )

    @property
    def total_activation_overhead_cycles(
        self,
    ) -> int:

        return sum(
            layer.layer_result
            .activation_overhead_cycles

            for layer
            in self.layers
        )

    @property
    def total_compute_work_cycles(
        self,
    ) -> int:
        """
        这里是所有 SC 上 compute work 的总和，
        不是 Token latency。

        因为不同 SC 可以并行。
        """

        return sum(
            layer.layer_result
            .compute_cycles

            for layer
            in self.layers
        )

    @property
    def total_busy_cycles(
        self,
    ) -> int:
        """
        所有 SC 的 busy time 总和：

            compute
            +
            activation/switch

        仍然不是 latency。
        """

        return sum(
            layer.layer_result
            .busy_cycles

            for layer
            in self.layers
        )

    @property
    def total_wait_cycles(
        self,
    ) -> int:
        """
        所有 Weight-Cube task
        因目标 Sub-Cube 忙碌而产生的等待总和。
        """

        return sum(
            layer.layer_result
            .wait_cycles

            for layer
            in self.layers
        )

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
    # 查询
    # ========================================================

    def layer(
        self,
        layer_id: int,
    ) -> TokenLayerExecution:

        if not (
            0
            <= layer_id
            < len(self.layers)
        ):

            raise TokenSchedulerError(
                f"layer_id={layer_id} "
                "超出范围。"
            )

        layer = (
            self.layers[
                layer_id
            ]
        )

        if (
            layer.layer_id
            != layer_id
        ):

            raise TokenSchedulerError(
                "layers 没有按照 "
                "layer_id 排序。"
            )

        return layer


# ============================================================
# Route 标准化
# ============================================================


def normalize_token_routes(
    *,
    index: RuntimeIndex,

    routed_experts_by_layer: Iterable[
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
    将输入标准化成：

        tuple[
            tuple[int,...],
            ...
        ]

    并利用 RuntimeIndex
    对每一层 Top-K 做合法性检查。
    """

    routes = tuple(
        tuple(
            route
        )
        for route
        in routed_experts_by_layer
    )

    # ========================================================
    # Layer 数
    # ========================================================

    if (
        len(routes)
        != index.num_layers
    ):

        raise TokenSchedulerError(
            "Token Route Layer 数错误："
            f"actual={len(routes)}, "
            f"expected="
            f"{index.num_layers}。"
        )

    # ========================================================
    # 每层 Top-K
    #
    # RuntimeIndex 会检查：
    #
    # 数量
    # ID 范围
    # 重复 Expert
    # ========================================================

    for (
        layer_id,
        route,
    ) in enumerate(
        routes
    ):

        index.resolve_active_expert_ids(
            layer_id=layer_id,

            routed_expert_ids=(
                route
            ),
        )

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
    一个 Token 开始前每个 SC 当前激活的 WC。

    第一个 Token 一般：

        None × 16

    后续如果要做 token pipeline / 连续 token：

        上一个 Token final state
            ->
        下一个 Token initial state
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

        raise TokenSchedulerError(
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
            or isinstance(
                value,
                bool,
            )
            or value < 0
        ):

            raise TokenSchedulerError(
                "Initial active cube "
                "必须为 None 或非负整数。"
            )

    return state


# ============================================================
# 主函数
# ============================================================


def schedule_token(
    *,
    index: RuntimeIndex,

    routed_experts_by_layer: Iterable[
        Iterable[int]
    ],

    rules: ExecutionRules | None = None,

    initial_active_cube_by_subcube: (
        Iterable[
            int | None
        ]
        | None
    ) = None,

    charge_initial_activation: bool = True,
) -> TokenScheduleResult:
    """
    模拟一个完整 Token 经过所有 MoE Layer。

    当前严格：

        Layer-0 完成
            ↓
        Layer-1
            ↓
        ...
            ↓
        Layer-57

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
        normalize_token_routes(
            index=index,

            routed_experts_by_layer=(
                routed_experts_by_layer
            ),
        )
    )

    # ========================================================
    # 初始 SC active state
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
    # 全局 Token 时间轴
    # ========================================================

    global_time = 0

    layer_executions: list[
        TokenLayerExecution
    ] = []

    # ========================================================
    # 58 层严格顺序执行
    # ========================================================

    for layer_id in range(
        index.num_layers
    ):

        route = (
            routes[
                layer_id
            ]
        )

        # ====================================================
        # 单层调度
        # ====================================================

        layer_result = (
            schedule_layer(
                index=index,

                layer_id=layer_id,

                routed_expert_ids=(
                    route
                ),

                rules=rules,

                initial_active_cube_by_subcube=(
                    current_state
                ),

                charge_initial_activation=(
                    charge_initial_activation
                ),
            )
        )

        global_start = (
            global_time
        )

        global_finish = (
            global_start
            + layer_result
            .total_cycles
        )

        execution = (
            TokenLayerExecution(
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

        layer_executions.append(
            execution
        )

        # ====================================================
        # 下一层继续使用当前 SC 中
        # 已经 active 的 WeightCube。
        # ====================================================

        current_state = (
            layer_result
            .final_active_cube_by_subcube
        )

        global_time = (
            global_finish
        )

    # ========================================================
    # Final Result
    # ========================================================

    result = (
        TokenScheduleResult(
            routed_experts_by_layer=(
                routes
            ),

            layers=tuple(
                layer_executions
            ),

            initial_active_cube_by_subcube=(
                initial_state
            ),

            final_active_cube_by_subcube=(
                current_state
            ),

            total_cycles=(
                global_time
            ),
        )
    )

    validate_token_schedule(
        result=result,
        index=index,
    )

    return result


# ============================================================
# 最终检查
# ============================================================


def validate_token_schedule(
    *,
    result: TokenScheduleResult,

    index: RuntimeIndex,
) -> None:
    """
    检查完整 Token 的：

    1. Layer 数；
    2. Layer 顺序；
    3. Layer 全局时间连续；
    4. active state 连续；
    5. total_cycles；
    6. task 数。
    """

    # ========================================================
    # Layer 数
    # ========================================================

    if (
        result.num_layers
        != index.num_layers
    ):

        raise TokenSchedulerError(
            "TokenScheduleResult "
            "Layer 数错误。"
        )

    if (
        len(
            result.routed_experts_by_layer
        )
        != index.num_layers
    ):

        raise TokenSchedulerError(
            "Route Layer 数错误。"
        )

    # ========================================================
    # 时间必须连续
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

            raise TokenSchedulerError(
                "Layer 顺序错误。"
            )

        if (
            execution.global_start_time
            != expected_start
        ):

            raise TokenSchedulerError(
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

            raise TokenSchedulerError(
                f"Layer-{layer_id} "
                "初始 WeightCube 状态 "
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

        raise TokenSchedulerError(
            "最终 active state 错误。"
        )

    # ========================================================
    # Total cycles
    # ========================================================

    if (
        expected_start
        != result.total_cycles
    ):

        raise TokenSchedulerError(
            "Token total_cycles "
            "与 Layer 累加结果不一致。"
        )

    expected_sum = sum(
        layer.layer_result
        .total_cycles

        for layer
        in result.layers
    )

    if (
        expected_sum
        != result.total_cycles
    ):

        raise TokenSchedulerError(
            "Token total_cycles "
            "不等于各 Layer 周期之和。"
        )

    # ========================================================
    # 每层任务数
    #
    # active experts:
    #
    #     Router Top-K
    #     +
    #     Shared
    #
    # 每 Expert 三个矩阵。
    # ========================================================

    active_experts_per_layer = (
        index.model_config
        .experts_per_token
        +
        int(
            index.model_config
            .include_shared_expert
        )
    )

    expected_tasks_per_layer = (
        active_experts_per_layer
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

            raise TokenSchedulerError(
                f"Layer-"
                f"{execution.layer_id} "
                "任务数量错误。"
            )


# ============================================================
# 输出
# ============================================================


def print_token_schedule_summary(
    result: TokenScheduleResult,
) -> None:
    """
    打印完整 Token 调度摘要。
    """

    print(
        "\n"
        "========== Token Schedule =========="
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
        f"Total Token Cycles："
        f"{result.total_cycles}"
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
    # 每层周期
    # ========================================================

    print(
        "\nLayer Timeline："
    )

    for execution in (
        result.layers
    ):

        layer_result = (
            execution
            .layer_result
        )

        print(
            f"  L{execution.layer_id}: "
            f"{execution.global_start_time}"
            " -> "
            f"{execution.global_finish_time}"
            "  "
            f"cycles={execution.cycles}, "
            f"switch="
            f"{layer_result.switch_count}, "
            f"wait="
            f"{layer_result.wait_cycles}"
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

    CLI 模式会把同一条 route
    临时重复到全部 58 Layer。

    只用于 smoke test。

    正式实验不会这样做，
    下一步会读取真实 Trace。
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
                "模拟一个完整 Token "
                "通过全部 MoE Layer。"
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
            "该 Route 会重复到所有 Layer。"
        ),
    )

    parser.add_argument(
        "--no-initial-activation-cost",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    index = (
        load_runtime_index(
            args.mapping
        )
    )

    # ========================================================
    # 这里只是 Smoke Test：
    #
    # 同一个 Top-8 重复到全部 Layer。
    #
    # 正式实验下一步从真实 trace
    # 得到每层不同的 Top-8。
    # ========================================================

    routes = tuple(
        args.route

        for _ in range(
            index.num_layers
        )
    )

    result = (
        schedule_token(
            index=index,

            routed_experts_by_layer=(
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

    print_token_schedule_summary(
        result
    )


if __name__ == "__main__":
    main()