"""
第五步：Chinese-SimpleQA Workload 推理周期评估。

完整流程：

    TraceToken
        ↓
    58 Layer Routes
        ↓
    Token Latency
        ↓
    Workload Statistics

------------------------------------------------------------

提供两种模式：

1. exact

    直接调用：

        schedule_token()

    完整创建所有 ScheduledTask，
    最适合：

        correctness test
        少量 Token 分析
        debug

2. fast

    针对当前固定 Baseline 调度规则，
    只计算最终 Layer / Token latency。

    不创建：

        ScheduledTask
        LayerScheduleResult
        SubcubeLayerStats

    用于：

        1000
        10000
        甚至更多真实 Token

------------------------------------------------------------

FAST 模式为什么可以保持精确？

当前规则：

    switch = 1
    compute = 1

且：

    charge_initial_activation = True

因此：

    一个 Weight-Cube Task
        =
    2 cycles

无论一个 SC：

    初始没有 active WC

还是：

    已经激活上一层的其他 WC

下一任务都需要 1 cycle activation/switch。

------------------------------------------------------------

另外：

所有 gate/up：

    ready_time = 0

而 down：

    ready_time > 0

layer_scheduler.py 的优先级首先比较：

    ready_time

因此同一个 SC：

    所有 gate/up
        ↓
    down

不会出现 down 插到尚未执行的 pre-task 前面的情况。

所以可以直接计算：

    Pre completion time
        ↓
    Down ready time
        ↓
    Down serial schedule

而不需要事件堆。

------------------------------------------------------------

注意：

FAST 模式当前只支持：

    charge_initial_activation=True

这是正式 Baseline 使用的模式。

如果以后：

    initial activation != switch
    或
    switch_cycles 动态变化
    或
    引入 cross-SC communication
    或
    改 Scheduler priority

则必须同步修改 fast evaluator。
"""

from __future__ import annotations

import argparse
import math
import time

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable


from config import (
    ExecutionRules,
)

from scheduling.runtime_index import (
    DEFAULT_MAPPING_PATH,
    RuntimeIndex,
    load_runtime_index,
)

from scheduling.token_scheduler import (
    schedule_token,
)

from scheduling.trace_workload import (
    DEFAULT_TRACE_ROOT,
    TraceToken,
    TraceWorkloadStats,
    iter_trace_tokens,
)


# ============================================================
# Mode
# ============================================================


MODE_EXACT = "exact"

MODE_FAST = "fast"

VALID_MODES = {
    MODE_EXACT,
    MODE_FAST,
}


# ============================================================
# 异常
# ============================================================


class WorkloadEvaluatorError(
    ValueError
):
    """Workload 推理评估失败。"""


# ============================================================
# Category Result
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class CategoryEvaluation:
    """
    一个 Chinese-SimpleQA 类别的统计。
    """

    category: str

    token_count: int

    mean_cycles: float

    min_cycles: int

    p50_cycles: int

    p95_cycles: int

    max_cycles: int


# ============================================================
# Workload Result
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class WorkloadEvaluationResult:
    """
    一次 Workload 实验结果。
    """

    mode: str

    token_count: int

    total_cycles: int

    mean_cycles: float

    min_cycles: int

    p50_cycles: int

    p95_cycles: int

    p99_cycles: int

    max_cycles: int

    # 实际如果逐个 exact 模拟，需要执行的
    # Weight-Cube task 数量。
    equivalent_task_count: int

    verified_token_count: int

    runtime_seconds: float

    cycles: tuple[
        int,
        ...
    ]

    category_results: tuple[
        CategoryEvaluation,
        ...
    ]

    trace_stats: (
        TraceWorkloadStats
    )

    @property
    def tokens_per_second_simulation(
        self,
    ) -> float:
        """
        注意：

        这是 Python Simulator 自身处理速度，

        不是 PIM 硬件实际吞吐率。
        """

        if (
            self.runtime_seconds
            <= 0
        ):

            return 0.0

        return (
            self.token_count
            / self.runtime_seconds
        )


# ============================================================
# Percentile
# ============================================================


def percentile_nearest_rank(
    values: Iterable[int],
    percentile: float,
) -> int:
    """
    Nearest-rank percentile。

    例如：

        P95

    index：

        ceil(0.95 * N) - 1
    """

    data = sorted(
        values
    )

    if not data:

        raise WorkloadEvaluatorError(
            "不能对空数据计算 percentile。"
        )

    if not (
        0.0
        < percentile
        <= 1.0
    ):

        raise WorkloadEvaluatorError(
            "percentile 必须位于 "
            "(0, 1]。"
        )

    rank = math.ceil(
        percentile
        * len(data)
    )

    index = max(
        0,
        rank - 1,
    )

    return int(
        data[
            index
        ]
    )


# ============================================================
# Fast Rules 检查
# ============================================================


def validate_fast_rules(
    rules: ExecutionRules,
    *,
    charge_initial_activation: bool,
) -> None:
    """
    FAST 模式成立需要的条件。
    """

    if not (
        charge_initial_activation
    ):

        raise WorkloadEvaluatorError(
            "FAST 模式当前要求 "
            "charge_initial_activation=True。"
        )

    if (
        rules.compute_cycles
        <= 0
    ):

        raise WorkloadEvaluatorError(
            "compute_cycles 必须大于 0。"
        )

    if (
        rules.switch_cycles
        < 0
    ):

        raise WorkloadEvaluatorError(
            "switch_cycles 不能小于 0。"
        )

    if (
        rules.cross_subcube_cycles
        != 0
    ):

        raise WorkloadEvaluatorError(
            "FAST Baseline 当前要求 "
            "cross_subcube_cycles=0。"
        )

    if not (
        rules
        .unlimited_parallel_subcubes
    ):

        raise WorkloadEvaluatorError(
            "FAST Baseline 要求 "
            "Sub-Cube 间完全并行。"
        )

    if not (
        rules
        .one_active_weight_cube_per_subcube
    ):

        raise WorkloadEvaluatorError(
            "FAST Baseline 要求 "
            "每个 SC 同时只能执行 "
            "一个 Weight-Cube。"
        )


# ============================================================
# FAST：单 Layer
# ============================================================


def fast_schedule_layer_cycles(
    *,
    index: RuntimeIndex,

    layer_id: int,

    routed_expert_ids: Iterable[int],

    rules: ExecutionRules | None = None,

    charge_initial_activation: bool = True,
) -> int:
    """
    精简计算一个 Layer 的 latency。

    --------------------------------------------------------

    当前：

        active experts
            =
        Top-K Routed + Shared

    每个 Expert：

        gate
        up
        down

    --------------------------------------------------------

    Step A：Pre Stage

    所有 gate/up：

        ready_time = 0

    因为 route_rank 是唯一的，
    同一个 SC 中的执行顺序就是：

        route rank 0
        route rank 1
        ...

    gate/up 又被 Step4 强制分到不同 SC，
    所以不需要额外排序。

    --------------------------------------------------------

    Step B：Down

    对 Expert-e：

        down_ready[e]
            =
        max(
            gate_finish[e],
            up_finish[e]
        )

    down 所在 SC：

        与 gate 相同。

    --------------------------------------------------------

    Step C：

    每个 SC：

        pre 全部执行结束
            ↓
        按：
            down_ready
            route_rank

        执行 down。

    最慢 SC 的 finish time：

        = Layer latency。
    """

    if rules is None:

        rules = (
            ExecutionRules()
        )

    validate_fast_rules(
        rules,

        charge_initial_activation=(
            charge_initial_activation
        ),
    )

    active_ids = (
        index.resolve_active_expert_ids(
            layer_id=layer_id,

            routed_expert_ids=(
                routed_expert_ids
            ),
        )
    )

    # ========================================================
    # 当前每个 Task 都固定：
    #
    # switch/activation + compute
    # ========================================================

    service_cycles = (
        rules.switch_cycles
        +
        rules.compute_cycles
    )

    if (
        service_cycles
        <= 0
    ):

        raise WorkloadEvaluatorError(
            "Task service cycles "
            "必须大于 0。"
        )

    num_sc = (
        index.num_subcubes
    )

    # ========================================================
    # 每个 SC 已经排入多少个 pre task
    # ========================================================

    pre_count = [
        0
        for _ in range(
            num_sc
        )
    ]

    # ========================================================
    # 每个 SC 的 down：
    #
    # (
    #     ready_time,
    #     route_rank
    # )
    # ========================================================

    down_by_sc: list[
        list[
            tuple[
                int,
                int,
            ]
        ]
    ] = [
        []
        for _ in range(
            num_sc
        )
    ]

    # ========================================================
    # 按 route rank 直接遍历。
    #
    # 这与 exact scheduler 的 pre-task
    # 顺序一致。
    # ========================================================

    for (
        route_rank,
        expert_id,
    ) in enumerate(
        active_ids
    ):

        expert = (
            index.expert(
                layer_id,
                expert_id,
            )
        )

        gate_sc = (
            expert.gate_subcube
        )

        up_sc = (
            expert.up_subcube
        )

        down_sc = (
            expert.down_subcube
        )

        # Step4 硬约束
        if (
            gate_sc
            == up_sc
        ):

            raise WorkloadEvaluatorError(
                f"Layer-{layer_id} "
                f"Expert-{expert_id} "
                "gate/up 位于同一个 SC。"
            )

        if (
            gate_sc
            != down_sc
        ):

            raise WorkloadEvaluatorError(
                f"Layer-{layer_id} "
                f"Expert-{expert_id} "
                "gate/down 没有共址。"
            )

        # ====================================================
        # gate 被排入自己的 SC
        # ====================================================

        pre_count[
            gate_sc
        ] += 1

        gate_finish = (
            pre_count[
                gate_sc
            ]
            * service_cycles
        )

        # ====================================================
        # up
        # ====================================================

        pre_count[
            up_sc
        ] += 1

        up_finish = (
            pre_count[
                up_sc
            ]
            * service_cycles
        )

        # ====================================================
        # down dependency
        # ====================================================

        down_ready = max(
            gate_finish,
            up_finish,
        )

        down_by_sc[
            down_sc
        ].append(
            (
                down_ready,
                route_rank,
            )
        )

    # ========================================================
    # Pre 阶段结束时间
    # ========================================================

    finish_by_sc = [
        count
        * service_cycles

        for count
        in pre_count
    ]

    # ========================================================
    # Down
    # ========================================================

    for sc in range(
        num_sc
    ):

        downs = (
            down_by_sc[
                sc
            ]
        )

        if not downs:
            continue

        # exact scheduler：
        #
        # ready_time
        # -> route_rank
        downs.sort()

        current_time = (
            finish_by_sc[
                sc
            ]
        )

        for (
            ready_time,
            _route_rank,
        ) in downs:

            current_time = max(
                current_time,
                ready_time,
            )

            current_time += (
                service_cycles
            )

        finish_by_sc[
            sc
        ] = (
            current_time
        )

    return max(
        finish_by_sc,
        default=0,
    )


# ============================================================
# FAST：完整 Token
# ============================================================


def fast_schedule_token_cycles(
    *,
    index: RuntimeIndex,

    routed_experts_by_layer: Iterable[
        Iterable[int]
    ],

    rules: ExecutionRules | None = None,

    charge_initial_activation: bool = True,
) -> int:
    """
    FAST 模式计算一个完整 Token。

    当前 Layer 严格串行，所以：

        token cycles
            =
        Σ layer cycles
    """

    if rules is None:

        rules = (
            ExecutionRules()
        )

    routes = tuple(
        tuple(route)

        for route
        in routed_experts_by_layer
    )

    if (
        len(routes)
        != index.num_layers
    ):

        raise WorkloadEvaluatorError(
            "Token Route Layer 数错误："
            f"actual={len(routes)}, "
            f"expected={index.num_layers}。"
        )

    total = 0

    for (
        layer_id,
        route,
    ) in enumerate(
        routes
    ):

        total += (
            fast_schedule_layer_cycles(
                index=index,

                layer_id=(
                    layer_id
                ),

                routed_expert_ids=(
                    route
                ),

                rules=rules,

                charge_initial_activation=(
                    charge_initial_activation
                ),
            )
        )

    return total


# ============================================================
# FAST vs EXACT
# ============================================================


def verify_fast_token(
    *,
    index: RuntimeIndex,

    token: TraceToken,

    rules: ExecutionRules | None = None,
) -> tuple[
    int,
    int,
]:
    """
    对同一个真实 Token：

        FAST
        vs
        EXACT

    周期必须完全一致。

    返回：

        (
            fast_cycles,
            exact_cycles
        )
    """

    if rules is None:

        rules = (
            ExecutionRules()
        )

    fast_cycles = (
        fast_schedule_token_cycles(
            index=index,

            routed_experts_by_layer=(
                token
                .routed_experts_by_layer
            ),

            rules=rules,

            charge_initial_activation=True,
        )
    )

    exact_result = (
        schedule_token(
            index=index,

            routed_experts_by_layer=(
                token
                .routed_experts_by_layer
            ),

            rules=rules,

            charge_initial_activation=True,
        )
    )

    exact_cycles = (
        exact_result
        .total_cycles
    )

    if (
        fast_cycles
        != exact_cycles
    ):

        raise WorkloadEvaluatorError(
            "FAST / EXACT 周期不一致："
            f"token={token.token_id}, "
            f"file={token.relative_file}, "
            f"segment={token.segment_index}, "
            f"token_index="
            f"{token.token_index_in_segment}, "
            f"fast={fast_cycles}, "
            f"exact={exact_cycles}。"
        )

    return (
        fast_cycles,
        exact_cycles,
    )


# ============================================================
# Category Helper
# ============================================================


def _build_category_results(
    category_cycles: dict[
        str,
        list[int],
    ],
) -> tuple[
    CategoryEvaluation,
    ...
]:

    results = []

    for category in sorted(
        category_cycles
    ):

        values = (
            category_cycles[
                category
            ]
        )

        if not values:
            continue

        result = (
            CategoryEvaluation(
                category=category,

                token_count=(
                    len(values)
                ),

                mean_cycles=(
                    fmean(values)
                ),

                min_cycles=(
                    min(values)
                ),

                p50_cycles=(
                    percentile_nearest_rank(
                        values,
                        0.50,
                    )
                ),

                p95_cycles=(
                    percentile_nearest_rank(
                        values,
                        0.95,
                    )
                ),

                max_cycles=(
                    max(values)
                ),
            )
        )

        results.append(
            result
        )

    return tuple(
        results
    )


# ============================================================
# Workload Evaluator
# ============================================================


def evaluate_trace_workload(
    *,
    index: RuntimeIndex,

    trace_root: Path | str = (
        DEFAULT_TRACE_ROOT
    ),

    mode: str = MODE_FAST,

    max_files: int | None = None,

    max_tokens: int | None = None,

    verify_first: int = 10,

    rules: ExecutionRules | None = None,

    verbose: bool = True,
) -> WorkloadEvaluationResult:
    """
    真正评估 Chinese-SimpleQA Workload。

    --------------------------------------------------------

    mode="exact"

        每个 Token 直接：

            schedule_token()

    --------------------------------------------------------

    mode="fast"

        使用快速 latency evaluator。

        如果：

            verify_first > 0

        则前 N 个 Token 会同时运行：

            FAST
            EXACT

        并强制检查结果完全一致。
    """

    if mode not in VALID_MODES:

        raise WorkloadEvaluatorError(
            f"未知 mode={mode!r}，"
            f"必须为 {sorted(VALID_MODES)}。"
        )

    if rules is None:

        rules = (
            ExecutionRules()
        )

    if verify_first < 0:

        raise WorkloadEvaluatorError(
            "verify_first 不能小于 0。"
        )

    if mode == MODE_FAST:

        validate_fast_rules(
            rules,

            charge_initial_activation=True,
        )

    trace_stats = (
        TraceWorkloadStats()
    )

    all_cycles: list[int] = []

    category_cycles: dict[
        str,
        list[int],
    ] = {}

    verified = 0

    start_time = (
        time.perf_counter()
    )

    # ========================================================
    # Streaming Token
    # ========================================================

    for token in (
        iter_trace_tokens(
            trace_root=(
                trace_root
            ),

            max_files=(
                max_files
            ),

            max_tokens=(
                max_tokens
            ),

            stats=(
                trace_stats
            ),

            # Workload evaluator 自己打印进度，
            # 不再打印 file loader progress。
            verbose=False,
        )
    ):

        token_number = (
            len(all_cycles)
            + 1
        )

        # ====================================================
        # FAST
        # ====================================================

        if (
            mode
            == MODE_FAST
        ):

            cycles = (
                fast_schedule_token_cycles(
                    index=index,

                    routed_experts_by_layer=(
                        token
                        .routed_experts_by_layer
                    ),

                    rules=rules,

                    charge_initial_activation=True,
                )
            )

            # ================================================
            # 前 N Token Exact Validation
            # ================================================

            if (
                verified
                < verify_first
            ):

                exact_result = (
                    schedule_token(
                        index=index,

                        routed_experts_by_layer=(
                            token
                            .routed_experts_by_layer
                        ),

                        rules=rules,

                        charge_initial_activation=True,
                    )
                )

                if (
                    cycles
                    != exact_result
                    .total_cycles
                ):

                    raise WorkloadEvaluatorError(
                        "FAST / EXACT "
                        "验证失败："
                        f"Token-{token.token_id}, "
                        f"fast={cycles}, "
                        f"exact="
                        f"{exact_result.total_cycles}。"
                    )

                verified += 1

        # ====================================================
        # EXACT
        # ====================================================

        else:

            exact_result = (
                schedule_token(
                    index=index,

                    routed_experts_by_layer=(
                        token
                        .routed_experts_by_layer
                    ),

                    rules=rules,

                    charge_initial_activation=True,
                )
            )

            cycles = (
                exact_result
                .total_cycles
            )

        # ====================================================
        # Record
        # ====================================================

        all_cycles.append(
            cycles
        )

        category_cycles.setdefault(
            token.category,
            [],
        ).append(
            cycles
        )

        # ====================================================
        # Progress
        # ====================================================

        if verbose and (
            token_number == 1
            or
            token_number % 1000 == 0
        ):

            elapsed = (
                time.perf_counter()
                - start_time
            )

            speed = (
                token_number
                / elapsed

                if elapsed > 0
                else 0.0
            )

            print(
                f"[Evaluate] "
                f"tokens={token_number}, "
                f"last_cycles={cycles}, "
                f"mean="
                f"{fmean(all_cycles):.2f}, "
                f"speed="
                f"{speed:.1f} token/s"
            )

    runtime = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # Empty
    # ========================================================

    if not all_cycles:

        raise WorkloadEvaluatorError(
            "没有读取到任何有效 Token。"
        )

    token_count = (
        len(all_cycles)
    )

    # ========================================================
    # 当前每 Token task 数
    # ========================================================

    active_experts = (
        index.model_config
        .experts_per_token
        +
        int(
            index.model_config
            .include_shared_expert
        )
    )

    tasks_per_token = (
        index.num_layers
        * active_experts
        * 3
    )

    equivalent_tasks = (
        token_count
        * tasks_per_token
    )

    # ========================================================
    # Result
    # ========================================================

    result = (
        WorkloadEvaluationResult(
            mode=mode,

            token_count=(
                token_count
            ),

            total_cycles=(
                sum(
                    all_cycles
                )
            ),

            mean_cycles=(
                fmean(
                    all_cycles
                )
            ),

            min_cycles=(
                min(
                    all_cycles
                )
            ),

            p50_cycles=(
                percentile_nearest_rank(
                    all_cycles,
                    0.50,
                )
            ),

            p95_cycles=(
                percentile_nearest_rank(
                    all_cycles,
                    0.95,
                )
            ),

            p99_cycles=(
                percentile_nearest_rank(
                    all_cycles,
                    0.99,
                )
            ),

            max_cycles=(
                max(
                    all_cycles
                )
            ),

            equivalent_task_count=(
                equivalent_tasks
            ),

            verified_token_count=(
                verified
            ),

            runtime_seconds=(
                runtime
            ),

            cycles=tuple(
                all_cycles
            ),

            category_results=(
                _build_category_results(
                    category_cycles
                )
            ),

            trace_stats=(
                trace_stats
            ),
        )
    )

    return result


# ============================================================
# Print
# ============================================================


def print_workload_evaluation_summary(
    result: WorkloadEvaluationResult,
) -> None:
    """
    打印 Workload 评估摘要。
    """

    print(
        "\n"
        "=============================================="
    )

    print(
        "       Workload Evaluation Result"
    )

    print(
        "=============================================="
    )

    print(
        f"Mode："
        f"{result.mode}"
    )

    print(
        f"Tokens："
        f"{result.token_count}"
    )

    print(
        "Equivalent Weight-Cube Tasks："
        f"{result.equivalent_task_count}"
    )

    print(
        f"FAST Verified Tokens："
        f"{result.verified_token_count}"
    )

    print(
        "\nLatency (cycles/token)："
    )

    print(
        f"  Mean："
        f"{result.mean_cycles:.4f}"
    )

    print(
        f"  Min："
        f"{result.min_cycles}"
    )

    print(
        f"  P50："
        f"{result.p50_cycles}"
    )

    print(
        f"  P95："
        f"{result.p95_cycles}"
    )

    print(
        f"  P99："
        f"{result.p99_cycles}"
    )

    print(
        f"  Max："
        f"{result.max_cycles}"
    )

    print(
        "\nSimulation Performance："
    )

    print(
        f"  Runtime："
        f"{result.runtime_seconds:.3f} s"
    )

    print(
        f"  Simulator Throughput："
        f"{result.tokens_per_second_simulation:.2f} token/s"
    )

    print(
        "\nTrace："
    )

    print(
        f"  Files Processed："
        f"{result.trace_stats.processed_file_count}"
    )

    print(
        f"  Segments Seen："
        f"{result.trace_stats.trace_segment_count}"
    )

    print(
        f"  Valid Segments："
        f"{result.trace_stats.valid_segment_count}"
    )

    print(
        f"  Skipped Segments："
        f"{result.trace_stats.skipped_segment_count}"
    )

    print(
        "\nCategory Latency："
    )

    for category in (
        result.category_results
    ):

        print(
            f"  {category.category}: "
            f"n={category.token_count}, "
            f"mean="
            f"{category.mean_cycles:.2f}, "
            f"P50="
            f"{category.p50_cycles}, "
            f"P95="
            f"{category.p95_cycles}, "
            f"max="
            f"{category.max_cycles}"
        )


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "评估 Chinese-SimpleQA "
                "真实 MoE Token 推理周期。"
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
        "--trace-root",
        type=Path,
        default=(
            DEFAULT_TRACE_ROOT
        ),
    )

    parser.add_argument(
        "--mode",
        choices=sorted(
            VALID_MODES
        ),
        default=(
            MODE_FAST
        ),
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,

        help=(
            "默认先评估 1000 个 Token。"
            "正式全量可以显式设置更大的值。"
        ),
    )

    parser.add_argument(
        "--verify-first",
        type=int,
        default=10,

        help=(
            "FAST 模式下前 N 个 Token "
            "同时使用 EXACT 验证。"
        ),
    )

    parser.add_argument(
        "--quiet",
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

    result = (
        evaluate_trace_workload(
            index=index,

            trace_root=(
                args.trace_root
            ),

            mode=(
                args.mode
            ),

            max_files=(
                args.max_files
            ),

            max_tokens=(
                args.max_tokens
            ),

            verify_first=(
                args.verify_first
            ),

            rules=(
                ExecutionRules()
            ),

            verbose=(
                not args.quiet
            ),
        )
    )

    print_workload_evaluation_summary(
        result
    )


if __name__ == "__main__":
    main()