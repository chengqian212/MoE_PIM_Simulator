"""
第五步：Latency 深度分析。

目标：

1. 哪些 Layer 平均最慢？
2. 哪些 Layer 的 P95 最慢？
3. 哪些 Sub-Cube 最经常成为关键路径？
4. 高延迟 Token 是哪些？
5. Token 的动态 Sub-Cube 冲突程度
   与最终 latency 是否相关？

------------------------------------------------------------

本文件不修改任何调度规则。

FAST 部分：

    对大量 Token：
        计算每层 latency
        计算总 latency
        计算动态 collision score

EXACT 部分：

    对前 N 个 Token：
        调用 schedule_token()

    用于统计：
        各 SC task 数
        busy cycles
        wait cycles
        switch 数
        critical-path 出现次数

------------------------------------------------------------

为什么不是直接分析 Mapping Conflict Cost？

第四步：

    Mapping Conflict Cost

对当前固定 Mapping 来说只有一个值。

例如：

    133349805

而 10000 个 Token 各自 latency 不同。

一个常数无法和这些 Token latency
做 token-level correlation。

因此这里定义动态 collision score：

Pre：

    同一个 SC 上同时需要的 gate/up 越多，
    collision 越大。

Down：

    同一个 SC 上 active down 越多，
    collision 越大。

对每个 SC：

    C(n, 2)

然后对 58 Layer 求和。

这个指标可以随 Token Route 改变，
所以可以和 Token latency 计算相关性。
"""

from __future__ import annotations

import argparse
import heapq
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

from scheduling.workload_evaluator import (
    fast_schedule_layer_cycles,
    percentile_nearest_rank,
)


# ============================================================
# 异常
# ============================================================


class LatencyAnalyzerError(
    ValueError
):
    """Latency 分析失败。"""


# ============================================================
# Layer 统计
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class LayerLatencyStats:
    """
    某一个 MoE Layer 在所有 Token 上的 latency。
    """

    layer_id: int

    token_count: int

    mean_cycles: float

    min_cycles: int

    p50_cycles: int

    p95_cycles: int

    p99_cycles: int

    max_cycles: int


# ============================================================
# Sub-Cube 统计
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class SubcubeLatencyStats:
    """
    EXACT Sample 上一个 Sub-Cube 的运行统计。

    observed_layer_events：

        exact_token_count
        ×
        num_layers

    critical_layer_count：

        这个 SC 有多少次成为某一层最终
        finish time 的关键 SC。

    注意：

        如果两个 SC 同时最后完成，

        两个都会计为 critical。
    """

    subcube_id: int

    observed_layer_events: int

    critical_layer_count: int

    total_tasks: int

    total_busy_cycles: int

    total_wait_cycles: int

    total_switches: int

    total_initial_activations: int

    @property
    def critical_rate(
        self,
    ) -> float:

        if (
            self.observed_layer_events
            <= 0
        ):

            return 0.0

        return (
            self.critical_layer_count
            / self.observed_layer_events
        )

    @property
    def mean_tasks_per_layer(
        self,
    ) -> float:

        if (
            self.observed_layer_events
            <= 0
        ):

            return 0.0

        return (
            self.total_tasks
            / self.observed_layer_events
        )

    @property
    def mean_busy_cycles_per_layer(
        self,
    ) -> float:

        if (
            self.observed_layer_events
            <= 0
        ):

            return 0.0

        return (
            self.total_busy_cycles
            / self.observed_layer_events
        )

    @property
    def mean_wait_cycles_per_layer(
        self,
    ) -> float:

        if (
            self.observed_layer_events
            <= 0
        ):

            return 0.0

        return (
            self.total_wait_cycles
            / self.observed_layer_events
        )


# ============================================================
# Slow Token
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class SlowTokenRecord:
    """
    一个高延迟 Token 的信息。
    """

    token_id: int

    category: str

    relative_file: str

    segment_index: int

    token_index_in_segment: int

    total_cycles: int

    worst_layer_id: int

    worst_layer_cycles: int

    collision_score: int


# ============================================================
# 总结果
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class LatencyAnalysisResult:

    token_count: int

    exact_token_count: int

    runtime_seconds: float

    # ========================================================
    # Token latency
    # ========================================================

    mean_token_cycles: float

    min_token_cycles: int

    p50_token_cycles: int

    p95_token_cycles: int

    p99_token_cycles: int

    max_token_cycles: int

    # ========================================================
    # 动态冲突
    # ========================================================

    mean_collision_score: float

    latency_collision_correlation: (
        float | None
    )

    # ========================================================
    # Detail
    # ========================================================

    layer_stats: tuple[
        LayerLatencyStats,
        ...
    ]

    subcube_stats: tuple[
        SubcubeLatencyStats,
        ...
    ]

    slow_tokens: tuple[
        SlowTokenRecord,
        ...
    ]

    trace_stats: (
        TraceWorkloadStats | None
    )


# ============================================================
# Collision
# ============================================================


def _combination_2(
    value: int,
) -> int:
    """
    C(n,2)
    """

    if value < 2:
        return 0

    return (
        value
        * (
            value - 1
        )
        // 2
    )


def calculate_layer_collision_score(
    *,
    index: RuntimeIndex,

    layer_id: int,

    routed_expert_ids: Iterable[int],
) -> int:
    """
    某一层的动态 SC collision score。

    --------------------------------------------------------

    Pre：

        每个 Expert：

            gate
            up

        分别落到对应 SC。

    如果某个 SC：

        n 个 pre task

    冲突记为：

        C(n,2)

    --------------------------------------------------------

    Down：

        每个 Expert：

            1 个 down

    某个 SC 有 m 个 down：

        C(m,2)

    --------------------------------------------------------

    Layer Score：

        pre collision
        +
        down collision
    """

    active_ids = (
        index.resolve_active_expert_ids(
            layer_id=layer_id,

            routed_expert_ids=(
                routed_expert_ids
            ),
        )
    )

    pre_count = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    down_count = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    for expert_id in active_ids:

        expert = (
            index.expert(
                layer_id,
                expert_id,
            )
        )

        pre_count[
            expert.gate_subcube
        ] += 1

        pre_count[
            expert.up_subcube
        ] += 1

        down_count[
            expert.down_subcube
        ] += 1

    pre_collision = sum(
        _combination_2(
            count
        )
        for count
        in pre_count
    )

    down_collision = sum(
        _combination_2(
            count
        )
        for count
        in down_count
    )

    return (
        pre_collision
        + down_collision
    )


def calculate_token_collision_score(
    *,
    index: RuntimeIndex,

    routed_experts_by_layer: Iterable[
        Iterable[int]
    ],
) -> int:
    """
    58 Layer collision score 求和。
    """

    routes = tuple(
        tuple(route)
        for route
        in routed_experts_by_layer
    )

    if (
        len(routes)
        != index.num_layers
    ):

        raise LatencyAnalyzerError(
            "Token Route Layer 数错误。"
        )

    return sum(
        calculate_layer_collision_score(
            index=index,

            layer_id=layer_id,

            routed_expert_ids=route,
        )

        for (
            layer_id,
            route,
        ) in enumerate(
            routes
        )
    )


# ============================================================
# Pearson
# ============================================================


def pearson_correlation(
    first: Iterable[
        int | float
    ],

    second: Iterable[
        int | float
    ],
) -> float | None:
    """
    Pearson correlation coefficient。

    如果：

        样本 < 2

    或：

        任意变量方差为 0

    返回 None。
    """

    x = tuple(
        float(value)
        for value
        in first
    )

    y = tuple(
        float(value)
        for value
        in second
    )

    if (
        len(x)
        != len(y)
    ):

        raise LatencyAnalyzerError(
            "Pearson 两组数据长度不同。"
        )

    if len(x) < 2:
        return None

    mean_x = fmean(
        x
    )

    mean_y = fmean(
        y
    )

    dx = [
        value - mean_x
        for value
        in x
    ]

    dy = [
        value - mean_y
        for value
        in y
    ]

    sum_x2 = sum(
        value * value
        for value
        in dx
    )

    sum_y2 = sum(
        value * value
        for value
        in dy
    )

    if (
        sum_x2 == 0
        or
        sum_y2 == 0
    ):

        return None

    covariance = sum(
        a * b
        for (
            a,
            b,
        ) in zip(
            dx,
            dy,
        )
    )

    return (
        covariance
        / math.sqrt(
            sum_x2
            * sum_y2
        )
    )


# ============================================================
# Layer Stats
# ============================================================


def _build_layer_stats(
    layer_cycles: list[
        list[int]
    ],
) -> tuple[
    LayerLatencyStats,
    ...
]:

    results = []

    for (
        layer_id,
        values,
    ) in enumerate(
        layer_cycles
    ):

        if not values:

            raise LatencyAnalyzerError(
                f"Layer-{layer_id} "
                "没有 latency 数据。"
            )

        results.append(
            LayerLatencyStats(
                layer_id=(
                    layer_id
                ),

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

                p99_cycles=(
                    percentile_nearest_rank(
                        values,
                        0.99,
                    )
                ),

                max_cycles=(
                    max(values)
                ),
            )
        )

    return tuple(
        results
    )


# ============================================================
# EXACT SC Accumulator
# ============================================================


@dataclass(
    slots=True,
)
class _SubcubeAccumulator:

    observed_layer_events: int = 0

    critical_layer_count: int = 0

    total_tasks: int = 0

    total_busy_cycles: int = 0

    total_wait_cycles: int = 0

    total_switches: int = 0

    total_initial_activations: int = 0


def _build_subcube_stats(
    accumulators: list[
        _SubcubeAccumulator
    ],
) -> tuple[
    SubcubeLatencyStats,
    ...
]:

    return tuple(
        SubcubeLatencyStats(
            subcube_id=(
                subcube_id
            ),

            observed_layer_events=(
                accumulator
                .observed_layer_events
            ),

            critical_layer_count=(
                accumulator
                .critical_layer_count
            ),

            total_tasks=(
                accumulator
                .total_tasks
            ),

            total_busy_cycles=(
                accumulator
                .total_busy_cycles
            ),

            total_wait_cycles=(
                accumulator
                .total_wait_cycles
            ),

            total_switches=(
                accumulator
                .total_switches
            ),

            total_initial_activations=(
                accumulator
                .total_initial_activations
            ),
        )

        for (
            subcube_id,
            accumulator,
        ) in enumerate(
            accumulators
        )
    )


# ============================================================
# 核心分析函数
# ============================================================


def analyze_tokens(
    *,
    index: RuntimeIndex,

    tokens: Iterable[
        TraceToken
    ],

    exact_tokens: int = 100,

    top_k_slow_tokens: int = 10,

    rules: ExecutionRules | None = None,

    verbose: bool = True,

    trace_stats: (
        TraceWorkloadStats | None
    ) = None,
) -> LatencyAnalysisResult:
    """
    分析一批已经提供的 Token。

    这是核心函数。

    analyze_trace_latency()
    只是负责把真实 Trace Generator 接进来。
    """

    if rules is None:

        rules = (
            ExecutionRules()
        )

    if exact_tokens < 0:

        raise LatencyAnalyzerError(
            "exact_tokens 不能小于 0。"
        )

    if top_k_slow_tokens <= 0:

        raise LatencyAnalyzerError(
            "top_k_slow_tokens "
            "必须大于 0。"
        )

    # ========================================================
    # 每层 latency
    # ========================================================

    layer_cycles: list[
        list[int]
    ] = [
        []
        for _ in range(
            index.num_layers
        )
    ]

    # ========================================================
    # Token
    # ========================================================

    token_cycles: list[int] = []

    collision_scores: list[int] = []

    # ========================================================
    # Top-K Slow Token Heap
    #
    # (
    #     latency,
    #     token_id,
    #     record
    # )
    # ========================================================

    slow_heap: list[
        tuple[
            int,
            int,
            SlowTokenRecord,
        ]
    ] = []

    # ========================================================
    # SC Exact Stats
    # ========================================================

    sc_accumulators = [
        _SubcubeAccumulator()

        for _ in range(
            index.num_subcubes
        )
    ]

    exact_count = 0

    start_time = (
        time.perf_counter()
    )

    # ========================================================
    # Token Loop
    # ========================================================

    for token in tokens:

        routes = (
            token
            .routed_experts_by_layer
        )

        if (
            len(routes)
            != index.num_layers
        ):

            raise LatencyAnalyzerError(
                f"Token-{token.token_id} "
                "Layer Route 数与 "
                "RuntimeIndex 不一致。"
            )

        # ====================================================
        # FAST Layer latency
        # ====================================================

        current_layer_cycles = []

        total_cycles = 0

        for (
            layer_id,
            route,
        ) in enumerate(
            routes
        ):

            cycles = (
                fast_schedule_layer_cycles(
                    index=index,

                    layer_id=(
                        layer_id
                    ),

                    routed_expert_ids=(
                        route
                    ),

                    rules=rules,

                    charge_initial_activation=True,
                )
            )

            layer_cycles[
                layer_id
            ].append(
                cycles
            )

            current_layer_cycles.append(
                cycles
            )

            total_cycles += (
                cycles
            )

        # ====================================================
        # Collision
        # ====================================================

        collision_score = (
            calculate_token_collision_score(
                index=index,

                routed_experts_by_layer=(
                    routes
                ),
            )
        )

        token_cycles.append(
            total_cycles
        )

        collision_scores.append(
            collision_score
        )

        # ====================================================
        # Worst Layer
        # ====================================================

        worst_layer_id = max(
            range(
                index.num_layers
            ),

            key=lambda layer_id: (
                current_layer_cycles[
                    layer_id
                ]
            ),
        )

        worst_layer_cycles = (
            current_layer_cycles[
                worst_layer_id
            ]
        )

        # ====================================================
        # Slow Token
        # ====================================================

        record = (
            SlowTokenRecord(
                token_id=(
                    token.token_id
                ),

                category=(
                    token.category
                ),

                relative_file=(
                    token.relative_file
                ),

                segment_index=(
                    token.segment_index
                ),

                token_index_in_segment=(
                    token
                    .token_index_in_segment
                ),

                total_cycles=(
                    total_cycles
                ),

                worst_layer_id=(
                    worst_layer_id
                ),

                worst_layer_cycles=(
                    worst_layer_cycles
                ),

                collision_score=(
                    collision_score
                ),
            )
        )

        heap_item = (
            total_cycles,
            token.token_id,
            record,
        )

        if (
            len(slow_heap)
            < top_k_slow_tokens
        ):

            heapq.heappush(
                slow_heap,
                heap_item,
            )

        elif (
            heap_item[
                :2
            ]
            >
            slow_heap[
                0
            ][
                :2
            ]
        ):

            heapq.heapreplace(
                slow_heap,
                heap_item,
            )

        # ====================================================
        # EXACT Sample
        # ====================================================

        if (
            exact_count
            < exact_tokens
        ):

            exact_result = (
                schedule_token(
                    index=index,

                    routed_experts_by_layer=(
                        routes
                    ),

                    rules=rules,

                    charge_initial_activation=True,
                )
            )

            if (
                exact_result
                .total_cycles
                != total_cycles
            ):

                raise LatencyAnalyzerError(
                    "FAST / EXACT 不一致："
                    f"Token-{token.token_id}, "
                    f"fast={total_cycles}, "
                    f"exact="
                    f"{exact_result.total_cycles}。"
                )

            # ================================================
            # 每层 SC 信息
            # ================================================

            for layer_execution in (
                exact_result.layers
            ):

                layer_result = (
                    layer_execution
                    .layer_result
                )

                layer_total = (
                    layer_result
                    .total_cycles
                )

                for stat in (
                    layer_result
                    .subcube_stats
                ):

                    accumulator = (
                        sc_accumulators[
                            stat.subcube_id
                        ]
                    )

                    accumulator.observed_layer_events += 1

                    accumulator.total_tasks += (
                        stat.task_count
                    )

                    accumulator.total_busy_cycles += (
                        stat.busy_cycles
                    )

                    accumulator.total_wait_cycles += (
                        stat.wait_cycles
                    )

                    accumulator.total_switches += (
                        stat.switch_count
                    )

                    accumulator.total_initial_activations += (
                        stat
                        .initial_activation_count
                    )

                    # ========================================
                    # 只有真正执行任务的 SC
                    # 才可能视作 critical。
                    # ========================================

                    if (
                        stat.task_count > 0
                        and
                        stat.last_finish_time
                        == layer_total
                    ):

                        accumulator.critical_layer_count += 1

            exact_count += 1

        # ====================================================
        # Progress
        # ====================================================

        count = len(
            token_cycles
        )

        if verbose and (
            count == 1
            or
            count % 1000 == 0
        ):

            elapsed = (
                time.perf_counter()
                - start_time
            )

            speed = (
                count / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"[Analyze] "
                f"tokens={count}, "
                f"mean="
                f"{fmean(token_cycles):.2f}, "
                f"last="
                f"{total_cycles}, "
                f"collision="
                f"{collision_score}, "
                f"speed="
                f"{speed:.1f} token/s"
            )

    # ========================================================
    # Empty
    # ========================================================

    if not token_cycles:

        raise LatencyAnalyzerError(
            "没有 Token 可以分析。"
        )

    runtime = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # Slow Tokens
    # ========================================================

    slow_tokens = tuple(
        item[2]

        for item
        in sorted(
            slow_heap,

            key=lambda item: (
                item[0],
                item[1],
            ),

            reverse=True,
        )
    )

    # ========================================================
    # Result
    # ========================================================

    return LatencyAnalysisResult(
        token_count=(
            len(
                token_cycles
            )
        ),

        exact_token_count=(
            exact_count
        ),

        runtime_seconds=(
            runtime
        ),

        mean_token_cycles=(
            fmean(
                token_cycles
            )
        ),

        min_token_cycles=(
            min(
                token_cycles
            )
        ),

        p50_token_cycles=(
            percentile_nearest_rank(
                token_cycles,
                0.50,
            )
        ),

        p95_token_cycles=(
            percentile_nearest_rank(
                token_cycles,
                0.95,
            )
        ),

        p99_token_cycles=(
            percentile_nearest_rank(
                token_cycles,
                0.99,
            )
        ),

        max_token_cycles=(
            max(
                token_cycles
            )
        ),

        mean_collision_score=(
            fmean(
                collision_scores
            )
        ),

        latency_collision_correlation=(
            pearson_correlation(
                token_cycles,
                collision_scores,
            )
        ),

        layer_stats=(
            _build_layer_stats(
                layer_cycles
            )
        ),

        subcube_stats=(
            _build_subcube_stats(
                sc_accumulators
            )
        ),

        slow_tokens=(
            slow_tokens
        ),

        trace_stats=(
            trace_stats
        ),
    )


# ============================================================
# 真实 Trace 入口
# ============================================================


def analyze_trace_latency(
    *,
    index: RuntimeIndex,

    trace_root: Path | str = (
        DEFAULT_TRACE_ROOT
    ),

    max_files: int | None = None,

    max_tokens: int | None = 10000,

    exact_tokens: int = 100,

    top_k_slow_tokens: int = 10,

    rules: ExecutionRules | None = None,

    verbose: bool = True,
) -> LatencyAnalysisResult:
    """
    直接分析 Chinese-SimpleQA。
    """

    stats = (
        TraceWorkloadStats()
    )

    tokens = (
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

            stats=stats,

            verbose=False,
        )
    )

    return analyze_tokens(
        index=index,

        tokens=tokens,

        exact_tokens=(
            exact_tokens
        ),

        top_k_slow_tokens=(
            top_k_slow_tokens
        ),

        rules=rules,

        verbose=verbose,

        trace_stats=stats,
    )


# ============================================================
# 输出
# ============================================================


def print_latency_analysis(
    result: LatencyAnalysisResult,
    *,
    top_layers: int = 10,
    top_subcubes: int = 10,
) -> None:
    """
    打印分析结果。
    """

    print(
        "\n"
        "=============================================="
    )

    print(
        "            Latency Analysis"
    )

    print(
        "=============================================="
    )

    print(
        f"Tokens："
        f"{result.token_count}"
    )

    print(
        f"EXACT Sample Tokens："
        f"{result.exact_token_count}"
    )

    print(
        f"Runtime："
        f"{result.runtime_seconds:.3f} s"
    )

    # ========================================================
    # Token latency
    # ========================================================

    print(
        "\nToken Latency："
    )

    print(
        f"  Mean："
        f"{result.mean_token_cycles:.4f}"
    )

    print(
        f"  Min："
        f"{result.min_token_cycles}"
    )

    print(
        f"  P50："
        f"{result.p50_token_cycles}"
    )

    print(
        f"  P95："
        f"{result.p95_token_cycles}"
    )

    print(
        f"  P99："
        f"{result.p99_token_cycles}"
    )

    print(
        f"  Max："
        f"{result.max_token_cycles}"
    )

    # ========================================================
    # Collision
    # ========================================================

    print(
        "\nDynamic Collision："
    )

    print(
        "  Mean Collision Score："
        f"{result.mean_collision_score:.4f}"
    )

    correlation = (
        result
        .latency_collision_correlation
    )

    if correlation is None:

        correlation_text = (
            "N/A"
        )

    else:

        correlation_text = (
            f"{correlation:.4f}"
        )

    print(
        "  Pearson("
        "Collision, Latency)："
        f"{correlation_text}"
    )

    # ========================================================
    # Layer Mean
    # ========================================================

    print(
        "\nTop Slow Layers by Mean："
    )

    slow_mean = sorted(
        result.layer_stats,

        key=lambda stat: (
            stat.mean_cycles
        ),

        reverse=True,
    )

    for stat in (
        slow_mean[
            :top_layers
        ]
    ):

        print(
            f"  L{stat.layer_id}: "
            f"mean="
            f"{stat.mean_cycles:.3f}, "
            f"P50="
            f"{stat.p50_cycles}, "
            f"P95="
            f"{stat.p95_cycles}, "
            f"P99="
            f"{stat.p99_cycles}, "
            f"max="
            f"{stat.max_cycles}"
        )

    # ========================================================
    # Layer P95
    # ========================================================

    print(
        "\nTop Slow Layers by P95："
    )

    slow_p95 = sorted(
        result.layer_stats,

        key=lambda stat: (
            stat.p95_cycles,
            stat.mean_cycles,
        ),

        reverse=True,
    )

    for stat in (
        slow_p95[
            :top_layers
        ]
    ):

        print(
            f"  L{stat.layer_id}: "
            f"P95="
            f"{stat.p95_cycles}, "
            f"mean="
            f"{stat.mean_cycles:.3f}, "
            f"max="
            f"{stat.max_cycles}"
        )

    # ========================================================
    # SC
    # ========================================================

    if (
        result.exact_token_count
        > 0
    ):

        print(
            "\nTop Critical Sub-Cubes "
            "(EXACT Sample)："
        )

        critical_sc = sorted(
            result.subcube_stats,

            key=lambda stat: (
                stat.critical_layer_count,
                stat.total_busy_cycles,
            ),

            reverse=True,
        )

        for stat in (
            critical_sc[
                :top_subcubes
            ]
        ):

            print(
                f"  SC-{stat.subcube_id}: "
                f"critical="
                f"{stat.critical_layer_count}, "
                f"rate="
                f"{stat.critical_rate:.2%}, "
                f"mean_tasks="
                f"{stat.mean_tasks_per_layer:.3f}, "
                f"mean_busy="
                f"{stat.mean_busy_cycles_per_layer:.3f}, "
                f"mean_wait="
                f"{stat.mean_wait_cycles_per_layer:.3f}, "
                f"switches="
                f"{stat.total_switches}"
            )

    # ========================================================
    # Slow Tokens
    # ========================================================

    print(
        "\nSlowest Tokens："
    )

    for (
        rank,
        token,
    ) in enumerate(
        result.slow_tokens,
        start=1,
    ):

        print(
            f"  #{rank} "
            f"Token-{token.token_id}: "
            f"cycles="
            f"{token.total_cycles}, "
            f"worst=L"
            f"{token.worst_layer_id}"
            f"({token.worst_layer_cycles}), "
            f"collision="
            f"{token.collision_score}, "
            f"{token.relative_file}, "
            f"segment="
            f"{token.segment_index}, "
            f"token="
            f"{token.token_index_in_segment}"
        )


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "分析 MoE PIM "
                "Token latency 来源。"
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
        "--max-files",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--exact-tokens",
        type=int,
        default=100,

        help=(
            "前 N 个 Token 使用 "
            "完整事件 Scheduler，"
            "用于 SC 级分析。"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
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
        analyze_trace_latency(
            index=index,

            trace_root=(
                args.trace_root
            ),

            max_files=(
                args.max_files
            ),

            max_tokens=(
                args.max_tokens
            ),

            exact_tokens=(
                args.exact_tokens
            ),

            top_k_slow_tokens=(
                args.top_k
            ),

            rules=(
                ExecutionRules()
            ),

            verbose=(
                not args.quiet
            ),
        )
    )

    print_latency_analysis(
        result,

        top_layers=(
            args.top_k
        ),

        top_subcubes=(
            args.top_k
        ),
    )


if __name__ == "__main__":
    main()