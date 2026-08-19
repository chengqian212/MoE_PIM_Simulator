"""
第六步：真实 Chinese-SimpleQA Prefill 批量评估。

前置文件：

    scheduling/prefill_workload.py
        读取原始 Trace，保留 Segment 边界

    scheduling/prefill_layer_scheduler.py
        调度：
            一个 Prefill Batch × 一个 MoE Layer

    scheduling/prefill_scheduler.py
        调度：
            一个 Prefill Batch × 58 个 MoE Layer

本文件负责：

    Chinese-SimpleQA
        ↓
    每个 JSON 的 Prefill Candidate
        ↓
    真实 token_count × 58 × Top-8
        ↓
    schedule_prefill_batch()
        ↓
    批量汇总 Prefill 指标

------------------------------------------------------------

当前数据已经扫描确认：

    2020 / 2020 JSON

均为：

    [N>1, 1, 1, 1, ...]

因此当前：

    segment0
        ->
    Prefill Candidate

后续 singleton segment：
    不在本文件评估，
    留给 Decode 评估路径。

------------------------------------------------------------

注意：

本文件输出的是：

    MoE Expert 部分的 Prefill Latency

不包含：

    Attention
    KV Cache
    Embedding
    LM Head
    其他非 MoE 模块

所以不能直接称为完整 TTFT。
"""

from __future__ import annotations

import argparse
import json
import math

from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import Path
from statistics import mean
from typing import Iterable


from config import (
    ExecutionRules,
)

from mapping.trace_profile import (
    DEFAULT_TRACE_ROOT,
)

from scheduling.prefill_workload import (
    STAGE_PREFILL_CANDIDATE,
    PrefillWorkloadStats,
    TraceSegmentBatch,
    iter_trace_segment_batches,
)

from scheduling.prefill_scheduler import (
    PrefillScheduleResult,
    schedule_prefill_batch,
)

from scheduling.runtime_index import (
    DEFAULT_MAPPING_PATH,
    RuntimeIndex,
    load_runtime_index,
)


# ============================================================
# 默认输出
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "prefill"
    / "prefill_evaluation.json"
)


# ============================================================
# 异常
# ============================================================


class PrefillEvaluatorError(
    ValueError
):
    """真实 Prefill 批量评估失败。"""


# ============================================================
# Percentile
# ============================================================


def percentile(
    values: Iterable[
        int | float
    ],
    q: float,
) -> float:
    """
    线性插值 percentile。

    q：

        0.50 -> P50
        0.95 -> P95
        0.99 -> P99
    """

    data = sorted(
        float(value)
        for value
        in values
    )

    if not data:

        return 0.0

    if not (
        0.0
        <= q
        <= 1.0
    ):

        raise PrefillEvaluatorError(
            "percentile q 必须位于 [0,1]。"
        )

    if len(data) == 1:

        return data[0]

    position = (
        (len(data) - 1)
        * q
    )

    lower = int(
        math.floor(
            position
        )
    )

    upper = int(
        math.ceil(
            position
        )
    )

    if lower == upper:

        return data[
            lower
        ]

    fraction = (
        position
        - lower
    )

    return (
        data[lower]
        * (1.0 - fraction)
        +
        data[upper]
        * fraction
    )


# ============================================================
# Pearson
# ============================================================


def pearson_correlation(
    xs: Iterable[
        int | float
    ],
    ys: Iterable[
        int | float
    ],
) -> float:
    """
    计算 Pearson correlation。

    用于观察：

        Prompt Token 数
            与
        MoE Prefill Latency

    的相关性。
    """

    x = [
        float(value)
        for value
        in xs
    ]

    y = [
        float(value)
        for value
        in ys
    ]

    if (
        len(x)
        != len(y)
    ):

        raise PrefillEvaluatorError(
            "Pearson 输入长度不一致。"
        )

    if len(x) < 2:

        return 0.0

    mean_x = mean(
        x
    )

    mean_y = mean(
        y
    )

    numerator = sum(
        (
            a - mean_x
        )
        *
        (
            b - mean_y
        )

        for a, b
        in zip(
            x,
            y,
        )
    )

    denominator_x = math.sqrt(
        sum(
            (
                value
                - mean_x
            )
            ** 2

            for value
            in x
        )
    )

    denominator_y = math.sqrt(
        sum(
            (
                value
                - mean_y
            )
            ** 2

            for value
            in y
        )
    )

    denominator = (
        denominator_x
        * denominator_y
    )

    if denominator == 0.0:

        return 0.0

    return (
        numerator
        / denominator
    )


# ============================================================
# Scalar Summary
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class ScalarSummary:
    """
    一组标量的基本分布。
    """

    count: int

    minimum: float

    mean: float

    p50: float

    p95: float

    p99: float

    maximum: float


def summarize_values(
    values: Iterable[
        int | float
    ],
) -> ScalarSummary:

    data = [
        float(value)
        for value
        in values
    ]

    if not data:

        return ScalarSummary(
            count=0,
            minimum=0.0,
            mean=0.0,
            p50=0.0,
            p95=0.0,
            p99=0.0,
            maximum=0.0,
        )

    return ScalarSummary(
        count=len(
            data
        ),

        minimum=min(
            data
        ),

        mean=float(
            mean(
                data
            )
        ),

        p50=percentile(
            data,
            0.50,
        ),

        p95=percentile(
            data,
            0.95,
        ),

        p99=percentile(
            data,
            0.99,
        ),

        maximum=max(
            data
        ),
    )


# ============================================================
# 单个真实 Prefill 记录
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PrefillEvaluationRecord:
    """
    一个 JSON / 一个 Prefill Batch 的核心结果。

    不保存全部 task，
    避免 2020 个 Batch 占用过多内存。
    """

    batch_id: int

    category: str

    relative_file: str

    segment_index: int

    input_tokens: int

    total_cycles: int

    cycles_per_input_token: float

    input_tokens_per_cycle: float

    total_tasks: int

    switches: int

    initial_activations: int

    activation_overhead_cycles: int

    compute_work_cycles: int

    busy_work_cycles: int

    wait_cycles: int

    max_task_wait_cycles: int

    layer_cycles: tuple[
        int,
        ...
    ]

    subcube_busy_cycles: tuple[
        int,
        ...
    ]

    subcube_switches: tuple[
        int,
        ...
    ]


# ============================================================
# Layer Aggregate
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class LayerEvaluationSummary:

    layer_id: int

    cycles: ScalarSummary

    switch_mean: float

    wait_mean: float


# ============================================================
# SC Aggregate
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class SubcubeEvaluationSummary:

    subcube_id: int

    task_count: int

    busy_cycles: int

    switch_count: int

    initial_activation_count: int

    wait_cycles: int

    critical_layer_count: int

    # ========================================================
    # 注意：
    #
    # 分母是：
    #
    #     所有 Prefill Batch 的 total_cycles 之和
    #
    # 所以这是跨全部真实请求的
    # 时间加权整体利用率。
    # ========================================================

    weighted_utilization: float


# ============================================================
# Category
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class CategoryEvaluationSummary:

    category: str

    batch_count: int

    input_tokens: ScalarSummary

    total_cycles: ScalarSummary

    mean_cycles_per_input_token: float


# ============================================================
# Final Summary
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PrefillEvaluationSummary:

    batch_count: int

    total_input_tokens: int

    prompt_tokens: ScalarSummary

    total_cycles: ScalarSummary

    cycles_per_input_token: ScalarSummary

    input_tokens_per_cycle: ScalarSummary

    switches: ScalarSummary

    wait_cycles: ScalarSummary

    # ========================================================
    # 两种整体归一化值
    #
    # global_cycles_per_input_token:
    #
    #     sum(batch latency)
    #     /
    #     sum(batch input tokens)
    #
    # global_input_tokens_per_cycle:
    #
    #     sum(batch input tokens)
    #     /
    #     sum(batch latency)
    #
    # ========================================================

    global_cycles_per_input_token: float

    global_input_tokens_per_cycle: float

    prompt_length_latency_pearson: float

    layers: tuple[
        LayerEvaluationSummary,
        ...
    ]

    subcubes: tuple[
        SubcubeEvaluationSummary,
        ...
    ]

    categories: tuple[
        CategoryEvaluationSummary,
        ...
    ]


# ============================================================
# 单 Batch 转 Record
# ============================================================


def make_record(
    *,
    batch: TraceSegmentBatch,
    result: PrefillScheduleResult,
) -> PrefillEvaluationRecord:

    if (
        batch.token_count
        != result.token_count
    ):

        raise PrefillEvaluatorError(
            "Batch token_count "
            "与 Scheduler 结果不一致。"
        )

    return PrefillEvaluationRecord(
        batch_id=(
            batch.batch_id
        ),

        category=(
            batch.category
        ),

        relative_file=(
            batch.relative_file
        ),

        segment_index=(
            batch.segment_index
        ),

        input_tokens=(
            batch.token_count
        ),

        total_cycles=(
            result.total_cycles
        ),

        cycles_per_input_token=(
            result
            .cycles_per_input_token
        ),

        input_tokens_per_cycle=(
            result
            .input_tokens_per_cycle
        ),

        total_tasks=(
            result.total_tasks
        ),

        switches=(
            result.total_switches
        ),

        initial_activations=(
            result
            .total_initial_activations
        ),

        activation_overhead_cycles=(
            result
            .total_activation_overhead_cycles
        ),

        compute_work_cycles=(
            result
            .total_compute_work_cycles
        ),

        busy_work_cycles=(
            result.total_busy_cycles
        ),

        wait_cycles=(
            result.total_wait_cycles
        ),

        max_task_wait_cycles=(
            result
            .max_task_wait_cycles
        ),

        layer_cycles=tuple(
            execution.cycles

            for execution
            in result.layers
        ),

        subcube_busy_cycles=tuple(
            stat.busy_cycles

            for stat
            in result.subcube_stats
        ),

        subcube_switches=tuple(
            stat.switch_count

            for stat
            in result.subcube_stats
        ),
    )


# ============================================================
# Summary Builder
# ============================================================


def build_summary(
    *,
    records: list[
        PrefillEvaluationRecord
    ],

    layer_cycle_values: list[
        list[int]
    ],

    layer_switch_values: list[
        list[int]
    ],

    layer_wait_values: list[
        list[int]
    ],

    sc_task_count: list[int],

    sc_busy_cycles: list[int],

    sc_switch_count: list[int],

    sc_initial_count: list[int],

    sc_wait_cycles: list[int],

    sc_critical_layer_count: list[int],
) -> PrefillEvaluationSummary:

    if not records:

        raise PrefillEvaluatorError(
            "没有可汇总的 Prefill Batch。"
        )

    prompt_lengths = [
        record.input_tokens
        for record
        in records
    ]

    latencies = [
        record.total_cycles
        for record
        in records
    ]

    cycles_per_token = [
        record.cycles_per_input_token
        for record
        in records
    ]

    tokens_per_cycle = [
        record.input_tokens_per_cycle
        for record
        in records
    ]

    switches = [
        record.switches
        for record
        in records
    ]

    waits = [
        record.wait_cycles
        for record
        in records
    ]

    total_input_tokens = sum(
        prompt_lengths
    )

    total_latency_cycles = sum(
        latencies
    )

    # ========================================================
    # Layer
    # ========================================================

    layers: list[
        LayerEvaluationSummary
    ] = []

    for layer_id in range(
        len(
            layer_cycle_values
        )
    ):

        cycle_values = (
            layer_cycle_values[
                layer_id
            ]
        )

        switch_values = (
            layer_switch_values[
                layer_id
            ]
        )

        wait_values = (
            layer_wait_values[
                layer_id
            ]
        )

        layers.append(
            LayerEvaluationSummary(
                layer_id=(
                    layer_id
                ),

                cycles=(
                    summarize_values(
                        cycle_values
                    )
                ),

                switch_mean=(
                    float(
                        mean(
                            switch_values
                        )
                    )
                    if switch_values
                    else 0.0
                ),

                wait_mean=(
                    float(
                        mean(
                            wait_values
                        )
                    )
                    if wait_values
                    else 0.0
                ),
            )
        )

    # ========================================================
    # Sub-Cube
    # ========================================================

    subcubes: list[
        SubcubeEvaluationSummary
    ] = []

    for sc in range(
        len(
            sc_task_count
        )
    ):

        utilization = (
            sc_busy_cycles[
                sc
            ]
            / total_latency_cycles
            if total_latency_cycles > 0
            else 0.0
        )

        subcubes.append(
            SubcubeEvaluationSummary(
                subcube_id=sc,

                task_count=(
                    sc_task_count[
                        sc
                    ]
                ),

                busy_cycles=(
                    sc_busy_cycles[
                        sc
                    ]
                ),

                switch_count=(
                    sc_switch_count[
                        sc
                    ]
                ),

                initial_activation_count=(
                    sc_initial_count[
                        sc
                    ]
                ),

                wait_cycles=(
                    sc_wait_cycles[
                        sc
                    ]
                ),

                critical_layer_count=(
                    sc_critical_layer_count[
                        sc
                    ]
                ),

                weighted_utilization=(
                    utilization
                ),
            )
        )

    # ========================================================
    # Category
    # ========================================================

    grouped: dict[
        str,
        list[
            PrefillEvaluationRecord
        ],
    ] = {}

    for record in records:

        grouped.setdefault(
            record.category,
            [],
        ).append(
            record
        )

    categories: list[
        CategoryEvaluationSummary
    ] = []

    for category in sorted(
        grouped
    ):

        items = grouped[
            category
        ]

        category_tokens = [
            item.input_tokens
            for item
            in items
        ]

        category_cycles = [
            item.total_cycles
            for item
            in items
        ]

        category_cycles_per_token = [
            item.cycles_per_input_token
            for item
            in items
        ]

        categories.append(
            CategoryEvaluationSummary(
                category=(
                    category
                ),

                batch_count=len(
                    items
                ),

                input_tokens=(
                    summarize_values(
                        category_tokens
                    )
                ),

                total_cycles=(
                    summarize_values(
                        category_cycles
                    )
                ),

                mean_cycles_per_input_token=(
                    float(
                        mean(
                            category_cycles_per_token
                        )
                    )
                ),
            )
        )

    return PrefillEvaluationSummary(
        batch_count=len(
            records
        ),

        total_input_tokens=(
            total_input_tokens
        ),

        prompt_tokens=(
            summarize_values(
                prompt_lengths
            )
        ),

        total_cycles=(
            summarize_values(
                latencies
            )
        ),

        cycles_per_input_token=(
            summarize_values(
                cycles_per_token
            )
        ),

        input_tokens_per_cycle=(
            summarize_values(
                tokens_per_cycle
            )
        ),

        switches=(
            summarize_values(
                switches
            )
        ),

        wait_cycles=(
            summarize_values(
                waits
            )
        ),

        global_cycles_per_input_token=(
            total_latency_cycles
            / total_input_tokens
            if total_input_tokens > 0
            else 0.0
        ),

        global_input_tokens_per_cycle=(
            total_input_tokens
            / total_latency_cycles
            if total_latency_cycles > 0
            else 0.0
        ),

        prompt_length_latency_pearson=(
            pearson_correlation(
                prompt_lengths,
                latencies,
            )
        ),

        layers=tuple(
            layers
        ),

        subcubes=tuple(
            subcubes
        ),

        categories=tuple(
            categories
        ),
    )


# ============================================================
# 主 Evaluator
# ============================================================


def evaluate_prefill_workload(
    *,
    index: RuntimeIndex,

    trace_root: Path | str = (
        DEFAULT_TRACE_ROOT
    ),

    rules: ExecutionRules | None = None,

    max_files: int | None = None,

    max_batches: int | None = None,

    charge_initial_activation: bool = True,

    progress_every: int = 50,

    verbose: bool = True,
) -> tuple[
    PrefillEvaluationSummary,
    tuple[
        PrefillEvaluationRecord,
        ...
    ],
]:
    """
    批量评估真实 Prefill。

    只处理：

        batch.stage
        ==
        prefill_candidate

    即当前数据中的 segment0。

    --------------------------------------------------------

    每一个 Prefill Batch：

        独立从：

            None × 16

        开始。

    也就是说：

        不把 JSON-A 的最后 active cube
        传给 JSON-B。

    这是当前 latency evaluation
    更清晰的请求级口径。
    """

    if rules is None:

        rules = (
            ExecutionRules()
        )

    if (
        max_files
        is not None
        and
        max_files <= 0
    ):

        raise PrefillEvaluatorError(
            "max_files 必须大于 0。"
        )

    if (
        max_batches
        is not None
        and
        max_batches <= 0
    ):

        raise PrefillEvaluatorError(
            "max_batches 必须大于 0。"
        )

    if progress_every <= 0:

        raise PrefillEvaluatorError(
            "progress_every 必须大于 0。"
        )

    workload_stats = (
        PrefillWorkloadStats()
    )

    records: list[
        PrefillEvaluationRecord
    ] = []

    # ========================================================
    # Layer aggregates
    # ========================================================

    layer_cycle_values: list[
        list[int]
    ] = [
        []
        for _ in range(
            index.num_layers
        )
    ]

    layer_switch_values: list[
        list[int]
    ] = [
        []
        for _ in range(
            index.num_layers
        )
    ]

    layer_wait_values: list[
        list[int]
    ] = [
        []
        for _ in range(
            index.num_layers
        )
    ]

    # ========================================================
    # SC aggregates
    # ========================================================

    sc_task_count = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    sc_busy_cycles = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    sc_switch_count = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    sc_initial_count = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    sc_wait_cycles = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    # ========================================================
    # Critical SC：
    #
    # 对每个 Batch / 每个 Layer，
    # 哪个 SC 的 last_finish_time
    # 等于当前 Layer total_cycles，
    # 就认为它落在这一层关键路径末端。
    #
    # 允许并列。
    # ========================================================

    sc_critical_layer_count = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    evaluated = 0

    # ========================================================
    # Workload
    # ========================================================

    for batch in (
        iter_trace_segment_batches(
            trace_root=(
                trace_root
            ),

            max_files=(
                max_files
            ),

            # 注意：
            #
            # 这里故意不传 max_batches。
            #
            # prefill_workload.py 的 max_batches
            # 统计的是所有 Segment，
            # 包括 Decode singleton。
            #
            # 我们这里需要的是：
            #
            #     最多评估 N 个 Prefill Batch。
            max_batches=None,

            stats=(
                workload_stats
            ),

            verbose=False,
        )
    ):

        # ====================================================
        # 只取 Prefill
        # ====================================================

        if (
            batch.stage
            != STAGE_PREFILL_CANDIDATE
        ):

            continue

        # ====================================================
        # 真实 58 层 Prefill
        #
        # 每个 JSON 独立冷启动：
        #
        # initial_active_cube_by_subcube=None
        # ====================================================

        result = (
            schedule_prefill_batch(
                index=index,

                routed_experts_by_token=(
                    batch
                    .routed_experts_by_token
                ),

                rules=(
                    rules
                ),

                initial_active_cube_by_subcube=None,

                charge_initial_activation=(
                    charge_initial_activation
                ),
            )
        )

        record = (
            make_record(
                batch=batch,
                result=result,
            )
        )

        records.append(
            record
        )

        evaluated += 1

        # ====================================================
        # Layer
        # ====================================================

        for execution in (
            result.layers
        ):

            layer_id = (
                execution.layer_id
            )

            layer_result = (
                execution.layer_result
            )

            layer_cycle_values[
                layer_id
            ].append(
                layer_result.total_cycles
            )

            layer_switch_values[
                layer_id
            ].append(
                layer_result.switch_count
            )

            layer_wait_values[
                layer_id
            ].append(
                layer_result.wait_cycles
            )

            # ================================================
            # Critical SC
            # ================================================

            for stat in (
                layer_result
                .subcube_stats
            ):

                if (
                    stat.task_count
                    > 0

                    and

                    stat.last_finish_time
                    == layer_result.total_cycles
                ):

                    sc_critical_layer_count[
                        stat.subcube_id
                    ] += 1

        # ====================================================
        # SC
        # ====================================================

        for stat in (
            result.subcube_stats
        ):

            sc = (
                stat.subcube_id
            )

            sc_task_count[
                sc
            ] += (
                stat.task_count
            )

            sc_busy_cycles[
                sc
            ] += (
                stat.busy_cycles
            )

            sc_switch_count[
                sc
            ] += (
                stat.switch_count
            )

            sc_initial_count[
                sc
            ] += (
                stat.initial_activation_count
            )

            sc_wait_cycles[
                sc
            ] += (
                stat.wait_cycles
            )

        # ====================================================
        # Progress
        # ====================================================

        if (
            verbose

            and

            (
                evaluated == 1
                or
                evaluated
                % progress_every
                == 0
            )
        ):

            current_mean = float(
                mean(
                    item.total_cycles
                    for item
                    in records
                )
            )

            print(
                "[PrefillEval] "
                f"batches={evaluated}, "
                f"last_tokens="
                f"{batch.token_count}, "
                f"last_cycles="
                f"{result.total_cycles}, "
                f"mean_cycles="
                f"{current_mean:.2f}"
            )

        # ====================================================
        # max_batches：
        #
        # 这里按 Prefill Batch 数量截断。
        # ====================================================

        if (
            max_batches
            is not None

            and

            evaluated
            >= max_batches
        ):

            break

    if not records:

        raise PrefillEvaluatorError(
            "没有找到 Prefill Candidate。"
        )

    summary = (
        build_summary(
            records=(
                records
            ),

            layer_cycle_values=(
                layer_cycle_values
            ),

            layer_switch_values=(
                layer_switch_values
            ),

            layer_wait_values=(
                layer_wait_values
            ),

            sc_task_count=(
                sc_task_count
            ),

            sc_busy_cycles=(
                sc_busy_cycles
            ),

            sc_switch_count=(
                sc_switch_count
            ),

            sc_initial_count=(
                sc_initial_count
            ),

            sc_wait_cycles=(
                sc_wait_cycles
            ),

            sc_critical_layer_count=(
                sc_critical_layer_count
            ),
        )
    )

    return (
        summary,
        tuple(
            records
        ),
    )


# ============================================================
# Print
# ============================================================


def _print_scalar_summary(
    *,
    name: str,
    summary: ScalarSummary,
    decimals: int = 4,
) -> None:

    print(
        f"\n{name}："
    )

    print(
        f"  Count："
        f"{summary.count}"
    )

    print(
        f"  Min："
        f"{summary.minimum:.{decimals}f}"
    )

    print(
        f"  Mean："
        f"{summary.mean:.{decimals}f}"
    )

    print(
        f"  P50："
        f"{summary.p50:.{decimals}f}"
    )

    print(
        f"  P95："
        f"{summary.p95:.{decimals}f}"
    )

    print(
        f"  P99："
        f"{summary.p99:.{decimals}f}"
    )

    print(
        f"  Max："
        f"{summary.maximum:.{decimals}f}"
    )


def print_prefill_evaluation_summary(
    summary: PrefillEvaluationSummary,
    *,
    top_layers: int = 10,
    top_subcubes: int = 10,
) -> None:

    print(
        "\n"
        "========== Real Prefill Evaluation =========="
    )

    print(
        f"Prefill Batches："
        f"{summary.batch_count}"
    )

    print(
        f"Total Input Tokens："
        f"{summary.total_input_tokens}"
    )

    _print_scalar_summary(
        name=(
            "Prompt Length "
            "(Input Tokens)"
        ),

        summary=(
            summary.prompt_tokens
        ),

        decimals=2,
    )

    _print_scalar_summary(
        name=(
            "MoE Prefill Total Cycles"
        ),

        summary=(
            summary.total_cycles
        ),

        decimals=2,
    )

    _print_scalar_summary(
        name=(
            "Cycles / Input Token"
        ),

        summary=(
            summary
            .cycles_per_input_token
        ),

        decimals=4,
    )

    _print_scalar_summary(
        name=(
            "Input Tokens / Cycle"
        ),

        summary=(
            summary
            .input_tokens_per_cycle
        ),

        decimals=6,
    )

    _print_scalar_summary(
        name=(
            "Weight-Cube Switches / Batch"
        ),

        summary=(
            summary.switches
        ),

        decimals=2,
    )

    print(
        "\nGlobal Normalized Metrics："
    )

    print(
        "  Sum Cycles / Sum Input Tokens："
        f"{summary.global_cycles_per_input_token:.4f}"
    )

    print(
        "  Sum Input Tokens / Sum Cycles："
        f"{summary.global_input_tokens_per_cycle:.6f}"
    )

    print(
        "\nPrompt Length vs Prefill Latency："
    )

    print(
        "  Pearson："
        f"{summary.prompt_length_latency_pearson:.4f}"
    )

    # ========================================================
    # Slow Layers
    # ========================================================

    ranked_layers = sorted(
        summary.layers,

        key=lambda item: (
            -item.cycles.mean,
            item.layer_id,
        ),
    )

    print(
        "\n"
        f"Top-{min(top_layers, len(ranked_layers))} "
        "Layers by Mean Prefill Cycles："
    )

    for item in (
        ranked_layers[
            :top_layers
        ]
    ):

        print(
            f"  L{item.layer_id}: "
            f"mean="
            f"{item.cycles.mean:.2f}, "
            f"p95="
            f"{item.cycles.p95:.2f}, "
            f"max="
            f"{item.cycles.maximum:.2f}, "
            f"switch_mean="
            f"{item.switch_mean:.2f}, "
            f"wait_mean="
            f"{item.wait_mean:.2f}"
        )

    # ========================================================
    # Critical SC
    # ========================================================

    ranked_sc = sorted(
        summary.subcubes,

        key=lambda item: (
            -item.critical_layer_count,
            -item.weighted_utilization,
            item.subcube_id,
        ),
    )

    print(
        "\n"
        f"Top-{min(top_subcubes, len(ranked_sc))} "
        "Critical Sub-Cubes："
    )

    for item in (
        ranked_sc[
            :top_subcubes
        ]
    ):

        print(
            f"  SC-{item.subcube_id}: "
            f"critical_layers="
            f"{item.critical_layer_count}, "
            f"util="
            f"{item.weighted_utilization:.2%}, "
            f"switch="
            f"{item.switch_count}, "
            f"wait="
            f"{item.wait_cycles}"
        )

    # ========================================================
    # Category
    # ========================================================

    print(
        "\nCategory Summary："
    )

    for item in (
        summary.categories
    ):

        print(
            f"  {item.category}: "
            f"batches="
            f"{item.batch_count}, "
            f"mean_tokens="
            f"{item.input_tokens.mean:.2f}, "
            f"mean_cycles="
            f"{item.total_cycles.mean:.2f}, "
            f"mean_cycles/token="
            f"{item.mean_cycles_per_input_token:.4f}"
        )


# ============================================================
# JSON
# ============================================================


def save_prefill_evaluation(
    *,
    output_path: Path | str,

    summary: PrefillEvaluationSummary,

    records: Iterable[
        PrefillEvaluationRecord
    ],

    mapping_path: Path | str,

    trace_root: Path | str,

    charge_initial_activation: bool,
) -> Path:

    path = Path(
        output_path
    ).resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "evaluation_version": 1,

        "metric_scope": (
            "MoE Expert Prefill only; "
            "not full-model TTFT"
        ),

        "mapping": str(
            Path(
                mapping_path
            ).resolve()
        ),

        "trace_root": str(
            Path(
                trace_root
            ).resolve()
        ),

        "charge_initial_activation": (
            charge_initial_activation
        ),

        "summary": (
            asdict(
                summary
            )
        ),

        "records": [
            asdict(
                record
            )

            for record
            in records
        ],
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return path


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "批量评估 Chinese-SimpleQA "
                "真实 segment0 Prefill。"
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
        "--root",
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
        "--max-batches",
        type=int,
        default=None,

        help=(
            "最多评估多少个 Prefill Batch。"
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--top-layers",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--top-subcubes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
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

    summary, records = (
        evaluate_prefill_workload(
            index=index,

            trace_root=(
                args.root
            ),

            rules=(
                ExecutionRules()
            ),

            max_files=(
                args.max_files
            ),

            max_batches=(
                args.max_batches
            ),

            charge_initial_activation=(
                not args
                .no_initial_activation_cost
            ),

            progress_every=(
                args.progress_every
            ),

            verbose=(
                not args.quiet
            ),
        )
    )

    print_prefill_evaluation_summary(
        summary,

        top_layers=(
            args.top_layers
        ),

        top_subcubes=(
            args.top_subcubes
        ),
    )

    if not args.no_save:

        saved = (
            save_prefill_evaluation(
                output_path=(
                    args.output
                ),

                summary=(
                    summary
                ),

                records=(
                    records
                ),

                mapping_path=(
                    args.mapping
                ),

                trace_root=(
                    args.root
                ),

                charge_initial_activation=(
                    not args
                    .no_initial_activation_cost
                ),
            )
        )

        print(
            "\nSaved："
            f"{saved}"
        )


if __name__ == "__main__":
    main()
