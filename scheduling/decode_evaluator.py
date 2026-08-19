"""
纯 Decode 批量评估（连续请求状态版）。

正式数据口径：

    一个 JSON = 一个完整请求

    segment0
        -> Prefill
        -> 只用于生成 Prefill 结束后的 16 个 SC active WC 状态
        -> 不计入 Decode cycles/token

    segment1
        -> Decode Token-1
        -> 继承 Prefill final state

    segment2
        -> Decode Token-2
        -> 继承 Decode Token-1 final state

    ...

即：

    Prefill final state
        ↓
    Decode-1 final state
        ↓
    Decode-2 final state
        ↓
    ...

不同 JSON 之间互相独立：
    每个新请求重新从 None × 16 开始执行自己的 Prefill。

------------------------------------------------------------

核心指标：

    MoE Decode Cycles / Token

不包含：

    Attention
    KV Cache
    Embedding
    LM Head
    其他非 MoE 模块

因此不能直接称为完整模型 TPOT。

------------------------------------------------------------

调试兼容：

    --cold-start-each-token

会恢复旧 evaluator 行为：

    每个 Decode Token 都从 None × 16 开始。

这个选项只用于回归检查，不作为正式 Decode 口径。
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


from config import ExecutionRules

from mapping.trace_profile import (
    DEFAULT_TRACE_ROOT,
    NUM_MOE_LAYERS,
)

from scheduling.decode_workload import (
    DecodeWorkloadStats,
    iter_decode_tokens,
)

from scheduling.prefill_scheduler import (
    schedule_prefill_batch,
)

from scheduling.runtime_index import (
    DEFAULT_MAPPING_PATH,
    RuntimeIndex,
    load_runtime_index,
)

from scheduling.token_scheduler import (
    TokenScheduleResult,
    schedule_token,
)

from scheduling.trace_workload import (
    TraceToken,
    TraceWorkloadError,
    collect_segment_routes,
    validate_runtime_route,
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
    / "decode"
    / "decode_evaluation.json"
)


# ============================================================
# 异常
# ============================================================


class DecodeEvaluatorError(
    ValueError
):
    """纯 Decode 评估失败。"""


# ============================================================
# 基础统计
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class ScalarSummary:

    count: int

    minimum: float

    mean: float

    p50: float

    p95: float

    p99: float

    maximum: float


@dataclass(
    frozen=True,
    slots=True,
)
class DecodeEvaluationRecord:

    token_id: int

    category: str

    relative_file: str

    segment_index: int

    token_index_in_segment: int

    total_cycles: int

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


@dataclass(
    frozen=True,
    slots=True,
)
class DecodeLayerSummary:

    layer_id: int

    cycles: ScalarSummary

    switch_mean: float

    wait_mean: float


@dataclass(
    frozen=True,
    slots=True,
)
class DecodeSubcubeSummary:

    subcube_id: int

    task_count: int

    busy_cycles: int

    switch_count: int

    initial_activation_count: int

    wait_cycles: int

    critical_layer_count: int

    critical_rate: float


@dataclass(
    frozen=True,
    slots=True,
)
class DecodeCategorySummary:

    category: str

    token_count: int

    cycles: ScalarSummary


@dataclass(
    frozen=True,
    slots=True,
)
class SlowDecodeToken:

    token_id: int

    total_cycles: int

    category: str

    relative_file: str

    segment_index: int


@dataclass(
    frozen=True,
    slots=True,
)
class DecodeEvaluationSummary:

    state_mode: str

    request_count: int

    token_count: int

    total_tasks_per_token: int

    cycles_per_token: ScalarSummary

    switches_per_token: ScalarSummary

    initial_activations_per_token: ScalarSummary

    wait_cycles_per_token: ScalarSummary

    layers: tuple[
        DecodeLayerSummary,
        ...
    ]

    subcubes: tuple[
        DecodeSubcubeSummary,
        ...
    ]

    categories: tuple[
        DecodeCategorySummary,
        ...
    ]

    slowest_tokens: tuple[
        SlowDecodeToken,
        ...
    ]


# ============================================================
# Percentile
# ============================================================


def percentile(
    values: Iterable[
        int | float
    ],
    q: float,
) -> float:

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

        raise DecodeEvaluatorError(
            "percentile q 必须位于 [0,1]。"
        )

    if len(data) == 1:

        return data[
            0
        ]

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
        data[
            lower
        ]
        * (
            1.0
            - fraction
        )
        +
        data[
            upper
        ]
        * fraction
    )


def summarize(
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
# 从某个 JSON 的 segment0 恢复真实 Prefill Route
# ============================================================


def load_prefill_routes_for_request(
    *,
    trace_root: Path | str,
    relative_file: str,
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
    返回 shape：

        PrefillToken × 58 Layer × Top-8

    这里只读取目标 JSON 的 segment0。

    这样 Decode evaluator 不需要为了给一个请求取
    Prefill state 而重新扫描全部 2020 个文件。
    """

    root = (
        Path(
            trace_root
        )
        .resolve()
    )

    path = (
        root
        / relative_file
    ).resolve()

    # ========================================================
    # 防止 relative_file 跑出 root
    # ========================================================

    try:

        path.relative_to(
            root
        )

    except ValueError as exc:

        raise DecodeEvaluatorError(
            f"Trace 文件不在 root 内："
            f"{relative_file}"
        ) from exc

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = (
                json.load(
                    file
                )
            )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise DecodeEvaluatorError(
            f"无法读取请求 Trace："
            f"{path}"
        ) from exc

    if (
        not isinstance(
            data,
            list,
        )
        or
        not data
    ):

        raise DecodeEvaluatorError(
            f"{path}: "
            "JSON 最外层必须是非空 list。"
        )

    # ========================================================
    # segment0
    # ========================================================

    segment0 = (
        data[
            0
        ]
    )

    if not isinstance(
        segment0,
        dict,
    ):

        raise DecodeEvaluatorError(
            f"{path}: segment0 必须是 dict。"
        )

    raw_routes_by_layer = (
        collect_segment_routes(
            segment=(
                segment0
            )
        )
    )

    if (
        raw_routes_by_layer
        is None
        or
        not raw_routes_by_layer
    ):

        raise DecodeEvaluatorError(
            f"{path}: segment0 "
            "不是完整 Prefill Segment。"
        )

    if (
        len(
            raw_routes_by_layer
        )
        != NUM_MOE_LAYERS
    ):

        raise DecodeEvaluatorError(
            f"{path}: segment0 "
            "MoE Layer 数错误。"
        )

    token_count = len(
        raw_routes_by_layer[
            0
        ][
            1
        ]
    )

    if (
        token_count
        <= 1
    ):

        raise DecodeEvaluatorError(
            f"{path}: segment0 "
            "不是多 Token Prefill："
            f"token_count={token_count}。"
        )

    # ========================================================
    # Layer-major：
    #
    # validated_by_layer[layer][token]
    # ========================================================

    validated_by_layer: list[
        list[
            tuple[
                int,
                ...
            ]
        ]
    ] = []

    for (
        trace_layer_id,
        routes,
    ) in raw_routes_by_layer:

        if (
            len(
                routes
            )
            != token_count
        ):

            raise DecodeEvaluatorError(
                f"{path}: segment0 "
                "各 Layer Token 数不一致。"
            )

        layer_routes: list[
            tuple[
                int,
                ...
            ]
        ] = []

        for (
            token_index,
            raw_route,
        ) in enumerate(
            routes
        ):

            try:

                route = (
                    validate_runtime_route(
                        route=(
                            raw_route
                        ),

                        path=(
                            path
                        ),

                        segment_index=0,

                        trace_layer_id=(
                            trace_layer_id
                        ),

                        token_index=(
                            token_index
                        ),
                    )
                )

            except TraceWorkloadError as exc:

                raise DecodeEvaluatorError(
                    str(
                        exc
                    )
                ) from exc

            layer_routes.append(
                route
            )

        validated_by_layer.append(
            layer_routes
        )

    # ========================================================
    # 转置：
    #
    # Layer × Token × Top8
    #
    # ->
    #
    # Token × Layer × Top8
    # ========================================================

    routed_experts_by_token = tuple(
        tuple(
            validated_by_layer[
                layer_id
            ][
                token_index
            ]

            for layer_id
            in range(
                NUM_MOE_LAYERS
            )
        )

        for token_index
        in range(
            token_count
        )
    )

    return (
        routed_experts_by_token
    )


# ============================================================
# Record
# ============================================================


def make_record(
    *,
    token: TraceToken,
    result: TokenScheduleResult,
) -> DecodeEvaluationRecord:

    max_wait = max(
        (
            execution
            .layer_result
            .max_task_wait_cycles

            for execution
            in result.layers
        ),
        default=0,
    )

    return DecodeEvaluationRecord(
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
            token.token_index_in_segment
        ),

        total_cycles=(
            result.total_cycles
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
            result
            .total_busy_cycles
        ),

        wait_cycles=(
            result
            .total_wait_cycles
        ),

        max_task_wait_cycles=(
            max_wait
        ),

        layer_cycles=tuple(
            execution.cycles

            for execution
            in result.layers
        ),
    )


# ============================================================
# 主 Evaluator
# ============================================================


def evaluate_decode_workload(
    *,
    index: RuntimeIndex,

    trace_root: Path | str = (
        DEFAULT_TRACE_ROOT
    ),

    rules: ExecutionRules | None = None,

    max_files: int | None = None,

    max_tokens: int | None = None,

    charge_initial_activation: bool = True,

    progress_every: int = 100,

    top_slowest_tokens: int = 10,

    continuous_request_state: bool = True,

    verbose: bool = True,
) -> tuple[
    DecodeEvaluationSummary,
    tuple[
        DecodeEvaluationRecord,
        ...
    ],
    DecodeWorkloadStats,
]:

    if rules is None:

        rules = (
            ExecutionRules()
        )

    if (
        max_tokens
        is not None
        and
        max_tokens <= 0
    ):

        raise DecodeEvaluatorError(
            "max_tokens 必须大于 0。"
        )

    if (
        max_files
        is not None
        and
        max_files <= 0
    ):

        raise DecodeEvaluatorError(
            "max_files 必须大于 0。"
        )

    if (
        progress_every
        <= 0
    ):

        raise DecodeEvaluatorError(
            "progress_every 必须大于 0。"
        )

    # ========================================================
    # Stats
    # ========================================================

    stats = (
        DecodeWorkloadStats()
    )

    records: list[
        DecodeEvaluationRecord
    ] = []

    # ========================================================
    # Layer aggregates
    # ========================================================

    layer_cycles = [
        []
        for _ in range(
            index.num_layers
        )
    ]

    layer_switches = [
        []
        for _ in range(
            index.num_layers
        )
    ]

    layer_waits = [
        []
        for _ in range(
            index.num_layers
        )
    ]

    # ========================================================
    # SC aggregates
    # ========================================================

    sc_tasks = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    sc_busy = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    sc_switch = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    sc_initial = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    sc_wait = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    sc_critical = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    # ========================================================
    # Request-level active state
    # ========================================================

    current_file: str | None = (
        None
    )

    current_state: tuple[
        int | None,
        ...
    ] | None = (
        None
    )

    previous_segment_index = 0

    request_count = 0

    # ========================================================
    # Decode Tokens
    # ========================================================

    for token in (
        iter_decode_tokens(
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
                stats
            ),

            strict_singleton=True,

            verbose=False,
        )
    ):

        # ====================================================
        # 进入一个新的 JSON / Request
        # ====================================================

        if (
            token.relative_file
            != current_file
        ):

            current_file = (
                token.relative_file
            )

            previous_segment_index = 0

            request_count += 1

            if (
                continuous_request_state
            ):

                # ============================================
                # 重新读取这个 JSON 的 segment0，
                # 跑真实 Prefill，
                # 只拿 final active state。
                #
                # Prefill cycles 不加入 Decode metric。
                # ============================================

                prefill_routes = (
                    load_prefill_routes_for_request(
                        trace_root=(
                            trace_root
                        ),

                        relative_file=(
                            current_file
                        ),
                    )
                )

                prefill_result = (
                    schedule_prefill_batch(
                        index=index,

                        routed_experts_by_token=(
                            prefill_routes
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

                current_state = (
                    prefill_result
                    .final_active_cube_by_subcube
                )

                if (
                    verbose
                    and
                    (
                        request_count
                        == 1
                        or
                        request_count
                        % 100
                        == 0
                    )
                ):

                    print(
                        "[DecodeRequest] "
                        f"request="
                        f"{request_count}, "
                        f"file="
                        f"{current_file}, "
                        f"prefill_tokens="
                        f"{prefill_result.token_count}, "
                        f"prefill_cycles="
                        f"{prefill_result.total_cycles}, "
                        "state_seeded=yes"
                    )

            else:

                # ============================================
                # 旧行为：
                # 每个 Token 都冷启动。
                # ============================================

                current_state = (
                    None
                )

        # ====================================================
        # 同一 Request 内 Segment 必须递增
        # ====================================================

        if (
            token.segment_index
            <= previous_segment_index
        ):

            raise DecodeEvaluatorError(
                f"{token.relative_file}: "
                "Decode Segment 顺序异常："
                f"previous="
                f"{previous_segment_index}, "
                f"current="
                f"{token.segment_index}。"
            )

        previous_segment_index = (
            token.segment_index
        )

        # ====================================================
        # 正式 Decode
        # ====================================================

        if (
            continuous_request_state
        ):

            initial_state = (
                current_state
            )

        else:

            # 旧冷启动口径：
            # 每个 Token 都 None × 16。
            initial_state = (
                None
            )

        result = (
            schedule_token(
                index=index,

                routed_experts_by_layer=(
                    token
                    .routed_experts_by_layer
                ),

                rules=(
                    rules
                ),

                initial_active_cube_by_subcube=(
                    initial_state
                ),

                charge_initial_activation=(
                    charge_initial_activation
                ),
            )
        )

        # ====================================================
        # 连续模式下检查：
        # schedule_token 确实吃到了上一阶段状态。
        # ====================================================

        if (
            continuous_request_state
        ):

            expected_state = tuple(
                current_state
                if current_state
                is not None
                else (
                    None
                    for _ in range(
                        index.num_subcubes
                    )
                )
            )

            if (
                result
                .initial_active_cube_by_subcube
                != expected_state
            ):

                raise DecodeEvaluatorError(
                    "Decode initial state "
                    "没有正确继承上一阶段 final state。"
                )

            # ================================================
            # Decode-i final
            # ->
            # Decode-(i+1) initial
            # ================================================

            current_state = (
                result
                .final_active_cube_by_subcube
            )

        record = (
            make_record(
                token=(
                    token
                ),

                result=(
                    result
                ),
            )
        )

        records.append(
            record
        )

        # ====================================================
        # Layer + SC aggregate
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

            layer_cycles[
                layer_id
            ].append(
                layer_result
                .total_cycles
            )

            layer_switches[
                layer_id
            ].append(
                layer_result
                .switch_count
            )

            layer_waits[
                layer_id
            ].append(
                layer_result
                .wait_cycles
            )

            for stat in (
                layer_result
                .subcube_stats
            ):

                sc = (
                    stat.subcube_id
                )

                sc_tasks[
                    sc
                ] += (
                    stat.task_count
                )

                sc_busy[
                    sc
                ] += (
                    stat.busy_cycles
                )

                sc_switch[
                    sc
                ] += (
                    stat.switch_count
                )

                sc_initial[
                    sc
                ] += (
                    stat
                    .initial_activation_count
                )

                sc_wait[
                    sc
                ] += (
                    stat.wait_cycles
                )

                if (
                    stat.task_count
                    > 0
                    and
                    stat.last_finish_time
                    == layer_result.total_cycles
                ):

                    sc_critical[
                        sc
                    ] += 1

        # ====================================================
        # Progress
        # ====================================================

        evaluated = len(
            records
        )

        if (
            verbose
            and
            (
                evaluated
                == 1
                or
                evaluated
                % progress_every
                == 0
            )
        ):

            print(
                "[DecodeEval] "
                f"tokens="
                f"{evaluated}, "
                f"last_cycles="
                f"{result.total_cycles}, "
                f"mean_cycles="
                f"{mean(record.total_cycles for record in records):.2f}, "
                f"file="
                f"{token.relative_file}, "
                f"segment="
                f"{token.segment_index}"
            )

    # ========================================================
    # Final checks
    # ========================================================

    if not records:

        raise DecodeEvaluatorError(
            "没有读取到纯 Decode Token。"
        )

    task_counts = {
        record.total_tasks

        for record
        in records
    }

    if (
        len(
            task_counts
        )
        != 1
    ):

        raise DecodeEvaluatorError(
            "不同 Decode Token "
            "任务数不一致。"
        )

    # ========================================================
    # Layer summary
    # ========================================================

    layer_summary = tuple(
        DecodeLayerSummary(
            layer_id=(
                layer_id
            ),

            cycles=(
                summarize(
                    layer_cycles[
                        layer_id
                    ]
                )
            ),

            switch_mean=(
                float(
                    mean(
                        layer_switches[
                            layer_id
                        ]
                    )
                )
                if
                layer_switches[
                    layer_id
                ]
                else 0.0
            ),

            wait_mean=(
                float(
                    mean(
                        layer_waits[
                            layer_id
                        ]
                    )
                )
                if
                layer_waits[
                    layer_id
                ]
                else 0.0
            ),
        )

        for layer_id
        in range(
            index.num_layers
        )
    )

    # ========================================================
    # SC summary
    # ========================================================

    total_layer_evaluations = (
        len(
            records
        )
        *
        index.num_layers
    )

    subcube_summary = tuple(
        DecodeSubcubeSummary(
            subcube_id=(
                sc
            ),

            task_count=(
                sc_tasks[
                    sc
                ]
            ),

            busy_cycles=(
                sc_busy[
                    sc
                ]
            ),

            switch_count=(
                sc_switch[
                    sc
                ]
            ),

            initial_activation_count=(
                sc_initial[
                    sc
                ]
            ),

            wait_cycles=(
                sc_wait[
                    sc
                ]
            ),

            critical_layer_count=(
                sc_critical[
                    sc
                ]
            ),

            critical_rate=(
                sc_critical[
                    sc
                ]
                /
                total_layer_evaluations

                if
                total_layer_evaluations
                > 0

                else
                0.0
            ),
        )

        for sc
        in range(
            index.num_subcubes
        )
    )

    # ========================================================
    # Category summary
    # ========================================================

    grouped: dict[
        str,
        list[
            DecodeEvaluationRecord
        ],
    ] = {}

    for record in (
        records
    ):

        grouped.setdefault(
            record.category,
            [],
        ).append(
            record
        )

    category_summary = tuple(
        DecodeCategorySummary(
            category=(
                category
            ),

            token_count=len(
                items
            ),

            cycles=(
                summarize(
                    item.total_cycles

                    for item
                    in items
                )
            ),
        )

        for (
            category,
            items,
        )
        in sorted(
            grouped.items()
        )
    )

    # ========================================================
    # Slowest Tokens
    # ========================================================

    ranked = sorted(
        records,

        key=lambda record: (
            -record.total_cycles,
            record.token_id,
        ),
    )

    slowest = tuple(
        SlowDecodeToken(
            token_id=(
                record.token_id
            ),

            total_cycles=(
                record.total_cycles
            ),

            category=(
                record.category
            ),

            relative_file=(
                record.relative_file
            ),

            segment_index=(
                record.segment_index
            ),
        )

        for record
        in ranked[
            :top_slowest_tokens
        ]
    )

    # ========================================================
    # Final summary
    # ========================================================

    state_mode = (
        "continuous_request_state"
        if continuous_request_state
        else "cold_start_each_token"
    )

    summary = (
        DecodeEvaluationSummary(
            state_mode=(
                state_mode
            ),

            request_count=(
                request_count
            ),

            token_count=len(
                records
            ),

            total_tasks_per_token=next(
                iter(
                    task_counts
                )
            ),

            cycles_per_token=(
                summarize(
                    record.total_cycles

                    for record
                    in records
                )
            ),

            switches_per_token=(
                summarize(
                    record.switches

                    for record
                    in records
                )
            ),

            initial_activations_per_token=(
                summarize(
                    record
                    .initial_activations

                    for record
                    in records
                )
            ),

            wait_cycles_per_token=(
                summarize(
                    record.wait_cycles

                    for record
                    in records
                )
            ),

            layers=(
                layer_summary
            ),

            subcubes=(
                subcube_summary
            ),

            categories=(
                category_summary
            ),

            slowest_tokens=(
                slowest
            ),
        )
    )

    return (
        summary,
        tuple(
            records
        ),
        stats,
    )


# ============================================================
# Print
# ============================================================


def print_scalar(
    name: str,
    summary: ScalarSummary,
    decimals: int = 2,
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


def print_summary(
    *,
    summary: DecodeEvaluationSummary,
    stats: DecodeWorkloadStats,
    top_layers: int = 10,
    top_subcubes: int = 10,
) -> None:

    print(
        "\n"
        "========== Real Decode Evaluation =========="
    )

    print(
        f"State Mode："
        f"{summary.state_mode}"
    )

    print(
        f"Requests Seen："
        f"{summary.request_count}"
    )

    print(
        f"Pure Decode Tokens："
        f"{summary.token_count}"
    )

    print(
        f"Tasks / Token："
        f"{summary.total_tasks_per_token}"
    )

    print(
        "Prefill Segment0 Excluded "
        "From Decode Metric："
        f"{stats.prefill_segment_count}"
    )

    print(
        "Non-singleton Decode Segments："
        f"{stats.non_singleton_decode_segment_count}"
    )

    # ========================================================
    # 核心 Decode 指标
    # ========================================================

    print_scalar(
        "MoE Decode Cycles / Token",
        summary
        .cycles_per_token,
    )

    print_scalar(
        "Weight-Cube Switches / Token",
        summary
        .switches_per_token,
    )

    print_scalar(
        "Initial Activations / Token",
        summary
        .initial_activations_per_token,
    )

    print_scalar(
        "Task Wait Cycles / Token",
        summary
        .wait_cycles_per_token,
    )

    # ========================================================
    # Layers
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
        f"Top-"
        f"{min(top_layers, len(ranked_layers))} "
        "Layers by Mean Decode Cycles："
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
            f"p99="
            f"{item.cycles.p99:.2f}, "
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
            -item
            .critical_layer_count,
            item.subcube_id,
        ),
    )

    print(
        "\n"
        f"Top-"
        f"{min(top_subcubes, len(ranked_sc))} "
        "Critical Sub-Cubes："
    )

    for item in (
        ranked_sc[
            :top_subcubes
        ]
    ):

        print(
            f"  SC-"
            f"{item.subcube_id}: "
            f"critical="
            f"{item.critical_layer_count}, "
            f"rate="
            f"{item.critical_rate:.2%}, "
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
            f"tokens="
            f"{item.token_count}, "
            f"mean="
            f"{item.cycles.mean:.2f}, "
            f"p95="
            f"{item.cycles.p95:.2f}, "
            f"p99="
            f"{item.cycles.p99:.2f}"
        )

    # ========================================================
    # Slow Tokens
    # ========================================================

    print(
        "\nSlowest Decode Tokens："
    )

    for item in (
        summary
        .slowest_tokens
    ):

        print(
            f"  token="
            f"{item.token_id}, "
            f"cycles="
            f"{item.total_cycles}, "
            f"category="
            f"{item.category}, "
            f"file="
            f"{item.relative_file}, "
            f"segment="
            f"{item.segment_index}"
        )


# ============================================================
# Save
# ============================================================


def save_result(
    path: Path | str,
    *,
    summary: DecodeEvaluationSummary,
    records: tuple[
        DecodeEvaluationRecord,
        ...
    ],
    stats: DecodeWorkloadStats,
    mapping: Path | str,
    trace_root: Path | str,
    charge_initial_activation: bool,
) -> Path:

    output = (
        Path(
            path
        )
        .resolve()
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "evaluation_version": 2,

        "metric_scope": (
            "MoE Expert Decode only; "
            "not full-model TPOT"
        ),

        "phase_rule": (
            "segment0=Prefill; "
            "segment1+=Decode singleton"
        ),

        "state_rule": (
            summary.state_mode
        ),

        "scheduler": (
            "exact prefill_scheduler "
            "+ exact token_scheduler"
        ),

        "mapping": str(
            Path(
                mapping
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

        "workload_stats": (
            asdict(
                stats
            )
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

    output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        output
    )


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "评估 Chinese-SimpleQA "
                "segment1+ 的纯 Decode Token，"
                "默认继承 Prefill/前一 Decode Token "
                "的 active WC 状态。"
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

    # 默认仍先跑 1000，
    # 不直接 exact 扫完整 25 万级 Token。
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
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
        "--top-slowest-tokens",
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

    parser.add_argument(
        "--cold-start-each-token",
        action="store_true",

        help=(
            "仅用于回归验证："
            "恢复旧行为，每个 Decode Token "
            "都从 None×16 开始。"
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

    summary, records, stats = (
        evaluate_decode_workload(
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

            max_tokens=(
                args.max_tokens
            ),

            charge_initial_activation=(
                not args
                .no_initial_activation_cost
            ),

            progress_every=(
                args.progress_every
            ),

            top_slowest_tokens=(
                args
                .top_slowest_tokens
            ),

            continuous_request_state=(
                not args
                .cold_start_each_token
            ),

            verbose=(
                not args.quiet
            ),
        )
    )

    print_summary(
        summary=(
            summary
        ),

        stats=(
            stats
        ),

        top_layers=(
            args.top_layers
        ),

        top_subcubes=(
            args.top_subcubes
        ),
    )

    if not (
        args.no_save
    ):

        saved = (
            save_result(
                args.output,

                summary=(
                    summary
                ),

                records=(
                    records
                ),

                stats=(
                    stats
                ),

                mapping=(
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
