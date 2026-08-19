"""
纯 Decode Fast Evaluator。

用途：
    在已经完成 Prefill / Decode 口径分离之后，
    快速评估 Chinese-SimpleQA 中全部 segment1+ Decode Token。

核心原则：
    Fast 只能减少 Python 对象/事件记录开销，
    不能改变 layer_scheduler.py 的调度语义。

因此本文件默认先做：

    FAST
      vs
    EXACT token_scheduler

逐 Token、逐 Layer 完全一致校验。

只要任意一个 Token / Layer 不一致：
    立即报错并停止，
    不允许继续产生“近似结果”。

------------------------------------------------------------

为什么当前 Decode 可以省掉重复 Prefill？

当前 Mapping 中：

    每个 Layer 的 Weight-Cube ID 都是不同的。

一个请求：

    Prefill L57 final WC
        ->
    Decode Token-1 L0 WC

以及：

    Decode Token-i L57 final WC
        ->
    Decode Token-(i+1) L0 WC

都不可能直接命中同一个 WC。

同时当前规则：

    initial activation = 1 cycle
    WC switch          = 1 cycle

所以：

    冷启动第一次访问
和
    从上一阶段 WC 切到本层 WC

对 latency 的代价相同。

我们已经用 exact continuous evaluator 验证：
    cold-start
和
    continuous-request-state

得到完全相同的 Decode cycles/token。

Fast evaluator 仍然在前 exact_check 个 Token 上
逐层验证 FAST == EXACT。

------------------------------------------------------------

当前 Fast Scheduler 仍严格保留：

    gate / up 初始 ready

    gate(e) ----\
                 -> down(e)
    up(e) ------/

    不同 SC 并行
    同一 SC 串行
    compute = 1
    switch = 1

以及原 layer_scheduler.py 的 SC 内选择顺序：

    1. ready_time
    2. active WC（Decode 单 Token 当前层不会重复 WC）
    3. route_rank
    4. gate -> up -> down
    5. cube_id

------------------------------------------------------------

指标范围：

    MoE Expert Decode Cycles / Token

不包含：
    Attention
    KV Cache
    Embedding
    LM Head
    其他非 MoE 模块

因此不能直接称完整 TPOT。
"""

from __future__ import annotations

import argparse
import json
import math

from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Iterable


from config import ExecutionRules

from mapping.trace_profile import DEFAULT_TRACE_ROOT

from scheduling.decode_workload import (
    DecodeWorkloadStats,
    iter_decode_tokens,
)

from scheduling.runtime_index import (
    DEFAULT_MAPPING_PATH,
    RuntimeIndex,
    load_runtime_index,
)

from scheduling.token_scheduler import (
    schedule_token,
)


# ============================================================
# 默认输出
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "decode"
    / "decode_fast_evaluation.json"
)


# ============================================================
# 异常
# ============================================================


class DecodeFastEvaluatorError(ValueError):
    pass


# ============================================================
# 数据结构
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class FastMatrix:
    subcube_id: int
    cube_id: int


@dataclass(
    frozen=True,
    slots=True,
)
class FastExpert:
    gate: FastMatrix
    up: FastMatrix
    down: FastMatrix


@dataclass(
    frozen=True,
    slots=True,
)
class FastLayerResult:
    total_cycles: int

    # 当前层所有任务累计 wait
    wait_cycles: int

    # 最晚完成、决定当前层结束时间的 SC
    critical_subcubes: tuple[int, ...]

    # 每个 SC 当前层任务数
    task_count_by_sc: tuple[int, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class FastTokenResult:
    total_cycles: int

    layer_cycles: tuple[int, ...]

    layer_wait_cycles: tuple[int, ...]

    # 每层 Critical SC，可并列
    critical_subcubes_by_layer: tuple[
        tuple[int, ...],
        ...
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class DecodeFastRecord:
    token_id: int
    category: str
    relative_file: str
    segment_index: int
    total_cycles: int


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
class LayerFastSummary:
    layer_id: int
    cycles: ScalarSummary
    wait_mean: float


@dataclass(
    frozen=True,
    slots=True,
)
class SubcubeFastSummary:
    subcube_id: int
    critical_count: int
    critical_rate: float


@dataclass(
    frozen=True,
    slots=True,
)
class CategoryFastSummary:
    category: str
    token_count: int
    cycles: ScalarSummary


@dataclass(
    frozen=True,
    slots=True,
)
class SlowToken:
    token_id: int
    category: str
    relative_file: str
    segment_index: int
    total_cycles: int


@dataclass(
    frozen=True,
    slots=True,
)
class DecodeFastSummary:
    scheduler_mode: str
    exact_checked_tokens: int

    token_count: int

    cycles_per_token: ScalarSummary

    layers: tuple[LayerFastSummary, ...]

    subcubes: tuple[SubcubeFastSummary, ...]

    categories: tuple[CategoryFastSummary, ...]

    slowest_tokens: tuple[SlowToken, ...]

    cache_hits: int
    cache_misses: int
    cache_currsize: int


# ============================================================
# Percentile
# ============================================================


def percentile(
    values: Iterable[int | float],
    q: float,
) -> float:

    data = sorted(
        float(value)
        for value in values
    )

    if not data:
        return 0.0

    if not 0.0 <= q <= 1.0:
        raise DecodeFastEvaluatorError(
            "percentile q 必须在 [0, 1]。"
        )

    if len(data) == 1:
        return data[0]

    position = (
        (len(data) - 1)
        * q
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return data[lower]

    fraction = position - lower

    return (
        data[lower] * (1.0 - fraction)
        +
        data[upper] * fraction
    )


def summarize(
    values: Iterable[int | float],
) -> ScalarSummary:

    data = [
        float(value)
        for value in values
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
        count=len(data),
        minimum=min(data),
        mean=float(mean(data)),
        p50=percentile(data, 0.50),
        p95=percentile(data, 0.95),
        p99=percentile(data, 0.99),
        maximum=max(data),
    )


# ============================================================
# 构造轻量 Mapping Table
# ============================================================


def build_fast_tables(
    index: RuntimeIndex,
) -> tuple[
    tuple[
        tuple[
            FastExpert,
            ...
        ],
        ...
    ],
    ...
]:
    """
    预先把 RuntimeIndex：

        layer
        -> expert
        -> matrix object

    压缩成：

        layer
        -> expert
        -> (SC, cube_id)

    Fast loop 之后不再创建 RuntimeMatrixLocation 对象。
    """

    layers = []

    for layer_id in range(
        index.num_layers
    ):

        experts = []

        for expert_id in range(
            index.experts_per_layer
        ):

            expert = index.expert(
                layer_id,
                expert_id,
            )

            experts.append(
                FastExpert(
                    gate=FastMatrix(
                        subcube_id=(
                            expert.gate.subcube_id
                        ),
                        cube_id=(
                            expert.gate.cube_id
                        ),
                    ),

                    up=FastMatrix(
                        subcube_id=(
                            expert.up.subcube_id
                        ),
                        cube_id=(
                            expert.up.cube_id
                        ),
                    ),

                    down=FastMatrix(
                        subcube_id=(
                            expert.down.subcube_id
                        ),
                        cube_id=(
                            expert.down.cube_id
                        ),
                    ),
                )
            )

        layers.append(
            tuple(experts)
        )

    return tuple(layers)


# ============================================================
# Fast Layer Scheduler
# ============================================================


# Task tuple：
#
# (
#     ready_time,      0
#     route_rank,      1
#     matrix_priority, 2
#     cube_id,         3
#     active_rank,     4
#     matrix_code,     5
# )
#
# matrix_code:
#     0 gate
#     1 up
#     2 down


def _fast_schedule_layer_uncached(
    *,
    layer_table: tuple[
        FastExpert,
        ...
    ],

    routed_expert_ids: tuple[int, ...],

    shared_expert_id: int | None,

    num_subcubes: int,

    compute_cycles: int,

    switch_cycles: int,
) -> FastLayerResult:
    """
    轻量实现 layer_scheduler.py 的 event loop。

    当前正式 Decode continuous 模式中：
        进入本 Layer 前 SC 已有上一个 Layer/Token 的 WC。

    由于不同 Layer 的 cube_id 不重复，
    且一个单 Token 当前 Layer 内 27 个 WC 也不重复，

    每次 task 都需要：

        switch_cycles + compute_cycles

    active-WC tie-break 不会命中当前队列中的重复 cube。

    这个前提会由 exact-check 逐 Token / 逐 Layer验证。
    """

    if shared_expert_id is None:
        active_ids = routed_expert_ids
    else:
        active_ids = (
            routed_expert_ids
            +
            (
                shared_expert_id,
            )
        )

    active_count = len(active_ids)

    # --------------------------------------------------------
    # Ready queues
    # --------------------------------------------------------

    queues: list[list[tuple]] = [
        []
        for _ in range(
            num_subcubes
        )
    ]

    # rank -> FastExpert
    active_experts = [
        layer_table[
            expert_id
        ]
        for expert_id in active_ids
    ]

    # t=0 所有 gate/up ready
    for rank, expert in enumerate(
        active_experts
    ):

        gate = expert.gate

        queues[
            gate.subcube_id
        ].append(
            (
                0,
                rank,
                0,
                gate.cube_id,
                rank,
                0,
            )
        )

        up = expert.up

        queues[
            up.subcube_id
        ].append(
            (
                0,
                rank,
                1,
                up.cube_id,
                rank,
                1,
            )
        )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    current_time = 0

    # 0 表示 idle。
    # 合法 finish_time 至少 > 0。
    running_finish = [
        0
        for _ in range(
            num_subcubes
        )
    ]

    running_task: list[
        tuple | None
    ] = [
        None
        for _ in range(
            num_subcubes
        )
    ]

    gate_finish = [
        -1
        for _ in range(
            active_count
        )
    ]

    up_finish = [
        -1
        for _ in range(
            active_count
        )
    ]

    down_created = [
        False
        for _ in range(
            active_count
        )
    ]

    completed_count = 0

    expected_count = (
        active_count
        * 3
    )

    wait_total = 0

    task_count_by_sc = [
        0
        for _ in range(
            num_subcubes
        )
    ]

    last_finish_by_sc = [
        0
        for _ in range(
            num_subcubes
        )
    ]

    service_cycles = (
        switch_cycles
        +
        compute_cycles
    )

    # --------------------------------------------------------
    # Event Loop
    # --------------------------------------------------------

    while (
        completed_count
        < expected_count
    ):

        # ====================================================
        # 所有空闲 SC 各 dispatch 一个 ready task
        # ====================================================

        for sc in range(
            num_subcubes
        ):

            if (
                running_finish[
                    sc
                ]
                != 0
            ):
                continue

            queue = queues[
                sc
            ]

            best_index = -1
            best_key = None

            # layer_scheduler:
            # 只在 ready_time <= current_time 中选最小
            for index_in_queue, task in enumerate(
                queue
            ):

                if (
                    task[0]
                    > current_time
                ):
                    continue

                # Decode 单 Token当前层没有重复 cube，
                # already_active 恒不成为有效区分项。
                #
                # 因此完整 key 等价于：
                # ready_time, route_rank, matrix_priority, cube_id
                key = (
                    task[0],
                    task[1],
                    task[2],
                    task[3],
                )

                if (
                    best_key is None
                    or
                    key < best_key
                ):

                    best_key = key
                    best_index = (
                        index_in_queue
                    )

            if best_index < 0:
                continue

            task = queue.pop(
                best_index
            )

            wait_total += (
                current_time
                - task[0]
            )

            running_task[
                sc
            ] = task

            running_finish[
                sc
            ] = (
                current_time
                + service_cycles
            )

            task_count_by_sc[
                sc
            ] += 1

        # ====================================================
        # 找下一完成事件
        # ====================================================

        next_finish = 0

        for finish in (
            running_finish
        ):

            if finish <= 0:
                continue

            if (
                next_finish == 0
                or
                finish < next_finish
            ):

                next_finish = finish

        # 理论上 down 未来 ready 时，至少仍有某个 gate/up 在跑，
        # 因此正常情况下不会出现 running 全空但 future task 存在。
        # 仍保留兜底逻辑。
        if next_finish == 0:

            future_ready = None

            for queue in queues:

                for task in queue:

                    ready = task[0]

                    if (
                        ready
                        > current_time
                        and
                        (
                            future_ready is None
                            or
                            ready < future_ready
                        )
                    ):

                        future_ready = ready

            if future_ready is None:

                raise DecodeFastEvaluatorError(
                    "Fast Layer Scheduler 死锁。"
                )

            current_time = (
                future_ready
            )

            continue

        current_time = (
            next_finish
        )

        # ====================================================
        # 同一时刻可能多个 SC 完成
        # ====================================================

        affected_ranks = set()

        for sc in range(
            num_subcubes
        ):

            if (
                running_finish[
                    sc
                ]
                != current_time
            ):

                continue

            task = (
                running_task[
                    sc
                ]
            )

            if task is None:

                raise DecodeFastEvaluatorError(
                    "Fast running state 错误。"
                )

            running_finish[
                sc
            ] = 0

            running_task[
                sc
            ] = None

            last_finish_by_sc[
                sc
            ] = (
                current_time
            )

            rank = task[4]
            matrix_code = task[5]

            completed_count += 1

            if matrix_code == 0:

                gate_finish[
                    rank
                ] = (
                    current_time
                )

                affected_ranks.add(
                    rank
                )

            elif matrix_code == 1:

                up_finish[
                    rank
                ] = (
                    current_time
                )

                affected_ranks.add(
                    rank
                )

            elif matrix_code != 2:

                raise DecodeFastEvaluatorError(
                    "非法 matrix_code。"
                )

        # ====================================================
        # 动态创建 down
        # ====================================================

        for rank in sorted(
            affected_ranks
        ):

            if (
                down_created[
                    rank
                ]
            ):
                continue

            gate_t = (
                gate_finish[
                    rank
                ]
            )

            up_t = (
                up_finish[
                    rank
                ]
            )

            if (
                gate_t < 0
                or
                up_t < 0
            ):
                continue

            ready_time = max(
                gate_t,
                up_t,
            )

            expert = (
                active_experts[
                    rank
                ]
            )

            down = (
                expert.down
            )

            queues[
                down.subcube_id
            ].append(
                (
                    ready_time,
                    rank,
                    2,
                    down.cube_id,
                    rank,
                    2,
                )
            )

            down_created[
                rank
            ] = True

    # ========================================================
    # Final
    # ========================================================

    total_cycles = max(
        last_finish_by_sc
    )

    critical = tuple(
        sc

        for sc in range(
            num_subcubes
        )

        if (
            task_count_by_sc[
                sc
            ]
            > 0

            and

            last_finish_by_sc[
                sc
            ]
            == total_cycles
        )
    )

    return FastLayerResult(
        total_cycles=(
            total_cycles
        ),

        wait_cycles=(
            wait_total
        ),

        critical_subcubes=(
            critical
        ),

        task_count_by_sc=tuple(
            task_count_by_sc
        ),
    )


# ============================================================
# Fast Scheduler Engine + LRU
# ============================================================


class FastDecodeScheduler:
    """
    包装轻量 Layer Scheduler，并缓存：

        (layer_id, Top8 route)
            ->
        FastLayerResult

    当前 Decode latency 在正式规则下
    不依赖跨 Layer/Token 的具体 active cube ID，
    因为跨 Layer cube 不会相同，
    且 initial activation == switch cost。

    exact-check 会验证这个前提。
    """

    def __init__(
        self,
        *,
        index: RuntimeIndex,
        rules: ExecutionRules,
        cache_size: int,
    ) -> None:

        if cache_size <= 0:
            raise DecodeFastEvaluatorError(
                "cache_size 必须大于 0。"
            )

        if (
            rules.compute_cycles
            != 1
            or
            rules.switch_cycles
            != 1
            or
            rules.cross_subcube_cycles
            != 0
            or
            not rules
            .unlimited_parallel_subcubes
            or
            not rules
            .one_active_weight_cube_per_subcube
        ):

            raise DecodeFastEvaluatorError(
                "Fast Scheduler 当前只支持"
                "本项目已确定的 Baseline："
                "compute=1, switch=1, crossSC=0, "
                "SC间并行、SC内互斥。"
            )

        self.index = index
        self.rules = rules

        self.tables = (
            build_fast_tables(
                index
            )
        )

        shared_expert_id = (
            index.shared_expert_id
        )

        num_subcubes = (
            index.num_subcubes
        )

        compute_cycles = (
            rules.compute_cycles
        )

        switch_cycles = (
            rules.switch_cycles
        )

        tables = self.tables

        @lru_cache(
            maxsize=cache_size
        )
        def cached_layer(
            layer_id: int,
            route: tuple[int, ...],
        ) -> FastLayerResult:

            return (
                _fast_schedule_layer_uncached(
                    layer_table=(
                        tables[
                            layer_id
                        ]
                    ),

                    routed_expert_ids=(
                        route
                    ),

                    shared_expert_id=(
                        shared_expert_id
                    ),

                    num_subcubes=(
                        num_subcubes
                    ),

                    compute_cycles=(
                        compute_cycles
                    ),

                    switch_cycles=(
                        switch_cycles
                    ),
                )
            )

        self._cached_layer = (
            cached_layer
        )

    def schedule_token(
        self,
        routed_experts_by_layer: Iterable[
            Iterable[int]
        ],
    ) -> FastTokenResult:

        routes = tuple(
            tuple(route)
            for route
            in routed_experts_by_layer
        )

        if (
            len(routes)
            != self.index.num_layers
        ):

            raise DecodeFastEvaluatorError(
                "Fast Token 的 Layer 数错误："
                f"actual={len(routes)}, "
                f"expected="
                f"{self.index.num_layers}。"
            )

        layer_results = []

        for layer_id, route in enumerate(
            routes
        ):

            # RuntimeIndex 做 route 合法性检查。
            #
            # DecodeWorkload 本身已经验证过，
            # 这里保留以防单独调用 Fast Scheduler。
            self.index.resolve_active_expert_ids(
                layer_id=(
                    layer_id
                ),

                routed_expert_ids=(
                    route
                ),
            )

            layer_results.append(
                self._cached_layer(
                    layer_id,
                    route,
                )
            )

        layer_cycles = tuple(
            result.total_cycles
            for result
            in layer_results
        )

        return FastTokenResult(
            total_cycles=sum(
                layer_cycles
            ),

            layer_cycles=(
                layer_cycles
            ),

            layer_wait_cycles=tuple(
                result.wait_cycles
                for result
                in layer_results
            ),

            critical_subcubes_by_layer=tuple(
                result.critical_subcubes
                for result
                in layer_results
            ),
        )

    def cache_info(
        self,
    ):
        return (
            self._cached_layer
            .cache_info()
        )


# ============================================================
# EXACT vs FAST 校验
# ============================================================


def validate_fast_against_exact(
    *,
    index: RuntimeIndex,

    rules: ExecutionRules,

    fast_result: FastTokenResult,

    routed_experts_by_layer: tuple[
        tuple[int, ...],
        ...
    ],

    token_id: int,
) -> None:
    """
    exact 使用原始 token_scheduler。

    这里只比较：
        total cycles
        以及 58 层每层 cycles

    这是最终 latency 的完整正确性条件。

    exact 用 cold state。
    当前 continuous 与 cold 的 cycles 已经由前一步实验验证相同；
    Fast 的结构性原因也写在文件顶部。
    """

    exact = (
        schedule_token(
            index=index,

            routed_experts_by_layer=(
                routed_experts_by_layer
            ),

            rules=rules,

            initial_active_cube_by_subcube=None,

            charge_initial_activation=True,
        )
    )

    exact_layer_cycles = tuple(
        execution.cycles

        for execution
        in exact.layers
    )

    if (
        exact.total_cycles
        != fast_result.total_cycles
    ):

        raise DecodeFastEvaluatorError(
            "FAST != EXACT："
            f"Token-{token_id} "
            f"total exact={exact.total_cycles}, "
            f"fast={fast_result.total_cycles}。"
        )

    if (
        exact_layer_cycles
        != fast_result.layer_cycles
    ):

        mismatch = next(
            (
                layer_id

                for layer_id, (
                    exact_cycle,
                    fast_cycle,
                )

                in enumerate(
                    zip(
                        exact_layer_cycles,
                        fast_result
                        .layer_cycles,
                    )
                )

                if (
                    exact_cycle
                    != fast_cycle
                )
            ),
            None,
        )

        raise DecodeFastEvaluatorError(
            "FAST != EXACT："
            f"Token-{token_id}, "
            f"Layer-{mismatch}, "
            f"exact="
            f"{exact_layer_cycles[mismatch]}, "
            f"fast="
            f"{fast_result.layer_cycles[mismatch]}。"
        )


# ============================================================
# Evaluator
# ============================================================


def evaluate_decode_fast(
    *,
    index: RuntimeIndex,

    trace_root: Path | str = (
        DEFAULT_TRACE_ROOT
    ),

    max_files: int | None = None,

    max_tokens: int | None = None,

    exact_check: int = 100,

    cache_size: int = 200000,

    progress_every: int = 10000,

    top_slowest_tokens: int = 10,

    verbose: bool = True,
) -> tuple[
    DecodeFastSummary,
    tuple[
        DecodeFastRecord,
        ...
    ],
    DecodeWorkloadStats,
]:

    if (
        exact_check
        < 0
    ):

        raise DecodeFastEvaluatorError(
            "exact_check 不能小于 0。"
        )

    if (
        max_tokens
        is not None
        and
        max_tokens <= 0
    ):

        raise DecodeFastEvaluatorError(
            "max_tokens 必须大于 0。"
        )

    if (
        max_files
        is not None
        and
        max_files <= 0
    ):

        raise DecodeFastEvaluatorError(
            "max_files 必须大于 0。"
        )

    if progress_every <= 0:

        raise DecodeFastEvaluatorError(
            "progress_every 必须大于 0。"
        )

    if top_slowest_tokens < 0:

        raise DecodeFastEvaluatorError(
            "top_slowest_tokens "
            "不能小于 0。"
        )

    rules = (
        ExecutionRules()
    )

    fast_scheduler = (
        FastDecodeScheduler(
            index=index,
            rules=rules,
            cache_size=(
                cache_size
            ),
        )
    )

    workload_stats = (
        DecodeWorkloadStats()
    )

    records: list[
        DecodeFastRecord
    ] = []

    layer_cycles: list[
        list[int]
    ] = [
        []
        for _ in range(
            index.num_layers
        )
    ]

    layer_waits: list[
        list[int]
    ] = [
        []
        for _ in range(
            index.num_layers
        )
    ]

    critical_count = [
        0
        for _ in range(
            index.num_subcubes
        )
    ]

    category_cycles: dict[
        str,
        list[int],
    ] = defaultdict(
        list
    )

    checked = 0

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
                workload_stats
            ),

            strict_singleton=True,

            verbose=False,
        )
    ):

        routes = (
            token
            .routed_experts_by_layer
        )

        result = (
            fast_scheduler
            .schedule_token(
                routes
            )
        )

        # ====================================================
        # 前 exact_check Token 硬校验
        # ====================================================

        if (
            checked
            < exact_check
        ):

            validate_fast_against_exact(
                index=index,

                rules=rules,

                fast_result=(
                    result
                ),

                routed_experts_by_layer=(
                    routes
                ),

                token_id=(
                    token.token_id
                ),
            )

            checked += 1

            if (
                verbose
                and
                (
                    checked == 1
                    or
                    checked
                    == exact_check
                    or
                    checked % 10 == 0
                )
            ):

                print(
                    "[FastCheck] "
                    f"{checked}/"
                    f"{exact_check} "
                    "FAST == EXACT"
                )

        # ====================================================
        # Record
        # ====================================================

        record = (
            DecodeFastRecord(
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

                total_cycles=(
                    result.total_cycles
                ),
            )
        )

        records.append(
            record
        )

        category_cycles[
            token.category
        ].append(
            result.total_cycles
        )

        # ====================================================
        # Layer
        # ====================================================

        for layer_id in range(
            index.num_layers
        ):

            layer_cycles[
                layer_id
            ].append(
                result
                .layer_cycles[
                    layer_id
                ]
            )

            layer_waits[
                layer_id
            ].append(
                result
                .layer_wait_cycles[
                    layer_id
                ]
            )

            for sc in (
                result
                .critical_subcubes_by_layer[
                    layer_id
                ]
            ):

                critical_count[
                    sc
                ] += 1

        # ====================================================
        # Progress
        # ====================================================

        n = len(
            records
        )

        if (
            verbose
            and
            (
                n == 1
                or
                n % progress_every
                == 0
            )
        ):

            cache_info = (
                fast_scheduler
                .cache_info()
            )

            print(
                "[DecodeFast] "
                f"tokens={n}, "
                f"last={result.total_cycles}, "
                f"mean="
                f"{mean(item.total_cycles for item in records):.2f}, "
                f"cache_hit="
                f"{cache_info.hits}, "
                f"cache_miss="
                f"{cache_info.misses}"
            )

    if not records:

        raise DecodeFastEvaluatorError(
            "没有读取到纯 Decode Token。"
        )

    # ========================================================
    # Summary
    # ========================================================

    layer_summary = tuple(
        LayerFastSummary(
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

            wait_mean=(
                float(
                    mean(
                        layer_waits[
                            layer_id
                        ]
                    )
                )
            ),
        )

        for layer_id
        in range(
            index.num_layers
        )
    )

    total_layer_evaluations = (
        len(records)
        *
        index.num_layers
    )

    sc_summary = tuple(
        SubcubeFastSummary(
            subcube_id=(
                sc
            ),

            critical_count=(
                critical_count[
                    sc
                ]
            ),

            critical_rate=(
                critical_count[
                    sc
                ]
                /
                total_layer_evaluations
            ),
        )

        for sc
        in range(
            index.num_subcubes
        )
    )

    category_summary = tuple(
        CategoryFastSummary(
            category=(
                category
            ),

            token_count=len(
                values
            ),

            cycles=(
                summarize(
                    values
                )
            ),
        )

        for (
            category,
            values,
        )
        in sorted(
            category_cycles.items()
        )
    )

    ranked = sorted(
        records,

        key=lambda record: (
            -record.total_cycles,
            record.token_id,
        ),
    )

    slowest = tuple(
        SlowToken(
            token_id=(
                record.token_id
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

            total_cycles=(
                record.total_cycles
            ),
        )

        for record
        in ranked[
            :top_slowest_tokens
        ]
    )

    cache_info = (
        fast_scheduler
        .cache_info()
    )

    summary = (
        DecodeFastSummary(
            scheduler_mode=(
                "fast_exact-validated"
            ),

            exact_checked_tokens=(
                checked
            ),

            token_count=len(
                records
            ),

            cycles_per_token=(
                summarize(
                    record.total_cycles

                    for record
                    in records
                )
            ),

            layers=(
                layer_summary
            ),

            subcubes=(
                sc_summary
            ),

            categories=(
                category_summary
            ),

            slowest_tokens=(
                slowest
            ),

            cache_hits=(
                cache_info.hits
            ),

            cache_misses=(
                cache_info.misses
            ),

            cache_currsize=(
                cache_info.currsize
            ),
        )
    )

    return (
        summary,
        tuple(
            records
        ),
        workload_stats,
    )


# ============================================================
# Print
# ============================================================


def print_scalar(
    name: str,
    summary: ScalarSummary,
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
        f"{summary.minimum:.2f}"
    )

    print(
        f"  Mean："
        f"{summary.mean:.2f}"
    )

    print(
        f"  P50："
        f"{summary.p50:.2f}"
    )

    print(
        f"  P95："
        f"{summary.p95:.2f}"
    )

    print(
        f"  P99："
        f"{summary.p99:.2f}"
    )

    print(
        f"  Max："
        f"{summary.maximum:.2f}"
    )


def print_summary(
    *,
    summary: DecodeFastSummary,
    workload_stats: DecodeWorkloadStats,
    top_layers: int = 10,
    top_subcubes: int = 10,
) -> None:

    print(
        "\n"
        "========== Fast Real Decode Evaluation =========="
    )

    print(
        f"Scheduler："
        f"{summary.scheduler_mode}"
    )

    print(
        f"FAST == EXACT Checked Tokens："
        f"{summary.exact_checked_tokens}"
    )

    print(
        f"Pure Decode Tokens："
        f"{summary.token_count}"
    )

    print(
        "Non-singleton Decode Segments："
        f"{workload_stats.non_singleton_decode_segment_count}"
    )

    print_scalar(
        "MoE Decode Cycles / Token",
        summary.cycles_per_token,
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
        f"Top-{min(top_layers, len(ranked_layers))} "
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
            f"wait_mean="
            f"{item.wait_mean:.2f}"
        )

    # ========================================================
    # Critical SC
    # ========================================================

    ranked_sc = sorted(
        summary.subcubes,

        key=lambda item: (
            -item.critical_count,
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
            f"critical="
            f"{item.critical_count}, "
            f"rate="
            f"{item.critical_rate:.2%}"
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
    # Slow
    # ========================================================

    print(
        "\nSlowest Decode Tokens："
    )

    for item in (
        summary.slowest_tokens
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

    print(
        "\nFast Layer Cache："
    )

    print(
        f"  Hits："
        f"{summary.cache_hits}"
    )

    print(
        f"  Misses："
        f"{summary.cache_misses}"
    )

    print(
        f"  Current Size："
        f"{summary.cache_currsize}"
    )


# ============================================================
# Save
# ============================================================


def save_result(
    *,
    output_path: Path | str,

    summary: DecodeFastSummary,

    records: tuple[
        DecodeFastRecord,
        ...
    ],

    workload_stats: DecodeWorkloadStats,

    mapping: Path | str,

    trace_root: Path | str,
) -> Path:

    path = (
        Path(
            output_path
        )
        .resolve()
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "evaluation_version": 1,

        "metric_scope": (
            "MoE Expert Decode only; "
            "not full-model TPOT"
        ),

        "phase_rule": (
            "segment0=Prefill; "
            "segment1+=Decode"
        ),

        "scheduler": (
            "compact fast layer event scheduler "
            "validated against exact token_scheduler"
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

        "workload_stats": (
            asdict(
                workload_stats
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

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "快速评估纯 Decode Token；"
                "默认前100 Token逐层验证 "
                "FAST == EXACT。"
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
        "--max-tokens",
        type=int,
        default=None,

        help=(
            "不填则读取全部纯 Decode Token。"
        ),
    )

    parser.add_argument(
        "--exact-check",
        type=int,
        default=100,

        help=(
            "前多少个 Token 同时运行 EXACT，"
            "逐层验证 FAST 完全一致。"
        ),
    )

    parser.add_argument(
        "--cache-size",
        type=int,
        default=200000,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
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

    args = (
        parser.parse_args()
    )

    index = (
        load_runtime_index(
            args.mapping
        )
    )

    summary, records, stats = (
        evaluate_decode_fast(
            index=index,

            trace_root=(
                args.root
            ),

            max_files=(
                args.max_files
            ),

            max_tokens=(
                args.max_tokens
            ),

            exact_check=(
                args.exact_check
            ),

            cache_size=(
                args.cache_size
            ),

            progress_every=(
                args.progress_every
            ),

            top_slowest_tokens=(
                args.top_slowest_tokens
            ),

            verbose=(
                not args.quiet
            ),
        )
    )

    print_summary(
        summary=summary,

        workload_stats=stats,

        top_layers=(
            args.top_layers
        ),

        top_subcubes=(
            args.top_subcubes
        ),
    )

    if not args.no_save:

        saved = (
            save_result(
                output_path=(
                    args.output
                ),

                summary=summary,

                records=records,

                workload_stats=stats,

                mapping=(
                    args.mapping
                ),

                trace_root=(
                    args.root
                ),
            )
        )

        print(
            "\nSaved："
            f"{saved}"
        )


if __name__ == "__main__":
    main()
