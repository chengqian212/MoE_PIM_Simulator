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
import multiprocessing as mp
import os

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Iterable, Iterator


from config import ExecutionRules

from mapping.trace_profile import DEFAULT_TRACE_ROOT, discover_trace_files

from scheduling.decode_workload import (
    DecodeWorkloadStats,
    _load_json as _load_decode_json,
    build_decode_token,
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

from scheduling.trace_workload import TraceToken, collect_segment_routes


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
    layer_table: tuple[FastExpert, ...],
    routed_expert_ids: tuple[int, ...],
    shared_expert_id: int | None,
    num_subcubes: int,
    compute_cycles: int,
    switch_cycles: int,
) -> FastLayerResult:
    """
    Decode 的等价直接计算版。

    与原 event-loop 语义完全相同，但利用当前正式 Baseline 的两个性质：

    1. 单 Token 当前 Layer 内每个 WC 只访问一次；
    2. initial activation 和 WC switch 的代价相同。

    因此每个 gate/up/down task 的 service time 固定为：

        switch_cycles + compute_cycles

    gate/up 全部 ready=0，可以直接得到它们在各 SC 中的完成位置；
    down 的 ready_time 再由对应 gate/up 的完成时间推出。

    这样不再创建 27 个事件任务、ready queue、running heap，
    但 total_cycles / wait_cycles / critical SC / task count 与原实现一致。
    """

    if shared_expert_id is None:
        active_ids = routed_expert_ids
    else:
        active_ids = routed_expert_ids + (shared_expert_id,)

    service_cycles = switch_cycles + compute_cycles
    if service_cycles <= 0:
        raise DecodeFastEvaluatorError(
            "Fast Decode task service cycles 必须大于 0。"
        )

    # 每个 SC 已经排入多少个 gate/up。
    pre_count = [0] * num_subcubes

    # 每个 SC 的 down：(ready_time, route_rank)。
    down_by_sc: list[list[tuple[int, int]]] = [
        [] for _ in range(num_subcubes)
    ]

    wait_total = 0

    # gate/up 的 exact 顺序由 route_rank（再由 matrix priority）确定。
    # Step4 保证同一 Expert 的 gate/up 不在同一 SC，因此这里按 route
    # 顺序递增计数即可得到对应完成时间。
    for route_rank, expert_id in enumerate(active_ids):
        expert = layer_table[expert_id]

        gate_sc = expert.gate.subcube_id
        gate_dispatch = pre_count[gate_sc] * service_cycles
        wait_total += gate_dispatch
        pre_count[gate_sc] += 1
        gate_finish = pre_count[gate_sc] * service_cycles

        up_sc = expert.up.subcube_id
        up_dispatch = pre_count[up_sc] * service_cycles
        wait_total += up_dispatch
        pre_count[up_sc] += 1
        up_finish = pre_count[up_sc] * service_cycles

        down_ready = max(gate_finish, up_finish)
        down_by_sc[expert.down.subcube_id].append(
            (down_ready, route_rank)
        )

    finish_by_sc = [
        count * service_cycles for count in pre_count
    ]
    task_count_by_sc = pre_count.copy()

    # down 的 exact priority：ready_time -> route_rank。
    for sc, downs in enumerate(down_by_sc):
        if not downs:
            continue

        downs.sort()
        current_time = finish_by_sc[sc]

        for ready_time, _route_rank in downs:
            if current_time < ready_time:
                current_time = ready_time

            wait_total += current_time - ready_time
            current_time += service_cycles

        finish_by_sc[sc] = current_time
        task_count_by_sc[sc] += len(downs)

    total_cycles = max(finish_by_sc, default=0)

    critical = tuple(
        sc
        for sc in range(num_subcubes)
        if task_count_by_sc[sc] > 0
        and finish_by_sc[sc] == total_cycles
    )

    return FastLayerResult(
        total_cycles=total_cycles,
        wait_cycles=wait_total,
        critical_subcubes=critical,
        task_count_by_sc=tuple(task_count_by_sc),
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
        routed_experts_by_layer: Iterable[Iterable[int]],
        *,
        validate_routes: bool = True,
    ) -> FastTokenResult:
        """
        计算一个 Decode Token 的 58 层结果。

        evaluate_decode_fast() 读取的 Trace 已经在 decode_workload.py 中
        做过完整 Top-8 校验，因此正式全量路径传 validate_routes=False，
        避免 255k Token × 58 Layer 的重复合法性检查。

        单独调用本方法时默认仍然 validate_routes=True，保留安全检查。
        """

        if isinstance(routed_experts_by_layer, tuple) and all(
            isinstance(route, tuple)
            for route in routed_experts_by_layer
        ):
            routes = routed_experts_by_layer
        else:
            routes = tuple(
                tuple(route)
                for route in routed_experts_by_layer
            )

        if len(routes) != self.index.num_layers:
            raise DecodeFastEvaluatorError(
                "Fast Token 的 Layer 数错误："
                f"actual={len(routes)}, "
                f"expected={self.index.num_layers}。"
            )

        layer_cycles: list[int] = []
        layer_wait_cycles: list[int] = []
        critical_by_layer: list[tuple[int, ...]] = []
        total_cycles = 0

        for layer_id, route in enumerate(routes):
            if validate_routes:
                self.index.resolve_active_expert_ids(
                    layer_id=layer_id,
                    routed_expert_ids=route,
                )

            result = self._cached_layer(layer_id, route)
            cycles = result.total_cycles
            total_cycles += cycles
            layer_cycles.append(cycles)
            layer_wait_cycles.append(result.wait_cycles)
            critical_by_layer.append(result.critical_subcubes)

        return FastTokenResult(
            total_cycles=total_cycles,
            layer_cycles=tuple(layer_cycles),
            layer_wait_cycles=tuple(layer_wait_cycles),
            critical_subcubes_by_layer=tuple(critical_by_layer),
        )

    def cache_info(
        self,
    ):
        return (
            self._cached_layer
            .cache_info()
        )



# ============================================================
# 多进程 Fast Decode
# ============================================================


@dataclass(frozen=True, slots=True)
class _DecodeParallelChunkResult:
    """一个 worker 处理一批 Decode Token 后返回的紧凑统计。"""

    records: tuple[DecodeFastRecord, ...]
    layer_cycles: tuple[tuple[int, ...], ...]
    layer_wait_sums: tuple[int, ...]
    critical_count: tuple[int, ...]
    category_cycles: dict[str, tuple[int, ...]]
    total_cycles: int

    # 每个 worker 都有自己的 LRU cache。
    # hits / misses 返回本 chunk 的增量；currsize 返回该 worker 当前值。
    cache_hits_delta: int
    cache_misses_delta: int
    worker_pid: int
    cache_currsize: int


_DECODE_WORKER_SCHEDULER: FastDecodeScheduler | None = None


def _resolve_worker_count(workers: int) -> int:
    """
    workers=0：自动选择。

    默认最多 8 个进程，避免 RuntimeIndex + LRU cache 把内存放大过多。
    workers=1：完全回退到原单进程 Fast 路径。
    """

    if workers < 0:
        raise DecodeFastEvaluatorError("workers 不能小于 0。")

    if workers == 0:
        cpu = os.cpu_count() or 1
        if cpu <= 2:
            return 1
        return max(1, min(8, cpu - 1))

    return workers


def _iter_chunks(
    items: Iterable[TraceToken],
    chunk_size: int,
) -> Iterator[tuple[TraceToken, ...]]:
    if chunk_size <= 0:
        raise DecodeFastEvaluatorError("parallel_chunk_size 必须大于 0。")

    chunk: list[TraceToken] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield tuple(chunk)
            chunk.clear()

    if chunk:
        yield tuple(chunk)


def _init_decode_worker(
    index: RuntimeIndex,
    rules: ExecutionRules,
    cache_size: int,
) -> None:
    """Windows spawn worker 初始化：每个进程只构造一次 Scheduler。"""

    global _DECODE_WORKER_SCHEDULER
    _DECODE_WORKER_SCHEDULER = FastDecodeScheduler(
        index=index,
        rules=rules,
        cache_size=cache_size,
    )


def _decode_worker_chunk(
    tokens: tuple[TraceToken, ...],
) -> _DecodeParallelChunkResult:
    scheduler = _DECODE_WORKER_SCHEDULER
    if scheduler is None:
        raise DecodeFastEvaluatorError("Decode worker 尚未初始化。")

    num_layers = scheduler.index.num_layers
    num_subcubes = scheduler.index.num_subcubes

    records: list[DecodeFastRecord] = []
    layer_cycles: list[list[int]] = [[] for _ in range(num_layers)]
    layer_wait_sums = [0] * num_layers
    critical_count = [0] * num_subcubes
    category_cycles: dict[str, list[int]] = defaultdict(list)
    total_cycles = 0

    before = scheduler.cache_info()

    for token in tokens:
        result = scheduler.schedule_token(
            token.routed_experts_by_layer,
            validate_routes=False,
        )

        records.append(
            DecodeFastRecord(
                token_id=token.token_id,
                category=token.category,
                relative_file=token.relative_file,
                segment_index=token.segment_index,
                total_cycles=result.total_cycles,
            )
        )
        total_cycles += result.total_cycles
        category_cycles[token.category].append(result.total_cycles)

        for layer_id in range(num_layers):
            layer_cycles[layer_id].append(result.layer_cycles[layer_id])
            layer_wait_sums[layer_id] += result.layer_wait_cycles[layer_id]
            for sc in result.critical_subcubes_by_layer[layer_id]:
                critical_count[sc] += 1

    after = scheduler.cache_info()

    return _DecodeParallelChunkResult(
        records=tuple(records),
        layer_cycles=tuple(tuple(values) for values in layer_cycles),
        layer_wait_sums=tuple(layer_wait_sums),
        critical_count=tuple(critical_count),
        category_cycles={
            category: tuple(values)
            for category, values in category_cycles.items()
        },
        total_cycles=total_cycles,
        cache_hits_delta=after.hits - before.hits,
        cache_misses_delta=after.misses - before.misses,
        worker_pid=os.getpid(),
        cache_currsize=after.currsize,
    )


def _bounded_parallel_decode_chunks(
    *,
    executor: ProcessPoolExecutor,
    chunks: Iterable[tuple[TraceToken, ...]],
    max_pending: int,
) -> Iterator[_DecodeParallelChunkResult]:
    """限制同时挂起的 chunk，避免一次把 25 万 Token 全部 pickle 进队列。"""

    if max_pending <= 0:
        raise DecodeFastEvaluatorError("max_pending 必须大于 0。")

    iterator = iter(chunks)
    pending = set()

    def submit_one() -> bool:
        try:
            chunk = next(iterator)
        except StopIteration:
            return False
        pending.add(executor.submit(_decode_worker_chunk, chunk))
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
class _DecodeFileTokenResult:
    file_order: int
    segment_index: int
    total_cycles: int


@dataclass(frozen=True, slots=True)
class _DecodeFileShardResult:
    tokens: tuple[_DecodeFileTokenResult, ...]
    layer_cycle_hist: tuple[dict[int, int], ...]
    layer_wait_sums: tuple[int, ...]
    critical_count: tuple[int, ...]
    total_cycles: int
    processed_file_count: int
    trace_segment_count: int
    prefill_segment_count: int
    valid_decode_segment_count: int
    skipped_decode_segment_count: int
    non_singleton_decode_segment_count: int
    cache_hits_delta: int
    cache_misses_delta: int
    worker_pid: int
    cache_currsize: int


_DECODE_WORKER_TRACE_ROOT: Path | None = None


def _init_decode_file_worker(
    index: RuntimeIndex,
    rules: ExecutionRules,
    cache_size: int,
    trace_root: str,
) -> None:
    _init_decode_worker(index, rules, cache_size)
    global _DECODE_WORKER_TRACE_ROOT
    _DECODE_WORKER_TRACE_ROOT = Path(trace_root).resolve()


def _decode_worker_file_shard(
    shard: tuple[tuple[int, str], ...],
) -> _DecodeFileShardResult:
    scheduler = _DECODE_WORKER_SCHEDULER
    root = _DECODE_WORKER_TRACE_ROOT
    if scheduler is None or root is None:
        raise DecodeFastEvaluatorError("Decode file worker 尚未初始化。")

    nl = scheduler.index.num_layers
    ns = scheduler.index.num_subcubes
    token_rows: list[_DecodeFileTokenResult] = []
    layer_hist = [Counter() for _ in range(nl)]
    layer_wait_sums = [0] * nl
    critical_count = [0] * ns
    total_cycles = 0
    trace_segment_count = 0
    prefill_segment_count = 0
    valid_decode_segment_count = 0
    skipped_decode_segment_count = 0
    non_singleton_decode_segment_count = 0

    before = scheduler.cache_info()

    for file_order, relative_file in shard:
        path = root / relative_file
        data = _load_decode_json(path)
        trace_segment_count += len(data)
        if data:
            prefill_segment_count += 1

        relative = Path(relative_file)
        category = relative.parts[0] if len(relative.parts) >= 2 else "__root__"

        for segment_index, segment in enumerate(data):
            if segment_index == 0:
                continue
            if not isinstance(segment, dict):
                raise DecodeFastEvaluatorError(
                    f"{path}: segment-{segment_index} 必须是 dict。"
                )

            raw_routes = collect_segment_routes(segment=segment)
            if raw_routes is None:
                skipped_decode_segment_count += 1
                continue

            token_count = len(raw_routes[0][1])
            if token_count != 1:
                non_singleton_decode_segment_count += 1
                raise DecodeFastEvaluatorError(
                    f"{path}: segment-{segment_index} 不是 singleton Decode："
                    f"token_count={token_count}。"
                )

            token = build_decode_token(
                path=path,
                relative_file=relative_file,
                category=category,
                token_id=0,
                segment_index=segment_index,
                segment=segment,
                strict_singleton=True,
                raw_routes_by_layer=raw_routes,
            )
            if token is None:
                skipped_decode_segment_count += 1
                continue

            result = scheduler.schedule_token(
                token.routed_experts_by_layer,
                validate_routes=False,
            )
            valid_decode_segment_count += 1
            total_cycles += result.total_cycles
            token_rows.append(
                _DecodeFileTokenResult(
                    file_order=file_order,
                    segment_index=segment_index,
                    total_cycles=result.total_cycles,
                )
            )

            for layer_id in range(nl):
                layer_hist[layer_id][result.layer_cycles[layer_id]] += 1
                layer_wait_sums[layer_id] += result.layer_wait_cycles[layer_id]
                for sc in result.critical_subcubes_by_layer[layer_id]:
                    critical_count[sc] += 1

    after = scheduler.cache_info()
    return _DecodeFileShardResult(
        tokens=tuple(token_rows),
        layer_cycle_hist=tuple(dict(x) for x in layer_hist),
        layer_wait_sums=tuple(layer_wait_sums),
        critical_count=tuple(critical_count),
        total_cycles=total_cycles,
        processed_file_count=len(shard),
        trace_segment_count=trace_segment_count,
        prefill_segment_count=prefill_segment_count,
        valid_decode_segment_count=valid_decode_segment_count,
        skipped_decode_segment_count=skipped_decode_segment_count,
        non_singleton_decode_segment_count=non_singleton_decode_segment_count,
        cache_hits_delta=after.hits - before.hits,
        cache_misses_delta=after.misses - before.misses,
        worker_pid=os.getpid(),
        cache_currsize=after.currsize,
    )


def _make_file_shards(
    files: list[Path],
    root: Path,
    workers: int,
) -> list[tuple[tuple[int, str], ...]]:
    if not files:
        return []
    shard_count = min(len(files), max(workers, workers * 4))
    shards: list[list[tuple[int, str]]] = [[] for _ in range(shard_count)]
    for file_order, path in enumerate(files):
        shards[file_order % shard_count].append((file_order, str(path.relative_to(root))))
    return [tuple(shard) for shard in shards if shard]


def _summarize_histogram(hist: dict[int, int]) -> ScalarSummary:
    if not hist:
        return ScalarSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    keys = sorted(hist)
    count = sum(hist.values())
    total = sum(value * hist[value] for value in keys)

    def value_at(index: int) -> float:
        seen = 0
        for value in keys:
            seen += hist[value]
            if index < seen:
                return float(value)
        return float(keys[-1])

    def p(q: float) -> float:
        if count == 1:
            return float(keys[0])
        position = (count - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return value_at(lower)
        fraction = position - lower
        return value_at(lower) * (1.0 - fraction) + value_at(upper) * fraction

    return ScalarSummary(
        count=count,
        minimum=float(keys[0]),
        mean=float(total / count),
        p50=p(0.50),
        p95=p(0.95),
        p99=p(0.99),
        maximum=float(keys[-1]),
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



def _evaluate_decode_fast_file_parallel(
    *,
    index: RuntimeIndex,
    trace_root: Path | str,
    max_files: int | None,
    exact_check: int,
    cache_size: int,
    progress_every: int,
    top_slowest_tokens: int,
    verbose: bool,
    workers: int,
) -> tuple[DecodeFastSummary, tuple[DecodeFastRecord, ...], DecodeWorkloadStats]:
    """
    全量 Decode 专用的粗粒度并行：主进程只分发“文件名”，Trace JSON 的
    读取、Route 构造和 Fast 调度都在 worker 内完成，避免 255k 个
    TraceToken 在 Windows 进程间反复 pickle。
    """

    root = Path(trace_root).resolve()
    files = list(discover_trace_files(root))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise DecodeFastEvaluatorError("没有找到 Decode Trace 文件。")

    rules = ExecutionRules()
    per_worker_cache = max(1000, cache_size // workers)

    # --------------------------------------------------------
    # exact-check 仍使用原完整调度器；只校验，不参与最终统计。
    # --------------------------------------------------------
    checked = 0
    exact_scheduler = FastDecodeScheduler(
        index=index,
        rules=rules,
        cache_size=per_worker_cache,
    )
    if exact_check > 0:
        exact_stats = DecodeWorkloadStats()
        for token in iter_decode_tokens(
            trace_root=root,
            max_files=max_files,
            max_tokens=exact_check,
            stats=exact_stats,
            strict_singleton=True,
            verbose=False,
        ):
            fast_result = exact_scheduler.schedule_token(
                token.routed_experts_by_layer,
                validate_routes=False,
            )
            validate_fast_against_exact(
                index=index,
                rules=rules,
                fast_result=fast_result,
                routed_experts_by_layer=token.routed_experts_by_layer,
                token_id=token.token_id,
            )
            checked += 1
            if verbose and (checked == 1 or checked == exact_check or checked % 10 == 0):
                print(f"[FastCheck] {checked}/{exact_check} FAST == EXACT")

    file_info: list[tuple[str, str]] = []
    for path in files:
        relative = path.relative_to(root)
        relative_text = str(relative)
        category = relative.parts[0] if len(relative.parts) >= 2 else "__root__"
        file_info.append((category, relative_text))

    token_rows: list[_DecodeFileTokenResult] = []
    layer_hist = [Counter() for _ in range(index.num_layers)]
    layer_wait_sums = [0] * index.num_layers
    critical_count = [0] * index.num_subcubes
    running_total_cycles = 0
    cache_hits = 0
    cache_misses = 0
    cache_curr_by_pid: dict[int, int] = {}

    stats = DecodeWorkloadStats()
    stats.discovered_file_count = len(files)
    stats.processed_file_count = 0
    stats.trace_segment_count = 0
    stats.prefill_segment_count = 0
    stats.valid_decode_segment_count = 0
    stats.skipped_decode_segment_count = 0
    stats.non_singleton_decode_segment_count = 0
    stats.decode_token_count = 0

    shards = _make_file_shards(files, root, workers)
    if verbose:
        print(
            "[DecodeFastFileParallel] "
            f"workers={workers}, shards={len(shards)}, files={len(files)}, "
            f"cache/worker={per_worker_cache}"
        )

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_decode_file_worker,
        initargs=(index, rules, per_worker_cache, str(root)),
    ) as executor:
        futures = [executor.submit(_decode_worker_file_shard, shard) for shard in shards]
        pending = set(futures)
        last_report = 0

        while pending:
            done, not_done = wait(pending, return_when=FIRST_COMPLETED)
            pending = set(not_done)
            for future in done:
                part = future.result()
                token_rows.extend(part.tokens)
                running_total_cycles += part.total_cycles
                cache_hits += part.cache_hits_delta
                cache_misses += part.cache_misses_delta
                cache_curr_by_pid[part.worker_pid] = part.cache_currsize

                stats.processed_file_count += part.processed_file_count
                stats.trace_segment_count += part.trace_segment_count
                stats.prefill_segment_count += part.prefill_segment_count
                stats.valid_decode_segment_count += part.valid_decode_segment_count
                stats.skipped_decode_segment_count += part.skipped_decode_segment_count
                stats.non_singleton_decode_segment_count += part.non_singleton_decode_segment_count

                for layer_id in range(index.num_layers):
                    layer_hist[layer_id].update(part.layer_cycle_hist[layer_id])
                    layer_wait_sums[layer_id] += part.layer_wait_sums[layer_id]
                for sc in range(index.num_subcubes):
                    critical_count[sc] += part.critical_count[sc]

                if verbose and len(token_rows) - last_report >= progress_every:
                    last_report = len(token_rows)
                    print(
                        "[DecodeFastFileParallel] "
                        f"tokens={len(token_rows)}, "
                        f"mean={running_total_cycles / len(token_rows):.2f}, "
                        f"files={stats.processed_file_count}/{len(files)}"
                    )

    if not token_rows:
        raise DecodeFastEvaluatorError("没有读取到纯 Decode Token。")

    token_rows.sort(key=lambda x: (x.file_order, x.segment_index))
    records: list[DecodeFastRecord] = []
    category_cycles: dict[str, list[int]] = defaultdict(list)

    for token_id, row in enumerate(token_rows):
        category, relative_file = file_info[row.file_order]
        record = DecodeFastRecord(
            token_id=token_id,
            category=category,
            relative_file=relative_file,
            segment_index=row.segment_index,
            total_cycles=row.total_cycles,
        )
        records.append(record)
        category_cycles[category].append(row.total_cycles)

    stats.decode_token_count = len(records)

    layer_summary = tuple(
        LayerFastSummary(
            layer_id=layer_id,
            cycles=_summarize_histogram(dict(layer_hist[layer_id])),
            wait_mean=layer_wait_sums[layer_id] / len(records),
        )
        for layer_id in range(index.num_layers)
    )

    total_layer_evaluations = len(records) * index.num_layers
    sc_summary = tuple(
        SubcubeFastSummary(
            subcube_id=sc,
            critical_count=critical_count[sc],
            critical_rate=critical_count[sc] / total_layer_evaluations,
        )
        for sc in range(index.num_subcubes)
    )
    category_summary = tuple(
        CategoryFastSummary(
            category=category,
            token_count=len(values),
            cycles=summarize(values),
        )
        for category, values in sorted(category_cycles.items())
    )
    ranked = sorted(records, key=lambda r: (-r.total_cycles, r.token_id))
    slowest = tuple(
        SlowToken(
            token_id=r.token_id,
            category=r.category,
            relative_file=r.relative_file,
            segment_index=r.segment_index,
            total_cycles=r.total_cycles,
        )
        for r in ranked[:top_slowest_tokens]
    )

    exact_cache = exact_scheduler.cache_info()
    summary = DecodeFastSummary(
        scheduler_mode="fast_exact-validated",
        exact_checked_tokens=checked,
        token_count=len(records),
        cycles_per_token=summarize(r.total_cycles for r in records),
        layers=layer_summary,
        subcubes=sc_summary,
        categories=category_summary,
        slowest_tokens=slowest,
        cache_hits=cache_hits + exact_cache.hits,
        cache_misses=cache_misses + exact_cache.misses,
        cache_currsize=sum(cache_curr_by_pid.values()) + exact_cache.currsize,
    )
    return summary, tuple(records), stats


def evaluate_decode_fast(
    *,
    index: RuntimeIndex,
    trace_root: Path | str = DEFAULT_TRACE_ROOT,
    max_files: int | None = None,
    max_tokens: int | None = None,
    exact_check: int = 100,
    cache_size: int = 200000,
    progress_every: int = 10000,
    top_slowest_tokens: int = 10,
    verbose: bool = True,
    workers: int = 1,
    parallel_chunk_size: int = 256,
) -> tuple[
    DecodeFastSummary,
    tuple[DecodeFastRecord, ...],
    DecodeWorkloadStats,
]:
    """
    Fast Decode evaluator。

    workers=1：原单进程 Fast 路径。
    workers>1：exact-check 前 N Token 仍在主进程硬校验，其余 Token
               分 chunk 交给多进程 Fast Scheduler。
    workers=0：自动选择（最多 8 个）。

    并行只改变 Python 模拟器执行方式，不改变任何周期语义。
    """

    if exact_check < 0:
        raise DecodeFastEvaluatorError("exact_check 不能小于 0。")
    if max_tokens is not None and max_tokens <= 0:
        raise DecodeFastEvaluatorError("max_tokens 必须大于 0。")
    if max_files is not None and max_files <= 0:
        raise DecodeFastEvaluatorError("max_files 必须大于 0。")
    if progress_every <= 0:
        raise DecodeFastEvaluatorError("progress_every 必须大于 0。")
    if top_slowest_tokens < 0:
        raise DecodeFastEvaluatorError("top_slowest_tokens 不能小于 0。")
    if parallel_chunk_size <= 0:
        raise DecodeFastEvaluatorError("parallel_chunk_size 必须大于 0。")

    resolved_workers = _resolve_worker_count(workers)

    # 全量 Decode 使用文件级粗粒度并行。Smoke / max_tokens 小实验保留
    # 单进程 Fast，避免 Windows spawn + IPC 启动开销反而拖慢。
    if resolved_workers > 1 and max_tokens is None:
        return _evaluate_decode_fast_file_parallel(
            index=index,
            trace_root=trace_root,
            max_files=max_files,
            exact_check=exact_check,
            cache_size=cache_size,
            progress_every=progress_every,
            top_slowest_tokens=top_slowest_tokens,
            verbose=verbose,
            workers=resolved_workers,
        )

    if resolved_workers > 1 and max_tokens is not None and verbose:
        print(
            "[DecodeFast] max_tokens 已设置，使用单进程 Fast；"
            "全量评估时才启用文件级多进程。"
        )
    resolved_workers = 1
    rules = ExecutionRules()

    # cache_size 仍被视为一次 evaluator 的总预算。
    # 多进程时按 worker 分摊，避免 8×200000 的内存放大。
    per_worker_cache_size = (
        cache_size
        if resolved_workers <= 1
        else max(1000, cache_size // resolved_workers)
    )

    main_scheduler = FastDecodeScheduler(
        index=index,
        rules=rules,
        cache_size=per_worker_cache_size,
    )

    workload_stats = DecodeWorkloadStats()
    token_iterator = iter_decode_tokens(
        trace_root=trace_root,
        max_files=max_files,
        max_tokens=max_tokens,
        stats=workload_stats,
        strict_singleton=True,
        verbose=False,
    )

    records: list[DecodeFastRecord] = []
    layer_cycles: list[list[int]] = [[] for _ in range(index.num_layers)]
    # wait 只需要 mean，不再保存 14M 级别的逐 Token wait Python int。
    layer_wait_sums = [0] * index.num_layers
    critical_count = [0] * index.num_subcubes
    category_cycles: dict[str, list[int]] = defaultdict(list)

    checked = 0
    running_total_cycles = 0
    worker_cache_hits = 0
    worker_cache_misses = 0
    worker_cache_currsize: dict[int, int] = {}

    def accumulate_one(
        token: TraceToken,
        result: FastTokenResult,
    ) -> None:
        nonlocal running_total_cycles

        records.append(
            DecodeFastRecord(
                token_id=token.token_id,
                category=token.category,
                relative_file=token.relative_file,
                segment_index=token.segment_index,
                total_cycles=result.total_cycles,
            )
        )
        running_total_cycles += result.total_cycles
        category_cycles[token.category].append(result.total_cycles)

        for layer_id in range(index.num_layers):
            layer_cycles[layer_id].append(result.layer_cycles[layer_id])
            layer_wait_sums[layer_id] += result.layer_wait_cycles[layer_id]
            for sc in result.critical_subcubes_by_layer[layer_id]:
                critical_count[sc] += 1

    def accumulate_chunk(chunk: _DecodeParallelChunkResult) -> None:
        nonlocal running_total_cycles, worker_cache_hits, worker_cache_misses

        records.extend(chunk.records)
        running_total_cycles += chunk.total_cycles
        worker_cache_hits += chunk.cache_hits_delta
        worker_cache_misses += chunk.cache_misses_delta
        worker_cache_currsize[chunk.worker_pid] = chunk.cache_currsize

        for layer_id in range(index.num_layers):
            layer_cycles[layer_id].extend(chunk.layer_cycles[layer_id])
            layer_wait_sums[layer_id] += chunk.layer_wait_sums[layer_id]

        for sc in range(index.num_subcubes):
            critical_count[sc] += chunk.critical_count[sc]

        for category, values in chunk.category_cycles.items():
            category_cycles[category].extend(values)

    # ========================================================
    # 前 exact_check 个 Token：主进程 FAST == EXACT
    # ========================================================

    while checked < exact_check:
        try:
            token = next(token_iterator)
        except StopIteration:
            break

        routes = token.routed_experts_by_layer
        result = main_scheduler.schedule_token(routes, validate_routes=False)
        validate_fast_against_exact(
            index=index,
            rules=rules,
            fast_result=result,
            routed_experts_by_layer=routes,
            token_id=token.token_id,
        )
        checked += 1
        accumulate_one(token, result)

        if verbose and (
            checked == 1
            or checked == exact_check
            or checked % 10 == 0
        ):
            print(f"[FastCheck] {checked}/{exact_check} FAST == EXACT")

    # ========================================================
    # 剩余 Token：单进程或多进程
    # ========================================================

    if resolved_workers <= 1:
        for token in token_iterator:
            result = main_scheduler.schedule_token(
                token.routed_experts_by_layer,
                validate_routes=False,
            )
            accumulate_one(token, result)

            n = len(records)
            if verbose and (n == 1 or n % progress_every == 0):
                info = main_scheduler.cache_info()
                print(
                    "[DecodeFast] "
                    f"tokens={n}, last={result.total_cycles}, "
                    f"mean={running_total_cycles / n:.2f}, "
                    f"workers=1, cache_hit={info.hits}, cache_miss={info.misses}"
                )
    else:
        if verbose:
            print(
                "[DecodeFastParallel] "
                f"workers={resolved_workers}, chunk={parallel_chunk_size}, "
                f"cache/worker={per_worker_cache_size}"
            )

        ctx = mp.get_context("spawn")
        chunks = _iter_chunks(token_iterator, parallel_chunk_size)
        with ProcessPoolExecutor(
            max_workers=resolved_workers,
            mp_context=ctx,
            initializer=_init_decode_worker,
            initargs=(index, rules, per_worker_cache_size),
        ) as executor:
            for chunk_result in _bounded_parallel_decode_chunks(
                executor=executor,
                chunks=chunks,
                max_pending=max(2, resolved_workers * 2),
            ):
                accumulate_chunk(chunk_result)

                n = len(records)
                if verbose and (n == 1 or n % progress_every < parallel_chunk_size):
                    print(
                        "[DecodeFastParallel] "
                        f"tokens={n}, mean={running_total_cycles / n:.2f}, "
                        f"cache_hit={worker_cache_hits}, "
                        f"cache_miss={worker_cache_misses}"
                    )

    if not records:
        raise DecodeFastEvaluatorError("没有读取到纯 Decode Token。")

    # 并行完成顺序不固定；恢复原 token_id 顺序，保证输出 JSON 稳定。
    records.sort(key=lambda record: record.token_id)

    layer_summary = tuple(
        LayerFastSummary(
            layer_id=layer_id,
            cycles=summarize(layer_cycles[layer_id]),
            wait_mean=(layer_wait_sums[layer_id] / len(records)),
        )
        for layer_id in range(index.num_layers)
    )

    total_layer_evaluations = len(records) * index.num_layers
    sc_summary = tuple(
        SubcubeFastSummary(
            subcube_id=sc,
            critical_count=critical_count[sc],
            critical_rate=critical_count[sc] / total_layer_evaluations,
        )
        for sc in range(index.num_subcubes)
    )

    category_summary = tuple(
        CategoryFastSummary(
            category=category,
            token_count=len(values),
            cycles=summarize(values),
        )
        for category, values in sorted(category_cycles.items())
    )

    ranked = sorted(
        records,
        key=lambda record: (-record.total_cycles, record.token_id),
    )
    slowest = tuple(
        SlowToken(
            token_id=record.token_id,
            category=record.category,
            relative_file=record.relative_file,
            segment_index=record.segment_index,
            total_cycles=record.total_cycles,
        )
        for record in ranked[:top_slowest_tokens]
    )

    main_cache = main_scheduler.cache_info()
    aggregate_cache_hits = main_cache.hits + worker_cache_hits
    aggregate_cache_misses = main_cache.misses + worker_cache_misses
    aggregate_cache_currsize = main_cache.currsize + sum(worker_cache_currsize.values())

    summary = DecodeFastSummary(
        scheduler_mode="fast_exact-validated",
        exact_checked_tokens=checked,
        token_count=len(records),
        cycles_per_token=summarize(record.total_cycles for record in records),
        layers=layer_summary,
        subcubes=sc_summary,
        categories=category_summary,
        slowest_tokens=slowest,
        cache_hits=aggregate_cache_hits,
        cache_misses=aggregate_cache_misses,
        cache_currsize=aggregate_cache_currsize,
    )

    return summary, tuple(records), workload_stats


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
        "--workers",
        type=int,
        default=0,
        help="并行 worker 数；0=自动，1=关闭多进程。",
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

            workers=(
                args.workers
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
