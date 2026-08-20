"""
第六步：Prefill / Decode Segment Workload 读取。

目标：

    保留 Chinese-SimpleQA 原始 Segment 边界，
    不再像 trace_workload.py 那样
    把 Segment 内的 Token 全部拆散。

------------------------------------------------------------

原始 Trace：

一个 JSON：

[
    segment-0,
    segment-1,
    ...
]

每个完整 segment：

{
    "3": [
        token-route-0,
        token-route-1,
        ...
    ],

    ...

    "60": [
        token-route-0,
        token-route-1,
        ...
    ]
}

同一个 Segment：

    Layer 3~60

必须拥有完全相同的 Token 数量。

------------------------------------------------------------

本文件输出：

TraceSegmentBatch

    Segment
        ├── Token-0
        │     ├── Project Layer-0 -> Top-8
        │     ├── ...
        │     └── Project Layer-57 -> Top-8
        │
        ├── Token-1
        └── ...

数据 shape：

    Token
        ->
    Layer
        ->
    Top-8

即：

    token_count × 58 × 8

------------------------------------------------------------

Prefill / Decode：

目前原始 Trace 中没有显式：

    "stage": "prefill"

这样的字段。

所以这里只做结构性候选判断：

1. segment_index == 0
   且 token_count > 1

       -> prefill_candidate

2. segment_index > 0
   且 token_count == 1

       -> decode_candidate

3. segment_index == 0
   且 token_count == 1

       -> ambiguous_singleton

   因为 Prompt 长度也可能只有 1，
   不能武断认为一定是 Decode。

4. segment_index > 0
   且 token_count > 1

       -> unexpected_multi_token

运行完整数据扫描后，再根据统计结果确认
Chinese-SimpleQA 的真实结构是否稳定满足：

    第一个 Segment：多 Token
    后续 Segment：单 Token

------------------------------------------------------------

这个文件暂时：

    不做调度
    不算周期
    不读取 Mapping

只解决：

    Trace
        ->
    Segment Batch
        ->
    Prefill / Decode 数据结构确认
"""

from __future__ import annotations

import argparse
import json

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


from mapping.trace_profile import (
    DEFAULT_TRACE_ROOT,
    EXPERTS_PER_TOKEN,
    NUM_MOE_LAYERS,
    TRACE_FIRST_MOE_LAYER,
    discover_trace_files,
)

from scheduling.trace_workload import (
    TraceWorkloadError,
    collect_segment_routes,
    validate_runtime_route,
)


# ============================================================
# Stage Labels
# ============================================================


STAGE_PREFILL_CANDIDATE = (
    "prefill_candidate"
)

STAGE_DECODE_CANDIDATE = (
    "decode_candidate"
)

STAGE_AMBIGUOUS_SINGLETON = (
    "ambiguous_singleton"
)

STAGE_UNEXPECTED_MULTI_TOKEN = (
    "unexpected_multi_token"
)


# ============================================================
# 异常
# ============================================================


class PrefillWorkloadError(
    ValueError
):
    """
    Prefill / Decode Workload 读取失败。
    """


# ============================================================
# 一个 Segment Batch
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class TraceSegmentBatch:
    """
    一个完整有效 Segment。

    routed_experts_by_token：

        shape =

            token_count
                ×
            58
                ×
            8

    例如：

        routed_experts_by_token[3][10]

    表示：

        当前 Segment
        Token-3
        Project MoE Layer-10
        的 Routed Top-8。
    """

    batch_id: int

    category: str

    relative_file: str

    segment_index: int

    stage: str

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
    # Validation
    # ========================================================

    def __post_init__(
        self,
    ) -> None:

        if self.batch_id < 0:

            raise PrefillWorkloadError(
                "batch_id 不能为负数。"
            )

        if self.segment_index < 0:

            raise PrefillWorkloadError(
                "segment_index 不能为负数。"
            )

        if not (
            self.routed_experts_by_token
        ):

            raise PrefillWorkloadError(
                "Segment Batch "
                "至少需要一个 Token。"
            )

        valid_stages = {
            STAGE_PREFILL_CANDIDATE,
            STAGE_DECODE_CANDIDATE,
            STAGE_AMBIGUOUS_SINGLETON,
            STAGE_UNEXPECTED_MULTI_TOKEN,
        }

        if (
            self.stage
            not in valid_stages
        ):

            raise PrefillWorkloadError(
                "未知 stage："
                f"{self.stage!r}。"
            )

        # ====================================================
        # Token
        # ====================================================

        for (
            token_index,
            routes_by_layer,
        ) in enumerate(
            self.routed_experts_by_token
        ):

            if (
                len(
                    routes_by_layer
                )
                != NUM_MOE_LAYERS
            ):

                raise PrefillWorkloadError(
                    f"Batch-{self.batch_id} "
                    f"Token-{token_index} "
                    "必须具有 "
                    f"{NUM_MOE_LAYERS} "
                    "层路由。"
                )

            # ================================================
            # Layer
            # ================================================

            for (
                layer_id,
                route,
            ) in enumerate(
                routes_by_layer
            ):

                if (
                    len(route)
                    != EXPERTS_PER_TOKEN
                ):

                    raise PrefillWorkloadError(
                        f"Batch-{self.batch_id} "
                        f"Token-{token_index} "
                        f"Layer-{layer_id} "
                        "Top-K 数量错误。"
                    )

    # ========================================================
    # Properties
    # ========================================================

    @property
    def token_count(
        self,
    ) -> int:
        """
        当前 Segment 中的 Token 数量。
        """

        return len(
            self.routed_experts_by_token
        )

    @property
    def is_multi_token(
        self,
    ) -> bool:

        return (
            self.token_count
            > 1
        )

    @property
    def is_single_token(
        self,
    ) -> bool:

        return (
            self.token_count
            == 1
        )

    @property
    def is_prefill_candidate(
        self,
    ) -> bool:

        return (
            self.stage
            == STAGE_PREFILL_CANDIDATE
        )

    @property
    def is_decode_candidate(
        self,
    ) -> bool:

        return (
            self.stage
            == STAGE_DECODE_CANDIDATE
        )

    # ========================================================
    # Query
    # ========================================================

    def route(
        self,
        token_index: int,
        layer_id: int,
    ) -> tuple[
        int,
        ...
    ]:
        """
        查询：

            某个 Token
            在某个 Project MoE Layer
            的 Top-8。
        """

        if not (
            0
            <= token_index
            < self.token_count
        ):

            raise PrefillWorkloadError(
                f"token_index="
                f"{token_index} "
                "超出范围。"
            )

        if not (
            0
            <= layer_id
            < NUM_MOE_LAYERS
        ):

            raise PrefillWorkloadError(
                f"layer_id="
                f"{layer_id} "
                "超出范围。"
            )

        return (
            self
            .routed_experts_by_token[
                token_index
            ][
                layer_id
            ]
        )

    def layer_routes(
        self,
        layer_id: int,
    ) -> tuple[
        tuple[
            int,
            ...
        ],
        ...
    ]:
        """
        返回：

            当前 Segment
            所有 Token
            在某一层的 Top-8。

        shape：

            token_count × 8

        这个接口后面会直接交给：

            prefill_layer_scheduler.py
        """

        if not (
            0
            <= layer_id
            < NUM_MOE_LAYERS
        ):

            raise PrefillWorkloadError(
                f"layer_id="
                f"{layer_id} "
                "超出范围。"
            )

        return tuple(
            token_routes[
                layer_id
            ]

            for token_routes
            in self
            .routed_experts_by_token
        )


# ============================================================
# Stats
# ============================================================


@dataclass(
    slots=True,
)
class PrefillWorkloadStats:
    """
    Segment 级完整扫描统计。
    """

    discovered_file_count: int = 0

    processed_file_count: int = 0

    trace_segment_count: int = 0

    valid_segment_count: int = 0

    skipped_segment_count: int = 0

    yielded_batch_count: int = 0

    # ========================================================
    # Stage
    # ========================================================

    prefill_candidate_count: int = 0

    decode_candidate_count: int = 0

    ambiguous_singleton_count: int = 0

    unexpected_multi_token_count: int = 0

    # ========================================================
    # Token
    # ========================================================

    total_token_count: int = 0

    prefill_candidate_token_count: int = 0

    decode_candidate_token_count: int = 0

    # ========================================================
    # File Pattern
    #
    # canonical：
    #
    #     第一个有效 Segment > 1 Token
    #     后续所有有效 Segment = 1 Token
    # ========================================================

    canonical_file_count: int = 0

    singleton_only_file_count: int = 0

    noncanonical_file_count: int = 0

    # ========================================================
    # Distribution
    # ========================================================

    segment_token_histogram: (
        Counter[int]
        | None
    ) = None

    prefill_token_histogram: (
        Counter[int]
        | None
    ) = None

    category_file_counts: (
        dict[
            str,
            int,
        ]
        | None
    ) = None

    def __post_init__(
        self,
    ) -> None:

        if (
            self.segment_token_histogram
            is None
        ):

            self.segment_token_histogram = (
                Counter()
            )

        if (
            self.prefill_token_histogram
            is None
        ):

            self.prefill_token_histogram = (
                Counter()
            )

        if (
            self.category_file_counts
            is None
        ):

            self.category_file_counts = {}


# ============================================================
# Stage Classification
# ============================================================


def classify_segment_stage(
    *,
    segment_index: int,
    token_count: int,
) -> str:
    """
    这里只根据结构做候选分类。

    注意：

    不是声称 Trace 官方已经明确
    标注了 Prefill / Decode。
    """

    if token_count <= 0:

        raise PrefillWorkloadError(
            "token_count 必须大于 0。"
        )

    # ========================================================
    # 第一个 Segment，多 Token
    # ========================================================

    if (
        segment_index == 0
        and
        token_count > 1
    ):

        return (
            STAGE_PREFILL_CANDIDATE
        )

    # ========================================================
    # 后续 Segment，单 Token
    # ========================================================

    if (
        segment_index > 0
        and
        token_count == 1
    ):

        return (
            STAGE_DECODE_CANDIDATE
        )

    # ========================================================
    # 第一个 Segment 只有一个 Token
    #
    # 不能直接判断：
    #
    #     Decode
    #
    # 因为 Prompt 也可能长度 = 1。
    # ========================================================

    if (
        segment_index == 0
        and
        token_count == 1
    ):

        return (
            STAGE_AMBIGUOUS_SINGLETON
        )

    # ========================================================
    # 后续 Segment 又出现多 Token
    # ========================================================

    return (
        STAGE_UNEXPECTED_MULTI_TOKEN
    )


# ============================================================
# JSON
# ============================================================


def _load_json(
    path: Path,
) -> list:
    """
    读取单个 Trace JSON。
    """

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise PrefillWorkloadError(
            f"无法读取 JSON：{path}"
        ) from exc

    if not isinstance(
        data,
        list,
    ):

        raise PrefillWorkloadError(
            f"{path}: "
            "JSON 最外层必须是 list。"
        )

    if not data:

        raise PrefillWorkloadError(
            f"{path}: "
            "JSON 最外层不能为空。"
        )

    return data


# ============================================================
# Segment Validation + Transpose
# ============================================================


def build_segment_batch(
    *,
    path: Path,
    relative_file: str,
    category: str,
    batch_id: int,
    segment_index: int,
    segment: object,
) -> (
    TraceSegmentBatch
    | None
):
    """
    将一个原始 Segment：

        Layer
            ->
        Token
            ->
        Top8

    转成：

        Token
            ->
        Layer
            ->
        Top8

    不完整 Segment 返回 None。
    """

    if not isinstance(
        segment,
        dict,
    ):

        raise PrefillWorkloadError(
            f"{path}: "
            f"segment-{segment_index} "
            "必须是 dict。"
        )

    raw_routes_by_layer = (
        collect_segment_routes(
            segment=segment
        )
    )

    # ========================================================
    # 不完整 Segment
    # ========================================================

    if (
        raw_routes_by_layer
        is None
    ):

        return None

    # ========================================================
    # 必须先验证完整 Segment。
    #
    # 防止已经构造部分 Token，
    # 后面才发现非法 Route。
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

        validated_routes: list[
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
                        route=raw_route,

                        path=path,

                        segment_index=(
                            segment_index
                        ),

                        trace_layer_id=(
                            trace_layer_id
                        ),

                        token_index=(
                            token_index
                        ),
                    )
                )

            except TraceWorkloadError as exc:

                raise PrefillWorkloadError(
                    str(exc)
                ) from exc

            validated_routes.append(
                route
            )

        validated_by_layer.append(
            validated_routes
        )

    # ========================================================
    # 58 层都有相同 Token 数。
    #
    # collect_segment_routes()
    # 已经检查过。
    # ========================================================

    token_count = len(
        validated_by_layer[
            0
        ]
    )

    if token_count <= 0:

        return None

    # ========================================================
    # Transpose
    #
    # 原始：
    #
    #     Layer
    #       ->
    #     Token
    #       ->
    #     Top8
    #
    # 转成：
    #
    #     Token
    #       ->
    #     Layer
    #       ->
    #     Top8
    # ========================================================

    routed_experts_by_token = tuple(

        tuple(
            validated_by_layer[
                project_layer_id
            ][
                token_index
            ]

            for project_layer_id
            in range(
                NUM_MOE_LAYERS
            )
        )

        for token_index
        in range(
            token_count
        )
    )

    stage = (
        classify_segment_stage(
            segment_index=(
                segment_index
            ),

            token_count=(
                token_count
            ),
        )
    )

    return (
        TraceSegmentBatch(
            batch_id=(
                batch_id
            ),

            category=(
                category
            ),

            relative_file=(
                relative_file
            ),

            segment_index=(
                segment_index
            ),

            stage=(
                stage
            ),

            routed_experts_by_token=(
                routed_experts_by_token
            ),
        )
    )


# ============================================================
# File Pattern
# ============================================================


def _update_file_pattern_stats(
    *,
    valid_token_counts: list[int],

    stats: PrefillWorkloadStats,
) -> None:
    """
    判断一个 JSON 文件是否符合：

        [N, 1, 1, 1, ...]
        N > 1

    这种结构。
    """

    if not valid_token_counts:

        stats.noncanonical_file_count += 1

        return

    # ========================================================
    # 全部都是单 Token
    #
    # 可能是：
    #
    #     Prompt 长度本来就是 1
    #
    # 也可能有其他数据语义。
    #
    # 所以单独统计，不直接归为错误。
    # ========================================================

    if all(
        count == 1

        for count
        in valid_token_counts
    ):

        stats.singleton_only_file_count += 1

        return

    # ========================================================
    # Canonical
    # ========================================================

    if (
        valid_token_counts[
            0
        ] > 1

        and

        all(
            count == 1

            for count
            in valid_token_counts[
                1:
            ]
        )
    ):

        stats.canonical_file_count += 1

        return

    # ========================================================
    # 其他结构
    # ========================================================

    stats.noncanonical_file_count += 1


# ============================================================
# Stats Update
# ============================================================


def _update_batch_stats(
    *,
    batch: TraceSegmentBatch,

    stats: PrefillWorkloadStats,
) -> None:

    stats.valid_segment_count += 1

    stats.yielded_batch_count += 1

    stats.total_token_count += (
        batch.token_count
    )

    assert (
        stats.segment_token_histogram
        is not None
    )

    stats.segment_token_histogram[
        batch.token_count
    ] += 1

    # ========================================================
    # Stage
    # ========================================================

    if (
        batch.stage
        == STAGE_PREFILL_CANDIDATE
    ):

        stats.prefill_candidate_count += 1

        stats.prefill_candidate_token_count += (
            batch.token_count
        )

        assert (
            stats.prefill_token_histogram
            is not None
        )

        stats.prefill_token_histogram[
            batch.token_count
        ] += 1

    elif (
        batch.stage
        == STAGE_DECODE_CANDIDATE
    ):

        stats.decode_candidate_count += 1

        stats.decode_candidate_token_count += (
            batch.token_count
        )

    elif (
        batch.stage
        == STAGE_AMBIGUOUS_SINGLETON
    ):

        stats.ambiguous_singleton_count += 1

    elif (
        batch.stage
        ==
        STAGE_UNEXPECTED_MULTI_TOKEN
    ):

        stats.unexpected_multi_token_count += 1

    else:

        raise PrefillWorkloadError(
            "内部 stage 状态错误。"
        )


# ============================================================
# 主 Generator
# ============================================================


def iter_trace_segment_batches(
    trace_root: Path | str = (
        DEFAULT_TRACE_ROOT
    ),
    *,
    max_files: int | None = None,
    max_batches: int | None = None,
    stats: PrefillWorkloadStats | None = None,
    verbose: bool = True,
) -> Iterator[
    TraceSegmentBatch
]:
    """
    流式遍历所有有效 Segment。

    与 iter_trace_tokens() 的区别：

        iter_trace_tokens：

            Segment
                ->
            Token
                ->
            一个个 yield

        本函数：

            Segment
                ->
            整个 Segment 一次 yield

    后续 Prefill Scheduler
    应直接消费这个 Generator。
    """

    root = (
        Path(trace_root)
        .resolve()
    )

    files = list(
        discover_trace_files(
            root
        )
    )

    # ========================================================
    # max_files
    # ========================================================

    if (
        max_files
        is not None
    ):

        if max_files <= 0:

            raise PrefillWorkloadError(
                "max_files 必须大于 0。"
            )

        files = files[
            :max_files
        ]

    # ========================================================
    # max_batches
    # ========================================================

    if (
        max_batches
        is not None
    ):

        if max_batches <= 0:

            raise PrefillWorkloadError(
                "max_batches 必须大于 0。"
            )

    # ========================================================
    # Stats
    # ========================================================

    if stats is None:

        stats = (
            PrefillWorkloadStats()
        )

    # 每次遍历重新初始化。
    stats.discovered_file_count = (
        len(files)
    )

    stats.processed_file_count = 0

    stats.trace_segment_count = 0

    stats.valid_segment_count = 0

    stats.skipped_segment_count = 0

    stats.yielded_batch_count = 0

    stats.prefill_candidate_count = 0

    stats.decode_candidate_count = 0

    stats.ambiguous_singleton_count = 0

    stats.unexpected_multi_token_count = 0

    stats.total_token_count = 0

    stats.prefill_candidate_token_count = 0

    stats.decode_candidate_token_count = 0

    stats.canonical_file_count = 0

    stats.singleton_only_file_count = 0

    stats.noncanonical_file_count = 0

    stats.segment_token_histogram = (
        Counter()
    )

    stats.prefill_token_histogram = (
        Counter()
    )

    stats.category_file_counts = {}

    batch_id = 0

    total_files = len(
        files
    )

    # ========================================================
    # Files
    # ========================================================

    for (
        file_index,
        path,
    ) in enumerate(
        files,
        start=1,
    ):

        relative = (
            path.relative_to(
                root
            )
        )

        relative_text = str(
            relative
        )

        # ====================================================
        # Category
        # ====================================================

        if (
            len(
                relative.parts
            )
            >= 2
        ):

            category = (
                relative.parts[
                    0
                ]
            )

        else:

            category = (
                "__root__"
            )

        assert (
            stats.category_file_counts
            is not None
        )

        stats.category_file_counts[
            category
        ] = (
            stats
            .category_file_counts
            .get(
                category,
                0,
            )
            + 1
        )

        # ====================================================
        # JSON
        # ====================================================

        data = (
            _load_json(
                path
            )
        )

        stats.processed_file_count += 1

        if (
            verbose

            and
            (
                file_index == 1
                or
                file_index == total_files
                or
                file_index % 100 == 0
            )
        ):

            print(
                f"[PrefillWorkload] "
                f"{file_index}/"
                f"{total_files} "
                f"{relative}"
            )

        # ====================================================
        # 用于检查当前文件模式：
        #
        #     [17, 1, 1, ...]
        # ====================================================

        valid_token_counts: list[int] = []

        file_completed = True

        # ====================================================
        # Segments
        # ====================================================

        for (
            segment_index,
            segment,
        ) in enumerate(
            data
        ):

            stats.trace_segment_count += 1

            batch = (
                build_segment_batch(
                    path=path,

                    relative_file=(
                        relative_text
                    ),

                    category=(
                        category
                    ),

                    batch_id=(
                        batch_id
                    ),

                    segment_index=(
                        segment_index
                    ),

                    segment=(
                        segment
                    ),
                )
            )

            # ================================================
            # Invalid Segment
            # ================================================

            if batch is None:

                stats.skipped_segment_count += 1

                continue

            valid_token_counts.append(
                batch.token_count
            )

            _update_batch_stats(
                batch=batch,
                stats=stats,
            )

            yield batch

            batch_id += 1

            # ================================================
            # max_batches
            # ================================================

            if (
                max_batches
                is not None

                and
                batch_id
                >= max_batches
            ):

                # 当前文件没有完整扫描，
                # 因此不能把它用于 file pattern 统计。
                file_completed = False

                return

        # ====================================================
        # 当前 JSON 完整扫描结束
        # ====================================================

        if file_completed:

            _update_file_pattern_stats(
                valid_token_counts=(
                    valid_token_counts
                ),

                stats=stats,
            )


# ============================================================
# Prefill-only Fast Iterator
# ============================================================


def iter_prefill_batches(
    trace_root: Path | str = DEFAULT_TRACE_ROOT,
    *,
    max_files: int | None = None,
    max_batches: int | None = None,
    stats: PrefillWorkloadStats | None = None,
    verbose: bool = True,
) -> Iterator[TraceSegmentBatch]:
    """
    只读取每个 JSON 的 segment0。

    当前 Chinese-SimpleQA 已经确认 2020/2020 文件结构均为：

        [N>1, 1, 1, ...]

    因此正式 Prefill 评估没有必要为了取 segment0，继续把后续几十万
    Decode singleton segment 逐层解析、逐 Top-8 校验后再丢弃。

    本函数与 iter_trace_segment_batches() 使用同一个 build_segment_batch()
    做 segment0 的完整合法性检查，所以不会改变 Prefill route 内容或调度语义；
    它只减少与 Prefill 无关的 Trace 解析工作。

    注意：这里的 stats 是“Prefill-only 扫描统计”，不再代表整个文件中
    所有 Decode segment 的完整统计。需要数据集结构审计时仍使用
    scan_prefill_workload()/iter_trace_segment_batches()。
    """

    root = Path(trace_root).resolve()
    files = list(discover_trace_files(root))

    if max_files is not None:
        if max_files <= 0:
            raise PrefillWorkloadError("max_files 必须大于 0。")
        files = files[:max_files]

    if max_batches is not None and max_batches <= 0:
        raise PrefillWorkloadError("max_batches 必须大于 0。")

    if stats is None:
        stats = PrefillWorkloadStats()

    # Prefill-only reset
    stats.discovered_file_count = len(files)
    stats.processed_file_count = 0
    stats.trace_segment_count = 0
    stats.valid_segment_count = 0
    stats.skipped_segment_count = 0
    stats.yielded_batch_count = 0
    stats.prefill_candidate_count = 0
    stats.decode_candidate_count = 0
    stats.ambiguous_singleton_count = 0
    stats.unexpected_multi_token_count = 0
    stats.total_token_count = 0
    stats.prefill_candidate_token_count = 0
    stats.decode_candidate_token_count = 0
    stats.canonical_file_count = 0
    stats.singleton_only_file_count = 0
    stats.noncanonical_file_count = 0
    stats.segment_token_histogram = Counter()
    stats.prefill_token_histogram = Counter()
    stats.category_file_counts = {}

    batch_id = 0
    total_files = len(files)

    for file_index, path in enumerate(files, start=1):
        relative = path.relative_to(root)
        relative_text = str(relative)
        category = relative.parts[0] if len(relative.parts) >= 2 else "__root__"

        assert stats.category_file_counts is not None
        stats.category_file_counts[category] = (
            stats.category_file_counts.get(category, 0) + 1
        )

        data = _load_json(path)
        stats.processed_file_count += 1

        # 不解析后续 segment，但记录原文件里一共有多少 segment，方便进度观察。
        stats.trace_segment_count += len(data)

        if verbose and (
            file_index == 1
            or file_index == total_files
            or file_index % 100 == 0
        ):
            print(
                f"[PrefillOnlyWorkload] {file_index}/{total_files} {relative}"
            )

        segment0 = data[0]
        batch = build_segment_batch(
            path=path,
            relative_file=relative_text,
            category=category,
            batch_id=batch_id,
            segment_index=0,
            segment=segment0,
        )

        if batch is None:
            stats.skipped_segment_count += 1
            continue

        # 正式 Prefill 只接受 segment0 多 Token candidate。
        if batch.stage != STAGE_PREFILL_CANDIDATE:
            stats.skipped_segment_count += 1
            continue

        _update_batch_stats(batch=batch, stats=stats)
        stats.yielded_batch_count += 1
        yield batch

        batch_id += 1
        if max_batches is not None and batch_id >= max_batches:
            return


# ============================================================
# Scan
# ============================================================


def scan_prefill_workload(
    trace_root: Path | str = (
        DEFAULT_TRACE_ROOT
    ),
    *,
    max_files: int | None = None,
    max_batches: int | None = None,
    verbose: bool = True,
) -> PrefillWorkloadStats:
    """
    完整扫描，不保存 Batch。
    """

    stats = (
        PrefillWorkloadStats()
    )

    for _batch in (
        iter_trace_segment_batches(
            trace_root=(
                trace_root
            ),

            max_files=(
                max_files
            ),

            max_batches=(
                max_batches
            ),

            stats=(
                stats
            ),

            verbose=(
                verbose
            ),
        )
    ):

        pass

    return stats


# ============================================================
# Histogram Utilities
# ============================================================


def _hist_total(
    histogram: Counter[int],
) -> int:

    return sum(
        histogram.values()
    )


def _hist_mean(
    histogram: Counter[int],
) -> float:

    total = (
        _hist_total(
            histogram
        )
    )

    if total <= 0:

        return 0.0

    weighted_sum = sum(
        value * count

        for (
            value,
            count,
        ) in histogram.items()
    )

    return (
        weighted_sum
        / total
    )


def _hist_percentile(
    histogram: Counter[int],
    percentile: float,
) -> int | None:
    """
    离散 nearest-rank percentile。
    """

    if not (
        0.0
        <= percentile
        <= 1.0
    ):

        raise ValueError(
            "percentile 必须位于 [0,1]。"
        )

    total = (
        _hist_total(
            histogram
        )
    )

    if total <= 0:

        return None

    # nearest rank
    rank = max(
        1,
        int(
            percentile
            * total
            + 0.999999999
        ),
    )

    cumulative = 0

    for value in sorted(
        histogram
    ):

        cumulative += (
            histogram[
                value
            ]
        )

        if cumulative >= rank:

            return value

    return max(
        histogram
    )


# ============================================================
# Bucket
# ============================================================


def _prefill_length_buckets(
    histogram: Counter[int],
) -> dict[
    str,
    int,
]:
    """
    Prefill 长度分桶。
    """

    buckets = {
        "2-16": 0,
        "17-32": 0,
        "33-64": 0,
        "65-128": 0,
        "129-256": 0,
        "257+": 0,
    }

    for (
        token_count,
        segment_count,
    ) in histogram.items():

        if (
            2
            <= token_count
            <= 16
        ):

            buckets[
                "2-16"
            ] += segment_count

        elif (
            17
            <= token_count
            <= 32
        ):

            buckets[
                "17-32"
            ] += segment_count

        elif (
            33
            <= token_count
            <= 64
        ):

            buckets[
                "33-64"
            ] += segment_count

        elif (
            65
            <= token_count
            <= 128
        ):

            buckets[
                "65-128"
            ] += segment_count

        elif (
            129
            <= token_count
            <= 256
        ):

            buckets[
                "129-256"
            ] += segment_count

        elif token_count >= 257:

            buckets[
                "257+"
            ] += segment_count

    return buckets


# ============================================================
# Print One Batch
# ============================================================


def print_segment_batch(
    batch: TraceSegmentBatch,
    *,
    show_tokens: int = 2,
    show_layers: tuple[
        int,
        ...
    ] = (
        0,
        1,
        28,
        57,
    ),
) -> None:
    """
    打印一个 Segment 的部分信息。
    """

    print(
        "\n"
        "========== Trace Segment Batch =========="
    )

    print(
        f"Batch ID："
        f"{batch.batch_id}"
    )

    print(
        f"Category："
        f"{batch.category}"
    )

    print(
        f"File："
        f"{batch.relative_file}"
    )

    print(
        f"Segment："
        f"{batch.segment_index}"
    )

    print(
        f"Stage："
        f"{batch.stage}"
    )

    print(
        f"Token Count："
        f"{batch.token_count}"
    )

    token_limit = min(
        show_tokens,
        batch.token_count,
    )

    for token_index in range(
        token_limit
    ):

        print(
            f"\nToken-{token_index}:"
        )

        for layer_id in (
            show_layers
        ):

            route = (
                batch.route(
                    token_index,
                    layer_id,
                )
            )

            trace_layer_id = (
                layer_id
                + TRACE_FIRST_MOE_LAYER
            )

            print(
                f"  Project L{layer_id} "
                f"(Trace L"
                f"{trace_layer_id}): "
                f"{route}"
            )


# ============================================================
# Print Stats
# ============================================================


def print_prefill_workload_stats(
    stats: PrefillWorkloadStats,
) -> None:

    print(
        "\n"
        "========== Prefill / Decode "
        "Workload Stats =========="
    )

    print(
        f"Discovered JSON Files："
        f"{stats.discovered_file_count}"
    )

    print(
        f"Processed JSON Files："
        f"{stats.processed_file_count}"
    )

    print(
        f"Trace Segments："
        f"{stats.trace_segment_count}"
    )

    print(
        f"Valid Segments："
        f"{stats.valid_segment_count}"
    )

    print(
        f"Skipped Segments："
        f"{stats.skipped_segment_count}"
    )

    print(
        f"Total Tokens："
        f"{stats.total_token_count}"
    )

    # ========================================================
    # Stage
    # ========================================================

    print(
        "\nStage Candidates："
    )

    print(
        "  Prefill Candidate："
        f"{stats.prefill_candidate_count}"
    )

    print(
        "  Decode Candidate："
        f"{stats.decode_candidate_count}"
    )

    print(
        "  Ambiguous First "
        "Singleton："
        f"{stats.ambiguous_singleton_count}"
    )

    print(
        "  Unexpected Multi-Token："
        f"{stats.unexpected_multi_token_count}"
    )

    print(
        "\nCandidate Tokens："
    )

    print(
        "  Prefill Tokens："
        f"{stats.prefill_candidate_token_count}"
    )

    print(
        "  Decode Tokens："
        f"{stats.decode_candidate_token_count}"
    )

    # ========================================================
    # File Structure
    # ========================================================

    print(
        "\nFile Structure："
    )

    print(
        "  Canonical "
        "[N>1, 1, 1, ...]："
        f"{stats.canonical_file_count}"
    )

    print(
        "  Singleton-only："
        f"{stats.singleton_only_file_count}"
    )

    print(
        "  Non-canonical："
        f"{stats.noncanonical_file_count}"
    )

    # ========================================================
    # Segment Distribution
    # ========================================================

    histogram = (
        stats.segment_token_histogram
        or Counter()
    )

    if histogram:

        minimum = min(
            histogram
        )

        maximum = max(
            histogram
        )

        mean = (
            _hist_mean(
                histogram
            )
        )

        p50 = (
            _hist_percentile(
                histogram,
                0.50,
            )
        )

        p95 = (
            _hist_percentile(
                histogram,
                0.95,
            )
        )

        p99 = (
            _hist_percentile(
                histogram,
                0.99,
            )
        )

        print(
            "\nAll Valid Segment "
            "Token Counts："
        )

        print(
            f"  Min：{minimum}"
        )

        print(
            f"  Mean：{mean:.4f}"
        )

        print(
            f"  P50：{p50}"
        )

        print(
            f"  P95：{p95}"
        )

        print(
            f"  P99：{p99}"
        )

        print(
            f"  Max：{maximum}"
        )

    # ========================================================
    # Prefill Distribution
    # ========================================================

    prefill_histogram = (
        stats.prefill_token_histogram
        or Counter()
    )

    if prefill_histogram:

        print(
            "\nPrefill Candidate "
            "Token Counts："
        )

        print(
            f"  Min："
            f"{min(prefill_histogram)}"
        )

        print(
            f"  Mean："
            f"{_hist_mean(prefill_histogram):.4f}"
        )

        print(
            "  P50："
            f"{_hist_percentile(prefill_histogram, 0.50)}"
        )

        print(
            "  P95："
            f"{_hist_percentile(prefill_histogram, 0.95)}"
        )

        print(
            "  P99："
            f"{_hist_percentile(prefill_histogram, 0.99)}"
        )

        print(
            f"  Max："
            f"{max(prefill_histogram)}"
        )

        print(
            "\nPrefill Length Buckets："
        )

        buckets = (
            _prefill_length_buckets(
                prefill_histogram
            )
        )

        for (
            name,
            count,
        ) in buckets.items():

            print(
                f"  {name}: "
                f"{count}"
            )

    # ========================================================
    # Category
    # ========================================================

    print(
        "\nCategory Files："
    )

    for (
        category,
        count,
    ) in sorted(
        (
            stats
            .category_file_counts
            or {}
        ).items()
    ):

        print(
            f"  {category}: "
            f"{count}"
        )


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "读取 Chinese-SimpleQA "
                "Segment 级 "
                "Prefill / Decode "
                "候选 Workload。"
            )
        )
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
    )

    parser.add_argument(
        "--show-first",
        type=int,
        default=3,

        help=(
            "打印前 N 个有效 Segment。"
        ),
    )

    parser.add_argument(
        "--show-tokens",
        type=int,
        default=2,

        help=(
            "每个 Segment 最多展示 "
            "前 N 个 Token。"
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    stats = (
        PrefillWorkloadStats()
    )

    shown = 0

    for batch in (
        iter_trace_segment_batches(
            trace_root=(
                args.root
            ),

            max_files=(
                args.max_files
            ),

            max_batches=(
                args.max_batches
            ),

            stats=(
                stats
            ),

            verbose=(
                not args.quiet
            ),
        )
    ):

        if (
            shown
            < args.show_first
        ):

            print_segment_batch(
                batch,

                show_tokens=(
                    args.show_tokens
                ),
            )

            shown += 1

    print_prefill_workload_stats(
        stats
    )


if __name__ == "__main__":
    main()