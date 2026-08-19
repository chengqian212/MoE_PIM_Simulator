"""
第五步：Chinese-SimpleQA 真实 Token Workload 读取。

与 mapping.trace_profile.py 的区别：

trace_profile.py：

    读取全部 Trace
        ↓
    只保留：
        frequency
        coactivation

本文件：

    读取全部 Trace
        ↓
    恢复真正的 Token：

        Token-i
            Layer-0 -> Top-8
            Layer-1 -> Top-8
            ...
            Layer-57 -> Top-8

然后交给：

    scheduling.token_scheduler.schedule_token()

------------------------------------------------------------

真实 Trace：

一个 JSON：

[
    segment-0,
    segment-1,
    ...
]

每个 segment：

{
    "0": null,
    "1": null,
    "2": null,

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

同一个完整 segment 内：

    Layer 3~60

必须具有相同的 token route 数量。

因此：

    各 Layer 的 route[token_index]

共同组成一个完整 Token。

例如：

    Layer-3 routes[5]
    Layer-4 routes[5]
    ...
    Layer-60 routes[5]

组成：

    一个 Token 在 58 个 MoE Layer
    上的完整路由。

------------------------------------------------------------

不完整 Segment：

如果出现：

    缺 Layer
    Layer = null
    各 Layer token 数不同
    空 routes

则整个 Segment 跳过。

这与 trace_profile.py 的处理原则保持一致。

------------------------------------------------------------

重要：

这里不会把 Top-8 Expert ID 排序。

原因：

调度阶段可能需要保留 Trace 中的原始 Router 顺序。

这与 trace_profile.py 不同：

    trace_profile.py 为了统计无序 Expert Pair，
    可以排序。

    trace_workload.py 必须保留原始顺序。
"""

from __future__ import annotations

import argparse
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


from mapping.trace_profile import (
    DEFAULT_TRACE_ROOT,
    EXPERTS_PER_TOKEN,
    NUM_MOE_LAYERS,
    NUM_ROUTED_EXPERTS,
    TRACE_FIRST_MOE_LAYER,
    TRACE_LAST_MOE_LAYER,
    discover_trace_files,
)


# ============================================================
# 异常
# ============================================================


class TraceWorkloadError(ValueError):
    """Chinese-SimpleQA Token Workload 读取失败。"""


# ============================================================
# 一个真实 Token
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class TraceToken:
    """
    Chinese-SimpleQA 中一个完整有效 Token。

    routed_experts_by_layer：

        shape = 58 × 8

    例如：

        routed_experts_by_layer[0]

    表示：

        Project MoE Layer-0
        = Trace Layer-3

    的 Routed Top-8。
    """

    token_id: int

    category: str

    relative_file: str

    segment_index: int

    token_index_in_segment: int

    routed_experts_by_layer: tuple[
        tuple[
            int,
            ...
        ],
        ...
    ]

    def __post_init__(
        self,
    ) -> None:

        if self.token_id < 0:
            raise TraceWorkloadError(
                "token_id 不能为负数。"
            )

        if self.segment_index < 0:
            raise TraceWorkloadError(
                "segment_index 不能为负数。"
            )

        if self.token_index_in_segment < 0:
            raise TraceWorkloadError(
                "token_index_in_segment "
                "不能为负数。"
            )

        if (
            len(
                self.routed_experts_by_layer
            )
            != NUM_MOE_LAYERS
        ):
            raise TraceWorkloadError(
                "TraceToken 必须具有 "
                f"{NUM_MOE_LAYERS} 层路由。"
            )

        for (
            layer_id,
            route,
        ) in enumerate(
            self.routed_experts_by_layer
        ):

            if (
                len(route)
                != EXPERTS_PER_TOKEN
            ):
                raise TraceWorkloadError(
                    f"Token-{self.token_id} "
                    f"Layer-{layer_id} "
                    "Top-K 数量错误。"
                )

    def route(
        self,
        layer_id: int,
    ) -> tuple[
        int,
        ...
    ]:
        """
        查询某个 Project MoE Layer 的 Top-8。
        """

        if not (
            0
            <= layer_id
            < NUM_MOE_LAYERS
        ):

            raise TraceWorkloadError(
                f"layer_id={layer_id} "
                "超出范围。"
            )

        return (
            self.routed_experts_by_layer[
                layer_id
            ]
        )


# ============================================================
# 流式读取统计
# ============================================================


@dataclass(
    slots=True,
)
class TraceWorkloadStats:
    """
    iter_trace_tokens() 运行过程中累积的统计。

    注意：

    如果调用者提前停止 Generator，
    这里记录的是：

        已经处理过的数据

    而不是完整数据集统计。
    """

    discovered_file_count: int = 0

    processed_file_count: int = 0

    trace_segment_count: int = 0

    valid_segment_count: int = 0

    skipped_segment_count: int = 0

    yielded_token_count: int = 0

    category_file_counts: dict[
        str,
        int
    ] | None = None

    def __post_init__(
        self,
    ) -> None:

        if (
            self.category_file_counts
            is None
        ):

            self.category_file_counts = {}


# ============================================================
# Top-8 验证
# ============================================================


def validate_runtime_route(
    *,
    route: object,

    path: Path,

    segment_index: int,

    trace_layer_id: int,

    token_index: int,
) -> tuple[
    int,
    ...
]:
    """
    验证运行时 Top-8。

    与 trace_profile.validate_route()
    最大区别：

        这里不排序！

    返回顺序与 JSON 完全一致。

    要求：

    1. list；
    2. 正好 8 个；
    3. int；
    4. ID ∈ [0,255]；
    5. Top-8 内不能重复。
    """

    location = (
        f"{path} | "
        f"segment={segment_index}, "
        f"trace_layer={trace_layer_id}, "
        f"token={token_index}"
    )

    if not isinstance(
        route,
        list,
    ):

        raise TraceWorkloadError(
            f"{location}: "
            "route 必须是 list。"
        )

    if (
        len(route)
        != EXPERTS_PER_TOKEN
    ):

        raise TraceWorkloadError(
            f"{location}: "
            f"每个 token 必须选择 "
            f"{EXPERTS_PER_TOKEN} 个 Expert，"
            f"实际为 {len(route)}。"
        )

    normalized: list[int] = []

    for expert_id in route:

        if (
            not isinstance(
                expert_id,
                int,
            )
            or isinstance(
                expert_id,
                bool,
            )
        ):

            raise TraceWorkloadError(
                f"{location}: "
                "Expert ID 必须是整数，"
                f"当前为 {expert_id!r}。"
            )

        if not (
            0
            <= expert_id
            < NUM_ROUTED_EXPERTS
        ):

            raise TraceWorkloadError(
                f"{location}: "
                f"Expert ID={expert_id} "
                "超出范围 [0,255]。"
            )

        normalized.append(
            expert_id
        )

    if (
        len(
            set(normalized)
        )
        != EXPERTS_PER_TOKEN
    ):

        raise TraceWorkloadError(
            f"{location}: "
            "Top-8 中存在重复 Expert。"
        )

    # ========================================================
    # 重要：
    #
    # 不排序。
    #
    # 保留 Trace 原始 Router 顺序。
    # ========================================================

    return tuple(
        normalized
    )


# ============================================================
# 一个 Segment 完整性检查
# ============================================================


def collect_segment_routes(
    *,
    segment: object,
) -> list[
    tuple[
        int,
        list,
    ]
] | None:
    """
    检查一个 Segment 是否具有完整的
    Trace Layer 3~60。

    返回：

        [
            (3, routes),
            (4, routes),
            ...
            (60, routes)
        ]

    不完整则返回 None。

    --------------------------------------------------------

    以下情况直接跳过整个 Segment：

    1. segment 不是 dict；
       这个属于结构错误，直接报错；

    2. 某个 Layer 缺失；

    3. 某个 Layer = null；

    4. routes 不是 list；

    5. 58 层 token 数不一致；

    6. routes 数量为 0。
    """

    if not isinstance(
        segment,
        dict,
    ):

        raise TraceWorkloadError(
            "Segment 必须是 dict。"
        )

    routes_by_layer: list[
        tuple[
            int,
            list,
        ]
    ] = []

    for trace_layer_id in range(
        TRACE_FIRST_MOE_LAYER,
        TRACE_LAST_MOE_LAYER + 1,
    ):

        key = str(
            trace_layer_id
        )

        if key not in segment:

            return None

        routes = (
            segment[
                key
            ]
        )

        if routes is None:

            return None

        if not isinstance(
            routes,
            list,
        ):

            return None

        routes_by_layer.append(
            (
                trace_layer_id,
                routes,
            )
        )

    # ========================================================
    # 58 层 token 数必须完全一样
    # ========================================================

    route_counts = {
        len(routes)

        for (
            _trace_layer_id,
            routes,
        ) in routes_by_layer
    }

    if (
        len(route_counts)
        != 1
    ):

        return None

    route_count = next(
        iter(
            route_counts
        )
    )

    if (
        route_count
        <= 0
    ):

        return None

    return routes_by_layer


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

        raise TraceWorkloadError(
            f"无法读取 JSON：{path}"
        ) from exc

    if not isinstance(
        data,
        list,
    ):

        raise TraceWorkloadError(
            f"{path}: "
            "JSON 最外层必须是 list。"
        )

    if not data:

        raise TraceWorkloadError(
            f"{path}: "
            "JSON 最外层不能为空。"
        )

    return data


# ============================================================
# 主 Generator
# ============================================================


def iter_trace_tokens(
    trace_root: Path | str = (
        DEFAULT_TRACE_ROOT
    ),
    *,
    max_files: int | None = None,
    max_tokens: int | None = None,
    stats: TraceWorkloadStats | None = None,
    verbose: bool = True,
) -> Iterator[
    TraceToken
]:
    """
    流式遍历 Chinese-SimpleQA 中
    所有有效 Token。

    --------------------------------------------------------

    为什么使用 Generator，而不是：

        list[TraceToken]

    ？

    当前有效 Token：

        285369

    每个 Token：

        58 × 8 Expert ID

    全部一次性构造成 Python 对象
    会占用大量内存。

    因此：

        读一个 JSON
        ↓
        产生 Token
        ↓
        调度
        ↓
        释放

    更合理。

    --------------------------------------------------------

    max_files：

        快速实验时只处理前 N 个 JSON。

    max_tokens：

        最多 yield N 个 Token。

        例如：

            max_tokens=100

        非常适合第五步 Smoke Test。

    --------------------------------------------------------

    stats：

        如果传入 TraceWorkloadStats，
        Generator 会实时更新统计。
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

    if max_files is not None:

        if max_files <= 0:

            raise TraceWorkloadError(
                "max_files 必须大于 0。"
            )

        files = files[
            :max_files
        ]

    # ========================================================
    # max_tokens
    # ========================================================

    if max_tokens is not None:

        if max_tokens <= 0:

            raise TraceWorkloadError(
                "max_tokens 必须大于 0。"
            )

    # ========================================================
    # Stats
    # ========================================================

    if stats is None:

        stats = (
            TraceWorkloadStats()
        )

    # 每次新的遍历从 0 开始。
    stats.discovered_file_count = (
        len(files)
    )

    stats.processed_file_count = 0

    stats.trace_segment_count = 0

    stats.valid_segment_count = 0

    stats.skipped_segment_count = 0

    stats.yielded_token_count = 0

    stats.category_file_counts = {}

    token_id = 0

    total_files = len(
        files
    )

    # ========================================================
    # File
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

        data = _load_json(
            path
        )

        stats.processed_file_count += 1

        # ====================================================
        # Progress
        # ====================================================

        if verbose and (
            file_index == 1
            or file_index == total_files
            or file_index % 100 == 0
        ):

            print(
                f"[Workload] "
                f"{file_index}/{total_files} "
                f"{relative}"
            )

        # ====================================================
        # Segment
        # ====================================================

        for (
            segment_index,
            segment,
        ) in enumerate(
            data
        ):

            stats.trace_segment_count += 1

            if not isinstance(
                segment,
                dict,
            ):

                raise TraceWorkloadError(
                    f"{path}: "
                    f"segment-{segment_index} "
                    "必须是 dict。"
                )

            raw_routes_by_layer = (
                collect_segment_routes(
                    segment=segment
                )
            )

            # =================================================
            # 不完整 Segment
            # =================================================

            if (
                raw_routes_by_layer
                is None
            ):

                stats.skipped_segment_count += 1

                continue

            # =================================================
            # 必须先把整个 Segment
            # 全部 Route 验证完成。
            #
            # 防止已经 yield 一半 Token 后，
            # 才发现后面的 Layer 有非法 route。
            # =================================================

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

                    route = (
                        validate_runtime_route(
                            route=(
                                raw_route
                            ),

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

                    validated_routes.append(
                        route
                    )

                validated_by_layer.append(
                    validated_routes
                )

            stats.valid_segment_count += 1

            # =================================================
            # Transpose：
            #
            # 原始：
            #
            # Layer -> Token -> Top8
            #
            # 转为：
            #
            # Token -> Layer -> Top8
            # =================================================

            token_count_in_segment = len(
                validated_by_layer[
                    0
                ]
            )

            for token_index in range(
                token_count_in_segment
            ):

                routes_by_project_layer = tuple(
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

                token = (
                    TraceToken(
                        token_id=(
                            token_id
                        ),

                        category=(
                            category
                        ),

                        relative_file=str(
                            relative
                        ),

                        segment_index=(
                            segment_index
                        ),

                        token_index_in_segment=(
                            token_index
                        ),

                        routed_experts_by_layer=(
                            routes_by_project_layer
                        ),
                    )
                )

                yield token

                token_id += 1

                stats.yielded_token_count += 1

                # =============================================
                # max_tokens
                # =============================================

                if (
                    max_tokens
                    is not None
                    and
                    token_id
                    >= max_tokens
                ):

                    return


# ============================================================
# 完整 Scan
# ============================================================


def scan_trace_workload(
    trace_root: Path | str = (
        DEFAULT_TRACE_ROOT
    ),
    *,
    max_files: int | None = None,
    max_tokens: int | None = None,
    verbose: bool = True,
) -> TraceWorkloadStats:
    """
    遍历 Workload，但不保存 Token。

    主要用于：

        检查数据数量
        验证与 TraceProfile 是否一致
    """

    stats = (
        TraceWorkloadStats()
    )

    for _token in (
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

            verbose=verbose,
        )
    ):

        pass

    return stats


# ============================================================
# 输出
# ============================================================


def print_trace_token(
    token: TraceToken,
    *,
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
    打印一个 Token 的部分层 Route。
    """

    print(
        "\n"
        "========== Trace Token =========="
    )

    print(
        f"Token ID："
        f"{token.token_id}"
    )

    print(
        f"Category："
        f"{token.category}"
    )

    print(
        f"File："
        f"{token.relative_file}"
    )

    print(
        f"Segment："
        f"{token.segment_index}"
    )

    print(
        "Token Index in Segment："
        f"{token.token_index_in_segment}"
    )

    print(
        "\nRoutes："
    )

    for layer_id in (
        show_layers
    ):

        route = (
            token.route(
                layer_id
            )
        )

        trace_layer_id = (
            layer_id
            + TRACE_FIRST_MOE_LAYER
        )

        print(
            f"  Project L{layer_id} "
            f"(Trace L{trace_layer_id}): "
            f"{route}"
        )


def print_workload_stats(
    stats: TraceWorkloadStats,
) -> None:
    """
    打印读取统计。
    """

    print(
        "\n"
        "========== Trace Workload Stats =========="
    )

    print(
        "Discovered JSON Files："
        f"{stats.discovered_file_count}"
    )

    print(
        "Processed JSON Files："
        f"{stats.processed_file_count}"
    )

    print(
        "Trace Segments："
        f"{stats.trace_segment_count}"
    )

    print(
        "Valid Segments："
        f"{stats.valid_segment_count}"
    )

    print(
        "Skipped Segments："
        f"{stats.skipped_segment_count}"
    )

    print(
        "Yielded Tokens："
        f"{stats.yielded_token_count}"
    )

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
                "真实 Token Route Workload。"
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
        "--max-tokens",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--show-first",
        type=int,
        default=1,

        help=(
            "打印前 N 个 Token。"
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
        TraceWorkloadStats()
    )

    shown = 0

    for token in (
        iter_trace_tokens(
            trace_root=(
                args.root
            ),

            max_files=(
                args.max_files
            ),

            max_tokens=(
                args.max_tokens
            ),

            stats=stats,

            verbose=(
                not args.quiet
            ),
        )
    ):

        if (
            shown
            < args.show_first
        ):

            print_trace_token(
                token
            )

            shown += 1

    print_workload_stats(
        stats
    )


if __name__ == "__main__":
    main()