"""
第六步：纯 Decode Workload 读取。

目标：

    从 Chinese-SimpleQA 原始 Trace 中
    明确排除每个 JSON 的 segment0（Prefill），

    只读取：

        segment1
        segment2
        ...
        segmentN

    作为正式 Decode Token。

------------------------------------------------------------

当前全量数据已经确认：

    2020 / 2020 个 JSON

均满足有效 Segment 结构：

    [N>1, 1, 1, 1, ...]

因此：

    segment0
        ->
    Prefill

    segment1+
        ->
    Decode

------------------------------------------------------------

Decode 语义：

每个有效 Decode Segment 必须：

    1. Layer 3~60 完整；
    2. 共 58 个 MoE Layer；
    3. 每层只有 1 个 Token；
    4. 每个 Token 每层恰好 Top-8 Routed Expert；
    5. 保留 Router 原始顺序，不排序。

输出：

    TraceToken

shape：

    58 × 8

后续可以直接交给：

    scheduling.token_scheduler.schedule_token()

------------------------------------------------------------

注意：

这个文件只负责 Workload 切分。

不做：

    周期计算
    Weight-Cube 调度
    Prefill
    Attention
    KV Cache
"""

from __future__ import annotations

import argparse
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


from mapping.trace_profile import (
    DEFAULT_TRACE_ROOT,
    NUM_MOE_LAYERS,
    discover_trace_files,
)

from scheduling.trace_workload import (
    TraceToken,
    TraceWorkloadError,
    collect_segment_routes,
    validate_runtime_route,
)


# ============================================================
# 异常
# ============================================================


class DecodeWorkloadError(
    ValueError
):
    """Decode Workload 读取失败。"""


# ============================================================
# Stats
# ============================================================


@dataclass(
    slots=True,
)
class DecodeWorkloadStats:
    """
    纯 Decode Workload 扫描统计。
    """

    discovered_file_count: int = 0

    processed_file_count: int = 0

    trace_segment_count: int = 0

    # 每个 JSON 的 segment0
    prefill_segment_count: int = 0

    # segment1+ 中有效的 Decode Segment
    valid_decode_segment_count: int = 0

    # segment1+ 中不完整 / 无效 Segment
    skipped_decode_segment_count: int = 0

    # segment1+ 中出现多 Token
    # 当前数据理论上应为 0
    non_singleton_decode_segment_count: int = 0

    # 最终真正 yield 的 Decode Token 数
    decode_token_count: int = 0


# ============================================================
# JSON
# ============================================================


def _load_json(
    path: Path,
) -> list:
    """
    读取一个 Trace JSON。
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

        raise DecodeWorkloadError(
            f"无法读取 JSON：{path}"
        ) from exc

    if not isinstance(
        data,
        list,
    ):

        raise DecodeWorkloadError(
            f"{path}: "
            "JSON 最外层必须是 list。"
        )

    if not data:

        raise DecodeWorkloadError(
            f"{path}: "
            "JSON 最外层不能为空。"
        )

    return data


# ============================================================
# 单 Decode Segment -> TraceToken
# ============================================================


def build_decode_token(
    *,
    path: Path,
    relative_file: str,
    category: str,
    token_id: int,
    segment_index: int,
    segment: object,
    strict_singleton: bool = True,
) -> TraceToken | None:
    """
    将一个 segment1+ 转成一个纯 Decode TraceToken。

    不完整 Segment：
        return None

    有效但 token_count != 1：
        strict_singleton=True 时直接报错。

    当前 Chinese-SimpleQA 全量数据
    理论上所有有效 segment1+ 都是单 Token。
    """

    if not isinstance(
        segment,
        dict,
    ):

        raise DecodeWorkloadError(
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

    if not raw_routes_by_layer:

        return None

    token_count = len(
        raw_routes_by_layer[
            0
        ][
            1
        ]
    )

    # ========================================================
    # Decode 必须单 Token
    # ========================================================

    if token_count != 1:

        if strict_singleton:

            raise DecodeWorkloadError(
                f"{path}: "
                f"segment-{segment_index} "
                "作为 Decode Segment "
                "却不是单 Token："
                f"token_count={token_count}。"
            )

        return None

    # ========================================================
    # Layer -> Top-8
    # ========================================================

    routed_experts_by_layer: list[
        tuple[
            int,
            ...
        ]
    ] = []

    for (
        trace_layer_id,
        routes,
    ) in raw_routes_by_layer:

        # 当前已确认 token_count == 1
        raw_route = (
            routes[
                0
            ]
        )

        try:

            route = (
                validate_runtime_route(
                    route=(
                        raw_route
                    ),

                    path=(
                        path
                    ),

                    segment_index=(
                        segment_index
                    ),

                    trace_layer_id=(
                        trace_layer_id
                    ),

                    token_index=0,
                )
            )

        except TraceWorkloadError as exc:

            raise DecodeWorkloadError(
                str(exc)
            ) from exc

        routed_experts_by_layer.append(
            route
        )

    if (
        len(
            routed_experts_by_layer
        )
        != NUM_MOE_LAYERS
    ):

        raise DecodeWorkloadError(
            f"{path}: "
            f"segment-{segment_index} "
            "Decode Layer 数错误："
            f"actual="
            f"{len(routed_experts_by_layer)}, "
            f"expected="
            f"{NUM_MOE_LAYERS}。"
        )

    return TraceToken(
        token_id=(
            token_id
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

        token_index_in_segment=0,

        routed_experts_by_layer=tuple(
            routed_experts_by_layer
        ),
    )


# ============================================================
# 主 Generator
# ============================================================


def iter_decode_tokens(
    trace_root: Path | str = (
        DEFAULT_TRACE_ROOT
    ),
    *,
    max_files: int | None = None,
    max_tokens: int | None = None,
    stats: DecodeWorkloadStats | None = None,
    strict_singleton: bool = True,
    verbose: bool = True,
) -> Iterator[
    TraceToken
]:
    """
    流式读取纯 Decode Token。

    规则：

        每个 JSON：

            segment0
                -> 跳过（Prefill）

            segment1+
                -> 只读取有效 singleton

    --------------------------------------------------------

    max_tokens：

        按真正 Decode Token 数截断。

    不会把 Prefill Token 计入。
    """

    root = (
        Path(
            trace_root
        )
        .resolve()
    )

    files = list(
        discover_trace_files(
            root
        )
    )

    if (
        max_files
        is not None
    ):

        if max_files <= 0:

            raise DecodeWorkloadError(
                "max_files 必须大于 0。"
            )

        files = (
            files[
                :max_files
            ]
        )

    if (
        max_tokens
        is not None
        and
        max_tokens <= 0
    ):

        raise DecodeWorkloadError(
            "max_tokens 必须大于 0。"
        )

    if stats is None:

        stats = (
            DecodeWorkloadStats()
        )

    # ========================================================
    # reset
    # ========================================================

    stats.discovered_file_count = (
        len(
            files
        )
    )

    stats.processed_file_count = 0
    stats.trace_segment_count = 0
    stats.prefill_segment_count = 0
    stats.valid_decode_segment_count = 0
    stats.skipped_decode_segment_count = 0
    stats.non_singleton_decode_segment_count = 0
    stats.decode_token_count = 0

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

        relative_file = str(
            relative
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
                "[DecodeWorkload] "
                f"{file_index}/"
                f"{total_files} "
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

            # ================================================
            # segment0 = Prefill
            # ================================================

            if (
                segment_index
                == 0
            ):

                stats.prefill_segment_count += 1

                continue

            # ================================================
            # Decode
            # ================================================

            # 为了给统计区分：
            # 先检查是否有效以及 token_count。
            if not isinstance(
                segment,
                dict,
            ):

                raise DecodeWorkloadError(
                    f"{path}: "
                    f"segment-{segment_index} "
                    "必须是 dict。"
                )

            raw_routes_by_layer = (
                collect_segment_routes(
                    segment=segment
                )
            )

            if (
                raw_routes_by_layer
                is None
            ):

                stats.skipped_decode_segment_count += 1

                continue

            token_count = len(
                raw_routes_by_layer[
                    0
                ][
                    1
                ]
            )

            if token_count != 1:

                stats.non_singleton_decode_segment_count += 1

                if strict_singleton:

                    raise DecodeWorkloadError(
                        f"{path}: "
                        f"segment-{segment_index} "
                        "不是 singleton Decode："
                        f"token_count={token_count}。"
                    )

                continue

            # ================================================
            # 真正构造 TraceToken
            # ================================================

            token = (
                build_decode_token(
                    path=(
                        path
                    ),

                    relative_file=(
                        relative_file
                    ),

                    category=(
                        category
                    ),

                    token_id=(
                        token_id
                    ),

                    segment_index=(
                        segment_index
                    ),

                    segment=(
                        segment
                    ),

                    strict_singleton=(
                        strict_singleton
                    ),
                )
            )

            if token is None:

                stats.skipped_decode_segment_count += 1

                continue

            stats.valid_decode_segment_count += 1
            stats.decode_token_count += 1

            yield token

            token_id += 1

            # ================================================
            # max_tokens
            # ================================================

            if (
                max_tokens
                is not None
                and
                token_id
                >= max_tokens
            ):

                return


# ============================================================
# Scan
# ============================================================


def scan_decode_workload(
    trace_root: Path | str = (
        DEFAULT_TRACE_ROOT
    ),
    *,
    max_files: int | None = None,
    max_tokens: int | None = None,
    strict_singleton: bool = True,
    verbose: bool = True,
) -> DecodeWorkloadStats:

    stats = (
        DecodeWorkloadStats()
    )

    for _token in (
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

            strict_singleton=(
                strict_singleton
            ),

            verbose=(
                verbose
            ),
        )
    ):

        pass

    return stats


# ============================================================
# Print
# ============================================================


def print_decode_token(
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

    print(
        "\n"
        "========== Decode Token =========="
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
        f"Token Index In Segment："
        f"{token.token_index_in_segment}"
    )

    for layer_id in (
        show_layers
    ):

        print(
            f"  L{layer_id}: "
            f"{token.routed_experts_by_layer[layer_id]}"
        )


def print_decode_workload_stats(
    stats: DecodeWorkloadStats,
) -> None:

    print(
        "\n"
        "========== Decode Workload Stats =========="
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
        f"Trace Segments Seen："
        f"{stats.trace_segment_count}"
    )

    print(
        f"Prefill Segment0 Skipped："
        f"{stats.prefill_segment_count}"
    )

    print(
        f"Valid Decode Segments："
        f"{stats.valid_decode_segment_count}"
    )

    print(
        f"Skipped Invalid Decode Segments："
        f"{stats.skipped_decode_segment_count}"
    )

    print(
        f"Non-singleton Decode Segments："
        f"{stats.non_singleton_decode_segment_count}"
    )

    print(
        f"Pure Decode Tokens："
        f"{stats.decode_token_count}"
    )


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "读取 Chinese-SimpleQA "
                "纯 Decode singleton Segment。"
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
        default=3,
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    parser.add_argument(
        "--allow-non-singleton",
        action="store_true",

        help=(
            "遇到 segment1+ 多 Token 时跳过，"
            "而不是报错。"
        ),
    )

    args = (
        parser.parse_args()
    )

    stats = (
        DecodeWorkloadStats()
    )

    shown = 0

    for token in (
        iter_decode_tokens(
            trace_root=(
                args.root
            ),

            max_files=(
                args.max_files
            ),

            max_tokens=(
                args.max_tokens
            ),

            stats=(
                stats
            ),

            strict_singleton=(
                not args
                .allow_non_singleton
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

            print_decode_token(
                token
            )

            shown += 1

    print_decode_workload_stats(
        stats
    )


if __name__ == "__main__":
    main()
