"""
Web UI 多 Token Workload 评估 API。

作用：

    Chinese-SimpleQA Trace
            ↓
    连续读取真实 Token
            ↓
    每个 Token：
        58 Layer
        × Top-8 Routed Experts
            ↓
    token_schedule_api.schedule_token()
            ↓
    多 Token 延迟统计


输出：

    Mean
    P50
    P95
    P99
    Max

    每层平均 latency

    每个 Sub-Cube：
        critical 次数
        task 数
        switch 数

    最慢 Token


重要规则：

1. Trace Layer 3~60
   → Project Layer 0~57

2. 每个 Token 每层恰好 Top-8 Routed Expert。

3. 保持 Router 原始顺序，不排序。

4. Shared Expert 256 不存在于 Trace。
   Scheduler 会自动加入。

5. 不完整 segment 整体跳过。

6. 当前多个 Token 之间按独立 Token 评估，
   不实现 inter-token pipeline。
"""

from __future__ import annotations

import argparse
import json
import math
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from fastapi import (
    APIRouter,
    HTTPException,
)

from pydantic import BaseModel

from webui.backend.token_schedule_api import (
    schedule_token,
)


# ============================================================
# Router
# ============================================================


router = APIRouter(
    prefix="/api/workload",
    tags=["Workload"],
)


# ============================================================
# Trace 参数
# ============================================================


TRACE_FIRST_LAYER = 3
TRACE_LAST_LAYER = 60

NUM_LAYERS = 58

TOP_K = 8

NUM_ROUTED_EXPERTS = 256

NUM_SUBCUBES = 16


# ============================================================
# 项目路径
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


DEFAULT_TRACE_ROOT = (
    PROJECT_ROOT
    / "deepseek_r1_trace"
    / "cognitivecomputations"
    / "DeepSeek-R1-AWQ"
    / "Chinese-SimpleQA"
)


# ============================================================
# Request
# ============================================================


class WorkloadRequest(
    BaseModel
):
    """
    例如：

    {
        "token_count": 100,
        "category": null
    }

    category = null：
        从所有类别读取。

    category = "中华文化"：
        只从这个类别读取。
    """

    token_count: int = 100

    category: str | None = None

    charge_initial_activation: bool = True


# ============================================================
# Trace Token
# ============================================================


@dataclass(
    slots=True,
)
class TraceToken:
    routes: list[list[int]]

    category: str

    file_name: str

    segment_index: int

    token_index: int


# ============================================================
# Category
# ============================================================


def discover_categories() -> list[str]:

    if not DEFAULT_TRACE_ROOT.exists():
        return []


    categories = []


    for path in sorted(
        DEFAULT_TRACE_ROOT.iterdir(),
        key=lambda item:
            item.name,
    ):

        if not path.is_dir():
            continue


        if any(
            path.rglob("*.json")
        ):
            categories.append(
                path.name
            )


    return categories


# ============================================================
# JSON 文件
# ============================================================


def discover_json_files(
    category: str | None,
) -> list[Path]:

    if not DEFAULT_TRACE_ROOT.exists():

        raise RuntimeError(
            "找不到 Chinese-SimpleQA Trace："
            f"{DEFAULT_TRACE_ROOT}"
        )


    # --------------------------------------------------------
    # 所有类别
    # --------------------------------------------------------

    if (
        category is None
        or category == ""
        or category == "ALL"
        or category == "全部"
    ):

        files = list(
            DEFAULT_TRACE_ROOT.rglob(
                "*.json"
            )
        )


    # --------------------------------------------------------
    # 指定类别
    # --------------------------------------------------------

    else:

        category_root = (
            DEFAULT_TRACE_ROOT
            / category
        )


        if not category_root.exists():

            raise ValueError(
                f"Trace 类别不存在：{category}"
            )


        files = list(
            category_root.rglob(
                "*.json"
            )
        )


    files.sort(
        key=lambda path:
            str(
                path.relative_to(
                    DEFAULT_TRACE_ROOT
                )
            )
    )


    return files


# ============================================================
# 判断文件属于哪个 category
# ============================================================


def category_from_path(
    path: Path,
) -> str:

    try:

        relative = (
            path.relative_to(
                DEFAULT_TRACE_ROOT
            )
        )


        if len(relative.parts) >= 2:

            return (
                relative.parts[0]
            )


    except ValueError:
        pass


    return "Unknown"


# ============================================================
# Route 验证
# ============================================================


def validate_route(
    route,
    *,
    path: Path,
    segment_index: int,
    layer_id: int,
    token_index: int,
) -> list[int]:

    if not isinstance(
        route,
        list,
    ):

        raise RuntimeError(
            f"{path}: "
            f"segment={segment_index}, "
            f"layer={layer_id}, "
            f"token={token_index} "
            "route 不是 list。"
        )


    if len(route) != TOP_K:

        raise RuntimeError(
            f"{path}: "
            f"segment={segment_index}, "
            f"layer={layer_id}, "
            f"token={token_index} "
            f"应该有 {TOP_K} 个 Expert，"
            f"实际为 {len(route)}。"
        )


    normalized = []


    for expert_id in route:

        if not isinstance(
            expert_id,
            int,
        ):

            raise RuntimeError(
                f"{path}: Expert ID "
                "必须是整数。"
            )


        if not (
            0
            <= expert_id
            < NUM_ROUTED_EXPERTS
        ):

            raise RuntimeError(
                f"{path}: 非法 Expert ID "
                f"{expert_id}。"
            )


        normalized.append(
            expert_id
        )


    if (
        len(
            set(
                normalized
            )
        )
        != TOP_K
    ):

        raise RuntimeError(
            f"{path}: "
            f"segment={segment_index}, "
            f"layer={layer_id}, "
            f"token={token_index} "
            "Top-8 中存在重复 Expert。"
        )


    # --------------------------------------------------------
    # 不排序。
    #
    # Router 顺序必须保留。
    # --------------------------------------------------------

    return normalized


# ============================================================
# 从单个 segment 提取 Token
# ============================================================


def extract_segment_tokens(
    segment,
    *,
    path: Path,
    segment_index: int,
) -> Iterator[TraceToken]:

    if not isinstance(
        segment,
        dict,
    ):
        return


    # ========================================================
    # 第一阶段：
    # 检查 58 层是否完整
    # ========================================================

    routes_by_layer = []


    for trace_layer in range(
        TRACE_FIRST_LAYER,
        TRACE_LAST_LAYER + 1,
    ):

        key = str(
            trace_layer
        )


        if key not in segment:

            # 整个 segment 跳过
            return


        routes = (
            segment[
                key
            ]
        )


        if routes is None:
            return


        if not isinstance(
            routes,
            list,
        ):

            raise RuntimeError(
                f"{path}: "
                f"segment={segment_index}, "
                f"Trace Layer {trace_layer} "
                "必须是 list。"
            )


        routes_by_layer.append(
            routes
        )


    # ========================================================
    # 第二阶段：
    # 58 层 token 数必须一致
    # ========================================================

    token_counts = {
        len(routes)
        for routes
        in routes_by_layer
    }


    if len(token_counts) != 1:

        # 与原 Trace 处理规则一致：
        # segment 整体跳过。
        return


    token_count = next(
        iter(
            token_counts
        )
    )


    if token_count <= 0:
        return


    category = (
        category_from_path(
            path
        )
    )


    # ========================================================
    # 第三个阶段：
    # 一个 token 一个 token 构建 58 层 route
    # ========================================================

    for token_index in range(
        token_count
    ):

        token_routes = []


        for project_layer in range(
            NUM_LAYERS
        ):

            trace_layer = (
                TRACE_FIRST_LAYER
                + project_layer
            )


            raw_route = (
                routes_by_layer[
                    project_layer
                ][
                    token_index
                ]
            )


            route = (
                validate_route(
                    raw_route,

                    path=path,

                    segment_index=(
                        segment_index
                    ),

                    layer_id=(
                        trace_layer
                    ),

                    token_index=(
                        token_index
                    ),
                )
            )


            token_routes.append(
                route
            )


        yield TraceToken(
            routes=token_routes,

            category=category,

            file_name=path.name,

            segment_index=(
                segment_index
            ),

            token_index=(
                token_index
            ),
        )


# ============================================================
# 流式读取 Trace
#
# 不把 28 万多个 Token 全部放进内存。
# ============================================================


def iter_trace_tokens(
    *,
    category: str | None,
) -> Iterator[TraceToken]:

    files = (
        discover_json_files(
            category
        )
    )


    for path in files:

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

            raise RuntimeError(
                f"读取 Trace 文件失败："
                f"{path}"
            ) from exc


        if not isinstance(
            data,
            list,
        ):

            raise RuntimeError(
                f"{path}: "
                "Trace JSON 最外层必须是 list。"
            )


        for (
            segment_index,
            segment,
        ) in enumerate(
            data
        ):

            yield from (
                extract_segment_tokens(
                    segment,

                    path=path,

                    segment_index=(
                        segment_index
                    ),
                )
            )


# ============================================================
# Percentile
#
# 使用线性插值：
#
# position = (n - 1) * p
# ============================================================


def percentile(
    sorted_values: list[int],
    percent: float,
) -> float:

    if not sorted_values:

        return 0.0


    if len(sorted_values) == 1:

        return float(
            sorted_values[0]
        )


    position = (
        len(sorted_values) - 1
    ) * percent


    lower_index = math.floor(
        position
    )


    upper_index = math.ceil(
        position
    )


    if (
        lower_index
        == upper_index
    ):

        return float(
            sorted_values[
                lower_index
            ]
        )


    fraction = (
        position
        - lower_index
    )


    lower = (
        sorted_values[
            lower_index
        ]
    )


    upper = (
        sorted_values[
            upper_index
        ]
    )


    return (
        lower
        +
        (
            upper - lower
        )
        * fraction
    )


# ============================================================
# Latency Histogram
#
# 给前端画分布图使用。
#
# 默认每 10 cycles 一个区间。
# ============================================================


def build_histogram(
    values: list[int],
    *,
    bin_size: int = 10,
) -> list[dict]:

    if not values:
        return []


    minimum = min(
        values
    )


    maximum = max(
        values
    )


    first_bin = (
        minimum
        // bin_size
        * bin_size
    )


    last_bin = (
        maximum
        // bin_size
        * bin_size
    )


    counts: dict[
        int,
        int,
    ] = {}


    for value in values:

        start = (
            value
            // bin_size
            * bin_size
        )


        counts[
            start
        ] = (
            counts.get(
                start,
                0,
            )
            + 1
        )


    result = []


    for start in range(
        first_bin,
        last_bin + bin_size,
        bin_size,
    ):

        result.append(
            {
                "start":
                    start,

                "end":
                    start
                    + bin_size
                    - 1,

                "count":
                    counts.get(
                        start,
                        0,
                    ),
            }
        )


    return result


# ============================================================
# 多 Token 评估
# ============================================================


def evaluate_workload(
    *,
    token_count: int,

    category: str | None,

    charge_initial_activation: bool,
) -> dict:

    if token_count <= 0:

        raise ValueError(
            "token_count 必须大于 0。"
        )


    # Web UI 先限制一下。
    #
    # 后续如果确实要从网页直接跑 10000，
    # 再开放。
    if token_count > 10000:

        raise ValueError(
            "Web UI 单次最多评估 "
            "10000 个 Token。"
        )


    start_time = (
        time.perf_counter()
    )


    # ========================================================
    # Token latency
    # ========================================================

    latencies: list[int] = []


    # ========================================================
    # 每层累计 latency
    # ========================================================

    layer_cycle_sum = [
        0
        for _ in range(
            NUM_LAYERS
        )
    ]


    layer_cycle_max = [
        0
        for _ in range(
            NUM_LAYERS
        )
    ]


    # ========================================================
    # SC 累计
    # ========================================================

    sc_task_count = [
        0
        for _ in range(
            NUM_SUBCUBES
        )
    ]


    sc_switch_count = [
        0
        for _ in range(
            NUM_SUBCUBES
        )
    ]


    sc_critical_count = [
        0
        for _ in range(
            NUM_SUBCUBES
        )
    ]


    # ========================================================
    # 最慢 Token
    # ========================================================

    slowest_tokens = []


    # ========================================================
    # 来源统计
    # ========================================================

    category_counts: dict[
        str,
        int,
    ] = {}


    evaluated = 0


    # ========================================================
    # Streaming
    # ========================================================

    for trace_token in (
        iter_trace_tokens(
            category=category
        )
    ):

        schedule = (
            schedule_token(
                routes=(
                    trace_token.routes
                ),

                charge_initial_activation=(
                    charge_initial_activation
                ),

                # 多 Token 统计不需要返回
                # 1566 个详细 task。
                include_tasks=False,
            )
        )


        latency = int(
            schedule[
                "total_cycles"
            ]
        )


        latencies.append(
            latency
        )


        # ====================================================
        # Layer stats
        # ====================================================

        for layer in (
            schedule[
                "layers"
            ]
        ):

            layer_id = int(
                layer[
                    "layer_id"
                ]
            )


            cycles = int(
                layer[
                    "layer_cycles"
                ]
            )


            layer_cycle_sum[
                layer_id
            ] += cycles


            layer_cycle_max[
                layer_id
            ] = max(
                layer_cycle_max[
                    layer_id
                ],
                cycles,
            )


        # ====================================================
        # SC stats
        # ====================================================

        for sc in (
            schedule[
                "subcubes"
            ]
        ):

            sc_id = int(
                sc[
                    "subcube_id"
                ]
            )


            sc_task_count[
                sc_id
            ] += int(
                sc.get(
                    "task_count",
                    0,
                )
            )


            sc_switch_count[
                sc_id
            ] += int(
                sc.get(
                    "switch_count",
                    0,
                )
            )


            sc_critical_count[
                sc_id
            ] += int(
                sc.get(
                    "critical_layer_count",
                    0,
                )
            )


        # ====================================================
        # Category count
        # ====================================================

        category_counts[
            trace_token.category
        ] = (
            category_counts.get(
                trace_token.category,
                0,
            )
            + 1
        )


        # ====================================================
        # Slowest token candidate
        # ====================================================

        slowest_tokens.append(
            {
                "latency":
                    latency,

                "category":
                    trace_token.category,

                "file_name":
                    trace_token.file_name,

                "segment_index":
                    trace_token.segment_index,

                "token_index":
                    trace_token.token_index,
            }
        )


        # 只保留当前最慢的 20 个，
        # 防止列表无限增长。
        if (
            len(
                slowest_tokens
            )
            > 20
        ):

            slowest_tokens.sort(
                key=lambda item: (
                    -item[
                        "latency"
                    ],

                    item[
                        "file_name"
                    ],

                    item[
                        "segment_index"
                    ],

                    item[
                        "token_index"
                    ],
                )
            )


            del slowest_tokens[
                20:
            ]


        evaluated += 1


        if (
            evaluated
            >= token_count
        ):
            break


    # ========================================================
    # Token 不足
    # ========================================================

    if evaluated == 0:

        raise RuntimeError(
            "没有找到可用于评估的完整 Trace Token。"
        )


    # 如果指定数量超过了该 category
    # 实际拥有的 Token，则返回实际数量。
    # 不把它当成错误。
    # ========================================================


    elapsed = (
        time.perf_counter()
        - start_time
    )


    # ========================================================
    # Latency statistics
    # ========================================================

    sorted_latency = sorted(
        latencies
    )


    mean_latency = (
        sum(
            latencies
        )
        /
        evaluated
    )


    minimum = (
        sorted_latency[0]
    )


    maximum = (
        sorted_latency[-1]
    )


    p50 = percentile(
        sorted_latency,
        0.50,
    )


    p95 = percentile(
        sorted_latency,
        0.95,
    )


    p99 = percentile(
        sorted_latency,
        0.99,
    )


    # ========================================================
    # Layer statistics
    # ========================================================

    layers = []


    for layer_id in range(
        NUM_LAYERS
    ):

        layers.append(
            {
                "layer_id":
                    layer_id,

                "mean_cycles":
                    (
                        layer_cycle_sum[
                            layer_id
                        ]
                        /
                        evaluated
                    ),

                "max_cycles":
                    layer_cycle_max[
                        layer_id
                    ],
            }
        )


    # 最慢平均 Layer
    slowest_layers = sorted(
        layers,

        key=lambda item: (
            -item[
                "mean_cycles"
            ],

            item[
                "layer_id"
            ],
        ),
    )[:10]


    # ========================================================
    # SC statistics
    #
    # 分母：
    #
    # evaluated × 58
    #
    # 因为每个 Token 有 58 个 Layer。
    # ========================================================

    total_layer_events = (
        evaluated
        * NUM_LAYERS
    )


    subcubes = []


    for sc_id in range(
        NUM_SUBCUBES
    ):

        subcubes.append(
            {
                "subcube_id":
                    sc_id,

                "task_count":
                    sc_task_count[
                        sc_id
                    ],

                "switch_count":
                    sc_switch_count[
                        sc_id
                    ],

                "critical_layer_count":
                    sc_critical_count[
                        sc_id
                    ],

                "critical_layer_rate":
                    (
                        sc_critical_count[
                            sc_id
                        ]
                        /
                        total_layer_events
                    ),
            }
        )


    # ========================================================
    # Slowest tokens
    # ========================================================

    slowest_tokens.sort(
        key=lambda item: (
            -item[
                "latency"
            ],

            item[
                "file_name"
            ],

            item[
                "segment_index"
            ],

            item[
                "token_index"
            ],
        )
    )


    slowest_tokens = (
        slowest_tokens[
            :10
        ]
    )


    # ========================================================
    # Result
    # ========================================================

    return {
        "requested_token_count":
            token_count,

        "evaluated_token_count":
            evaluated,

        "category":
            category
            or "ALL",

        "elapsed_seconds":
            elapsed,

        "latency": {
            "mean":
                mean_latency,

            "min":
                minimum,

            "p50":
                p50,

            "p95":
                p95,

            "p99":
                p99,

            "max":
                maximum,
        },

        "histogram":
            build_histogram(
                latencies,
                bin_size=10,
            ),

        "layers":
            layers,

        "slowest_layers":
            slowest_layers,

        "subcubes":
            subcubes,

        "slowest_tokens":
            slowest_tokens,

        "source_counts":
            category_counts,
    }


# ============================================================
# API：Health
# ============================================================


@router.get(
    "/health"
)
def workload_health():

    return {
        "status":
            "ok",

        "trace_root":
            str(
                DEFAULT_TRACE_ROOT
            ),

        "trace_exists":
            DEFAULT_TRACE_ROOT.exists(),

        "categories":
            discover_categories(),
    }


# ============================================================
# API：Categories
# ============================================================


@router.get(
    "/categories"
)
def workload_categories():

    return {
        "categories":
            discover_categories()
    }


# ============================================================
# API：Evaluate
# ============================================================


@router.post(
    "/evaluate"
)
def evaluate(
    request: WorkloadRequest,
):

    try:

        return evaluate_workload(
            token_count=(
                request.token_count
            ),

            category=(
                request.category
            ),

            charge_initial_activation=(
                request
                .charge_initial_activation
            ),
        )


    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


    except RuntimeError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# CLI Self Test
#
# python -m webui.backend.workload_api --tokens 10
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser()
    )


    parser.add_argument(
        "--tokens",

        type=int,

        default=10,
    )


    parser.add_argument(
        "--category",

        type=str,

        default=None,
    )


    args = (
        parser.parse_args()
    )


    print(
        "========== "
        "Workload API Self Test "
        "=========="
    )


    print(
        "Trace root:",
        DEFAULT_TRACE_ROOT,
    )


    print(
        "Categories:",
        discover_categories(),
    )


    print(
        "Requested tokens:",
        args.tokens,
    )


    print(
        "Category:",
        args.category
        or "ALL",
    )


    result = (
        evaluate_workload(
            token_count=(
                args.tokens
            ),

            category=(
                args.category
            ),

            charge_initial_activation=True,
        )
    )


    print(
        "\nEvaluated:",
        result[
            "evaluated_token_count"
        ],
    )


    print(
        "Elapsed:",
        f"{result['elapsed_seconds']:.3f}s",
    )


    latency = (
        result[
            "latency"
        ]
    )


    print(
        "\nLatency:"
    )


    print(
        "  Mean:",
        f"{latency['mean']:.4f}",
    )


    print(
        "  Min :",
        latency[
            "min"
        ],
    )


    print(
        "  P50 :",
        latency[
            "p50"
        ],
    )


    print(
        "  P95 :",
        latency[
            "p95"
        ],
    )


    print(
        "  P99 :",
        latency[
            "p99"
        ],
    )


    print(
        "  Max :",
        latency[
            "max"
        ],
    )


    print(
        "\nSlowest layers:"
    )


    for layer in (
        result[
            "slowest_layers"
        ][
            :5
        ]
    ):

        print(
            "  "
            f"L{layer['layer_id']:>2} "
            f"mean="
            f"{layer['mean_cycles']:.4f} "
            f"max="
            f"{layer['max_cycles']}"
        )


    print(
        "\nCritical SC:"
    )


    top_sc = sorted(
        result[
            "subcubes"
        ],

        key=lambda item:
            -item[
                "critical_layer_count"
            ],
    )[:5]


    for sc in top_sc:

        print(
            "  "
            f"SC-{sc['subcube_id']}: "
            f"{sc['critical_layer_count']} "
            "critical layers, "
            f"rate="
            f"{sc['critical_layer_rate']:.4%}"
        )


    assert (
        result[
            "evaluated_token_count"
        ]
        > 0
    )


    assert (
        len(
            result[
                "layers"
            ]
        )
        == 58
    )


    assert (
        len(
            result[
                "subcubes"
            ]
        )
        == 16
    )


    print(
        "\nPASS"
    )


if __name__ == "__main__":
    main()