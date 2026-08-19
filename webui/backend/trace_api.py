"""
Web UI：Chinese-SimpleQA Trace API

作用：

1. 找到 Chinese-SimpleQA trace 数据集；
2. 返回所有类别；
3. 返回某类别下的 JSON 文件；
4. 从真实 trace 中读取一个 Token；
5. 将 DeepSeek Trace Layer 3~60 映射成项目 Layer 0~57；
6. 保留原始 Top-8 Routed Expert 顺序。

注意：

Router trace 只包含：

    Expert 0 ~ 255

Shared Expert 256 不会在这里人为加入。

Shared Expert 会在真正的调度阶段由 scheduler
按照 always-active 规则加入。
"""

from __future__ import annotations

import json
import random

from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)


# ============================================================
# Router
# ============================================================


router = APIRouter(
    prefix="/api/trace",
    tags=["Trace"],
)


# ============================================================
# 固定参数
# ============================================================


TRACE_FIRST_MOE_LAYER = 3
TRACE_LAST_MOE_LAYER = 60

NUM_MOE_LAYERS = (
    TRACE_LAST_MOE_LAYER
    - TRACE_FIRST_MOE_LAYER
    + 1
)

NUM_ROUTED_EXPERTS = 256

EXPERTS_PER_TOKEN = 8


# ============================================================
# 项目路径
#
# 当前文件：
#
# MoE_PIM_Simulator/
# └─ webui/
#    └─ backend/
#       └─ trace_api.py
#
# parents[2]：
#
# MoE_PIM_Simulator/
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
# Trace 错误
# ============================================================


class WebTraceError(
    ValueError
):
    pass


# ============================================================
# 路径检查
# ============================================================


def get_trace_root() -> Path:
    """
    返回 Chinese-SimpleQA trace 根目录。
    """

    root = (
        DEFAULT_TRACE_ROOT
        .resolve()
    )

    if not root.exists():

        raise WebTraceError(
            "找不到 Chinese-SimpleQA Trace："
            f"{root}"
        )

    if not root.is_dir():

        raise WebTraceError(
            "Trace Root 不是目录："
            f"{root}"
        )

    return root


# ============================================================
# Category
# ============================================================


def discover_categories() -> list[dict[str, Any]]:
    """
    读取所有 category。

    返回：

    [
        {
            "name": "中华文化",
            "file_count": 279
        },
        ...
    ]
    """

    root = get_trace_root()

    categories = []


    for path in sorted(
        root.iterdir(),
        key=lambda item:
            item.name,
    ):

        if not path.is_dir():
            continue


        files = sorted(
            path.glob("*.json"),
            key=lambda item:
                item.name,
        )


        if not files:
            continue


        categories.append(
            {
                "name":
                    path.name,

                "file_count":
                    len(files),
            }
        )


    return categories


# ============================================================
# 找某 category
# ============================================================


def get_category_path(
    category: str,
) -> Path:
    """
    安全获得 category 路径。

    防止前端传入：

        ../../xxx
    """

    root = get_trace_root()


    candidate = (
        root
        / category
    ).resolve()


    try:

        candidate.relative_to(
            root
        )

    except ValueError:

        raise WebTraceError(
            "非法 category 路径。"
        )


    if not candidate.exists():

        raise WebTraceError(
            f"不存在类别：{category}"
        )


    if not candidate.is_dir():

        raise WebTraceError(
            f"类别不是目录：{category}"
        )


    return candidate


# ============================================================
# 文件列表
# ============================================================


def discover_category_files(
    category: str,
) -> list[Path]:

    category_path = (
        get_category_path(
            category
        )
    )


    return sorted(
        category_path.glob(
            "*.json"
        ),
        key=lambda item:
            item.name,
    )


# ============================================================
# 安全取得 JSON 文件
# ============================================================


def get_trace_file(
    *,
    category: str,
    filename: str,
) -> Path:

    category_path = (
        get_category_path(
            category
        )
    )


    path = (
        category_path
        / filename
    ).resolve()


    try:

        path.relative_to(
            category_path
        )

    except ValueError:

        raise WebTraceError(
            "非法 Trace 文件路径。"
        )


    if not path.exists():

        raise WebTraceError(
            "找不到 Trace 文件："
            f"{category}/{filename}"
        )


    if not path.is_file():

        raise WebTraceError(
            "Trace 路径不是文件。"
        )


    if (
        path.suffix.lower()
        != ".json"
    ):

        raise WebTraceError(
            "Trace 文件必须是 JSON。"
        )


    return path


# ============================================================
# 读取 JSON
# ============================================================


def load_trace_json(
    path: Path,
) -> list[Any]:

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

    except Exception as exc:

        raise WebTraceError(
            "读取 Trace JSON 失败："
            f"{path.name}；"
            f"{exc}"
        ) from exc


    if not isinstance(
        data,
        list,
    ):

        raise WebTraceError(
            "Trace JSON 最外层必须是 list："
            f"{path.name}"
        )


    return data


# ============================================================
# Top-8 检查
# ============================================================


def validate_route(
    route: Any,
) -> tuple[int, ...]:
    """
    正确 route：

    [
        77,
        96,
        127,
        142,
        149,
        198,
        204,
        146
    ]
    """

    if not isinstance(
        route,
        list,
    ):

        raise WebTraceError(
            "Route 必须是 list。"
        )


    if (
        len(route)
        != EXPERTS_PER_TOKEN
    ):

        raise WebTraceError(
            "每个 Token 必须包含 "
            f"{EXPERTS_PER_TOKEN} "
            "个 Routed Expert，"
            f"当前为 {len(route)}。"
        )


    experts: list[int] = []


    for expert in route:

        if not isinstance(
            expert,
            int,
        ):

            raise WebTraceError(
                "Expert ID 必须是 int。"
            )


        if not (
            0
            <= expert
            < NUM_ROUTED_EXPERTS
        ):

            raise WebTraceError(
                "Routed Expert ID "
                "必须位于 0~255，"
                f"当前为 {expert}。"
            )


        experts.append(
            expert
        )


    if (
        len(set(experts))
        != EXPERTS_PER_TOKEN
    ):

        raise WebTraceError(
            "同一个 Top-8 Route "
            "不能包含重复 Expert。"
        )


    # ========================================================
    # 不排序
    #
    # runtime simulation 要保留 Router 原始顺序。
    # ========================================================

    return tuple(
        experts
    )


# ============================================================
# Segment 检查
# ============================================================


def inspect_segment(
    segment: Any,
) -> int | None:
    """
    检查一个 segment 是否完整。

    返回：

        token_count

    如果该 segment 不完整：

        None

    完整要求：

    1. Layer 3~60 全部存在；
    2. 每层必须为 list；
    3. 每层不能为空；
    4. 58 层 token 数必须完全相同。
    """

    if not isinstance(
        segment,
        dict,
    ):

        return None


    token_count:int | None = None


    for trace_layer_id in range(
        TRACE_FIRST_MOE_LAYER,
        TRACE_LAST_MOE_LAYER + 1,
    ):

        key = str(
            trace_layer_id
        )


        if key not in segment:

            return None


        layer_routes = (
            segment[key]
        )


        if not isinstance(
            layer_routes,
            list,
        ):

            return None


        if not layer_routes:

            return None


        current_count = (
            len(layer_routes)
        )


        if token_count is None:

            token_count = (
                current_count
            )

        elif (
            current_count
            != token_count
        ):

            return None


    return token_count


# ============================================================
# 构造一个真实 Token
# ============================================================


def build_token(
    *,
    category: str,
    filename: str,
    segment_index: int,
    token_index: int,
) -> dict[str, Any]:
    """
    从真实 JSON 中提取一个 Token。

    返回：

    {
        "source": {...},

        "num_layers": 58,

        "layers": [
            {
                "layer_id": 0,
                "trace_layer_id": 3,
                "routed_experts": [...]
            },
            ...
        ]
    }
    """

    path = get_trace_file(
        category=category,
        filename=filename,
    )


    data = load_trace_json(
        path
    )


    if not (
        0
        <= segment_index
        < len(data)
    ):

        raise WebTraceError(
            "segment_index 越界："
            f"{segment_index}；"
            f"segment 数={len(data)}"
        )


    segment = (
        data[segment_index]
    )


    token_count = (
        inspect_segment(
            segment
        )
    )


    if token_count is None:

        raise WebTraceError(
            "选择的 segment 不完整，"
            "不能用于 58 层 Token 模拟。"
        )


    if not (
        0
        <= token_index
        < token_count
    ):

        raise WebTraceError(
            "token_index 越界："
            f"{token_index}；"
            f"该 segment token 数="
            f"{token_count}"
        )


    layers = []


    for project_layer_id in range(
        NUM_MOE_LAYERS
    ):

        trace_layer_id = (
            project_layer_id
            + TRACE_FIRST_MOE_LAYER
        )


        route = (
            segment[
                str(
                    trace_layer_id
                )
            ][
                token_index
            ]
        )


        experts = (
            validate_route(
                route
            )
        )


        layers.append(
            {
                "layer_id":
                    project_layer_id,

                "trace_layer_id":
                    trace_layer_id,

                "routed_experts":
                    list(
                        experts
                    ),

                # Router trace 中没有 Shared-256。
                # 这里单独告诉前端：
                "shared_expert":
                    256,

                "active_expert_count":
                    9,
            }
        )


    return {
        "source": {
            "dataset":
                "Chinese-SimpleQA",

            "category":
                category,

            "filename":
                filename,

            "segment_index":
                segment_index,

            "token_index":
                token_index,
        },

        "num_layers":
            NUM_MOE_LAYERS,

        "routed_experts_per_layer":
            EXPERTS_PER_TOKEN,

        "shared_expert_id":
            256,

        "layers":
            layers,
    }


# ============================================================
# 找一个文件中所有完整 segment
# ============================================================


def get_valid_segments(
    *,
    category: str,
    filename: str,
) -> list[dict[str, int]]:

    path = get_trace_file(
        category=category,
        filename=filename,
    )


    data = load_trace_json(
        path
    )


    valid = []


    for segment_index, segment in enumerate(
        data
    ):

        token_count = (
            inspect_segment(
                segment
            )
        )


        if token_count is None:
            continue


        valid.append(
            {
                "segment_index":
                    segment_index,

                "token_count":
                    token_count,
            }
        )


    return valid


# ============================================================
# 随机找一个有效 Token
# ============================================================


def random_real_token(
    category: str | None = None,
) -> dict[str, Any]:
    """
    从真实数据中随机选择一个合法 Token。
    """

    categories = (
        discover_categories()
    )


    if not categories:

        raise WebTraceError(
            "Chinese-SimpleQA "
            "中没有找到任何类别。"
        )


    if category is None:

        category = (
            random.choice(
                categories
            )["name"]
        )


    files = (
        discover_category_files(
            category
        )
    )


    if not files:

        raise WebTraceError(
            f"{category} 中没有 JSON 文件。"
        )


    # ========================================================
    # 避免遇到不完整文件以后立刻失败。
    #
    # 随机最多尝试 100 个文件。
    # ========================================================

    candidate_files = (
        files.copy()
    )


    random.shuffle(
        candidate_files
    )


    for path in candidate_files[
        :100
    ]:

        valid_segments = (
            get_valid_segments(
                category=category,
                filename=path.name,
            )
        )


        if not valid_segments:

            continue


        chosen_segment = (
            random.choice(
                valid_segments
            )
        )


        token_index = (
            random.randrange(
                chosen_segment[
                    "token_count"
                ]
            )
        )


        return build_token(
            category=category,

            filename=path.name,

            segment_index=(
                chosen_segment[
                    "segment_index"
                ]
            ),

            token_index=token_index,
        )


    raise WebTraceError(
        "随机扫描 100 个 Trace 文件后，"
        "没有找到完整的 58 层 Token。"
    )


# ============================================================
# API:
#
# GET /api/trace/summary
# ============================================================


@router.get(
    "/summary"
)
def trace_summary():

    try:

        categories = (
            discover_categories()
        )


        return {
            "dataset":
                "Chinese-SimpleQA",

            "trace_root":
                str(
                    get_trace_root()
                ),

            "num_moe_layers":
                NUM_MOE_LAYERS,

            "trace_layer_range": [
                TRACE_FIRST_MOE_LAYER,
                TRACE_LAST_MOE_LAYER,
            ],

            "project_layer_range": [
                0,
                NUM_MOE_LAYERS - 1,
            ],

            "num_routed_experts":
                NUM_ROUTED_EXPERTS,

            "experts_per_token":
                EXPERTS_PER_TOKEN,

            "shared_expert_id":
                256,

            "category_count":
                len(
                    categories
                ),

            "file_count":
                sum(
                    item[
                        "file_count"
                    ]
                    for item
                    in categories
                ),
        }


    except WebTraceError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# API:
#
# GET /api/trace/categories
# ============================================================


@router.get(
    "/categories"
)
def trace_categories():

    try:

        items = (
            discover_categories()
        )


        return {
            "items":
                items,

            "count":
                len(items),
        }


    except WebTraceError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# API:
#
# GET /api/trace/files?category=中华文化
# ============================================================


@router.get(
    "/files"
)
def trace_files(
    category: str,

    offset: int = Query(
        default=0,
        ge=0,
    ),

    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
):

    try:

        files = (
            discover_category_files(
                category
            )
        )


        selected = (
            files[
                offset:
                offset + limit
            ]
        )


        return {
            "category":
                category,

            "total":
                len(files),

            "offset":
                offset,

            "limit":
                limit,

            "items": [
                {
                    "filename":
                        path.name,
                }
                for path
                in selected
            ],
        }


    except WebTraceError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


# ============================================================
# API:
#
# GET /api/trace/segments
#     ?category=中华文化
#     &filename=704.json
# ============================================================


@router.get(
    "/segments"
)
def trace_segments(
    category: str,
    filename: str,
):

    try:

        items = (
            get_valid_segments(
                category=category,
                filename=filename,
            )
        )


        return {
            "category":
                category,

            "filename":
                filename,

            "items":
                items,

            "valid_segment_count":
                len(items),
        }


    except WebTraceError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


# ============================================================
# API:
#
# GET /api/trace/token
# ============================================================


@router.get(
    "/token"
)
def trace_token(
    category: str,
    filename: str,

    segment_index: int = Query(
        default=0,
        ge=0,
    ),

    token_index: int = Query(
        default=0,
        ge=0,
    ),
):

    try:

        return build_token(
            category=category,
            filename=filename,
            segment_index=segment_index,
            token_index=token_index,
        )


    except WebTraceError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


# ============================================================
# API:
#
# GET /api/trace/random-token
#
# 或：
#
# GET /api/trace/random-token?category=自然与自然科学
# ============================================================


@router.get(
    "/random-token"
)
def trace_random_token(
    category: str | None = None,
):

    try:

        return random_real_token(
            category=category
        )


    except WebTraceError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


# ============================================================
# 命令行自测
#
# python -m webui.backend.trace_api
# ============================================================


def main() -> None:

    print(
        "========== Web Trace API Self Test =========="
    )


    print(
        "\nTrace Root:"
    )

    print(
        get_trace_root()
    )


    print(
        "\nCategories:"
    )


    categories = (
        discover_categories()
    )


    for item in categories:

        print(
            f"  {item['name']}: "
            f"{item['file_count']} files"
        )


    print(
        "\nRandom Real Token:"
    )


    token = (
        random_real_token()
    )


    source = (
        token["source"]
    )


    print(
        "  Category:",
        source["category"],
    )


    print(
        "  File:",
        source["filename"],
    )


    print(
        "  Segment:",
        source[
            "segment_index"
        ],
    )


    print(
        "  Token:",
        source[
            "token_index"
        ],
    )


    print(
        "\nFirst 5 Layers:"
    )


    for layer in token[
        "layers"
    ][:5]:

        print(
            f"  L{layer['layer_id']}: "
            f"{layer['routed_experts']} "
            f"+ Shared E256"
        )


    print(
        "\nPASS"
    )


if __name__ == "__main__":
    main()