"""
第四步：Chinese-SimpleQA Expert 路由 Trace 统计。

作用：
1. 递归读取 Chinese-SimpleQA 下所有类别文件夹中的 JSON；
2. 将 DeepSeek trace 的 Layer 3~60 映射到本项目 MoE Layer 0~57；
3. 统计 Routed Expert 访问频率：
       frequency[layer][expert]
4. 统计 Routed Expert 两两共激活次数：
       coactivation[layer][expert_a][expert_b]

说明：
- Trace 中只包含 Routed Expert 0~255；
- Shared Expert 256 不在本文件中补入，
  后续映射阶段按 always-active 单独处理；
- 如果某个 segment 的 58 个 MoE 层中存在：
    缺层
    null
    空路由
    各层 token 数不一致
  则整个 segment 跳过；
- 对一个完整 segment 内部的非法 Top-8：
    Expert 数不是 8
    Expert ID 越界
    Expert 重复
  则直接报错。

这样可以保证：
58 个 MoE Layer 始终使用完全相同的一批有效 token 样本。
"""

from __future__ import annotations

import argparse
import json

from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ============================================================
# Trace 固定参数
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
# 默认数据集路径
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


DEFAULT_TRACE_ROOT = (
    PROJECT_ROOT
    / "deepseek_r1_trace"
    / "cognitivecomputations"
    / "DeepSeek-R1-AWQ"
    / "Chinese-SimpleQA"
)


# ============================================================
# 异常
# ============================================================


class TraceProfileError(ValueError):
    """Trace 格式或统计过程出现错误。"""


# ============================================================
# TraceProfile
# ============================================================


@dataclass(slots=True)
class TraceProfile:
    """
    Chinese-SimpleQA 的 Routed Expert 路由统计。
    """

    # ========================================================
    # 文件与 Segment
    # ========================================================

    file_count: int

    # JSON 最外层 segment 总数
    trace_segment_count: int

    # 因缺层 / null / 长度不一致等被跳过的 segment
    skipped_segment_count: int

    # 每个类别读取了多少 JSON
    category_file_counts: dict[str, int]

    # ========================================================
    # 路由统计
    # ========================================================

    # shape:
    #     58 × 256
    #
    # frequency[layer][expert]
    frequency: tuple[
        tuple[int, ...],
        ...
    ]

    # shape:
    #     58 × (256 * 256)
    #
    # 实际只使用：
    #
    #     expert_a < expert_b
    #
    # 对应的上三角位置。
    coactivation: tuple[
        array,
        ...
    ]

    # 每个项目 MoE Layer
    # 实际统计到多少个 token
    token_count_by_layer: tuple[
        int,
        ...
    ]

    # ========================================================
    # ID 检查
    # ========================================================

    def validate_layer_id(
        self,
        layer_id: int,
    ) -> None:

        if not (
            0
            <= layer_id
            < NUM_MOE_LAYERS
        ):
            raise TraceProfileError(
                "layer_id 必须位于 "
                f"[0, {NUM_MOE_LAYERS - 1}]，"
                f"当前为 {layer_id}。"
            )

    def validate_expert_id(
        self,
        expert_id: int,
    ) -> None:

        if not (
            0
            <= expert_id
            < NUM_ROUTED_EXPERTS
        ):
            raise TraceProfileError(
                "Routed Expert ID 必须位于 "
                f"[0, {NUM_ROUTED_EXPERTS - 1}]，"
                f"当前为 {expert_id}。"
            )

    # ========================================================
    # Frequency 查询
    # ========================================================

    def frequency_count(
        self,
        layer_id: int,
        expert_id: int,
    ) -> int:
        """
        返回某层某 Routed Expert
        被选择的总次数。
        """

        self.validate_layer_id(
            layer_id
        )

        self.validate_expert_id(
            expert_id
        )

        return self.frequency[
            layer_id
        ][
            expert_id
        ]

    # ========================================================
    # Coactivation 查询
    # ========================================================

    def coactivation_count(
        self,
        layer_id: int,
        expert_a: int,
        expert_b: int,
    ) -> int:
        """
        返回两个 Routed Expert
        在同一层、同一 token 中
        被共同选择的次数。

        G(a,b) = G(b,a)
        """

        self.validate_layer_id(
            layer_id
        )

        self.validate_expert_id(
            expert_a
        )

        self.validate_expert_id(
            expert_b
        )

        if expert_a == expert_b:
            raise TraceProfileError(
                "共激活查询要求两个不同 Expert。"
            )

        first = min(
            expert_a,
            expert_b,
        )

        second = max(
            expert_a,
            expert_b,
        )

        index = (
            first
            * NUM_ROUTED_EXPERTS
            + second
        )

        return int(
            self.coactivation[
                layer_id
            ][
                index
            ]
        )

    # ========================================================
    # 热门 Expert
    # ========================================================

    def top_experts(
        self,
        layer_id: int,
        k: int = 10,
    ) -> tuple[
        tuple[int, int],
        ...
    ]:
        """
        返回某层访问最高的 k 个 Expert。

        返回：

        (
            (expert_id, frequency),
            ...
        )
        """

        self.validate_layer_id(
            layer_id
        )

        if k <= 0:
            raise TraceProfileError(
                "k 必须大于 0。"
            )

        k = min(
            k,
            NUM_ROUTED_EXPERTS,
        )

        ranked = sorted(
            range(
                NUM_ROUTED_EXPERTS
            ),
            key=lambda expert_id: (
                -self.frequency[
                    layer_id
                ][
                    expert_id
                ],
                expert_id,
            ),
        )

        return tuple(
            (
                expert_id,
                self.frequency[
                    layer_id
                ][
                    expert_id
                ],
            )
            for expert_id
            in ranked[:k]
        )

    # ========================================================
    # 汇总属性
    # ========================================================

    @property
    def valid_segment_count(
        self,
    ) -> int:
        """
        真正参与统计的 segment 数。
        """

        return (
            self.trace_segment_count
            - self.skipped_segment_count
        )

    @property
    def tokens_per_layer(
        self,
    ) -> int:
        """
        每个 MoE Layer 最终统计到的 token 数。

        当前处理方式保证 58 层应该相同。
        """

        if not self.token_count_by_layer:
            return 0

        return (
            self.token_count_by_layer[0]
        )

    @property
    def total_layer_token_events(
        self,
    ) -> int:
        """
        所有 Layer-Token 事件总数。

        如果每层有 T 个 token：

            58 × T
        """

        return sum(
            self.token_count_by_layer
        )

    @property
    def total_expert_selections(
        self,
    ) -> int:
        """
        Routed Expert 总选择次数。

        理论上：

            tokens_per_layer
            × 58
            × 8
        """

        return sum(
            sum(
                layer_frequency
            )
            for layer_frequency
            in self.frequency
        )


# ============================================================
# Layer ID 映射
# ============================================================


def trace_layer_to_project_layer(
    trace_layer_id: int,
) -> int:
    """
    Trace Layer：

        3 ... 60

    转换为项目：

        0 ... 57
    """

    if not (
        TRACE_FIRST_MOE_LAYER
        <= trace_layer_id
        <= TRACE_LAST_MOE_LAYER
    ):
        raise TraceProfileError(
            "Trace MoE Layer 必须位于 "
            f"[{TRACE_FIRST_MOE_LAYER}, "
            f"{TRACE_LAST_MOE_LAYER}]，"
            f"当前为 {trace_layer_id}。"
        )

    return (
        trace_layer_id
        - TRACE_FIRST_MOE_LAYER
    )


def project_layer_to_trace_layer(
    layer_id: int,
) -> int:
    """
    项目 Layer：

        0 ... 57

    转换回：

        3 ... 60
    """

    if not (
        0
        <= layer_id
        < NUM_MOE_LAYERS
    ):
        raise TraceProfileError(
            "Project layer_id 必须位于 "
            f"[0, {NUM_MOE_LAYERS - 1}]。"
        )

    return (
        layer_id
        + TRACE_FIRST_MOE_LAYER
    )


# ============================================================
# Top-8 路由合法性
# ============================================================


def validate_route(
    *,
    route: object,
    path: Path,
    segment_index: int,
    trace_layer_id: int,
    token_index: int,
) -> tuple[int, ...]:
    """
    验证一次 token 的 Top-8 路由。

    要求：

    1. route 是 list；
    2. 恰好 8 个 Expert；
    3. Expert ID 是整数；
    4. ID ∈ [0,255]；
    5. 同一次 Top-8 不允许重复 Expert。

    返回排序后的 tuple。

    排序只为了统计 Expert pair，
    不改变路由语义。
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
        raise TraceProfileError(
            f"{location}: "
            "路由结果必须是 list。"
        )

    if (
        len(route)
        != EXPERTS_PER_TOKEN
    ):
        raise TraceProfileError(
            f"{location}: "
            f"每个 token 必须选择 "
            f"{EXPERTS_PER_TOKEN} 个 Expert，"
            f"实际为 {len(route)}。"
        )

    normalized: list[int] = []

    for expert_id in route:

        # bool 是 int 的子类，
        # 因此必须额外排除。
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
            raise TraceProfileError(
                f"{location}: "
                "Expert ID 必须是整数，"
                f"当前为 {expert_id!r}。"
            )

        if not (
            0
            <= expert_id
            < NUM_ROUTED_EXPERTS
        ):
            raise TraceProfileError(
                f"{location}: "
                f"Expert ID={expert_id} "
                "超出 Routed Expert 范围 "
                "[0,255]。"
            )

        normalized.append(
            expert_id
        )

    if (
        len(normalized)
        != len(
            set(normalized)
        )
    ):
        raise TraceProfileError(
            f"{location}: "
            "同一个 token 的 Top-8 中 "
            "出现重复 Expert。"
        )

    return tuple(
        sorted(
            normalized
        )
    )


# ============================================================
# JSON 文件发现
# ============================================================


def discover_trace_files(
    trace_root: Path | str,
) -> tuple[
    Path,
    ...
]:
    """
    递归读取 Chinese-SimpleQA
    下面所有类别目录中的 JSON。

    例如：

    Chinese-SimpleQA/
        工程、技术与应用科学/
        人文与社会科学/
        社会/
        生活、艺术与文化/
        中华文化/
        自然与自然科学/
    """

    root = (
        Path(trace_root)
        .resolve()
    )

    if not root.exists():
        raise TraceProfileError(
            "Chinese-SimpleQA 路径不存在："
            f"{root}"
        )

    if not root.is_dir():
        raise TraceProfileError(
            "Chinese-SimpleQA 路径不是目录："
            f"{root}"
        )

    files = tuple(
        sorted(
            (
                path
                for path
                in root.rglob(
                    "*.json"
                )
                if path.is_file()
            ),
            key=lambda path: (
                str(
                    path.relative_to(
                        root
                    )
                )
            ),
        )
    )

    if not files:
        raise TraceProfileError(
            "没有在 Chinese-SimpleQA 下 "
            "找到任何 JSON 文件："
            f"{root}"
        )

    return files


# ============================================================
# 单次路由统计
# ============================================================


def accumulate_route(
    *,
    layer_id: int,
    route: tuple[
        int,
        ...
    ],
    frequency: list[
        list[int]
    ],
    coactivation: list[
        array
    ],
) -> None:
    """
    将一个 token 在一个 Layer 中的 Top-8
    加入统计。

    Frequency：

        8 个 Expert 各 +1

    Coactivation：

        Top-8 两两组合：

        C(8,2) = 28

        每个 pair +1。
    """

    layer_frequency = (
        frequency[layer_id]
    )

    layer_coactivation = (
        coactivation[layer_id]
    )

    # ========================================================
    # Frequency
    # ========================================================

    for expert_id in route:

        layer_frequency[
            expert_id
        ] += 1

    # ========================================================
    # Coactivation
    # ========================================================

    for i in range(
        len(route) - 1
    ):

        expert_a = route[i]

        base_index = (
            expert_a
            * NUM_ROUTED_EXPERTS
        )

        for j in range(
            i + 1,
            len(route),
        ):

            expert_b = route[j]

            pair_index = (
                base_index
                + expert_b
            )

            layer_coactivation[
                pair_index
            ] += 1


# ============================================================
# Segment 完整性检查
# ============================================================


def collect_complete_segment_routes(
    *,
    segment: dict,
) -> list[
    tuple[
        int,
        list,
    ]
] | None:
    """
    检查一个 segment 是否拥有完整的 58 层路由。

    完整：
        返回
        [
            (3, routes),
            (4, routes),
            ...
            (60, routes)
        ]

    不完整：
        返回 None

    下列情况整个 segment 跳过：

    1. 某个 MoE Layer 缺失；
    2. 某个 MoE Layer = null；
    3. 某个 routes 不是 list；
    4. 58 层 token 数量不一致；
    5. token 数量为 0。

    注意：

    这里只判断 segment 是否完整。

    具体 Top-8 route 是否非法，
    后面由 validate_route() 检查。
    """

    raw_routes_by_layer: list[
        tuple[
            int,
            list,
        ]
    ] = []

    # ========================================================
    # 检查 3~60
    # ========================================================

    for trace_layer_id in range(
        TRACE_FIRST_MOE_LAYER,
        TRACE_LAST_MOE_LAYER + 1,
    ):

        key = str(
            trace_layer_id
        )

        if key not in segment:
            return None

        routes = segment[key]

        if routes is None:
            return None

        if not isinstance(
            routes,
            list,
        ):
            return None

        raw_routes_by_layer.append(
            (
                trace_layer_id,
                routes,
            )
        )

    # ========================================================
    # 检查各层 token 数是否一致
    # ========================================================

    route_counts = {
        len(routes)
        for _, routes
        in raw_routes_by_layer
    }

    if len(route_counts) != 1:
        return None

    route_count = next(
        iter(
            route_counts
        )
    )

    if route_count == 0:
        return None

    return raw_routes_by_layer


# ============================================================
# 单 JSON 文件处理
# ============================================================


def process_trace_file(
    *,
    path: Path,
    frequency: list[
        list[int]
    ],
    coactivation: list[
        array
    ],
    token_count_by_layer: list[
        int
    ],
) -> tuple[
    int,
    int,
]:
    """
    读取一个 JSON 文件。

    返回：

        (
            segment 总数,
            跳过的不完整 segment 数
        )

    重要：

    必须先验证整个 segment，
    再真正写入 frequency / coactivation。

    防止：

        Layer 3~58 已经统计
        ↓
        Layer 59 才发现错误
        ↓
        前面的统计已经无法撤回
    """

    # ========================================================
    # 读取 JSON
    # ========================================================

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

        raise TraceProfileError(
            f"无法读取 JSON：{path}"
        ) from exc

    if not isinstance(
        data,
        list,
    ):
        raise TraceProfileError(
            f"{path}: "
            "JSON 最外层必须是 list。"
        )

    if not data:
        raise TraceProfileError(
            f"{path}: "
            "JSON 最外层为空。"
        )

    skipped_segment_count = 0

    # ========================================================
    # 遍历 Segment
    # ========================================================

    for (
        segment_index,
        segment,
    ) in enumerate(
        data
    ):

        if not isinstance(
            segment,
            dict,
        ):
            raise TraceProfileError(
                f"{path}: "
                f"segment-{segment_index} "
                "必须是 dict。"
            )

        # ====================================================
        # 1. 先检查完整性
        # ====================================================

        raw_routes_by_layer = (
            collect_complete_segment_routes(
                segment=segment
            )
        )

        if (
            raw_routes_by_layer
            is None
        ):

            skipped_segment_count += 1

            continue

        # ====================================================
        # 2. 先验证整个 segment 的所有 route
        #
        # 暂时仍然不写统计。
        # ====================================================

        validated_routes_by_layer: list[
            tuple[
                int,
                list[
                    tuple[int, ...]
                ],
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

                route = validate_route(
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

                validated_routes.append(
                    route
                )

            validated_routes_by_layer.append(
                (
                    trace_layer_id,
                    validated_routes,
                )
            )

        # ====================================================
        # 3. 整个 segment 都合法，
        #    现在才真正统计
        # ====================================================

        for (
            trace_layer_id,
            routes,
        ) in validated_routes_by_layer:

            layer_id = (
                trace_layer_to_project_layer(
                    trace_layer_id
                )
            )

            for route in routes:

                accumulate_route(
                    layer_id=layer_id,
                    route=route,
                    frequency=frequency,
                    coactivation=coactivation,
                )

                token_count_by_layer[
                    layer_id
                ] += 1

    return (
        len(data),
        skipped_segment_count,
    )


# ============================================================
# 加载整个 Chinese-SimpleQA
# ============================================================


def load_chinese_simpleqa_profile(
    trace_root: Path | str = (
        DEFAULT_TRACE_ROOT
    ),
    *,
    max_files: int | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> TraceProfile:
    """
    递归读取整个 Chinese-SimpleQA。

    max_files：

        None：
            全部读取

        10：
            只读取排序后的前 10 个 JSON

    推荐：

        第一次：
            --max-files 10

        确认正常后：
            全部运行
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
            raise TraceProfileError(
                "max_files 必须大于 0。"
            )

        files = files[
            :max_files
        ]

    # ========================================================
    # Frequency
    #
    # 58 × 256
    # ========================================================

    frequency: list[
        list[int]
    ] = [
        [
            0
            for _ in range(
                NUM_ROUTED_EXPERTS
            )
        ]
        for _ in range(
            NUM_MOE_LAYERS
        )
    ]

    # ========================================================
    # Coactivation
    #
    # 58 × 256 × 256
    #
    # 用 array("Q") 避免大量 Python int 对象。
    # ========================================================

    pair_matrix_size = (
        NUM_ROUTED_EXPERTS
        * NUM_ROUTED_EXPERTS
    )

    coactivation: list[
        array
    ] = [
        array(
            "Q",
            [0],
        )
        * pair_matrix_size
        for _ in range(
            NUM_MOE_LAYERS
        )
    ]

    # ========================================================
    # Token 数量
    # ========================================================

    token_count_by_layer = [
        0
        for _ in range(
            NUM_MOE_LAYERS
        )
    ]

    # ========================================================
    # 文件类别
    # ========================================================

    category_file_counts: dict[
        str,
        int
    ] = {}

    trace_segment_count = 0

    skipped_segment_count = 0

    total_files = len(
        files
    )

    # ========================================================
    # 遍历 JSON
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

        # Chinese-SimpleQA/category/file.json
        if len(
            relative.parts
        ) >= 2:

            category = (
                relative.parts[0]
            )

        else:

            category = "__root__"

        category_file_counts[
            category
        ] = (
            category_file_counts.get(
                category,
                0,
            )
            + 1
        )

        (
            segment_count,
            skipped_count,
        ) = process_trace_file(
            path=path,
            frequency=frequency,
            coactivation=coactivation,
            token_count_by_layer=(
                token_count_by_layer
            ),
        )

        trace_segment_count += (
            segment_count
        )

        skipped_segment_count += (
            skipped_count
        )

        # ====================================================
        # 适量打印进度
        # ====================================================

        if verbose and (
            file_index == 1
            or file_index == total_files
            or file_index % 100 == 0
        ):

            print(
                f"[Trace] "
                f"{file_index}/{total_files} "
                f"{relative}"
            )

    # ========================================================
    # 冻结 Frequency
    # ========================================================

    frozen_frequency = tuple(
        tuple(
            layer
        )
        for layer in frequency
    )

    # ========================================================
    # 构造 Profile
    # ========================================================

    profile = TraceProfile(
        file_count=(
            total_files
        ),

        trace_segment_count=(
            trace_segment_count
        ),

        skipped_segment_count=(
            skipped_segment_count
        ),

        category_file_counts=dict(
            sorted(
                category_file_counts.items()
            )
        ),

        frequency=(
            frozen_frequency
        ),

        coactivation=tuple(
            coactivation
        ),

        token_count_by_layer=tuple(
            token_count_by_layer
        ),
    )

    # ========================================================
    # 最终验证
    # ========================================================

    validate_profile(
        profile,
        strict=strict,
    )

    return profile


# ============================================================
# Profile 最终检查
# ============================================================


def validate_profile(
    profile: TraceProfile,
    *,
    strict: bool = True,
) -> None:
    """
    检查最终统计结果。
    """

    # ========================================================
    # 1. Layer 数量
    # ========================================================

    if (
        len(
            profile.frequency
        )
        != NUM_MOE_LAYERS
    ):
        raise TraceProfileError(
            "frequency Layer 数错误。"
        )

    if (
        len(
            profile.coactivation
        )
        != NUM_MOE_LAYERS
    ):
        raise TraceProfileError(
            "coactivation Layer 数错误。"
        )

    if (
        len(
            profile.token_count_by_layer
        )
        != NUM_MOE_LAYERS
    ):
        raise TraceProfileError(
            "token_count_by_layer "
            "Layer 数错误。"
        )

    # ========================================================
    # 2. Frequency shape
    # ========================================================

    for (
        layer_id,
        layer_frequency,
    ) in enumerate(
        profile.frequency
    ):

        if (
            len(
                layer_frequency
            )
            != NUM_ROUTED_EXPERTS
        ):
            raise TraceProfileError(
                f"Layer-{layer_id} "
                "frequency Expert 数错误。"
            )

    # ========================================================
    # 3. Coactivation shape
    # ========================================================

    expected_pair_size = (
        NUM_ROUTED_EXPERTS
        * NUM_ROUTED_EXPERTS
    )

    for (
        layer_id,
        layer_pairs,
    ) in enumerate(
        profile.coactivation
    ):

        if (
            len(
                layer_pairs
            )
            != expected_pair_size
        ):
            raise TraceProfileError(
                f"Layer-{layer_id} "
                "coactivation shape 错误。"
            )

    # ========================================================
    # 4. 每层 selection 数
    #
    # 每个 token 必须贡献 8 次选择。
    # ========================================================

    for layer_id in range(
        NUM_MOE_LAYERS
    ):

        actual = sum(
            profile.frequency[
                layer_id
            ]
        )

        expected = (
            profile.token_count_by_layer[
                layer_id
            ]
            * EXPERTS_PER_TOKEN
        )

        if actual != expected:
            raise TraceProfileError(
                f"Layer-{layer_id} "
                "Expert 选择总数错误："
                f"actual={actual}, "
                f"expected={expected}。"
            )

    # ========================================================
    # 5. 严格一致性检查
    # ========================================================

    if strict:

        unique_token_counts = set(
            profile.token_count_by_layer
        )

        if (
            len(
                unique_token_counts
            )
            != 1
        ):

            descriptions = (
                ", ".join(
                    (
                        f"L{layer_id}="
                        f"{count}"
                    )
                    for (
                        layer_id,
                        count,
                    ) in enumerate(
                        profile.token_count_by_layer
                    )
                )
            )

            raise TraceProfileError(
                "58 个 MoE Layer 的 token "
                "路由数量不一致："
                + descriptions
            )

        if (
            profile.tokens_per_layer
            == 0
        ):
            raise TraceProfileError(
                "没有读取到任何有效 token 路由。"
            )

        expected_total_selections = (
            profile.tokens_per_layer
            * NUM_MOE_LAYERS
            * EXPERTS_PER_TOKEN
        )

        if (
            profile.total_expert_selections
            != expected_total_selections
        ):
            raise TraceProfileError(
                "总 Routed Expert Selections "
                "不一致："
                f"actual="
                f"{profile.total_expert_selections}, "
                f"expected="
                f"{expected_total_selections}。"
            )


# ============================================================
# 打印统计
# ============================================================


def print_profile_summary(
    profile: TraceProfile,
    *,
    top_k: int = 5,
    show_layers: Iterable[
        int
    ] = (
        0,
        1,
        28,
        57,
    ),
) -> None:
    """
    打印 Trace 统计摘要。
    """

    print(
        "\n"
        "========== Chinese-SimpleQA Trace Profile =========="
    )

    print(
        f"JSON 文件数："
        f"{profile.file_count}"
    )

    print(
        f"Trace segment 总数："
        f"{profile.trace_segment_count}"
    )

    print(
        f"有效 Segment："
        f"{profile.valid_segment_count}"
    )

    print(
        f"跳过的不完整 Segment："
        f"{profile.skipped_segment_count}"
    )

    print(
        f"MoE Layers："
        f"{NUM_MOE_LAYERS}"
    )

    print(
        f"Routed Experts / Layer："
        f"{NUM_ROUTED_EXPERTS}"
    )

    print(
        f"Experts / Token："
        f"{EXPERTS_PER_TOKEN}"
    )

    print(
        f"Tokens / Layer："
        f"{profile.tokens_per_layer}"
    )

    print(
        f"总 Layer-Token Events："
        f"{profile.total_layer_token_events}"
    )

    print(
        f"总 Routed Expert Selections："
        f"{profile.total_expert_selections}"
    )

    # ========================================================
    # 类别
    # ========================================================

    print(
        "\n类别文件数量："
    )

    for (
        category,
        count,
    ) in (
        profile
        .category_file_counts
        .items()
    ):

        print(
            f"  {category}: "
            f"{count}"
        )

    # ========================================================
    # 部分层热门 Expert
    # ========================================================

    print(
        "\n部分 Layer 热门 Expert："
    )

    for layer_id in show_layers:

        profile.validate_layer_id(
            layer_id
        )

        top = profile.top_experts(
            layer_id=layer_id,
            k=top_k,
        )

        description = (
            ", ".join(
                (
                    f"E{expert_id}="
                    f"{count}"
                )
                for (
                    expert_id,
                    count,
                ) in top
            )
        )

        trace_layer = (
            project_layer_to_trace_layer(
                layer_id
            )
        )

        print(
            f"  Project Layer-{layer_id} "
            f"(Trace Layer-{trace_layer}): "
            f"{description}"
        )


# ============================================================
# 命令行入口
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "统计 Chinese-SimpleQA "
                "DeepSeek-R1 Expert Trace"
            )
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
        help=(
            "Chinese-SimpleQA 根目录。"
        ),
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help=(
            "只读取前 N 个 JSON，"
            "用于快速测试。"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "打印示例 Layer 中 "
            "访问最频繁的前 K 个 Expert。"
        ),
    )

    args = (
        parser.parse_args()
    )

    print(
        "Trace Root："
        f"{args.root.resolve()}"
    )

    profile = (
        load_chinese_simpleqa_profile(
            trace_root=args.root,
            max_files=args.max_files,
            strict=True,
            verbose=True,
        )
    )

    print_profile_summary(
        profile,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()