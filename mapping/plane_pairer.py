"""
第四步：真实矩阵 -> LogicalPlane 配对。

当前固定空间结构：

    H = 7168
    W = 4096

每个物理 Plane 中正好有两个：

    7168 × 2048

槽位。

完整模型：

    58 个 MoE Layer

每层：

    256 Routed Expert
    1 Shared Expert

每个 Expert：

    gate
    up
    down

总矩阵数量：

    58 × 257 × 3
    = 44718

因此最终必须形成：

    44718 / 2
    = 22359

个 LogicalPlane。

------------------------------------------------------------

当前 Plane 配对策略：

1. gate + down

   对每个 Expert 固定：

       [ gate(e) | down(e) ]

   一共：

       58 × 257
       = 14906 Plane

2. Routed up + Routed up

   每层有：

       256 个 Routed up

   根据 Chinese-SimpleQA trace 中的：

       Expert pair coactivation

   进行低共激活配对。

   每层：

       256 / 2 = 128 Plane

   全模型：

       58 × 128
       = 7424 Plane

3. Shared up + Shared up

   每层剩：

       1 个 Shared up

   共：

       58 个

   当前 Baseline 使用确定性的跨层配对：

       Layer-0 + Layer-1
       Layer-2 + Layer-3
       ...
       Layer-56 + Layer-57

   得到：

       29 Plane

最终：

    14906
    + 7424
    + 29
    = 22359 Plane

------------------------------------------------------------

注意：

本文件仍然不决定：

    physical_plane_id
    subcube_id
    z
    slot_id
    x
    y
    rotated

这些属于下一阶段：

    LogicalPlane -> Sub-Cube
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from mapping.logical_plane import (
    LogicalPlane,
    create_gate_down_plane,
    create_up_up_plane,
    validate_logical_planes,
)

from mapping.logical_weight import (
    LogicalWeightCube,
    MATRIX_DOWN,
    MATRIX_GATE,
    MATRIX_UP,
)

from mapping.trace_profile import (
    NUM_MOE_LAYERS,
    NUM_ROUTED_EXPERTS,
    TraceProfile,
)


# ============================================================
# 异常
# ============================================================


class PlanePairingError(ValueError):
    """LogicalPlane 配对失败。"""


# ============================================================
# Pairing 模式
# ============================================================


PAIRING_MODE_SEQUENTIAL = "sequential"
PAIRING_MODE_RANDOM = "random"
PAIRING_MODE_FREQUENCY_AWARE = "frequency_aware"
PAIRING_MODE_GREEDY = "greedy"
PAIRING_MODE_TRACE_AWARE = "trace_aware"
PAIRING_MODE_OPTIMAL = "optimal"

PAIRING_MODES = (
    PAIRING_MODE_SEQUENTIAL,
    PAIRING_MODE_RANDOM,
    PAIRING_MODE_FREQUENCY_AWARE,
    PAIRING_MODE_GREEDY,
    PAIRING_MODE_TRACE_AWARE,
    PAIRING_MODE_OPTIMAL,
)

DEFAULT_PAIRING_RANDOM_SEED = 42


# ============================================================
# 类型
# ============================================================


ExpertMatrixMap = dict[
    str,
    LogicalWeightCube,
]

ExpertIndex = dict[
    tuple[int, int],
    ExpertMatrixMap,
]


# ============================================================
# PairingResult
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PairingResult:
    """
    完整 LogicalPlane 配对结果。

    planes：

        全部 22359 个 LogicalPlane。

    routed_up_pairs_by_layer：

        每层 128 对 Routed Expert：

            (
                (expert_a, expert_b),
                ...
            )

    routed_up_coactivation_cost_by_layer：

        每层所有 up-up Plane 的
        共激活次数总和。

        越小越好。

    shared_up_layer_pairs：

        Shared up 的跨层配对：

            (0,1)
            (2,3)
            ...
    """

    planes: tuple[
        LogicalPlane,
        ...
    ]

    routed_up_pairs_by_layer: tuple[
        tuple[
            tuple[int, int],
            ...
        ],
        ...
    ]

    routed_up_coactivation_cost_by_layer: tuple[
        int,
        ...
    ]

    shared_up_layer_pairs: tuple[
        tuple[int, int],
        ...
    ]

    # ========================================================
    # 统计
    # ========================================================

    @property
    def total_planes(
        self,
    ) -> int:

        return len(
            self.planes
        )

    @property
    def gate_down_plane_count(
        self,
    ) -> int:

        return sum(
            1
            for plane
            in self.planes
            if plane.is_gate_down
        )

    @property
    def up_up_plane_count(
        self,
    ) -> int:

        return sum(
            1
            for plane
            in self.planes
            if plane.is_up_up
        )

    @property
    def routed_up_plane_count(
        self,
    ) -> int:

        return sum(
            len(layer_pairs)
            for layer_pairs
            in self.routed_up_pairs_by_layer
        )

    @property
    def shared_up_plane_count(
        self,
    ) -> int:

        return len(
            self.shared_up_layer_pairs
        )

    @property
    def total_routed_up_coactivation_cost(
        self,
    ) -> int:
        """
        如果这些 up-up Plane 真正共址，
        在整个 Chinese-SimpleQA trace 上，
        总共有多少次这两个 Expert 同时被激活。

        越低越好。
        """

        return sum(
            self
            .routed_up_coactivation_cost_by_layer
        )


# ============================================================
# Expert 索引
# ============================================================


def build_expert_index(
    cubes: Iterable[
        LogicalWeightCube
    ],
) -> ExpertIndex:
    """
    构造：

        (layer_id, expert_id)
            ->
        {
            gate_proj: cube,
            up_proj: cube,
            down_proj: cube
        }

    不依赖 cube_id 的排列顺序。
    """

    cube_list = tuple(
        cubes
    )

    if not cube_list:

        raise PlanePairingError(
            "LogicalWeightCube 集合不能为空。"
        )

    index: ExpertIndex = {}

    for cube in cube_list:

        expert_key = (
            cube.expert_key
        )

        matrices = (
            index.setdefault(
                expert_key,
                {},
            )
        )

        if (
            cube.matrix_name
            in matrices
        ):

            raise PlanePairingError(
                f"Expert {expert_key} "
                f"存在重复矩阵 "
                f"{cube.matrix_name}。"
            )

        matrices[
            cube.matrix_name
        ] = cube

    # ========================================================
    # 检查每个 Expert 都必须有：
    #
    # gate / up / down
    # ========================================================

    expected_names = {
        MATRIX_GATE,
        MATRIX_UP,
        MATRIX_DOWN,
    }

    for (
        expert_key,
        matrices,
    ) in index.items():

        if (
            set(matrices)
            != expected_names
        ):

            raise PlanePairingError(
                f"Expert {expert_key} "
                "矩阵不完整："
                f"{sorted(matrices)}。"
            )

        cubes_of_expert = tuple(
            matrices.values()
        )

        if (
            len(
                {
                    cube.is_shared
                    for cube
                    in cubes_of_expert
                }
            )
            != 1
        ):

            raise PlanePairingError(
                f"Expert {expert_key} "
                "三个矩阵的 is_shared "
                "属性不一致。"
            )

    return index


# ============================================================
# Trace 快速查询
# ============================================================


def _coactivation_cost(
    profile: TraceProfile,
    layer_id: int,
    expert_a: int,
    expert_b: int,
) -> int:
    """
    G[layer][a][b]

    为了 Plane 配对速度，
    直接访问 TraceProfile 内部 array。
    """

    if expert_a == expert_b:

        raise PlanePairingError(
            "同一个 Routed Expert "
            "不能与自己形成 up-up Plane。"
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
        profile.coactivation[
            layer_id
        ][
            index
        ]
    )


def _frequency(
    profile: TraceProfile,
    layer_id: int,
    expert_id: int,
) -> int:
    """
    f[layer][expert]
    """

    return int(
        profile.frequency[
            layer_id
        ][
            expert_id
        ]
    )


# ============================================================
# Routed up 顺序配对（Naive / Ablation Baseline）
# ============================================================


def sequential_pair_routed_up_experts(
    *,
    layer_id: int,
) -> tuple[
    tuple[int, int],
    ...
]:
    """
    对某一层的 256 个 Routed up 做固定顺序配对。

    配对方式：

        E0  + E1
        E2  + E3
        ...
        E254 + E255

    该策略完全不读取 frequency / coactivation，
    用作 Trace-aware Pairing 的确定性消融 Baseline。

    注意：

    后续仍然可以用真实 Trace 计算这组配对的
    coactivation cost，作为事后评价指标；
    但 Trace 不参与配对决策本身。
    """

    if not (
        0
        <= layer_id
        < NUM_MOE_LAYERS
    ):

        raise PlanePairingError(
            "layer_id 必须位于 "
            f"[0,{NUM_MOE_LAYERS - 1}]。"
        )

    if (
        NUM_ROUTED_EXPERTS
        % 2
        != 0
    ):

        raise PlanePairingError(
            "Sequential Pairing 要求 "
            "Routed Expert 数量为偶数。"
        )

    return tuple(
        (
            expert_id,
            expert_id + 1,
        )
        for expert_id
        in range(
            0,
            NUM_ROUTED_EXPERTS,
            2,
        )
    )


# ============================================================
# Routed up 随机配对（Random Baseline）
# ============================================================


def random_pair_routed_up_experts(
    *,
    layer_id: int,
    seed: int = DEFAULT_PAIRING_RANDOM_SEED,
) -> tuple[
    tuple[int, int],
    ...
]:
    """
    对某一层 256 个 Routed up 做可复现随机配对。

    Random Baseline 不读取 frequency / coactivation。
    为保证实验可复现，每层使用：

        effective_seed = seed + layer_id

    相同 seed 下结果必须完全一致。
    """

    if not (
        0
        <= layer_id
        < NUM_MOE_LAYERS
    ):
        raise PlanePairingError(
            "layer_id 必须位于 "
            f"[0,{NUM_MOE_LAYERS - 1}]。"
        )

    if NUM_ROUTED_EXPERTS % 2 != 0:
        raise PlanePairingError(
            "Random Pairing 要求 Routed Expert 数量为偶数。"
        )

    rng = random.Random(
        int(seed) + layer_id
    )

    experts = list(
        range(NUM_ROUTED_EXPERTS)
    )

    rng.shuffle(experts)

    pairs = [
        (
            min(experts[i], experts[i + 1]),
            max(experts[i], experts[i + 1]),
        )
        for i in range(
            0,
            NUM_ROUTED_EXPERTS,
            2,
        )
    ]

    return tuple(pairs)


# ============================================================
# Routed up Frequency-aware 配对
# ============================================================


def frequency_aware_pair_routed_up_experts(
    *,
    layer_id: int,
    profile: TraceProfile,
) -> tuple[
    tuple[int, int],
    ...
]:
    """
    只使用 Expert 访问频率，不使用共激活矩阵。

    规则：

        hottest + coldest
        second hottest + second coldest
        ...

    目的：
    将高频 Expert 与低频 Expert 组成同一 up-up Plane，
    作为“仅使用热度信息”的 Pairing baseline。

    该方法不读取 G(a,b)，因此可用于区分：

        仅知道 Expert 热度
            vs
        真正使用 Expert 共激活关系

    的差别。
    """

    if not (
        0
        <= layer_id
        < NUM_MOE_LAYERS
    ):
        raise PlanePairingError(
            "layer_id 必须位于 "
            f"[0,{NUM_MOE_LAYERS - 1}]。"
        )

    ranked = sorted(
        range(NUM_ROUTED_EXPERTS),
        key=lambda expert_id: (
            -_frequency(
                profile,
                layer_id,
                expert_id,
            ),
            expert_id,
        ),
    )

    pairs: list[
        tuple[int, int]
    ] = []

    left = 0
    right = len(ranked) - 1

    while left < right:
        a = ranked[left]
        b = ranked[right]

        pairs.append(
            (
                min(a, b),
                max(a, b),
            )
        )

        left += 1
        right -= 1

    if len(pairs) * 2 != NUM_ROUTED_EXPERTS:
        raise PlanePairingError(
            f"Layer-{layer_id} Frequency-aware Pairing 数量错误。"
        )

    return tuple(pairs)


# ============================================================
# Routed up Minimum-Weight Perfect Matching
# ============================================================


def optimal_pair_routed_up_experts(
    *,
    layer_id: int,
    profile: TraceProfile,
) -> tuple[
    tuple[int, int],
    ...
]:
    """
    求解当前层 Pairing Cost 的全局最优完美匹配：

        min sum G(a,b)

    每个 Routed Expert 恰好出现一次。

    优先使用 SciPy/HiGHS MILP（通常明显快于纯 Python
    Blossom）；如果当前环境没有 scipy.optimize.milp，
    再回退到 NetworkX minimum-weight matching。

    注意：这里最优的是 Pairing Cost 中间目标，
    不等价于最终 Prefill / Decode latency 的全局最优。
    """

    if not (
        0
        <= layer_id
        < NUM_MOE_LAYERS
    ):
        raise PlanePairingError(
            "layer_id 必须位于 "
            f"[0,{NUM_MOE_LAYERS - 1}]。"
        )

    # --------------------------------------------------------
    # 构造无向完全图边表
    # --------------------------------------------------------

    edges: list[
        tuple[int, int]
    ] = []

    primary_costs: list[int] = []

    for expert_a in range(
        NUM_ROUTED_EXPERTS - 1
    ):
        for expert_b in range(
            expert_a + 1,
            NUM_ROUTED_EXPERTS,
        ):
            edges.append(
                (
                    expert_a,
                    expert_b,
                )
            )

            primary_costs.append(
                _coactivation_cost(
                    profile,
                    layer_id,
                    expert_a,
                    expert_b,
                )
            )

    # 128 条匹配边的 tie-break 总和最大约 8.4M。
    # 10M 可以保证 coactivation cost 永远是主目标。
    primary_scale = 10_000_000

    objective = [
        (
            primary * primary_scale
            + expert_a * NUM_ROUTED_EXPERTS
            + expert_b
        )
        for (
            primary,
            (expert_a, expert_b),
        ) in zip(
            primary_costs,
            edges,
        )
    ]

    # --------------------------------------------------------
    # 首选：SciPy MILP / HiGHS
    # --------------------------------------------------------

    pairs: tuple[
        tuple[int, int],
        ...
    ] | None = None

    try:
        import numpy as np
        from scipy.optimize import (
            Bounds,
            LinearConstraint,
            milp,
        )
        from scipy.sparse import coo_matrix

        row_indices: list[int] = []
        col_indices: list[int] = []
        values: list[int] = []

        for edge_id, (
            expert_a,
            expert_b,
        ) in enumerate(edges):
            row_indices.extend(
                (
                    expert_a,
                    expert_b,
                )
            )
            col_indices.extend(
                (
                    edge_id,
                    edge_id,
                )
            )
            values.extend((1, 1))

        incidence = coo_matrix(
            (
                values,
                (
                    row_indices,
                    col_indices,
                ),
            ),
            shape=(
                NUM_ROUTED_EXPERTS,
                len(edges),
            ),
        ).tocsr()

        ones = np.ones(
            NUM_ROUTED_EXPERTS,
            dtype=float,
        )

        result = milp(
            c=np.asarray(
                objective,
                dtype=float,
            ),
            integrality=np.ones(
                len(edges),
                dtype=np.int8,
            ),
            bounds=Bounds(
                np.zeros(len(edges)),
                np.ones(len(edges)),
            ),
            constraints=LinearConstraint(
                incidence,
                ones,
                ones,
            ),
            options={
                "presolve": True,
            },
        )

        if (
            result.success
            and result.x is not None
        ):
            selected = [
                edges[edge_id]
                for edge_id, value
                in enumerate(result.x)
                if value > 0.5
            ]

            pairs = tuple(
                sorted(selected)
            )

    except (
        ImportError,
        AttributeError,
    ):
        pairs = None

    # --------------------------------------------------------
    # 回退：NetworkX Blossom
    # --------------------------------------------------------

    if pairs is None:
        try:
            import networkx as nx
        except ImportError as exc:
            raise PlanePairingError(
                "optimal Pairing 需要 scipy>=1.9 或 networkx。"
                "建议先执行：pip install scipy networkx"
            ) from exc

        graph = nx.Graph()
        graph.add_nodes_from(
            range(NUM_ROUTED_EXPERTS)
        )

        for (
            (expert_a, expert_b),
            weight,
        ) in zip(
            edges,
            objective,
        ):
            graph.add_edge(
                expert_a,
                expert_b,
                weight=weight,
            )

        matching = (
            nx.algorithms.matching
            .min_weight_matching(
                graph,
                weight="weight",
            )
        )

        pairs = tuple(
            sorted(
                (
                    min(int(a), int(b)),
                    max(int(a), int(b)),
                )
                for a, b in matching
            )
        )

    # --------------------------------------------------------
    # 完美匹配检查
    # --------------------------------------------------------

    if len(pairs) != NUM_ROUTED_EXPERTS // 2:
        raise PlanePairingError(
            f"Layer-{layer_id} Optimal Pairing 未得到完美匹配："
            f"pairs={len(pairs)}。"
        )

    flat = [
        expert_id
        for pair in pairs
        for expert_id in pair
    ]

    if (
        len(flat) != NUM_ROUTED_EXPERTS
        or len(set(flat)) != NUM_ROUTED_EXPERTS
    ):
        raise PlanePairingError(
            f"Layer-{layer_id} Optimal Pairing Expert 覆盖错误。"
        )

    return pairs


# ============================================================
# Routed up 初始贪心配对
# ============================================================


def greedy_pair_routed_up_experts(
    *,
    layer_id: int,
    profile: TraceProfile,
) -> tuple[
    tuple[int, int],
    ...
]:
    """
    对某一层的 256 个 Routed up 做初始配对。

    --------------------------------------------------------

    核心原则：

    两个 Expert 如果频繁同时被 Router 选中，
    它们的 up 不应该共享一个 Plane。

    因为：

        同 Plane
            =>
        同 Sub-Cube

    如果某 token 同时激活 e1/e2：

        up(e1)
        up(e2)

    就会争抢同一个 Sub-Cube。

    --------------------------------------------------------

    Greedy：

    1. Expert 按访问频率从高到低处理；

    2. 对当前 Expert a，
       从尚未配对 Expert 中选择 b：

       第一：
           G(a,b) 最小

       第二：
           如果 G 相同，
           frequency(b) 更高优先

       第三：
           expert_id 更小优先

    --------------------------------------------------------

    为什么高频 Expert 先处理？

    高频 Expert 对最终推理周期影响更大，
    应优先获得更好的低冲突伙伴。

    --------------------------------------------------------

    为什么 G 相同时优先选另一个高频 Expert？

    如果两个高频 Expert 在整个 trace 中
    实际从不同时出现：

        G = 0

    那么把它们放一起并不会产生
    同 token 并行冲突。

    同时也可以避免把另一个高频 Expert
    留到最后被迫接受较差配对。
    """

    if not (
        0
        <= layer_id
        < NUM_MOE_LAYERS
    ):

        raise PlanePairingError(
            "layer_id 必须位于 "
            f"[0,{NUM_MOE_LAYERS - 1}]。"
        )

    remaining = set(
        range(
            NUM_ROUTED_EXPERTS
        )
    )

    pairs: list[
        tuple[int, int]
    ] = []

    # ========================================================
    # 高频 Expert 优先
    # ========================================================

    hot_order = sorted(
        remaining,
        key=lambda expert_id: (
            -_frequency(
                profile,
                layer_id,
                expert_id,
            ),
            expert_id,
        ),
    )

    # ========================================================
    # Greedy
    # ========================================================

    for expert_a in hot_order:

        if (
            expert_a
            not in remaining
        ):
            continue

        remaining.remove(
            expert_a
        )

        if not remaining:

            raise PlanePairingError(
                f"Layer-{layer_id} "
                "Routed up 数量不是偶数。"
            )

        expert_b = min(
            remaining,
            key=lambda candidate: (
                # 1. 共激活越低越好
                _coactivation_cost(
                    profile,
                    layer_id,
                    expert_a,
                    candidate,
                ),

                # 2. 如果共激活相同，
                #    热 Expert 优先处理
                -_frequency(
                    profile,
                    layer_id,
                    candidate,
                ),

                # 3. 确定性 tie-break
                candidate,
            ),
        )

        remaining.remove(
            expert_b
        )

        pairs.append(
            (
                min(
                    expert_a,
                    expert_b,
                ),
                max(
                    expert_a,
                    expert_b,
                ),
            )
        )

    # ========================================================
    # 检查
    # ========================================================

    if remaining:

        raise PlanePairingError(
            f"Layer-{layer_id} "
            "配对结束仍有未使用 Expert。"
        )

    if (
        len(pairs) * 2
        != NUM_ROUTED_EXPERTS
    ):

        raise PlanePairingError(
            f"Layer-{layer_id} "
            "Routed up 配对数量错误。"
        )

    return tuple(
        pairs
    )


# ============================================================
# Routed up 局部交换优化
# ============================================================


def improve_routed_up_pairs(
    *,
    layer_id: int,
    pairs: Iterable[
        tuple[int, int]
    ],
    profile: TraceProfile,
    max_rounds: int = 4,
) -> tuple[
    tuple[int, int],
    ...
]:
    """
    对 Greedy 结果做局部交换优化。

    --------------------------------------------------------

    假设现在有两对：

        (a,b)
        (c,d)

    当前代价：

        G(a,b) + G(c,d)

    尝试另外两种重组：

        (a,c) + (b,d)

    或：

        (a,d) + (b,c)

    如果任意一种总共激活次数更低，
    就接受交换。

    --------------------------------------------------------

    这解决 Greedy 的一个问题：

    前面的 Expert 配得很好，
    可能导致后面的 Expert
    被迫形成很高冲突的 pair。

    局部交换可以在不引入额外依赖库的情况下
    对结果进一步优化。
    """

    if max_rounds < 0:

        raise PlanePairingError(
            "max_rounds 不能为负数。"
        )

    current = [
        (
            min(a, b),
            max(a, b),
        )
        for a, b
        in pairs
    ]

    # ========================================================
    # 输入必须是 256 Expert 的完美配对
    # ========================================================

    flat = [
        expert_id
        for pair in current
        for expert_id in pair
    ]

    if (
        len(flat)
        != NUM_ROUTED_EXPERTS
    ):

        raise PlanePairingError(
            f"Layer-{layer_id} "
            "pair 中 Expert 数量错误。"
        )

    if (
        len(
            set(flat)
        )
        != NUM_ROUTED_EXPERTS
    ):

        raise PlanePairingError(
            f"Layer-{layer_id} "
            "pair 中存在重复 Expert。"
        )

    if (
        set(flat)
        != set(
            range(
                NUM_ROUTED_EXPERTS
            )
        )
    ):

        raise PlanePairingError(
            f"Layer-{layer_id} "
            "没有完整覆盖 Expert 0~255。"
        )

    # ========================================================
    # Local Search
    # ========================================================

    for _ in range(
        max_rounds
    ):

        improved = False

        for i in range(
            len(current) - 1
        ):

            a, b = (
                current[i]
            )

            for j in range(
                i + 1,
                len(current),
            ):

                c, d = (
                    current[j]
                )

                # ============================================
                # 当前方案
                # ============================================

                current_cost = (
                    _coactivation_cost(
                        profile,
                        layer_id,
                        a,
                        b,
                    )
                    +
                    _coactivation_cost(
                        profile,
                        layer_id,
                        c,
                        d,
                    )
                )

                # ============================================
                # 交换方案 1：
                #
                # (a,c)
                # (b,d)
                # ============================================

                option_1 = (
                    _coactivation_cost(
                        profile,
                        layer_id,
                        a,
                        c,
                    )
                    +
                    _coactivation_cost(
                        profile,
                        layer_id,
                        b,
                        d,
                    )
                )

                # ============================================
                # 交换方案 2：
                #
                # (a,d)
                # (b,c)
                # ============================================

                option_2 = (
                    _coactivation_cost(
                        profile,
                        layer_id,
                        a,
                        d,
                    )
                    +
                    _coactivation_cost(
                        profile,
                        layer_id,
                        b,
                        c,
                    )
                )

                # 没有提升
                if (
                    option_1
                    >= current_cost
                    and
                    option_2
                    >= current_cost
                ):

                    continue

                # ============================================
                # 选择更好的方案
                # ============================================

                if (
                    option_1
                    < option_2
                ):

                    new_first = (
                        min(a, c),
                        max(a, c),
                    )

                    new_second = (
                        min(b, d),
                        max(b, d),
                    )

                elif (
                    option_2
                    < option_1
                ):

                    new_first = (
                        min(a, d),
                        max(a, d),
                    )

                    new_second = (
                        min(b, c),
                        max(b, c),
                    )

                else:
                    # 两个方案 cost 完全相同，
                    # 使用 Expert ID 顺序保证结果确定。

                    candidate_1 = tuple(
                        sorted(
                            (
                                (
                                    min(a, c),
                                    max(a, c),
                                ),
                                (
                                    min(b, d),
                                    max(b, d),
                                ),
                            )
                        )
                    )

                    candidate_2 = tuple(
                        sorted(
                            (
                                (
                                    min(a, d),
                                    max(a, d),
                                ),
                                (
                                    min(b, c),
                                    max(b, c),
                                ),
                            )
                        )
                    )

                    chosen = min(
                        candidate_1,
                        candidate_2,
                    )

                    (
                        new_first,
                        new_second,
                    ) = chosen

                # ============================================
                # 接受交换
                # ============================================

                current[i] = (
                    new_first
                )

                current[j] = (
                    new_second
                )

                improved = True

                # 当前 i 对已经改变，
                # 后续比较使用新 pair。
                a, b = (
                    current[i]
                )

        if not improved:

            break

    return tuple(
        current
    )


# ============================================================
# Pairing Cost
# ============================================================


def routed_up_pairing_cost(
    *,
    layer_id: int,
    pairs: Iterable[
        tuple[int, int]
    ],
    profile: TraceProfile,
) -> int:
    """
    计算一层 Routed up Plane 的总共激活代价。

    Cost：

        Σ G(e1,e2)

    越低越好。
    """

    return sum(
        _coactivation_cost(
            profile,
            layer_id,
            expert_a,
            expert_b,
        )
        for (
            expert_a,
            expert_b,
        ) in pairs
    )


# ============================================================
# 完整 LogicalPlane 构造
# ============================================================


def build_logical_planes(
    *,
    cubes: Iterable[
        LogicalWeightCube
    ],
    profile: TraceProfile,
    pairing_mode: str = (
        PAIRING_MODE_TRACE_AWARE
    ),
    improve_pairs: bool = True,
    local_search_rounds: int = 4,
    random_seed: int = DEFAULT_PAIRING_RANDOM_SEED,
) -> PairingResult:
    """
    构造完整的 22359 个 LogicalPlane。

    顺序：

    1. 全部 gate+down；
    2. 每层 Routed up+up；
    3. Shared up 跨层配对。

    --------------------------------------------------------

    pairing_mode：

        sequential：固定编号顺序配对。

        random：固定 seed 的随机配对。

        frequency_aware：只使用 Expert 访问频率。

        greedy：只使用共激活感知 Greedy。

        trace_aware：Greedy + 可选 Local Search。

        optimal：Minimum-Weight Perfect Matching，
            对 Pairing Cost 给出全局最优参考。

    improve_pairs：

        仅对 trace_aware 生效。

        False：
            只使用 Greedy

        True：
            Greedy + Local Search

    后续做消融实验时可以直接比较。
    """

    if pairing_mode not in PAIRING_MODES:

        raise PlanePairingError(
            f"非法 pairing_mode={pairing_mode!r}，"
            f"允许值为 {PAIRING_MODES}。"
        )

    cube_list = tuple(
        cubes
    )

    # ========================================================
    # Trace Layer 数检查
    # ========================================================

    if (
        len(
            profile.frequency
        )
        != NUM_MOE_LAYERS
    ):

        raise PlanePairingError(
            "TraceProfile 的 MoE Layer "
            "数量不正确。"
        )

    # ========================================================
    # 建立 Expert Index
    # ========================================================

    expert_index = (
        build_expert_index(
            cube_list
        )
    )

    expected_experts = (
        NUM_MOE_LAYERS
        * (
            NUM_ROUTED_EXPERTS
            + 1
        )
    )

    if (
        len(expert_index)
        != expected_experts
    ):

        raise PlanePairingError(
            "Expert 总数错误："
            f"actual="
            f"{len(expert_index)}, "
            f"expected="
            f"{expected_experts}。"
        )

    # ========================================================
    # 检查每层：
    #
    # Routed = 0~255
    # Shared = 256
    # ========================================================

    for layer_id in range(
        NUM_MOE_LAYERS
    ):

        expected_keys = {
            (
                layer_id,
                expert_id,
            )
            for expert_id
            in range(
                NUM_ROUTED_EXPERTS
                + 1
            )
        }

        actual_keys = {
            key
            for key
            in expert_index
            if (
                key[0]
                == layer_id
            )
        }

        if (
            actual_keys
            != expected_keys
        ):

            raise PlanePairingError(
                f"Layer-{layer_id} "
                "Expert ID 集合不完整。"
            )

        # Routed 0~255
        for expert_id in range(
            NUM_ROUTED_EXPERTS
        ):

            matrices = (
                expert_index[
                    (
                        layer_id,
                        expert_id,
                    )
                ]
            )

            if (
                matrices[
                    MATRIX_GATE
                ].is_shared
            ):

                raise PlanePairingError(
                    f"Layer-{layer_id} "
                    f"Expert-{expert_id} "
                    "应为 Routed Expert。"
                )

        # Shared = 256
        shared = (
            expert_index[
                (
                    layer_id,
                    NUM_ROUTED_EXPERTS,
                )
            ]
        )

        if not (
            shared[
                MATRIX_GATE
            ].is_shared
        ):

            raise PlanePairingError(
                f"Layer-{layer_id} "
                f"Expert-"
                f"{NUM_ROUTED_EXPERTS} "
                "应为 Shared Expert。"
            )

    # ========================================================
    # 正式开始生成 Plane
    # ========================================================

    planes: list[
        LogicalPlane
    ] = []

    next_plane_id = 0

    # ========================================================
    # 第一部分：
    #
    # 每个 Expert：
    #
    # [ gate | down ]
    #
    # 58 × 257 = 14906 Plane
    # ========================================================

    for layer_id in range(
        NUM_MOE_LAYERS
    ):

        for expert_id in range(
            NUM_ROUTED_EXPERTS
            + 1
        ):

            matrices = (
                expert_index[
                    (
                        layer_id,
                        expert_id,
                    )
                ]
            )

            plane = (
                create_gate_down_plane(
                    logical_plane_id=(
                        next_plane_id
                    ),
                    gate=matrices[
                        MATRIX_GATE
                    ],
                    down=matrices[
                        MATRIX_DOWN
                    ],
                )
            )

            planes.append(
                plane
            )

            next_plane_id += 1

    # ========================================================
    # 第二部分：
    #
    # Routed up + Routed up
    # ========================================================

    routed_up_pairs_by_layer: list[
        tuple[
            tuple[int, int],
            ...
        ]
    ] = []

    routed_up_costs: list[
        int
    ] = []

    for layer_id in range(
        NUM_MOE_LAYERS
    ):

        # ====================================================
        # Pairing Strategy
        # ====================================================

        if (
            pairing_mode
            == PAIRING_MODE_SEQUENTIAL
        ):

            pairs = (
                sequential_pair_routed_up_experts(
                    layer_id=layer_id,
                )
            )

        elif (
            pairing_mode
            == PAIRING_MODE_RANDOM
        ):

            pairs = (
                random_pair_routed_up_experts(
                    layer_id=layer_id,
                    seed=random_seed,
                )
            )

        elif (
            pairing_mode
            == PAIRING_MODE_FREQUENCY_AWARE
        ):

            pairs = (
                frequency_aware_pair_routed_up_experts(
                    layer_id=layer_id,
                    profile=profile,
                )
            )

        elif (
            pairing_mode
            == PAIRING_MODE_OPTIMAL
        ):

            pairs = (
                optimal_pair_routed_up_experts(
                    layer_id=layer_id,
                    profile=profile,
                )
            )

        elif pairing_mode in (
            PAIRING_MODE_GREEDY,
            PAIRING_MODE_TRACE_AWARE,
        ):

            pairs = (
                greedy_pair_routed_up_experts(
                    layer_id=layer_id,
                    profile=profile,
                )
            )

            # greedy 模式明确只做初始 Greedy。
            # trace_aware 才允许继续 Local Search。
            if (
                pairing_mode
                == PAIRING_MODE_TRACE_AWARE
                and improve_pairs
            ):

                pairs = (
                    improve_routed_up_pairs(
                        layer_id=layer_id,
                        pairs=pairs,
                        profile=profile,
                        max_rounds=(
                            local_search_rounds
                        ),
                    )
                )

        else:
            raise PlanePairingError(
                f"未实现 pairing_mode={pairing_mode!r}。"
            )

        # ====================================================
        # 记录 Cost
        # ====================================================

        cost = (
            routed_up_pairing_cost(
                layer_id=layer_id,
                pairs=pairs,
                profile=profile,
            )
        )

        routed_up_pairs_by_layer.append(
            pairs
        )

        routed_up_costs.append(
            cost
        )

        # ====================================================
        # Expert Pair -> LogicalPlane
        # ====================================================

        for (
            expert_a,
            expert_b,
        ) in pairs:

            up_a = (
                expert_index[
                    (
                        layer_id,
                        expert_a,
                    )
                ][
                    MATRIX_UP
                ]
            )

            up_b = (
                expert_index[
                    (
                        layer_id,
                        expert_b,
                    )
                ][
                    MATRIX_UP
                ]
            )

            plane = (
                create_up_up_plane(
                    logical_plane_id=(
                        next_plane_id
                    ),
                    first_up=up_a,
                    second_up=up_b,
                )
            )

            planes.append(
                plane
            )

            next_plane_id += 1

    # ========================================================
    # 第三部分：
    #
    # Shared up 跨层配对
    #
    # Baseline：
    #
    # L0 + L1
    # L2 + L3
    # ...
    # L56 + L57
    # ========================================================

    shared_up_layer_pairs: list[
        tuple[int, int]
    ] = []

    if (
        NUM_MOE_LAYERS
        % 2
        != 0
    ):

        raise PlanePairingError(
            "Shared up 跨层配对要求 "
            "MoE Layer 数量为偶数。"
        )

    for first_layer in range(
        0,
        NUM_MOE_LAYERS,
        2,
    ):

        second_layer = (
            first_layer
            + 1
        )

        shared_up_a = (
            expert_index[
                (
                    first_layer,
                    NUM_ROUTED_EXPERTS,
                )
            ][
                MATRIX_UP
            ]
        )

        shared_up_b = (
            expert_index[
                (
                    second_layer,
                    NUM_ROUTED_EXPERTS,
                )
            ][
                MATRIX_UP
            ]
        )

        plane = (
            create_up_up_plane(
                logical_plane_id=(
                    next_plane_id
                ),
                first_up=(
                    shared_up_a
                ),
                second_up=(
                    shared_up_b
                ),
            )
        )

        planes.append(
            plane
        )

        shared_up_layer_pairs.append(
            (
                first_layer,
                second_layer,
            )
        )

        next_plane_id += 1

    # ========================================================
    # 全部矩阵必须恰好出现一次
    # ========================================================

    validate_logical_planes(
        planes=planes,
        cubes=cube_list,
        require_complete=True,
    )

    # ========================================================
    # 数量检查
    # ========================================================

    if (
        len(cube_list)
        % 2
        != 0
    ):

        raise PlanePairingError(
            "LogicalWeightCube "
            "总数量必须为偶数。"
        )

    expected_planes = (
        len(cube_list)
        // 2
    )

    if (
        len(planes)
        != expected_planes
    ):

        raise PlanePairingError(
            "LogicalPlane 总数错误："
            f"actual={len(planes)}, "
            f"expected={expected_planes}。"
        )

    return PairingResult(
        planes=tuple(
            planes
        ),

        routed_up_pairs_by_layer=tuple(
            routed_up_pairs_by_layer
        ),

        routed_up_coactivation_cost_by_layer=tuple(
            routed_up_costs
        ),

        shared_up_layer_pairs=tuple(
            shared_up_layer_pairs
        ),
    )


# ============================================================
# 输出统计
# ============================================================

def print_pairing_summary(
    result: PairingResult,
) -> None:
    """
    打印 LogicalPlane 配对摘要。
    """

    print(
        "\n"
        "========== Logical Plane Pairing =========="
    )

    print(
        f"Total Logical Planes："
        f"{result.total_planes}"
    )

    print(
        f"gate+down Planes："
        f"{result.gate_down_plane_count}"
    )

    print(
        f"Routed up+up Planes："
        f"{result.routed_up_plane_count}"
    )

    print(
        f"Shared up+up Planes："
        f"{result.shared_up_plane_count}"
    )

    print(
        f"All up+up Planes："
        f"{result.up_up_plane_count}"
    )

    total_cost = (
        result.total_routed_up_coactivation_cost
    )

    print(
        "Routed up Pairing "
        f"Coactivation Cost：{total_cost}"
    )

    layer_costs = (
        result
        .routed_up_coactivation_cost_by_layer
    )

    if layer_costs:

        max_cost = max(
            layer_costs
        )

        min_cost = min(
            layer_costs
        )

        avg_cost = (
            sum(layer_costs)
            / len(layer_costs)
        )

        print(
            f"Max Layer Pairing Cost："
            f"{max_cost}"
        )

        print(
            f"Min Layer Pairing Cost："
            f"{min_cost}"
        )

        print(
            f"Avg Layer Pairing Cost："
            f"{avg_cost:.4f}"
        )