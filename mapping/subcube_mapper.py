"""
第四步：LogicalPlane -> Sub-Cube 静态映射。

输入：

    1. PairingResult
       已经完成真实矩阵 -> LogicalPlane 配对；

    2. LogicalWeightCube
       用于恢复每个 Plane 内的真实 Expert 身份；

    3. TraceProfile
       Chinese-SimpleQA 的 Expert 热度与共激活统计；

    4. ResolvedHardwareConfig
       已经确定 N、H、W、D。

输出：

    每个 LogicalPlane：

        logical_plane_id
        subcube_id
        z

------------------------------------------------------------

当前核心原则：

1. 一个 LogicalPlane 整体属于一个 Sub-Cube；

2. 每个 Sub-Cube 最多放 D 个 Plane；

3. 同一个 Expert：

       gate
       up

   必须映射到不同 Sub-Cube；

4. gate + down 已经共享一个 Plane，因此：

       gate(e)
       down(e)

   必然属于同一个 Sub-Cube；

5. 利用 Chinese-SimpleQA trace：

       frequency
       coactivation

   尽量减少同时活跃的计算落入同一个 Sub-Cube；

6. 在冲突相同时：

       优先平衡单层负载；
       再平衡全局负载；
       再平衡 Plane 数量。

------------------------------------------------------------

注意：

本文件仍然没有绑定第三步的匿名 PhysicalSlot。

当前只得到：

    LogicalPlane
        ->
    (subcube_id, z)

下一阶段再完成：

    LogicalPlane
        ->
    anonymous Physical Plane
        ->
    PhysicalSlot
        ->
    LogicalWeightCube
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from evaluation.hardware_resolver import (
    ResolvedHardwareConfig,
)

from mapping.logical_plane import (
    LogicalPlane,
    build_cube_index,
    get_plane_cubes,
)

from mapping.logical_weight import (
    LogicalWeightCube,
    MATRIX_DOWN,
    MATRIX_GATE,
    MATRIX_UP,
)

from mapping.plane_pairer import (
    PairingResult,
)

from mapping.trace_profile import (
    NUM_MOE_LAYERS,
    NUM_ROUTED_EXPERTS,
    TraceProfile,
)


# ============================================================
# 常量
# ============================================================


SHARED_EXPERT_ID = (
    NUM_ROUTED_EXPERTS
)


# ============================================================
# Mapping 模式
# ============================================================


MAPPING_MODE_TRACE_AWARE = "trace_aware"
MAPPING_MODE_ROUND_ROBIN = "round_robin"

MAPPING_MODES = (
    MAPPING_MODE_TRACE_AWARE,
    MAPPING_MODE_ROUND_ROBIN,
)


# ============================================================
# 异常
# ============================================================


class SubcubeMappingError(ValueError):
    """LogicalPlane -> Sub-Cube 映射失败。"""


# ============================================================
# 单张 LogicalPlane 的最终逻辑位置
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class LogicalPlanePlacement:
    """
    一张 LogicalPlane 最终对应：

        Sub-Cube
        z

    注意：

    logical_plane_id：

        是第四步逻辑 Plane 编号。

    这里仍然没有绑定：

        第三步 physical plane_id。
    """

    logical_plane_id: int

    subcube_id: int

    z: int

    def __post_init__(
        self,
    ) -> None:

        if (
            self.logical_plane_id
            < 0
        ):
            raise SubcubeMappingError(
                "logical_plane_id "
                "不能为负数。"
            )

        if (
            self.subcube_id
            < 0
        ):
            raise SubcubeMappingError(
                "subcube_id "
                "不能为负数。"
            )

        if self.z < 0:
            raise SubcubeMappingError(
                "z 不能为负数。"
            )


# ============================================================
# 完整 Mapping Result
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class SubcubeMappingResult:
    """
    LogicalPlane -> Sub-Cube 的完整结果。
    """

    hardware: (
        ResolvedHardwareConfig
    )

    placements: tuple[
        LogicalPlanePlacement,
        ...
    ]

    # 每个 Sub-Cube 实际使用 Plane 数量
    subcube_plane_counts: tuple[
        int,
        ...
    ]

    # ========================================================
    # gate/down Plane 所在 SC
    #
    # shape:
    #
    #     58 × 257
    #
    # gate_down_subcube_by_layer[l][e]
    # ========================================================

    gate_down_subcube_by_layer: tuple[
        tuple[
            int,
            ...
        ],
        ...
    ]

    # ========================================================
    # Trace 加权负载
    #
    # pre:
    #     gate + up
    #
    # down:
    #     down
    #
    # shape:
    #
    #     58 × N²
    # ========================================================

    pre_weighted_load_by_layer: tuple[
        tuple[
            int,
            ...
        ],
        ...
    ]

    down_weighted_load_by_layer: tuple[
        tuple[
            int,
            ...
        ],
        ...
    ]

    # ========================================================
    # Trace 冲突代价
    # ========================================================

    pre_conflict_cost: int

    down_conflict_cost: int

    # ========================================================
    # 属性
    # ========================================================

    @property
    def total_planes(
        self,
    ) -> int:

        return len(
            self.placements
        )

    @property
    def total_conflict_cost(
        self,
    ) -> int:

        return (
            self.pre_conflict_cost
            + self.down_conflict_cost
        )

    @property
    def max_planes_in_subcube(
        self,
    ) -> int:

        return max(
            self.subcube_plane_counts,
            default=0,
        )

    @property
    def min_planes_in_subcube(
        self,
    ) -> int:

        return min(
            self.subcube_plane_counts,
            default=0,
        )

    @property
    def empty_plane_slots(
        self,
    ) -> int:
        """
        Q - P
        """

        return (
            self.hardware
            .total_plane_slots
            - self.total_planes
        )

    @property
    def max_pre_weighted_load(
        self,
    ) -> int:

        return max(
            (
                value
                for layer
                in self.pre_weighted_load_by_layer
                for value
                in layer
            ),
            default=0,
        )

    @property
    def max_down_weighted_load(
        self,
    ) -> int:

        return max(
            (
                value
                for layer
                in self.down_weighted_load_by_layer
                for value
                in layer
            ),
            default=0,
        )

    # ========================================================
    # 查询
    # ========================================================

    def placement_of(
        self,
        logical_plane_id: int,
    ) -> LogicalPlanePlacement:
        """
        查询某张 LogicalPlane 的位置。
        """

        if not (
            0
            <= logical_plane_id
            < len(self.placements)
        ):
            raise SubcubeMappingError(
                "logical_plane_id="
                f"{logical_plane_id} "
                "超出范围。"
            )

        placement = (
            self.placements[
                logical_plane_id
            ]
        )

        if (
            placement.logical_plane_id
            != logical_plane_id
        ):
            raise SubcubeMappingError(
                "placements 没有按照 "
                "logical_plane_id 排序。"
            )

        return placement

    def gate_down_subcube(
        self,
        layer_id: int,
        expert_id: int,
    ) -> int:
        """
        查询某 Expert 的 gate+down Plane
        位于哪个 Sub-Cube。
        """

        if not (
            0
            <= layer_id
            < NUM_MOE_LAYERS
        ):
            raise SubcubeMappingError(
                "layer_id 超出范围。"
            )

        if not (
            0
            <= expert_id
            <= SHARED_EXPERT_ID
        ):
            raise SubcubeMappingError(
                "expert_id 超出范围。"
            )

        return (
            self
            .gate_down_subcube_by_layer[
                layer_id
            ][
                expert_id
            ]
        )


# ============================================================
# Expert 激活次数
# ============================================================


def _activation_count(
    profile: TraceProfile,
    layer_id: int,
    expert_id: int,
) -> int:
    """
    Routed Expert：

        frequency[l][e]

    Shared Expert：

        每个 token 都执行

        = token_count_by_layer[l]
    """

    if (
        expert_id
        == SHARED_EXPERT_ID
    ):

        return int(
            profile
            .token_count_by_layer[
                layer_id
            ]
        )

    if not (
        0
        <= expert_id
        < NUM_ROUTED_EXPERTS
    ):
        raise SubcubeMappingError(
            f"非法 Expert ID="
            f"{expert_id}。"
        )

    return int(
        profile.frequency[
            layer_id
        ][
            expert_id
        ]
    )


# ============================================================
# Expert 共激活次数
# ============================================================


def _coactivation_count(
    profile: TraceProfile,
    layer_id: int,
    expert_a: int,
    expert_b: int,
) -> int:
    """
    Routed / Routed：

        G[l][a][b]

    Shared / Routed：

        Shared 永远激活。

        因此：

        G(shared,e)
        =
        frequency[e]
    """

    if (
        expert_a
        == expert_b
    ):
        raise SubcubeMappingError(
            "同一个 Expert 不应通过 "
            "_coactivation_count() "
            "进行比较。"
        )

    a_shared = (
        expert_a
        == SHARED_EXPERT_ID
    )

    b_shared = (
        expert_b
        == SHARED_EXPERT_ID
    )

    # 同一层只有一个 Shared Expert
    if (
        a_shared
        and b_shared
    ):
        raise SubcubeMappingError(
            "同一层只有一个 "
            "Shared Expert。"
        )

    if a_shared:

        return _activation_count(
            profile,
            layer_id,
            expert_b,
        )

    if b_shared:

        return _activation_count(
            profile,
            layer_id,
            expert_a,
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


# ============================================================
# 提取 gate/down Plane 身份
# ============================================================


def _extract_gate_down_expert(
    plane: LogicalPlane,
    cube_index: dict[
        int,
        LogicalWeightCube,
    ],
) -> tuple[
    int,
    int,
]:
    """
    返回：

        (layer_id, expert_id)
    """

    cube_a, cube_b = (
        get_plane_cubes(
            plane=plane,
            cube_index=cube_index,
        )
    )

    names = {
        cube_a.matrix_name,
        cube_b.matrix_name,
    }

    if names != {
        MATRIX_GATE,
        MATRIX_DOWN,
    }:

        raise SubcubeMappingError(
            f"LogicalPlane-"
            f"{plane.logical_plane_id} "
            "标记为 gate_down，"
            "但实际不是 gate+down。"
        )

    if (
        cube_a.layer_id
        != cube_b.layer_id
        or
        cube_a.expert_id
        != cube_b.expert_id
    ):

        raise SubcubeMappingError(
            f"LogicalPlane-"
            f"{plane.logical_plane_id} "
            "中的 gate/down "
            "不属于同一个 Expert。"
        )

    return (
        cube_a.layer_id,
        cube_a.expert_id,
    )


# ============================================================
# 提取 up-up Plane 身份
# ============================================================


def _extract_up_members(
    plane: LogicalPlane,
    cube_index: dict[
        int,
        LogicalWeightCube,
    ],
) -> tuple[
    tuple[int, int],
    tuple[int, int],
]:
    """
    返回：

        (
            (layer_a, expert_a),
            (layer_b, expert_b)
        )

    支持当前两种结构：

    1. 同层两个 Routed up；

    2. 不同层两个 Shared up。
    """

    cube_a, cube_b = (
        get_plane_cubes(
            plane=plane,
            cube_index=cube_index,
        )
    )

    if (
        cube_a.matrix_name
        != MATRIX_UP
        or
        cube_b.matrix_name
        != MATRIX_UP
    ):

        raise SubcubeMappingError(
            f"LogicalPlane-"
            f"{plane.logical_plane_id} "
            "标记为 up_up，"
            "但实际并非两个 up。"
        )

    member_a = (
        cube_a.layer_id,
        cube_a.expert_id,
    )

    member_b = (
        cube_b.layer_id,
        cube_b.expert_id,
    )

    # ========================================================
    # 当前不允许 Shared + Routed
    # ========================================================

    if (
        cube_a.is_shared
        != cube_b.is_shared
    ):

        raise SubcubeMappingError(
            "当前 Baseline 不允许 "
            "Shared up 与 Routed up "
            "混合组成 Plane。"
        )

    # ========================================================
    # Shared + Shared
    # ========================================================

    if cube_a.is_shared:

        if (
            cube_a.expert_id
            != SHARED_EXPERT_ID
            or
            cube_b.expert_id
            != SHARED_EXPERT_ID
        ):

            raise SubcubeMappingError(
                "Shared Expert ID "
                "必须为 256。"
            )

        if (
            cube_a.layer_id
            == cube_b.layer_id
        ):

            raise SubcubeMappingError(
                "两个 Shared up "
                "必须来自不同 Layer。"
            )

    # ========================================================
    # Routed + Routed
    # ========================================================

    else:

        if (
            cube_a.layer_id
            != cube_b.layer_id
        ):

            raise SubcubeMappingError(
                "两个 Routed up "
                "必须来自同一 Layer。"
            )

        if (
            cube_a.expert_id
            >= NUM_ROUTED_EXPERTS
            or
            cube_b.expert_id
            >= NUM_ROUTED_EXPERTS
        ):

            raise SubcubeMappingError(
                "Routed Expert ID "
                "必须位于 0~255。"
            )

    return (
        member_a,
        member_b,
    )


# ============================================================
# 将 Plane 分组
# ============================================================


def _build_plane_groups(
    pairing: PairingResult,
    cubes: Iterable[
        LogicalWeightCube
    ],
) -> tuple[
    dict[
        tuple[int, int],
        LogicalPlane,
    ],
    list[
        LogicalPlane
    ],
    dict[
        int,
        LogicalWeightCube,
    ],
]:
    """
    得到：

        gate_down_planes[
            (layer, expert)
        ]

    和：

        up_planes
    """

    cube_list = tuple(
        cubes
    )

    cube_index = (
        build_cube_index(
            cube_list
        )
    )

    gate_down_planes: dict[
        tuple[int, int],
        LogicalPlane,
    ] = {}

    up_planes: list[
        LogicalPlane
    ] = []

    for plane in pairing.planes:

        # ====================================================
        # gate + down
        # ====================================================

        if plane.is_gate_down:

            key = (
                _extract_gate_down_expert(
                    plane=plane,
                    cube_index=cube_index,
                )
            )

            if (
                key
                in gate_down_planes
            ):

                raise SubcubeMappingError(
                    f"Expert {key} "
                    "出现重复 gate_down Plane。"
                )

            gate_down_planes[
                key
            ] = plane

        # ====================================================
        # up + up
        # ====================================================

        elif plane.is_up_up:

            _extract_up_members(
                plane=plane,
                cube_index=cube_index,
            )

            up_planes.append(
                plane
            )

        else:

            raise SubcubeMappingError(
                "未知 LogicalPlane 类型："
                f"{plane.pair_kind}。"
            )

    expected_gate_down = (
        NUM_MOE_LAYERS
        * (
            NUM_ROUTED_EXPERTS
            + 1
        )
    )

    if (
        len(gate_down_planes)
        != expected_gate_down
    ):

        raise SubcubeMappingError(
            "gate_down Plane 数量错误："
            f"actual="
            f"{len(gate_down_planes)}, "
            f"expected="
            f"{expected_gate_down}。"
        )

    return (
        gate_down_planes,
        up_planes,
        cube_index,
    )


# ============================================================
# gate/down Plane 候选评分
# ============================================================


def _choose_gate_down_subcube(
    *,
    layer_id: int,
    expert_id: int,
    profile: TraceProfile,

    gate_experts_by_layer_sc: list[
        list[
            list[int]
        ]
    ],

    pre_load: list[
        list[int]
    ],

    down_load: list[
        list[int]
    ],

    global_load: list[int],

    plane_counts: list[int],

    gate_phase_cap: int,
) -> int:
    """
    gate/down Plane 的候选评分：

    1. 新增共激活冲突最少；
    2. 该层 projected stage load 最小；
    3. 全局 weighted load 最小；
    4. 当前 Plane 数较少；
    5. subcube_id。

    --------------------------------------------------------

    gate/down 共 Plane，因此：

        一个 Expert 与另一 Expert
        共享同一个 SC

    如果它们共同激活，会同时影响：

        gate stage
        down stage

    所以这里 conflict × 2。
    """

    activation = (
        _activation_count(
            profile,
            layer_id,
            expert_id,
        )
    )

    candidates = [
        sc
        for sc
        in range(
            len(plane_counts)
        )
        if (
            plane_counts[sc]
            < gate_phase_cap
        )
    ]

    if not candidates:

        raise SubcubeMappingError(
            f"Layer-{layer_id} "
            f"Expert-{expert_id} "
            "找不到 gate_down "
            "可用 Sub-Cube。"
        )

    current_pre_peak = max(
        pre_load[
            layer_id
        ]
    )

    current_down_peak = max(
        down_load[
            layer_id
        ]
    )

    def score(
        sc: int,
    ) -> tuple[
        int,
        int,
        int,
        int,
        int,
    ]:

        same_sc_experts = (
            gate_experts_by_layer_sc[
                layer_id
            ][
                sc
            ]
        )

        # ====================================================
        # 与已经位于当前 SC 的 gate/down Expert
        # 的共激活冲突
        # ====================================================

        pair_conflict = sum(
            _coactivation_count(
                profile,
                layer_id,
                expert_id,
                other_expert,
            )
            for other_expert
            in same_sc_experts
        )

        # gate + down 两个阶段都会受影响
        conflict_delta = (
            2
            * pair_conflict
        )

        projected_pre = (
            pre_load[
                layer_id
            ][
                sc
            ]
            + activation
        )

        projected_down = (
            down_load[
                layer_id
            ][
                sc
            ]
            + activation
        )

        projected_layer_peak = max(
            current_pre_peak,
            current_down_peak,
            projected_pre,
            projected_down,
        )

        projected_global_load = (
            global_load[sc]
            + 2 * activation
        )

        return (
            conflict_delta,
            projected_layer_peak,
            projected_global_load,
            plane_counts[sc],
            sc,
        )

    return min(
        candidates,
        key=score,
    )


# ============================================================
# up Plane 排序
# ============================================================


def _up_plane_sort_key(
    *,
    plane: LogicalPlane,

    cube_index: dict[
        int,
        LogicalWeightCube,
    ],

    gate_down_subcube: list[
        list[int]
    ],

    profile: TraceProfile,
) -> tuple[
    int,
    int,
    int,
]:
    """
    up Plane 优先级：

    1. forbidden SC 更多的先放；
    2. 更热的 Plane 先放；
    3. logical_plane_id。

    这样可以减少最后因为容量接近 D
    而找不到合法 SC 的概率。
    """

    member_a, member_b = (
        _extract_up_members(
            plane=plane,
            cube_index=cube_index,
        )
    )

    forbidden = {
        gate_down_subcube[
            layer_id
        ][
            expert_id
        ]

        for (
            layer_id,
            expert_id,
        )
        in (
            member_a,
            member_b,
        )
    }

    activation = sum(
        _activation_count(
            profile,
            layer_id,
            expert_id,
        )
        for (
            layer_id,
            expert_id,
        )
        in (
            member_a,
            member_b,
        )
    )

    return (
        -len(forbidden),
        -activation,
        plane.logical_plane_id,
    )


# ============================================================
# up Plane 候选评分
# ============================================================


def _choose_up_subcube(
    *,
    plane: LogicalPlane,

    members: tuple[
        tuple[int, int],
        tuple[int, int],
    ],

    profile: TraceProfile,

    gate_down_subcube: list[
        list[int]
    ],

    gate_experts_by_layer_sc: list[
        list[
            list[int]
        ]
    ],

    up_experts_by_layer_sc: list[
        list[
            list[int]
        ]
    ],

    pre_load: list[
        list[int]
    ],

    global_load: list[int],

    plane_counts: list[int],

    D: int,
) -> int:
    """
    为一个 up+up Plane 选择 Sub-Cube。

    --------------------------------------------------------

    第一条是硬约束：

        up(e)
        不能和
        gate(e)

    位于同一个 Sub-Cube。

    一个 up-up Plane 有两个成员：

        up(e1)
        up(e2)

    因此需要同时避开：

        gate(e1) 所在 SC
        gate(e2) 所在 SC

    --------------------------------------------------------

    剩余候选按照：

    1. 新增 trace conflict；
    2. affected Layer 的 pre load peak；
    3. 全局 load；
    4. Plane 数；
    5. subcube_id。

    进行字典序选择。
    """

    # ========================================================
    # Gate/Up 分离
    # ========================================================

    forbidden = {
        gate_down_subcube[
            layer_id
        ][
            expert_id
        ]

        for (
            layer_id,
            expert_id,
        )
        in members
    }

    candidates = [
        sc

        for sc
        in range(
            len(plane_counts)
        )

        if (
            sc not in forbidden
            and
            plane_counts[sc] < D
        )
    ]

    if not candidates:

        raise SubcubeMappingError(
            f"LogicalPlane-"
            f"{plane.logical_plane_id} "
            "不存在同时满足 "
            "gate/up 分离和容量约束的 "
            "Sub-Cube。"
        )

    # ========================================================
    # 按 Layer 整理成员
    #
    # Routed up Plane：
    #
    #     同一 Layer 两个 Expert
    #
    # Shared up Plane：
    #
    #     两个不同 Layer
    # ========================================================

    additions_by_layer: dict[
        int,
        list[int],
    ] = {}

    total_activation = 0

    for (
        layer_id,
        expert_id,
    ) in members:

        additions_by_layer.setdefault(
            layer_id,
            [],
        ).append(
            expert_id
        )

        total_activation += (
            _activation_count(
                profile,
                layer_id,
                expert_id,
            )
        )

    current_layer_peaks = {
        layer_id: max(
            pre_load[
                layer_id
            ]
        )

        for layer_id
        in additions_by_layer
    }

    # ========================================================
    # 候选评分
    # ========================================================

    def score(
        sc: int,
    ) -> tuple[
        int,
        int,
        int,
        int,
        int,
    ]:

        conflict_delta = 0

        # ====================================================
        # 1. 与该 SC 中已有 gate/up 的冲突
        # ====================================================

        for (
            layer_id,
            expert_ids,
        ) in (
            additions_by_layer.items()
        ):

            for expert_id in expert_ids:

                # --------------------------------------------
                # up(e) 与已有 gate(other)
                # --------------------------------------------

                for other_expert in (
                    gate_experts_by_layer_sc[
                        layer_id
                    ][
                        sc
                    ]
                ):

                    conflict_delta += (
                        _coactivation_count(
                            profile,
                            layer_id,
                            expert_id,
                            other_expert,
                        )
                    )

                # --------------------------------------------
                # up(e) 与已有 up(other)
                # --------------------------------------------

                for other_expert in (
                    up_experts_by_layer_sc[
                        layer_id
                    ][
                        sc
                    ]
                ):

                    conflict_delta += (
                        _coactivation_count(
                            profile,
                            layer_id,
                            expert_id,
                            other_expert,
                        )
                    )

        # ====================================================
        # 2. affected Layer 的 pre 最大负载
        # ====================================================

        projected_affected_peak = 0

        for (
            layer_id,
            expert_ids,
        ) in (
            additions_by_layer.items()
        ):

            added_load = sum(
                _activation_count(
                    profile,
                    layer_id,
                    expert_id,
                )

                for expert_id
                in expert_ids
            )

            projected_affected_peak = max(
                projected_affected_peak,

                current_layer_peaks[
                    layer_id
                ],

                pre_load[
                    layer_id
                ][
                    sc
                ]
                + added_load,
            )

        # ====================================================
        # 3. 全局 weighted load
        # ====================================================

        projected_global_load = (
            global_load[sc]
            + total_activation
        )

        return (
            conflict_delta,
            projected_affected_peak,
            projected_global_load,
            plane_counts[sc],
            sc,
        )

    return min(
        candidates,
        key=score,
    )


# ============================================================
# 最终 Conflict 精确重新统计
# ============================================================


def _calculate_conflicts(
    *,
    profile: TraceProfile,

    gate_experts_by_layer_sc: list[
        list[
            list[int]
        ]
    ],

    up_experts_by_layer_sc: list[
        list[
            list[int]
        ]
    ],
) -> tuple[
    int,
    int,
]:
    """
    最后根据完整映射重新计算：

        pre conflict
        down conflict

    --------------------------------------------------------

    pre stage：

        gate + up

    只要两个同时可能执行的任务：

        属于不同 Expert
        且位于同一 SC

    就按照 Expert 共激活次数累计。

    --------------------------------------------------------

    down stage：

        所有 down

    因为 down 与 gate 共 Plane，

    所以 down 的 SC 分布
    等于 gate_experts_by_layer_sc。
    """

    pre_conflict = 0

    down_conflict = 0

    num_subcubes = len(
        gate_experts_by_layer_sc[
            0
        ]
    )

    for layer_id in range(
        NUM_MOE_LAYERS
    ):

        for sc in range(
            num_subcubes
        ):

            gate_experts = (
                gate_experts_by_layer_sc[
                    layer_id
                ][
                    sc
                ]
            )

            up_experts = (
                up_experts_by_layer_sc[
                    layer_id
                ][
                    sc
                ]
            )

            # =================================================
            # Pre Stage
            # =================================================

            pre_tasks = (
                [
                    (
                        expert_id,
                        "gate",
                    )
                    for expert_id
                    in gate_experts
                ]
                +
                [
                    (
                        expert_id,
                        "up",
                    )
                    for expert_id
                    in up_experts
                ]
            )

            for i in range(
                len(pre_tasks) - 1
            ):

                expert_a, _ = (
                    pre_tasks[i]
                )

                for j in range(
                    i + 1,
                    len(pre_tasks),
                ):

                    expert_b, _ = (
                        pre_tasks[j]
                    )

                    # 同 Expert gate/up 不允许共址
                    if (
                        expert_a
                        == expert_b
                    ):

                        raise SubcubeMappingError(
                            f"Layer-{layer_id} "
                            f"SC-{sc} "
                            "出现同 Expert "
                            "gate/up 共址。"
                        )

                    pre_conflict += (
                        _coactivation_count(
                            profile,
                            layer_id,
                            expert_a,
                            expert_b,
                        )
                    )

            # =================================================
            # Down Stage
            # =================================================

            for i in range(
                len(gate_experts) - 1
            ):

                expert_a = (
                    gate_experts[i]
                )

                for j in range(
                    i + 1,
                    len(gate_experts),
                ):

                    expert_b = (
                        gate_experts[j]
                    )

                    down_conflict += (
                        _coactivation_count(
                            profile,
                            layer_id,
                            expert_a,
                            expert_b,
                        )
                    )

    return (
        pre_conflict,
        down_conflict,
    )



# ============================================================
# Round-Robin Baseline
# ============================================================


def _cyclic_distance(
    *,
    start: int,
    target: int,
    size: int,
) -> int:
    """
    从 start 开始循环扫描到 target 的距离。

    仅用于 Round-Robin 的确定性 tie-break。
    """

    return (
        target - start
    ) % size


def _choose_round_robin_subcube(
    *,
    plane_counts: list[int],
    D: int,
    cursor: int,
    forbidden: set[int],
) -> int:
    """
    受约束的 Round-Robin。

    不读取：
        frequency
        coactivation
        weighted load

    只遵守两个硬约束：

    1. 每个 Sub-Cube 最多 D 个 Plane；
    2. forbidden 中的 Sub-Cube 不可选
       （当前用于保持同 Expert gate/up 分离）。

    为避免在接近容量上限时过早塞满少数 SC，
    先选择“当前 Plane 数最少”的合法 SC，
    再按照 cursor -> cursor+1 -> ... 的轮询顺序
    做确定性 tie-break。

    因此它仍然是一个不看 Trace 的、
    容量均衡的 Round-Robin Baseline。
    """

    num_subcubes = len(
        plane_counts
    )

    if num_subcubes <= 0:
        raise SubcubeMappingError(
            "Round-Robin：Sub-Cube 数量必须大于 0。"
        )

    if not (
        0
        <= cursor
        < num_subcubes
    ):
        raise SubcubeMappingError(
            f"Round-Robin cursor={cursor} 非法。"
        )

    candidates = [
        sc
        for sc
        in range(
            num_subcubes
        )
        if (
            sc not in forbidden
            and
            plane_counts[sc] < D
        )
    ]

    if not candidates:
        raise SubcubeMappingError(
            "Round-Robin 找不到满足容量与 "
            "gate/up 分离约束的 Sub-Cube。"
        )

    min_count = min(
        plane_counts[sc]
        for sc
        in candidates
    )

    balanced_candidates = [
        sc
        for sc
        in candidates
        if (
            plane_counts[sc]
            == min_count
        )
    ]

    return min(
        balanced_candidates,
        key=lambda sc: (
            _cyclic_distance(
                start=cursor,
                target=sc,
                size=num_subcubes,
            ),
            sc,
        ),
    )


def _map_logical_planes_round_robin(
    *,
    pairing: PairingResult,

    cubes: Iterable[
        LogicalWeightCube
    ],

    profile: TraceProfile,

    hardware: (
        ResolvedHardwareConfig
    ),
) -> SubcubeMappingResult:
    """
    不使用 Trace 做决策的 Sub-Cube Mapping Baseline。

    映射顺序：

    1. gate/down Plane 按 logical_plane_id 排序；
    2. up/up Plane 按 logical_plane_id 排序；
    3. 每张 Plane 使用受约束 Round-Robin 分配；
    4. gate/up 分离仍作为硬约束保留；
    5. 最后才使用 Trace 统计 conflict / weighted load，
       这些统计不会反过来影响 Round-Robin 的选择。

    这正好用于消融：

        Pairing Only:
            Trace-aware Pairing
            +
            Round-Robin Mapping

        Naive:
            Sequential Pairing
            +
            Round-Robin Mapping
    """

    cube_list = tuple(
        cubes
    )

    (
        gate_down_planes,
        up_planes,
        cube_index,
    ) = _build_plane_groups(
        pairing=pairing,
        cubes=cube_list,
    )

    num_subcubes = (
        hardware.num_subcubes
    )

    plane_counts = [
        0
        for _ in range(
            num_subcubes
        )
    ]

    gate_experts_by_layer_sc = [
        [
            []
            for _ in range(
                num_subcubes
            )
        ]
        for _ in range(
            NUM_MOE_LAYERS
        )
    ]

    up_experts_by_layer_sc = [
        [
            []
            for _ in range(
                num_subcubes
            )
        ]
        for _ in range(
            NUM_MOE_LAYERS
        )
    ]

    pre_load = [
        [
            0
            for _ in range(
                num_subcubes
            )
        ]
        for _ in range(
            NUM_MOE_LAYERS
        )
    ]

    down_load = [
        [
            0
            for _ in range(
                num_subcubes
            )
        ]
        for _ in range(
            NUM_MOE_LAYERS
        )
    ]

    gate_down_subcube = [
        [
            -1
            for _ in range(
                NUM_ROUTED_EXPERTS
                + 1
            )
        ]
        for _ in range(
            NUM_MOE_LAYERS
        )
    ]

    plane_to_subcube: dict[
        int,
        int,
    ] = {}

    # ========================================================
    # 第一阶段：gate/down
    # ========================================================

    cursor = 0

    gate_down_order = sorted(
        gate_down_planes.values(),
        key=lambda plane: (
            plane.logical_plane_id
        ),
    )

    for plane in gate_down_order:

        (
            layer_id,
            expert_id,
        ) = _extract_gate_down_expert(
            plane=plane,
            cube_index=cube_index,
        )

        sc = _choose_round_robin_subcube(
            plane_counts=plane_counts,
            D=hardware.D,
            cursor=cursor,
            forbidden=set(),
        )

        cursor = (
            sc + 1
        ) % num_subcubes

        activation = (
            _activation_count(
                profile,
                layer_id,
                expert_id,
            )
        )

        plane_to_subcube[
            plane.logical_plane_id
        ] = sc

        gate_down_subcube[
            layer_id
        ][
            expert_id
        ] = sc

        gate_experts_by_layer_sc[
            layer_id
        ][
            sc
        ].append(
            expert_id
        )

        pre_load[
            layer_id
        ][
            sc
        ] += activation

        down_load[
            layer_id
        ][
            sc
        ] += activation

        plane_counts[
            sc
        ] += 1

    # ========================================================
    # 第二阶段：up/up
    # ========================================================

    up_order = sorted(
        up_planes,
        key=lambda plane: (
            plane.logical_plane_id
        ),
    )

    for plane in up_order:

        members = (
            _extract_up_members(
                plane=plane,
                cube_index=cube_index,
            )
        )

        forbidden = {
            gate_down_subcube[
                layer_id
            ][
                expert_id
            ]
            for (
                layer_id,
                expert_id,
            ) in members
        }

        if (
            -1
            in forbidden
        ):
            raise SubcubeMappingError(
                f"LogicalPlane-{plane.logical_plane_id} "
                "对应 Expert 的 gate/down "
                "尚未完成 Round-Robin 映射。"
            )

        sc = _choose_round_robin_subcube(
            plane_counts=plane_counts,
            D=hardware.D,
            cursor=cursor,
            forbidden=forbidden,
        )

        cursor = (
            sc + 1
        ) % num_subcubes

        plane_to_subcube[
            plane.logical_plane_id
        ] = sc

        plane_counts[
            sc
        ] += 1

        for (
            layer_id,
            expert_id,
        ) in members:

            activation = (
                _activation_count(
                    profile,
                    layer_id,
                    expert_id,
                )
            )

            up_experts_by_layer_sc[
                layer_id
            ][
                sc
            ].append(
                expert_id
            )

            pre_load[
                layer_id
            ][
                sc
            ] += activation

    # ========================================================
    # 第三阶段：z
    # ========================================================

    plane_ids_by_sc = [
        []
        for _ in range(
            num_subcubes
        )
    ]

    for plane in pairing.planes:

        try:
            sc = plane_to_subcube[
                plane.logical_plane_id
            ]
        except KeyError as exc:
            raise SubcubeMappingError(
                f"LogicalPlane-{plane.logical_plane_id} "
                "未完成 Round-Robin 映射。"
            ) from exc

        plane_ids_by_sc[
            sc
        ].append(
            plane.logical_plane_id
        )

    placement_by_id: dict[
        int,
        LogicalPlanePlacement,
    ] = {}

    for sc in range(
        num_subcubes
    ):

        plane_ids = sorted(
            plane_ids_by_sc[
                sc
            ]
        )

        if (
            len(plane_ids)
            > hardware.D
        ):
            raise SubcubeMappingError(
                f"Round-Robin：Sub-Cube-{sc} "
                f"使用 {len(plane_ids)} 个 Plane，"
                f"超过 D={hardware.D}。"
            )

        for (
            z,
            logical_plane_id,
        ) in enumerate(
            plane_ids
        ):

            placement_by_id[
                logical_plane_id
            ] = LogicalPlanePlacement(
                logical_plane_id=(
                    logical_plane_id
                ),
                subcube_id=sc,
                z=z,
            )

    placements = tuple(
        placement_by_id[
            plane.logical_plane_id
        ]
        for plane in sorted(
            pairing.planes,
            key=lambda item: (
                item.logical_plane_id
            ),
        )
    )

    # ========================================================
    # 第四阶段：事后统计
    #
    # 注意：
    # 这里使用 Trace 只用于“测量结果”，
    # 不参与任何 Round-Robin 选择。
    # ========================================================

    (
        pre_conflict,
        down_conflict,
    ) = _calculate_conflicts(
        profile=profile,
        gate_experts_by_layer_sc=(
            gate_experts_by_layer_sc
        ),
        up_experts_by_layer_sc=(
            up_experts_by_layer_sc
        ),
    )

    result = SubcubeMappingResult(
        hardware=hardware,

        placements=placements,

        subcube_plane_counts=tuple(
            plane_counts
        ),

        gate_down_subcube_by_layer=tuple(
            tuple(row)
            for row
            in gate_down_subcube
        ),

        pre_weighted_load_by_layer=tuple(
            tuple(row)
            for row
            in pre_load
        ),

        down_weighted_load_by_layer=tuple(
            tuple(row)
            for row
            in down_load
        ),

        pre_conflict_cost=(
            pre_conflict
        ),

        down_conflict_cost=(
            down_conflict
        ),
    )

    validate_subcube_mapping(
        result=result,
        pairing=pairing,
        cubes=cube_list,
        profile=profile,
    )

    return result


# ============================================================
# 主函数
# ============================================================


def map_logical_planes_to_subcubes(
    *,
    pairing: PairingResult,

    cubes: Iterable[
        LogicalWeightCube
    ],

    profile: TraceProfile,

    hardware: (
        ResolvedHardwareConfig
    ),

    mapping_mode: str = (
        MAPPING_MODE_TRACE_AWARE
    ),
) -> SubcubeMappingResult:
    """
    将所有 LogicalPlane 映射到 Sub-Cube。

    mapping_mode：

        trace_aware：
            当前正式 Trace-aware Mapping。

        round_robin：
            不使用 Trace 决策的受约束轮询 Baseline；
            仅保留容量与 gate/up 分离硬约束。

    trace_aware 当前顺序：

    第一阶段：
        gate/down Plane

    第二阶段：
        up/up Plane

    第三阶段：
        为每个 SC 内 Plane 分配 z

    第四阶段：
        完整合法性检查
    """

    cube_list = tuple(
        cubes
    )

    if (
        mapping_mode
        not in MAPPING_MODES
    ):
        raise SubcubeMappingError(
            f"未知 mapping_mode={mapping_mode!r}，"
            f"允许值为 {MAPPING_MODES}。"
        )

    # ========================================================
    # 0. 基础检查
    # ========================================================

    if (
        hardware.num_subcubes
        <= 0
    ):
        raise SubcubeMappingError(
            "Sub-Cube 数量必须大于 0。"
        )

    if (
        len(pairing.planes)
        > hardware.total_plane_slots
    ):

        raise SubcubeMappingError(
            "LogicalPlane 数量超过 "
            "硬件容量："
            f"P={len(pairing.planes)}, "
            f"Q="
            f"{hardware.total_plane_slots}。"
        )

    if (
        len(profile.frequency)
        != NUM_MOE_LAYERS
    ):

        raise SubcubeMappingError(
            "TraceProfile 的 Layer 数错误。"
        )

    # ========================================================
    # Round-Robin Baseline
    # ========================================================

    if (
        mapping_mode
        == MAPPING_MODE_ROUND_ROBIN
    ):

        return (
            _map_logical_planes_round_robin(
                pairing=pairing,
                cubes=cube_list,
                profile=profile,
                hardware=hardware,
            )
        )

    # ========================================================
    # Trace-aware Mapping
    # ========================================================

    # ========================================================
    # Plane 分组
    # ========================================================

    (
        gate_down_planes,
        up_planes,
        cube_index,
    ) = _build_plane_groups(
        pairing=pairing,
        cubes=cube_list,
    )

    num_subcubes = (
        hardware.num_subcubes
    )

    # ========================================================
    # 状态
    # ========================================================

    plane_counts = [
        0
        for _ in range(
            num_subcubes
        )
    ]

    # 所有 Layer 加起来的 trace 加权任务量
    global_load = [
        0
        for _ in range(
            num_subcubes
        )
    ]

    # ========================================================
    # 每层每 SC 已经有哪些 gate Expert
    # ========================================================

    gate_experts_by_layer_sc = [
        [
            []
            for _ in range(
                num_subcubes
            )
        ]

        for _ in range(
            NUM_MOE_LAYERS
        )
    ]

    # ========================================================
    # 每层每 SC 已经有哪些 up Expert
    # ========================================================

    up_experts_by_layer_sc = [
        [
            []
            for _ in range(
                num_subcubes
            )
        ]

        for _ in range(
            NUM_MOE_LAYERS
        )
    ]

    # ========================================================
    # Weighted Load
    # ========================================================

    pre_load = [
        [
            0
            for _ in range(
                num_subcubes
            )
        ]

        for _ in range(
            NUM_MOE_LAYERS
        )
    ]

    down_load = [
        [
            0
            for _ in range(
                num_subcubes
            )
        ]

        for _ in range(
            NUM_MOE_LAYERS
        )
    ]

    # ========================================================
    # gate/down 所在 SC
    #
    # 58 × 257
    # ========================================================

    gate_down_subcube = [
        [
            -1
            for _ in range(
                NUM_ROUTED_EXPERTS
                + 1
            )
        ]

        for _ in range(
            NUM_MOE_LAYERS
        )
    ]

    # ========================================================
    # logical_plane_id -> sc
    # ========================================================

    plane_to_subcube: dict[
        int,
        int,
    ] = {}

    # ========================================================
    # 第一阶段：
    # gate + down
    #
    # 14906 Plane
    # ========================================================

    # 为防止第一阶段把部分 SC 塞得太满，
    # gate/down Plane 本身先做近似平均分配。

    gate_phase_cap = (
        (
            len(gate_down_planes)
            + num_subcubes
            - 1
        )
        // num_subcubes
    )

    for layer_id in range(
        NUM_MOE_LAYERS
    ):

        # ====================================================
        # Shared 最先放；
        # 然后 Routed 按热度降序。
        # ====================================================

        experts = list(
            range(
                NUM_ROUTED_EXPERTS
                + 1
            )
        )

        experts.sort(
            key=lambda expert_id: (
                # Shared 第一
                expert_id
                != SHARED_EXPERT_ID,

                # 越热越早
                -_activation_count(
                    profile,
                    layer_id,
                    expert_id,
                ),

                expert_id,
            )
        )

        for expert_id in experts:

            plane = (
                gate_down_planes[
                    (
                        layer_id,
                        expert_id,
                    )
                ]
            )

            sc = (
                _choose_gate_down_subcube(
                    layer_id=layer_id,
                    expert_id=expert_id,

                    profile=profile,

                    gate_experts_by_layer_sc=(
                        gate_experts_by_layer_sc
                    ),

                    pre_load=pre_load,
                    down_load=down_load,

                    global_load=(
                        global_load
                    ),

                    plane_counts=(
                        plane_counts
                    ),

                    gate_phase_cap=(
                        gate_phase_cap
                    ),
                )
            )

            activation = (
                _activation_count(
                    profile,
                    layer_id,
                    expert_id,
                )
            )

            # =================================================
            # 提交
            # =================================================

            plane_to_subcube[
                plane.logical_plane_id
            ] = sc

            gate_down_subcube[
                layer_id
            ][
                expert_id
            ] = sc

            gate_experts_by_layer_sc[
                layer_id
            ][
                sc
            ].append(
                expert_id
            )

            pre_load[
                layer_id
            ][
                sc
            ] += activation

            down_load[
                layer_id
            ][
                sc
            ] += activation

            # 一个 gate/down Plane：
            #
            # gate 一次
            # down 一次
            global_load[
                sc
            ] += (
                2
                * activation
            )

            plane_counts[
                sc
            ] += 1

    # ========================================================
    # 第二阶段：
    # up + up
    # ========================================================

    # 先放限制更强、负载更高的 Plane。
    up_planes.sort(
        key=lambda plane: (
            _up_plane_sort_key(
                plane=plane,

                cube_index=(
                    cube_index
                ),

                gate_down_subcube=(
                    gate_down_subcube
                ),

                profile=profile,
            )
        )
    )

    for plane in up_planes:

        members = (
            _extract_up_members(
                plane=plane,
                cube_index=cube_index,
            )
        )

        sc = (
            _choose_up_subcube(
                plane=plane,
                members=members,

                profile=profile,

                gate_down_subcube=(
                    gate_down_subcube
                ),

                gate_experts_by_layer_sc=(
                    gate_experts_by_layer_sc
                ),

                up_experts_by_layer_sc=(
                    up_experts_by_layer_sc
                ),

                pre_load=pre_load,

                global_load=(
                    global_load
                ),

                plane_counts=(
                    plane_counts
                ),

                D=hardware.D,
            )
        )

        # ====================================================
        # 提交 Plane
        # ====================================================

        plane_to_subcube[
            plane.logical_plane_id
        ] = sc

        plane_counts[
            sc
        ] += 1

        # ====================================================
        # 提交两个 up
        # ====================================================

        for (
            layer_id,
            expert_id,
        ) in members:

            activation = (
                _activation_count(
                    profile,
                    layer_id,
                    expert_id,
                )
            )

            up_experts_by_layer_sc[
                layer_id
            ][
                sc
            ].append(
                expert_id
            )

            pre_load[
                layer_id
            ][
                sc
            ] += activation

            global_load[
                sc
            ] += activation

    # ========================================================
    # 第三阶段：
    # 给每个 SC 内 Plane 分 z
    #
    # z 不影响当前推理周期，
    # 因此用 logical_plane_id 排序保证确定性。
    # ========================================================

    plane_ids_by_sc = [
        []
        for _ in range(
            num_subcubes
        )
    ]

    for plane in pairing.planes:

        try:

            sc = (
                plane_to_subcube[
                    plane.logical_plane_id
                ]
            )

        except KeyError as exc:

            raise SubcubeMappingError(
                f"LogicalPlane-"
                f"{plane.logical_plane_id} "
                "未分配 Sub-Cube。"
            ) from exc

        plane_ids_by_sc[
            sc
        ].append(
            plane.logical_plane_id
        )

    placement_by_id: dict[
        int,
        LogicalPlanePlacement,
    ] = {}

    for sc in range(
        num_subcubes
    ):

        plane_ids = sorted(
            plane_ids_by_sc[
                sc
            ]
        )

        if (
            len(plane_ids)
            > hardware.D
        ):

            raise SubcubeMappingError(
                f"Sub-Cube-{sc} "
                f"使用 "
                f"{len(plane_ids)} "
                "个 Plane，"
                f"超过 D="
                f"{hardware.D}。"
            )

        for (
            z,
            logical_plane_id,
        ) in enumerate(
            plane_ids
        ):

            placement_by_id[
                logical_plane_id
            ] = (
                LogicalPlanePlacement(
                    logical_plane_id=(
                        logical_plane_id
                    ),

                    subcube_id=sc,

                    z=z,
                )
            )

    # ========================================================
    # 按 logical_plane_id 排序
    # ========================================================

    placements: list[
        LogicalPlanePlacement
    ] = []

    for plane in sorted(
        pairing.planes,
        key=lambda item: (
            item.logical_plane_id
        ),
    ):

        placements.append(
            placement_by_id[
                plane.logical_plane_id
            ]
        )

    # ========================================================
    # 第四阶段：
    # 重新计算最终冲突
    # ========================================================

    (
        pre_conflict,
        down_conflict,
    ) = _calculate_conflicts(
        profile=profile,

        gate_experts_by_layer_sc=(
            gate_experts_by_layer_sc
        ),

        up_experts_by_layer_sc=(
            up_experts_by_layer_sc
        ),
    )

    # ========================================================
    # 结果
    # ========================================================

    result = (
        SubcubeMappingResult(
            hardware=hardware,

            placements=tuple(
                placements
            ),

            subcube_plane_counts=tuple(
                plane_counts
            ),

            gate_down_subcube_by_layer=tuple(
                tuple(
                    row
                )
                for row
                in gate_down_subcube
            ),

            pre_weighted_load_by_layer=tuple(
                tuple(
                    row
                )
                for row
                in pre_load
            ),

            down_weighted_load_by_layer=tuple(
                tuple(
                    row
                )
                for row
                in down_load
            ),

            pre_conflict_cost=(
                pre_conflict
            ),

            down_conflict_cost=(
                down_conflict
            ),
        )
    )

    # ========================================================
    # 最终验证
    # ========================================================

    validate_subcube_mapping(
        result=result,
        pairing=pairing,
        cubes=cube_list,
        profile=profile,
    )

    return result


# ============================================================
# 最终合法性检查
# ============================================================


def validate_subcube_mapping(
    *,
    result: SubcubeMappingResult,

    pairing: PairingResult,

    cubes: Iterable[
        LogicalWeightCube
    ],

    profile: TraceProfile,
) -> None:
    """
    检查：

    1. 所有 LogicalPlane 恰好映射一次；
    2. subcube_id 合法；
    3. z < D；
    4. (subcube_id,z) 不重复；
    5. 每个 SC 不超过 D；
    6. 所有 Expert gate/up 分离；
    7. Q >= P；
    8. Q-P 正确。
    """

    placements = (
        result.placements
    )

    hardware = (
        result.hardware
    )

    # ========================================================
    # 1. 数量
    # ========================================================

    if (
        len(placements)
        != len(pairing.planes)
    ):

        raise SubcubeMappingError(
            "Placement 数量与 "
            "LogicalPlane 数量不一致。"
        )

    # ========================================================
    # 2. Plane ID 完整
    # ========================================================

    logical_plane_ids = [
        placement.logical_plane_id
        for placement
        in placements
    ]

    expected_ids = sorted(
        plane.logical_plane_id
        for plane
        in pairing.planes
    )

    if (
        logical_plane_ids
        != expected_ids
    ):

        raise SubcubeMappingError(
            "LogicalPlane 没有被 "
            "完整且唯一地映射。"
        )

    # ========================================================
    # 3. 坐标检查
    # ========================================================

    used_coordinates: set[
        tuple[int, int]
    ] = set()

    for placement in placements:

        if not (
            0
            <= placement.subcube_id
            < hardware.num_subcubes
        ):

            raise SubcubeMappingError(
                "非法 subcube_id="
                f"{placement.subcube_id}。"
            )

        if not (
            0
            <= placement.z
            < hardware.D
        ):

            raise SubcubeMappingError(
                f"非法 z="
                f"{placement.z}，"
                f"D="
                f"{hardware.D}。"
            )

        coordinate = (
            placement.subcube_id,
            placement.z,
        )

        if (
            coordinate
            in used_coordinates
        ):

            raise SubcubeMappingError(
                "物理 Plane 坐标重复："
                f"{coordinate}。"
            )

        used_coordinates.add(
            coordinate
        )

    # ========================================================
    # 4. Sub-Cube Plane 数
    # ========================================================

    actual_counts = [
        0
        for _ in range(
            hardware.num_subcubes
        )
    ]

    for placement in placements:

        actual_counts[
            placement.subcube_id
        ] += 1

    if (
        tuple(actual_counts)
        != result.subcube_plane_counts
    ):

        raise SubcubeMappingError(
            "subcube_plane_counts "
            "与实际 Placement 不一致。"
        )

    if any(
        count > hardware.D
        for count
        in actual_counts
    ):

        raise SubcubeMappingError(
            "存在 Sub-Cube 使用 Plane "
            "数量超过 D。"
        )

    # ========================================================
    # 5. gate/up 必须分离
    # ========================================================

    cube_index = (
        build_cube_index(
            tuple(cubes)
        )
    )

    placement_by_id = {
        placement.logical_plane_id:
        placement

        for placement
        in placements
    }

    gate_sc: dict[
        tuple[int, int],
        int,
    ] = {}

    up_sc: dict[
        tuple[int, int],
        int,
    ] = {}

    for plane in pairing.planes:

        placement = (
            placement_by_id[
                plane.logical_plane_id
            ]
        )

        cube_a, cube_b = (
            get_plane_cubes(
                plane=plane,
                cube_index=cube_index,
            )
        )

        for cube in (
            cube_a,
            cube_b,
        ):

            key = (
                cube.layer_id,
                cube.expert_id,
            )

            if (
                cube.matrix_name
                == MATRIX_GATE
            ):

                gate_sc[
                    key
                ] = (
                    placement.subcube_id
                )

            elif (
                cube.matrix_name
                == MATRIX_UP
            ):

                up_sc[
                    key
                ] = (
                    placement.subcube_id
                )

    expected_experts = (
        NUM_MOE_LAYERS
        * (
            NUM_ROUTED_EXPERTS
            + 1
        )
    )

    if (
        len(gate_sc)
        != expected_experts
    ):

        raise SubcubeMappingError(
            "没有找到所有 Expert "
            "的 gate 映射。"
        )

    if (
        len(up_sc)
        != expected_experts
    ):

        raise SubcubeMappingError(
            "没有找到所有 Expert "
            "的 up 映射。"
        )

    for key in gate_sc:

        if (
            gate_sc[key]
            == up_sc[key]
        ):

            raise SubcubeMappingError(
                f"Expert {key} 的 "
                "gate 和 up "
                "被映射到了同一个 "
                "Sub-Cube。"
            )

    # ========================================================
    # 6. Q、P
    # ========================================================

    if (
        result.total_planes
        > hardware.total_plane_slots
    ):

        raise SubcubeMappingError(
            "总 Plane 数超过硬件 Q。"
        )

    expected_empty = (
        hardware.total_plane_slots
        - len(pairing.planes)
    )

    if (
        result.empty_plane_slots
        != expected_empty
    ):

        raise SubcubeMappingError(
            "空 Plane 数 Q-P "
            "计算错误。"
        )


# ============================================================
# 输出
# ============================================================


def print_subcube_mapping_summary(
    result: SubcubeMappingResult,
) -> None:
    """
    打印 Plane -> Sub-Cube 结果。
    """

    print(
        "\n"
        "========== "
        "LogicalPlane -> Sub-Cube Mapping "
        "=========="
    )

    print(
        f"N："
        f"{result.hardware.N}"
    )

    print(
        f"Sub-Cubes："
        f"{result.hardware.num_subcubes}"
    )

    print(
        f"D："
        f"{result.hardware.D}"
    )

    print(
        f"Logical Planes："
        f"{result.total_planes}"
    )

    print(
        f"Empty Plane Slots："
        f"{result.empty_plane_slots}"
    )

    print(
        "Plane Count Range："
        f"{result.min_planes_in_subcube}"
        " ~ "
        f"{result.max_planes_in_subcube}"
    )

    print(
        f"Pre-stage Conflict Cost："
        f"{result.pre_conflict_cost}"
    )

    print(
        f"Down-stage Conflict Cost："
        f"{result.down_conflict_cost}"
    )

    print(
        f"Total Conflict Cost："
        f"{result.total_conflict_cost}"
    )

    print(
        "\n每个 Sub-Cube 的 Plane 数："
    )

    for (
        subcube_id,
        count,
    ) in enumerate(
        result.subcube_plane_counts
    ):

        print(
            f"  SC-{subcube_id}: "
            f"{count}/"
            f"{result.hardware.D}"
        )