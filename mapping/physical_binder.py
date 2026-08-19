"""
第四步最后阶段：逻辑映射 -> 匿名 PhysicalSlot 绑定。

输入：

1. LogicalWeightCube
2. PairingResult
       确定哪些真实矩阵共享 LogicalPlane
3. SubcubeMappingResult
       确定每个 LogicalPlane 的：
           subcube_id
           z
4. 第三步产生的匿名 Plane / PhysicalSlot

输出：

    WeightCubePlacement

其中包含最终静态物理位置：

    layer_id
    expert_id
    matrix_name

    subcube_id
    z

    physical_plane_id
    slot_id

    x
    y

    slot_rows
    slot_cols

    logical_cube_rotated

------------------------------------------------------------

重要：

第三步：

    PhysicalSlot.orientation_swapped

表示：

    匿名矩形在 MaxRects 装箱时
    相对于匿名需求方向是否发生交换。

第四步：

    logical_cube_rotated

表示：

    真实 gate/up/down 矩阵
    为了匹配这个 PhysicalSlot
    是否需要旋转。

两者不是一个概念。

------------------------------------------------------------

当前 H=7168, W=4096 时：

每张 Plane：

    2 × (7168 × 2048)

因此：

gate/up：

    7168 × 2048
    通常 logical_cube_rotated=False

down：

    2048 × 7168
    通常 logical_cube_rotated=True

------------------------------------------------------------

Physical Plane 与 LogicalPlane 的绑定：

不是简单假设：

    logical_plane_id == physical_plane_id

而是先比较 Plane Signature。

LogicalPlane Signature：

    由其内部 LogicalWeightCube.size_key 组成

Physical Plane Signature：

    使用第三步 Plane.signature()

只有两边 Signature 完全相同才能绑定。

当前最优配置中所有 Plane 的 Signature 相同，
但保留这个机制可以防止以后换 H/W 或切分模板时出错。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

from mapping.logical_plane import (
    LogicalPlane,
    build_cube_index,
    get_plane_cubes,
)

from mapping.logical_weight import (
    LogicalWeightCube,
)

from mapping.plane_pairer import (
    PairingResult,
)

from mapping.subcube_mapper import (
    SubcubeMappingResult,
)

from packing.physical_slot import (
    PhysicalSlot,
    PhysicalSlotError,
)

from packing.plane import (
    Plane,
)


# ============================================================
# 异常
# ============================================================


class PhysicalBindingError(ValueError):
    """LogicalWeightCube -> PhysicalSlot 绑定失败。"""


# ============================================================
# 最终 Weight-Cube Placement
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class WeightCubePlacement:
    """
    一个真实 Weight-Cube 的最终静态物理位置。

    这是第四步最终真正需要交给第五步
    调度模拟器的数据结构。
    """

    # ========================================================
    # 逻辑身份
    # ========================================================

    cube_id: int

    layer_id: int

    expert_id: int

    is_shared: bool

    matrix_name: str

    # ========================================================
    # LogicalPlane
    # ========================================================

    logical_plane_id: int

    # ========================================================
    # 硬件三维位置
    # ========================================================

    subcube_id: int

    z: int

    # ========================================================
    # 第三步匿名 Plane / Slot
    # ========================================================

    physical_plane_id: int

    slot_id: int

    x: int
    y: int

    slot_rows: int
    slot_cols: int

    # ========================================================
    # 原始逻辑矩阵尺寸
    # ========================================================

    logical_rows: int
    logical_cols: int

    # ========================================================
    # 旋转
    # ========================================================

    # 第三步匿名矩形是否交换方向
    slot_orientation_swapped: bool

    # 第四步真实矩阵是否旋转后绑定
    logical_cube_rotated: bool

    # 当前固定
    depth: int = 1

    # ========================================================
    # 检查
    # ========================================================

    def __post_init__(
        self,
    ) -> None:

        if self.cube_id < 0:
            raise PhysicalBindingError(
                "cube_id 不能为负数。"
            )

        if self.layer_id < 0:
            raise PhysicalBindingError(
                "layer_id 不能为负数。"
            )

        if self.expert_id < 0:
            raise PhysicalBindingError(
                "expert_id 不能为负数。"
            )

        if (
            self.logical_plane_id
            < 0
        ):
            raise PhysicalBindingError(
                "logical_plane_id "
                "不能为负数。"
            )

        if self.subcube_id < 0:
            raise PhysicalBindingError(
                "subcube_id "
                "不能为负数。"
            )

        if self.z < 0:
            raise PhysicalBindingError(
                "z 不能为负数。"
            )

        if (
            self.physical_plane_id
            < 0
        ):
            raise PhysicalBindingError(
                "physical_plane_id "
                "不能为负数。"
            )

        if self.slot_id < 0:
            raise PhysicalBindingError(
                "slot_id 不能为负数。"
            )

        if self.x < 0 or self.y < 0:
            raise PhysicalBindingError(
                "x、y 不能为负数。"
            )

        if (
            self.slot_rows <= 0
            or self.slot_cols <= 0
        ):
            raise PhysicalBindingError(
                "slot_rows、slot_cols "
                "必须大于 0。"
            )

        if (
            self.logical_rows <= 0
            or self.logical_cols <= 0
        ):
            raise PhysicalBindingError(
                "logical_rows、logical_cols "
                "必须大于 0。"
            )

        if self.depth != 1:
            raise PhysicalBindingError(
                "当前 Weight-Cube "
                "depth 必须恒为 1。"
            )

    # ========================================================
    # 常用属性
    # ========================================================

    @property
    def placed_rows(
        self,
    ) -> int:

        return self.slot_rows

    @property
    def placed_cols(
        self,
    ) -> int:

        return self.slot_cols

    @property
    def physical_coordinate(
        self,
    ) -> tuple[
        int,
        int,
        int,
        int,
    ]:
        """
        返回：

            (
                subcube_id,
                z,
                x,
                y
            )
        """

        return (
            self.subcube_id,
            self.z,
            self.x,
            self.y,
        )

    def summary(
        self,
    ) -> str:

        expert_type = (
            "shared"
            if self.is_shared
            else "routed"
        )

        return (
            f"Cube-{self.cube_id}: "
            f"L{self.layer_id}/"
            f"E{self.expert_id}/"
            f"{expert_type}/"
            f"{self.matrix_name} -> "
            f"SC{self.subcube_id}, "
            f"z={self.z}, "
            f"plane={self.physical_plane_id}, "
            f"slot={self.slot_id}, "
            f"xy=({self.x},{self.y}), "
            f"size="
            f"{self.slot_rows}×{self.slot_cols}, "
            f"rotated="
            f"{self.logical_cube_rotated}"
        )


# ============================================================
# 完整 Binding Result
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PhysicalBindingResult:
    """
    第四步最终静态映射结果。
    """

    placements: tuple[
        WeightCubePlacement,
        ...
    ]

    # (
    #     logical_plane_id,
    #     physical_plane_id,
    # )
    plane_bindings: tuple[
        tuple[int, int],
        ...
    ]

    logical_rotation_count: int

    # ========================================================
    # 属性
    # ========================================================

    @property
    def cube_count(
        self,
    ) -> int:

        return len(
            self.placements
        )

    @property
    def physical_plane_count(
        self,
    ) -> int:

        return len(
            self.plane_bindings
        )

    @property
    def unrotated_count(
        self,
    ) -> int:

        return (
            self.cube_count
            - self.logical_rotation_count
        )

    # ========================================================
    # 查询
    # ========================================================

    def placement_of_cube(
        self,
        cube_id: int,
    ) -> WeightCubePlacement:
        """
        placements 按 cube_id 排序，
        当前完整模型 cube_id 从 0 连续递增。
        """

        if not (
            0
            <= cube_id
            < len(self.placements)
        ):

            raise PhysicalBindingError(
                f"cube_id={cube_id} "
                "超出范围。"
            )

        placement = (
            self.placements[
                cube_id
            ]
        )

        if (
            placement.cube_id
            != cube_id
        ):

            raise PhysicalBindingError(
                "placements 没有按照 "
                "cube_id 排序。"
            )

        return placement


# ============================================================
# LogicalPlane Signature
# ============================================================


def logical_plane_signature(
    *,
    plane: LogicalPlane,

    cube_index: dict[
        int,
        LogicalWeightCube,
    ],
) -> tuple[
    tuple[int, int],
    ...
]:
    """
    LogicalPlane 的尺寸组成签名。

    与 packing.Plane.signature()
    使用完全相同的格式：

        (
            size_key_1,
            size_key_2,
            ...
        )

    并排序。
    """

    cube_a, cube_b = (
        get_plane_cubes(
            plane=plane,
            cube_index=cube_index,
        )
    )

    return tuple(
        sorted(
            (
                cube_a.size_key,
                cube_b.size_key,
            )
        )
    )


# ============================================================
# 基础输入检查
# ============================================================


def validate_binding_inputs(
    *,
    cubes: Iterable[
        LogicalWeightCube
    ],

    pairing: PairingResult,

    subcube_mapping: (
        SubcubeMappingResult
    ),

    physical_planes: Iterable[
        Plane
    ],
) -> None:
    """
    正式绑定前先检查：

    1. LogicalPlane 数 = Physical Plane 数；
    2. LogicalWeightCube 数 = PhysicalSlot 数；
    3. size_key 总数量一致；
    4. Plane Signature 数量一致；
    5. Physical Plane 尺寸与硬件 H/W 一致。
    """

    cube_list = tuple(
        cubes
    )

    plane_list = tuple(
        physical_planes
    )

    # ========================================================
    # 1. Plane 数
    # ========================================================

    if (
        len(pairing.planes)
        != len(plane_list)
    ):

        raise PhysicalBindingError(
            "LogicalPlane 数量与 "
            "Physical Plane 数量不一致："
            f"logical="
            f"{len(pairing.planes)}, "
            f"physical="
            f"{len(plane_list)}。"
        )

    if (
        len(
            subcube_mapping
            .placements
        )
        != len(pairing.planes)
    ):

        raise PhysicalBindingError(
            "Sub-Cube Mapping 中 Plane 数量 "
            "与 PairingResult 不一致。"
        )

    # ========================================================
    # 2. Physical Plane ID 唯一
    # ========================================================

    physical_plane_ids = [
        plane.plane_id
        for plane
        in plane_list
    ]

    if (
        len(physical_plane_ids)
        != len(
            set(
                physical_plane_ids
            )
        )
    ):

        raise PhysicalBindingError(
            "存在重复 physical plane_id。"
        )

    # ========================================================
    # 3. PhysicalSlot ID 唯一
    # ========================================================

    all_slots = [
        slot

        for plane
        in plane_list

        for slot
        in plane.slots
    ]

    slot_ids = [
        slot.slot_id
        for slot
        in all_slots
    ]

    if (
        len(slot_ids)
        != len(
            set(slot_ids)
        )
    ):

        raise PhysicalBindingError(
            "存在重复 PhysicalSlot.slot_id。"
        )

    # ========================================================
    # 4. Cube 数 = Slot 数
    # ========================================================

    if (
        len(cube_list)
        != len(all_slots)
    ):

        raise PhysicalBindingError(
            "LogicalWeightCube 数量与 "
            "PhysicalSlot 数量不一致："
            f"cubes={len(cube_list)}, "
            f"slots={len(all_slots)}。"
        )

    # ========================================================
    # 5. Cube ID 唯一
    # ========================================================

    cube_ids = [
        cube.cube_id
        for cube
        in cube_list
    ]

    if (
        len(cube_ids)
        != len(
            set(cube_ids)
        )
    ):

        raise PhysicalBindingError(
            "存在重复 cube_id。"
        )

    # ========================================================
    # 6. 全局 size_key 数量必须一致
    # ========================================================

    logical_histogram = Counter(
        cube.size_key
        for cube
        in cube_list
    )

    physical_histogram = Counter(
        slot.size_key
        for slot
        in all_slots
    )

    if (
        logical_histogram
        != physical_histogram
    ):

        raise PhysicalBindingError(
            "LogicalWeightCube 与 "
            "PhysicalSlot 的尺寸数量不一致。\n"
            f"logical={dict(logical_histogram)}\n"
            f"physical={dict(physical_histogram)}"
        )

    # ========================================================
    # 7. Plane H/W
    # ========================================================

    H = (
        subcube_mapping
        .hardware
        .H
    )

    W = (
        subcube_mapping
        .hardware
        .W
    )

    for plane in plane_list:

        if (
            plane.H != H
            or plane.W != W
        ):

            raise PhysicalBindingError(
                f"Physical Plane-"
                f"{plane.plane_id} "
                "尺寸与硬件不一致："
                f"plane="
                f"{plane.H}×{plane.W}, "
                f"hardware="
                f"{H}×{W}。"
            )

        plane.validate_layout()

    # ========================================================
    # 8. Plane Signature 数量一致
    # ========================================================

    cube_index = (
        build_cube_index(
            cube_list
        )
    )

    logical_signature_count = Counter(
        logical_plane_signature(
            plane=plane,
            cube_index=cube_index,
        )

        for plane
        in pairing.planes
    )

    physical_signature_count = Counter(
        plane.signature()

        for plane
        in plane_list
    )

    if (
        logical_signature_count
        != physical_signature_count
    ):

        raise PhysicalBindingError(
            "LogicalPlane 与 Physical Plane "
            "的 Signature 数量不一致。\n"
            f"logical="
            f"{dict(logical_signature_count)}\n"
            f"physical="
            f"{dict(physical_signature_count)}"
        )


# ============================================================
# Plane -> Plane 匹配
# ============================================================


def match_logical_planes_to_physical_planes(
    *,
    pairing: PairingResult,

    cubes: Iterable[
        LogicalWeightCube
    ],

    physical_planes: Iterable[
        Plane
    ],
) -> dict[
    int,
    Plane,
]:
    """
    根据 Signature：

        LogicalPlane
            ->
        Physical Plane

    当前配置中所有 Plane Signature 一样，
    所以最后基本等价于：

        logical_plane_id 顺序
            对应
        physical plane_id 顺序

    但这里保留 Signature 分类，
    以后换 H/W 或切分模板仍然可用。
    """

    cube_list = tuple(
        cubes
    )

    physical_plane_list = tuple(
        physical_planes
    )

    cube_index = (
        build_cube_index(
            cube_list
        )
    )

    # ========================================================
    # 按 Signature 分类
    # ========================================================

    logical_groups: dict[
        tuple[
            tuple[int, int],
            ...
        ],
        list[
            LogicalPlane
        ],
    ] = defaultdict(
        list
    )

    physical_groups: dict[
        tuple[
            tuple[int, int],
            ...
        ],
        list[
            Plane
        ],
    ] = defaultdict(
        list
    )

    for logical_plane in (
        pairing.planes
    ):

        signature = (
            logical_plane_signature(
                plane=logical_plane,
                cube_index=cube_index,
            )
        )

        logical_groups[
            signature
        ].append(
            logical_plane
        )

    for physical_plane in (
        physical_plane_list
    ):

        physical_groups[
            physical_plane.signature()
        ].append(
            physical_plane
        )

    if (
        set(
            logical_groups
        )
        != set(
            physical_groups
        )
    ):

        raise PhysicalBindingError(
            "LogicalPlane 与 PhysicalPlane "
            "Signature 类型集合不一致。"
        )

    # ========================================================
    # 一一绑定
    # ========================================================

    result: dict[
        int,
        Plane,
    ] = {}

    for signature in sorted(
        logical_groups
    ):

        logical_planes = sorted(
            logical_groups[
                signature
            ],
            key=lambda plane: (
                plane.logical_plane_id
            ),
        )

        physical_planes_of_type = (
            sorted(
                physical_groups[
                    signature
                ],
                key=lambda plane: (
                    plane.plane_id
                ),
            )
        )

        if (
            len(logical_planes)
            != len(
                physical_planes_of_type
            )
        ):

            raise PhysicalBindingError(
                "Signature="
                f"{signature} "
                "的 LogicalPlane / "
                "PhysicalPlane 数量不同。"
            )

        for (
            logical_plane,
            physical_plane,
        ) in zip(
            logical_planes,
            physical_planes_of_type,
        ):

            result[
                logical_plane
                .logical_plane_id
            ] = physical_plane

    if (
        len(result)
        != len(pairing.planes)
    ):

        raise PhysicalBindingError(
            "Plane 绑定没有完整覆盖 "
            "全部 LogicalPlane。"
        )

    return result


# ============================================================
# 一张 Plane 内：
# Cube -> Slot 最优绑定
# ============================================================


def choose_best_slot_assignment(
    *,
    logical_plane: LogicalPlane,

    physical_plane: Plane,

    cube_index: dict[
        int,
        LogicalWeightCube,
    ],
) -> tuple[
    tuple[
        LogicalWeightCube,
        PhysicalSlot,
        bool,
    ],
    ...,
]:
    """
    为一张 LogicalPlane 内的两个 Cube
    选择具体 PhysicalSlot。

    当前每张 LogicalPlane 有两个 Cube。

    当前每张 Physical Plane 也必须拥有
    两个匹配的 Slot。

    --------------------------------------------------------

    如果有多个可行方案：

    第一优先：
        真实矩阵旋转数量最少；

    第二优先：
        优先让前面的 cube 不旋转；

    第三优先：
        slot_id 较小。

    --------------------------------------------------------

    返回：

        (
            (
                logical_cube,
                physical_slot,
                logical_cube_rotated
            ),
            ...
        )
    """

    cube_a, cube_b = (
        get_plane_cubes(
            plane=logical_plane,
            cube_index=cube_index,
        )
    )

    cubes = (
        cube_a,
        cube_b,
    )

    slots = tuple(
        sorted(
            physical_plane.slots,
            key=lambda slot: (
                slot.slot_id
            ),
        )
    )

    if (
        len(slots)
        != len(cubes)
    ):

        raise PhysicalBindingError(
            f"LogicalPlane-"
            f"{logical_plane.logical_plane_id} "
            f"包含 {len(cubes)} 个 Cube，"
            f"但 PhysicalPlane-"
            f"{physical_plane.plane_id} "
            f"包含 {len(slots)} 个 Slot。"
        )

    candidates = []

    # 当前只有两个 Slot，
    # permutations 只需尝试 2! = 2 种。
    for slot_order in permutations(
        slots
    ):

        rotations: list[
            bool
        ] = []

        feasible = True

        for (
            cube,
            slot,
        ) in zip(
            cubes,
            slot_order,
        ):

            try:

                rotated = (
                    slot
                    .logical_rotation_required(
                        rows=(
                            cube.logical_rows
                        ),
                        cols=(
                            cube.logical_cols
                        ),
                    )
                )

            except PhysicalSlotError:

                feasible = False
                break

            rotations.append(
                rotated
            )

        if not feasible:
            continue

        score = (
            # 总旋转数量越少越好
            sum(
                int(value)
                for value
                in rotations
            ),

            # 如果旋转数量相同，
            # 优先让 cube_a 不旋转。
            tuple(
                int(value)
                for value
                in rotations
            ),

            # 最后保持确定性
            tuple(
                slot.slot_id
                for slot
                in slot_order
            ),
        )

        candidates.append(
            (
                score,
                slot_order,
                tuple(
                    rotations
                ),
            )
        )

    if not candidates:

        raise PhysicalBindingError(
            f"LogicalPlane-"
            f"{logical_plane.logical_plane_id} "
            "无法绑定到 "
            f"PhysicalPlane-"
            f"{physical_plane.plane_id}。"
        )

    (
        _best_score,
        best_slots,
        best_rotations,
    ) = min(
        candidates,
        key=lambda item: (
            item[0]
        ),
    )

    return tuple(
        (
            cube,
            slot,
            rotated,
        )

        for (
            cube,
            slot,
            rotated,
        ) in zip(
            cubes,
            best_slots,
            best_rotations,
        )
    )


# ============================================================
# 主绑定函数
# ============================================================


def bind_logical_mapping_to_physical_slots(
    *,
    cubes: Iterable[
        LogicalWeightCube
    ],

    pairing: PairingResult,

    subcube_mapping: (
        SubcubeMappingResult
    ),

    physical_planes: Iterable[
        Plane
    ],
) -> PhysicalBindingResult:
    """
    完成第四步最后的真实静态物理绑定。

    流程：

    LogicalPlane
        ↓
    找到同 Signature 的 PhysicalPlane
        ↓
    从 SubcubeMappingResult 取得：
        subcube_id
        z
        ↓
    两个 LogicalWeightCube
        ↓
    最优匹配两个 PhysicalSlot
        ↓
    得到 WeightCubePlacement
    """

    cube_list = tuple(
        cubes
    )

    physical_plane_list = tuple(
        physical_planes
    )

    # ========================================================
    # 1. 输入验证
    # ========================================================

    validate_binding_inputs(
        cubes=cube_list,

        pairing=pairing,

        subcube_mapping=(
            subcube_mapping
        ),

        physical_planes=(
            physical_plane_list
        ),
    )

    cube_index = (
        build_cube_index(
            cube_list
        )
    )

    # ========================================================
    # 2. LogicalPlane -> PhysicalPlane
    # ========================================================

    logical_to_physical = (
        match_logical_planes_to_physical_planes(
            pairing=pairing,

            cubes=cube_list,

            physical_planes=(
                physical_plane_list
            ),
        )
    )

    # ========================================================
    # 3. LogicalPlane -> SubCube Placement
    # ========================================================

    mapping_by_logical_plane_id = {
        placement.logical_plane_id:
        placement

        for placement
        in subcube_mapping.placements
    }

    # ========================================================
    # 4. 逐 Plane 绑定
    # ========================================================

    weight_placements: list[
        WeightCubePlacement
    ] = []

    plane_bindings: list[
        tuple[int, int]
    ] = []

    logical_rotation_count = 0

    logical_planes_sorted = sorted(
        pairing.planes,
        key=lambda plane: (
            plane.logical_plane_id
        ),
    )

    for logical_plane in (
        logical_planes_sorted
    ):

        logical_plane_id = (
            logical_plane
            .logical_plane_id
        )

        # ====================================================
        # Physical Plane
        # ====================================================

        try:

            physical_plane = (
                logical_to_physical[
                    logical_plane_id
                ]
            )

        except KeyError as exc:

            raise PhysicalBindingError(
                f"LogicalPlane-"
                f"{logical_plane_id} "
                "没有对应 PhysicalPlane。"
            ) from exc

        # ====================================================
        # Sub-Cube + z
        # ====================================================

        try:

            logical_location = (
                mapping_by_logical_plane_id[
                    logical_plane_id
                ]
            )

        except KeyError as exc:

            raise PhysicalBindingError(
                f"LogicalPlane-"
                f"{logical_plane_id} "
                "没有 Sub-Cube Mapping。"
            ) from exc

        # ====================================================
        # Cube -> Slot
        # ====================================================

        assignments = (
            choose_best_slot_assignment(
                logical_plane=(
                    logical_plane
                ),

                physical_plane=(
                    physical_plane
                ),

                cube_index=(
                    cube_index
                ),
            )
        )

        for (
            cube,
            slot,
            logical_rotated,
        ) in assignments:

            if logical_rotated:

                logical_rotation_count += 1

            weight_placement = (
                WeightCubePlacement(
                    # =========================================
                    # 逻辑
                    # =========================================

                    cube_id=(
                        cube.cube_id
                    ),

                    layer_id=(
                        cube.layer_id
                    ),

                    expert_id=(
                        cube.expert_id
                    ),

                    is_shared=(
                        cube.is_shared
                    ),

                    matrix_name=(
                        cube.matrix_name
                    ),

                    logical_plane_id=(
                        logical_plane_id
                    ),

                    # =========================================
                    # 3D
                    # =========================================

                    subcube_id=(
                        logical_location
                        .subcube_id
                    ),

                    z=(
                        logical_location
                        .z
                    ),

                    # =========================================
                    # 匿名物理 Plane / Slot
                    # =========================================

                    physical_plane_id=(
                        physical_plane
                        .plane_id
                    ),

                    slot_id=(
                        slot.slot_id
                    ),

                    x=slot.x,
                    y=slot.y,

                    slot_rows=(
                        slot.slot_rows
                    ),

                    slot_cols=(
                        slot.slot_cols
                    ),

                    # =========================================
                    # 原矩阵
                    # =========================================

                    logical_rows=(
                        cube.logical_rows
                    ),

                    logical_cols=(
                        cube.logical_cols
                    ),

                    # =========================================
                    # Rotation
                    # =========================================

                    slot_orientation_swapped=(
                        slot.orientation_swapped
                    ),

                    logical_cube_rotated=(
                        logical_rotated
                    ),

                    depth=(
                        cube.depth
                    ),
                )
            )

            weight_placements.append(
                weight_placement
            )

        plane_bindings.append(
            (
                logical_plane_id,
                physical_plane.plane_id,
            )
        )

    # ========================================================
    # 5. 按 cube_id 排序
    # ========================================================

    weight_placements.sort(
        key=lambda placement: (
            placement.cube_id
        )
    )

    plane_bindings.sort(
        key=lambda item: (
            item[0]
        )
    )

    result = (
        PhysicalBindingResult(
            placements=tuple(
                weight_placements
            ),

            plane_bindings=tuple(
                plane_bindings
            ),

            logical_rotation_count=(
                logical_rotation_count
            ),
        )
    )

    # ========================================================
    # 6. 最终检查
    # ========================================================

    validate_physical_binding(
        result=result,

        cubes=cube_list,

        pairing=pairing,

        subcube_mapping=(
            subcube_mapping
        ),

        physical_planes=(
            physical_plane_list
        ),
    )

    return result


# ============================================================
# 最终 Binding 验证
# ============================================================


def validate_physical_binding(
    *,
    result: PhysicalBindingResult,

    cubes: Iterable[
        LogicalWeightCube
    ],

    pairing: PairingResult,

    subcube_mapping: (
        SubcubeMappingResult
    ),

    physical_planes: Iterable[
        Plane
    ],
) -> None:
    """
    最终严格检查：

    1. 每个 Cube 恰好出现一次；
    2. 每个 PhysicalSlot 恰好使用一次；
    3. 每个 LogicalPlane 恰好对应一个 PhysicalPlane；
    4. 同 LogicalPlane 的 Cube：
           physical_plane 相同
           subcube 相同
           z 相同
    5. rotation 与 PhysicalSlot 实际尺寸一致；
    6. 位置不越界；
    7. depth = 1。
    """

    cube_list = tuple(
        cubes
    )

    physical_plane_list = tuple(
        physical_planes
    )

    placements = (
        result.placements
    )

    # ========================================================
    # 1. Cube 数
    # ========================================================

    if (
        len(placements)
        != len(cube_list)
    ):

        raise PhysicalBindingError(
            "最终 Placement 数量与 "
            "LogicalWeightCube 数量不一致。"
        )

    # ========================================================
    # 2. Cube ID 必须完整且唯一
    # ========================================================

    placement_cube_ids = [
        placement.cube_id
        for placement
        in placements
    ]

    expected_cube_ids = sorted(
        cube.cube_id
        for cube
        in cube_list
    )

    if (
        placement_cube_ids
        != expected_cube_ids
    ):

        raise PhysicalBindingError(
            "最终 Placement 没有完整、唯一地 "
            "覆盖所有 LogicalWeightCube。"
        )

    # ========================================================
    # 3. Slot ID 必须完整且唯一
    # ========================================================

    placement_slot_ids = [
        placement.slot_id
        for placement
        in placements
    ]

    expected_slot_ids = sorted(
        slot.slot_id

        for plane
        in physical_plane_list

        for slot
        in plane.slots
    )

    if (
        sorted(
            placement_slot_ids
        )
        != expected_slot_ids
    ):

        raise PhysicalBindingError(
            "最终 Placement 没有完整、唯一地 "
            "使用全部 PhysicalSlot。"
        )

    if (
        len(
            set(
                placement_slot_ids
            )
        )
        != len(
            placement_slot_ids
        )
    ):

        raise PhysicalBindingError(
            "同一个 PhysicalSlot "
            "被多个 WeightCube 重复使用。"
        )

    # ========================================================
    # 4. Plane Binding 唯一
    # ========================================================

    logical_plane_ids = [
        logical_id
        for (
            logical_id,
            _physical_id,
        ) in result.plane_bindings
    ]

    physical_plane_ids = [
        physical_id
        for (
            _logical_id,
            physical_id,
        ) in result.plane_bindings
    ]

    if (
        len(
            set(
                logical_plane_ids
            )
        )
        != len(
            pairing.planes
        )
    ):

        raise PhysicalBindingError(
            "LogicalPlane 没有一一绑定。"
        )

    if (
        len(
            set(
                physical_plane_ids
            )
        )
        != len(
            physical_plane_list
        )
    ):

        raise PhysicalBindingError(
            "PhysicalPlane 没有一一绑定。"
        )

    # ========================================================
    # 5. 建索引
    # ========================================================

    slot_index: dict[
        int,
        PhysicalSlot,
    ] = {}

    for plane in (
        physical_plane_list
    ):

        for slot in (
            plane.slots
        ):

            slot_index[
                slot.slot_id
            ] = slot

    cube_index = (
        build_cube_index(
            cube_list
        )
    )

    logical_location_index = {
        placement.logical_plane_id:
        placement

        for placement
        in subcube_mapping.placements
    }

    binding_index = {
        logical_id:
        physical_id

        for (
            logical_id,
            physical_id,
        )
        in result.plane_bindings
    }

    # ========================================================
    # 6. 每个 Cube 检查
    # ========================================================

    for placement in (
        placements
    ):

        cube = (
            cube_index[
                placement.cube_id
            ]
        )

        slot = (
            slot_index[
                placement.slot_id
            ]
        )

        # ----------------------------------------------------
        # 逻辑身份
        # ----------------------------------------------------

        if (
            placement.layer_id
            != cube.layer_id
            or
            placement.expert_id
            != cube.expert_id
            or
            placement.matrix_name
            != cube.matrix_name
            or
            placement.is_shared
            != cube.is_shared
        ):

            raise PhysicalBindingError(
                f"Cube-{cube.cube_id} "
                "最终 Placement 的逻辑身份错误。"
            )

        # ----------------------------------------------------
        # Slot 必须属于记录的 Physical Plane
        # ----------------------------------------------------

        if (
            slot.plane_id
            != placement.physical_plane_id
        ):

            raise PhysicalBindingError(
                f"Cube-{cube.cube_id} "
                "slot.plane_id 与 "
                "physical_plane_id 不一致。"
            )

        # ----------------------------------------------------
        # Physical Plane 必须与 Plane Binding 一致
        # ----------------------------------------------------

        expected_physical_plane = (
            binding_index[
                placement.logical_plane_id
            ]
        )

        if (
            placement.physical_plane_id
            != expected_physical_plane
        ):

            raise PhysicalBindingError(
                f"Cube-{cube.cube_id} "
                "PhysicalPlane 与 "
                "LogicalPlane Binding 不一致。"
            )

        # ----------------------------------------------------
        # SubCube / z
        # ----------------------------------------------------

        logical_location = (
            logical_location_index[
                placement.logical_plane_id
            ]
        )

        if (
            placement.subcube_id
            != logical_location.subcube_id
            or
            placement.z
            != logical_location.z
        ):

            raise PhysicalBindingError(
                f"Cube-{cube.cube_id} "
                "Sub-Cube / z "
                "与 SubcubeMappingResult 不一致。"
            )

        # ----------------------------------------------------
        # XY 与尺寸
        # ----------------------------------------------------

        if (
            placement.x != slot.x
            or
            placement.y != slot.y
            or
            placement.slot_rows
            != slot.slot_rows
            or
            placement.slot_cols
            != slot.slot_cols
        ):

            raise PhysicalBindingError(
                f"Cube-{cube.cube_id} "
                "PhysicalSlot 几何信息错误。"
            )

        # ----------------------------------------------------
        # Rotation
        # ----------------------------------------------------

        try:

            expected_rotation = (
                slot
                .logical_rotation_required(
                    rows=(
                        cube.logical_rows
                    ),
                    cols=(
                        cube.logical_cols
                    ),
                )
            )

        except PhysicalSlotError as exc:

            raise PhysicalBindingError(
                f"Cube-{cube.cube_id} "
                "与最终 Slot 尺寸不匹配。"
            ) from exc

        if (
            placement
            .logical_cube_rotated
            != expected_rotation
        ):

            raise PhysicalBindingError(
                f"Cube-{cube.cube_id} "
                "logical_cube_rotated "
                "记录错误。"
            )

        # ----------------------------------------------------
        # 第三步 orientation
        # ----------------------------------------------------

        if (
            placement
            .slot_orientation_swapped
            != slot.orientation_swapped
        ):

            raise PhysicalBindingError(
                f"Cube-{cube.cube_id} "
                "slot_orientation_swapped "
                "记录错误。"
            )

        # ----------------------------------------------------
        # depth
        # ----------------------------------------------------

        if placement.depth != 1:

            raise PhysicalBindingError(
                f"Cube-{cube.cube_id} "
                "depth != 1。"
            )

    # ========================================================
    # 7. 同 LogicalPlane 内部必须完全共址
    # ========================================================

    placements_by_logical_plane: dict[
        int,
        list[
            WeightCubePlacement
        ],
    ] = defaultdict(
        list
    )

    for placement in placements:

        placements_by_logical_plane[
            placement.logical_plane_id
        ].append(
            placement
        )

    for logical_plane in (
        pairing.planes
    ):

        members = (
            placements_by_logical_plane[
                logical_plane
                .logical_plane_id
            ]
        )

        if len(members) != 2:

            raise PhysicalBindingError(
                f"LogicalPlane-"
                f"{logical_plane.logical_plane_id} "
                "最终不是恰好两个 WeightCube。"
            )

        first = members[0]
        second = members[1]

        if (
            first.physical_plane_id
            != second.physical_plane_id
        ):

            raise PhysicalBindingError(
                f"LogicalPlane-"
                f"{logical_plane.logical_plane_id} "
                "的两个 Cube "
                "没有绑定到同一个 PhysicalPlane。"
            )

        if (
            first.subcube_id
            != second.subcube_id
            or
            first.z
            != second.z
        ):

            raise PhysicalBindingError(
                f"LogicalPlane-"
                f"{logical_plane.logical_plane_id} "
                "的两个 Cube "
                "没有位于同一个 "
                "Sub-Cube / z。"
            )


# ============================================================
# 输出统计
# ============================================================


def print_physical_binding_summary(
    result: PhysicalBindingResult,
) -> None:
    """
    打印第四步最终绑定结果。
    """

    print(
        "\n"
        "========== Physical Binding =========="
    )

    print(
        f"WeightCube Placements："
        f"{result.cube_count}"
    )

    print(
        f"Physical Planes Used："
        f"{result.physical_plane_count}"
    )

    print(
        f"Logical Cube Rotated："
        f"{result.logical_rotation_count}"
    )

    print(
        f"Logical Cube Unrotated："
        f"{result.unrotated_count}"
    )

    if result.cube_count > 0:

        rotation_ratio = (
            result.logical_rotation_count
            / result.cube_count
        )

        print(
            f"Logical Rotation Ratio："
            f"{rotation_ratio:.4%}"
        )