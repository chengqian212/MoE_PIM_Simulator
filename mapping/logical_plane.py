"""
第四步第二阶段基础数据结构：LogicalPlane。

前三步已经生成：

    22359 个匿名 Physical Plane
    每个 Plane 有两个 7168×2048 PhysicalSlot

第四步第一阶段已经生成：

    44718 个 LogicalWeightCube

其中：

    14906 gate
    14906 up
    14906 down

当前文件只定义：

    两个 LogicalWeightCube
        ↓
    一个 LogicalPlane

重要：

LogicalPlane 只是“两个真实矩阵必须共享同一个物理 Plane”
这一逻辑关系。

本文件不决定：

    physical_plane_id
    subcube_id
    z
    slot_id
    x
    y
    rotated

这些都要到后续映射阶段确定。

当前规划中主要存在两类 LogicalPlane：

1. gate_down

       [ gate(e) | down(e) ]

   同一个 Expert 的 gate 和 down 共 Plane。

2. up_up

       [ up(e1) | up(e2) ]

   两个 up 共 Plane。

其中 up-up 到底哪两个 Expert 配对，
后面由 Chinese-SimpleQA trace 决定，
本文件不实现该策略。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from mapping.logical_weight import (
    LogicalWeightCube,
    MATRIX_DOWN,
    MATRIX_GATE,
    MATRIX_UP,
)


# ============================================================
# 常量
# ============================================================


PAIR_GATE_DOWN = "gate_down"
PAIR_UP_UP = "up_up"

PAIR_KINDS = (
    PAIR_GATE_DOWN,
    PAIR_UP_UP,
)


class LogicalPlaneError(ValueError):
    """LogicalPlane 构造或验证失败。"""


# ============================================================
# LogicalPlane
# ============================================================


@dataclass(frozen=True, slots=True)
class LogicalPlane:
    """
    一张逻辑 Plane。

    当前空间配置：

        H = 7168
        W = 4096

    每张匿名物理 Plane 中恰好存在两个：

        7168 × 2048

    的槽位。

    因此每个 LogicalPlane 也恰好包含两个
    LogicalWeightCube。

    --------------------------------------------------------

    注意：

    logical_plane_id：

        只是第四步逻辑层自己的编号。

    它不是第三步产生的 physical plane_id。

    两者只有到后面的 Physical Binding 阶段
    才会建立对应关系。
    """

    logical_plane_id: int

    # 两个 LogicalWeightCube 的 cube_id
    cube_a_id: int
    cube_b_id: int

    # gate_down / up_up
    pair_kind: str

    def __post_init__(self) -> None:
        self.validate_basic()

    def validate_basic(self) -> None:

        if self.logical_plane_id < 0:
            raise LogicalPlaneError(
                "logical_plane_id 不能为负数。"
            )

        if self.cube_a_id < 0:
            raise LogicalPlaneError(
                "cube_a_id 不能为负数。"
            )

        if self.cube_b_id < 0:
            raise LogicalPlaneError(
                "cube_b_id 不能为负数。"
            )

        if self.cube_a_id == self.cube_b_id:
            raise LogicalPlaneError(
                "同一个 LogicalWeightCube "
                "不能同时占据 Plane 的两个槽位。"
            )

        if self.pair_kind not in PAIR_KINDS:
            raise LogicalPlaneError(
                f"非法 pair_kind={self.pair_kind!r}，"
                f"允许值为 {PAIR_KINDS}。"
            )

    @property
    def cube_ids(self) -> tuple[int, int]:
        """
        返回该 Plane 中的两个 LogicalWeightCube ID。
        """
        return (
            self.cube_a_id,
            self.cube_b_id,
        )

    @property
    def is_gate_down(self) -> bool:
        return (
            self.pair_kind
            == PAIR_GATE_DOWN
        )

    @property
    def is_up_up(self) -> bool:
        return (
            self.pair_kind
            == PAIR_UP_UP
        )

    def contains_cube(
        self,
        cube_id: int,
    ) -> bool:
        """
        判断某个 LogicalWeightCube
        是否位于当前 LogicalPlane。
        """
        return (
            cube_id == self.cube_a_id
            or cube_id == self.cube_b_id
        )

    def summary(self) -> str:
        return (
            f"LogicalPlane-{self.logical_plane_id}: "
            f"kind={self.pair_kind}, "
            f"cubes="
            f"({self.cube_a_id}, "
            f"{self.cube_b_id})"
        )


# ============================================================
# cube_id 索引
# ============================================================


def build_cube_index(
    cubes: Iterable[
        LogicalWeightCube
    ],
) -> dict[
    int,
    LogicalWeightCube,
]:
    """
    建立：

        cube_id -> LogicalWeightCube

    后面的 Plane 配对、Sub-Cube 映射、
    PhysicalSlot 绑定都会频繁使用该索引。
    """

    cube_index: dict[
        int,
        LogicalWeightCube,
    ] = {}

    for cube in cubes:

        if cube.cube_id in cube_index:
            raise LogicalPlaneError(
                f"发现重复 cube_id="
                f"{cube.cube_id}。"
            )

        cube_index[
            cube.cube_id
        ] = cube

    if not cube_index:
        raise LogicalPlaneError(
            "LogicalWeightCube 集合不能为空。"
        )

    return cube_index


# ============================================================
# 创建一张 gate/down Plane
# ============================================================


def create_gate_down_plane(
    *,
    logical_plane_id: int,
    gate: LogicalWeightCube,
    down: LogicalWeightCube,
) -> LogicalPlane:
    """
    将同一个 Expert 的：

        gate
        down

    组成：

        [ gate | down ]

    LogicalPlane。

    硬约束：

    1. gate 必须真的是 gate_proj；
    2. down 必须真的是 down_proj；
    3. 两者必须来自同一 Layer；
    4. 两者必须来自同一 Expert；
    5. is_shared 必须一致。

    --------------------------------------------------------

    为什么允许 gate 和 down 共 Plane？

    因为当前执行依赖：

        gate ----\
                  -> down
        up ------/

    gate 和 down 不在同一执行阶段，
    因此这种共址不会直接破坏
    gate/up 的前置并行。

    注意：

    这里只建立“共享 Plane”关系，
    还没有分配具体 Sub-Cube。
    """

    if gate.matrix_name != MATRIX_GATE:
        raise LogicalPlaneError(
            "create_gate_down_plane() "
            "第一个参数必须是 gate_proj，"
            f"实际为 {gate.matrix_name}。"
        )

    if down.matrix_name != MATRIX_DOWN:
        raise LogicalPlaneError(
            "create_gate_down_plane() "
            "第二个参数必须是 down_proj，"
            f"实际为 {down.matrix_name}。"
        )

    if gate.layer_id != down.layer_id:
        raise LogicalPlaneError(
            "gate 和 down 不属于同一 Layer："
            f"gate_layer={gate.layer_id}, "
            f"down_layer={down.layer_id}。"
        )

    if gate.expert_id != down.expert_id:
        raise LogicalPlaneError(
            "gate 和 down 不属于同一 Expert："
            f"gate_expert={gate.expert_id}, "
            f"down_expert={down.expert_id}。"
        )

    if gate.is_shared != down.is_shared:
        raise LogicalPlaneError(
            "gate 和 down 的 Shared 属性不一致。"
        )

    return LogicalPlane(
        logical_plane_id=logical_plane_id,

        cube_a_id=gate.cube_id,
        cube_b_id=down.cube_id,

        pair_kind=PAIR_GATE_DOWN,
    )


# ============================================================
# 创建一张 up/up Plane
# ============================================================


def create_up_up_plane(
    *,
    logical_plane_id: int,
    first_up: LogicalWeightCube,
    second_up: LogicalWeightCube,
) -> LogicalPlane:
    """
    将两个 up：

        [ up(e1) | up(e2) ]

    组成一个 LogicalPlane。

    当前这里只验证合法性。

    不负责决定：

        e1 应该和哪个 e2 配对。

    该决定属于后面的 plane_pairer.py，
    需要利用 Chinese-SimpleQA 的 Expert
    共激活统计。

    --------------------------------------------------------

    允许：

        同层 Routed + Routed

    也允许：

        不同层 Shared + Shared

    具体哪些组合真正采用，
    由后续策略决定。

    --------------------------------------------------------

    禁止：

        同一个 WeightCube 和自己配对。
    """

    if (
        first_up.matrix_name
        != MATRIX_UP
    ):
        raise LogicalPlaneError(
            "first_up 必须是 up_proj，"
            f"实际为 "
            f"{first_up.matrix_name}。"
        )

    if (
        second_up.matrix_name
        != MATRIX_UP
    ):
        raise LogicalPlaneError(
            "second_up 必须是 up_proj，"
            f"实际为 "
            f"{second_up.matrix_name}。"
        )

    if (
        first_up.cube_id
        == second_up.cube_id
    ):
        raise LogicalPlaneError(
            "同一个 up Weight-Cube "
            "不能与自己配对。"
        )

    return LogicalPlane(
        logical_plane_id=logical_plane_id,

        cube_a_id=first_up.cube_id,
        cube_b_id=second_up.cube_id,

        pair_kind=PAIR_UP_UP,
    )


# ============================================================
# 根据 LogicalPlane 获取真实矩阵
# ============================================================


def get_plane_cubes(
    *,
    plane: LogicalPlane,
    cube_index: dict[
        int,
        LogicalWeightCube,
    ],
) -> tuple[
    LogicalWeightCube,
    LogicalWeightCube,
]:
    """
    根据 LogicalPlane 中保存的 cube_id，
    返回两个真实 LogicalWeightCube。
    """

    try:
        cube_a = cube_index[
            plane.cube_a_id
        ]

        cube_b = cube_index[
            plane.cube_b_id
        ]

    except KeyError as exc:

        raise LogicalPlaneError(
            f"LogicalPlane-"
            f"{plane.logical_plane_id} "
            "引用了不存在的 cube_id。"
        ) from exc

    return (
        cube_a,
        cube_b,
    )


# ============================================================
# 单张 Plane 深度验证
# ============================================================


def validate_logical_plane(
    *,
    plane: LogicalPlane,
    cube_index: dict[
        int,
        LogicalWeightCube,
    ],
) -> None:
    """
    验证一张 LogicalPlane 的逻辑内容是否合法。
    """

    cube_a, cube_b = (
        get_plane_cubes(
            plane=plane,
            cube_index=cube_index,
        )
    )

    # ========================================================
    # gate_down
    # ========================================================

    if plane.pair_kind == PAIR_GATE_DOWN:

        names = {
            cube_a.matrix_name,
            cube_b.matrix_name,
        }

        if names != {
            MATRIX_GATE,
            MATRIX_DOWN,
        }:
            raise LogicalPlaneError(
                f"LogicalPlane-"
                f"{plane.logical_plane_id} "
                "标记为 gate_down，"
                "但实际矩阵不是 gate+down："
                f"{names}。"
            )

        if (
            cube_a.layer_id
            != cube_b.layer_id
        ):
            raise LogicalPlaneError(
                "gate_down Plane 中两个矩阵 "
                "不属于同一 Layer。"
            )

        if (
            cube_a.expert_id
            != cube_b.expert_id
        ):
            raise LogicalPlaneError(
                "gate_down Plane 中两个矩阵 "
                "不属于同一 Expert。"
            )

        if (
            cube_a.is_shared
            != cube_b.is_shared
        ):
            raise LogicalPlaneError(
                "gate_down Plane 中两个矩阵 "
                "Shared 属性不一致。"
            )

    # ========================================================
    # up_up
    # ========================================================

    elif plane.pair_kind == PAIR_UP_UP:

        if (
            cube_a.matrix_name
            != MATRIX_UP
            or cube_b.matrix_name
            != MATRIX_UP
        ):
            raise LogicalPlaneError(
                f"LogicalPlane-"
                f"{plane.logical_plane_id} "
                "标记为 up_up，"
                "但实际并非两个 up_proj。"
            )

    else:
        raise LogicalPlaneError(
            f"未知 pair_kind="
            f"{plane.pair_kind!r}。"
        )


# ============================================================
# Plane 集合验证
# ============================================================


def validate_logical_planes(
    *,
    planes: Iterable[
        LogicalPlane
    ],
    cubes: Iterable[
        LogicalWeightCube
    ],
    require_complete: bool = False,
) -> None:
    """
    验证一组 LogicalPlane。

    检查：

    1. logical_plane_id 唯一；
    2. 每张 Plane 结构合法；
    3. 一个 LogicalWeightCube 不能被多个 Plane 重复使用；
    4. 如果 require_complete=True：
       所有 LogicalWeightCube 必须全部恰好出现一次。

    --------------------------------------------------------

    为什么默认 require_complete=False？

    因为我们下一步会先生成：

        全部 gate_down Plane

    此时所有 up 还没有完成 trace-aware pairing。

    所以中间阶段允许只绑定一部分 WeightCube。

    等 plane_pairer.py 完成全部 up-up 配对以后，
    再使用：

        require_complete=True

    做最终检查。
    """

    plane_list = tuple(planes)
    cube_list = tuple(cubes)

    cube_index = build_cube_index(
        cube_list
    )

    # ========================================================
    # 1. Plane ID 唯一
    # ========================================================

    plane_ids = [
        plane.logical_plane_id
        for plane in plane_list
    ]

    if (
        len(plane_ids)
        != len(set(plane_ids))
    ):
        raise LogicalPlaneError(
            "存在重复 logical_plane_id。"
        )

    # ========================================================
    # 2. 单 Plane 验证
    # ========================================================

    for plane in plane_list:

        validate_logical_plane(
            plane=plane,
            cube_index=cube_index,
        )

    # ========================================================
    # 3. Cube 不可重复使用
    # ========================================================

    used_cube_ids: list[int] = []

    for plane in plane_list:

        used_cube_ids.extend(
            plane.cube_ids
        )

    if (
        len(used_cube_ids)
        != len(set(used_cube_ids))
    ):
        raise LogicalPlaneError(
            "一个 LogicalWeightCube "
            "被多个 LogicalPlane 重复使用。"
        )

    # ========================================================
    # 4. 是否要求完整覆盖
    # ========================================================

    if require_complete:

        all_cube_ids = {
            cube.cube_id
            for cube in cube_list
        }

        used_cube_id_set = set(
            used_cube_ids
        )

        if (
            used_cube_id_set
            != all_cube_ids
        ):

            missing = (
                all_cube_ids
                - used_cube_id_set
            )

            extra = (
                used_cube_id_set
                - all_cube_ids
            )

            raise LogicalPlaneError(
                "LogicalPlane 没有完整覆盖 "
                "全部 LogicalWeightCube："
                f"missing={len(missing)}, "
                f"extra={len(extra)}。"
            )


# ============================================================
# 查找未绑定 WeightCube
# ============================================================


def find_unassigned_cubes(
    *,
    planes: Iterable[
        LogicalPlane
    ],
    cubes: Iterable[
        LogicalWeightCube
    ],
) -> tuple[
    LogicalWeightCube,
    ...
]:
    """
    返回当前尚未进入任何 LogicalPlane 的
    LogicalWeightCube。

    例如：

    如果目前只生成 gate_down Plane，

    那么返回值应该正好是：

        全部 14906 个 up。
    """

    plane_list = tuple(planes)
    cube_list = tuple(cubes)

    used_cube_ids: set[int] = set()

    for plane in plane_list:

        used_cube_ids.update(
            plane.cube_ids
        )

    return tuple(
        cube
        for cube in cube_list
        if cube.cube_id
        not in used_cube_ids
    )


# ============================================================
# 统计
# ============================================================


def logical_plane_statistics(
    *,
    planes: Iterable[
        LogicalPlane
    ],
) -> dict[str, int]:
    """
    统计 LogicalPlane 类型数量。
    """

    plane_list = tuple(planes)

    gate_down_count = sum(
        1
        for plane in plane_list
        if plane.is_gate_down
    )

    up_up_count = sum(
        1
        for plane in plane_list
        if plane.is_up_up
    )

    return {
        "total_planes": (
            len(plane_list)
        ),

        "gate_down_planes": (
            gate_down_count
        ),

        "up_up_planes": (
            up_up_count
        ),

        "total_bound_cubes": (
            len(plane_list) * 2
        ),
    }


def print_logical_plane_summary(
    *,
    planes: Iterable[
        LogicalPlane
    ],
) -> None:
    """
    打印 LogicalPlane 统计。
    """

    stats = logical_plane_statistics(
        planes=planes
    )

    print(
        "========== Logical Planes =========="
    )

    print(
        f"Total Planes："
        f"{stats['total_planes']}"
    )

    print(
        f"gate+down Planes："
        f"{stats['gate_down_planes']}"
    )

    print(
        f"up+up Planes："
        f"{stats['up_up_planes']}"
    )

    print(
        f"Bound Weight-Cubes："
        f"{stats['total_bound_cubes']}"
    )