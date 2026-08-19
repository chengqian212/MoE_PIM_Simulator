"""
第五步：运行时静态索引。

第四步已经产生最终静态 Placement：

    layer_id
    expert_id
    matrix_name

    cube_id
    logical_plane_id
    physical_plane_id
    slot_id

    subcube_id
    z

第五步调度器没有必要每执行一个 token
都遍历 44718 条 Placement。

因此本文件把第四步结果整理成：

    RuntimeIndex
        ↓
    Layer
        ↓
    Expert
        ↓
    gate / up / down

之后可以直接：

    index.expert(layer_id, expert_id)

或者：

    index.matrix(
        layer_id,
        expert_id,
        "gate_proj",
    )

O(1) 查询其 WeightCube 和 Sub-Cube。

------------------------------------------------------------

当前运行模型：

58 个 MoE Layer

每层：

    Routed Expert 0~255
    Shared Expert 256

每个 Routed token：

    Router 选 Top-8

随后加入：

    Shared Expert 256

因此每个 token / layer 实际需要执行：

    9 gate
    9 up
    9 down

但本文件只建立索引。

真正计算：

    gate || up
          ↓
        down

以及周期、切换、并行等，

留给后面的 scheduler。
"""

from __future__ import annotations

import argparse
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


from config import (
    ModelConfig,
)

from mapping.logical_weight import (
    MATRIX_DOWN,
    MATRIX_GATE,
    MATRIX_UP,
)


# ============================================================
# 默认路径
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


DEFAULT_MAPPING_PATH = (
    PROJECT_ROOT
    / "results"
    / "mappings"
    / "mapping_baseline_N4_H7168_W4096.json"
)


# ============================================================
# 矩阵名称
# ============================================================


MATRIX_NAMES = (
    MATRIX_GATE,
    MATRIX_UP,
    MATRIX_DOWN,
)


# ============================================================
# 异常
# ============================================================


class RuntimeIndexError(ValueError):
    """构造或查询 RuntimeIndex 失败。"""


# ============================================================
# 单个 WeightCube 的运行时位置
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimeMatrixLocation:
    """
    一个真实 WeightCube 在运行时需要知道的信息。

    调度阶段最重要的是：

        cube_id
        subcube_id

    其余位置保留用于：

        调试
        切换统计
        可视化
        最终结果追踪
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
    # Logical / Physical Plane
    # ========================================================

    logical_plane_id: int

    physical_plane_id: int

    slot_id: int

    # ========================================================
    # 3D Hardware
    # ========================================================

    subcube_id: int

    z: int

    # ========================================================
    # 常用属性
    # ========================================================

    @property
    def weight_cube_id(
        self,
    ) -> int:
        """
        当前一个逻辑矩阵对应一个 WeightCube，
        因此 cube_id 就是运行时 WeightCube ID。
        """

        return self.cube_id

    @property
    def physical_plane_coordinate(
        self,
    ) -> tuple[int, int]:
        """
        返回：

            (subcube_id, z)
        """

        return (
            self.subcube_id,
            self.z,
        )


# ============================================================
# 一个 Expert 的三个矩阵
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimeExpertLocation:
    """
    一个 Expert 的完整静态位置。

    每个 Expert：

        gate
        up
        down
    """

    layer_id: int

    expert_id: int

    is_shared: bool

    gate: RuntimeMatrixLocation

    up: RuntimeMatrixLocation

    down: RuntimeMatrixLocation

    def __post_init__(
        self,
    ) -> None:

        # ====================================================
        # 三个矩阵必须属于同一个 Expert
        # ====================================================

        for matrix in (
            self.gate,
            self.up,
            self.down,
        ):

            if (
                matrix.layer_id
                != self.layer_id
                or
                matrix.expert_id
                != self.expert_id
                or
                matrix.is_shared
                != self.is_shared
            ):

                raise RuntimeIndexError(
                    f"Layer-{self.layer_id} "
                    f"Expert-{self.expert_id} "
                    "三个矩阵身份不一致。"
                )

        # ====================================================
        # Matrix name
        # ====================================================

        if (
            self.gate.matrix_name
            != MATRIX_GATE
        ):
            raise RuntimeIndexError(
                "gate 矩阵名称错误。"
            )

        if (
            self.up.matrix_name
            != MATRIX_UP
        ):
            raise RuntimeIndexError(
                "up 矩阵名称错误。"
            )

        if (
            self.down.matrix_name
            != MATRIX_DOWN
        ):
            raise RuntimeIndexError(
                "down 矩阵名称错误。"
            )

        # ====================================================
        # Step4 固定策略：
        #
        # gate + down 共 Plane
        # ====================================================

        if (
            self.gate.logical_plane_id
            != self.down.logical_plane_id
        ):

            raise RuntimeIndexError(
                f"Layer-{self.layer_id} "
                f"Expert-{self.expert_id} "
                "gate/down 没有共享 "
                "LogicalPlane。"
            )

        if (
            self.gate.physical_plane_id
            != self.down.physical_plane_id
        ):

            raise RuntimeIndexError(
                f"Layer-{self.layer_id} "
                f"Expert-{self.expert_id} "
                "gate/down 没有共享 "
                "PhysicalPlane。"
            )

        if (
            self.gate.subcube_id
            != self.down.subcube_id
            or
            self.gate.z
            != self.down.z
        ):

            raise RuntimeIndexError(
                f"Layer-{self.layer_id} "
                f"Expert-{self.expert_id} "
                "gate/down 没有共址。"
            )

        # ====================================================
        # Step4 硬约束：
        #
        # gate 和 up 必须不同 Sub-Cube
        # ====================================================

        if (
            self.gate.subcube_id
            == self.up.subcube_id
        ):

            raise RuntimeIndexError(
                f"Layer-{self.layer_id} "
                f"Expert-{self.expert_id} "
                "gate 和 up 位于同一个 "
                "Sub-Cube。"
            )

    # ========================================================
    # 查询
    # ========================================================

    def matrix(
        self,
        matrix_name: str,
    ) -> RuntimeMatrixLocation:

        if (
            matrix_name
            == MATRIX_GATE
        ):
            return self.gate

        if (
            matrix_name
            == MATRIX_UP
        ):
            return self.up

        if (
            matrix_name
            == MATRIX_DOWN
        ):
            return self.down

        raise RuntimeIndexError(
            "未知 matrix_name："
            f"{matrix_name!r}。"
        )

    # ========================================================
    # 常用 SC
    # ========================================================

    @property
    def gate_subcube(
        self,
    ) -> int:

        return (
            self.gate.subcube_id
        )

    @property
    def up_subcube(
        self,
    ) -> int:

        return (
            self.up.subcube_id
        )

    @property
    def down_subcube(
        self,
    ) -> int:

        return (
            self.down.subcube_id
        )


# ============================================================
# 单层索引
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimeLayerIndex:
    """
    一个 MoE Layer 的全部 Expert。

    experts 按 expert_id 排序，因此：

        experts[expert_id]

    可以 O(1) 查询。
    """

    layer_id: int

    experts: tuple[
        RuntimeExpertLocation,
        ...
    ]

    def expert(
        self,
        expert_id: int,
    ) -> RuntimeExpertLocation:

        if not (
            0
            <= expert_id
            < len(self.experts)
        ):

            raise RuntimeIndexError(
                f"Layer-{self.layer_id} "
                f"expert_id={expert_id} "
                "超出范围。"
            )

        expert = (
            self.experts[
                expert_id
            ]
        )

        if (
            expert.expert_id
            != expert_id
        ):

            raise RuntimeIndexError(
                "RuntimeLayerIndex.experts "
                "没有按照 expert_id 排序。"
            )

        return expert


# ============================================================
# 完整 RuntimeIndex
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimeIndex:
    """
    第五步调度器使用的静态索引。
    """

    model_config: ModelConfig

    num_subcubes: int

    subcube_depth: int

    layers: tuple[
        RuntimeLayerIndex,
        ...
    ]

    source_mapping: str | None = None

    # ========================================================
    # 基础属性
    # ========================================================

    @property
    def num_layers(
        self,
    ) -> int:

        return len(
            self.layers
        )

    @property
    def experts_per_layer(
        self,
    ) -> int:

        return (
            self.model_config
            .routed_experts_per_layer
            + int(
                self.model_config
                .include_shared_expert
            )
        )

    @property
    def shared_expert_id(
        self,
    ) -> int | None:

        if not (
            self.model_config
            .include_shared_expert
        ):

            return None

        return (
            self.model_config
            .routed_experts_per_layer
        )

    @property
    def total_experts(
        self,
    ) -> int:

        return (
            self.num_layers
            * self.experts_per_layer
        )

    @property
    def total_matrices(
        self,
    ) -> int:

        return (
            self.total_experts
            * 3
        )

    # ========================================================
    # Layer 查询
    # ========================================================

    def layer(
        self,
        layer_id: int,
    ) -> RuntimeLayerIndex:

        if not (
            0
            <= layer_id
            < len(self.layers)
        ):

            raise RuntimeIndexError(
                f"layer_id={layer_id} "
                "超出范围。"
            )

        layer = (
            self.layers[
                layer_id
            ]
        )

        if (
            layer.layer_id
            != layer_id
        ):

            raise RuntimeIndexError(
                "RuntimeIndex.layers "
                "没有按照 layer_id 排序。"
            )

        return layer

    # ========================================================
    # Expert 查询
    # ========================================================

    def expert(
        self,
        layer_id: int,
        expert_id: int,
    ) -> RuntimeExpertLocation:

        return (
            self.layer(
                layer_id
            )
            .expert(
                expert_id
            )
        )

    # ========================================================
    # Matrix 查询
    # ========================================================

    def matrix(
        self,
        layer_id: int,
        expert_id: int,
        matrix_name: str,
    ) -> RuntimeMatrixLocation:

        return (
            self.expert(
                layer_id,
                expert_id,
            )
            .matrix(
                matrix_name
            )
        )

    # ========================================================
    # Router Top-K + Shared Expert
    # ========================================================

    def resolve_active_expert_ids(
        self,
        *,
        layer_id: int,
        routed_expert_ids: Iterable[int],
    ) -> tuple[int, ...]:
        """
        输入 Router 选择出的 Routed Expert。

        例如：

            [247, 116, ..., 35]

        当前 DeepSeek-R1：

            恰好 8 个。

        输出：

            Routed Top-8
            +
            Shared Expert 256

        即共 9 个 Expert。

        注意：

        不排序 Routed Expert，
        保留 trace 原来的顺序。
        """

        # 顺便检查 layer_id
        self.layer(
            layer_id
        )

        routed = tuple(
            routed_expert_ids
        )

        expected_top_k = (
            self.model_config
            .experts_per_token
        )

        if (
            len(routed)
            != expected_top_k
        ):

            raise RuntimeIndexError(
                f"Router 必须选择 "
                f"{expected_top_k} 个 Expert，"
                f"实际为 {len(routed)}。"
            )

        # ====================================================
        # ID
        # ====================================================

        for expert_id in routed:

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

                raise RuntimeIndexError(
                    "Routed Expert ID "
                    "必须是整数。"
                )

            if not (
                0
                <= expert_id
                < self.model_config
                .routed_experts_per_layer
            ):

                raise RuntimeIndexError(
                    "Routed Expert ID="
                    f"{expert_id} "
                    "超出范围。"
                )

        # ====================================================
        # 不能重复
        # ====================================================

        if (
            len(
                set(routed)
            )
            != len(routed)
        ):

            raise RuntimeIndexError(
                "Router Top-K 中 "
                "存在重复 Expert。"
            )

        # ====================================================
        # Shared
        # ====================================================

        if (
            self.model_config
            .include_shared_expert
        ):

            shared_id = (
                self.shared_expert_id
            )

            if shared_id is None:
                raise RuntimeIndexError(
                    "Shared Expert ID "
                    "内部状态错误。"
                )

            return (
                routed
                + (
                    shared_id,
                )
            )

        return routed

    def resolve_active_experts(
        self,
        *,
        layer_id: int,
        routed_expert_ids: Iterable[int],
    ) -> tuple[
        RuntimeExpertLocation,
        ...
    ]:
        """
        resolve_active_expert_ids()
        的 Expert 对象版本。
        """

        expert_ids = (
            self.resolve_active_expert_ids(
                layer_id=layer_id,
                routed_expert_ids=(
                    routed_expert_ids
                ),
            )
        )

        return tuple(
            self.expert(
                layer_id,
                expert_id,
            )
            for expert_id
            in expert_ids
        )


# ============================================================
# JSON 基础检查
# ============================================================


def _require_int(
    data: dict[
        str,
        Any,
    ],
    key: str,
) -> int:

    value = data.get(
        key
    )

    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
    ):

        raise RuntimeIndexError(
            f"{key} 必须是整数。"
        )

    return value


def _require_bool(
    data: dict[
        str,
        Any,
    ],
    key: str,
) -> bool:

    value = data.get(
        key
    )

    if not isinstance(
        value,
        bool,
    ):

        raise RuntimeIndexError(
            f"{key} 必须是 bool。"
        )

    return value


# ============================================================
# Placement -> RuntimeMatrixLocation
# ============================================================


def _parse_placement(
    record: dict[
        str,
        Any,
    ],
) -> RuntimeMatrixLocation:

    matrix_name = (
        record.get(
            "matrix_name"
        )
    )

    if (
        matrix_name
        not in MATRIX_NAMES
    ):

        raise RuntimeIndexError(
            "非法 matrix_name："
            f"{matrix_name!r}。"
        )

    location = (
        RuntimeMatrixLocation(
            cube_id=_require_int(
                record,
                "cube_id",
            ),

            layer_id=_require_int(
                record,
                "layer_id",
            ),

            expert_id=_require_int(
                record,
                "expert_id",
            ),

            is_shared=_require_bool(
                record,
                "is_shared",
            ),

            matrix_name=(
                matrix_name
            ),

            logical_plane_id=(
                _require_int(
                    record,
                    "logical_plane_id",
                )
            ),

            physical_plane_id=(
                _require_int(
                    record,
                    "physical_plane_id",
                )
            ),

            slot_id=_require_int(
                record,
                "slot_id",
            ),

            subcube_id=(
                _require_int(
                    record,
                    "subcube_id",
                )
            ),

            z=_require_int(
                record,
                "z",
            ),
        )
    )

    # ========================================================
    # 非负
    # ========================================================

    integer_fields = (
        location.cube_id,
        location.layer_id,
        location.expert_id,
        location.logical_plane_id,
        location.physical_plane_id,
        location.slot_id,
        location.subcube_id,
        location.z,
    )

    if any(
        value < 0
        for value
        in integer_fields
    ):

        raise RuntimeIndexError(
            "Placement 中存在负数 ID。"
        )

    return location


# ============================================================
# LogicalPlane 全局验证
# ============================================================


def _validate_plane_groups(
    locations: tuple[
        RuntimeMatrixLocation,
        ...
    ],
) -> None:
    """
    验证第四步 Plane 结构没有被 JSON 破坏。

    每个 LogicalPlane 必须正好两个 WeightCube。

    只允许：

        gate + down

    或：

        up + up
    """

    groups: dict[
        int,
        list[
            RuntimeMatrixLocation
        ],
    ] = {}

    for location in locations:

        groups.setdefault(
            location.logical_plane_id,
            [],
        ).append(
            location
        )

    physical_plane_ids: set[int] = set()

    coordinates: set[
        tuple[int, int]
    ] = set()

    for (
        logical_plane_id,
        members,
    ) in groups.items():

        if len(members) != 2:

            raise RuntimeIndexError(
                f"LogicalPlane-"
                f"{logical_plane_id} "
                "不是恰好两个 WeightCube。"
            )

        first, second = members

        # ====================================================
        # 必须完全共址
        # ====================================================

        if (
            first.subcube_id
            != second.subcube_id
            or
            first.z
            != second.z
        ):

            raise RuntimeIndexError(
                f"LogicalPlane-"
                f"{logical_plane_id} "
                "两个 WeightCube "
                "没有位于同一个 SC/z。"
            )

        if (
            first.physical_plane_id
            != second.physical_plane_id
        ):

            raise RuntimeIndexError(
                f"LogicalPlane-"
                f"{logical_plane_id} "
                "对应了不同 PhysicalPlane。"
            )

        # ====================================================
        # 一个 PhysicalPlane
        # 只能对应一个 LogicalPlane
        # ====================================================

        physical_plane_id = (
            first.physical_plane_id
        )

        if (
            physical_plane_id
            in physical_plane_ids
        ):

            raise RuntimeIndexError(
                f"PhysicalPlane-"
                f"{physical_plane_id} "
                "被多个 LogicalPlane 使用。"
            )

        physical_plane_ids.add(
            physical_plane_id
        )

        # ====================================================
        # 一个 SC/z 只能有一张 Plane
        # ====================================================

        coordinate = (
            first.subcube_id,
            first.z,
        )

        if (
            coordinate
            in coordinates
        ):

            raise RuntimeIndexError(
                "重复 Plane 坐标："
                f"{coordinate}。"
            )

        coordinates.add(
            coordinate
        )

        # ====================================================
        # 两个 Slot 必须不同
        # ====================================================

        if (
            first.slot_id
            == second.slot_id
        ):

            raise RuntimeIndexError(
                f"LogicalPlane-"
                f"{logical_plane_id} "
                "两个 WeightCube "
                "使用同一个 Slot。"
            )

        names = {
            first.matrix_name,
            second.matrix_name,
        }

        # ====================================================
        # gate + down
        # ====================================================

        if names == {
            MATRIX_GATE,
            MATRIX_DOWN,
        }:

            if (
                first.layer_id
                != second.layer_id
                or
                first.expert_id
                != second.expert_id
            ):

                raise RuntimeIndexError(
                    f"LogicalPlane-"
                    f"{logical_plane_id} "
                    "gate/down 不属于同一个 Expert。"
                )

            continue

        # ====================================================
        # up + up
        # ====================================================

        if (
            first.matrix_name
            == MATRIX_UP
            and
            second.matrix_name
            == MATRIX_UP
        ):

            # Routed up + Routed up：
            # 必须同层。
            if (
                not first.is_shared
                and
                not second.is_shared
            ):

                if (
                    first.layer_id
                    != second.layer_id
                ):

                    raise RuntimeIndexError(
                        f"LogicalPlane-"
                        f"{logical_plane_id} "
                        "Routed up-up "
                        "来自不同 Layer。"
                    )

                continue

            # Shared up + Shared up：
            # 当前 Baseline 必须跨层。
            if (
                first.is_shared
                and
                second.is_shared
            ):

                if (
                    first.layer_id
                    == second.layer_id
                ):

                    raise RuntimeIndexError(
                        f"LogicalPlane-"
                        f"{logical_plane_id} "
                        "Shared up-up "
                        "来自同一 Layer。"
                    )

                continue

        raise RuntimeIndexError(
            f"LogicalPlane-"
            f"{logical_plane_id} "
            "出现非法矩阵组合。"
        )


# ============================================================
# Mapping Dict -> RuntimeIndex
# ============================================================


def build_runtime_index(
    mapping_data: dict[
        str,
        Any,
    ],
    *,
    model_config: (
        ModelConfig | None
    ) = None,
    source_mapping: (
        str | None
    ) = None,
) -> RuntimeIndex:
    """
    从第四步最终 JSON 内容构造 RuntimeIndex。
    """

    if not isinstance(
        mapping_data,
        dict,
    ):

        raise RuntimeIndexError(
            "Mapping 最外层必须是 dict。"
        )

    # ========================================================
    # Version
    # ========================================================

    if (
        mapping_data.get(
            "mapping_version"
        )
        != 1
    ):

        raise RuntimeIndexError(
            "当前只支持 "
            "mapping_version=1。"
        )

    # ========================================================
    # 当前最终 Baseline 默认包含 Shared Expert
    # ========================================================

    if model_config is None:

        model_config = (
            ModelConfig(
                include_shared_expert=True
            )
        )

    # ========================================================
    # Spatial
    # ========================================================

    spatial = (
        mapping_data.get(
            "spatial"
        )
    )

    if not isinstance(
        spatial,
        dict,
    ):

        raise RuntimeIndexError(
            "Mapping 缺少 spatial。"
        )

    num_subcubes = (
        _require_int(
            spatial,
            "num_subcubes",
        )
    )

    D = _require_int(
        spatial,
        "D",
    )

    if (
        num_subcubes <= 0
        or D <= 0
    ):

        raise RuntimeIndexError(
            "num_subcubes 和 D "
            "必须大于 0。"
        )

    # ========================================================
    # Model metadata
    # ========================================================

    model_data = (
        mapping_data.get(
            "model"
        )
    )

    if not isinstance(
        model_data,
        dict,
    ):

        raise RuntimeIndexError(
            "Mapping 缺少 model。"
        )

    saved_shared = (
        model_data.get(
            "shared_expert_enabled"
        )
    )

    if (
        saved_shared
        != model_config
        .include_shared_expert
    ):

        raise RuntimeIndexError(
            "Mapping 的 Shared Expert 配置 "
            "与 ModelConfig 不一致。"
        )

    # ========================================================
    # Placements
    # ========================================================

    raw_placements = (
        mapping_data.get(
            "placements"
        )
    )

    if not isinstance(
        raw_placements,
        list,
    ):

        raise RuntimeIndexError(
            "Mapping 缺少 placements list。"
        )

    if not raw_placements:

        raise RuntimeIndexError(
            "placements 为空。"
        )

    locations: list[
        RuntimeMatrixLocation
    ] = []

    for index, record in enumerate(
        raw_placements
    ):

        if not isinstance(
            record,
            dict,
        ):

            raise RuntimeIndexError(
                f"placements[{index}] "
                "必须是 dict。"
            )

        location = (
            _parse_placement(
                record
            )
        )

        # ====================================================
        # Hardware 范围
        # ====================================================

        if not (
            0
            <= location.subcube_id
            < num_subcubes
        ):

            raise RuntimeIndexError(
                f"Cube-{location.cube_id} "
                "subcube_id 超出范围。"
            )

        if not (
            0
            <= location.z
            < D
        ):

            raise RuntimeIndexError(
                f"Cube-{location.cube_id} "
                "z 超出 D 范围。"
            )

        locations.append(
            location
        )

    location_tuple = tuple(
        locations
    )

    # ========================================================
    # 数量
    # ========================================================

    experts_per_layer = (
        model_config
        .routed_experts_per_layer
        + int(
            model_config
            .include_shared_expert
        )
    )

    expected_experts = (
        model_config.num_moe_layers
        * experts_per_layer
    )

    expected_matrices = (
        expected_experts
        * 3
    )

    if (
        len(location_tuple)
        != expected_matrices
    ):

        raise RuntimeIndexError(
            "Placement 数量错误："
            f"actual="
            f"{len(location_tuple)}, "
            f"expected="
            f"{expected_matrices}。"
        )

    saved_cube_count = (
        model_data.get(
            "num_logical_weight_cubes"
        )
    )

    if (
        saved_cube_count
        is not None
        and
        saved_cube_count
        != len(location_tuple)
    ):

        raise RuntimeIndexError(
            "model.num_logical_weight_cubes "
            "与 placements 数量不一致。"
        )

    # ========================================================
    # Cube ID / Slot ID 唯一
    # ========================================================

    cube_ids = [
        location.cube_id
        for location
        in location_tuple
    ]

    if (
        len(cube_ids)
        != len(
            set(cube_ids)
        )
    ):

        raise RuntimeIndexError(
            "存在重复 cube_id。"
        )

    slot_ids = [
        location.slot_id
        for location
        in location_tuple
    ]

    if (
        len(slot_ids)
        != len(
            set(slot_ids)
        )
    ):

        raise RuntimeIndexError(
            "存在重复 slot_id。"
        )

    # ========================================================
    # Plane 结构
    # ========================================================

    _validate_plane_groups(
        location_tuple
    )

    # ========================================================
    # 按 Expert 分组
    # ========================================================

    expert_groups: dict[
        tuple[int, int],
        dict[
            str,
            RuntimeMatrixLocation,
        ],
    ] = {}

    for location in (
        location_tuple
    ):

        # ====================================================
        # Layer
        # ====================================================

        if not (
            0
            <= location.layer_id
            < model_config
            .num_moe_layers
        ):

            raise RuntimeIndexError(
                f"Cube-{location.cube_id} "
                "layer_id 超出范围。"
            )

        # ====================================================
        # Expert
        # ====================================================

        if not (
            0
            <= location.expert_id
            < experts_per_layer
        ):

            raise RuntimeIndexError(
                f"Cube-{location.cube_id} "
                "expert_id 超出范围。"
            )

        expected_shared = (
            model_config
            .include_shared_expert
            and
            location.expert_id
            == model_config
            .routed_experts_per_layer
        )

        if (
            location.is_shared
            != expected_shared
        ):

            raise RuntimeIndexError(
                f"Layer-{location.layer_id} "
                f"Expert-{location.expert_id} "
                "is_shared 标记错误。"
            )

        key = (
            location.layer_id,
            location.expert_id,
        )

        matrices = (
            expert_groups.setdefault(
                key,
                {},
            )
        )

        if (
            location.matrix_name
            in matrices
        ):

            raise RuntimeIndexError(
                f"Layer-{location.layer_id} "
                f"Expert-{location.expert_id} "
                f"存在重复 "
                f"{location.matrix_name}。"
            )

        matrices[
            location.matrix_name
        ] = location

    # ========================================================
    # 构造 Layer
    # ========================================================

    layers: list[
        RuntimeLayerIndex
    ] = []

    expected_matrix_names = {
        MATRIX_GATE,
        MATRIX_UP,
        MATRIX_DOWN,
    }

    for layer_id in range(
        model_config.num_moe_layers
    ):

        experts: list[
            RuntimeExpertLocation
        ] = []

        for expert_id in range(
            experts_per_layer
        ):

            key = (
                layer_id,
                expert_id,
            )

            if (
                key
                not in expert_groups
            ):

                raise RuntimeIndexError(
                    f"缺少 Layer-{layer_id} "
                    f"Expert-{expert_id}。"
                )

            matrices = (
                expert_groups[
                    key
                ]
            )

            if (
                set(matrices)
                != expected_matrix_names
            ):

                raise RuntimeIndexError(
                    f"Layer-{layer_id} "
                    f"Expert-{expert_id} "
                    "没有完整 gate/up/down。"
                )

            expert = (
                RuntimeExpertLocation(
                    layer_id=(
                        layer_id
                    ),

                    expert_id=(
                        expert_id
                    ),

                    is_shared=(
                        matrices[
                            MATRIX_GATE
                        ].is_shared
                    ),

                    gate=(
                        matrices[
                            MATRIX_GATE
                        ]
                    ),

                    up=(
                        matrices[
                            MATRIX_UP
                        ]
                    ),

                    down=(
                        matrices[
                            MATRIX_DOWN
                        ]
                    ),
                )
            )

            experts.append(
                expert
            )

        layers.append(
            RuntimeLayerIndex(
                layer_id=layer_id,

                experts=tuple(
                    experts
                ),
            )
        )

    return RuntimeIndex(
        model_config=(
            model_config
        ),

        num_subcubes=(
            num_subcubes
        ),

        subcube_depth=D,

        layers=tuple(
            layers
        ),

        source_mapping=(
            source_mapping
        ),
    )


# ============================================================
# JSON 文件加载
# ============================================================


def load_runtime_index(
    mapping_path: Path | str = (
        DEFAULT_MAPPING_PATH
    ),
    *,
    model_config: (
        ModelConfig | None
    ) = None,
) -> RuntimeIndex:
    """
    第五步最常用入口。

    直接读取：

        results/mappings/
        mapping_baseline_N4_H7168_W4096.json
    """

    path = Path(
        mapping_path
    ).resolve()

    if not path.exists():

        raise RuntimeIndexError(
            f"Mapping 文件不存在："
            f"{path}"
        )

    if not path.is_file():

        raise RuntimeIndexError(
            f"Mapping 路径不是文件："
            f"{path}"
        )

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

        raise RuntimeIndexError(
            f"无法读取 Mapping："
            f"{path}"
        ) from exc

    return build_runtime_index(
        data,

        model_config=(
            model_config
        ),

        source_mapping=str(
            path
        ),
    )


# ============================================================
# 输出
# ============================================================


def print_runtime_index_summary(
    index: RuntimeIndex,
) -> None:
    """
    打印第五步运行时索引摘要。
    """

    print(
        "\n"
        "========== Runtime Index =========="
    )

    print(
        f"MoE Layers："
        f"{index.num_layers}"
    )

    print(
        f"Experts / Layer："
        f"{index.experts_per_layer}"
    )

    print(
        f"Routed Experts / Layer："
        f"{index.model_config.routed_experts_per_layer}"
    )

    print(
        f"Shared Expert ID："
        f"{index.shared_expert_id}"
    )

    print(
        f"Total Experts："
        f"{index.total_experts}"
    )

    print(
        f"Total Weight-Cubes："
        f"{index.total_matrices}"
    )

    print(
        f"Sub-Cubes："
        f"{index.num_subcubes}"
    )

    print(
        f"Sub-Cube Depth："
        f"{index.subcube_depth}"
    )

    # ========================================================
    # 示例
    # ========================================================

    expert_0 = (
        index.expert(
            0,
            0,
        )
    )

    print(
        "\nExample：Layer-0 Expert-0"
    )

    print(
        f"  gate："
        f"Cube-{expert_0.gate.cube_id}, "
        f"SC-{expert_0.gate_subcube}"
    )

    print(
        f"  up："
        f"Cube-{expert_0.up.cube_id}, "
        f"SC-{expert_0.up_subcube}"
    )

    print(
        f"  down："
        f"Cube-{expert_0.down.cube_id}, "
        f"SC-{expert_0.down_subcube}"
    )

    if (
        index.shared_expert_id
        is not None
    ):

        shared = (
            index.expert(
                0,
                index.shared_expert_id,
            )
        )

        print(
            "\nExample：Layer-0 Shared Expert"
        )

        print(
            f"  gate："
            f"SC-{shared.gate_subcube}"
        )

        print(
            f"  up："
            f"SC-{shared.up_subcube}"
        )

        print(
            f"  down："
            f"SC-{shared.down_subcube}"
        )


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "读取第四步 Mapping，"
                "构造第五步 RuntimeIndex。"
            )
        )
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
    )

    args = (
        parser.parse_args()
    )

    index = (
        load_runtime_index(
            args.mapping
        )
    )

    print_runtime_index_summary(
        index
    )


if __name__ == "__main__":
    main()