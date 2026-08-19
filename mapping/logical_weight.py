"""
第四步第一阶段：生成真实 LogicalWeightCube。

前三步中：

    gate / up / down
    layer_id
    expert_id

一直被隐藏，只进行匿名空间规划。

第四步开始恢复真实逻辑身份。

当前固定模型：

    58 个 MoE Layer

    每层：
        256 个 Routed Expert
        1 个 Shared Expert

    每个 Expert：
        gate_proj : 7168 × 2048
        up_proj   : 7168 × 2048
        down_proj : 2048 × 7168

当前空间配置：

    H = 7168
    W = 4096

因此每个完整矩阵都可以作为一个
depth=1 的 Weight-Cube：

    gate/up:
        7168 × 2048

    down:
        2048 × 7168
        物理放置时允许旋转为 7168 × 2048

注意：

本文件只恢复逻辑身份。

本阶段仍然不决定：

    plane_id
    slot_id
    subcube_id
    z
    x
    y
    rotated

这些属于第四步后面的 Plane 配对和 Sub-Cube 映射。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from config import ModelConfig
from model_geometry import SizeKey, make_size_key


# ============================================================
# 常量
# ============================================================


MATRIX_GATE = "gate_proj"
MATRIX_UP = "up_proj"
MATRIX_DOWN = "down_proj"

MATRIX_NAMES = (
    MATRIX_GATE,
    MATRIX_UP,
    MATRIX_DOWN,
)


class LogicalWeightError(ValueError):
    """LogicalWeightCube 构造或验证失败。"""


# ============================================================
# LogicalWeightCube
# ============================================================


@dataclass(frozen=True, slots=True)
class LogicalWeightCube:
    """
    一个真实的逻辑 Weight-Cube。

    当前 H=7168, W=4096 时，
    一个完整 gate/up/down 矩阵就是一个 Weight-Cube，
    因此当前没有继续切分。

    logical_rows / logical_cols：

        始终表示矩阵数学上的原始方向。

    例如：

        gate:
            7168 × 2048

        down:
            2048 × 7168

    即使以后 down 被旋转放进物理槽位，
    这里也不能把 logical_rows / logical_cols 改成
    7168 × 2048。

    物理旋转属于后续映射阶段。
    """

    # 全局唯一编号
    cube_id: int

    # MoE Layer
    layer_id: int

    # Expert 编号
    #
    # 0 ~ 255:
    #     Routed Expert
    #
    # 256:
    #     Shared Expert
    expert_id: int

    # 是否为 Shared Expert
    is_shared: bool

    # gate_proj / up_proj / down_proj
    matrix_name: str

    # 矩阵原始逻辑尺寸
    logical_rows: int
    logical_cols: int

    # 当前课题固定
    depth: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:

        if self.cube_id < 0:
            raise LogicalWeightError(
                f"cube_id 不能为负数，当前为 {self.cube_id}。"
            )

        if self.layer_id < 0:
            raise LogicalWeightError(
                f"layer_id 不能为负数，当前为 {self.layer_id}。"
            )

        if self.expert_id < 0:
            raise LogicalWeightError(
                f"expert_id 不能为负数，当前为 {self.expert_id}。"
            )

        if self.matrix_name not in MATRIX_NAMES:
            raise LogicalWeightError(
                "matrix_name 非法："
                f"{self.matrix_name!r}，"
                f"允许值为 {MATRIX_NAMES}。"
            )

        if self.logical_rows <= 0:
            raise LogicalWeightError(
                "logical_rows 必须大于 0。"
            )

        if self.logical_cols <= 0:
            raise LogicalWeightError(
                "logical_cols 必须大于 0。"
            )

        if self.depth != 1:
            raise LogicalWeightError(
                "当前课题中 Weight-Cube depth 必须恒为 1。"
            )

    @property
    def area(self) -> int:
        """有效权重面积。"""
        return (
            self.logical_rows
            * self.logical_cols
        )

    @property
    def size_key(self) -> SizeKey:
        """
        与方向无关的尺寸类型。

        gate/up:
            7168 × 2048

        down:
            2048 × 7168

        三者都会得到：

            (2048, 7168)

        因此后续都可以绑定到相同类型的匿名槽位。
        """
        return make_size_key(
            self.logical_rows,
            self.logical_cols,
        )

    @property
    def expert_key(self) -> tuple[int, int]:
        """
        唯一标识一个 Expert：

            (layer_id, expert_id)
        """
        return (
            self.layer_id,
            self.expert_id,
        )

    @property
    def logical_key(
        self,
    ) -> tuple[int, int, str]:
        """
        唯一标识一个真实矩阵：

            (layer_id, expert_id, matrix_name)
        """
        return (
            self.layer_id,
            self.expert_id,
            self.matrix_name,
        )

    @property
    def is_gate(self) -> bool:
        return self.matrix_name == MATRIX_GATE

    @property
    def is_up(self) -> bool:
        return self.matrix_name == MATRIX_UP

    @property
    def is_down(self) -> bool:
        return self.matrix_name == MATRIX_DOWN

    def summary(self) -> str:
        expert_type = (
            "shared"
            if self.is_shared
            else "routed"
        )

        return (
            f"LogicalWeightCube-{self.cube_id}: "
            f"layer={self.layer_id}, "
            f"expert={self.expert_id}, "
            f"type={expert_type}, "
            f"matrix={self.matrix_name}, "
            f"shape="
            f"{self.logical_rows}×{self.logical_cols}, "
            f"depth={self.depth}"
        )


# ============================================================
# 单个 Expert
# ============================================================


def build_expert_weight_cubes(
    *,
    layer_id: int,
    expert_id: int,
    is_shared: bool,
    hidden_size: int,
    intermediate_size: int,
    first_cube_id: int,
) -> tuple[LogicalWeightCube, ...]:
    """
    为一个 Expert 创建：

        gate
        up
        down

    三个 LogicalWeightCube。

    cube_id 顺序固定：

        gate
        up
        down

    这样可以保证完整程序完全可复现。
    """

    if layer_id < 0:
        raise LogicalWeightError(
            "layer_id 不能为负数。"
        )

    if expert_id < 0:
        raise LogicalWeightError(
            "expert_id 不能为负数。"
        )

    if hidden_size <= 0:
        raise LogicalWeightError(
            "hidden_size 必须大于 0。"
        )

    if intermediate_size <= 0:
        raise LogicalWeightError(
            "intermediate_size 必须大于 0。"
        )

    if first_cube_id < 0:
        raise LogicalWeightError(
            "first_cube_id 不能为负数。"
        )

    gate = LogicalWeightCube(
        cube_id=first_cube_id,
        layer_id=layer_id,
        expert_id=expert_id,
        is_shared=is_shared,
        matrix_name=MATRIX_GATE,

        logical_rows=hidden_size,
        logical_cols=intermediate_size,

        depth=1,
    )

    up = LogicalWeightCube(
        cube_id=first_cube_id + 1,
        layer_id=layer_id,
        expert_id=expert_id,
        is_shared=is_shared,
        matrix_name=MATRIX_UP,

        logical_rows=hidden_size,
        logical_cols=intermediate_size,

        depth=1,
    )

    down = LogicalWeightCube(
        cube_id=first_cube_id + 2,
        layer_id=layer_id,
        expert_id=expert_id,
        is_shared=is_shared,
        matrix_name=MATRIX_DOWN,

        logical_rows=intermediate_size,
        logical_cols=hidden_size,

        depth=1,
    )

    return (
        gate,
        up,
        down,
    )


# ============================================================
# 完整模型
# ============================================================


def build_logical_weight_cubes(
    config: ModelConfig,
) -> tuple[LogicalWeightCube, ...]:
    """
    根据 ModelConfig 构造完整模型的真实 LogicalWeightCube。

    当前正式方案必须开启 Shared Expert：

        每层：
            256 Routed
            + 1 Shared
            = 257 Expert

        总 Expert：
            58 × 257 = 14906

        总矩阵：
            14906 × 3 = 44718
    """

    config.validate()

    if not config.include_shared_expert:
        raise LogicalWeightError(
            "第四步当前方案要求启用 Shared Expert，"
            "请将 ModelConfig.include_shared_expert 设置为 True。"
        )

    cubes: list[LogicalWeightCube] = []

    next_cube_id = 0

    for layer_id in range(
        config.num_moe_layers
    ):

        # ====================================================
        # 1. Routed Experts
        # ====================================================

        for expert_id in range(
            config.routed_experts_per_layer
        ):

            expert_cubes = (
                build_expert_weight_cubes(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    is_shared=False,

                    hidden_size=(
                        config.hidden_size
                    ),

                    intermediate_size=(
                        config.expert_intermediate_size
                    ),

                    first_cube_id=(
                        next_cube_id
                    ),
                )
            )

            cubes.extend(
                expert_cubes
            )

            next_cube_id += 3

        # ====================================================
        # 2. Shared Expert
        #
        # 当前定义：
        #
        #   Routed Expert:
        #       0 ... 255
        #
        #   Shared Expert:
        #       256
        # ====================================================

        shared_expert_id = (
            config.routed_experts_per_layer
        )

        shared_cubes = (
            build_expert_weight_cubes(
                layer_id=layer_id,
                expert_id=shared_expert_id,
                is_shared=True,

                hidden_size=(
                    config.hidden_size
                ),

                intermediate_size=(
                    config.expert_intermediate_size
                ),

                first_cube_id=next_cube_id,
            )
        )

        cubes.extend(
            shared_cubes
        )

        next_cube_id += 3

    result = tuple(cubes)

    validate_logical_weight_cubes(
        cubes=result,
        config=config,
    )

    return result


# ============================================================
# 验证
# ============================================================


def validate_logical_weight_cubes(
    *,
    cubes: Iterable[LogicalWeightCube],
    config: ModelConfig,
) -> None:
    """
    对完整 LogicalWeightCube 集合进行严格检查。

    检查：

    1. 总数量正确；
    2. cube_id 唯一且连续；
    3. logical_key 唯一；
    4. 每层 Expert 数正确；
    5. 每个 Expert 恰好 gate/up/down 各一个；
    6. 每层恰好一个 Shared Expert；
    7. 矩阵尺寸正确；
    8. 所有 depth=1；
    9. 总面积与 ModelConfig 一致。
    """

    cube_list = tuple(cubes)

    # ========================================================
    # 1. 总数量
    # ========================================================

    expected_experts_per_layer = (
        config.routed_experts_per_layer
        + 1
    )

    expected_total_experts = (
        config.num_moe_layers
        * expected_experts_per_layer
    )

    expected_total_cubes = (
        expected_total_experts
        * 3
    )

    if len(cube_list) != expected_total_cubes:
        raise LogicalWeightError(
            "LogicalWeightCube 总数量错误："
            f"actual={len(cube_list)}, "
            f"expected={expected_total_cubes}。"
        )

    # ========================================================
    # 2. cube_id
    # ========================================================

    cube_ids = [
        cube.cube_id
        for cube in cube_list
    ]

    expected_ids = list(
        range(expected_total_cubes)
    )

    if cube_ids != expected_ids:
        raise LogicalWeightError(
            "cube_id 必须从 0 开始连续递增。"
        )

    # ========================================================
    # 3. logical_key 唯一
    # ========================================================

    logical_keys = [
        cube.logical_key
        for cube in cube_list
    ]

    if (
        len(logical_keys)
        != len(set(logical_keys))
    ):
        raise LogicalWeightError(
            "存在重复的真实矩阵 logical_key。"
        )

    # ========================================================
    # 4. 按 Expert 分组
    # ========================================================

    expert_to_cubes: dict[
        tuple[int, int],
        list[LogicalWeightCube],
    ] = {}

    for cube in cube_list:

        expert_to_cubes.setdefault(
            cube.expert_key,
            [],
        ).append(cube)

    if (
        len(expert_to_cubes)
        != expected_total_experts
    ):
        raise LogicalWeightError(
            "Expert 总数量错误："
            f"actual={len(expert_to_cubes)}, "
            f"expected={expected_total_experts}。"
        )

    # ========================================================
    # 5. 每个 Expert 必须恰好三个矩阵
    # ========================================================

    expected_matrix_names = set(
        MATRIX_NAMES
    )

    for expert_key, expert_cubes in (
        expert_to_cubes.items()
    ):

        if len(expert_cubes) != 3:
            raise LogicalWeightError(
                f"Expert {expert_key} "
                "不是恰好三个矩阵。"
            )

        actual_names = {
            cube.matrix_name
            for cube in expert_cubes
        }

        if actual_names != expected_matrix_names:
            raise LogicalWeightError(
                f"Expert {expert_key} "
                "没有完整的 gate/up/down："
                f"{actual_names}。"
            )

    # ========================================================
    # 6. 每层 Shared Expert 数量
    # ========================================================

    for layer_id in range(
        config.num_moe_layers
    ):

        layer_shared_expert_ids = {
            cube.expert_id
            for cube in cube_list
            if (
                cube.layer_id == layer_id
                and cube.is_shared
            )
        }

        expected_shared_id = (
            config.routed_experts_per_layer
        )

        if layer_shared_expert_ids != {
            expected_shared_id
        }:
            raise LogicalWeightError(
                f"Layer-{layer_id} "
                "Shared Expert 配置错误："
                f"{layer_shared_expert_ids}。"
            )

    # ========================================================
    # 7. 尺寸检查
    # ========================================================

    for cube in cube_list:

        if cube.matrix_name in (
            MATRIX_GATE,
            MATRIX_UP,
        ):

            expected_shape = (
                config.hidden_size,
                config.expert_intermediate_size,
            )

        else:

            expected_shape = (
                config.expert_intermediate_size,
                config.hidden_size,
            )

        actual_shape = (
            cube.logical_rows,
            cube.logical_cols,
        )

        if actual_shape != expected_shape:
            raise LogicalWeightError(
                f"{cube.logical_key} "
                "矩阵尺寸错误："
                f"actual={actual_shape}, "
                f"expected={expected_shape}。"
            )

    # ========================================================
    # 8. 总面积
    # ========================================================

    actual_total_area = sum(
        cube.area
        for cube in cube_list
    )

    expected_total_area = (
        expected_total_cubes
        * config.hidden_size
        * config.expert_intermediate_size
    )

    if actual_total_area != expected_total_area:
        raise LogicalWeightError(
            "LogicalWeightCube 总面积错误："
            f"actual={actual_total_area}, "
            f"expected={expected_total_area}。"
        )


# ============================================================
# 统计
# ============================================================


def logical_weight_statistics(
    cubes: Iterable[LogicalWeightCube],
) -> dict:
    """
    输出第四步第一阶段的基本统计。
    """

    cube_list = tuple(cubes)

    matrix_counter = Counter(
        cube.matrix_name
        for cube in cube_list
    )

    routed_count = sum(
        1
        for cube in cube_list
        if not cube.is_shared
    )

    shared_count = sum(
        1
        for cube in cube_list
        if cube.is_shared
    )

    expert_keys = {
        cube.expert_key
        for cube in cube_list
    }

    layer_ids = {
        cube.layer_id
        for cube in cube_list
    }

    return {
        "num_layers": len(layer_ids),
        "num_experts": len(expert_keys),

        "total_weight_cubes": (
            len(cube_list)
        ),

        "routed_weight_cubes": (
            routed_count
        ),

        "shared_weight_cubes": (
            shared_count
        ),

        "gate_count": (
            matrix_counter[MATRIX_GATE]
        ),

        "up_count": (
            matrix_counter[MATRIX_UP]
        ),

        "down_count": (
            matrix_counter[MATRIX_DOWN]
        ),

        "total_weight_area": sum(
            cube.area
            for cube in cube_list
        ),
    }


def print_logical_weight_summary(
    cubes: Iterable[LogicalWeightCube],
) -> None:
    """
    打印第四步第一阶段结果。
    """

    stats = logical_weight_statistics(
        cubes
    )

    print(
        "========== Logical Weight Cubes =========="
    )

    print(
        f"MoE Layers："
        f"{stats['num_layers']}"
    )

    print(
        f"Experts："
        f"{stats['num_experts']}"
    )

    print(
        f"Logical Weight-Cubes："
        f"{stats['total_weight_cubes']}"
    )

    print(
        f"Routed Weight-Cubes："
        f"{stats['routed_weight_cubes']}"
    )

    print(
        f"Shared Weight-Cubes："
        f"{stats['shared_weight_cubes']}"
    )

    print(
        f"gate：{stats['gate_count']}"
    )

    print(
        f"up：{stats['up_count']}"
    )

    print(
        f"down：{stats['down_count']}"
    )

    print(
        f"Total Weight Area："
        f"{stats['total_weight_area']}"
    )


# ============================================================
# 单文件测试运行
# ============================================================


if __name__ == "__main__":

    model = ModelConfig(
        include_shared_expert=True,
    )

    logical_cubes = (
        build_logical_weight_cubes(
            config=model,
        )
    )

    print_logical_weight_summary(
        logical_cubes
    )

    print(
        "\n前 6 个 LogicalWeightCube："
    )

    for cube in logical_cubes[:6]:
        print(
            cube.summary()
        )

    print(
        "\nLayer-0 Shared Expert："
    )

    shared_id = (
        model.routed_experts_per_layer
    )

    for cube in logical_cubes:
        if (
            cube.layer_id == 0
            and cube.expert_id == shared_id
        ):
            print(
                cube.summary()
            )