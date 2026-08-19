# config.py
"""
项目基础配置。

本文件只定义：
1. 模型规模；
2. 候选硬件平面参数 N、H、W；
3. 固定执行规则。

注意：
- 前三步只进行匿名空间规划；
- 本文件不生成真实 Weight-Cube；
- D 不由用户直接设置，而是在第三步根据实际使用平面数 P 计算：
      D = ceil(P / N^2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ============================================================
# 题目规定的硬件参数范围
# ============================================================

MIN_N = 2
MAX_N = 4

MIN_H = 4096
MAX_H = 16384

MIN_W = 4096
MAX_W = 16384


class ConfigError(ValueError):
    """配置参数不合法时抛出的异常。"""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """
    MoE 模型的固定几何配置。

    前三步只关心矩阵的形状和数量，不区分具体矩阵属于：
    - 哪一层；
    - 哪一个 Expert；
    - gate_proj、up_proj 还是 down_proj。
    """

    # MoE 层数量
    num_moe_layers: int = 58

    # 每层 Routed Expert 数量
    routed_experts_per_layer: int = 256

    # 每个 token 在每层选择的 Routed Expert 数量
    experts_per_token: int = 8

    # 模型隐藏维度
    hidden_size: int = 7168

    # Expert 中间维度
    expert_intermediate_size: int = 2048

    # 每个 Expert 只考虑 gate、up、down 三个矩阵
    matrices_per_expert: int = 3

    # 当前 Baseline 默认不考虑 Shared Expert
    include_shared_expert: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """检查模型配置是否合法。"""

        if self.num_moe_layers <= 0:
            raise ConfigError("num_moe_layers 必须大于 0。")

        if self.routed_experts_per_layer <= 0:
            raise ConfigError(
                "routed_experts_per_layer 必须大于 0。"
            )

        if not (
            1
            <= self.experts_per_token
            <= self.routed_experts_per_layer
        ):
            raise ConfigError(
                "experts_per_token 必须位于 "
                "[1, routed_experts_per_layer] 范围内。"
            )

        if self.hidden_size <= 0:
            raise ConfigError("hidden_size 必须大于 0。")

        if self.expert_intermediate_size <= 0:
            raise ConfigError(
                "expert_intermediate_size 必须大于 0。"
            )

        if self.matrices_per_expert != 3:
            raise ConfigError(
                "当前课题只考虑 gate、up、down 三个矩阵，"
                "matrices_per_expert 必须为 3。"
            )

    @property
    def experts_per_layer(self) -> int:
        """
        当前实际计入部署的 Expert 数量。

        Baseline 默认只计算 Routed Expert。
        如果以后明确需要加入 Shared Expert，可以通过配置扩展。
        """
        return self.routed_experts_per_layer + int(
            self.include_shared_expert
        )

    @property
    def total_experts(self) -> int:
        """完整模型中的 Expert 总数。"""
        return self.num_moe_layers * self.experts_per_layer

    @property
    def total_matrices(self) -> int:
        """
        完整模型需要部署的矩阵总数。

        Baseline：
            58 × 256 × 3 = 44544
        """
        return self.total_experts * self.matrices_per_expert

    @property
    def canonical_matrix_rows(self) -> int:
        """
        前三步采用的标准匿名矩阵长边。

        gate、up 的形状为 7168×2048；
        down 的形状为 2048×7168。

        因为允许物理旋转，所以纯空间阶段统一使用：
            7168×2048
        """
        return max(
            self.hidden_size,
            self.expert_intermediate_size,
        )

    @property
    def canonical_matrix_cols(self) -> int:
        """前三步采用的标准匿名矩阵短边。"""
        return min(
            self.hidden_size,
            self.expert_intermediate_size,
        )

    @property
    def area_per_matrix(self) -> int:
        """单个矩阵的有效权重面积。"""
        return (
            self.hidden_size
            * self.expert_intermediate_size
        )

    @property
    def total_weight_area(self) -> int:
        """
        全部有效权重的总面积 S。

        当前不考虑：
        - 参数位宽；
        - 补零；
        - 权重复制；
        - Shared Expert；
        - 其他模型层。
        """
        return self.total_matrices * self.area_per_matrix


@dataclass(frozen=True, slots=True)
class GeometryCandidate:
    """
    前三步使用的候选硬件几何参数。

    注意：
    - N 表示 Global Cube 每个方向上的 Sub-Cube 数量；
    - Sub-Cube 总数为 N^2；
    - H、W 是每个深度层的二维平面尺寸；
    - 此处不包含 D；
    - D 需要第三步得到实际平面数 P 后再计算。
    """

    N: int
    H: int
    W: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """检查候选硬件参数是否满足题目范围。"""

        if not MIN_N <= self.N <= MAX_N:
            raise ConfigError(
                f"N 必须位于 [{MIN_N}, {MAX_N}]，"
                f"当前为 {self.N}。"
            )

        if not MIN_H <= self.H <= MAX_H:
            raise ConfigError(
                f"H 必须位于 [{MIN_H}, {MAX_H}]，"
                f"当前为 {self.H}。"
            )

        if not MIN_W <= self.W <= MAX_W:
            raise ConfigError(
                f"W 必须位于 [{MIN_W}, {MAX_W}]，"
                f"当前为 {self.W}。"
            )

    @property
    def num_subcubes(self) -> int:
        """Sub-Cube 总数。"""
        return self.N * self.N

    @property
    def plane_area(self) -> int:
        """每个 H×W 平面的面积。"""
        return self.H * self.W


@dataclass(frozen=True, slots=True)
class ExecutionRules:
    """
    推理阶段使用的固定执行规则。

    前三步暂时不执行调度，但提前统一规则，
    避免第四、第五步采用不一致的计算口径。
    """

    # 已确认每个 Weight-Cube 的深度恒为 1
    weight_cube_depth: int = 1

    # 一个深度为 1 的 Weight-Cube 计算耗时
    compute_cycles: int = 1

    # 同一 Sub-Cube 切换到另一个 Weight-Cube 的开销
    switch_cycles: int = 1

    # 跨 Sub-Cube 不产生额外通信或切换开销
    cross_subcube_cycles: int = 0

    # 不同 Sub-Cube 可以同时计算
    unlimited_parallel_subcubes: bool = True

    # 同一 Sub-Cube 同一时刻只能执行一个 Weight-Cube
    one_active_weight_cube_per_subcube: bool = True

    # Weight-Cube 允许旋转 90°
    allow_rotation: bool = True

    # 部署完成后权重位置不允许在推理时迁移
    weight_stationary: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """检查执行规则是否符合当前已确认设定。"""

        if self.weight_cube_depth != 1:
            raise ConfigError(
                "当前课题已确认 Weight-Cube 的深度必须为 1。"
            )

        if self.compute_cycles != 1:
            raise ConfigError(
                "当前 Baseline 中单个 Weight-Cube 的计算耗时应为 1 周期。"
            )

        if self.switch_cycles != 1:
            raise ConfigError(
                "当前规则中切换 Weight-Cube 的耗时应为 1 周期。"
            )

        if self.cross_subcube_cycles != 0:
            raise ConfigError(
                "当前规则中跨 Sub-Cube 不产生额外开销。"
            )

        if not self.unlimited_parallel_subcubes:
            raise ConfigError(
                "当前规则不限制可并行工作的 Sub-Cube 数量。"
            )

        if not self.one_active_weight_cube_per_subcube:
            raise ConfigError(
                "同一 Sub-Cube 同一时刻只能执行一个 Weight-Cube。"
            )

        if not self.allow_rotation:
            raise ConfigError(
                "当前规则允许 Weight-Cube 旋转 90° 放置。"
            )

        if not self.weight_stationary:
            raise ConfigError(
                "当前课题要求部署后采用 Weight Stationary。"
            )


def build_default_geometry_candidates() -> tuple[GeometryCandidate, ...]:
    """
    构造第一批建议测试的 N、H、W 候选配置。

    这些只是实验候选，不代表最终最优参数。

    H、W 不要求相等，因此保留与标准矩阵 7168×2048
    几何形状较匹配的 7168×4096 配置。
    """

    plane_shapes = (
        (4096, 4096),
        (7168, 4096),
        (8192, 4096),
        (8192, 8192),
        (16384, 4096),
        (16384, 8192),
        (16384, 16384),
    )

    candidates: list[GeometryCandidate] = []

    for n in range(MIN_N, MAX_N + 1):
        for h, w in plane_shapes:
            candidates.append(
                GeometryCandidate(
                    N=n,
                    H=h,
                    W=w,
                )
            )

    return tuple(candidates)


def validate_unique_candidates(
    candidates: Iterable[GeometryCandidate],
) -> None:
    """
    检查候选配置是否重复。

    Raises:
        ConfigError: 存在重复的 N、H、W 组合。
    """

    seen: set[tuple[int, int, int]] = set()

    for candidate in candidates:
        key = (
            candidate.N,
            candidate.H,
            candidate.W,
        )

        if key in seen:
            raise ConfigError(
                f"发现重复硬件候选配置：{key}。"
            )

        seen.add(key)


def print_config_summary(
    model: ModelConfig,
    rules: ExecutionRules,
) -> None:
    """打印第一步的配置摘要。"""

    print("========== Model Configuration ==========")
    print(f"MoE 层数：{model.num_moe_layers}")
    print(
        "每层 Routed Expert 数："
        f"{model.routed_experts_per_layer}"
    )
    print(f"是否包含 Shared Expert：{model.include_shared_expert}")
    print(f"矩阵总数：{model.total_matrices}")
    print(
        "标准匿名矩阵形状："
        f"{model.canonical_matrix_rows}"
        f" × {model.canonical_matrix_cols}"
    )
    print(f"单矩阵面积：{model.area_per_matrix}")
    print(f"有效权重总面积 S：{model.total_weight_area}")

    print("\n========== Execution Rules ==========")
    print(
        "Weight-Cube depth："
        f"{rules.weight_cube_depth}"
    )
    print(f"计算周期：{rules.compute_cycles}")
    print(f"切换周期：{rules.switch_cycles}")
    print(
        "跨 Sub-Cube 开销："
        f"{rules.cross_subcube_cycles}"
    )
    print(f"允许旋转：{rules.allow_rotation}")
    print(f"Weight Stationary：{rules.weight_stationary}")


if __name__ == "__main__":
    model_config = ModelConfig()
    execution_rules = ExecutionRules()

    geometry_candidates = build_default_geometry_candidates()
    validate_unique_candidates(geometry_candidates)

    print_config_summary(
        model=model_config,
        rules=execution_rules,
    )

    print("\n========== Geometry Candidates ==========")
    for candidate in geometry_candidates:
        print(
            f"N={candidate.N}, "
            f"H={candidate.H}, "
            f"W={candidate.W}, "
            f"Sub-Cubes={candidate.num_subcubes}, "
            f"Plane Area={candidate.plane_area}"
        )
        