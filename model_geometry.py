# model_geometry.py
"""
将模型配置转换为匿名矩阵几何需求。

本文件只描述：
1. 标准匿名矩阵的形状；
2. 标准匿名矩阵的数量；
3. 有效权重总面积；
4. 某种矩阵形状是否能直接放入候选 H×W 平面。

注意：
- 不生成真实 Weight-Cube；
- 不保存 layer_id、expert_id、matrix_name；
- gate、up、down 在纯空间阶段被统一看作等价的匿名矩阵；
- down 的 2048×7168 可以通过旋转与 7168×2048 统一。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from config import GeometryCandidate, ModelConfig


class GeometryError(ValueError):
    """模型几何信息不合法时抛出的异常。"""


SizeKey = tuple[int, int]


def make_size_key(rows: int, cols: int) -> SizeKey:
    """
    生成与方向无关的尺寸标识。

    例如：
        7168×2048
        2048×7168

    都得到：
        (2048, 7168)

    Args:
        rows: 矩形第一维。
        cols: 矩形第二维。

    Returns:
        与方向无关的尺寸元组。
    """

    if rows <= 0 or cols <= 0:
        raise GeometryError(
            f"矩形尺寸必须为正数，当前为 {rows}×{cols}。"
        )

    return min(rows, cols), max(rows, cols)


@dataclass(frozen=True, slots=True)
class MatrixShapeDemand:
    """
    一类匿名矩阵的几何需求。

    该对象只表示：
        有 count 个 rows×cols 的匿名矩阵需要存储。

    它不表示这些矩阵属于哪个层、哪个 Expert，
    也不记录它们是 gate、up 还是 down。
    """

    rows: int
    cols: int
    count: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """检查匿名矩阵需求是否合法。"""

        if self.rows <= 0:
            raise GeometryError(
                f"rows 必须大于 0，当前为 {self.rows}。"
            )

        if self.cols <= 0:
            raise GeometryError(
                f"cols 必须大于 0，当前为 {self.cols}。"
            )

        if self.count <= 0:
            raise GeometryError(
                f"count 必须大于 0，当前为 {self.count}。"
            )

    @property
    def size_key(self) -> SizeKey:
        """返回与方向无关的矩阵尺寸标识。"""
        return make_size_key(self.rows, self.cols)

    @property
    def area_per_matrix(self) -> int:
        """单个矩阵的有效面积。"""
        return self.rows * self.cols

    @property
    def total_area(self) -> int:
        """该类所有匿名矩阵的有效总面积。"""
        return self.area_per_matrix * self.count

    def can_fit_without_partition(
        self,
        H: int,
        W: int,
        allow_rotation: bool = True,
    ) -> bool:
        """
        判断整个矩阵是否可以不切分地放入一个 H×W 平面。

        这里只判断几何尺寸，不进行实际装箱。

        Args:
            H: 平面第一维。
            W: 平面第二维。
            allow_rotation: 是否允许将矩形旋转 90°。

        Returns:
            至少存在一个合法方向时返回 True。
        """

        if H <= 0 or W <= 0:
            raise GeometryError(
                f"平面尺寸必须为正数，当前为 {H}×{W}。"
            )

        normal_fit = self.rows <= H and self.cols <= W

        rotated_fit = (
            allow_rotation
            and self.cols <= H
            and self.rows <= W
        )

        return normal_fit or rotated_fit

    def minimum_area_planes(self, H: int, W: int) -> int:
        """
        仅按照面积计算该类矩阵所需平面数下界。

        注意：
        这只是面积下界，不代表矩形一定可以达到该装箱结果。
        """

        if H <= 0 or W <= 0:
            raise GeometryError(
                f"平面尺寸必须为正数，当前为 {H}×{W}。"
            )

        plane_area = H * W

        return (
            self.total_area + plane_area - 1
        ) // plane_area


@dataclass(frozen=True, slots=True)
class ModelGeometry:
    """
    模型在纯空间阶段的匿名几何摘要。
    """

    demands: tuple[MatrixShapeDemand, ...]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """检查几何需求列表是否合法。"""

        if not self.demands:
            raise GeometryError(
                "ModelGeometry 至少需要一个矩阵形状需求。"
            )

        seen: set[SizeKey] = set()

        for demand in self.demands:
            demand.validate()

            if demand.size_key in seen:
                raise GeometryError(
                    "ModelGeometry 中存在未合并的重复形状："
                    f"{demand.size_key}。"
                )

            seen.add(demand.size_key)

    @property
    def total_matrix_count(self) -> int:
        """全部匿名矩阵数量。"""
        return sum(demand.count for demand in self.demands)

    @property
    def total_weight_area(self) -> int:
        """全部匿名矩阵的有效权重总面积。"""
        return sum(demand.total_area for demand in self.demands)

    @property
    def shape_count(self) -> int:
        """匿名矩阵形状类别数量。"""
        return len(self.demands)

    def get_demand_by_size_key(
        self,
        size_key: SizeKey,
    ) -> MatrixShapeDemand:
        """
        根据无方向尺寸查找矩阵需求。
        """

        normalized_key = make_size_key(*size_key)

        for demand in self.demands:
            if demand.size_key == normalized_key:
                return demand

        raise GeometryError(
            f"没有找到尺寸类型 {normalized_key}。"
        )


def merge_shape_demands(
    demands: Iterable[MatrixShapeDemand],
) -> tuple[MatrixShapeDemand, ...]:
    """
    合并方向等价的匿名矩阵需求。

    例如：
        7168×2048，数量 2
        2048×7168，数量 1

    合并为：
        7168×2048，数量 3

    输出统一采用：
        rows = 较长边
        cols = 较短边
    """

    merged_counts: dict[SizeKey, int] = {}

    for demand in demands:
        demand.validate()

        key = demand.size_key
        merged_counts[key] = (
            merged_counts.get(key, 0)
            + demand.count
        )

    merged: list[MatrixShapeDemand] = []

    for short_side, long_side in sorted(merged_counts):
        merged.append(
            MatrixShapeDemand(
                rows=long_side,
                cols=short_side,
                count=merged_counts[
                    (short_side, long_side)
                ],
            )
        )

    return tuple(merged)


def build_matrix_shape_demands(
    config: ModelConfig,
) -> tuple[MatrixShapeDemand, ...]:
    """
    根据模型配置生成匿名矩阵几何需求。

    当前每个 Expert 有三个面积相同的矩阵：

        7168×2048
        7168×2048
        2048×7168

    因为纯空间阶段允许旋转，因此将三者合并为：

        7168×2048，数量 = Expert 总数 × 3

    Args:
        config: 模型配置。

    Returns:
        合并后的匿名矩阵需求。
    """

    config.validate()

    raw_demands = (
        MatrixShapeDemand(
            rows=config.hidden_size,
            cols=config.expert_intermediate_size,
            count=config.total_experts * 2,
        ),
        MatrixShapeDemand(
            rows=config.expert_intermediate_size,
            cols=config.hidden_size,
            count=config.total_experts,
        ),
    )

    demands = merge_shape_demands(raw_demands)

    validate_demands_against_model(
        config=config,
        demands=demands,
    )

    return demands


def build_model_geometry(
    config: ModelConfig,
) -> ModelGeometry:
    """
    构造完整匿名模型几何摘要。
    """

    return ModelGeometry(
        demands=build_matrix_shape_demands(config)
    )


def validate_demands_against_model(
    config: ModelConfig,
    demands: Iterable[MatrixShapeDemand],
) -> None:
    """
    检查匿名矩阵需求是否与模型配置一致。

    验证内容：
    1. 匿名矩阵总数正确；
    2. 有效权重总面积正确；
    3. 当前 Baseline 只有一个方向归一化后的形状类型。
    """

    demand_list = tuple(demands)

    total_count = sum(
        demand.count
        for demand in demand_list
    )

    total_area = sum(
        demand.total_area
        for demand in demand_list
    )

    if total_count != config.total_matrices:
        raise GeometryError(
            "匿名矩阵总数与模型配置不一致："
            f"需求中为 {total_count}，"
            f"模型应为 {config.total_matrices}。"
        )

    if total_area != config.total_weight_area:
        raise GeometryError(
            "匿名矩阵总面积与模型配置不一致："
            f"需求中为 {total_area}，"
            f"模型应为 {config.total_weight_area}。"
        )

    expected_key = make_size_key(
        config.hidden_size,
        config.expert_intermediate_size,
    )

    for demand in demand_list:
        if demand.size_key != expected_key:
            raise GeometryError(
                "当前 Baseline 出现了非预期矩阵形状："
                f"{demand.rows}×{demand.cols}。"
            )


def analyze_geometry_candidate(
    geometry: ModelGeometry,
    candidate: GeometryCandidate,
    allow_rotation: bool = True,
) -> dict[str, int | bool]:
    """
    对一个候选 H、W 做简单几何分析。

    这里只判断：
    - 整个矩阵能否直接放入单平面；
    - 纯面积平面数下界。

    不执行矩阵切分，也不执行 MaxRects 装箱。
    """

    if geometry.shape_count != 1:
        raise GeometryError(
            "当前分析函数暂时要求只有一种匿名矩阵形状。"
        )

    demand = geometry.demands[0]

    plane_area = candidate.plane_area

    area_lower_bound = (
        geometry.total_weight_area
        + plane_area
        - 1
    ) // plane_area

    return {
        "N": candidate.N,
        "H": candidate.H,
        "W": candidate.W,
        "num_subcubes": candidate.num_subcubes,
        "matrix_fits_without_partition": (
            demand.can_fit_without_partition(
                H=candidate.H,
                W=candidate.W,
                allow_rotation=allow_rotation,
            )
        ),
        "area_plane_lower_bound": area_lower_bound,
    }


def print_geometry_summary(
    geometry: ModelGeometry,
) -> None:
    """打印匿名模型几何摘要。"""

    print("========== Anonymous Model Geometry ==========")
    print(f"形状类别数量：{geometry.shape_count}")
    print(f"匿名矩阵总数：{geometry.total_matrix_count}")
    print(f"有效权重总面积 S：{geometry.total_weight_area}")

    for index, demand in enumerate(geometry.demands):
        print(f"\n形状需求 {index}:")
        print(
            f"  标准方向："
            f"{demand.rows} × {demand.cols}"
        )
        print(f"  size_key：{demand.size_key}")
        print(f"  数量：{demand.count}")
        print(f"  单矩阵面积：{demand.area_per_matrix}")
        print(f"  总面积：{demand.total_area}")


if __name__ == "__main__":
    model_config = ModelConfig()

    model_geometry = build_model_geometry(
        config=model_config
    )

    print_geometry_summary(model_geometry)

    example_candidates = (
        GeometryCandidate(
            N=2,
            H=4096,
            W=4096,
        ),
        GeometryCandidate(
            N=2,
            H=7168,
            W=4096,
        ),
        GeometryCandidate(
            N=3,
            H=4096,
            W=8192,
        ),
    )

    print("\n========== Candidate Analysis ==========")

    for candidate in example_candidates:
        result = analyze_geometry_candidate(
            geometry=model_geometry,
            candidate=candidate,
        )

        print(
            f"N={result['N']}, "
            f"H={result['H']}, "
            f"W={result['W']}, "
            f"无需切分={result['matrix_fits_without_partition']}, "
            f"面积平面下界={result['area_plane_lower_bound']}"
        )