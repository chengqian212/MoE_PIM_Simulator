# evaluation/hardware_resolver.py
"""
根据匿名二维装箱结果 P，解析最终硬件深度 D。

第三步前半段已经完成：

    匿名矩形
        ↓
    H×W Plane 二维装箱
        ↓
    得到实际平面数量 P

本文件负责：

    P
        ↓
    枚举 N = 2, 3, 4
        ↓
    D = ceil(P / N^2)
        ↓
    Q = N^2 * D
        ↓
    C = Q * H * W
        ↓
    判断 S <= C <= 2S

其中：

    N^2：
        Sub-Cube 数量

    D：
        每个 Sub-Cube 必须统一配置的深度

    Q：
        整个硬件一共拥有多少个 H×W 平面槽位

    P：
        实际装箱真正用到的匿名 Plane 数量

注意：

1. D 是派生值，不是独立搜索变量；
2. 本文件不会把某个 Plane 分配给某个 Sub-Cube；
3. 不生成 subcube_id；
4. 不生成 z；
5. Plane → Sub-Cube 的具体映射留到第四步。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

from config import (
    MAX_N,
    MIN_N,
)


class HardwareResolutionError(ValueError):
    """硬件参数解析过程中出现非法状态时抛出的异常。"""


@dataclass(frozen=True, slots=True)
class ResolvedHardwareConfig:
    """
    一个已经由 P 解析出的完整硬件配置。

    与前面的 GeometryCandidate 不同：

        GeometryCandidate:
            N, H, W

    ResolvedHardwareConfig:
            N, H, W, D

    D 在这里才第一次真正确定。
    """

    N: int
    H: int
    W: int
    D: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """检查解析后的硬件参数是否合法。"""

        if not MIN_N <= self.N <= MAX_N:
            raise HardwareResolutionError(
                f"N 必须位于 [{MIN_N}, {MAX_N}]，"
                f"当前为 {self.N}。"
            )

        if self.H <= 0:
            raise HardwareResolutionError(
                f"H 必须大于 0，当前为 {self.H}。"
            )

        if self.W <= 0:
            raise HardwareResolutionError(
                f"W 必须大于 0，当前为 {self.W}。"
            )

        if self.D <= 0:
            raise HardwareResolutionError(
                f"D 必须大于 0，当前为 {self.D}。"
            )

    @property
    def num_subcubes(self) -> int:
        """
        Sub-Cube 总数量：

            N^2
        """
        return self.N * self.N

    @property
    def plane_area(self) -> int:
        """
        一个二维平面的面积：

            H * W
        """
        return self.H * self.W

    @property
    def total_plane_slots(self) -> int:
        """
        整个硬件一共具有多少个二维平面槽位：

            Q = N^2 * D
        """
        return (
            self.num_subcubes
            * self.D
        )

    @property
    def total_capacity(self) -> int:
        """
        整个硬件总容量：

            C = N^2 * D * H * W
        """
        return (
            self.total_plane_slots
            * self.plane_area
        )


@dataclass(frozen=True, slots=True)
class HardwareResolutionResult:
    """
    针对某个 N 得到的完整空间解析结果。

    除了硬件参数本身，还保存：

    - 实际需要的 Plane 数量 P；
    - 硬件总 Plane 槽位数量 Q；
    - 完整空 Plane 数量 Q-P；
    - 总容量 C；
    - 空间利用率 S/C；
    - 是否满足容量限制。
    """

    hardware: ResolvedHardwareConfig

    # 有效模型权重面积
    total_weight_area: int

    # 实际二维装箱使用的 Plane 数量
    used_plane_count: int

    # 是否满足 S <= C
    enough_capacity: bool

    # 是否满足 C <= 2S
    within_double_capacity: bool

    @property
    def valid(self) -> bool:
        """
        当前硬件配置是否合法。

        必须同时满足：

            S <= C <= 2S
        """
        return (
            self.enough_capacity
            and self.within_double_capacity
        )

    @property
    def total_plane_slots(self) -> int:
        """
        Q = N^2 * D
        """
        return self.hardware.total_plane_slots

    @property
    def empty_plane_slots(self) -> int:
        """
        因统一 D 产生的完整空平面数量：

            Q - P
        """

        return (
            self.total_plane_slots
            - self.used_plane_count
        )

    @property
    def total_capacity(self) -> int:
        """
        C = Q * H * W
        """
        return self.hardware.total_capacity

    @property
    def hardware_utilization(self) -> float:
        """
        整体硬件空间利用率：

            U = S / C

        与二维装箱利用率：

            S / (P*H*W)

        不同。

        前者同时考虑：

        1. Plane 内部二维碎片；
        2. Q > P 导致的完整空平面。
        """

        if self.total_capacity <= 0:
            return 0.0

        return (
            self.total_weight_area
            / self.total_capacity
        )

    @property
    def capacity_ratio(self) -> float:
        """
        硬件总容量相对于有效权重面积的倍数：

            C / S

        合法范围应为：

            1 <= C/S <= 2
        """

        if self.total_weight_area <= 0:
            return float("inf")

        return (
            self.total_capacity
            / self.total_weight_area
        )

    @property
    def empty_plane_capacity(self) -> int:
        """
        因 Q>P 产生的完整空 Plane 容量：

            (Q - P) * H * W
        """

        return (
            self.empty_plane_slots
            * self.hardware.plane_area
        )

    def summary(self) -> str:
        """返回简短结果摘要。"""

        return (
            f"N={self.hardware.N}, "
            f"SubCubes={self.hardware.num_subcubes}, "
            f"H={self.hardware.H}, "
            f"W={self.hardware.W}, "
            f"P={self.used_plane_count}, "
            f"D={self.hardware.D}, "
            f"Q={self.total_plane_slots}, "
            f"empty_planes={self.empty_plane_slots}, "
            f"C/S={self.capacity_ratio:.6f}, "
            f"util={self.hardware_utilization:.6%}, "
            f"valid={self.valid}"
        )


def calculate_required_depth(
    used_plane_count: int,
    N: int,
) -> int:
    """
    根据实际使用的 Plane 数 P 和 N 计算最小 D。

    公式：

        D = ceil(P / N^2)

    例如：

        P = 17

        N = 2:
            Sub-Cube 数 = 4
            D = ceil(17 / 4) = 5

        N = 3:
            Sub-Cube 数 = 9
            D = ceil(17 / 9) = 2

        N = 4:
            Sub-Cube 数 = 16
            D = ceil(17 / 16) = 2
    """

    if used_plane_count <= 0:
        raise HardwareResolutionError(
            "used_plane_count 必须大于 0。"
        )

    if not MIN_N <= N <= MAX_N:
        raise HardwareResolutionError(
            f"N 必须位于 [{MIN_N}, {MAX_N}]，"
            f"当前为 {N}。"
        )

    num_subcubes = N * N

    return ceil(
        used_plane_count
        / num_subcubes
    )


def resolve_hardware(
    N: int,
    H: int,
    W: int,
    used_plane_count: int,
    total_weight_area: int,
) -> HardwareResolutionResult:
    """
    根据一个固定 N、H、W 和实际 P，
    计算对应的最小合法深度 D。

    Args:
        N:
            每个方向上的 Sub-Cube 数量。

        H, W:
            单个二维 Plane 尺寸。

        used_plane_count:
            第三步二维装箱得到的实际平面数量 P。

        total_weight_area:
            有效模型权重总面积 S。

    Returns:
        HardwareResolutionResult
    """

    if H <= 0 or W <= 0:
        raise HardwareResolutionError(
            f"H、W 必须大于 0，当前为 {H}×{W}。"
        )

    if used_plane_count <= 0:
        raise HardwareResolutionError(
            f"P 必须大于 0，当前为 {used_plane_count}。"
        )

    if total_weight_area <= 0:
        raise HardwareResolutionError(
            "total_weight_area 必须大于 0。"
        )

    # ========================================================
    # 1. 被动计算 D
    # ========================================================

    D = calculate_required_depth(
        used_plane_count=used_plane_count,
        N=N,
    )

    # ========================================================
    # 2. 构造完整硬件配置
    # ========================================================

    hardware = ResolvedHardwareConfig(
        N=N,
        H=H,
        W=W,
        D=D,
    )

    # ========================================================
    # 3. 计算总容量 C
    # ========================================================

    total_capacity = (
        hardware.total_capacity
    )

    # ========================================================
    # 4. 检查容量
    # ========================================================

    enough_capacity = (
        total_capacity
        >= total_weight_area
    )

    within_double_capacity = (
        total_capacity
        <= 2 * total_weight_area
    )

    result = HardwareResolutionResult(
        hardware=hardware,

        total_weight_area=(
            total_weight_area
        ),

        used_plane_count=(
            used_plane_count
        ),

        enough_capacity=(
            enough_capacity
        ),

        within_double_capacity=(
            within_double_capacity
        ),
    )

    # ========================================================
    # 5. 逻辑检查
    # ========================================================

    if result.total_plane_slots < used_plane_count:
        raise HardwareResolutionError(
            "出现不可能状态："
            f"Q={result.total_plane_slots} "
            f"小于 P={used_plane_count}。"
        )

    return result


def resolve_all_n(
    H: int,
    W: int,
    used_plane_count: int,
    total_weight_area: int,
    n_values: Iterable[int] = (2, 3, 4),
) -> tuple[HardwareResolutionResult, ...]:
    """
    对同一个二维装箱结果 P，
    枚举多个 N。

    重要：

    对固定：

        H
        W
        PartitionTemplate

    二维装箱得到的：

        P

    与 N 无关。

    因此正确顺序是：

        先得到 P
            ↓
        再尝试 N=2、3、4
            ↓
        分别计算 D

    而不是为每个 N 重跑一次二维装箱。
    """

    results: list[
        HardwareResolutionResult
    ] = []

    seen_n: set[int] = set()

    for N in n_values:

        if N in seen_n:
            raise HardwareResolutionError(
                f"n_values 中存在重复 N={N}。"
            )

        seen_n.add(N)

        result = resolve_hardware(
            N=N,
            H=H,
            W=W,
            used_plane_count=(
                used_plane_count
            ),
            total_weight_area=(
                total_weight_area
            ),
        )

        results.append(result)

    return tuple(results)


def filter_valid_hardware(
    results: Iterable[
        HardwareResolutionResult
    ],
) -> tuple[
    HardwareResolutionResult,
    ...
]:
    """
    只保留满足：

        S <= C <= 2S

    的硬件配置。
    """

    return tuple(
        result
        for result in results
        if result.valid
    )


def sort_hardware_by_space_efficiency(
    results: Iterable[
        HardwareResolutionResult
    ],
) -> tuple[
    HardwareResolutionResult,
    ...
]:
    """
    按纯空间效率排列硬件候选。

    优先级：

    1. 合法配置优先；
    2. C 越小越好；
    3. 完整空 Plane 数越少越好；
    4. N 越小越优先作为确定性 tie-break；
    5. D 越小越优先。

    注意：

    这个排序只用于第三步空间结果展示。

    它不能直接决定最终最好方案。

    因为第四步和第五步还需要考虑推理周期。
    """

    return tuple(
        sorted(
            results,
            key=lambda result: (
                not result.valid,
                result.total_capacity,
                result.empty_plane_slots,
                result.hardware.N,
                result.hardware.D,
            ),
        )
    )


def print_hardware_resolution(
    results: Iterable[
        HardwareResolutionResult
    ],
) -> None:
    """
    打印 N=2、3、4 的硬件解析结果。
    """

    result_list = list(results)

    print(
        "========== Hardware Resolution =========="
    )

    for result in result_list:

        hardware = result.hardware

        print(
            f"\nN = {hardware.N}"
        )

        print(
            f"Sub-Cube 数量 N²："
            f"{hardware.num_subcubes}"
        )

        print(
            f"H×W："
            f"{hardware.H}×{hardware.W}"
        )

        print(
            f"实际使用 Plane 数 P："
            f"{result.used_plane_count}"
        )

        print(
            f"派生深度 D："
            f"{hardware.D}"
        )

        print(
            f"硬件总 Plane 槽位 Q："
            f"{result.total_plane_slots}"
        )

        print(
            f"完整空 Plane 数 Q-P："
            f"{result.empty_plane_slots}"
        )

        print(
            f"有效权重面积 S："
            f"{result.total_weight_area}"
        )

        print(
            f"硬件总容量 C："
            f"{result.total_capacity}"
        )

        print(
            f"C/S："
            f"{result.capacity_ratio:.6f}"
        )

        print(
            f"硬件空间利用率 S/C："
            f"{result.hardware_utilization:.6%}"
        )

        print(
            f"S <= C："
            f"{result.enough_capacity}"
        )

        print(
            f"C <= 2S："
            f"{result.within_double_capacity}"
        )

        print(
            f"最终合法："
            f"{result.valid}"
        )


if __name__ == "__main__":

    # ========================================================
    # 示例：
    #
    # 假设匿名二维装箱最终用了：
    #     P = 17
    #
    # Plane：
    #     4096 × 4096
    #
    # 这里只为了演示 D、Q 的计算。
    # ========================================================

    P = 17

    H = 4096
    W = 4096

    # 示例模型面积。
    #
    # 正式运行时应从：
    #
    #     ModelConfig.total_weight_area
    #
    # 获得。
    #
    # 这里为了让示例容量合法，
    # 使用 17 个 Plane 已使用容量中的一个测试值。
    S = (
        16
        * H
        * W
    )

    results = resolve_all_n(
        H=H,
        W=W,
        used_plane_count=P,
        total_weight_area=S,
    )

    print_hardware_resolution(
        results
    )

    print(
        "\n========== Sorted =========="
    )

    sorted_results = (
        sort_hardware_by_space_efficiency(
            results
        )
    )

    for result in sorted_results:
        print(
            result.summary()
        )