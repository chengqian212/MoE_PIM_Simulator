# evaluation/spatial_metrics.py
"""
第三步纯空间指标统计与合法性检查。

输入：

1. PackingResult
   - 第三步二维装箱结果；
   - 包含 P 个匿名 Plane；
   - 包含全部匿名 PhysicalSlot。

2. HardwareResolutionResult
   - 根据 P、N、H、W 计算出的 D、Q、C。

本文件负责计算：

    S
    P_lower
    P

    Plane 内部二维碎片：
        P*H*W - S

    完整空 Plane 数：
        Q - P

    完整空 Plane 容量：
        (Q-P)*H*W

    总空闲容量：
        C - S

    二维装箱利用率：
        S / (P*H*W)

    整体硬件利用率：
        S / C

并执行最终空间合法性检查。

注意：
- 本文件仍然属于第三步；
- 不出现 layer_id；
- 不出现 expert_id；
- 不出现 matrix_name；
- 不出现 subcube_id；
- 不出现 z；
- 不涉及真实 Weight-Cube；
- 不涉及推理周期。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil

from model_geometry import SizeKey

from packing.anonymous_packer import (
    PackingResult,
)

from evaluation.hardware_resolver import (
    HardwareResolutionResult,
)


class SpatialMetricsError(ValueError):
    """空间指标计算或验证失败时抛出的异常。"""


@dataclass(frozen=True, slots=True)
class SpatialMetrics:
    """
    一个完整第三步空间候选的评价结果。

    一个候选由：

        H
        W
        PartitionTemplate
        PackingResult
        N
        D

    共同确定。
    """

    template_id: str

    N: int
    H: int
    W: int
    D: int

    # 有效模型权重面积
    total_weight_area: int

    # 理论面积下界
    plane_lower_bound: int

    # 实际二维装箱使用平面数
    used_plane_count: int

    # 硬件总平面槽位数
    total_plane_slots: int

    # Plane 内二维碎片
    internal_fragmentation: int

    # 因统一 D 产生的完整空 Plane 数
    empty_plane_slots: int

    # 完整空 Plane 对应的容量
    empty_plane_capacity: int

    # 总硬件空闲容量
    total_unused_capacity: int

    # 二维装箱利用率
    packing_utilization: float

    # 整体硬件空间利用率
    hardware_utilization: float

    # 总容量
    total_capacity: int

    # C / S
    capacity_ratio: float

    # 空间结果是否合法
    valid: bool

    def summary(self) -> str:
        """返回简洁摘要。"""

        return (
            f"SpatialMetrics<{self.template_id}>: "
            f"N={self.N}, "
            f"H={self.H}, "
            f"W={self.W}, "
            f"D={self.D}, "
            f"P_lower={self.plane_lower_bound}, "
            f"P={self.used_plane_count}, "
            f"Q={self.total_plane_slots}, "
            f"packing_util="
            f"{self.packing_utilization:.6%}, "
            f"hardware_util="
            f"{self.hardware_utilization:.6%}, "
            f"C/S={self.capacity_ratio:.6f}, "
            f"valid={self.valid}"
        )


def calculate_plane_lower_bound(
    total_weight_area: int,
    H: int,
    W: int,
) -> int:
    """
    根据面积计算理论最少 Plane 数。

        P_lower = ceil(S / (H*W))

    注意：

    这是纯面积下界。

    它没有考虑矩形几何形状，因此通常：

        P >= P_lower
    """

    if total_weight_area <= 0:
        raise SpatialMetricsError(
            "total_weight_area 必须大于 0。"
        )

    if H <= 0 or W <= 0:
        raise SpatialMetricsError(
            f"H、W 必须大于 0，当前为 {H}×{W}。"
        )

    return ceil(
        total_weight_area
        / (H * W)
    )


def validate_packing_result(
    packing: PackingResult,
) -> None:
    """
    检查匿名二维装箱结果本身是否合法。

    验证：

    1. Plane 数量与 plane_count 一致；
    2. Slot 数量正确；
    3. 所有 Plane 尺寸都等于 H×W；
    4. 所有 Plane 内槽位不越界；
    5. 所有 Plane 内槽位不重叠；
    6. slot_id 唯一；
    7. plane_id 连续；
    8. slot.plane_id 与所属 Plane 一致；
    9. 总 Slot 面积等于 S。
    """

    if packing.plane_count <= 0:
        raise SpatialMetricsError(
            "PackingResult 至少应该包含一个 Plane。"
        )

    # ========================================================
    # 1. plane_id 检查
    # ========================================================

    expected_plane_ids = set(
        range(packing.plane_count)
    )

    actual_plane_ids = {
        plane.plane_id
        for plane in packing.planes
    }

    if actual_plane_ids != expected_plane_ids:
        raise SpatialMetricsError(
            "Plane ID 不连续或存在重复："
            f"expected={expected_plane_ids}, "
            f"actual={actual_plane_ids}。"
        )

    # ========================================================
    # 2. slot_id 唯一性检查
    # ========================================================

    slot_ids = [
        slot.slot_id
        for slot in packing.slots
    ]

    if len(slot_ids) != len(set(slot_ids)):
        raise SpatialMetricsError(
            "PhysicalSlot 中存在重复 slot_id。"
        )

    # ========================================================
    # 3. 每个 Plane 的布局检查
    # ========================================================

    slots_seen_from_planes = []

    for plane in packing.planes:

        if (
            plane.H != packing.H
            or plane.W != packing.W
        ):
            raise SpatialMetricsError(
                f"Plane-{plane.plane_id} 尺寸 "
                f"{plane.H}×{plane.W} "
                "与 PackingResult 的 "
                f"{packing.H}×{packing.W} 不一致。"
            )

        plane.validate_layout()

        for slot in plane.slots:

            if slot.plane_id != plane.plane_id:
                raise SpatialMetricsError(
                    f"Slot-{slot.slot_id} 的 plane_id="
                    f"{slot.plane_id}，"
                    f"但实际位于 Plane-{plane.plane_id}。"
                )

            slots_seen_from_planes.append(
                slot.slot_id
            )

    # ========================================================
    # 4. PackingResult.slots 与 Plane.slots 一致性
    # ========================================================

    if set(slots_seen_from_planes) != set(slot_ids):
        raise SpatialMetricsError(
            "PackingResult.slots 与各 Plane 中保存的槽位集合不一致。"
        )

    # ========================================================
    # 5. Slot 总面积检查
    # ========================================================

    total_slot_area = sum(
        slot.area
        for slot in packing.slots
    )

    if total_slot_area != packing.total_block_area:
        raise SpatialMetricsError(
            "PhysicalSlot 总面积与 total_block_area 不一致："
            f"slot_area={total_slot_area}, "
            f"block_area={packing.total_block_area}。"
        )

    # ========================================================
    # 6. P×H×W 必须至少能装下 S
    # ========================================================

    used_plane_capacity = (
        packing.plane_count
        * packing.H
        * packing.W
    )

    if used_plane_capacity < packing.total_block_area:
        raise SpatialMetricsError(
            "出现不可能状态："
            f"P×H×W={used_plane_capacity} "
            f"小于 S={packing.total_block_area}。"
        )


def validate_slot_histogram(
    packing: PackingResult,
    expected_histogram: dict[SizeKey, int],
) -> None:
    """
    检查各尺寸 PhysicalSlot 数量是否与第二步模板需求一致。

    例如第二步要求：

        (2048,4096): 44544
        (2048,3072): 44544

    第三步最终也必须严格得到相同数量。

    因为第三步只能改变：

        位置
        物理方向

    不能改变块的数量和尺寸类型。
    """

    actual_histogram = dict(
        Counter(
            slot.size_key
            for slot in packing.slots
        )
    )

    if actual_histogram != expected_histogram:
        raise SpatialMetricsError(
            "PhysicalSlot 尺寸统计与第二步匿名需求不一致。\n"
            f"expected={expected_histogram}\n"
            f"actual={actual_histogram}"
        )


def validate_hardware_against_packing(
    packing: PackingResult,
    hardware_result: HardwareResolutionResult,
) -> None:
    """
    检查 HardwareResolutionResult 是否确实对应当前 PackingResult。

    重点验证：

        H、W 一致；
        P 一致；
        Q >= P；
        D = ceil(P/N²)；
        C = N²*D*H*W。
    """

    hardware = hardware_result.hardware

    # ========================================================
    # 1. H、W 必须一致
    # ========================================================

    if (
        hardware.H != packing.H
        or hardware.W != packing.W
    ):
        raise SpatialMetricsError(
            "Hardware 与 Packing 的 H、W 不一致："
            f"packing={packing.H}×{packing.W}, "
            f"hardware={hardware.H}×{hardware.W}。"
        )

    # ========================================================
    # 2. P 必须一致
    # ========================================================

    if (
        hardware_result.used_plane_count
        != packing.plane_count
    ):
        raise SpatialMetricsError(
            "HardwareResolutionResult 中的 P "
            "与 PackingResult 不一致："
            f"hardware_P="
            f"{hardware_result.used_plane_count}, "
            f"packing_P={packing.plane_count}。"
        )

    # ========================================================
    # 3. S 必须一致
    # ========================================================

    if (
        hardware_result.total_weight_area
        != packing.total_block_area
    ):
        raise SpatialMetricsError(
            "硬件解析使用的 S 与匿名装箱使用的 S 不一致："
            f"hardware_S="
            f"{hardware_result.total_weight_area}, "
            f"packing_S={packing.total_block_area}。"
        )

    # ========================================================
    # 4. D 必须等于 ceil(P/N²)
    # ========================================================

    expected_D = ceil(
        packing.plane_count
        / hardware.num_subcubes
    )

    if hardware.D != expected_D:
        raise SpatialMetricsError(
            "D 计算错误："
            f"actual_D={hardware.D}, "
            f"expected_D={expected_D}。"
        )

    # ========================================================
    # 5. Q >= P
    # ========================================================

    if (
        hardware.total_plane_slots
        < packing.plane_count
    ):
        raise SpatialMetricsError(
            "硬件总 Plane 槽位 Q 小于实际 Plane 数 P。"
        )

    # ========================================================
    # 6. C 公式检查
    # ========================================================

    expected_capacity = (
        hardware.N
        * hardware.N
        * hardware.D
        * hardware.H
        * hardware.W
    )

    if (
        hardware.total_capacity
        != expected_capacity
    ):
        raise SpatialMetricsError(
            "硬件容量计算错误："
            f"actual={hardware.total_capacity}, "
            f"expected={expected_capacity}。"
        )


def evaluate_spatial_metrics(
    packing: PackingResult,
    hardware_result: HardwareResolutionResult,
) -> SpatialMetrics:
    """
    计算一个完整第三步候选的空间指标。

    ----------------------------------------------------------
    空间浪费分成两部分：

    1. 已使用 P 个 Plane 内的二维碎片：

        F_inside = P*H*W - S

    2. 因统一 D 导致 Q>P 的完整空 Plane：

        F_empty = (Q-P)*H*W

    因此总空闲容量：

        C-S
        =
        (P*H*W-S)
        +
        (Q-P)*H*W

    即：

        C-S
        =
        internal_fragmentation
        +
        empty_plane_capacity
    ----------------------------------------------------------
    """

    # ========================================================
    # 1. 先执行一致性检查
    # ========================================================

    validate_packing_result(
        packing
    )

    validate_hardware_against_packing(
        packing=packing,
        hardware_result=hardware_result,
    )

    hardware = hardware_result.hardware

    S = packing.total_block_area
    P = packing.plane_count

    Q = hardware.total_plane_slots
    C = hardware.total_capacity

    # ========================================================
    # 2. 理论面积下界
    # ========================================================

    P_lower = calculate_plane_lower_bound(
        total_weight_area=S,
        H=packing.H,
        W=packing.W,
    )

    if P < P_lower:
        raise SpatialMetricsError(
            "出现不可能状态："
            f"实际 P={P} 小于面积理论下界 "
            f"P_lower={P_lower}。"
        )

    # ========================================================
    # 3. Plane 内部二维碎片
    # ========================================================

    internal_fragmentation = (
        P
        * packing.H
        * packing.W
        - S
    )

    if internal_fragmentation < 0:
        raise SpatialMetricsError(
            "internal_fragmentation 不应为负数。"
        )

    # ========================================================
    # 4. 完整空 Plane
    # ========================================================

    empty_plane_slots = Q - P

    if empty_plane_slots < 0:
        raise SpatialMetricsError(
            "Q-P 不应为负数。"
        )

    empty_plane_capacity = (
        empty_plane_slots
        * packing.H
        * packing.W
    )

    # ========================================================
    # 5. 总空闲容量
    # ========================================================

    total_unused_capacity = (
        C - S
    )

    if total_unused_capacity < 0:
        raise SpatialMetricsError(
            "总容量 C 小于有效权重面积 S。"
        )

    # ========================================================
    # 6. 验证空间浪费分解公式
    #
    # C-S
    # =
    # (P*H*W-S)
    # +
    # (Q-P)*H*W
    # ========================================================

    decomposed_unused_capacity = (
        internal_fragmentation
        + empty_plane_capacity
    )

    if (
        total_unused_capacity
        != decomposed_unused_capacity
    ):
        raise SpatialMetricsError(
            "空间浪费分解公式不成立："
            f"C-S={total_unused_capacity}, "
            "internal_fragmentation + "
            "empty_plane_capacity="
            f"{decomposed_unused_capacity}。"
        )

    # ========================================================
    # 7. 利用率
    # ========================================================

    packing_utilization = (
        S
        / (
            P
            * packing.H
            * packing.W
        )
    )

    hardware_utilization = (
        S / C
    )

    capacity_ratio = (
        C / S
    )

    # ========================================================
    # 8. 合法性
    # ========================================================

    valid = (
        hardware_result.valid
        and S <= C
        and C <= 2 * S
        and P >= P_lower
    )

    # ========================================================
    # 9. 返回
    # ========================================================

    return SpatialMetrics(
        template_id=packing.template_id,

        N=hardware.N,
        H=hardware.H,
        W=hardware.W,
        D=hardware.D,

        total_weight_area=S,

        plane_lower_bound=P_lower,
        used_plane_count=P,
        total_plane_slots=Q,

        internal_fragmentation=(
            internal_fragmentation
        ),

        empty_plane_slots=(
            empty_plane_slots
        ),

        empty_plane_capacity=(
            empty_plane_capacity
        ),

        total_unused_capacity=(
            total_unused_capacity
        ),

        packing_utilization=(
            packing_utilization
        ),

        hardware_utilization=(
            hardware_utilization
        ),

        total_capacity=C,

        capacity_ratio=(
            capacity_ratio
        ),

        valid=valid,
    )


def print_spatial_metrics(
    metrics: SpatialMetrics,
) -> None:
    """
    打印一个空间候选的完整指标。
    """

    print(
        "========== Spatial Metrics =========="
    )

    print(
        f"template_id：{metrics.template_id}"
    )

    print(
        f"N：{metrics.N}"
    )

    print(
        f"Sub-Cube 数量：{metrics.N ** 2}"
    )

    print(
        f"H×W：{metrics.H}×{metrics.W}"
    )

    print(
        f"D：{metrics.D}"
    )

    print()

    print(
        f"有效权重面积 S："
        f"{metrics.total_weight_area}"
    )

    print(
        f"理论 Plane 下界 P_lower："
        f"{metrics.plane_lower_bound}"
    )

    print(
        f"实际 Plane 数 P："
        f"{metrics.used_plane_count}"
    )

    print(
        f"总 Plane 槽位数 Q："
        f"{metrics.total_plane_slots}"
    )

    print()

    print(
        "Plane 内部二维碎片："
        f"{metrics.internal_fragmentation}"
    )

    print(
        "完整空 Plane 数 Q-P："
        f"{metrics.empty_plane_slots}"
    )

    print(
        "完整空 Plane 容量："
        f"{metrics.empty_plane_capacity}"
    )

    print(
        "总空闲容量 C-S："
        f"{metrics.total_unused_capacity}"
    )

    print()

    print(
        "二维装箱利用率："
        f"{metrics.packing_utilization:.6%}"
    )

    print(
        "整体硬件利用率："
        f"{metrics.hardware_utilization:.6%}"
    )

    print(
        f"硬件总容量 C："
        f"{metrics.total_capacity}"
    )

    print(
        f"C/S："
        f"{metrics.capacity_ratio:.6f}"
    )

    print()

    print(
        f"容量是否合法：{metrics.valid}"
    )


def compare_spatial_metrics(
    metrics_list: list[SpatialMetrics],
) -> list[SpatialMetrics]:
    """
    对多个第三步空间候选进行纯空间排序。

    排序优先级：

    1. 合法方案优先；
    2. 总容量 C 越小越好；
    3. 实际 Plane 数 P 越少越好；
    4. Plane 内二维碎片越少越好；
    5. 完整空 Plane 数越少越好；
    6. N 越小作为确定性 tie-break；
    7. template_id 保证结果稳定。

    注意：

    这个排序只代表“空间上更优”。

    不代表最终推理性能更优。

    第四步之后还需要根据真实调度周期进行综合比较。
    """

    return sorted(
        metrics_list,
        key=lambda item: (
            not item.valid,
            item.total_capacity,
            item.used_plane_count,
            item.internal_fragmentation,
            item.empty_plane_slots,
            item.N,
            item.template_id,
        ),
    )


if __name__ == "__main__":

    from config import ModelConfig

    from partition.partition_generator import (
        generate_partition_templates,
    )

    from partition.partition_validator import (
        validate_partition_templates,
    )

    from packing.anonymous_packer import (
        pack_anonymous_blocks,
    )

    from evaluation.hardware_resolver import (
        resolve_all_n,
    )

    # ========================================================
    # 小规模功能测试
    #
    # 为了避免直接跑 44544 个矩阵，
    # 先用 8 个匿名矩阵。
    # ========================================================

    model = ModelConfig()

    matrix_rows = (
        model.canonical_matrix_rows
    )

    matrix_cols = (
        model.canonical_matrix_cols
    )

    H = 4096
    W = 4096

    matrix_count = 8

    templates = generate_partition_templates(
        matrix_rows=matrix_rows,
        matrix_cols=matrix_cols,
        H=H,
        W=W,
    )

    validate_partition_templates(
        templates=templates,
        H=H,
        W=W,
        allow_rotation=True,
    )

    all_metrics: list[
        SpatialMetrics
    ] = []

    for template in templates:

        packing = pack_anonymous_blocks(
            template=template,
            matrix_count=matrix_count,
            H=H,
            W=W,
            allow_rotation=True,
        )

        hardware_results = resolve_all_n(
            H=H,
            W=W,
            used_plane_count=(
                packing.plane_count
            ),
            total_weight_area=(
                packing.total_block_area
            ),
        )

        for hardware_result in hardware_results:

            metrics = evaluate_spatial_metrics(
                packing=packing,
                hardware_result=hardware_result,
            )

            all_metrics.append(metrics)

            print()
            print_spatial_metrics(
                metrics
            )

    # ========================================================
    # 按纯空间效果排序
    # ========================================================

    print(
        "\n"
        "========== Spatial Ranking =========="
    )

    ranked = compare_spatial_metrics(
        all_metrics
    )

    for rank, metrics in enumerate(
        ranked,
        start=1,
    ):
        print(
            f"{rank}. {metrics.summary()}"
        )