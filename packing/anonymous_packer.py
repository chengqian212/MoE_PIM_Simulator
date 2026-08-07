# packing/anonymous_packer.py
"""
整个模型的匿名矩形二维装箱。

本文件负责：

1. 将 PartitionTemplate 扩展成完整模型的匿名矩形需求；
2. 按“面积降序 + 最长边降序 + 匿名编号”排序；
3. 对所有已有 Plane 搜索 MaxRects-BSSF 最佳位置；
4. 如果所有已有 Plane 都放不下，则创建新的 H×W Plane；
5. 为每个匿名矩形生成 PhysicalSlot；
6. 最终得到实际使用的二维平面数量 P。

注意：
- 当前仍然属于第三步；
- 不出现 layer_id；
- 不出现 expert_id；
- 不出现 matrix_name；
- 不出现 gate / up / down；
- 不出现 subcube_id；
- 不出现 z；
- 这里生成的只是匿名 PhysicalSlot。

第四步才会把真实逻辑 Weight-Cube 绑定到这些槽位。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from model_geometry import SizeKey, make_size_key

from partition.partition_template import (
    PartitionTemplate,
)

from packing.physical_slot import (
    PhysicalSlot,
)

from packing.plane import (
    Plane,
    create_empty_plane,
)

from packing.maxrects import (
    PlacementCandidate,
    commit_placement,
    find_best_position,
    validate_free_rectangles,
)


class AnonymousPackingError(ValueError):
    """匿名矩形装箱失败时抛出的异常。"""


# ============================================================
# 匿名矩形
# ============================================================


@dataclass(frozen=True, slots=True)
class AnonymousBlock:
    """
    第三步使用的一个匿名矩形实例。

    例如：

        anonymous_block_id = 0
        rows = 4096
        cols = 2048

    它只说明：

        “整个模型中有这样一个矩形需要存储。”

    不知道这个矩形将来属于：
        - 哪一层；
        - 哪个 Expert；
        - gate / up / down 中的哪一个。

    source_template_chunk_id 只表示它来自第二步模板中的哪类块，
    不是实际矩阵的逻辑 chunk_id。
    """

    anonymous_block_id: int

    rows: int
    cols: int

    source_template_chunk_id: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.anonymous_block_id < 0:
            raise AnonymousPackingError(
                "anonymous_block_id 不能为负数。"
            )

        if self.rows <= 0:
            raise AnonymousPackingError(
                f"rows 必须大于 0，当前为 {self.rows}。"
            )

        if self.cols <= 0:
            raise AnonymousPackingError(
                f"cols 必须大于 0，当前为 {self.cols}。"
            )

        if self.source_template_chunk_id < 0:
            raise AnonymousPackingError(
                "source_template_chunk_id 不能为负数。"
            )

    @property
    def area(self) -> int:
        """匿名块面积。"""
        return self.rows * self.cols

    @property
    def longest_side(self) -> int:
        """匿名块最长边。"""
        return max(
            self.rows,
            self.cols,
        )

    @property
    def shortest_side(self) -> int:
        """匿名块最短边。"""
        return min(
            self.rows,
            self.cols,
        )

    @property
    def size_key(self) -> SizeKey:
        """与方向无关的尺寸类型。"""
        return make_size_key(
            self.rows,
            self.cols,
        )


# ============================================================
# 装箱结果
# ============================================================


@dataclass(frozen=True, slots=True)
class PackingResult:
    """
    一个 PartitionTemplate 完成整个模型匿名装箱后的结果。

    注意：

    此时只有：

        plane_id
        slot_id
        x
        y
        slot_rows
        slot_cols

    还没有：

        subcube_id
        z
        layer_id
        expert_id
        matrix_name
    """

    template_id: str

    matrix_count: int

    H: int
    W: int

    planes: tuple[Plane, ...]
    slots: tuple[PhysicalSlot, ...]

    total_block_area: int

    expected_block_count: int

    orientation_swapped_count: int

    def __post_init__(self) -> None:
        self.validate_basic()

    def validate_basic(self) -> None:
        if not self.template_id:
            raise AnonymousPackingError(
                "template_id 不能为空。"
            )

        if self.matrix_count <= 0:
            raise AnonymousPackingError(
                "matrix_count 必须大于 0。"
            )

        if self.H <= 0 or self.W <= 0:
            raise AnonymousPackingError(
                f"H、W 必须大于 0，当前为 {self.H}×{self.W}。"
            )

        if self.expected_block_count < 0:
            raise AnonymousPackingError(
                "expected_block_count 不能为负数。"
            )

        if len(self.slots) != self.expected_block_count:
            raise AnonymousPackingError(
                "实际 PhysicalSlot 数量与预期匿名块数量不一致："
                f"slots={len(self.slots)}, "
                f"expected={self.expected_block_count}。"
            )

    @property
    def plane_count(self) -> int:
        """
        实际使用的二维平面数量 P。
        """
        return len(self.planes)

    @property
    def slot_count(self) -> int:
        return len(self.slots)

    @property
    def plane_area(self) -> int:
        return self.H * self.W

    @property
    def total_used_plane_area(self) -> int:
        """
        P 个已使用 Plane 的总容量：

            P × H × W
        """
        return (
            self.plane_count
            * self.H
            * self.W
        )

    @property
    def internal_fragmentation(self) -> int:
        """
        已使用 Plane 内部的真实空间碎片：

            P×H×W - S

        这里只包含二维装箱碎片。

        尚未包含后面因为：
            Q = N²D > P

        而产生的整层空闲容量。
        """
        return (
            self.total_used_plane_area
            - self.total_block_area
        )

    @property
    def packing_utilization(self) -> float:
        """
        P 个实际使用平面的平均空间利用率。

            U_packing = S / (P×H×W)
        """

        if self.total_used_plane_area == 0:
            return 0.0

        return (
            self.total_block_area
            / self.total_used_plane_area
        )

    @property
    def orientation_original_count(self) -> int:
        return (
            self.slot_count
            - self.orientation_swapped_count
        )

    def size_histogram(
        self,
    ) -> dict[SizeKey, int]:
        """
        统计最终 PhysicalSlot 的尺寸类型数量。
        """

        counter = Counter(
            slot.size_key
            for slot in self.slots
        )

        return dict(counter)

    def summary(self) -> str:
        return (
            f"PackingResult<{self.template_id}>: "
            f"H={self.H}, "
            f"W={self.W}, "
            f"blocks={self.slot_count}, "
            f"P={self.plane_count}, "
            f"packing_utilization="
            f"{self.packing_utilization:.4%}, "
            f"internal_fragmentation="
            f"{self.internal_fragmentation}"
        )


# ============================================================
# 从 PartitionTemplate 生成匿名矩形
# ============================================================


def build_anonymous_blocks(
    template: PartitionTemplate,
    matrix_count: int,
) -> list[AnonymousBlock]:
    """
    将一个 PartitionTemplate 扩展成整个模型的匿名矩形实例。

    例如：

        template：
            Chunk-0 = 4096×2048
            Chunk-1 = 3072×2048

        matrix_count = 44544

    最终：

        44544 个 4096×2048
        44544 个 3072×2048

    总共：

        89088 个 AnonymousBlock。

    ------------------------------------------------------------

    注意：

    这里虽然使用 matrix_index 进行循环，
    但不会把它保存成：
        layer_id
        expert_id
        matrix_name

    它仅仅用于把模板复制 matrix_count 次。
    """

    if matrix_count <= 0:
        raise AnonymousPackingError(
            "matrix_count 必须大于 0。"
        )

    blocks: list[AnonymousBlock] = []

    anonymous_block_id = 0

    for _matrix_index in range(matrix_count):

        for chunk in template.chunks:

            blocks.append(
                AnonymousBlock(
                    anonymous_block_id=(
                        anonymous_block_id
                    ),
                    rows=chunk.rows,
                    cols=chunk.cols,
                    source_template_chunk_id=(
                        chunk.chunk_id
                    ),
                )
            )

            anonymous_block_id += 1

    return blocks


def sort_anonymous_blocks(
    blocks: list[AnonymousBlock],
) -> list[AnonymousBlock]:
    """
    使用确定性的 Baseline 顺序排列匿名矩形。

    优先级：

    1. 面积越大越先放；
    2. 面积相同，最长边越长越先放；
    3. 再相同，最短边越长越先放；
    4. 最后使用 anonymous_block_id 保证结果固定。

    大块先放的原因：

        大块可选择位置更少。

    如果先放很多小块，
    容易将大块需要的连续空间切碎。
    """

    return sorted(
        blocks,
        key=lambda block: (
            -block.area,
            -block.longest_side,
            -block.shortest_side,
            block.anonymous_block_id,
        ),
    )


# ============================================================
# 多 Plane 全局候选选择
# ============================================================


def find_best_existing_plane_candidate(
    planes: list[Plane],
    block: AnonymousBlock,
    allow_rotation: bool = True,
) -> tuple[Plane, PlacementCandidate] | None:
    """
    在所有已有 Plane 中寻找一个全局最优 BSSF 候选。

    对每个 Plane：

        find_best_position()

    会先找到该 Plane 内部的最佳候选。

    然后本函数再在所有 Plane 的候选之间比较。

    PlacementCandidate.score_tuple() 已经包含：

        short_side_fit
        long_side_fit
        plane_id
        y
        x
        orientation_swapped
        free_rect_index

    因此可以直接按字典序选择最小值。
    """

    best_plane: Plane | None = None

    best_candidate: PlacementCandidate | None = None

    for plane in planes:

        candidate = find_best_position(
            plane=plane,
            rows=block.rows,
            cols=block.cols,
            allow_rotation=allow_rotation,
        )

        if candidate is None:
            continue

        if (
            best_candidate is None
            or candidate.score_tuple()
            < best_candidate.score_tuple()
        ):
            best_plane = plane
            best_candidate = candidate

    if (
        best_plane is None
        or best_candidate is None
    ):
        return None

    return (
        best_plane,
        best_candidate,
    )


# ============================================================
# 新建 Plane
# ============================================================


def create_plane_for_block(
    plane_id: int,
    H: int,
    W: int,
    block: AnonymousBlock,
    allow_rotation: bool,
) -> tuple[Plane, PlacementCandidate]:
    """
    创建一个新的空 H×W Plane，并在其中为 block 找位置。

    如果一个刚创建的空 Plane 都无法容纳 block，
    说明第二步产生了一个非法切分模板，
    应立即报错。
    """

    plane = create_empty_plane(
        plane_id=plane_id,
        H=H,
        W=W,
    )

    candidate = find_best_position(
        plane=plane,
        rows=block.rows,
        cols=block.cols,
        allow_rotation=allow_rotation,
    )

    if candidate is None:
        raise AnonymousPackingError(
            "匿名块无法放入一个全新的空 Plane，"
            "说明第二步模板与当前 H、W 不兼容："
            f"block={block.rows}×{block.cols}, "
            f"H×W={H}×{W}。"
        )

    return (
        plane,
        candidate,
    )


# ============================================================
# 完整装箱
# ============================================================


def pack_anonymous_blocks(
    template: PartitionTemplate,
    matrix_count: int,
    H: int,
    W: int,
    allow_rotation: bool = True,
) -> PackingResult:
    """
    第三步匿名二维装箱的主要入口函数。

    流程：

        PartitionTemplate
                ↓
        扩展到完整模型 AnonymousBlock
                ↓
        面积降序排列
                ↓
        逐块处理
                ↓
        在所有已有 Plane 中寻找全局最佳 BSSF
                ↓
        如果存在：
            放入已有 Plane
        如果不存在：
            新建一个 H×W Plane
                ↓
        创建 PhysicalSlot
                ↓
        完成所有块
                ↓
        得到 P = len(planes)

    ------------------------------------------------------------

    重要：

    这里的新建 Plane 只是：

        “二维装箱新增一个逻辑箱子”

    完全没有决定：

        Plane 属于哪个 Sub-Cube
        Plane 的 z 是多少

    这些都留到第四步。
    """

    if matrix_count <= 0:
        raise AnonymousPackingError(
            "matrix_count 必须大于 0。"
        )

    if H <= 0 or W <= 0:
        raise AnonymousPackingError(
            f"H、W 必须大于 0，当前为 {H}×{W}。"
        )

    # ========================================================
    # 1. 将模板扩展成整个模型的匿名块
    # ========================================================

    blocks = build_anonymous_blocks(
        template=template,
        matrix_count=matrix_count,
    )

    # ========================================================
    # 2. 大块优先排序
    # ========================================================

    sorted_blocks = sort_anonymous_blocks(
        blocks
    )

    # ========================================================
    # 3. 开始多 Plane 装箱
    # ========================================================

    planes: list[Plane] = []

    slots: list[PhysicalSlot] = []

    next_slot_id = 0

    orientation_swapped_count = 0

    for block in sorted_blocks:

        # ----------------------------------------------------
        # 3.1 先搜索所有已有 Plane
        # ----------------------------------------------------

        existing_result = (
            find_best_existing_plane_candidate(
                planes=planes,
                block=block,
                allow_rotation=allow_rotation,
            )
        )

        # ----------------------------------------------------
        # 3.2 如果没有已有 Plane 可以容纳，
        #     新建一个 Plane
        # ----------------------------------------------------

        if existing_result is None:

            new_plane_id = len(planes)

            plane, candidate = (
                create_plane_for_block(
                    plane_id=new_plane_id,
                    H=H,
                    W=W,
                    block=block,
                    allow_rotation=allow_rotation,
                )
            )

            planes.append(plane)

        else:

            plane, candidate = existing_result

        # ----------------------------------------------------
        # 3.3 正式提交该匿名矩形
        # ----------------------------------------------------

        slot = commit_placement(
            plane=plane,
            candidate=candidate,
            slot_id=next_slot_id,
        )

        slots.append(slot)

        if slot.orientation_swapped:
            orientation_swapped_count += 1

        next_slot_id += 1

    # ========================================================
    # 4. 最终空间合法性检查
    # ========================================================

    for plane in planes:

        plane.validate_layout()

        validate_free_rectangles(
            plane
        )

    # ========================================================
    # 5. 面积检查
    # ========================================================

    total_block_area = sum(
        block.area
        for block in blocks
    )

    expected_area = (
        template.base_area
        * matrix_count
    )

    if total_block_area != expected_area:
        raise AnonymousPackingError(
            "匿名块总面积与模板扩展后的理论面积不一致："
            f"blocks={total_block_area}, "
            f"expected={expected_area}。"
        )

    # ========================================================
    # 6. 匿名块数量检查
    # ========================================================

    expected_block_count = (
        template.chunk_count
        * matrix_count
    )

    if len(slots) != expected_block_count:
        raise AnonymousPackingError(
            "PhysicalSlot 数量与匿名块需求不一致："
            f"slots={len(slots)}, "
            f"expected={expected_block_count}。"
        )

    # ========================================================
    # 7. 尺寸类型检查
    # ========================================================

    expected_histogram = (
        template.total_size_histogram(
            matrix_count=matrix_count
        )
    )

    actual_histogram = dict(
        Counter(
            slot.size_key
            for slot in slots
        )
    )

    if actual_histogram != expected_histogram:
        raise AnonymousPackingError(
            "最终 PhysicalSlot 尺寸数量与模板需求不一致。\n"
            f"expected={expected_histogram}\n"
            f"actual={actual_histogram}"
        )

    # ========================================================
    # 8. 返回结果
    # ========================================================

    return PackingResult(
        template_id=template.template_id,

        matrix_count=matrix_count,

        H=H,
        W=W,

        planes=tuple(planes),
        slots=tuple(slots),

        total_block_area=total_block_area,

        expected_block_count=expected_block_count,

        orientation_swapped_count=(
            orientation_swapped_count
        ),
    )


# ============================================================
# 输出辅助
# ============================================================


def print_packing_result(
    result: PackingResult,
    show_planes: bool = False,
    max_planes_to_show: int = 20,
) -> None:
    """
    打印匿名装箱结果。

    完整模型 Plane 数可能很多，
    因此默认不逐个打印。
    """

    print(
        "========== Anonymous Packing Result =========="
    )

    print(
        f"template_id：{result.template_id}"
    )

    print(
        f"H×W：{result.H}×{result.W}"
    )

    print(
        f"匿名矩阵数量：{result.matrix_count}"
    )

    print(
        f"匿名块 / Slot 数量：{result.slot_count}"
    )

    print(
        f"实际平面数 P：{result.plane_count}"
    )

    print(
        f"有效权重面积 S：{result.total_block_area}"
    )

    print(
        "已使用平面总容量 P×H×W："
        f"{result.total_used_plane_area}"
    )

    print(
        "二维内部碎片："
        f"{result.internal_fragmentation}"
    )

    print(
        "二维装箱利用率："
        f"{result.packing_utilization:.6%}"
    )

    print(
        "交换方向放置数量："
        f"{result.orientation_swapped_count}"
    )

    print(
        "未交换方向数量："
        f"{result.orientation_original_count}"
    )

    print(
        "PhysicalSlot 尺寸统计："
        f"{result.size_histogram()}"
    )

    if show_planes:

        print(
            "\n========== Plane Details =========="
        )

        for plane in result.planes[
            :max_planes_to_show
        ]:

            print(
                plane.summary()
            )

        if (
            result.plane_count
            > max_planes_to_show
        ):
            print(
                f"... 其余 "
                f"{result.plane_count - max_planes_to_show} "
                "个 Plane 未显示"
            )


# ============================================================
# 简单测试
# ============================================================


if __name__ == "__main__":

    from partition.partition_generator import (
        generate_partition_templates,
    )

    from partition.partition_validator import (
        validate_partition_templates,
    )

    # ========================================================
    # 为了快速测试，不直接跑 44544 个矩阵。
    #
    # 先使用 4 个匿名矩阵验证算法。
    # ========================================================

    matrix_rows = 7168
    matrix_cols = 2048

    H = 4096
    W = 4096

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
        raise_on_error=True,
    )

    print(
        f"候选模板数量：{len(templates)}"
    )

    for template in templates:

        print(
            "\n========================================"
        )

        print(
            template.summary()
        )

        result = pack_anonymous_blocks(
            template=template,

            # 快速测试先用 4
            matrix_count=4,

            H=H,
            W=W,

            allow_rotation=True,
        )

        print_packing_result(
            result=result,
            show_planes=True,
        )