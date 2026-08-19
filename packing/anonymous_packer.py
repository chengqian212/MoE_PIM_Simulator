# packing/anonymous_packer.py
"""
高效版匿名二维装箱器。

核心算法仍然保持：

    面积降序
    + MaxRects-BSSF
    + 允许单块旋转

相比旧版，主要优化：

旧版：
    每放一个匿名块
        -> 扫描所有已有 Plane
        -> 每个 Plane 再扫描 FreeRectangle

当块数量达到 8~9 万时非常慢。

新版：
    1. 不再创建 8~9 万个 AnonymousBlock 后排序；
    2. 直接使用 “尺寸 + count” 批量处理；
    3. 对同一种尺寸，只为每个 Plane 计算一次当前最佳候选；
    4. 使用 heap 保存所有 Plane 的 BSSF 候选；
    5. 每放一个块，只重新计算刚刚发生变化的那个 Plane。

这样可以大幅减少重复扫描。

注意：
- 仍然只做匿名空间规划；
- 没有 layer_id；
- 没有 expert_id；
- 没有 matrix_name；
- 没有 subcube_id；
- 没有 z；
- 第四步才恢复真实逻辑身份。
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from itertools import count

from model_geometry import (
    SizeKey,
    make_size_key,
)

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
    """匿名矩形装箱异常。"""


# ============================================================
# 1. 单个匿名块
#
# 保留这个类主要用于调试和小规模测试。
#
# 完整模型装箱不会再创建 89088 个这样的对象。
# ============================================================


@dataclass(frozen=True, slots=True)
class AnonymousBlock:
    """
    一个匿名矩形实例。

    完整装箱时不会大规模创建该对象，
    这里只保留用于兼容调试代码。
    """

    anonymous_block_id: int

    rows: int
    cols: int

    source_template_chunk_id: int

    def __post_init__(self) -> None:
        if self.anonymous_block_id < 0:
            raise AnonymousPackingError(
                "anonymous_block_id 不能为负数。"
            )

        if self.rows <= 0 or self.cols <= 0:
            raise AnonymousPackingError(
                "AnonymousBlock 尺寸必须大于 0。"
            )

        if self.source_template_chunk_id < 0:
            raise AnonymousPackingError(
                "source_template_chunk_id 不能为负数。"
            )

    @property
    def area(self) -> int:
        return self.rows * self.cols

    @property
    def longest_side(self) -> int:
        return max(
            self.rows,
            self.cols,
        )

    @property
    def shortest_side(self) -> int:
        return min(
            self.rows,
            self.cols,
        )

    @property
    def size_key(self) -> SizeKey:
        return make_size_key(
            self.rows,
            self.cols,
        )


# ============================================================
# 2. 批量匿名块需求
#
# 完整模型实际使用这个结构。
# ============================================================


@dataclass(frozen=True, slots=True)
class AnonymousBlockDemand:
    """
    表示：

        有 count 个 rows×cols 匿名矩形需要装箱。

    例如：

        AnonymousBlockDemand(
            rows=4096,
            cols=2048,
            count=44544,
        )

    用一个对象代替 44544 个 AnonymousBlock。
    """

    rows: int
    cols: int
    count: int

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise AnonymousPackingError(
                f"rows 必须大于 0，当前为 {self.rows}。"
            )

        if self.cols <= 0:
            raise AnonymousPackingError(
                f"cols 必须大于 0，当前为 {self.cols}。"
            )

        if self.count <= 0:
            raise AnonymousPackingError(
                f"count 必须大于 0，当前为 {self.count}。"
            )

    @property
    def area(self) -> int:
        return (
            self.rows
            * self.cols
        )

    @property
    def total_area(self) -> int:
        return (
            self.area
            * self.count
        )

    @property
    def longest_side(self) -> int:
        return max(
            self.rows,
            self.cols,
        )

    @property
    def shortest_side(self) -> int:
        return min(
            self.rows,
            self.cols,
        )

    @property
    def size_key(self) -> SizeKey:
        return make_size_key(
            self.rows,
            self.cols,
        )


# ============================================================
# 3. 装箱结果
# ============================================================


@dataclass(frozen=True, slots=True)
class PackingResult:
    """
    第三步匿名二维装箱结果。
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
                f"H、W 必须大于 0，"
                f"当前为 {self.H}×{self.W}。"
            )

        if (
            len(self.slots)
            != self.expected_block_count
        ):
            raise AnonymousPackingError(
                "PhysicalSlot 数量与理论匿名块数量不一致："
                f"slots={len(self.slots)}, "
                f"expected={self.expected_block_count}。"
            )

    @property
    def plane_count(self) -> int:
        """
        实际二维平面数量 P。
        """
        return len(
            self.planes
        )

    @property
    def slot_count(self) -> int:
        return len(
            self.slots
        )

    @property
    def plane_area(self) -> int:
        return (
            self.H
            * self.W
        )

    @property
    def total_used_plane_area(self) -> int:
        """
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
        已使用 Plane 内部的二维碎片：

            P×H×W - S
        """
        return (
            self.total_used_plane_area
            - self.total_block_area
        )

    @property
    def packing_utilization(self) -> float:

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

        return dict(
            Counter(
                slot.size_key
                for slot in self.slots
            )
        )

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
# 4. 小规模调试函数
#
# 旧接口保留。
# ============================================================


def build_anonymous_blocks(
    template: PartitionTemplate,
    matrix_count: int,
) -> list[AnonymousBlock]:
    """
    真正创建每一个 AnonymousBlock。

    注意：
    这个函数只建议用于测试。

    完整模型装箱不要调用它，
    否则又会产生几万个 Python 对象。
    """

    if matrix_count <= 0:
        raise AnonymousPackingError(
            "matrix_count 必须大于 0。"
        )

    blocks: list[
        AnonymousBlock
    ] = []

    block_id = 0

    for _ in range(
        matrix_count
    ):

        for chunk in template.chunks:

            blocks.append(
                AnonymousBlock(
                    anonymous_block_id=block_id,

                    rows=chunk.rows,
                    cols=chunk.cols,

                    source_template_chunk_id=(
                        chunk.chunk_id
                    ),
                )
            )

            block_id += 1

    return blocks


# ============================================================
# 5. 构造批量需求
# ============================================================


def build_anonymous_block_demands(
    template: PartitionTemplate,
    matrix_count: int,
) -> list[AnonymousBlockDemand]:
    """
    将模板直接转换为：

        shape -> count

    不生成每一个实际匿名块。

    ------------------------------------------------

    示例：

    Template：

        4096×2048
        3072×2048

    matrix_count：

        44544

    得到：

        4096×2048 × 44544
        3072×2048 × 44544

    只创建两个 AnonymousBlockDemand。
    """

    if matrix_count <= 0:
        raise AnonymousPackingError(
            "matrix_count 必须大于 0。"
        )

    # 这里保留 rows、cols 的实际方向，
    # 因为 orientation_swapped 需要相对于这个方向判断。
    shape_counts: Counter[
        tuple[int, int]
    ] = Counter()

    for chunk in template.chunks:

        shape_counts[
            (
                chunk.rows,
                chunk.cols,
            )
        ] += matrix_count

    demands = [

        AnonymousBlockDemand(
            rows=rows,
            cols=cols,
            count=block_count,
        )

        for (
            rows,
            cols
        ), block_count in shape_counts.items()
    ]

    return demands


def sort_anonymous_block_demands(
    demands: list[AnonymousBlockDemand],
) -> list[AnonymousBlockDemand]:
    """
    与原 Baseline 保持相同思想：

        1. 面积降序；
        2. 最长边降序；
        3. 最短边降序；
        4. rows；
        5. cols。

    相同尺寸块本身完全匿名，
    所以不需要逐个排序。
    """

    return sorted(
        demands,
        key=lambda demand: (
            -demand.area,
            -demand.longest_side,
            -demand.shortest_side,
            -demand.rows,
            -demand.cols,
        ),
    )


# ============================================================
# 6. Heap 中保存的候选
# ============================================================


HeapEntry = tuple[
    tuple,
    int,
    int,
    int,
    PlacementCandidate,
]


def _push_plane_candidate(
    heap: list[HeapEntry],

    plane: Plane,

    rows: int,
    cols: int,

    allow_rotation: bool,

    plane_versions: dict[int, int],

    unique_counter,
) -> None:
    """
    重新计算某个 Plane 对当前尺寸的最佳 BSSF 候选，
    并加入 heap。

    如果当前 Plane 已无法容纳该尺寸，则什么也不做。
    """

    candidate = find_best_position(
        plane=plane,
        rows=rows,
        cols=cols,
        allow_rotation=allow_rotation,
    )

    if candidate is None:
        return

    version = plane_versions[
        plane.plane_id
    ]

    heapq.heappush(
        heap,
        (
            candidate.score_tuple(),

            next(unique_counter),

            plane.plane_id,

            version,

            candidate,
        )
    )


def _build_candidate_heap(
    planes: list[Plane],

    rows: int,
    cols: int,

    allow_rotation: bool,

    plane_versions: dict[int, int],

    unique_counter,
) -> list[HeapEntry]:
    """
    对“当前尺寸”只扫描所有历史 Plane 一次。

    这一步是新版与旧版最大的区别之一。

    旧版：
        每一个块都重新扫描全部 Plane。

    新版：
        同尺寸的一批块开始时扫一次；
        后续只有被修改的那个 Plane 重新计算候选。
    """

    heap: list[
        HeapEntry
    ] = []

    for plane in planes:

        _push_plane_candidate(
            heap=heap,

            plane=plane,

            rows=rows,
            cols=cols,

            allow_rotation=allow_rotation,

            plane_versions=plane_versions,

            unique_counter=unique_counter,
        )

    return heap


def _pop_valid_candidate(
    heap: list[HeapEntry],
    plane_versions: dict[int, int],
) -> tuple[int, PlacementCandidate] | None:
    """
    从 heap 取出当前仍然有效的最优候选。

    version 用来防止旧候选失效后仍然被使用。
    """

    while heap:

        (
            _score,
            _unique_id,
            plane_id,
            stored_version,
            candidate,
        ) = heapq.heappop(
            heap
        )

        current_version = (
            plane_versions.get(
                plane_id
            )
        )

        if current_version is None:
            continue

        if (
            stored_version
            != current_version
        ):
            # 这个 Plane 已经发生变化，
            # heap 中的是旧候选。
            continue

        return (
            plane_id,
            candidate,
        )

    return None


# ============================================================
# 7. 完整高效装箱
# ============================================================


def pack_anonymous_blocks(
    template: PartitionTemplate,

    matrix_count: int,

    H: int,
    W: int,

    allow_rotation: bool = True,

    verbose: bool = False,

    progress_interval: int = 5000,
) -> PackingResult:
    """
    高效版完整匿名装箱。

    ------------------------------------------------

    保持算法思想：

        面积降序
        + MaxRects-BSSF
        + Rotation

    但是实现方式从：

        Block
        × 所有 Plane
        × 所有 FreeRectangle

    改为：

        尺寸批次
        × Plane 一次初始化
        + 修改 Plane 时局部重新计算

    ------------------------------------------------

    Args:

        template:
            第二步产生的匿名矩阵切分模板。

        matrix_count:
            匿名标准矩阵数量。

            当前完整模型通常为：

                58 × 256 × 3
                = 44544

        H, W:
            Plane 尺寸。

        allow_rotation:
            是否允许匿名块旋转。

        verbose:
            是否输出进度。

        progress_interval:
            每处理多少匿名块输出一次进度。

    Returns:

        PackingResult
    """

    if matrix_count <= 0:
        raise AnonymousPackingError(
            "matrix_count 必须大于 0。"
        )

    if H <= 0 or W <= 0:
        raise AnonymousPackingError(
            f"H、W 必须大于 0，"
            f"当前为 {H}×{W}。"
        )

    if progress_interval <= 0:
        raise AnonymousPackingError(
            "progress_interval 必须大于 0。"
        )

    # ========================================================
    # 1. 直接构造尺寸批次
    #
    # 不创建 89088 个 AnonymousBlock。
    # ========================================================

    demands = (
        build_anonymous_block_demands(
            template=template,
            matrix_count=matrix_count,
        )
    )

    demands = (
        sort_anonymous_block_demands(
            demands
        )
    )

    total_blocks = sum(
        demand.count
        for demand in demands
    )

    total_block_area = sum(
        demand.total_area
        for demand in demands
    )

    expected_area = (
        template.base_area
        * matrix_count
    )

    if (
        total_block_area
        != expected_area
    ):
        raise AnonymousPackingError(
            "匿名需求总面积错误："
            f"{total_block_area} "
            f"!= {expected_area}。"
        )

    # ========================================================
    # 2. 全局状态
    # ========================================================

    planes: list[
        Plane
    ] = []

    slots: list[
        PhysicalSlot
    ] = []

    # plane_id 与 list 下标保持一致，
    # 因此后面可以 O(1) 找到 Plane。
    plane_versions: dict[
        int,
        int
    ] = {}

    unique_counter = count()

    next_slot_id = 0

    orientation_swapped_count = 0

    processed_blocks = 0

    next_progress = (
        progress_interval
    )

    if verbose:

        print(
            "========== Fast Anonymous Packing =========="
        )

        print(
            f"template：{template.template_id}"
        )

        print(
            f"H×W：{H}×{W}"
        )

        print(
            f"matrix_count：{matrix_count}"
        )

        print(
            f"匿名块总数：{total_blocks}"
        )

        print(
            f"尺寸类型数：{len(demands)}"
        )

    # ========================================================
    # 3. 按尺寸批量处理
    # ========================================================

    for demand_index, demand in enumerate(
        demands
    ):

        rows = demand.rows
        cols = demand.cols

        if verbose:

            print(
                "\n----------------------------------------"
            )

            print(
                f"尺寸类型 "
                f"{demand_index + 1}/{len(demands)}："
                f"{rows}×{cols}"
            )

            print(
                f"数量：{demand.count}"
            )

            print(
                f"当前已有 Plane：{len(planes)}"
            )

        # ====================================================
        # 对这个尺寸：
        #
        # 所有历史 Plane 只扫描一次。
        # ====================================================

        heap = _build_candidate_heap(
            planes=planes,

            rows=rows,
            cols=cols,

            allow_rotation=allow_rotation,

            plane_versions=plane_versions,

            unique_counter=unique_counter,
        )

        # ====================================================
        # 开始放这一批完全相同的匿名块
        # ====================================================

        for _ in range(
            demand.count
        ):

            # ------------------------------------------------
            # 先尝试已有 Plane 中的全局最佳候选
            # ------------------------------------------------

            best = _pop_valid_candidate(
                heap=heap,
                plane_versions=plane_versions,
            )

            # ------------------------------------------------
            # 所有已有 Plane 都装不下
            # → 新建 Plane
            # ------------------------------------------------

            if best is None:

                plane_id = len(
                    planes
                )

                plane = create_empty_plane(
                    plane_id=plane_id,
                    H=H,
                    W=W,
                )

                candidate = (
                    find_best_position(
                        plane=plane,
                        rows=rows,
                        cols=cols,
                        allow_rotation=allow_rotation,
                    )
                )

                if candidate is None:
                    raise AnonymousPackingError(
                        "一个全新的空 Plane 都无法容纳当前匿名块："
                        f"block={rows}×{cols}, "
                        f"Plane={H}×{W}。"
                    )

                planes.append(
                    plane
                )

                plane_versions[
                    plane_id
                ] = 0

            else:

                (
                    plane_id,
                    candidate,
                ) = best

                plane = planes[
                    plane_id
                ]

            # ------------------------------------------------
            # 正式放置
            # ------------------------------------------------

            slot = commit_placement(
                plane=plane,
                candidate=candidate,
                slot_id=next_slot_id,
            )

            slots.append(
                slot
            )

            if (
                slot.orientation_swapped
            ):
                orientation_swapped_count += 1

            next_slot_id += 1
            processed_blocks += 1

            # ------------------------------------------------
            # Plane 状态已经变化
            # ------------------------------------------------

            plane_versions[
                plane_id
            ] += 1

            # ------------------------------------------------
            # 只重新计算这个 Plane。
            #
            # 其他几万个 Plane 完全没有变化，
            # 不需要重新检查。
            # ------------------------------------------------

            _push_plane_candidate(
                heap=heap,

                plane=plane,

                rows=rows,
                cols=cols,

                allow_rotation=allow_rotation,

                plane_versions=plane_versions,

                unique_counter=unique_counter,
            )

            # ------------------------------------------------
            # 可选进度显示
            # ------------------------------------------------

            if (
                verbose
                and processed_blocks
                >= next_progress
            ):

                progress = (
                    processed_blocks
                    / total_blocks
                )

                print(
                    f"进度："
                    f"{processed_blocks}/{total_blocks} "
                    f"({progress:.2%}), "
                    f"Plane={len(planes)}"
                )

                while (
                    next_progress
                    <= processed_blocks
                ):
                    next_progress += (
                        progress_interval
                    )

    # ========================================================
    # 4. 数量检查
    # ========================================================

    if (
        len(slots)
        != total_blocks
    ):
        raise AnonymousPackingError(
            "最终 Slot 数量错误："
            f"{len(slots)} != {total_blocks}。"
        )

    # ========================================================
    # 5. 空间布局检查
    # ========================================================

    if verbose:

        print(
            "\n正在执行最终空间合法性检查..."
        )

    for plane in planes:

        plane.validate_layout()

        validate_free_rectangles(
            plane
        )

    # ========================================================
    # 6. 尺寸数量检查
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

    if (
        actual_histogram
        != expected_histogram
    ):
        raise AnonymousPackingError(
            "PhysicalSlot 尺寸统计与模板需求不一致。\n"
            f"expected={expected_histogram}\n"
            f"actual={actual_histogram}"
        )

    # ========================================================
    # 7. 返回结果
    # ========================================================

    result = PackingResult(
        template_id=template.template_id,

        matrix_count=matrix_count,

        H=H,
        W=W,

        planes=tuple(
            planes
        ),

        slots=tuple(
            slots
        ),

        total_block_area=(
            total_block_area
        ),

        expected_block_count=(
            total_blocks
        ),

        orientation_swapped_count=(
            orientation_swapped_count
        ),
    )

    if verbose:

        print(
            "\n========== Packing Finished =========="
        )

        print(
            result.summary()
        )

    return result


# ============================================================
# 8. 输出结果
# ============================================================


def print_packing_result(
    result: PackingResult,

    show_planes: bool = False,

    max_planes_to_show: int = 20,
) -> None:

    print(
        "========== Anonymous Packing Result =========="
    )

    print(
        f"template_id："
        f"{result.template_id}"
    )

    print(
        f"H×W："
        f"{result.H}×{result.W}"
    )

    print(
        f"匿名矩阵数量："
        f"{result.matrix_count}"
    )

    print(
        f"匿名块 / Slot 数量："
        f"{result.slot_count}"
    )

    print(
        f"实际平面数量 P："
        f"{result.plane_count}"
    )

    print(
        f"有效权重面积 S："
        f"{result.total_block_area}"
    )

    print(
        "P×H×W："
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
        "旋转放置数量："
        f"{result.orientation_swapped_count}"
    )

    print(
        "未旋转数量："
        f"{result.orientation_original_count}"
    )

    print(
        "槽位尺寸统计："
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

        remaining = (
            result.plane_count
            - max_planes_to_show
        )

        if remaining > 0:

            print(
                f"... 还有 {remaining} 个 Plane 未显示"
            )


# ============================================================
# 9. 单文件测试
# ============================================================


if __name__ == "__main__":

    from partition.partition_generator import (
        generate_partition_templates,
    )

    from partition.partition_validator import (
        validate_partition_templates,
    )

    matrix_rows = 7168
    matrix_cols = 2048

    H = 4096
    W = 4096

    # ========================================================
    # 生成模板
    # ========================================================

    templates = (
        generate_partition_templates(
            matrix_rows=matrix_rows,
            matrix_cols=matrix_cols,
            H=H,
            W=W,
        )
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

    # ========================================================
    # 先做中等规模测试
    #
    # 不建议第一次就直接跑 44544。
    # ========================================================

    test_matrix_count = 44544

    for template in templates:

        print(
            "\n========================================"
        )

        print(
            template.summary()
        )

        result = pack_anonymous_blocks(
            template=template,

            matrix_count=test_matrix_count,

            H=H,
            W=W,

            allow_rotation=True,

            verbose=True,

            progress_interval=1000,
        )

        print_packing_result(
            result=result,
            show_planes=False,
        )