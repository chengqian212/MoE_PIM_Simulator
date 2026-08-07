# packing/maxrects.py
"""
MaxRects-BSSF 二维矩形装箱算法。

本文件负责：

1. 在一个 Plane 的所有 FreeRectangle 中寻找合法放置位置；
2. 同时尝试原方向和旋转 90° 后的方向；
3. 使用 Best Short Side Fit（BSSF）选择最优位置；
4. 提交 PhysicalSlot；
5. 根据新放入的矩形更新 FreeRectangle；
6. 删除被其他 FreeRectangle 完全包含的冗余空闲矩形。

注意：
- 本文件只解决一个 Plane 内的二维几何放置；
- 不生成 layer_id、expert_id、matrix_name；
- 不涉及 Sub-Cube；
- 不涉及 z；
- 不处理 gate/up/down；
- 多个 Plane 之间如何选择，由 anonymous_packer.py 负责。
"""

from __future__ import annotations

from dataclasses import dataclass

from packing.physical_slot import PhysicalSlot
from packing.plane import (
    FreeRectangle,
    Plane,
)


class MaxRectsError(ValueError):
    """MaxRects 算法运行异常时抛出的错误。"""


@dataclass(frozen=True, slots=True)
class PlacementCandidate:
    """
    一个匿名矩形在某个 Plane 中的候选放置位置。

    注意：
    这只是候选结果，尚未真正修改 Plane。
    """

    plane_id: int

    # 对应哪个 FreeRectangle
    free_rect_index: int

    # 实际放置起点
    x: int
    y: int

    # 实际物理方向
    placed_rows: int
    placed_cols: int

    # 是否相对于输入 rows×cols 交换了方向
    orientation_swapped: bool

    # BSSF 第一评分
    short_side_fit: int

    # BSSF 第二评分
    long_side_fit: int

    @property
    def area(self) -> int:
        return self.placed_rows * self.placed_cols

    @property
    def row_end(self) -> int:
        return self.x + self.placed_rows

    @property
    def col_end(self) -> int:
        return self.y + self.placed_cols

    def score_tuple(self) -> tuple:
        """
        候选位置的确定性比较规则。

        优先级：

        1. short_side_fit 越小越好；
        2. long_side_fit 越小越好；
        3. plane_id 越小越好；
        4. y 越小越好；
        5. x 越小越好；
        6. 相同时优先不旋转；
        7. free_rect_index 越小越好。

        前两项是 MaxRects-BSSF 的核心。
        后面的项只是为了保证程序完全可复现。
        """

        return (
            self.short_side_fit,
            self.long_side_fit,
            self.plane_id,
            self.y,
            self.x,
            self.orientation_swapped,
            self.free_rect_index,
        )


def _validate_shape(
    rows: int,
    cols: int,
) -> None:
    if rows <= 0 or cols <= 0:
        raise MaxRectsError(
            f"待放矩形尺寸必须大于 0，当前为 {rows}×{cols}。"
        )


def calculate_bssf_score(
    free_rect: FreeRectangle,
    rows: int,
    cols: int,
) -> tuple[int, int] | None:
    """
    计算某个方向在一个 FreeRectangle 中的 BSSF 分数。

    如果放不进去，返回 None。

    假设：

        FreeRectangle = Fr × Fc
        待放矩形       = r  × c

    剩余边长：

        delta_r = Fr - r
        delta_c = Fc - c

    BSSF：

        short_side_fit = min(delta_r, delta_c)
        long_side_fit  = max(delta_r, delta_c)

    优先最小化 short_side_fit，
    再最小化 long_side_fit。
    """

    _validate_shape(rows, cols)

    if not free_rect.fits(
        rows=rows,
        cols=cols,
    ):
        return None

    delta_rows = free_rect.rows - rows
    delta_cols = free_rect.cols - cols

    short_side_fit = min(
        delta_rows,
        delta_cols,
    )

    long_side_fit = max(
        delta_rows,
        delta_cols,
    )

    return (
        short_side_fit,
        long_side_fit,
    )


def find_best_position(
    plane: Plane,
    rows: int,
    cols: int,
    allow_rotation: bool = True,
) -> PlacementCandidate | None:
    """
    在一个 Plane 中寻找 rows×cols 的最佳放置位置。

    会遍历该 Plane 当前所有 FreeRectangle。

    对每个 FreeRectangle：

        尝试原方向 rows×cols

    如果允许旋转：

        再尝试 cols×rows

    最终返回 BSSF 得分最小的 PlacementCandidate。

    如果当前 Plane 完全放不下，则返回 None。
    """

    _validate_shape(rows, cols)

    best_candidate: PlacementCandidate | None = None

    for free_rect_index, free_rect in enumerate(
        plane.free_rectangles
    ):

        # ====================================================
        # 1. 尝试原方向
        # ====================================================

        normal_score = calculate_bssf_score(
            free_rect=free_rect,
            rows=rows,
            cols=cols,
        )

        if normal_score is not None:

            candidate = PlacementCandidate(
                plane_id=plane.plane_id,
                free_rect_index=free_rect_index,

                x=free_rect.x,
                y=free_rect.y,

                placed_rows=rows,
                placed_cols=cols,

                orientation_swapped=False,

                short_side_fit=normal_score[0],
                long_side_fit=normal_score[1],
            )

            if (
                best_candidate is None
                or candidate.score_tuple()
                < best_candidate.score_tuple()
            ):
                best_candidate = candidate

        # ====================================================
        # 2. 尝试旋转方向
        # ====================================================

        # 正方形旋转没有意义，不重复测试。
        if (
            allow_rotation
            and rows != cols
        ):

            rotated_score = calculate_bssf_score(
                free_rect=free_rect,
                rows=cols,
                cols=rows,
            )

            if rotated_score is not None:

                candidate = PlacementCandidate(
                    plane_id=plane.plane_id,
                    free_rect_index=free_rect_index,

                    x=free_rect.x,
                    y=free_rect.y,

                    placed_rows=cols,
                    placed_cols=rows,

                    orientation_swapped=True,

                    short_side_fit=rotated_score[0],
                    long_side_fit=rotated_score[1],
                )

                if (
                    best_candidate is None
                    or candidate.score_tuple()
                    < best_candidate.score_tuple()
                ):
                    best_candidate = candidate

    return best_candidate


def _regions_overlap(
    ax: int,
    ay: int,
    a_rows: int,
    a_cols: int,
    bx: int,
    by: int,
    b_rows: int,
    b_cols: int,
) -> bool:
    """
    判断两个二维矩形是否真正重叠。

    只边界接触不算重叠。
    """

    a_row_end = ax + a_rows
    a_col_end = ay + a_cols

    b_row_end = bx + b_rows
    b_col_end = by + b_cols

    row_overlap = (
        ax < b_row_end
        and bx < a_row_end
    )

    col_overlap = (
        ay < b_col_end
        and by < a_col_end
    )

    return row_overlap and col_overlap


def split_free_rectangle(
    free_rect: FreeRectangle,
    placed_x: int,
    placed_y: int,
    placed_rows: int,
    placed_cols: int,
) -> list[FreeRectangle]:
    """
    将一个与新放置矩形发生重叠的 FreeRectangle 拆分。

    MaxRects 的经典做法是：

    如果新矩形与 free_rect 重叠，
    最多产生四个新的候选空闲矩形：

        1. 放置矩形之前的 H 方向区域；
        2. 放置矩形之后的 H 方向区域；
        3. 放置矩形之前的 W 方向区域；
        4. 放置矩形之后的 W 方向区域。

    --------------------------------------------------

    注意：

    这些新的 FreeRectangle 之间允许重叠。

    这是 MaxRects 的正常行为。

    因此后续不能通过：

        sum(free_rect.area)

    来计算实际剩余面积。
    """

    if placed_rows <= 0 or placed_cols <= 0:
        raise MaxRectsError(
            "placed_rows 和 placed_cols 必须大于 0。"
        )

    # 如果根本没有重叠，该 FreeRectangle 保持不变。
    if not _regions_overlap(
        free_rect.x,
        free_rect.y,
        free_rect.rows,
        free_rect.cols,

        placed_x,
        placed_y,
        placed_rows,
        placed_cols,
    ):
        return [free_rect]

    result: list[FreeRectangle] = []

    free_row_end = free_rect.row_end
    free_col_end = free_rect.col_end

    placed_row_end = (
        placed_x + placed_rows
    )

    placed_col_end = (
        placed_y + placed_cols
    )

    # ========================================================
    # 1. placed 矩形在 H 方向前面的区域
    #
    # free_rect:
    #
    # +----------------+
    # |      free      |
    # |                |
    # +----------------+
    #
    # 如果 placed_x > free.x：
    #
    # +----+-----------+
    # |new |           |
    # |    |  placed   |
    # +----+-----------+
    # ========================================================

    if (
        placed_x > free_rect.x
        and placed_x < free_row_end
    ):

        new_rows = (
            placed_x - free_rect.x
        )

        if new_rows > 0:
            result.append(
                FreeRectangle(
                    x=free_rect.x,
                    y=free_rect.y,
                    rows=new_rows,
                    cols=free_rect.cols,
                )
            )

    # ========================================================
    # 2. placed 矩形在 H 方向后面的区域
    # ========================================================

    if (
        placed_row_end < free_row_end
        and placed_row_end > free_rect.x
    ):

        new_rows = (
            free_row_end
            - placed_row_end
        )

        if new_rows > 0:
            result.append(
                FreeRectangle(
                    x=placed_row_end,
                    y=free_rect.y,
                    rows=new_rows,
                    cols=free_rect.cols,
                )
            )

    # ========================================================
    # 3. placed 矩形在 W 方向前面的区域
    # ========================================================

    if (
        placed_y > free_rect.y
        and placed_y < free_col_end
    ):

        new_cols = (
            placed_y - free_rect.y
        )

        if new_cols > 0:
            result.append(
                FreeRectangle(
                    x=free_rect.x,
                    y=free_rect.y,
                    rows=free_rect.rows,
                    cols=new_cols,
                )
            )

    # ========================================================
    # 4. placed 矩形在 W 方向后面的区域
    # ========================================================

    if (
        placed_col_end < free_col_end
        and placed_col_end > free_rect.y
    ):

        new_cols = (
            free_col_end
            - placed_col_end
        )

        if new_cols > 0:
            result.append(
                FreeRectangle(
                    x=free_rect.x,
                    y=placed_col_end,
                    rows=free_rect.rows,
                    cols=new_cols,
                )
            )

    return result


def remove_duplicate_free_rectangles(
    rectangles: list[FreeRectangle],
) -> list[FreeRectangle]:
    """
    删除几何位置和尺寸完全相同的重复 FreeRectangle。
    """

    seen: set[
        tuple[int, int, int, int]
    ] = set()

    unique: list[FreeRectangle] = []

    for rect in rectangles:

        key = rect.geometry_tuple()

        if key in seen:
            continue

        seen.add(key)
        unique.append(rect)

    return unique


def prune_contained_free_rectangles(
    rectangles: list[FreeRectangle],
) -> list[FreeRectangle]:
    """
    删除完全被其他 FreeRectangle 包含的冗余矩形。

    例如：

        Rect-A = 4096×2048
        Rect-B = 3072×1024

    如果 Rect-B 完全位于 Rect-A 内，
    那么 Rect-B 没有单独保留的意义。

    因为：

        能放入 Rect-B 的矩形，
        一定也能放入 Rect-A。

    因此删除 Rect-B。
    """

    rectangles = (
        remove_duplicate_free_rectangles(
            rectangles
        )
    )

    keep = [True] * len(rectangles)

    for i in range(len(rectangles)):

        if not keep[i]:
            continue

        for j in range(len(rectangles)):

            if i == j:
                continue

            if not keep[i]:
                break

            first = rectangles[i]
            second = rectangles[j]

            # 如果 second 完全包含 first，
            # 则 first 是冗余的。
            if second.contains_rectangle(first):
                keep[i] = False

    return [
        rect
        for rect, should_keep in zip(
            rectangles,
            keep,
        )
        if should_keep
    ]


def update_free_rectangles_after_placement(
    plane: Plane,
    candidate: PlacementCandidate,
) -> list[FreeRectangle]:
    """
    根据一次 PlacementCandidate，
    计算放置后的新 FreeRectangle 集合。

    注意：

    这个函数只计算结果，
    不直接修改 Plane。
    """

    if candidate.plane_id != plane.plane_id:
        raise MaxRectsError(
            "PlacementCandidate 的 plane_id "
            "与 Plane 不一致。"
        )

    new_free_rectangles: list[
        FreeRectangle
    ] = []

    # ========================================================
    # 每一个旧 FreeRectangle 都需要检查是否与新块重叠
    # ========================================================

    for free_rect in plane.free_rectangles:

        split_result = split_free_rectangle(
            free_rect=free_rect,

            placed_x=candidate.x,
            placed_y=candidate.y,

            placed_rows=candidate.placed_rows,
            placed_cols=candidate.placed_cols,
        )

        new_free_rectangles.extend(
            split_result
        )

    # ========================================================
    # 删除重复和被包含区域
    # ========================================================

    new_free_rectangles = (
        prune_contained_free_rectangles(
            new_free_rectangles
        )
    )

    return new_free_rectangles


def commit_placement(
    plane: Plane,
    candidate: PlacementCandidate,
    slot_id: int,
) -> PhysicalSlot:
    """
    正式将 PlacementCandidate 提交到 Plane。

    执行：

        PlacementCandidate
                ↓
        创建 PhysicalSlot
                ↓
        更新 FreeRectangle
                ↓
        保存到 Plane

    Returns:
        新生成的 PhysicalSlot。
    """

    if slot_id < 0:
        raise MaxRectsError(
            f"slot_id 不能为负数，当前为 {slot_id}。"
        )

    if candidate.plane_id != plane.plane_id:
        raise MaxRectsError(
            "candidate.plane_id 与 plane.plane_id 不一致："
            f"{candidate.plane_id} != {plane.plane_id}。"
        )

    # ========================================================
    # 1. 先计算新的空闲矩形集合
    # ========================================================

    new_free_rectangles = (
        update_free_rectangles_after_placement(
            plane=plane,
            candidate=candidate,
        )
    )

    # ========================================================
    # 2. 创建 PhysicalSlot
    # ========================================================

    slot = PhysicalSlot(
        slot_id=slot_id,
        plane_id=plane.plane_id,

        x=candidate.x,
        y=candidate.y,

        slot_rows=candidate.placed_rows,
        slot_cols=candidate.placed_cols,

        orientation_swapped=(
            candidate.orientation_swapped
        ),
    )

    # ========================================================
    # 3. 向 Plane 中加入槽位
    #
    # Plane.add_slot 会再次做：
    # - 越界检查；
    # - 与已有槽位的重叠检查。
    # ========================================================

    plane.add_slot(slot)

    # ========================================================
    # 4. 更新 FreeRectangle
    # ========================================================

    plane.replace_free_rectangles(
        new_free_rectangles
    )

    return slot


def place_shape_in_plane(
    plane: Plane,
    rows: int,
    cols: int,
    slot_id: int,
    allow_rotation: bool = True,
) -> PhysicalSlot | None:
    """
    方便使用的高级接口。

    在一个 Plane 中：

        1. 搜索最佳位置；
        2. 如果没有位置，返回 None；
        3. 如果有位置，直接提交并返回 PhysicalSlot。

    anonymous_packer.py 可以调用它，
    但如果需要比较多个 Plane 的候选分数，
    应先分别调用 find_best_position()，
    再从所有 Plane 中选全局最佳候选。
    """

    candidate = find_best_position(
        plane=plane,
        rows=rows,
        cols=cols,
        allow_rotation=allow_rotation,
    )

    if candidate is None:
        return None

    return commit_placement(
        plane=plane,
        candidate=candidate,
        slot_id=slot_id,
    )


def validate_free_rectangles(
    plane: Plane,
) -> None:
    """
    检查 Plane 当前所有 FreeRectangle 是否合法。

    这里只检查：
    - 不越界；
    - 尺寸为正。

    不检查 FreeRectangle 是否互相重叠，
    因为 MaxRects 允许空闲候选矩形重叠。
    """

    for rect in plane.free_rectangles:

        if rect.rows <= 0 or rect.cols <= 0:
            raise MaxRectsError(
                "发现非正尺寸 FreeRectangle："
                f"{rect.geometry_tuple()}。"
            )

        if (
            rect.x < 0
            or rect.y < 0
            or rect.row_end > plane.H
            or rect.col_end > plane.W
        ):
            raise MaxRectsError(
                "FreeRectangle 越界："
                f"{rect.geometry_tuple()}, "
                f"Plane={plane.H}×{plane.W}。"
            )


if __name__ == "__main__":

    from packing.plane import (
        create_empty_plane,
    )

    # ========================================================
    # 测试一个 4096×4096 Plane
    # ========================================================

    plane = create_empty_plane(
        plane_id=0,
        H=4096,
        W=4096,
    )

    print(
        "初始：",
        plane.summary(),
    )

    # ========================================================
    # 放入第一个匿名矩形：4096×2048
    # ========================================================

    candidate_0 = find_best_position(
        plane=plane,
        rows=4096,
        cols=2048,
        allow_rotation=True,
    )

    if candidate_0 is None:
        raise RuntimeError(
            "第一个匿名矩形不应该放置失败。"
        )

    print(
        "\nCandidate-0：",
        candidate_0,
    )

    slot_0 = commit_placement(
        plane=plane,
        candidate=candidate_0,
        slot_id=0,
    )

    print(
        "Slot-0：",
        slot_0.summary(),
    )

    print(
        "放置后：",
        plane.summary(),
    )

    print(
        "Free Rectangles："
    )

    for rect in plane.free_rectangles:
        print(
            " ",
            rect.geometry_tuple(),
        )

    # ========================================================
    # 再放入 3072×2048
    # ========================================================

    candidate_1 = find_best_position(
        plane=plane,
        rows=3072,
        cols=2048,
        allow_rotation=True,
    )

    if candidate_1 is None:
        raise RuntimeError(
            "第二个匿名矩形不应该放置失败。"
        )

    print(
        "\nCandidate-1：",
        candidate_1,
    )

    slot_1 = commit_placement(
        plane=plane,
        candidate=candidate_1,
        slot_id=1,
    )

    print(
        "Slot-1：",
        slot_1.summary(),
    )

    print(
        "最终 Plane：",
        plane.summary(),
    )

    print(
        "最终 Free Rectangles："
    )

    for rect in plane.free_rectangles:
        print(
            " ",
            rect.geometry_tuple(),
        )

    # ========================================================
    # 最终检查
    # ========================================================

    validate_free_rectangles(plane)
    plane.validate_layout()

    print(
        "\nMaxRects 基础测试通过。"
    )