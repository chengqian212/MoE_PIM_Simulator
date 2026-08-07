# packing/plane.py
"""
匿名二维平面 Plane 与空闲矩形 FreeRectangle。

第三步二维装箱时，每个 Plane 表示一个独立的 H×W 平面。

Plane 负责维护：

1. 当前已经生成的 PhysicalSlot；
2. 当前剩余的 FreeRectangle；
3. 基础边界检查；
4. 空闲面积、已用面积等统计。

注意：
- 本文件不决定 MaxRects 的“最佳位置”；
- 具体候选搜索和 BSSF 评分放在 maxrects.py；
- 不出现 layer_id、expert_id、matrix_name；
- 不出现 subcube_id、z；
- 一个 Plane 目前还不知道最终属于哪个 Sub-Cube。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from packing.physical_slot import (
    PhysicalSlot,
    PhysicalSlotError,
    validate_slot_inside_plane,
    validate_slots_no_overlap,
)


class PlaneError(ValueError):
    """Plane 或 FreeRectangle 非法时抛出的异常。"""


@dataclass(frozen=True, slots=True)
class FreeRectangle:
    """
    Plane 中当前仍可使用的一块矩形区域。

    坐标定义与 PhysicalSlot 一致：

        x：沿 H 方向
        y：沿 W 方向

    区域采用左闭右开：

        [x, x + rows)
        [y, y + cols)

    FreeRectangle 只是 MaxRects 算法中的内部状态，
    并不是最终物理槽位。
    """

    x: int
    y: int

    rows: int
    cols: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.x < 0:
            raise PlaneError(
                f"FreeRectangle.x 不能为负数，当前为 {self.x}。"
            )

        if self.y < 0:
            raise PlaneError(
                f"FreeRectangle.y 不能为负数，当前为 {self.y}。"
            )

        if self.rows <= 0:
            raise PlaneError(
                f"FreeRectangle.rows 必须大于 0，当前为 {self.rows}。"
            )

        if self.cols <= 0:
            raise PlaneError(
                f"FreeRectangle.cols 必须大于 0，当前为 {self.cols}。"
            )

    @property
    def row_end(self) -> int:
        return self.x + self.rows

    @property
    def col_end(self) -> int:
        return self.y + self.cols

    @property
    def area(self) -> int:
        return self.rows * self.cols

    @property
    def longest_side(self) -> int:
        return max(self.rows, self.cols)

    @property
    def shortest_side(self) -> int:
        return min(self.rows, self.cols)

    def fits(
        self,
        rows: int,
        cols: int,
    ) -> bool:
        """
        判断 rows×cols 是否能直接放入该空闲矩形。

        这里只判断当前方向。
        是否尝试旋转由 maxrects.py 控制。
        """

        if rows <= 0 or cols <= 0:
            raise PlaneError(
                f"待放矩形尺寸必须为正数，当前为 {rows}×{cols}。"
            )

        return (
            rows <= self.rows
            and cols <= self.cols
        )

    def contains_rectangle(
        self,
        other: "FreeRectangle",
    ) -> bool:
        """
        判断当前 FreeRectangle 是否完全包含另一个 FreeRectangle。
        """

        return (
            self.x <= other.x
            and self.y <= other.y
            and self.row_end >= other.row_end
            and self.col_end >= other.col_end
        )

    def overlaps_region(
        self,
        x: int,
        y: int,
        rows: int,
        cols: int,
    ) -> bool:
        """
        判断当前空闲矩形是否与给定矩形区域发生重叠。
        """

        region_row_end = x + rows
        region_col_end = y + cols

        row_overlap = (
            self.x < region_row_end
            and x < self.row_end
        )

        col_overlap = (
            self.y < region_col_end
            and y < self.col_end
        )

        return row_overlap and col_overlap

    def geometry_tuple(
        self,
    ) -> tuple[int, int, int, int]:
        return (
            self.x,
            self.y,
            self.rows,
            self.cols,
        )


@dataclass(slots=True)
class Plane:
    """
    一个匿名 H×W 二维平面。

    Plane 创建时默认只有一个 FreeRectangle：

        (0, 0, H, W)

    随着匿名矩形被放入：

        free_rectangles 会不断被 MaxRects 更新；
        slots 会不断增加。

    注意：
    plane_id 只是匿名平面编号。

    第四步才会决定：

        Plane-k → SubCube-j
        Plane-k → z=t
    """

    plane_id: int

    H: int
    W: int

    slots: list[PhysicalSlot] = field(
        default_factory=list
    )

    free_rectangles: list[FreeRectangle] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self.validate_basic()

        if not self.free_rectangles:
            self.free_rectangles.append(
                FreeRectangle(
                    x=0,
                    y=0,
                    rows=self.H,
                    cols=self.W,
                )
            )

    def validate_basic(self) -> None:
        """检查 Plane 的基础配置。"""

        if self.plane_id < 0:
            raise PlaneError(
                f"plane_id 不能为负数，当前为 {self.plane_id}。"
            )

        if self.H <= 0:
            raise PlaneError(
                f"H 必须大于 0，当前为 {self.H}。"
            )

        if self.W <= 0:
            raise PlaneError(
                f"W 必须大于 0，当前为 {self.W}。"
            )

    @property
    def area(self) -> int:
        """整个 Plane 的面积。"""
        return self.H * self.W

    @property
    def used_area(self) -> int:
        """已经生成的匿名槽位总面积。"""

        return sum(
            slot.area
            for slot in self.slots
        )

    @property
    def unused_area(self) -> int:
        """
        尚未被 PhysicalSlot 使用的真实面积。

        注意：
        这里直接使用：

            Plane 总面积 - 已用槽位面积

        而不是 free_rectangles 面积之和。

        因为 MaxRects 的 free_rectangles 之间可能重叠，
        它们只是“候选空闲矩形”，不能直接求和。
        """

        return self.area - self.used_area

    @property
    def utilization(self) -> float:
        """Plane 的实际空间利用率。"""

        if self.area == 0:
            return 0.0

        return self.used_area / self.area

    @property
    def slot_count(self) -> int:
        """当前匿名槽位数量。"""

        return len(self.slots)

    @property
    def free_rectangle_count(self) -> int:
        """当前 MaxRects 空闲矩形数量。"""

        return len(self.free_rectangles)

    def add_slot(
        self,
        slot: PhysicalSlot,
    ) -> None:
        """
        向 Plane 中提交一个 PhysicalSlot。

        注意：
        这个函数只负责最终提交和安全检查。

        FreeRectangle 如何切分、更新，
        由 maxrects.py 负责。
        """

        if slot.plane_id != self.plane_id:
            raise PlaneError(
                "PhysicalSlot 的 plane_id 与 Plane 不一致："
                f"slot.plane_id={slot.plane_id}, "
                f"plane.plane_id={self.plane_id}。"
            )

        validate_slot_inside_plane(
            slot=slot,
            H=self.H,
            W=self.W,
        )

        for existing in self.slots:
            if slot.overlaps(existing):
                raise PlaneError(
                    f"新增槽位 {slot.slot_id} 与已有槽位 "
                    f"{existing.slot_id} 在 Plane-{self.plane_id} "
                    "中发生重叠。"
                )

        self.slots.append(slot)

    def replace_free_rectangles(
        self,
        rectangles: list[FreeRectangle],
    ) -> None:
        """
        用 MaxRects 更新后的空闲矩形集合替换当前状态。

        maxrects.py 在完成一次放置后，会生成新的列表，
        然后通过该函数统一提交。
        """

        for rect in rectangles:
            if (
                rect.x < 0
                or rect.y < 0
                or rect.row_end > self.H
                or rect.col_end > self.W
            ):
                raise PlaneError(
                    "发现越出 Plane 边界的 FreeRectangle："
                    f"{rect.geometry_tuple()}, "
                    f"plane={self.H}×{self.W}。"
                )

        self.free_rectangles = rectangles

    def can_fit_shape(
        self,
        rows: int,
        cols: int,
        allow_rotation: bool = True,
    ) -> bool:
        """
        快速判断当前 Plane 是否至少存在一个 FreeRectangle
        可以容纳 rows×cols。

        这里只做可行性判断，不做 BSSF 排序。
        """

        if rows <= 0 or cols <= 0:
            raise PlaneError(
                f"矩形尺寸必须为正数，当前为 {rows}×{cols}。"
            )

        for rect in self.free_rectangles:

            if rect.fits(
                rows=rows,
                cols=cols,
            ):
                return True

            if (
                allow_rotation
                and rows != cols
                and rect.fits(
                    rows=cols,
                    cols=rows,
                )
            ):
                return True

        return False

    def validate_layout(self) -> None:
        """
        对当前 Plane 的最终槽位布局进行完整检查。

        检查：
        - 所有槽位都在 Plane 内；
        - 任意两个槽位不重叠；
        - used_area 不超过 Plane 面积。
        """

        for slot in self.slots:
            validate_slot_inside_plane(
                slot=slot,
                H=self.H,
                W=self.W,
            )

        try:
            validate_slots_no_overlap(
                self.slots
            )
        except PhysicalSlotError as exc:
            raise PlaneError(str(exc)) from exc

        if self.used_area > self.area:
            raise PlaneError(
                f"Plane-{self.plane_id} 已用面积 "
                f"{self.used_area} 超过平面总面积 {self.area}。"
            )

    def size_histogram(
        self,
    ) -> dict[tuple[int, int], int]:
        """
        统计该 Plane 中各类匿名槽位数量。

        返回的 key 是与方向无关的 size_key。
        """

        histogram: dict[
            tuple[int, int],
            int
        ] = {}

        for slot in self.slots:
            key = slot.size_key

            histogram[key] = (
                histogram.get(key, 0)
                + 1
            )

        return histogram

    def signature(
        self,
    ) -> tuple[tuple[int, int], ...]:
        """
        返回该 Plane 的匿名槽位组成签名。

        例如：

            两个 4096×2048 槽位

        返回：

            (
                (2048, 4096),
                (2048, 4096),
            )

        第四步如果需要比较不同匿名平面的槽位组成，
        可以直接使用这个 signature。

        注意：
        此处只是描述 Plane，
        第三步仍然不会把 Plane 分给 Sub-Cube。
        """

        return tuple(
            sorted(
                slot.size_key
                for slot in self.slots
            )
        )

    def summary(self) -> str:
        """返回 Plane 的简短描述。"""

        return (
            f"Plane-{self.plane_id}: "
            f"size={self.H}×{self.W}, "
            f"slots={self.slot_count}, "
            f"free_rects={self.free_rectangle_count}, "
            f"used_area={self.used_area}, "
            f"utilization={self.utilization:.4%}"
        )


def create_empty_plane(
    plane_id: int,
    H: int,
    W: int,
) -> Plane:
    """
    创建一个标准空 Plane。

    初始状态：

        slots = []

        free_rectangles = [
            FreeRectangle(
                x=0,
                y=0,
                rows=H,
                cols=W,
            )
        ]
    """

    return Plane(
        plane_id=plane_id,
        H=H,
        W=W,
    )


if __name__ == "__main__":

    # ========================================================
    # 基础测试
    # ========================================================

    plane = create_empty_plane(
        plane_id=0,
        H=4096,
        W=4096,
    )

    print(
        "初始 Plane："
    )

    print(
        plane.summary()
    )

    print(
        "初始 FreeRectangle：",
        plane.free_rectangles[0].geometry_tuple(),
    )

    # --------------------------------------------------------
    # 手动加入两个不重叠槽位
    # --------------------------------------------------------

    slot_0 = PhysicalSlot(
        slot_id=0,
        plane_id=0,
        x=0,
        y=0,
        slot_rows=4096,
        slot_cols=2048,
        orientation_swapped=False,
    )

    slot_1 = PhysicalSlot(
        slot_id=1,
        plane_id=0,
        x=0,
        y=2048,
        slot_rows=3072,
        slot_cols=2048,
        orientation_swapped=False,
    )

    plane.add_slot(slot_0)
    plane.add_slot(slot_1)

    print(
        "\n加入槽位后："
    )

    print(
        plane.summary()
    )

    print(
        "size_histogram：",
        plane.size_histogram(),
    )

    print(
        "signature：",
        plane.signature(),
    )

    plane.validate_layout()

    print(
        "\nPlane 布局检查通过。"
    )