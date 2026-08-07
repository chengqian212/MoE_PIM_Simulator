# packing/physical_slot.py
"""
匿名物理槽位 PhysicalSlot 的数据结构定义。

第三步中，匿名矩形经过二维装箱后，会在某个 H×W 平面中
形成一个实际占用位置。

这个位置就是 PhysicalSlot。

PhysicalSlot 只描述：
1. 位于哪个匿名 plane；
2. 在平面中的二维坐标 x、y；
3. 占用的 rows、cols；
4. 对应的无方向尺寸 size_key；
5. 匿名矩形在放置时是否交换了方向。

注意：
- 这里仍然不知道将来这个槽位存放的是 gate / up / down；
- 不知道属于哪一层；
- 不知道属于哪个 Expert；
- 不知道最终属于哪个 Sub-Cube；
- 不知道 z；
- 所以 PhysicalSlot 不是最终 Weight-Cube 映射结果。
"""

from __future__ import annotations

from dataclasses import dataclass

from model_geometry import SizeKey, make_size_key


class PhysicalSlotError(ValueError):
    """PhysicalSlot 数据不合法时抛出的异常。"""


@dataclass(frozen=True, slots=True)
class PhysicalSlot:
    """
    第三步二维装箱产生的匿名物理槽位。

    坐标约定：

        x：沿 H 方向的位置
        y：沿 W 方向的位置

    因此一个槽位满足：

        0 <= x
        0 <= y
        x + slot_rows <= H
        y + slot_cols <= W

    这里采用左闭右开区域：

        [x, x + slot_rows)
        [y, y + slot_cols)

    这样后续判断矩形重叠更方便。
    """

    # 全局唯一匿名槽位编号
    slot_id: int

    # 所属匿名二维平面编号
    plane_id: int

    # 在平面 H 方向上的起始坐标
    x: int

    # 在平面 W 方向上的起始坐标
    y: int

    # 实际物理占用尺寸
    slot_rows: int
    slot_cols: int

    # 匿名块在第三步放置时是否交换方向
    orientation_swapped: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """检查 PhysicalSlot 的基础字段是否合法。"""

        if self.slot_id < 0:
            raise PhysicalSlotError(
                f"slot_id 不能为负数，当前为 {self.slot_id}。"
            )

        if self.plane_id < 0:
            raise PhysicalSlotError(
                f"plane_id 不能为负数，当前为 {self.plane_id}。"
            )

        if self.x < 0:
            raise PhysicalSlotError(
                f"x 不能为负数，当前为 {self.x}。"
            )

        if self.y < 0:
            raise PhysicalSlotError(
                f"y 不能为负数，当前为 {self.y}。"
            )

        if self.slot_rows <= 0:
            raise PhysicalSlotError(
                "slot_rows 必须大于 0，"
                f"当前为 {self.slot_rows}。"
            )

        if self.slot_cols <= 0:
            raise PhysicalSlotError(
                "slot_cols 必须大于 0，"
                f"当前为 {self.slot_cols}。"
            )

    @property
    def row_end(self) -> int:
        """
        槽位沿 H 方向的结束位置。

        区间：
            [x, row_end)
        """
        return self.x + self.slot_rows

    @property
    def col_end(self) -> int:
        """
        槽位沿 W 方向的结束位置。

        区间：
            [y, col_end)
        """
        return self.y + self.slot_cols

    @property
    def area(self) -> int:
        """槽位实际占用面积。"""
        return self.slot_rows * self.slot_cols

    @property
    def size_key(self) -> SizeKey:
        """
        返回与方向无关的尺寸类别。

        例如：

            slot_rows = 4096
            slot_cols = 2048

        与：

            slot_rows = 2048
            slot_cols = 4096

        都得到：

            (2048, 4096)

        第四步会根据这个 size_key，
        把真实逻辑 Weight-Cube 与匿名槽位进行匹配。
        """
        return make_size_key(
            self.slot_rows,
            self.slot_cols,
        )

    @property
    def longest_side(self) -> int:
        """槽位最长边。"""
        return max(
            self.slot_rows,
            self.slot_cols,
        )

    @property
    def shortest_side(self) -> int:
        """槽位最短边。"""
        return min(
            self.slot_rows,
            self.slot_cols,
        )

    def fits_inside_plane(
        self,
        H: int,
        W: int,
    ) -> bool:
        """
        判断该槽位是否完全位于 H×W 平面中。
        """

        if H <= 0 or W <= 0:
            raise PhysicalSlotError(
                f"平面尺寸必须为正数，当前为 {H}×{W}。"
            )

        return (
            self.x >= 0
            and self.y >= 0
            and self.row_end <= H
            and self.col_end <= W
        )

    def overlaps(
        self,
        other: "PhysicalSlot",
    ) -> bool:
        """
        判断两个 PhysicalSlot 是否重叠。

        只有位于同一个 plane_id 中时才需要检查。

        两个矩形仅边界接触不算重叠。
        """

        if self.plane_id != other.plane_id:
            return False

        row_overlap = (
            self.x < other.row_end
            and other.x < self.row_end
        )

        col_overlap = (
            self.y < other.col_end
            and other.y < self.col_end
        )

        return row_overlap and col_overlap

    def can_host_shape(
        self,
        rows: int,
        cols: int,
        allow_rotation: bool = True,
    ) -> bool:
        """
        判断该匿名槽位是否能够容纳某个逻辑矩形。

        这个函数主要供第四步使用。

        注意：
        PhysicalSlot 的尺寸已经由第三步固定，
        第四步不能再改变它。

        因此只有两种情况合法：

        1. 逻辑矩形与槽位方向完全相同；
        2. 允许旋转，并且交换 rows、cols 后完全相同。

        这里不是判断“小矩形能不能放进大槽位”，
        而是要求尺寸严格匹配。

        原因是第三步匿名槽位数量是按照第二步的
        匿名块需求精确生成的，第四步只做身份绑定，
        不应重新制造空间浪费。
        """

        if rows <= 0 or cols <= 0:
            raise PhysicalSlotError(
                f"逻辑矩形尺寸必须为正数，当前为 {rows}×{cols}。"
            )

        normal_match = (
            rows == self.slot_rows
            and cols == self.slot_cols
        )

        rotated_match = (
            allow_rotation
            and rows == self.slot_cols
            and cols == self.slot_rows
        )

        return normal_match or rotated_match

    def logical_rotation_required(
        self,
        rows: int,
        cols: int,
    ) -> bool:
        """
        判断某个真实逻辑矩形绑定到该槽位时是否需要旋转。

        返回：
            False：
                rows×cols 与槽位完全一致。

            True：
                cols×rows 与槽位完全一致。

        如果两种情况都不匹配，则抛出异常。

        注意：
        这是第四步真正绑定逻辑 Weight-Cube 时才有意义的判断。
        """

        if (
            rows == self.slot_rows
            and cols == self.slot_cols
        ):
            return False

        if (
            rows == self.slot_cols
            and cols == self.slot_rows
        ):
            return True

        raise PhysicalSlotError(
            "逻辑矩形无法与该槽位匹配："
            f"logical={rows}×{cols}, "
            f"slot={self.slot_rows}×{self.slot_cols}。"
        )

    def geometry_tuple(
        self,
    ) -> tuple[int, int, int, int]:
        """
        返回槽位的二维几何描述：

            (x, y, slot_rows, slot_cols)
        """

        return (
            self.x,
            self.y,
            self.slot_rows,
            self.slot_cols,
        )

    def summary(self) -> str:
        """返回简洁文本描述。"""

        return (
            f"PhysicalSlot-{self.slot_id}: "
            f"plane={self.plane_id}, "
            f"pos=({self.x},{self.y}), "
            f"size={self.slot_rows}×{self.slot_cols}, "
            f"size_key={self.size_key}, "
            f"orientation_swapped="
            f"{self.orientation_swapped}"
        )


def slots_overlap(
    first: PhysicalSlot,
    second: PhysicalSlot,
) -> bool:
    """
    独立辅助函数：
    判断两个 PhysicalSlot 是否重叠。
    """

    return first.overlaps(second)


def validate_slot_inside_plane(
    slot: PhysicalSlot,
    H: int,
    W: int,
) -> None:
    """
    检查单个 PhysicalSlot 是否位于 H×W 平面内。

    不合法则直接抛出异常。
    """

    if not slot.fits_inside_plane(
        H=H,
        W=W,
    ):
        raise PhysicalSlotError(
            f"槽位 {slot.slot_id} 越界："
            f"plane={slot.plane_id}, "
            f"position=({slot.x},{slot.y}), "
            f"size={slot.slot_rows}×{slot.slot_cols}, "
            f"plane_size={H}×{W}。"
        )


def validate_slots_no_overlap(
    slots: list[PhysicalSlot],
) -> None:
    """
    检查一组槽位之间是否存在重叠。

    通常传入同一个 Plane 中的所有槽位。
    """

    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):

            first = slots[i]
            second = slots[j]

            if first.overlaps(second):
                raise PhysicalSlotError(
                    f"槽位 {first.slot_id} 与 "
                    f"槽位 {second.slot_id} "
                    f"在 Plane-{first.plane_id} 中发生重叠。"
                )


if __name__ == "__main__":

    # ========================================================
    # 基础示例
    # ========================================================

    H = 4096
    W = 4096

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

    print(slot_0.summary())
    print(slot_1.summary())

    print(
        "\nSlot-0 是否位于平面内：",
        slot_0.fits_inside_plane(
            H=H,
            W=W,
        ),
    )

    print(
        "Slot-1 是否位于平面内：",
        slot_1.fits_inside_plane(
            H=H,
            W=W,
        ),
    )

    print(
        "两个槽位是否重叠：",
        slot_0.overlaps(slot_1),
    )

    print(
        "\nSlot-0 能否容纳 4096×2048：",
        slot_0.can_host_shape(
            rows=4096,
            cols=2048,
        ),
    )

    print(
        "Slot-0 能否容纳 2048×4096：",
        slot_0.can_host_shape(
            rows=2048,
            cols=4096,
        ),
    )

    print(
        "2048×4096 绑定 Slot-0 是否需要旋转：",
        slot_0.logical_rotation_required(
            rows=2048,
            cols=4096,
        ),
    )

    validate_slot_inside_plane(
        slot=slot_0,
        H=H,
        W=W,
    )

    validate_slot_inside_plane(
        slot=slot_1,
        H=H,
        W=W,
    )

    validate_slots_no_overlap(
        [slot_0, slot_1]
    )

    print("\nPhysicalSlot 检查通过。")