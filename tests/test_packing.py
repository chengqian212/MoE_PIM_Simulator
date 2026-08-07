# tests/test_packing.py

import pytest

from packing.physical_slot import (
    PhysicalSlot,
    PhysicalSlotError,
    validate_slot_inside_plane,
    validate_slots_no_overlap,
)

from packing.plane import (
    FreeRectangle,
    Plane,
    create_empty_plane,
)

from packing.maxrects import (
    calculate_bssf_score,
    find_best_position,
    commit_placement,
    split_free_rectangle,
    prune_contained_free_rectangles,
    validate_free_rectangles,
)

from packing.anonymous_packer import (
    AnonymousBlock,
    build_anonymous_blocks,
    sort_anonymous_blocks,
    pack_anonymous_blocks,
)

from partition.partition_generator import (
    generate_original_template,
    generate_transposed_template,
)


# ============================================================
# PhysicalSlot 基础测试
# ============================================================


def test_physical_slot_basic():
    """
    一个合法槽位应该正确计算：
    - 面积
    - 结束坐标
    - size_key
    """

    slot = PhysicalSlot(
        slot_id=0,
        plane_id=0,
        x=0,
        y=0,
        slot_rows=4096,
        slot_cols=2048,
        orientation_swapped=False,
    )

    assert slot.row_end == 4096
    assert slot.col_end == 2048

    assert slot.area == 4096 * 2048

    assert slot.size_key == (
        2048,
        4096,
    )


def test_slot_inside_plane():
    """
    4096×2048 放在 4096×4096 中应该合法。
    """

    slot = PhysicalSlot(
        slot_id=0,
        plane_id=0,
        x=0,
        y=0,
        slot_rows=4096,
        slot_cols=2048,
    )

    assert (
        slot.fits_inside_plane(
            H=4096,
            W=4096,
        )
        is True
    )

    validate_slot_inside_plane(
        slot=slot,
        H=4096,
        W=4096,
    )


def test_slot_out_of_bounds():
    """
    从 y=3000 开始放 2048 列，
    会超过 W=4096。
    """

    slot = PhysicalSlot(
        slot_id=0,
        plane_id=0,
        x=0,
        y=3000,
        slot_rows=4096,
        slot_cols=2048,
    )

    assert (
        slot.fits_inside_plane(
            H=4096,
            W=4096,
        )
        is False
    )

    with pytest.raises(
        PhysicalSlotError
    ):
        validate_slot_inside_plane(
            slot=slot,
            H=4096,
            W=4096,
        )


def test_slot_overlap_detection():
    """
    同一个 Plane 中两个矩形发生重叠时，
    必须能够检测出来。
    """

    slot_0 = PhysicalSlot(
        slot_id=0,
        plane_id=0,
        x=0,
        y=0,
        slot_rows=3000,
        slot_cols=3000,
    )

    slot_1 = PhysicalSlot(
        slot_id=1,
        plane_id=0,
        x=2000,
        y=2000,
        slot_rows=2000,
        slot_cols=2000,
    )

    assert slot_0.overlaps(slot_1)

    with pytest.raises(
        PhysicalSlotError
    ):
        validate_slots_no_overlap(
            [slot_0, slot_1]
        )


def test_slots_on_different_planes_do_not_overlap():
    """
    即使二维坐标完全一样，
    不同 Plane 中的槽位也不算重叠。
    """

    slot_0 = PhysicalSlot(
        slot_id=0,
        plane_id=0,
        x=0,
        y=0,
        slot_rows=4096,
        slot_cols=2048,
    )

    slot_1 = PhysicalSlot(
        slot_id=1,
        plane_id=1,
        x=0,
        y=0,
        slot_rows=4096,
        slot_cols=2048,
    )

    assert slot_0.overlaps(slot_1) is False


# ============================================================
# Plane 基础测试
# ============================================================


def test_empty_plane():
    """
    新建 Plane 时应该只有一个完整 FreeRectangle。
    """

    plane = create_empty_plane(
        plane_id=0,
        H=4096,
        W=4096,
    )

    assert plane.slot_count == 0
    assert plane.free_rectangle_count == 1

    free_rect = (
        plane.free_rectangles[0]
    )

    assert free_rect.geometry_tuple() == (
        0,
        0,
        4096,
        4096,
    )

    assert plane.used_area == 0
    assert plane.unused_area == 4096 * 4096


def test_plane_add_two_non_overlapping_slots():
    """
    两个 4096×2048 可以并排填满一个
    4096×4096 Plane。
    """

    plane = create_empty_plane(
        plane_id=0,
        H=4096,
        W=4096,
    )

    slot_0 = PhysicalSlot(
        slot_id=0,
        plane_id=0,
        x=0,
        y=0,
        slot_rows=4096,
        slot_cols=2048,
    )

    slot_1 = PhysicalSlot(
        slot_id=1,
        plane_id=0,
        x=0,
        y=2048,
        slot_rows=4096,
        slot_cols=2048,
    )

    plane.add_slot(slot_0)
    plane.add_slot(slot_1)

    assert plane.slot_count == 2

    assert plane.used_area == (
        4096 * 4096
    )

    assert plane.unused_area == 0

    assert plane.utilization == 1.0

    plane.validate_layout()


# ============================================================
# BSSF 测试
# ============================================================


def test_bssf_score():
    """
    FreeRectangle：
        4096×4096

    待放：
        4096×2048

    剩余：
        0
        2048

    所以：

        short = 0
        long = 2048
    """

    free_rect = FreeRectangle(
        x=0,
        y=0,
        rows=4096,
        cols=4096,
    )

    score = calculate_bssf_score(
        free_rect=free_rect,
        rows=4096,
        cols=2048,
    )

    assert score == (
        0,
        2048,
    )


def test_bssf_cannot_fit():
    """
    7168×2048 不能原方向放入 4096×8192。
    """

    free_rect = FreeRectangle(
        x=0,
        y=0,
        rows=4096,
        cols=8192,
    )

    score = calculate_bssf_score(
        free_rect=free_rect,
        rows=7168,
        cols=2048,
    )

    assert score is None


# ============================================================
# 旋转测试
# ============================================================


def test_find_best_position_with_rotation():
    """
    7168×2048：

    原方向：
        7168 > H=4096
        放不下

    旋转后：
        2048×7168
        可以放进 4096×8192

    所以必须 orientation_swapped=True。
    """

    plane = create_empty_plane(
        plane_id=0,
        H=4096,
        W=8192,
    )

    candidate = find_best_position(
        plane=plane,
        rows=7168,
        cols=2048,
        allow_rotation=True,
    )

    assert candidate is not None

    assert (
        candidate.orientation_swapped
        is True
    )

    assert candidate.placed_rows == 2048
    assert candidate.placed_cols == 7168


def test_find_position_without_rotation_fails():
    """
    禁止旋转后，
    7168×2048 无法进入 4096×8192。
    """

    plane = create_empty_plane(
        plane_id=0,
        H=4096,
        W=8192,
    )

    candidate = find_best_position(
        plane=plane,
        rows=7168,
        cols=2048,
        allow_rotation=False,
    )

    assert candidate is None


# ============================================================
# FreeRectangle 切分测试
# ============================================================


def test_split_free_rectangle():
    """
    在 4096×4096 空区域左边放入：

        4096×2048

    理论上至少应该产生剩余的：

        4096×2048

    位于：

        x=0
        y=2048
    """

    free_rect = FreeRectangle(
        x=0,
        y=0,
        rows=4096,
        cols=4096,
    )

    new_rects = split_free_rectangle(
        free_rect=free_rect,

        placed_x=0,
        placed_y=0,

        placed_rows=4096,
        placed_cols=2048,
    )

    geometries = {
        rect.geometry_tuple()
        for rect in new_rects
    }

    assert (
        0,
        2048,
        4096,
        2048,
    ) in geometries


def test_prune_contained_free_rectangles():
    """
    小矩形完全包含在大矩形内，
    prune 后应只留下大矩形。
    """

    big = FreeRectangle(
        x=0,
        y=0,
        rows=4096,
        cols=4096,
    )

    small = FreeRectangle(
        x=0,
        y=0,
        rows=2048,
        cols=2048,
    )

    result = (
        prune_contained_free_rectangles(
            [big, small]
        )
    )

    assert len(result) == 1
    assert result[0] == big


# ============================================================
# commit_placement 测试
# ============================================================


def test_commit_two_blocks_fill_plane():
    """
    连续放两个 4096×2048，
    应该正好填满一个 4096×4096 Plane。
    """

    plane = create_empty_plane(
        plane_id=0,
        H=4096,
        W=4096,
    )

    candidate_0 = find_best_position(
        plane=plane,
        rows=4096,
        cols=2048,
        allow_rotation=True,
    )

    assert candidate_0 is not None

    commit_placement(
        plane=plane,
        candidate=candidate_0,
        slot_id=0,
    )

    candidate_1 = find_best_position(
        plane=plane,
        rows=4096,
        cols=2048,
        allow_rotation=True,
    )

    assert candidate_1 is not None

    commit_placement(
        plane=plane,
        candidate=candidate_1,
        slot_id=1,
    )

    assert plane.slot_count == 2
    assert plane.used_area == plane.area
    assert plane.unused_area == 0

    plane.validate_layout()
    validate_free_rectangles(plane)


# ============================================================
# AnonymousBlock 排序
# ============================================================


def test_anonymous_block_sorting():
    """
    应按照：

    1. 面积降序；
    2. 最长边降序；
    3. 最短边降序；
    4. block_id

    排序。
    """

    blocks = [
        AnonymousBlock(
            anonymous_block_id=0,
            rows=100,
            cols=100,
            source_template_chunk_id=0,
        ),

        AnonymousBlock(
            anonymous_block_id=1,
            rows=400,
            cols=100,
            source_template_chunk_id=0,
        ),

        AnonymousBlock(
            anonymous_block_id=2,
            rows=200,
            cols=200,
            source_template_chunk_id=0,
        ),
    ]

    sorted_blocks = (
        sort_anonymous_blocks(
            blocks
        )
    )

    # block-1:
    # 400×100 = 40000
    #
    # block-2:
    # 200×200 = 40000
    #
    # 面积相同，但 block-1 最长边更长。
    #
    # block-0:
    # 10000

    assert [
        block.anonymous_block_id
        for block in sorted_blocks
    ] == [
        1,
        2,
        0,
    ]


# ============================================================
# Template → AnonymousBlock
# ============================================================


def test_build_anonymous_blocks():
    """
    H=W=4096：

    一个矩阵：
        2 个 chunk

    matrix_count=3：

        应生成 6 个匿名块。
    """

    template = generate_original_template(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=4096,
    )

    blocks = build_anonymous_blocks(
        template=template,
        matrix_count=3,
    )

    assert len(blocks) == 6

    assert sum(
        block.area
        for block in blocks
    ) == (
        3
        * 7168
        * 2048
    )


# ============================================================
# 完整 AnonymousPacker：
# 4096×4096
# ============================================================


def test_pack_two_matrices_4096_4096():
    """
    每个 7168×2048 被切成：

        4096×2048
        3072×2048

    两个矩阵共有：

        2 个 4096×2048
        2 个 3072×2048

    MaxRects 应能使用 2 个 Plane：

        Plane-0：
            4096×2048
            4096×2048

        Plane-1：
            3072×2048
            3072×2048
    """

    template = generate_original_template(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=4096,
    )

    result = pack_anonymous_blocks(
        template=template,
        matrix_count=2,
        H=4096,
        W=4096,
        allow_rotation=True,
    )

    assert result.slot_count == 4

    assert result.plane_count == 2

    assert result.total_block_area == (
        2
        * 7168
        * 2048
    )

    for plane in result.planes:
        plane.validate_layout()


# ============================================================
# 完整 AnonymousPacker：
# 整块旋转
# ============================================================


def test_pack_whole_matrix_with_rotation():
    """
    H=4096
    W=8192

    转置方向模板允许整个：

        7168×2048

    不切分。

    第三步实际放置时应旋转为：

        2048×7168

    两个这样的矩形可以在 H 方向堆叠，
    正好进入同一个 4096×8192 Plane。
    """

    template = generate_transposed_template(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=8192,
    )

    assert template.chunk_count == 1

    result = pack_anonymous_blocks(
        template=template,
        matrix_count=2,
        H=4096,
        W=8192,
        allow_rotation=True,
    )

    assert result.slot_count == 2

    assert result.plane_count == 1

    assert (
        result.orientation_swapped_count
        == 2
    )

    for slot in result.slots:

        assert slot.slot_rows == 2048
        assert slot.slot_cols == 7168

        assert (
            slot.orientation_swapped
            is True
        )


# ============================================================
# 面积守恒
# ============================================================


@pytest.mark.parametrize(
    "matrix_count",
    [
        1,
        2,
        4,
        8,
    ],
)
def test_packing_area_preserved(
    matrix_count,
):
    """
    无论用了几个 Plane，

    PhysicalSlot 总面积始终必须等于：

        matrix_count × 7168 × 2048
    """

    template = generate_original_template(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=4096,
    )

    result = pack_anonymous_blocks(
        template=template,
        matrix_count=matrix_count,
        H=4096,
        W=4096,
        allow_rotation=True,
    )

    expected_area = (
        matrix_count
        * 7168
        * 2048
    )

    actual_area = sum(
        slot.area
        for slot in result.slots
    )

    assert actual_area == expected_area

    assert (
        result.total_block_area
        == expected_area
    )


# ============================================================
# Slot ID 唯一
# ============================================================


def test_slot_ids_unique():
    """
    所有 PhysicalSlot 的 slot_id 必须全局唯一。
    """

    template = generate_original_template(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=4096,
    )

    result = pack_anonymous_blocks(
        template=template,
        matrix_count=8,
        H=4096,
        W=4096,
        allow_rotation=True,
    )

    slot_ids = [
        slot.slot_id
        for slot in result.slots
    ]

    assert (
        len(slot_ids)
        == len(set(slot_ids))
    )


# ============================================================
# 所有 Plane 最终必须无重叠、无越界
# ============================================================


def test_all_planes_valid():
    """
    对一个稍大的小规模测试：

        matrix_count = 16

    检查最终所有 Plane。
    """

    template = generate_original_template(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=4096,
    )

    result = pack_anonymous_blocks(
        template=template,
        matrix_count=16,
        H=4096,
        W=4096,
        allow_rotation=True,
    )

    for plane in result.planes:

        plane.validate_layout()

        validate_free_rectangles(
            plane
        )

        for slot in plane.slots:

            assert (
                slot.fits_inside_plane(
                    H=4096,
                    W=4096,
                )
                is True
            )


# ============================================================
# P 不可能低于面积理论下界
# ============================================================


def test_plane_count_not_below_area_lower_bound():
    """
    检查：

        P >= ceil(S / (H*W))

    这里不要求 MaxRects 一定达到理论下界，
    只要求它不可能比理论下界还低。
    """

    from math import ceil

    template = generate_original_template(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=4096,
    )

    result = pack_anonymous_blocks(
        template=template,
        matrix_count=10,
        H=4096,
        W=4096,
        allow_rotation=True,
    )

    S = (
        10
        * 7168
        * 2048
    )

    P_lower = ceil(
        S
        / (
            4096
            * 4096
        )
    )

    assert (
        result.plane_count
        >= P_lower
    )