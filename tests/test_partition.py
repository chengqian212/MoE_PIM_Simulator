# tests/test_partition.py

import pytest

from partition.partition_generator import (
    generate_original_template,
    generate_transposed_template,
    generate_partition_templates,
)

from partition.partition_validator import (
    validate_partition_template,
    validate_partition_templates,
    validate_no_duplicate_templates,
    find_overlapping_pairs,
    find_out_of_bounds_chunks,
)

from partition.partition_template import (
    PartitionTemplate,
    TemplateChunk,
)


# ============================================================
# 基础：4096 × 4096
# ============================================================


def test_original_template_4096_4096():
    """
    7168×2048 在 H=W=4096 时：

        4096×2048
        3072×2048

    应该产生两个块。
    """

    template = generate_original_template(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=4096,
    )

    assert template.chunk_count == 2

    assert template.chunks[0].rows == 4096
    assert template.chunks[0].cols == 2048

    assert template.chunks[1].rows == 3072
    assert template.chunks[1].cols == 2048

    assert template.chunk_area_sum == 7168 * 2048


def test_template_valid_4096_4096():
    """
    上述模板应该完整合法。
    """

    template = generate_original_template(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=4096,
    )

    result = validate_partition_template(
        template=template,
        H=4096,
        W=4096,
        allow_rotation=True,
    )

    assert result.valid is True

    assert result.out_of_bounds_count == 0
    assert result.overlap_count == 0
    assert result.unplaceable_count == 0

    assert result.area_match is True
    assert result.complete_coverage is True


# ============================================================
# H=4096, W=8192
# ============================================================


def test_two_templates_4096_8192():
    """
    7168×2048，H=4096，W=8192。

    应存在两种不同匿名切分需求：

    原方向：
        4096×2048
        3072×2048

    转置方向：
        整体 7168×2048
        第三步可旋转成 2048×7168 放入 Plane。

    因此去重后应该保留两个模板。
    """

    templates = generate_partition_templates(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=8192,
    )

    assert len(templates) == 2

    chunk_counts = sorted(
        template.chunk_count
        for template in templates
    )

    assert chunk_counts == [1, 2]


def test_whole_matrix_requires_rotation_4096_8192():
    """
    整体 7168×2048：

        原方向放不进 4096×8192；
        旋转后 2048×7168 可以放入。

    第二步只检查“至少存在一种合法方向”。
    """

    template = generate_transposed_template(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=8192,
    )

    assert template.chunk_count == 1

    chunk = template.chunks[0]

    assert chunk.rows == 7168
    assert chunk.cols == 2048

    assert (
        chunk.fits_plane(
            H=4096,
            W=8192,
            allow_rotation=False,
        )
        is False
    )

    assert (
        chunk.fits_plane(
            H=4096,
            W=8192,
            allow_rotation=True,
        )
        is True
    )


# ============================================================
# H=W=8192
# ============================================================


def test_duplicate_template_removed_8192_8192():
    """
    H=W=8192 时，整个 7168×2048 可以直接放入。

    原方向与转置方向最终产生相同匿名 size_key 需求。

    因此应该被去重，只剩一个模板。
    """

    templates = generate_partition_templates(
        matrix_rows=7168,
        matrix_cols=2048,
        H=8192,
        W=8192,
    )

    assert len(templates) == 1

    template = templates[0]

    assert template.chunk_count == 1

    assert template.chunks[0].size_key == (
        2048,
        7168,
    )


# ============================================================
# 多方向模板统一验证
# ============================================================


@pytest.mark.parametrize(
    "H,W",
    [
        (4096, 4096),
        (4096, 8192),
        (7168, 4096),
        (8192, 4096),
        (8192, 8192),
        (16384, 4096),
        (16384, 8192),
        (16384, 16384),
    ],
)
def test_all_generated_templates_are_valid(
    H,
    W,
):
    """
    所有默认 H、W 组合生成的模板都必须：

    - 无越界；
    - 无重叠；
    - 面积守恒；
    - 完整覆盖；
    - 至少存在一种方向能进入 H×W。
    """

    templates = generate_partition_templates(
        matrix_rows=7168,
        matrix_cols=2048,
        H=H,
        W=W,
    )

    results = validate_partition_templates(
        templates=templates,
        H=H,
        W=W,
        allow_rotation=True,
        raise_on_error=False,
    )

    assert len(results) > 0

    for result in results:

        assert result.valid is True
        assert result.area_match is True
        assert result.complete_coverage is True
        assert result.overlap_count == 0
        assert result.out_of_bounds_count == 0
        assert result.unplaceable_count == 0


# ============================================================
# 去重验证
# ============================================================


@pytest.mark.parametrize(
    "H,W",
    [
        (4096, 4096),
        (4096, 8192),
        (7168, 4096),
        (8192, 4096),
        (8192, 8192),
        (16384, 4096),
        (16384, 8192),
        (16384, 16384),
    ],
)
def test_generated_templates_have_no_duplicates(
    H,
    W,
):
    """
    partition_generator 已经做过匿名尺寸需求去重。

    所以这里再次验证不能存在重复模板。
    """

    templates = generate_partition_templates(
        matrix_rows=7168,
        matrix_cols=2048,
        H=H,
        W=W,
    )

    # 不抛异常即通过
    validate_no_duplicate_templates(
        templates
    )


# ============================================================
# 人工构造非法模板：重叠
# ============================================================


def test_detect_overlap():
    """
    人工构造两个明显重叠的块，
    验证 validator 可以检测出来。
    """

    template = PartitionTemplate(
        template_id="bad_overlap",

        base_rows=7168,
        base_cols=2048,

        orientation_mode="original",

        chunks=(
            TemplateChunk(
                chunk_id=0,
                row_start=0,
                col_start=0,
                rows=4096,
                cols=2048,
            ),

            TemplateChunk(
                chunk_id=1,

                # 从 3000 开始，
                # 与前一个块 0~4096 明显重叠。
                row_start=3000,
                col_start=0,

                rows=4168,
                cols=2048,
            ),
        ),
    )

    overlaps = find_overlapping_pairs(
        template
    )

    assert len(overlaps) == 1

    result = validate_partition_template(
        template=template,
        H=8192,
        W=8192,
        allow_rotation=True,
    )

    assert result.valid is False

    assert result.overlap_count == 1


# ============================================================
# 人工构造非法模板：越界
# ============================================================


def test_detect_out_of_bounds():
    """
    人工构造一个越出 7168 行边界的块。
    """

    template = PartitionTemplate(
        template_id="bad_boundary",

        base_rows=7168,
        base_cols=2048,

        orientation_mode="original",

        chunks=(
            TemplateChunk(
                chunk_id=0,

                row_start=0,
                col_start=0,

                # 8000 > 7168
                rows=8000,
                cols=2048,
            ),
        ),
    )

    out_of_bounds = (
        find_out_of_bounds_chunks(
            template
        )
    )

    assert len(out_of_bounds) == 1

    result = validate_partition_template(
        template=template,
        H=8192,
        W=8192,
        allow_rotation=True,
    )

    assert result.valid is False
    assert result.out_of_bounds_count == 1


# ============================================================
# 人工构造非法模板：面积不足
# ============================================================


def test_detect_incomplete_area():
    """
    只覆盖 4096×2048，
    显然没有覆盖完整 7168×2048。
    """

    template = PartitionTemplate(
        template_id="bad_area",

        base_rows=7168,
        base_cols=2048,

        orientation_mode="original",

        chunks=(
            TemplateChunk(
                chunk_id=0,
                row_start=0,
                col_start=0,
                rows=4096,
                cols=2048,
            ),
        ),
    )

    result = validate_partition_template(
        template=template,
        H=4096,
        W=4096,
        allow_rotation=True,
    )

    assert result.valid is False

    assert result.area_match is False
    assert result.complete_coverage is False


# ============================================================
# 完整模型匿名块数量
# ============================================================


def test_full_model_block_count():
    """
    58 × 256 × 3 = 44544 个匿名矩阵。

    H=W=4096 时：

        每个矩阵 2 个块

    所以：

        44544 × 2 = 89088
    """

    template = generate_original_template(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=4096,
    )

    matrix_count = (
        58
        * 256
        * 3
    )

    assert matrix_count == 44544

    assert (
        template.total_block_count(
            matrix_count
        )
        == 89088
    )


# ============================================================
# 完整模型面积守恒
# ============================================================


def test_full_model_area_preserved():
    """
    无论怎么切：

        所有匿名块总面积

    必须始终等于：

        44544 × 7168 × 2048
    """

    matrix_count = (
        58
        * 256
        * 3
    )

    expected_area = (
        matrix_count
        * 7168
        * 2048
    )

    templates = generate_partition_templates(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=8192,
    )

    for template in templates:

        actual_area = (
            template.chunk_area_sum
            * matrix_count
        )

        assert actual_area == expected_area