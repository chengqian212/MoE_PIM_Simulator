# partition/partition_generator.py
"""
根据候选平面尺寸 H×W 生成匿名矩阵切分模板。

本文件负责：

1. 按标准矩阵原方向生成“最大块规则”切分模板；
2. 按标准矩阵转置方向生成另一套切分模板；
3. 将转置方向的坐标重新映射回标准矩阵坐标；
4. 删除对纯空间阶段完全等价的重复模板；
5. 返回全部合法候选模板。

注意：
- 本文件不生成真实 Weight-Cube；
- 不出现 layer_id；
- 不出现 expert_id；
- 不出现 matrix_name；
- 不选择 subcube_id、z；
- 不决定最终实际物理槽位；
- 不按“块数少”提前选唯一模板。

所有模板都交给第三步进行真实二维装箱比较。
"""

from __future__ import annotations

from typing import Iterable

from partition.partition_template import (
    PartitionTemplate,
    TemplateChunk,
    build_template,
    templates_have_same_size_demand,
)


class PartitionGenerationError(ValueError):
    """切分模板生成失败时抛出的异常。"""


def _validate_dimensions(
    matrix_rows: int,
    matrix_cols: int,
    H: int,
    W: int,
) -> None:
    """检查输入尺寸。"""

    if matrix_rows <= 0:
        raise PartitionGenerationError(
            f"matrix_rows 必须大于 0，当前为 {matrix_rows}。"
        )

    if matrix_cols <= 0:
        raise PartitionGenerationError(
            f"matrix_cols 必须大于 0，当前为 {matrix_cols}。"
        )

    if H <= 0:
        raise PartitionGenerationError(
            f"H 必须大于 0，当前为 {H}。"
        )

    if W <= 0:
        raise PartitionGenerationError(
            f"W 必须大于 0，当前为 {W}。"
        )


def _generate_axis_segments(
    total_length: int,
    max_length: int,
) -> list[tuple[int, int]]:
    """
    将一维区间按照“最大块规则”切分。

    Args:
        total_length:
            原始区间长度。

        max_length:
            每一块允许的最大长度。

    Returns:
        [
            (start, length),
            ...
        ]

    例如：

        total_length = 7168
        max_length = 4096

    返回：

        [
            (0, 4096),
            (4096, 3072),
        ]

    最后一块直接使用剩余长度，不进行补齐。
    """

    if total_length <= 0:
        raise PartitionGenerationError(
            "total_length 必须大于 0。"
        )

    if max_length <= 0:
        raise PartitionGenerationError(
            "max_length 必须大于 0。"
        )

    segments: list[tuple[int, int]] = []

    start = 0

    while start < total_length:
        remaining = total_length - start

        length = min(
            remaining,
            max_length,
        )

        segments.append(
            (start, length)
        )

        start += length

    return segments


def _generate_grid_chunks(
    rows: int,
    cols: int,
    max_rows: int,
    max_cols: int,
) -> list[TemplateChunk]:
    """
    对一个 rows×cols 矩阵按照二维最大块规则生成网格切分。

    行方向：
        每块最多 max_rows

    列方向：
        每块最多 max_cols

    最后剩多少就取多少，不补齐。

    例如：

        matrix = 7168 × 5000
        max_rows = 4096
        max_cols = 4096

    行切分：
        4096
        3072

    列切分：
        4096
        904

    最终产生四块：

        4096×4096
        4096×904
        3072×4096
        3072×904
    """

    row_segments = _generate_axis_segments(
        total_length=rows,
        max_length=max_rows,
    )

    col_segments = _generate_axis_segments(
        total_length=cols,
        max_length=max_cols,
    )

    chunks: list[TemplateChunk] = []

    chunk_id = 0

    # 固定遍历顺序：
    # 先沿行，再沿列。
    #
    # 这样可以保证结果完全可复现。
    for row_start, chunk_rows in row_segments:
        for col_start, chunk_cols in col_segments:

            chunks.append(
                TemplateChunk(
                    chunk_id=chunk_id,
                    row_start=row_start,
                    col_start=col_start,
                    rows=chunk_rows,
                    cols=chunk_cols,
                )
            )

            chunk_id += 1

    return chunks


def generate_original_template(
    matrix_rows: int,
    matrix_cols: int,
    H: int,
    W: int,
    template_id: str | None = None,
) -> PartitionTemplate:
    """
    生成原方向切分模板。

    标准矩阵保持：

        matrix_rows × matrix_cols

    然后按照：

        行最大长度 H
        列最大长度 W

    进行最大规则切分。

    例如：

        matrix = 7168×2048
        H = 4096
        W = 4096

    得到：

        4096×2048
        3072×2048
    """

    _validate_dimensions(
        matrix_rows=matrix_rows,
        matrix_cols=matrix_cols,
        H=H,
        W=W,
    )

    chunks = _generate_grid_chunks(
        rows=matrix_rows,
        cols=matrix_cols,
        max_rows=H,
        max_cols=W,
    )

    if template_id is None:
        template_id = (
            f"original_"
            f"{matrix_rows}x{matrix_cols}_"
            f"H{H}_W{W}"
        )

    return build_template(
        template_id=template_id,
        base_rows=matrix_rows,
        base_cols=matrix_cols,
        orientation_mode="original",
        chunks=chunks,
    )


def generate_transposed_template(
    matrix_rows: int,
    matrix_cols: int,
    H: int,
    W: int,
    template_id: str | None = None,
) -> PartitionTemplate:
    """
    生成转置方向切分模板。

    注意：
    这里并不是改变真实矩阵的数学含义。

    只是纯空间阶段考虑另一种几何方向：

        原矩阵：
            R × C

        临时转置几何：
            C × R

    先在 C×R 上按照 H×W 最大规则切分，
    然后把每个块的坐标映射回原始 R×C 坐标。

    ------------------------------------------------

    假设转置矩阵中的一个块为：

        transposed_row_start = tr
        transposed_col_start = tc

        transposed_rows = r
        transposed_cols = c

    那么映射回原矩阵后：

        row_start = tc
        col_start = tr

        rows = c
        cols = r

    ------------------------------------------------

    例如：

        原矩阵：
            7168×2048

        H=4096
        W=8192

    临时转置：

            2048×7168

    可以整块放入：

            2048×7168

    映射回标准矩阵逻辑坐标后仍描述：

            rows=7168
            cols=2048

    但是 orientation_mode="transposed"

    到第三步时，它可以作为旋转后的整体矩形装入平面。
    """

    _validate_dimensions(
        matrix_rows=matrix_rows,
        matrix_cols=matrix_cols,
        H=H,
        W=W,
    )

    # ========================================================
    # 1. 在“转置几何空间”中切分
    # ========================================================

    transposed_chunks = _generate_grid_chunks(
        rows=matrix_cols,
        cols=matrix_rows,
        max_rows=H,
        max_cols=W,
    )

    # ========================================================
    # 2. 映射回标准矩阵坐标
    # ========================================================

    original_coordinate_chunks: list[TemplateChunk] = []

    for new_chunk_id, chunk in enumerate(transposed_chunks):

        # 转置坐标：
        #
        # T[row, col] = A[col, row]
        #
        # 因此映射回原矩阵：
        #
        # original row_start = transposed col_start
        # original col_start = transposed row_start
        #
        # original rows = transposed cols
        # original cols = transposed rows

        original_coordinate_chunks.append(
            TemplateChunk(
                chunk_id=new_chunk_id,

                row_start=chunk.col_start,
                col_start=chunk.row_start,

                rows=chunk.cols,
                cols=chunk.rows,
            )
        )

    if template_id is None:
        template_id = (
            f"transposed_"
            f"{matrix_rows}x{matrix_cols}_"
            f"H{H}_W{W}"
        )

    return build_template(
        template_id=template_id,
        base_rows=matrix_rows,
        base_cols=matrix_cols,
        orientation_mode="transposed",
        chunks=original_coordinate_chunks,
    )


def remove_duplicate_templates(
    templates: Iterable[PartitionTemplate],
) -> list[PartitionTemplate]:
    """
    删除对纯空间规划而言重复的模板。

    判断标准：

        每一种 size_key 的匿名块数量完全一样。

    例如：

        Template-A：
            4096×2048
            3072×2048

        Template-B：
            2048×4096
            2048×3072

    对第三步的匿名二维装箱来说，两者产生的尺寸需求一样：

        (2048,4096): 1
        (2048,3072): 1

    因此只保留一个。

    ------------------------------------------------

    注意：

    这里不是按照“块数量少”删除模板。

    两个模板只要匿名尺寸需求不同，
    即使一个模板块数更多，也必须保留给第三步比较。
    """

    unique_templates: list[PartitionTemplate] = []

    for template in templates:

        duplicate_found = False

        for existing in unique_templates:
            if templates_have_same_size_demand(
                template,
                existing,
            ):
                duplicate_found = True
                break

        if not duplicate_found:
            unique_templates.append(template)

    return unique_templates


def generate_partition_templates(
    matrix_rows: int,
    matrix_cols: int,
    H: int,
    W: int,
) -> list[PartitionTemplate]:
    """
    第二步的主要入口函数。

    对指定矩阵和 H×W：

    1. 生成原方向切分模板；
    2. 生成转置方向切分模板；
    3. 删除纯空间意义下重复的模板；
    4. 返回所有候选。

    不在这里选择“最优模板”。

    最优与否需要第三步真实装箱后，
    根据 P、内部碎片、最终容量等指标判断。
    """

    _validate_dimensions(
        matrix_rows=matrix_rows,
        matrix_cols=matrix_cols,
        H=H,
        W=W,
    )

    original = generate_original_template(
        matrix_rows=matrix_rows,
        matrix_cols=matrix_cols,
        H=H,
        W=W,
    )

    transposed = generate_transposed_template(
        matrix_rows=matrix_rows,
        matrix_cols=matrix_cols,
        H=H,
        W=W,
    )

    templates = remove_duplicate_templates(
        [
            original,
            transposed,
        ]
    )

    return templates


def print_partition_templates(
    templates: Iterable[PartitionTemplate],
) -> None:
    """打印候选模板信息。"""

    template_list = list(templates)

    print(
        "========== Partition Templates =========="
    )

    print(
        f"候选模板数量：{len(template_list)}"
    )

    for template in template_list:

        print()

        print(template.summary())

        print(
            "  size_histogram="
            f"{template.size_histogram}"
        )

        print(
            "  geometry_signature="
            f"{template.geometry_signature}"
        )


if __name__ == "__main__":

    # ========================================================
    # 示例 1：
    # 7168×2048
    # H=W=4096
    # ========================================================

    print(
        "\n"
        "=========================================\n"
        "Example 1: H=4096, W=4096\n"
        "========================================="
    )

    templates_1 = generate_partition_templates(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=4096,
    )

    print_partition_templates(
        templates_1
    )

    # ========================================================
    # 示例 2：
    #
    # 原矩阵：
    #     7168×2048
    #
    # H=4096
    # W=8192
    #
    # 原方向：
    #     4096×2048
    #     3072×2048
    #
    # 转置方向：
    #     可以整体按 2048×7168 放置
    #
    # 因此应该保留两个不同模板。
    # ========================================================

    print(
        "\n"
        "=========================================\n"
        "Example 2: H=4096, W=8192\n"
        "========================================="
    )

    templates_2 = generate_partition_templates(
        matrix_rows=7168,
        matrix_cols=2048,
        H=4096,
        W=8192,
    )

    print_partition_templates(
        templates_2
    )

    # ========================================================
    # 示例 3：
    # H=8192, W=8192
    #
    # 整个矩阵无需切分。
    #
    # 原方向和转置方向从匿名尺寸需求上完全等价，
    # 因此去重后只保留一个模板。
    # ========================================================

    print(
        "\n"
        "=========================================\n"
        "Example 3: H=8192, W=8192\n"
        "========================================="
    )

    templates_3 = generate_partition_templates(
        matrix_rows=7168,
        matrix_cols=2048,
        H=8192,
        W=8192,
    )

    print_partition_templates(
        templates_3
    )