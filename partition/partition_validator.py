# partition/partition_validator.py
"""
匿名矩阵切分模板合法性检查。

本文件负责检查：

1. TemplateChunk 是否越界；
2. TemplateChunk 之间是否重叠；
3. 所有块面积之和是否等于原矩阵面积；
4. 是否完整覆盖原矩阵；
5. 每个块是否至少存在一种方向能放入 H×W 平面；
6. 模板集合中是否存在重复的匿名尺寸需求。

注意：
- 本文件仍属于第二步；
- 不出现 layer_id；
- 不出现 expert_id；
- 不出现 matrix_name；
- 不出现 subcube_id、z；
- 不生成真实 Weight-Cube。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from partition.partition_template import (
    PartitionTemplate,
    TemplateChunk,
    templates_have_same_size_demand,
)


class PartitionValidationError(ValueError):
    """切分模板验证失败时抛出的异常。"""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """
    单个 PartitionTemplate 的验证结果。
    """

    template_id: str

    valid: bool

    chunk_count: int

    base_area: int
    chunk_area_sum: int

    out_of_bounds_count: int
    overlap_count: int
    unplaceable_count: int

    area_match: bool
    complete_coverage: bool

    errors: tuple[str, ...]

    def raise_if_invalid(self) -> None:
        """
        如果模板非法，直接抛出异常。
        """

        if self.valid:
            return

        message = (
            f"PartitionTemplate {self.template_id!r} "
            "验证失败：\n"
            + "\n".join(
                f"- {error}"
                for error in self.errors
            )
        )

        raise PartitionValidationError(message)


def rectangles_overlap(
    first: TemplateChunk,
    second: TemplateChunk,
) -> bool:
    """
    判断两个模板矩形是否真正重叠。

    使用左闭右开区间：

        row: [row_start, row_end)
        col: [col_start, col_end)

    两个矩形仅仅边界接触不算重叠。
    """

    row_overlap = (
        first.row_start < second.row_end
        and second.row_start < first.row_end
    )

    col_overlap = (
        first.col_start < second.col_end
        and second.col_start < first.col_end
    )

    return row_overlap and col_overlap


def chunk_is_inside_base_matrix(
    chunk: TemplateChunk,
    base_rows: int,
    base_cols: int,
) -> bool:
    """
    判断一个 TemplateChunk 是否完全位于标准矩阵内部。
    """

    return (
        0 <= chunk.row_start
        and 0 <= chunk.col_start
        and chunk.row_end <= base_rows
        and chunk.col_end <= base_cols
    )


def find_out_of_bounds_chunks(
    template: PartitionTemplate,
) -> list[TemplateChunk]:
    """
    找出所有越界块。
    """

    return [
        chunk
        for chunk in template.chunks
        if not chunk_is_inside_base_matrix(
            chunk=chunk,
            base_rows=template.base_rows,
            base_cols=template.base_cols,
        )
    ]


def find_overlapping_pairs(
    template: PartitionTemplate,
) -> list[tuple[TemplateChunk, TemplateChunk]]:
    """
    找出模板中所有发生重叠的块对。

    当前最大规则切分得到的 chunk 数量通常很少，
    因此直接使用 O(n^2) 检查即可。

    这样实现简单而且可靠。
    """

    chunks = template.chunks

    overlaps: list[
        tuple[TemplateChunk, TemplateChunk]
    ] = []

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):

            first = chunks[i]
            second = chunks[j]

            if rectangles_overlap(
                first,
                second,
            ):
                overlaps.append(
                    (first, second)
                )

    return overlaps


def find_unplaceable_chunks(
    template: PartitionTemplate,
    H: int,
    W: int,
    allow_rotation: bool = True,
) -> list[TemplateChunk]:
    """
    找出无法以任何合法方向放入 H×W 平面的模板块。

    注意：

    第二步这里只检查：

        “是否至少有一个方向能放进去”

    不决定第三步实际使用哪个方向。
    """

    if H <= 0 or W <= 0:
        raise PartitionValidationError(
            f"H、W 必须大于 0，当前为 H={H}, W={W}。"
        )

    return [
        chunk
        for chunk in template.chunks
        if not chunk.fits_plane(
            H=H,
            W=W,
            allow_rotation=allow_rotation,
        )
    ]


def validate_partition_template(
    template: PartitionTemplate,
    H: int,
    W: int,
    allow_rotation: bool = True,
) -> ValidationResult:
    """
    对一个 PartitionTemplate 做完整合法性检查。

    验证规则：

    1. 所有 chunk 不越出原矩阵；
    2. 任意两个 chunk 不重叠；
    3. chunk 总面积等于原矩阵面积；
    4. 每个 chunk 至少存在一种方向可以放入 H×W；
    5. 上述条件同时满足时，可认为完整覆盖原矩阵。

    为什么“不重叠 + 不越界 + 面积相等”
    可以证明完整覆盖？

    因为：

        所有 chunk 都位于原矩阵内部；
        chunk 之间没有重复区域；
        它们的面积总和恰好等于原矩阵面积。

    因此不存在未覆盖区域。
    """

    errors: list[str] = []

    # ========================================================
    # 1. 越界检查
    # ========================================================

    out_of_bounds = find_out_of_bounds_chunks(
        template
    )

    for chunk in out_of_bounds:
        errors.append(
            "Chunk-"
            f"{chunk.chunk_id} 越界："
            f"start=({chunk.row_start}, "
            f"{chunk.col_start}), "
            f"size={chunk.rows}×{chunk.cols}, "
            f"base={template.base_rows}×"
            f"{template.base_cols}"
        )

    # ========================================================
    # 2. 重叠检查
    # ========================================================

    overlapping_pairs = find_overlapping_pairs(
        template
    )

    for first, second in overlapping_pairs:
        errors.append(
            f"Chunk-{first.chunk_id} 与 "
            f"Chunk-{second.chunk_id} 发生重叠。"
        )

    # ========================================================
    # 3. 面积守恒检查
    # ========================================================

    base_area = template.base_area
    chunk_area_sum = template.chunk_area_sum

    area_match = (
        chunk_area_sum == base_area
    )

    if not area_match:
        errors.append(
            "模板块总面积与标准矩阵面积不一致："
            f"chunk_area_sum={chunk_area_sum}, "
            f"base_area={base_area}。"
        )

    # ========================================================
    # 4. 判断是否完整覆盖
    # ========================================================

    complete_coverage = (
        len(out_of_bounds) == 0
        and len(overlapping_pairs) == 0
        and area_match
    )

    if not complete_coverage:
        errors.append(
            "模板未能证明完整覆盖标准矩阵。"
        )

    # ========================================================
    # 5. 平面适配检查
    # ========================================================

    unplaceable_chunks = find_unplaceable_chunks(
        template=template,
        H=H,
        W=W,
        allow_rotation=allow_rotation,
    )

    for chunk in unplaceable_chunks:
        errors.append(
            f"Chunk-{chunk.chunk_id} "
            f"尺寸 {chunk.rows}×{chunk.cols} "
            f"无法放入 H×W={H}×{W} 平面，"
            "即使考虑旋转也不合法。"
        )

    # ========================================================
    # 6. 最终结果
    # ========================================================

    valid = (
        complete_coverage
        and len(unplaceable_chunks) == 0
    )

    return ValidationResult(
        template_id=template.template_id,

        valid=valid,

        chunk_count=template.chunk_count,

        base_area=base_area,
        chunk_area_sum=chunk_area_sum,

        out_of_bounds_count=len(out_of_bounds),
        overlap_count=len(overlapping_pairs),
        unplaceable_count=len(unplaceable_chunks),

        area_match=area_match,
        complete_coverage=complete_coverage,

        errors=tuple(errors),
    )


def validate_partition_templates(
    templates: Iterable[PartitionTemplate],
    H: int,
    W: int,
    allow_rotation: bool = True,
    raise_on_error: bool = True,
) -> tuple[ValidationResult, ...]:
    """
    批量验证多个候选模板。

    Args:
        templates:
            第二步生成的候选模板。

        H, W:
            当前候选平面尺寸。

        allow_rotation:
            第三步是否允许矩形旋转。

        raise_on_error:
            如果为 True，只要出现非法模板立即抛异常。

    Returns:
        每个模板对应的 ValidationResult。
    """

    template_list = tuple(templates)

    if not template_list:
        raise PartitionValidationError(
            "待验证模板集合不能为空。"
        )

    results: list[ValidationResult] = []

    for template in template_list:

        result = validate_partition_template(
            template=template,
            H=H,
            W=W,
            allow_rotation=allow_rotation,
        )

        results.append(result)

        if raise_on_error:
            result.raise_if_invalid()

    return tuple(results)


def find_duplicate_template_pairs(
    templates: Iterable[PartitionTemplate],
) -> list[
    tuple[PartitionTemplate, PartitionTemplate]
]:
    """
    找出纯空间意义下重复的模板。

    “重复”的定义：

        两个模板产生完全相同的匿名 size_key 数量。

    不要求：
        chunk_id 相同；
        orientation_mode 相同；
        模板坐标相同。

    例如：

        4096×2048 + 3072×2048

    与：

        2048×4096 + 2048×3072

    在前三步纯空间规划中属于相同需求。
    """

    template_list = list(templates)

    duplicates: list[
        tuple[PartitionTemplate, PartitionTemplate]
    ] = []

    for i in range(len(template_list)):
        for j in range(i + 1, len(template_list)):

            first = template_list[i]
            second = template_list[j]

            if templates_have_same_size_demand(
                first,
                second,
            ):
                duplicates.append(
                    (first, second)
                )

    return duplicates


def validate_no_duplicate_templates(
    templates: Iterable[PartitionTemplate],
) -> None:
    """
    确认候选模板集合中不存在空间等价的重复模板。

    partition_generator.py 已经负责去重，
    因此这里主要作为防御性检查。
    """

    duplicate_pairs = find_duplicate_template_pairs(
        templates
    )

    if not duplicate_pairs:
        return

    descriptions = []

    for first, second in duplicate_pairs:
        descriptions.append(
            f"{first.template_id!r} "
            f"与 {second.template_id!r}"
        )

    raise PartitionValidationError(
        "发现匿名尺寸需求完全重复的模板：\n- "
        + "\n- ".join(descriptions)
    )


def print_validation_result(
    result: ValidationResult,
) -> None:
    """
    打印一个模板的验证结果。
    """

    print(
        f"========== {result.template_id} =========="
    )

    print(f"合法：{result.valid}")
    print(f"块数量：{result.chunk_count}")

    print(
        "面积："
        f"{result.chunk_area_sum} / "
        f"{result.base_area}"
    )

    print(
        f"面积守恒：{result.area_match}"
    )

    print(
        f"完整覆盖：{result.complete_coverage}"
    )

    print(
        f"越界块数量："
        f"{result.out_of_bounds_count}"
    )

    print(
        f"重叠块对数量："
        f"{result.overlap_count}"
    )

    print(
        f"无法装入平面的块数量："
        f"{result.unplaceable_count}"
    )

    if result.errors:
        print("错误：")

        for error in result.errors:
            print(f"  - {error}")


if __name__ == "__main__":

    # ========================================================
    # 与 partition_generator 联合测试
    # ========================================================

    from partition.partition_generator import (
        generate_partition_templates,
    )

    matrix_rows = 7168
    matrix_cols = 2048

    # --------------------------------------------------------
    # Example 1
    # --------------------------------------------------------

    H = 4096
    W = 4096

    templates = generate_partition_templates(
        matrix_rows=matrix_rows,
        matrix_cols=matrix_cols,
        H=H,
        W=W,
    )

    print(
        "\n"
        "=========================================\n"
        f"Validate H={H}, W={W}\n"
        "========================================="
    )

    results = validate_partition_templates(
        templates=templates,
        H=H,
        W=W,
        allow_rotation=True,
        raise_on_error=False,
    )

    for result in results:
        print_validation_result(result)
        print()

    validate_no_duplicate_templates(
        templates
    )

    # --------------------------------------------------------
    # Example 2
    # --------------------------------------------------------

    H = 4096
    W = 8192

    templates = generate_partition_templates(
        matrix_rows=matrix_rows,
        matrix_cols=matrix_cols,
        H=H,
        W=W,
    )

    print(
        "\n"
        "=========================================\n"
        f"Validate H={H}, W={W}\n"
        "========================================="
    )

    results = validate_partition_templates(
        templates=templates,
        H=H,
        W=W,
        allow_rotation=True,
        raise_on_error=False,
    )

    for result in results:
        print_validation_result(result)
        print()

    validate_no_duplicate_templates(
        templates
    )