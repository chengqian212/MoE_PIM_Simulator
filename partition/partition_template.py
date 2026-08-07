# partition/partition_template.py
"""
匿名矩阵切分模板的数据结构定义。

本文件只描述：

1. 一个标准匿名矩阵被切成哪些几何块；
2. 每个几何块在标准矩阵中的位置；
3. 一个完整切分模板由哪些匿名块组成。

注意：
- 这里没有 layer_id；
- 没有 expert_id；
- 没有 matrix_name；
- 没有 subcube_id；
- 没有 z；
- 没有真实 Weight-Cube。

TemplateChunk 只是“标准矩阵切分模板中的一个匿名矩形块”。

真实 gate/up/down 矩阵要到第四步才根据这些模板生成
LogicalWeightCube。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from model_geometry import SizeKey, make_size_key


class PartitionTemplateError(ValueError):
    """切分模板数据不合法时抛出的异常。"""


@dataclass(frozen=True, slots=True)
class TemplateChunk:
    """
    一个匿名切分模板中的矩形块。

    例如标准矩阵：

        7168 × 2048

    在 H=W=4096 时可能切为：

        Chunk-0:
            row_start = 0
            col_start = 0
            rows = 4096
            cols = 2048

        Chunk-1:
            row_start = 4096
            col_start = 0
            rows = 3072
            cols = 2048

    注意：
    row_start / col_start 只是标准匿名矩阵内部的模板坐标，
    并不属于某个真实 gate/up/down 矩阵。
    """

    chunk_id: int

    row_start: int
    col_start: int

    rows: int
    cols: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """检查单个模板块的基本合法性。"""

        if self.chunk_id < 0:
            raise PartitionTemplateError(
                f"chunk_id 不能为负数，当前为 {self.chunk_id}。"
            )

        if self.row_start < 0:
            raise PartitionTemplateError(
                f"row_start 不能为负数，当前为 {self.row_start}。"
            )

        if self.col_start < 0:
            raise PartitionTemplateError(
                f"col_start 不能为负数，当前为 {self.col_start}。"
            )

        if self.rows <= 0:
            raise PartitionTemplateError(
                f"rows 必须大于 0，当前为 {self.rows}。"
            )

        if self.cols <= 0:
            raise PartitionTemplateError(
                f"cols 必须大于 0，当前为 {self.cols}。"
            )

    @property
    def row_end(self) -> int:
        """
        矩形在标准矩阵中的行结束位置。

        使用左闭右开区间：
            [row_start, row_end)
        """
        return self.row_start + self.rows

    @property
    def col_end(self) -> int:
        """
        矩形在标准矩阵中的列结束位置。

        使用左闭右开区间：
            [col_start, col_end)
        """
        return self.col_start + self.cols

    @property
    def area(self) -> int:
        """该匿名块的面积。"""
        return self.rows * self.cols

    @property
    def size_key(self) -> SizeKey:
        """
        返回与方向无关的尺寸类型。

        例如：
            4096×2048
            2048×4096

        都返回：
            (2048, 4096)
        """
        return make_size_key(
            self.rows,
            self.cols,
        )

    @property
    def longest_side(self) -> int:
        """最长边。"""
        return max(self.rows, self.cols)

    @property
    def shortest_side(self) -> int:
        """最短边。"""
        return min(self.rows, self.cols)

    def fits_plane(
        self,
        H: int,
        W: int,
        allow_rotation: bool = True,
    ) -> bool:
        """
        判断该匿名块是否至少存在一个方向可以放入 H×W 平面。

        这里只判断尺寸，不决定真正放置方向。
        """

        if H <= 0 or W <= 0:
            raise PartitionTemplateError(
                f"平面尺寸必须为正数，当前为 {H}×{W}。"
            )

        normal_fit = (
            self.rows <= H
            and self.cols <= W
        )

        rotated_fit = (
            allow_rotation
            and self.cols <= H
            and self.rows <= W
        )

        return normal_fit or rotated_fit

    def geometry_tuple(self) -> tuple[int, int, int, int]:
        """
        返回包含模板位置的完整几何描述。

        用于验证、调试和模板比较。
        """
        return (
            self.row_start,
            self.col_start,
            self.rows,
            self.cols,
        )


@dataclass(frozen=True, slots=True)
class PartitionTemplate:
    """
    一个标准匿名矩阵的完整几何切分模板。

    例如：

        base_rows = 7168
        base_cols = 2048

        chunks:
            4096×2048
            3072×2048

    orientation_mode 表示该模板是怎样生成的。

    推荐取值：

        "original"
            按标准矩阵原方向生成。

        "transposed"
            先从转置后的几何方向生成切分方案，
            然后转换回标准矩阵坐标。

    注意：
    orientation_mode 只是模板生成方式，
    不代表第三步中每个匿名块最终是否旋转放置。
    """

    template_id: str

    base_rows: int
    base_cols: int

    orientation_mode: str

    chunks: tuple[TemplateChunk, ...]

    def __post_init__(self) -> None:
        self.validate_basic()

    def validate_basic(self) -> None:
        """
        只进行数据结构层面的基础检查。

        完整的：
        - 覆盖检查；
        - 重叠检查；
        - 越界检查；

        放到 partition_validator.py 中完成。
        """

        if not self.template_id:
            raise PartitionTemplateError(
                "template_id 不能为空。"
            )

        if self.base_rows <= 0:
            raise PartitionTemplateError(
                f"base_rows 必须大于 0，当前为 {self.base_rows}。"
            )

        if self.base_cols <= 0:
            raise PartitionTemplateError(
                f"base_cols 必须大于 0，当前为 {self.base_cols}。"
            )

        allowed_modes = {
            "original",
            "transposed",
        }

        if self.orientation_mode not in allowed_modes:
            raise PartitionTemplateError(
                "orientation_mode 必须为 "
                f"{sorted(allowed_modes)} 之一，"
                f"当前为 {self.orientation_mode!r}。"
            )

        if not self.chunks:
            raise PartitionTemplateError(
                "PartitionTemplate 至少需要一个 TemplateChunk。"
            )

        chunk_ids = [
            chunk.chunk_id
            for chunk in self.chunks
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise PartitionTemplateError(
                f"模板 {self.template_id} 中存在重复 chunk_id。"
            )

    @property
    def base_area(self) -> int:
        """标准匿名矩阵总面积。"""
        return self.base_rows * self.base_cols

    @property
    def chunk_count(self) -> int:
        """单个标准矩阵会产生多少个匿名块。"""
        return len(self.chunks)

    @property
    def chunk_area_sum(self) -> int:
        """所有模板块面积总和。"""
        return sum(
            chunk.area
            for chunk in self.chunks
        )

    @property
    def size_histogram(self) -> dict[SizeKey, int]:
        """
        统计单个矩阵对应的各类匿名块数量。

        例如：

            4096×2048
            3072×2048

        返回：

            {
                (2048, 4096): 1,
                (2048, 3072): 1
            }

        size_key 不区分旋转方向。
        """

        counter = Counter(
            chunk.size_key
            for chunk in self.chunks
        )

        return dict(counter)

    @property
    def oriented_size_histogram(
        self,
    ) -> dict[tuple[int, int], int]:
        """
        按模板中的原始 rows×cols 统计块数量。

        与 size_histogram 不同：

        - size_histogram 不区分旋转；
        - oriented_size_histogram 区分 rows、cols 顺序。

        该信息主要用于调试切分模板。
        """

        counter = Counter(
            (chunk.rows, chunk.cols)
            for chunk in self.chunks
        )

        return dict(counter)

    @property
    def geometry_signature(
        self,
    ) -> tuple[SizeKey, ...]:
        """
        返回忽略块位置、忽略旋转方向后的尺寸签名。

        用于第二步去除几何等价模板。

        例如：

            Template-A:
                4096×2048
                3072×2048

            Template-B:
                2048×4096
                2048×3072

        二者 signature 相同：

            (
                (2048, 3072),
                (2048, 4096),
            )

        对于前三步纯空间规划来说，它们可以视为
        相同的匿名尺寸需求。
        """

        return tuple(
            sorted(
                chunk.size_key
                for chunk in self.chunks
            )
        )

    def total_block_count(
        self,
        matrix_count: int,
    ) -> int:
        """
        计算完整模型对应的匿名块总数。

        例如：
            单矩阵 2 块；
            匿名矩阵数量 44544；

        则：
            89088 个匿名块。
        """

        if matrix_count <= 0:
            raise PartitionTemplateError(
                "matrix_count 必须大于 0。"
            )

        return self.chunk_count * matrix_count

    def total_size_histogram(
        self,
        matrix_count: int,
    ) -> dict[SizeKey, int]:
        """
        将单矩阵模板扩展到完整模型。

        例如单矩阵：

            (2048,4096): 1
            (2048,3072): 1

        matrix_count = 44544

        返回：

            (2048,4096): 44544
            (2048,3072): 44544
        """

        if matrix_count <= 0:
            raise PartitionTemplateError(
                "matrix_count 必须大于 0。"
            )

        return {
            size_key: count * matrix_count
            for size_key, count
            in self.size_histogram.items()
        }

    def get_chunk(
        self,
        chunk_id: int,
    ) -> TemplateChunk:
        """
        根据 chunk_id 获取模板块。
        """

        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk

        raise PartitionTemplateError(
            f"模板 {self.template_id} 中不存在 "
            f"chunk_id={chunk_id}。"
        )

    def summary(self) -> str:
        """生成用于调试的简短文本摘要。"""

        lines = [
            f"PartitionTemplate<{self.template_id}>",
            (
                f"  base="
                f"{self.base_rows}×{self.base_cols}"
            ),
            (
                f"  orientation_mode="
                f"{self.orientation_mode}"
            ),
            f"  chunk_count={self.chunk_count}",
        ]

        for chunk in sorted(
            self.chunks,
            key=lambda c: c.chunk_id,
        ):
            lines.append(
                "  "
                f"Chunk-{chunk.chunk_id}: "
                f"start=({chunk.row_start},"
                f"{chunk.col_start}), "
                f"size={chunk.rows}×{chunk.cols}, "
                f"size_key={chunk.size_key}"
            )

        return "\n".join(lines)


def build_template(
    template_id: str,
    base_rows: int,
    base_cols: int,
    orientation_mode: str,
    chunks: Iterable[TemplateChunk],
) -> PartitionTemplate:
    """
    辅助函数：从任意 iterable 创建不可变 PartitionTemplate。
    """

    return PartitionTemplate(
        template_id=template_id,
        base_rows=base_rows,
        base_cols=base_cols,
        orientation_mode=orientation_mode,
        chunks=tuple(chunks),
    )


def templates_have_same_size_demand(
    first: PartitionTemplate,
    second: PartitionTemplate,
) -> bool:
    """
    判断两个模板是否产生完全相同的匿名尺寸需求。

    这里只比较：

        每种 size_key 的数量

    不比较：

        chunk_id
        模板坐标
        orientation_mode

    这个函数将在 partition_generator.py 中用于
    删除对纯空间装箱而言完全重复的候选模板。
    """

    return (
        first.size_histogram
        == second.size_histogram
    )


if __name__ == "__main__":
    # 一个简单的手动示例：
    #
    # 7168×2048
    # 在 H=W=4096 时：
    #
    #     4096×2048
    #     3072×2048

    example_template = build_template(
        template_id="example_original",
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
                row_start=4096,
                col_start=0,
                rows=3072,
                cols=2048,
            ),
        ),
    )

    print(example_template.summary())

    print("\n========== Statistics ==========")
    print(
        f"标准矩阵面积："
        f"{example_template.base_area}"
    )
    print(
        f"模板块面积总和："
        f"{example_template.chunk_area_sum}"
    )
    print(
        f"单矩阵块数量："
        f"{example_template.chunk_count}"
    )
    print(
        f"单矩阵尺寸统计："
        f"{example_template.size_histogram}"
    )

    matrix_count = 58 * 256 * 3

    print(
        f"完整模型匿名块总数："
        f"{example_template.total_block_count(matrix_count)}"
    )
    print(
        "完整模型各尺寸匿名块数量："
        f"{example_template.total_size_histogram(matrix_count)}"
    )