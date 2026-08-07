# run_spatial_baseline.py
"""
前三步统一运行入口。

完整流程：

第一步：
    ModelConfig
        ↓
    得到模型匿名几何规模

第二步：
    对每个 H、W
        ↓
    生成 PartitionTemplate
        ↓
    验证模板

第三步：
    对每个 PartitionTemplate
        ↓
    完整匿名二维装箱
        ↓
    得到 P 个 Plane 和全部 PhysicalSlot
        ↓
    枚举 N = 2, 3, 4
        ↓
    D = ceil(P / N^2)
        ↓
    Q = N^2 * D
        ↓
    C = Q * H * W
        ↓
    计算空间指标
        ↓
    保存空间候选

注意：

前三步始终不出现：

    layer_id
    expert_id
    matrix_name
    gate/up/down 逻辑身份
    subcube_id
    z

第三步输出的仍然只是：
    匿名 Plane
    匿名 PhysicalSlot
    N、H、W、D
    空间指标

第四步才完成真实逻辑映射。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import (
    ModelConfig,
    ExecutionRules,
)

from model_geometry import (
    build_model_geometry,
    print_geometry_summary,
)

from partition.partition_generator import (
    generate_partition_templates,
)

from partition.partition_validator import (
    validate_partition_templates,
    validate_no_duplicate_templates,
)

from packing.anonymous_packer import (
    PackingResult,
    pack_anonymous_blocks,
    print_packing_result,
)

from evaluation.hardware_resolver import (
    HardwareResolutionResult,
    resolve_all_n,
)

from evaluation.spatial_metrics import (
    SpatialMetrics,
    compare_spatial_metrics,
    evaluate_spatial_metrics,
    print_spatial_metrics,
    validate_slot_histogram,
)


# ============================================================
# 默认实验参数
# ============================================================


DEFAULT_PLANE_SHAPES: tuple[
    tuple[int, int],
    ...
] = (
    (4096, 4096),
    (4096, 8192),
    (7168, 4096),
    (8192, 4096),
    (8192, 8192),
    (16384, 4096),
    (16384, 8192),
    (16384, 16384),
)


DEFAULT_N_VALUES: tuple[int, ...] = (
    2,
    3,
    4,
)


# ============================================================
# JSON 辅助
# ============================================================


def size_histogram_to_json(
    histogram: dict[tuple[int, int], int],
) -> dict[str, int]:
    """
    将：

        {(2048, 4096): 44544}

    转成 JSON 可保存形式：

        {"2048x4096": 44544}
    """

    return {
        f"{short_side}x{long_side}": count
        for (
            short_side,
            long_side
        ), count in sorted(histogram.items())
    }


def template_to_dict(
    template,
) -> dict[str, Any]:
    """
    将 PartitionTemplate 转成 JSON 数据。
    """

    return {
        "template_id": template.template_id,
        "base_rows": template.base_rows,
        "base_cols": template.base_cols,
        "orientation_mode": (
            template.orientation_mode
        ),
        "chunk_count": (
            template.chunk_count
        ),
        "size_histogram": (
            size_histogram_to_json(
                template.size_histogram
            )
        ),
        "chunks": [
            {
                "template_chunk_id": (
                    chunk.chunk_id
                ),
                "row_start": (
                    chunk.row_start
                ),
                "col_start": (
                    chunk.col_start
                ),
                "rows": chunk.rows,
                "cols": chunk.cols,
                "size_key": list(
                    chunk.size_key
                ),
            }
            for chunk in template.chunks
        ],
    }


def packing_layout_to_dict(
    packing: PackingResult,
    template,
) -> dict[str, Any]:
    """
    保存第三步真正需要交给第四步的匿名空间布局。

    注意：

    这里保存：
        plane_id
        slot_id
        x
        y
        slot_rows
        slot_cols

    不保存：
        subcube_id
        z
        layer_id
        expert_id
        matrix_name
    """

    return {
        "layout_version": 1,

        "template": (
            template_to_dict(template)
        ),

        "H": packing.H,
        "W": packing.W,

        "matrix_count": (
            packing.matrix_count
        ),

        "P": packing.plane_count,

        "slot_count": (
            packing.slot_count
        ),

        "total_weight_area": (
            packing.total_block_area
        ),

        "packing_utilization": (
            packing.packing_utilization
        ),

        "internal_fragmentation": (
            packing.internal_fragmentation
        ),

        "slot_size_histogram": (
            size_histogram_to_json(
                packing.size_histogram()
            )
        ),

        "planes": [
            {
                "plane_id": (
                    plane.plane_id
                ),

                "H": plane.H,
                "W": plane.W,

                "used_area": (
                    plane.used_area
                ),

                "unused_area": (
                    plane.unused_area
                ),

                "utilization": (
                    plane.utilization
                ),

                "signature": [
                    list(size_key)
                    for size_key
                    in plane.signature()
                ],

                "slots": [
                    {
                        "slot_id": (
                            slot.slot_id
                        ),

                        "plane_id": (
                            slot.plane_id
                        ),

                        "x": slot.x,
                        "y": slot.y,

                        "slot_rows": (
                            slot.slot_rows
                        ),

                        "slot_cols": (
                            slot.slot_cols
                        ),

                        "size_key": list(
                            slot.size_key
                        ),

                        "orientation_swapped": (
                            slot.orientation_swapped
                        ),
                    }
                    for slot in plane.slots
                ],
            }
            for plane in packing.planes
        ],
    }


def spatial_candidate_to_dict(
    metrics: SpatialMetrics,
    hardware_result: HardwareResolutionResult,
    layout_id: str,
) -> dict[str, Any]:
    """
    将一个 N 对应的空间候选转换成 JSON 摘要。

    注意：

    同一个 layout_id 可以对应三个 N：

        N=2
        N=3
        N=4

    因为二维匿名布局 P 与 N 无关。
    """

    hardware = (
        hardware_result.hardware
    )

    return {
        "layout_id": layout_id,

        "template_id": (
            metrics.template_id
        ),

        "N": hardware.N,
        "H": hardware.H,
        "W": hardware.W,
        "D": hardware.D,

        "num_subcubes": (
            hardware.num_subcubes
        ),

        "P_lower": (
            metrics.plane_lower_bound
        ),

        "P": (
            metrics.used_plane_count
        ),

        "Q": (
            metrics.total_plane_slots
        ),

        "total_weight_area_S": (
            metrics.total_weight_area
        ),

        "total_capacity_C": (
            metrics.total_capacity
        ),

        "capacity_ratio_C_over_S": (
            metrics.capacity_ratio
        ),

        "packing_utilization": (
            metrics.packing_utilization
        ),

        "hardware_utilization": (
            metrics.hardware_utilization
        ),

        "internal_fragmentation": (
            metrics.internal_fragmentation
        ),

        "empty_plane_slots": (
            metrics.empty_plane_slots
        ),

        "empty_plane_capacity": (
            metrics.empty_plane_capacity
        ),

        "total_unused_capacity": (
            metrics.total_unused_capacity
        ),

        "valid": metrics.valid,
    }


# ============================================================
# 文件输出
# ============================================================


def save_json(
    path: Path,
    data: Any,
) -> None:
    """
    保存格式化 JSON。
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def build_layout_id(
    H: int,
    W: int,
    template_id: str,
) -> str:
    """
    生成一个确定性的匿名布局编号。
    """

    safe_template_id = (
        template_id
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    return (
        f"H{H}_W{W}_"
        f"{safe_template_id}"
    )


# ============================================================
# 单个 H、W 的完整运行
# ============================================================


def run_one_plane_shape(
    *,
    H: int,
    W: int,
    matrix_rows: int,
    matrix_cols: int,
    matrix_count: int,
    total_weight_area: int,
    n_values: tuple[int, ...],
    allow_rotation: bool,
    output_dir: Path,
    save_layouts: bool,
    verbose: bool,
) -> tuple[
    list[SpatialMetrics],
    list[dict[str, Any]],
]:
    """
    对一个固定 H、W 执行第二步 + 第三步。

    返回：

        SpatialMetrics 列表
        JSON candidate 列表
    """

    print(
        "\n"
        "============================================================"
    )

    print(
        f"处理 Plane Shape：H={H}, W={W}"
    )

    print(
        "============================================================"
    )

    # ========================================================
    # 第二步：
    # 生成匿名切分模板
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
        allow_rotation=allow_rotation,
        raise_on_error=True,
    )

    validate_no_duplicate_templates(
        templates
    )

    print(
        f"合法切分模板数量："
        f"{len(templates)}"
    )

    all_metrics: list[
        SpatialMetrics
    ] = []

    all_candidate_dicts: list[
        dict[str, Any]
    ] = []

    # ========================================================
    # 对每个模板分别进行第三步装箱
    # ========================================================

    for template_index, template in enumerate(
        templates,
        start=1,
    ):

        print(
            "\n"
            "------------------------------------------------------------"
        )

        print(
            f"Template "
            f"{template_index}/{len(templates)}："
            f"{template.template_id}"
        )

        print(
            "单矩阵匿名块尺寸统计："
            f"{template.size_histogram}"
        )

        print(
            "整个模型匿名块总数："
            f"{template.total_block_count(matrix_count)}"
        )

        # ====================================================
        # 第三步 A：
        # 匿名二维装箱
        # ====================================================

        packing = pack_anonymous_blocks(
            template=template,
            matrix_count=matrix_count,
            H=H,
            W=W,
            allow_rotation=allow_rotation,
        )

        # ====================================================
        # 检查最终槽位尺寸数量是否正确
        # ====================================================

        expected_histogram = (
            template.total_size_histogram(
                matrix_count=matrix_count
            )
        )

        validate_slot_histogram(
            packing=packing,
            expected_histogram=(
                expected_histogram
            ),
        )

        print_packing_result(
            result=packing,
            show_planes=False,
        )

        # ====================================================
        # 保存匿名布局
        #
        # 注意：
        # 同一组 H、W、template 的布局与 N 无关。
        # 因此只保存一次。
        # ====================================================

        layout_id = build_layout_id(
            H=H,
            W=W,
            template_id=(
                template.template_id
            ),
        )

        if save_layouts:

            layout_path = (
                output_dir
                / "layouts"
                / f"{layout_id}.json"
            )

            layout_data = (
                packing_layout_to_dict(
                    packing=packing,
                    template=template,
                )
            )

            save_json(
                path=layout_path,
                data=layout_data,
            )

            print(
                "匿名布局已保存："
                f"{layout_path}"
            )

        # ====================================================
        # 第三步 B：
        # 对同一个 P 枚举 N
        #
        # 非常重要：
        # 这里不会重新运行二维装箱。
        # ====================================================

        hardware_results = resolve_all_n(
            H=H,
            W=W,

            used_plane_count=(
                packing.plane_count
            ),

            total_weight_area=(
                total_weight_area
            ),

            n_values=n_values,
        )

        for hardware_result in (
            hardware_results
        ):

            # ================================================
            # 空间指标
            # ================================================

            metrics = (
                evaluate_spatial_metrics(
                    packing=packing,
                    hardware_result=(
                        hardware_result
                    ),
                )
            )

            all_metrics.append(
                metrics
            )

            candidate_dict = (
                spatial_candidate_to_dict(
                    metrics=metrics,

                    hardware_result=(
                        hardware_result
                    ),

                    layout_id=layout_id,
                )
            )

            all_candidate_dicts.append(
                candidate_dict
            )

            if verbose:
                print()
                print_spatial_metrics(
                    metrics
                )

            else:
                print(
                    "  "
                    f"N={metrics.N}, "
                    f"D={metrics.D}, "
                    f"P={metrics.used_plane_count}, "
                    f"Q={metrics.total_plane_slots}, "
                    f"util="
                    f"{metrics.hardware_utilization:.4%}, "
                    f"valid={metrics.valid}"
                )

    return (
        all_metrics,
        all_candidate_dicts,
    )


# ============================================================
# 全部实验运行
# ============================================================


def run_spatial_baseline(
    *,
    plane_shapes: tuple[
        tuple[int, int],
        ...
    ],
    n_values: tuple[int, ...],
    matrix_count_override: int | None = None,
    output_dir: str = "results",
    save_layouts: bool = True,
    verbose: bool = False,
) -> None:
    """
    前三步完整入口。
    """

    output_path = Path(
        output_dir
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 第一步：
    # 配置
    # ========================================================

    model = ModelConfig()

    rules = ExecutionRules()

    geometry = build_model_geometry(
        config=model
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "第一步：模型与执行规则"
    )

    print(
        "============================================================"
    )

    print_geometry_summary(
        geometry
    )

    print(
        "\nWeight-Cube depth："
        f"{rules.weight_cube_depth}"
    )

    print(
        "允许旋转："
        f"{rules.allow_rotation}"
    )

    # ========================================================
    # 当前模型归一化后只有一种匿名矩阵形状
    # ========================================================

    if geometry.shape_count != 1:
        raise RuntimeError(
            "当前 Baseline 预期只有一种归一化后的匿名矩阵形状。"
        )

    matrix_demand = (
        geometry.demands[0]
    )

    matrix_rows = (
        matrix_demand.rows
    )

    matrix_cols = (
        matrix_demand.cols
    )

    # ========================================================
    # quick test 模式可以减少 matrix_count
    # ========================================================

    if matrix_count_override is None:

        matrix_count = (
            matrix_demand.count
        )

        total_weight_area = (
            geometry.total_weight_area
        )

    else:

        if matrix_count_override <= 0:
            raise ValueError(
                "matrix_count_override 必须大于 0。"
            )

        matrix_count = (
            matrix_count_override
        )

        total_weight_area = (
            matrix_count
            * matrix_rows
            * matrix_cols
        )

        print(
            "\n[测试模式]"
        )

        print(
            "匿名矩阵数量从 "
            f"{matrix_demand.count} "
            f"缩小到 {matrix_count}。"
        )

    print(
        "\n本次实际装箱匿名矩阵数量："
        f"{matrix_count}"
    )

    print(
        "本次有效权重面积 S："
        f"{total_weight_area}"
    )

    # ========================================================
    # 第二、三步
    # ========================================================

    all_metrics: list[
        SpatialMetrics
    ] = []

    all_candidate_dicts: list[
        dict[str, Any]
    ] = []

    for H, W in plane_shapes:

        metrics_list, candidate_dicts = (
            run_one_plane_shape(
                H=H,
                W=W,

                matrix_rows=matrix_rows,
                matrix_cols=matrix_cols,

                matrix_count=matrix_count,

                total_weight_area=(
                    total_weight_area
                ),

                n_values=n_values,

                allow_rotation=(
                    rules.allow_rotation
                ),

                output_dir=(
                    output_path
                ),

                save_layouts=(
                    save_layouts
                ),

                verbose=verbose,
            )
        )

        all_metrics.extend(
            metrics_list
        )

        all_candidate_dicts.extend(
            candidate_dicts
        )

    # ========================================================
    # 对全部空间候选排序
    # ========================================================

    ranked_metrics = (
        compare_spatial_metrics(
            all_metrics
        )
    )

    # 建立快速索引
    candidate_index = {
        (
            candidate[
                "template_id"
            ],
            candidate["N"],
            candidate["H"],
            candidate["W"],
            candidate["D"],
        ): candidate
        for candidate
        in all_candidate_dicts
    }

    ranked_candidate_dicts = []

    for rank, metrics in enumerate(
        ranked_metrics,
        start=1,
    ):

        key = (
            metrics.template_id,
            metrics.N,
            metrics.H,
            metrics.W,
            metrics.D,
        )

        candidate = dict(
            candidate_index[key]
        )

        candidate[
            "spatial_rank"
        ] = rank

        ranked_candidate_dicts.append(
            candidate
        )

    # ========================================================
    # 保存总结果
    # ========================================================

    summary = {
        "model": {
            "matrix_rows": (
                matrix_rows
            ),
            "matrix_cols": (
                matrix_cols
            ),
            "matrix_count": (
                matrix_count
            ),
            "total_weight_area_S": (
                total_weight_area
            ),
        },

        "rules": {
            "weight_cube_depth": (
                rules.weight_cube_depth
            ),
            "allow_rotation": (
                rules.allow_rotation
            ),
            "compute_cycles": (
                rules.compute_cycles
            ),
            "switch_cycles": (
                rules.switch_cycles
            ),
            "cross_subcube_cycles": (
                rules.cross_subcube_cycles
            ),
        },

        "plane_shapes": [
            {
                "H": H,
                "W": W,
            }
            for H, W in plane_shapes
        ],

        "N_values": list(
            n_values
        ),

        "candidate_count": len(
            ranked_candidate_dicts
        ),

        "valid_candidate_count": sum(
            1
            for candidate
            in ranked_candidate_dicts
            if candidate["valid"]
        ),

        "candidates": (
            ranked_candidate_dicts
        ),
    }

    summary_path = (
        output_path
        / "spatial_candidates.json"
    )

    save_json(
        path=summary_path,
        data=summary,
    )

    # ========================================================
    # 打印排行榜
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "前三步空间候选排名"
    )

    print(
        "============================================================"
    )

    valid_ranked = [
        metrics
        for metrics in ranked_metrics
        if metrics.valid
    ]

    if not valid_ranked:

        print(
            "没有找到满足容量限制的合法空间候选。"
        )

    else:

        # 只显示前 20 个
        for rank, metrics in enumerate(
            valid_ranked[:20],
            start=1,
        ):

            print(
                f"{rank:>2}. "
                f"H={metrics.H}, "
                f"W={metrics.W}, "
                f"N={metrics.N}, "
                f"D={metrics.D}, "
                f"P={metrics.used_plane_count}, "
                f"Q={metrics.total_plane_slots}, "
                f"packing="
                f"{metrics.packing_utilization:.4%}, "
                f"hardware="
                f"{metrics.hardware_utilization:.4%}, "
                f"C/S="
                f"{metrics.capacity_ratio:.4f}, "
                f"template="
                f"{metrics.template_id}"
            )

    print(
        "\n结果已保存到："
    )

    print(
        f"  {summary_path}"
    )

    if save_layouts:
        print(
            f"  {output_path / 'layouts'}"
        )


# ============================================================
# 命令行参数
# ============================================================


def parse_args() -> argparse.Namespace:
    """
    命令行参数。

    默认采用快速测试模式，
    避免一上来直接对 44544 个矩阵执行完整 MaxRects。

    正式运行：

        python run_spatial_baseline.py --full
    """

    parser = argparse.ArgumentParser(
        description=(
            "运行 MoE 前三步匿名空间规划 Baseline。"
        )
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "运行完整 44544 个匿名矩阵。"
            "不指定时使用小规模测试。"
        ),
    )

    parser.add_argument(
        "--test-matrices",
        type=int,
        default=32,
        help=(
            "快速测试模式使用多少个匿名矩阵，"
            "默认 32。"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="输出目录，默认 results。",
    )

    parser.add_argument(
        "--no-layouts",
        action="store_true",
        help=(
            "不保存详细 PhysicalSlot 布局。"
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印完整空间指标。",
    )

    return parser.parse_args()


# ============================================================
# main
# ============================================================


def main() -> None:

    args = parse_args()

    if args.full:
        matrix_count_override = None
    else:
        matrix_count_override = (
            args.test_matrices
        )

    run_spatial_baseline(
        plane_shapes=(
            DEFAULT_PLANE_SHAPES
        ),

        n_values=(
            DEFAULT_N_VALUES
        ),

        matrix_count_override=(
            matrix_count_override
        ),

        output_dir=(
            args.output_dir
        ),

        save_layouts=(
            not args.no_layouts
        ),

        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()