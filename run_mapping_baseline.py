"""
第四步完整 Baseline Runner。

运行流程：

Step 4.1
    构造全部 LogicalWeightCube

Step 4.2
    读取前三步保存的匿名空间 Layout

Step 4.3
    读取 Chinese-SimpleQA Expert Routing Trace

Step 4.4
    LogicalWeightCube -> LogicalPlane

Step 4.5
    LogicalPlane -> Sub-Cube

Step 4.6
    LogicalPlane -> PhysicalPlane
    LogicalWeightCube -> PhysicalSlot

Step 4.7
    保存最终静态映射结果

------------------------------------------------------------

当前默认 Baseline：

    N = 4
    H = 7168
    W = 4096

    58 MoE Layers
    256 Routed Experts / Layer
    1 Shared Expert / Layer

    44718 LogicalWeightCube
    22359 LogicalPlane

    D = 1398
    Q = 22368
    Empty Plane Slots = 9

------------------------------------------------------------

正式结果默认使用：

    全部 Chinese-SimpleQA Trace

Routed up 配对：

    Trace-aware Greedy
        +
    Local Pair Swap

------------------------------------------------------------

注意：

本文件不运行前三步 MaxRects。

空间布局直接从：

    results/spatial_candidates.json
    results/layouts/*.json

读取。
"""

from __future__ import annotations

import argparse
import json
import time

from pathlib import Path
from typing import Any


from config import (
    ModelConfig,
)

from mapping.logical_weight import (
    MATRIX_DOWN,
    MATRIX_GATE,
    MATRIX_UP,
    build_logical_weight_cubes,
)

from mapping.physical_binder import (
    PhysicalBindingResult,
    bind_logical_mapping_to_physical_slots,
    print_physical_binding_summary,
)

from mapping.plane_pairer import (
    PAIRING_MODE_SEQUENTIAL,
    PAIRING_MODE_TRACE_AWARE,
    PAIRING_MODES,
    PairingResult,
    build_logical_planes,
    print_pairing_summary,
)

from mapping.spatial_layout_loader import (
    DEFAULT_RESULTS_DIR,
    SpatialLayoutBundle,
    load_spatial_layout_bundle,
    print_spatial_layout_bundle,
)

from mapping.subcube_mapper import (
    MAPPING_MODE_ROUND_ROBIN,
    MAPPING_MODE_TRACE_AWARE,
    MAPPING_MODES,
    SubcubeMappingResult,
    map_logical_planes_to_subcubes,
    print_subcube_mapping_summary,
)

from mapping.trace_profile import (
    DEFAULT_TRACE_ROOT,
    TraceProfile,
    load_chinese_simpleqa_profile,
    print_profile_summary,
)


# ============================================================
# 当前 Baseline 默认硬件
# ============================================================


DEFAULT_N = 4

DEFAULT_H = 7168

DEFAULT_W = 4096


# ============================================================
# 异常
# ============================================================


class MappingBaselineError(ValueError):
    """第四步完整 Baseline 执行失败。"""


# ============================================================
# 时间输出
# ============================================================


def print_stage_time(
    stage_name: str,
    start_time: float,
) -> float:
    """
    打印一个阶段耗时。

    返回当前时间，
    可以直接作为下一阶段 start_time。
    """

    now = time.perf_counter()

    elapsed = (
        now
        - start_time
    )

    print(
        f"[Time] {stage_name}: "
        f"{elapsed:.3f} s"
    )

    return now


# ============================================================
# Logical WeightCube 摘要
# ============================================================


def print_logical_weight_summary(
    cubes,
) -> None:
    """
    打印真实逻辑矩阵数量。
    """

    total = len(
        cubes
    )

    gate_count = sum(
        1
        for cube in cubes
        if cube.matrix_name
        == MATRIX_GATE
    )

    up_count = sum(
        1
        for cube in cubes
        if cube.matrix_name
        == MATRIX_UP
    )

    down_count = sum(
        1
        for cube in cubes
        if cube.matrix_name
        == MATRIX_DOWN
    )

    shared_count = sum(
        1
        for cube in cubes
        if cube.is_shared
    )

    routed_count = (
        total
        - shared_count
    )

    print(
        "\n"
        "========== Logical Weight-Cubes =========="
    )

    print(
        f"Total Weight-Cubes："
        f"{total}"
    )

    print(
        f"Routed Weight-Cubes："
        f"{routed_count}"
    )

    print(
        f"Shared Weight-Cubes："
        f"{shared_count}"
    )

    print(
        f"gate："
        f"{gate_count}"
    )

    print(
        f"up："
        f"{up_count}"
    )

    print(
        f"down："
        f"{down_count}"
    )


# ============================================================
# 最终一致性检查
# ============================================================


def validate_complete_mapping(
    *,
    cubes,
    spatial: SpatialLayoutBundle,
    pairing: PairingResult,
    subcube_mapping: SubcubeMappingResult,
    binding: PhysicalBindingResult,
) -> None:
    """
    第四步结束后的总检查。

    这里不重新实现各模块内部验证，
    只检查几个跨模块最重要的数量关系。
    """

    cube_count = len(
        cubes
    )

    logical_plane_count = len(
        pairing.planes
    )

    physical_plane_count = len(
        spatial.physical_planes
    )

    placement_count = len(
        binding.placements
    )

    # ========================================================
    # Cube
    # ========================================================

    if (
        placement_count
        != cube_count
    ):
        raise MappingBaselineError(
            "最终 WeightCube Placement 数量错误："
            f"placements={placement_count}, "
            f"cubes={cube_count}。"
        )

    # ========================================================
    # Logical Plane / Physical Plane
    # ========================================================

    if (
        logical_plane_count
        != physical_plane_count
    ):
        raise MappingBaselineError(
            "LogicalPlane 与 PhysicalPlane "
            "数量不一致："
            f"logical={logical_plane_count}, "
            f"physical={physical_plane_count}。"
        )

    if (
        logical_plane_count
        != subcube_mapping.total_planes
    ):
        raise MappingBaselineError(
            "PairingResult 与 "
            "SubcubeMappingResult "
            "Plane 数量不一致。"
        )

    if (
        binding.physical_plane_count
        != logical_plane_count
    ):
        raise MappingBaselineError(
            "PhysicalBindingResult "
            "Plane 数量错误。"
        )

    # ========================================================
    # Slot
    # ========================================================

    if (
        spatial.slot_count
        != cube_count
    ):
        raise MappingBaselineError(
            "PhysicalSlot 数量与 "
            "LogicalWeightCube 数量不一致："
            f"slots={spatial.slot_count}, "
            f"cubes={cube_count}。"
        )

    # ========================================================
    # Capacity
    # ========================================================

    if (
        spatial.plane_count
        > spatial.hardware
        .total_plane_slots
    ):
        raise MappingBaselineError(
            "P > Q。"
        )

    # ========================================================
    # Placement ID 连续
    # ========================================================

    expected_cube_ids = list(
        range(
            cube_count
        )
    )

    actual_cube_ids = [
        placement.cube_id
        for placement
        in binding.placements
    ]

    if (
        actual_cube_ids
        != expected_cube_ids
    ):
        raise MappingBaselineError(
            "最终 Placement 的 cube_id "
            "不是连续的 0...N-1。"
        )


# ============================================================
# JSON Helper
# ============================================================


def _pair_list_to_json(
    pairs,
) -> list[
    list[int]
]:
    """
    tuple[(a,b), ...]
    ->
    list[[a,b], ...]
    """

    return [
        [
            int(a),
            int(b),
        ]
        for a, b
        in pairs
    ]


# ============================================================
# 保存最终第四步结果
# ============================================================


def build_output_dict(
    *,
    cubes,
    profile: TraceProfile,
    spatial: SpatialLayoutBundle,
    pairing: PairingResult,
    subcube_mapping: SubcubeMappingResult,
    binding: PhysicalBindingResult,
    pairing_mode: str,
    mapping_mode: str,
    local_search_enabled: bool,
    local_search_rounds: int,
) -> dict[
    str,
    Any,
]:
    """
    构造最终 JSON。

    不保存完整 coactivation 矩阵，
    因为它非常大，而且第五步不需要。

    保存真正需要的：

        WeightCube 最终物理位置
        Plane 配对
        Sub-Cube 信息
        Trace 摘要
    """

    # ========================================================
    # Routed up Pair
    # ========================================================

    routed_pairs_json = []

    for (
        layer_id,
        layer_pairs,
    ) in enumerate(
        pairing
        .routed_up_pairs_by_layer
    ):

        routed_pairs_json.append(
            {
                "layer_id": (
                    layer_id
                ),

                "pairs": (
                    _pair_list_to_json(
                        layer_pairs
                    )
                ),

                "coactivation_cost": int(
                    pairing
                    .routed_up_coactivation_cost_by_layer[
                        layer_id
                    ]
                ),
            }
        )

    # ========================================================
    # Final WeightCube Placements
    # ========================================================

    placements_json = []

    for placement in (
        binding.placements
    ):

        placements_json.append(
            {
                # ============================================
                # Logical Identity
                # ============================================

                "cube_id": (
                    placement.cube_id
                ),

                "layer_id": (
                    placement.layer_id
                ),

                "expert_id": (
                    placement.expert_id
                ),

                "is_shared": (
                    placement.is_shared
                ),

                "matrix_name": (
                    placement.matrix_name
                ),

                "logical_rows": (
                    placement.logical_rows
                ),

                "logical_cols": (
                    placement.logical_cols
                ),

                "depth": (
                    placement.depth
                ),

                # ============================================
                # Logical Plane
                # ============================================

                "logical_plane_id": (
                    placement
                    .logical_plane_id
                ),

                # ============================================
                # 3D Hardware
                # ============================================

                "subcube_id": (
                    placement.subcube_id
                ),

                "z": (
                    placement.z
                ),

                # ============================================
                # Physical Plane / Slot
                # ============================================

                "physical_plane_id": (
                    placement
                    .physical_plane_id
                ),

                "slot_id": (
                    placement.slot_id
                ),

                "x": (
                    placement.x
                ),

                "y": (
                    placement.y
                ),

                "slot_rows": (
                    placement.slot_rows
                ),

                "slot_cols": (
                    placement.slot_cols
                ),

                # ============================================
                # Rotation
                # ============================================

                "slot_orientation_swapped": (
                    placement
                    .slot_orientation_swapped
                ),

                "logical_cube_rotated": (
                    placement
                    .logical_cube_rotated
                ),
            }
        )

    # ========================================================
    # Plane Binding
    # ========================================================

    plane_bindings_json = [
        {
            "logical_plane_id": (
                logical_plane_id
            ),

            "physical_plane_id": (
                physical_plane_id
            ),
        }

        for (
            logical_plane_id,
            physical_plane_id,
        ) in binding.plane_bindings
    ]

    # ========================================================
    # Main Dict
    # ========================================================

    output = {
        "mapping_version": 1,

        # ====================================================
        # Model
        # ====================================================

        "model": {
            "num_logical_weight_cubes": (
                len(cubes)
            ),

            "num_logical_planes": (
                len(pairing.planes)
            ),

            "shared_expert_enabled": (
                True
            ),
        },

        # ====================================================
        # Trace
        # ====================================================

        "trace": {
            "dataset": (
                "Chinese-SimpleQA"
            ),

            "file_count": (
                profile.file_count
            ),

            "trace_segment_count": (
                profile
                .trace_segment_count
            ),

            "valid_segment_count": (
                profile
                .valid_segment_count
            ),

            "skipped_segment_count": (
                profile
                .skipped_segment_count
            ),

            "tokens_per_layer": (
                profile
                .tokens_per_layer
            ),

            "total_expert_selections": (
                profile
                .total_expert_selections
            ),

            "category_file_counts": (
                profile
                .category_file_counts
            ),
        },

        # ====================================================
        # Spatial
        # ====================================================

        "spatial": {
            "layout_id": (
                spatial.layout_id
            ),

            "template_id": (
                spatial.template_id
            ),

            "spatial_rank": (
                spatial.spatial_rank
            ),

            "N": (
                spatial.hardware.N
            ),

            "H": (
                spatial.hardware.H
            ),

            "W": (
                spatial.hardware.W
            ),

            "D": (
                spatial.hardware.D
            ),

            "num_subcubes": (
                spatial.hardware
                .num_subcubes
            ),

            "P": (
                spatial.plane_count
            ),

            "Q": (
                spatial.hardware
                .total_plane_slots
            ),

            "empty_plane_slots": (
                spatial
                .empty_plane_slots
            ),

            "slot_count": (
                spatial.slot_count
            ),
        },

        # ====================================================
        # Pairing
        # ====================================================

        "pairing": {
            "mode": (
                pairing_mode
            ),

            "local_search_enabled": (
                local_search_enabled
            ),

            "local_search_rounds": (
                local_search_rounds
            ),

            "gate_down_plane_count": (
                pairing
                .gate_down_plane_count
            ),

            "routed_up_plane_count": (
                pairing
                .routed_up_plane_count
            ),

            "shared_up_plane_count": (
                pairing
                .shared_up_plane_count
            ),

            "total_routed_up_coactivation_cost": (
                pairing
                .total_routed_up_coactivation_cost
            ),

            "shared_up_layer_pairs": (
                _pair_list_to_json(
                    pairing
                    .shared_up_layer_pairs
                )
            ),

            "routed_up_pairs_by_layer": (
                routed_pairs_json
            ),
        },

        # ====================================================
        # Sub-Cube Mapping
        # ====================================================

        "subcube_mapping": {
            "mode": (
                mapping_mode
            ),

            "plane_counts": [
                int(value)
                for value
                in subcube_mapping
                .subcube_plane_counts
            ],

            "pre_conflict_cost": (
                subcube_mapping
                .pre_conflict_cost
            ),

            "down_conflict_cost": (
                subcube_mapping
                .down_conflict_cost
            ),

            "total_conflict_cost": (
                subcube_mapping
                .total_conflict_cost
            ),

            "gate_down_subcube_by_layer": [
                [
                    int(value)
                    for value
                    in row
                ]
                for row
                in subcube_mapping
                .gate_down_subcube_by_layer
            ],

            "pre_weighted_load_by_layer": [
                [
                    int(value)
                    for value
                    in row
                ]
                for row
                in subcube_mapping
                .pre_weighted_load_by_layer
            ],

            "down_weighted_load_by_layer": [
                [
                    int(value)
                    for value
                    in row
                ]
                for row
                in subcube_mapping
                .down_weighted_load_by_layer
            ],
        },

        # ====================================================
        # Physical Binding
        # ====================================================

        "physical_binding": {
            "logical_rotation_count": (
                binding
                .logical_rotation_count
            ),

            "unrotated_count": (
                binding
                .unrotated_count
            ),

            "plane_bindings": (
                plane_bindings_json
            ),
        },

        # ====================================================
        # 最重要：
        # 第五步直接读取它即可
        # ====================================================

        "placements": (
            placements_json
        ),
    }

    return output


def save_mapping_result(
    *,
    output_path: Path,
    data: dict[
        str,
        Any,
    ],
) -> None:
    """
    保存第四步最终 JSON。
    """

    output_path = (
        output_path.resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,

            ensure_ascii=False,

            indent=2,
        )


# ============================================================
# 默认输出文件
# ============================================================


def build_default_output_path(
    *,
    results_dir: Path,
    spatial: SpatialLayoutBundle,
    pairing_mode: str,
    mapping_mode: str,
    max_files: int | None,
) -> Path:
    """
    根据 Pairing + Mapping 两个模式自动命名四组主消融：

        trace_aware + trace_aware
            -> mapping_baseline_...

        sequential + trace_aware
            -> mapping_mapping_only_...

        trace_aware + round_robin
            -> mapping_pairing_only_...

        sequential + round_robin
            -> mapping_naive_...

    Full Baseline 继续保留原文件名，
    不破坏现有 Prefill / Decode / WebUI 默认路径。
    """

    mode_pair = (
        pairing_mode,
        mapping_mode,
    )

    if mode_pair == (
        PAIRING_MODE_TRACE_AWARE,
        MAPPING_MODE_TRACE_AWARE,
    ):
        experiment_name = "baseline"

    elif mode_pair == (
        PAIRING_MODE_SEQUENTIAL,
        MAPPING_MODE_TRACE_AWARE,
    ):
        experiment_name = "mapping_only"

    elif mode_pair == (
        PAIRING_MODE_TRACE_AWARE,
        MAPPING_MODE_ROUND_ROBIN,
    ):
        experiment_name = "pairing_only"

    elif mode_pair == (
        PAIRING_MODE_SEQUENTIAL,
        MAPPING_MODE_ROUND_ROBIN,
    ):
        experiment_name = "naive"

    else:
        raise MappingBaselineError(
            "未知 Pairing/Mapping 模式组合："
            f"pairing={pairing_mode!r}, "
            f"mapping={mapping_mode!r}。"
        )

    hardware = spatial.hardware

    if max_files is None:

        filename = (
            f"mapping_{experiment_name}_"
            f"N{hardware.N}_"
            f"H{hardware.H}_"
            f"W{hardware.W}.json"
        )

    else:

        filename = (
            f"mapping_{experiment_name}_debug_"
            f"{max_files}files_"
            f"N{hardware.N}_"
            f"H{hardware.H}_"
            f"W{hardware.W}.json"
        )

    return (
        results_dir
        / "mappings"
        / filename
    )


# ============================================================
# Runner
# ============================================================


def run_mapping_baseline(
    args,
) -> None:
    """
    第四步完整执行入口。
    """

    total_start = (
        time.perf_counter()
    )

    stage_start = (
        total_start
    )

    print(
        "\n"
        "=============================================="
    )

    print(
        "       Step 4 Mapping Baseline"
    )

    print(
        "=============================================="
    )

    # ========================================================
    # 1. Logical WeightCube
    # ========================================================

    print(
        "\n[Stage 1] "
        "Build LogicalWeightCube"
    )

    model_config = (
        ModelConfig(
            include_shared_expert=True
        )
    )

    cubes = (
        build_logical_weight_cubes(
            model_config
        )
    )

    print_logical_weight_summary(
        cubes
    )

    stage_start = (
        print_stage_time(
            "LogicalWeightCube",
            stage_start,
        )
    )

    # ========================================================
    # 2. 加载前三步空间布局
    #
    # 先做这个，
    # 如果 results 是旧的 44544 版本，
    # 可以在读取 Trace 之前立即失败。
    # ========================================================

    print(
        "\n[Stage 2] "
        "Load Spatial Layout"
    )

    spatial = (
        load_spatial_layout_bundle(
            results_dir=(
                args.results_dir
            ),

            spatial_rank=(
                args.spatial_rank
            ),

            N=args.N,
            H=args.H,
            W=args.W,

            layout_id=(
                args.layout_id
            ),

            expected_matrix_count=(
                len(cubes)
            ),

            require_single_chunk=True,
        )
    )

    print_spatial_layout_bundle(
        spatial
    )

    stage_start = (
        print_stage_time(
            "Spatial Layout Loader",
            stage_start,
        )
    )

    # ========================================================
    # 3. Trace Profile
    # ========================================================

    print(
        "\n[Stage 3] "
        "Load Chinese-SimpleQA Trace"
    )

    profile = (
        load_chinese_simpleqa_profile(
            trace_root=(
                args.trace_root
            ),

            max_files=(
                args.max_files
            ),

            strict=True,

            verbose=(
                not args.quiet
            ),
        )
    )

    print_profile_summary(
        profile,
        top_k=5,
    )

    stage_start = (
        print_stage_time(
            "Trace Profile",
            stage_start,
        )
    )

    # ========================================================
    # 4. Matrix -> LogicalPlane
    # ========================================================

    print(
        "\n[Stage 4] "
        "Logical Plane Pairing"
    )

    # Sequential Pairing 本身不使用 Local Search。
    # --no-local-search 只对 trace_aware 模式有意义。
    local_search_enabled = (
        args.pairing_mode
        == PAIRING_MODE_TRACE_AWARE
        and
        not args.no_local_search
    )

    print(
        f"Pairing Mode：{args.pairing_mode}"
    )

    pairing = (
        build_logical_planes(
            cubes=cubes,

            profile=profile,

            pairing_mode=(
                args.pairing_mode
            ),

            improve_pairs=(
                local_search_enabled
            ),

            local_search_rounds=(
                args.local_search_rounds
            ),
        )
    )

    print_pairing_summary(
        pairing
    )

    # 在进入下一步前提前检查
    if (
        len(pairing.planes)
        != spatial.plane_count
    ):

        raise MappingBaselineError(
            "LogicalPlane 数量与 "
            "第三步 PhysicalPlane 数量不一致："
            f"logical="
            f"{len(pairing.planes)}, "
            f"physical="
            f"{spatial.plane_count}。"
        )

    stage_start = (
        print_stage_time(
            "Plane Pairing",
            stage_start,
        )
    )

    # ========================================================
    # 5. LogicalPlane -> Sub-Cube
    # ========================================================

    print(
        "\n[Stage 5] "
        "Map LogicalPlane to Sub-Cube"
    )

    print(
        f"Mapping Mode：{args.mapping_mode}"
    )

    subcube_mapping = (
        map_logical_planes_to_subcubes(
            pairing=pairing,

            cubes=cubes,

            profile=profile,

            hardware=(
                spatial.hardware
            ),

            mapping_mode=(
                args.mapping_mode
            ),
        )
    )

    print_subcube_mapping_summary(
        subcube_mapping
    )

    stage_start = (
        print_stage_time(
            "Sub-Cube Mapping",
            stage_start,
        )
    )

    # ========================================================
    # 6. Physical Binding
    # ========================================================

    print(
        "\n[Stage 6] "
        "Bind Logical Mapping to Physical Slots"
    )

    binding = (
        bind_logical_mapping_to_physical_slots(
            cubes=cubes,

            pairing=pairing,

            subcube_mapping=(
                subcube_mapping
            ),

            physical_planes=(
                spatial
                .physical_planes
            ),
        )
    )

    print_physical_binding_summary(
        binding
    )

    stage_start = (
        print_stage_time(
            "Physical Binding",
            stage_start,
        )
    )

    # ========================================================
    # 7. 跨模块总验证
    # ========================================================

    print(
        "\n[Stage 7] "
        "Final Validation"
    )

    validate_complete_mapping(
        cubes=cubes,

        spatial=spatial,

        pairing=pairing,

        subcube_mapping=(
            subcube_mapping
        ),

        binding=binding,
    )

    print(
        "Final Validation：PASS"
    )

    stage_start = (
        print_stage_time(
            "Final Validation",
            stage_start,
        )
    )

    # ========================================================
    # 8. 保存
    # ========================================================

    if not args.no_save:

        print(
            "\n[Stage 8] "
            "Save Mapping Result"
        )

        output_data = (
            build_output_dict(
                cubes=cubes,

                profile=profile,

                spatial=spatial,

                pairing=pairing,

                subcube_mapping=(
                    subcube_mapping
                ),

                binding=binding,

                pairing_mode=(
                    args.pairing_mode
                ),

                mapping_mode=(
                    args.mapping_mode
                ),

                local_search_enabled=(
                    local_search_enabled
                ),

                local_search_rounds=(
                    args.local_search_rounds
                ),
            )
        )

        if args.output is None:

            output_path = (
                build_default_output_path(
                    results_dir=(
                        args.results_dir
                    ),

                    spatial=spatial,

                    pairing_mode=(
                        args.pairing_mode
                    ),

                    mapping_mode=(
                        args.mapping_mode
                    ),

                    max_files=(
                        args.max_files
                    ),
                )
            )

        else:

            output_path = (
                args.output
            )

        save_mapping_result(
            output_path=(
                output_path
            ),

            data=(
                output_data
            ),
        )

        print(
            "Saved："
            f"{output_path.resolve()}"
        )

        stage_start = (
            print_stage_time(
                "Save Result",
                stage_start,
            )
        )

    # ========================================================
    # 最终摘要
    # ========================================================

    total_time = (
        time.perf_counter()
        - total_start
    )

    print(
        "\n"
        "=============================================="
    )

    print(
        "       Step 4 Mapping Completed"
    )

    print(
        "=============================================="
    )

    print(
        f"Weight-Cubes："
        f"{len(cubes)}"
    )

    print(
        f"Logical Planes："
        f"{len(pairing.planes)}"
    )

    print(
        f"Sub-Cubes："
        f"{spatial.hardware.num_subcubes}"
    )

    print(
        f"D："
        f"{spatial.hardware.D}"
    )

    print(
        f"P："
        f"{spatial.plane_count}"
    )

    print(
        f"Q："
        f"{spatial.hardware.total_plane_slots}"
    )

    print(
        f"Empty Plane Slots："
        f"{spatial.empty_plane_slots}"
    )

    print(
        f"Pairing Mode："
        f"{args.pairing_mode}"
    )

    print(
        f"Mapping Mode："
        f"{args.mapping_mode}"
    )

    print(
        "Pairing Coactivation Cost："
        f"{pairing.total_routed_up_coactivation_cost}"
    )

    print(
        "Mapping Conflict Cost："
        f"{subcube_mapping.total_conflict_cost}"
    )

    print(
        f"Logical Rotations："
        f"{binding.logical_rotation_count}"
    )

    print(
        f"Total Runtime："
        f"{total_time:.3f} s"
    )


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "运行第四步完整 "
                "MoE PIM Mapping Baseline。"
            )
        )
    )

    # ========================================================
    # Spatial
    # ========================================================

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )

    parser.add_argument(
        "--N",
        type=int,
        default=DEFAULT_N,
    )

    parser.add_argument(
        "--H",
        type=int,
        default=DEFAULT_H,
    )

    parser.add_argument(
        "--W",
        type=int,
        default=DEFAULT_W,
    )

    parser.add_argument(
        "--spatial-rank",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--layout-id",
        type=str,
        default=None,
    )

    # ========================================================
    # Trace
    # ========================================================

    parser.add_argument(
        "--trace-root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,

        help=(
            "只读取前 N 个 Trace JSON。"
            "仅用于快速测试；"
            "正式 Baseline 不要设置。"
        ),
    )

    # ========================================================
    # Pairing
    # ========================================================

    parser.add_argument(
        "--pairing-mode",
        type=str,
        default=(
            PAIRING_MODE_TRACE_AWARE
        ),
        choices=PAIRING_MODES,
        help=(
            "Routed up Pairing 策略："
            "trace_aware=当前 Greedy+Local Search；"
            "sequential=固定 E0-E1、E2-E3...，"
            "用于消融实验。"
        ),
    )

    parser.add_argument(
        "--no-local-search",
        action="store_true",

        help=(
            "关闭 Routed up Pair "
            "局部交换优化，"
            "仅使用 Greedy。"
        ),
    )

    parser.add_argument(
        "--local-search-rounds",
        type=int,
        default=4,
    )

    # ========================================================
    # Sub-Cube Mapping
    # ========================================================

    parser.add_argument(
        "--mapping-mode",
        type=str,
        default=(
            MAPPING_MODE_TRACE_AWARE
        ),
        choices=MAPPING_MODES,
        help=(
            "LogicalPlane -> Sub-Cube 策略："
            "trace_aware=当前正式 Trace-aware Mapping；"
            "round_robin=不使用 Trace 决策的受约束轮询，"
            "用于消融实验。"
        ),
    )

    # ========================================================
    # Output
    # ========================================================

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "减少 Trace 读取过程输出。"
        ),
    )

    args = (
        parser.parse_args()
    )

    if (
        args.local_search_rounds
        < 0
    ):

        parser.error(
            "--local-search-rounds "
            "不能小于 0。"
        )

    run_mapping_baseline(
        args
    )


if __name__ == "__main__":
    main()
