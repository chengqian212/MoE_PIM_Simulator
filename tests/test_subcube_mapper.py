from array import array

import pytest

from config import (
    ModelConfig,
)

from evaluation.hardware_resolver import (
    ResolvedHardwareConfig,
)

from mapping.logical_weight import (
    build_logical_weight_cubes,
)

from mapping.plane_pairer import (
    build_logical_planes,
)

from mapping.subcube_mapper import (
    MAPPING_MODE_ROUND_ROBIN,
    MAPPING_MODE_TRACE_AWARE,
    map_logical_planes_to_subcubes,
)

from mapping.trace_profile import (
    NUM_MOE_LAYERS,
    NUM_ROUTED_EXPERTS,
    TraceProfile,
)


def build_fake_profile():
    """
    所有 Routed Expert 热度相同，
    所有不同 Routed Expert pair 共激活统一设为 1。

    这里只测试映射结构是否合法。

    注意：不能把所有 Routed-Routed 共激活设为 0。
    Shared Expert 在当前模型中每个 token 都激活，
    如果 Routed-Routed=0 而 Shared-Routed>0，
    会形成一个极端退化 Trace，使 Trace-aware Mapper
    系统性回避 Shared 所在 SC，最终可能造成容量死角。
    """

    frequency = tuple(
        tuple(
            1
            for _ in range(
                NUM_ROUTED_EXPERTS
            )
        )
        for _ in range(
            NUM_MOE_LAYERS
        )
    )

    pair_size = (
        NUM_ROUTED_EXPERTS
        * NUM_ROUTED_EXPERTS
    )

    coactivation = tuple(
        array(
            "Q",
            [1],
        )
        * pair_size

        for _ in range(
            NUM_MOE_LAYERS
        )
    )

    token_count = tuple(
        1
        for _ in range(
            NUM_MOE_LAYERS
        )
    )

    return TraceProfile(
        file_count=1,

        trace_segment_count=1,

        skipped_segment_count=0,

        category_file_counts={
            "test": 1
        },

        frequency=frequency,

        coactivation=coactivation,

        token_count_by_layer=(
            token_count
        ),
    )


@pytest.fixture(
    scope="module"
)
def mapping_result():

    # ========================================================
    # LogicalWeightCube
    # ========================================================

    config = ModelConfig(
        include_shared_expert=True,
    )

    cubes = (
        build_logical_weight_cubes(
            config
        )
    )

    # ========================================================
    # Fake Trace
    # ========================================================

    profile = (
        build_fake_profile()
    )

    # ========================================================
    # Matrix -> Plane
    # ========================================================

    pairing = (
        build_logical_planes(
            cubes=cubes,
            profile=profile,

            # 测试结构即可，
            # 不需要跑 local search。
            improve_pairs=False,
        )
    )

    # ========================================================
    # 当前最佳硬件
    #
    # N=4
    # H=7168
    # W=4096
    # D=1398
    # ========================================================

    hardware = (
        ResolvedHardwareConfig(
            N=4,
            H=7168,
            W=4096,
            D=1398,
        )
    )

    # ========================================================
    # Plane -> Sub-Cube
    # ========================================================

    result = (
        map_logical_planes_to_subcubes(
            pairing=pairing,
            cubes=cubes,
            profile=profile,
            hardware=hardware,
            mapping_mode=(
                MAPPING_MODE_TRACE_AWARE
            ),
        )
    )

    return (
        cubes,
        pairing,
        result,
    )


def test_total_plane_count(
    mapping_result,
):

    _, _, result = (
        mapping_result
    )

    assert (
        result.total_planes
        == 22359
    )


def test_empty_plane_slots(
    mapping_result,
):

    _, _, result = (
        mapping_result
    )

    # Q = 16 × 1398 = 22368
    #
    # Q - P
    # = 22368 - 22359
    # = 9

    assert (
        result.empty_plane_slots
        == 9
    )


def test_subcube_count(
    mapping_result,
):

    _, _, result = (
        mapping_result
    )

    assert (
        len(
            result.subcube_plane_counts
        )
        == 16
    )


def test_all_planes_fit_depth(
    mapping_result,
):

    _, _, result = (
        mapping_result
    )

    assert all(
        count <= 1398

        for count
        in result.subcube_plane_counts
    )

    assert (
        sum(
            result.subcube_plane_counts
        )
        == 22359
    )


def test_all_physical_plane_coordinates_unique(
    mapping_result,
):

    _, _, result = (
        mapping_result
    )

    coordinates = {
        (
            placement.subcube_id,
            placement.z,
        )

        for placement
        in result.placements
    }

    assert (
        len(coordinates)
        == 22359
    )


def test_gate_up_are_separated(
    mapping_result,
):

    cubes, pairing, result = (
        mapping_result
    )

    from mapping.logical_plane import (
        build_cube_index,
        get_plane_cubes,
    )

    from mapping.logical_weight import (
        MATRIX_GATE,
        MATRIX_UP,
    )

    cube_index = (
        build_cube_index(
            cubes
        )
    )

    placement_by_id = {
        placement.logical_plane_id:
        placement

        for placement
        in result.placements
    }

    gate_sc = {}
    up_sc = {}

    for plane in pairing.planes:

        sc = (
            placement_by_id[
                plane.logical_plane_id
            ].subcube_id
        )

        cube_a, cube_b = (
            get_plane_cubes(
                plane=plane,
                cube_index=cube_index,
            )
        )

        for cube in (
            cube_a,
            cube_b,
        ):

            key = (
                cube.layer_id,
                cube.expert_id,
            )

            if (
                cube.matrix_name
                == MATRIX_GATE
            ):

                gate_sc[key] = sc

            elif (
                cube.matrix_name
                == MATRIX_UP
            ):

                up_sc[key] = sc

    assert (
        len(gate_sc)
        == 58 * 257
    )

    assert (
        len(up_sc)
        == 58 * 257
    )

    for key in gate_sc:

        assert (
            gate_sc[key]
            != up_sc[key]
        )


@pytest.fixture(
    scope="module"
)
def round_robin_mapping_result():

    config = ModelConfig(
        include_shared_expert=True,
    )

    cubes = (
        build_logical_weight_cubes(
            config
        )
    )

    profile = (
        build_fake_profile()
    )

    pairing = (
        build_logical_planes(
            cubes=cubes,
            profile=profile,
            improve_pairs=False,
        )
    )

    hardware = (
        ResolvedHardwareConfig(
            N=4,
            H=7168,
            W=4096,
            D=1398,
        )
    )

    result = (
        map_logical_planes_to_subcubes(
            pairing=pairing,
            cubes=cubes,
            profile=profile,
            hardware=hardware,
            mapping_mode=(
                MAPPING_MODE_ROUND_ROBIN
            ),
        )
    )

    return (
        cubes,
        pairing,
        profile,
        hardware,
        result,
    )


def test_round_robin_total_plane_count(
    round_robin_mapping_result,
):

    _, _, _, _, result = (
        round_robin_mapping_result
    )

    assert (
        result.total_planes
        == 22359
    )

    assert (
        sum(
            result.subcube_plane_counts
        )
        == 22359
    )


def test_round_robin_balances_plane_counts(
    round_robin_mapping_result,
):

    _, _, _, hardware, result = (
        round_robin_mapping_result
    )

    counts = (
        result.subcube_plane_counts
    )

    assert (
        max(counts)
        <= hardware.D
    )

    # 受约束轮询仍应尽量均衡。
    assert (
        max(counts)
        - min(counts)
        <= 1
    )


def test_round_robin_first_cycle_is_sc0_to_sc15(
    round_robin_mapping_result,
):

    _, _, _, hardware, result = (
        round_robin_mapping_result
    )

    first_cycle = tuple(
        placement.subcube_id
        for placement
        in result.placements[
            :hardware.num_subcubes
        ]
    )

    assert (
        first_cycle
        == tuple(
            range(
                hardware.num_subcubes
            )
        )
    )


def test_round_robin_gate_up_are_separated(
    round_robin_mapping_result,
):

    cubes, pairing, _, _, result = (
        round_robin_mapping_result
    )

    from mapping.logical_plane import (
        build_cube_index,
        get_plane_cubes,
    )

    from mapping.logical_weight import (
        MATRIX_GATE,
        MATRIX_UP,
    )

    cube_index = (
        build_cube_index(
            cubes
        )
    )

    placement_by_id = {
        placement.logical_plane_id:
        placement
        for placement
        in result.placements
    }

    gate_sc = {}
    up_sc = {}

    for plane in pairing.planes:

        sc = (
            placement_by_id[
                plane.logical_plane_id
            ].subcube_id
        )

        cube_a, cube_b = (
            get_plane_cubes(
                plane=plane,
                cube_index=cube_index,
            )
        )

        for cube in (
            cube_a,
            cube_b,
        ):

            key = (
                cube.layer_id,
                cube.expert_id,
            )

            if (
                cube.matrix_name
                == MATRIX_GATE
            ):
                gate_sc[key] = sc

            elif (
                cube.matrix_name
                == MATRIX_UP
            ):
                up_sc[key] = sc

    assert (
        len(gate_sc)
        == 58 * 257
    )

    assert (
        len(up_sc)
        == 58 * 257
    )

    for key in gate_sc:

        assert (
            gate_sc[key]
            != up_sc[key]
        )


def test_round_robin_decision_does_not_depend_on_trace(
    round_robin_mapping_result,
):
    """
    改变 frequency 后，Round-Robin 的物理选择
    必须完全不变。

    Trace 仍可用于事后统计 load/conflict，
    但不能影响 Plane -> SC 决策。
    """

    (
        cubes,
        pairing,
        base_profile,
        hardware,
        base_result,
    ) = round_robin_mapping_result

    hot_frequency = tuple(
        tuple(
            (
                1000
                if expert_id == (
                    layer_id
                    % NUM_ROUTED_EXPERTS
                )
                else 1
            )
            for expert_id
            in range(
                NUM_ROUTED_EXPERTS
            )
        )
        for layer_id
        in range(
            NUM_MOE_LAYERS
        )
    )

    hot_profile = TraceProfile(
        file_count=1,
        trace_segment_count=1,
        skipped_segment_count=0,
        category_file_counts={
            "test": 1
        },
        frequency=(
            hot_frequency
        ),
        coactivation=(
            base_profile.coactivation
        ),
        token_count_by_layer=tuple(
            1000
            for _ in range(
                NUM_MOE_LAYERS
            )
        ),
    )

    hot_result = (
        map_logical_planes_to_subcubes(
            pairing=pairing,
            cubes=cubes,
            profile=hot_profile,
            hardware=hardware,
            mapping_mode=(
                MAPPING_MODE_ROUND_ROBIN
            ),
        )
    )

    base_locations = tuple(
        (
            placement.subcube_id,
            placement.z,
        )
        for placement
        in base_result.placements
    )

    hot_locations = tuple(
        (
            placement.subcube_id,
            placement.z,
        )
        for placement
        in hot_result.placements
    )

    assert (
        hot_locations
        == base_locations
    )
