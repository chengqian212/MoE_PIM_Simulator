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
    所有 pair 共激活为 0。

    这里只测试映射结构是否合法。
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
            [0],
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