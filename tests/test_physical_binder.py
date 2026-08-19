from array import array

from evaluation.hardware_resolver import (
    ResolvedHardwareConfig,
)

from mapping.logical_plane import (
    create_gate_down_plane,
    create_up_up_plane,
)

from mapping.logical_weight import (
    LogicalWeightCube,
    MATRIX_DOWN,
    MATRIX_GATE,
    MATRIX_UP,
)

from mapping.physical_binder import (
    bind_logical_mapping_to_physical_slots,
)

from mapping.plane_pairer import (
    PairingResult,
)

from mapping.subcube_mapper import (
    LogicalPlanePlacement,
    SubcubeMappingResult,
)

from packing.physical_slot import (
    PhysicalSlot,
)

from packing.plane import (
    create_empty_plane,
)


# ============================================================
# Logical Cubes
# ============================================================


def build_cubes():

    # Expert-0
    gate_0 = LogicalWeightCube(
        cube_id=0,
        layer_id=0,
        expert_id=0,
        is_shared=False,
        matrix_name=MATRIX_GATE,
        logical_rows=7168,
        logical_cols=2048,
        depth=1,
    )

    up_0 = LogicalWeightCube(
        cube_id=1,
        layer_id=0,
        expert_id=0,
        is_shared=False,
        matrix_name=MATRIX_UP,
        logical_rows=7168,
        logical_cols=2048,
        depth=1,
    )

    down_0 = LogicalWeightCube(
        cube_id=2,
        layer_id=0,
        expert_id=0,
        is_shared=False,
        matrix_name=MATRIX_DOWN,
        logical_rows=2048,
        logical_cols=7168,
        depth=1,
    )

    # Expert-1
    gate_1 = LogicalWeightCube(
        cube_id=3,
        layer_id=0,
        expert_id=1,
        is_shared=False,
        matrix_name=MATRIX_GATE,
        logical_rows=7168,
        logical_cols=2048,
        depth=1,
    )

    up_1 = LogicalWeightCube(
        cube_id=4,
        layer_id=0,
        expert_id=1,
        is_shared=False,
        matrix_name=MATRIX_UP,
        logical_rows=7168,
        logical_cols=2048,
        depth=1,
    )

    down_1 = LogicalWeightCube(
        cube_id=5,
        layer_id=0,
        expert_id=1,
        is_shared=False,
        matrix_name=MATRIX_DOWN,
        logical_rows=2048,
        logical_cols=7168,
        depth=1,
    )

    return (
        gate_0,
        up_0,
        down_0,
        gate_1,
        up_1,
        down_1,
    )


# ============================================================
# Logical Planes
# ============================================================


def build_pairing(
    cubes,
):

    (
        gate_0,
        up_0,
        down_0,
        gate_1,
        up_1,
        down_1,
    ) = cubes

    plane_0 = (
        create_gate_down_plane(
            logical_plane_id=0,
            gate=gate_0,
            down=down_0,
        )
    )

    plane_1 = (
        create_gate_down_plane(
            logical_plane_id=1,
            gate=gate_1,
            down=down_1,
        )
    )

    plane_2 = (
        create_up_up_plane(
            logical_plane_id=2,
            first_up=up_0,
            second_up=up_1,
        )
    )

    return PairingResult(
        planes=(
            plane_0,
            plane_1,
            plane_2,
        ),

        routed_up_pairs_by_layer=(
            (
                (0, 1),
            ),
        ),

        routed_up_coactivation_cost_by_layer=(
            0,
        ),

        shared_up_layer_pairs=(),
    )


# ============================================================
# Fake Sub-Cube Mapping
# ============================================================


def build_subcube_mapping():

    hardware = (
        ResolvedHardwareConfig(
            N=2,
            H=7168,
            W=4096,
            D=1,
        )
    )

    placements = (
        LogicalPlanePlacement(
            logical_plane_id=0,
            subcube_id=0,
            z=0,
        ),

        LogicalPlanePlacement(
            logical_plane_id=1,
            subcube_id=1,
            z=0,
        ),

        LogicalPlanePlacement(
            logical_plane_id=2,
            subcube_id=2,
            z=0,
        ),
    )

    return SubcubeMappingResult(
        hardware=hardware,

        placements=placements,

        subcube_plane_counts=(
            1,
            1,
            1,
            0,
        ),

        gate_down_subcube_by_layer=(
            (
                0,
                1,
            ),
        ),

        pre_weighted_load_by_layer=(
            (
                1,
                1,
                2,
                0,
            ),
        ),

        down_weighted_load_by_layer=(
            (
                1,
                1,
                0,
                0,
            ),
        ),

        pre_conflict_cost=0,

        down_conflict_cost=0,
    )


# ============================================================
# Physical Planes
# ============================================================


def build_physical_planes():

    planes = []

    next_slot_id = 0

    for plane_id in range(3):

        plane = create_empty_plane(
            plane_id=plane_id,
            H=7168,
            W=4096,
        )

        slot_a = PhysicalSlot(
            slot_id=next_slot_id,
            plane_id=plane_id,

            x=0,
            y=0,

            slot_rows=7168,
            slot_cols=2048,

            orientation_swapped=False,
        )

        next_slot_id += 1

        slot_b = PhysicalSlot(
            slot_id=next_slot_id,
            plane_id=plane_id,

            x=0,
            y=2048,

            slot_rows=7168,
            slot_cols=2048,

            orientation_swapped=False,
        )

        next_slot_id += 1

        plane.add_slot(
            slot_a
        )

        plane.add_slot(
            slot_b
        )

        planes.append(
            plane
        )

    return tuple(
        planes
    )


# ============================================================
# Test Helper
# ============================================================


def build_result():

    cubes = build_cubes()

    pairing = (
        build_pairing(
            cubes
        )
    )

    subcube_mapping = (
        build_subcube_mapping()
    )

    physical_planes = (
        build_physical_planes()
    )

    result = (
        bind_logical_mapping_to_physical_slots(
            cubes=cubes,

            pairing=pairing,

            subcube_mapping=(
                subcube_mapping
            ),

            physical_planes=(
                physical_planes
            ),
        )
    )

    return (
        cubes,
        pairing,
        subcube_mapping,
        physical_planes,
        result,
    )


# ============================================================
# Tests
# ============================================================


def test_total_cube_count():

    (
        _,
        _,
        _,
        _,
        result,
    ) = build_result()

    assert (
        result.cube_count
        == 6
    )


def test_total_plane_count():

    (
        _,
        _,
        _,
        _,
        result,
    ) = build_result()

    assert (
        result.physical_plane_count
        == 3
    )


def test_all_slots_unique():

    (
        _,
        _,
        _,
        _,
        result,
    ) = build_result()

    slot_ids = [
        placement.slot_id
        for placement
        in result.placements
    ]

    assert len(slot_ids) == 6

    assert (
        len(
            set(slot_ids)
        )
        == 6
    )


def test_gate_down_share_plane():

    (
        _,
        _,
        _,
        _,
        result,
    ) = build_result()

    gate_0 = (
        result
        .placement_of_cube(0)
    )

    down_0 = (
        result
        .placement_of_cube(2)
    )

    assert (
        gate_0.physical_plane_id
        == down_0.physical_plane_id
    )

    assert (
        gate_0.subcube_id
        == down_0.subcube_id
    )

    assert (
        gate_0.z
        == down_0.z
    )


def test_up_pair_share_plane():

    (
        _,
        _,
        _,
        _,
        result,
    ) = build_result()

    up_0 = (
        result
        .placement_of_cube(1)
    )

    up_1 = (
        result
        .placement_of_cube(4)
    )

    assert (
        up_0.physical_plane_id
        == up_1.physical_plane_id
    )

    assert (
        up_0.subcube_id
        == up_1.subcube_id
    )


def test_down_requires_rotation():

    (
        _,
        _,
        _,
        _,
        result,
    ) = build_result()

    down_0 = (
        result
        .placement_of_cube(2)
    )

    down_1 = (
        result
        .placement_of_cube(5)
    )

    assert (
        down_0.logical_cube_rotated
        is True
    )

    assert (
        down_1.logical_cube_rotated
        is True
    )


def test_gate_up_do_not_rotate():

    (
        _,
        _,
        _,
        _,
        result,
    ) = build_result()

    for cube_id in (
        0,
        1,
        3,
        4,
    ):

        placement = (
            result
            .placement_of_cube(
                cube_id
            )
        )

        assert (
            placement
            .logical_cube_rotated
            is False
        )


def test_rotation_count():

    (
        _,
        _,
        _,
        _,
        result,
    ) = build_result()

    # 两个 down 需要旋转
    assert (
        result.logical_rotation_count
        == 2
    )


def test_subcube_mapping_is_preserved():

    (
        _,
        _,
        _,
        _,
        result,
    ) = build_result()

    # LogicalPlane-0 -> SC0
    gate_0 = (
        result
        .placement_of_cube(0)
    )

    assert (
        gate_0.subcube_id
        == 0
    )

    # LogicalPlane-1 -> SC1
    gate_1 = (
        result
        .placement_of_cube(3)
    )

    assert (
        gate_1.subcube_id
        == 1
    )

    # LogicalPlane-2 -> SC2
    up_0 = (
        result
        .placement_of_cube(1)
    )

    assert (
        up_0.subcube_id
        == 2
    )