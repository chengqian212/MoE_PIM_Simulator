from config import ModelConfig

from mapping.logical_weight import (
    build_logical_weight_cubes,
)

from mapping.logical_plane import (
    PAIR_GATE_DOWN,
    PAIR_UP_UP,
    create_gate_down_plane,
    create_up_up_plane,
    validate_logical_plane,
    build_cube_index,
)


def build_cubes():

    config = ModelConfig(
        include_shared_expert=True,
    )

    return build_logical_weight_cubes(
        config
    )


def test_create_gate_down_plane():

    cubes = build_cubes()

    gate = cubes[0]
    up = cubes[1]
    down = cubes[2]

    plane = create_gate_down_plane(
        logical_plane_id=0,
        gate=gate,
        down=down,
    )

    assert (
        plane.pair_kind
        == PAIR_GATE_DOWN
    )

    assert plane.cube_a_id == gate.cube_id
    assert plane.cube_b_id == down.cube_id

    cube_index = build_cube_index(
        cubes
    )

    validate_logical_plane(
        plane=plane,
        cube_index=cube_index,
    )


def test_create_up_up_plane():

    cubes = build_cubes()

    # Layer-0 Expert-0 up
    up_0 = cubes[1]

    # Layer-0 Expert-1：
    # gate=3, up=4, down=5
    up_1 = cubes[4]

    plane = create_up_up_plane(
        logical_plane_id=0,
        first_up=up_0,
        second_up=up_1,
    )

    assert (
        plane.pair_kind
        == PAIR_UP_UP
    )

    assert (
        plane.cube_a_id
        == up_0.cube_id
    )

    assert (
        plane.cube_b_id
        == up_1.cube_id
    )