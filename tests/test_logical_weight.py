from collections import Counter

from config import ModelConfig

from mapping.logical_weight import (
    MATRIX_DOWN,
    MATRIX_GATE,
    MATRIX_UP,
    build_logical_weight_cubes,
)


def build_test_config():
    return ModelConfig(
        include_shared_expert=True,
    )


def test_total_cube_count():

    config = build_test_config()

    cubes = build_logical_weight_cubes(
        config
    )

    assert len(cubes) == (
        58 * 257 * 3
    )

    assert len(cubes) == 44718


def test_each_matrix_count():

    config = build_test_config()

    cubes = build_logical_weight_cubes(
        config
    )

    counts = Counter(
        cube.matrix_name
        for cube in cubes
    )

    assert counts[MATRIX_GATE] == (
        58 * 257
    )

    assert counts[MATRIX_UP] == (
        58 * 257
    )

    assert counts[MATRIX_DOWN] == (
        58 * 257
    )

    assert counts[MATRIX_GATE] == 14906
    assert counts[MATRIX_UP] == 14906
    assert counts[MATRIX_DOWN] == 14906


def test_layer_zero_expert_count():

    config = build_test_config()

    cubes = build_logical_weight_cubes(
        config
    )

    expert_ids = {
        cube.expert_id
        for cube in cubes
        if cube.layer_id == 0
    }

    assert len(expert_ids) == 257

    assert min(expert_ids) == 0
    assert max(expert_ids) == 256


def test_shared_expert_id():

    config = build_test_config()

    cubes = build_logical_weight_cubes(
        config
    )

    shared = [
        cube
        for cube in cubes
        if cube.is_shared
    ]

    # 每层 1 个 Shared Expert × 3 矩阵
    assert len(shared) == 58 * 3

    assert all(
        cube.expert_id == 256
        for cube in shared
    )


def test_each_expert_has_three_matrices():

    config = build_test_config()

    cubes = build_logical_weight_cubes(
        config
    )

    layer = 0
    expert = 10

    target = [
        cube
        for cube in cubes
        if (
            cube.layer_id == layer
            and cube.expert_id == expert
        )
    ]

    assert len(target) == 3

    names = {
        cube.matrix_name
        for cube in target
    }

    assert names == {
        MATRIX_GATE,
        MATRIX_UP,
        MATRIX_DOWN,
    }


def test_matrix_shapes():

    config = build_test_config()

    cubes = build_logical_weight_cubes(
        config
    )

    for cube in cubes:

        if cube.matrix_name in (
            MATRIX_GATE,
            MATRIX_UP,
        ):
            assert (
                cube.logical_rows,
                cube.logical_cols,
            ) == (
                7168,
                2048,
            )

        elif cube.matrix_name == MATRIX_DOWN:
            assert (
                cube.logical_rows,
                cube.logical_cols,
            ) == (
                2048,
                7168,
            )


def test_all_size_keys_equal():

    config = build_test_config()

    cubes = build_logical_weight_cubes(
        config
    )

    size_keys = {
        cube.size_key
        for cube in cubes
    }

    assert size_keys == {
        (2048, 7168)
    }


def test_depth_is_one():

    config = build_test_config()

    cubes = build_logical_weight_cubes(
        config
    )

    assert all(
        cube.depth == 1
        for cube in cubes
    )


def test_cube_ids_continuous():

    config = build_test_config()

    cubes = build_logical_weight_cubes(
        config
    )

    assert [
        cube.cube_id
        for cube in cubes
    ] == list(
        range(44718)
    )