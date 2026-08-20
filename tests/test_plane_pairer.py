from array import array

from config import ModelConfig

from mapping.logical_weight import (
    build_logical_weight_cubes,
)

from mapping.logical_plane import (
    build_cube_index,
    get_plane_cubes,
)

from mapping.plane_pairer import (
    PAIRING_MODE_SEQUENTIAL,
    build_logical_planes,
    sequential_pair_routed_up_experts,
)

from mapping.trace_profile import (
    NUM_MOE_LAYERS,
    NUM_ROUTED_EXPERTS,
    TraceProfile,
)


def build_fake_profile():
    """
    构造一个最简单 TraceProfile。

    所有 Expert：

        frequency 相同

    所有 Expert pair：

        coactivation = 0

    只用于检查 Plane 数量和覆盖关系，
    不测试真实 trace 优化效果。
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
        token_count_by_layer=token_count,
    )


def build_cubes():

    config = ModelConfig(
        include_shared_expert=True,
    )

    return build_logical_weight_cubes(
        config
    )


def test_total_plane_count():

    cubes = build_cubes()

    profile = (
        build_fake_profile()
    )

    result = build_logical_planes(
        cubes=cubes,
        profile=profile,
        improve_pairs=False,
    )

    assert (
        result.total_planes
        == 22359
    )


def test_gate_down_plane_count():

    cubes = build_cubes()

    profile = (
        build_fake_profile()
    )

    result = build_logical_planes(
        cubes=cubes,
        profile=profile,
        improve_pairs=False,
    )

    assert (
        result.gate_down_plane_count
        == 58 * 257
    )

    assert (
        result.gate_down_plane_count
        == 14906
    )


def test_routed_up_plane_count():

    cubes = build_cubes()

    profile = (
        build_fake_profile()
    )

    result = build_logical_planes(
        cubes=cubes,
        profile=profile,
        improve_pairs=False,
    )

    assert (
        result.routed_up_plane_count
        == 58 * 128
    )

    assert (
        result.routed_up_plane_count
        == 7424
    )


def test_shared_up_plane_count():

    cubes = build_cubes()

    profile = (
        build_fake_profile()
    )

    result = build_logical_planes(
        cubes=cubes,
        profile=profile,
        improve_pairs=False,
    )

    assert (
        result.shared_up_plane_count
        == 29
    )


def test_all_up_plane_count():

    cubes = build_cubes()

    profile = (
        build_fake_profile()
    )

    result = build_logical_planes(
        cubes=cubes,
        profile=profile,
        improve_pairs=False,
    )

    assert (
        result.up_up_plane_count
        == 7424 + 29
    )

    assert (
        result.up_up_plane_count
        == 7453
    )


def test_all_cubes_used_once():

    cubes = build_cubes()

    profile = (
        build_fake_profile()
    )

    result = build_logical_planes(
        cubes=cubes,
        profile=profile,
        improve_pairs=False,
    )

    used = []

    for plane in result.planes:

        used.extend(
            plane.cube_ids
        )

    assert len(used) == 44718

    assert (
        len(set(used))
        == 44718
    )

    assert set(used) == {
        cube.cube_id
        for cube in cubes
    }


def test_gate_down_same_expert():

    cubes = build_cubes()

    cube_index = (
        build_cube_index(
            cubes
        )
    )

    profile = (
        build_fake_profile()
    )

    result = build_logical_planes(
        cubes=cubes,
        profile=profile,
        improve_pairs=False,
    )

    for plane in result.planes:

        if not plane.is_gate_down:
            continue

        cube_a, cube_b = (
            get_plane_cubes(
                plane=plane,
                cube_index=cube_index,
            )
        )

        assert (
            cube_a.layer_id
            == cube_b.layer_id
        )

        assert (
            cube_a.expert_id
            == cube_b.expert_id
        )


def test_each_layer_has_128_routed_up_pairs():

    cubes = build_cubes()

    profile = (
        build_fake_profile()
    )

    result = build_logical_planes(
        cubes=cubes,
        profile=profile,
        improve_pairs=False,
    )

    assert (
        len(
            result
            .routed_up_pairs_by_layer
        )
        == 58
    )

    for layer_pairs in (
        result
        .routed_up_pairs_by_layer
    ):

        assert (
            len(layer_pairs)
            == 128
        )

        experts = [
            expert_id
            for pair in layer_pairs
            for expert_id in pair
        ]

        assert len(experts) == 256

        assert set(experts) == set(
            range(256)
        )


def test_shared_up_layer_pairing():

    cubes = build_cubes()

    profile = (
        build_fake_profile()
    )

    result = build_logical_planes(
        cubes=cubes,
        profile=profile,
        improve_pairs=False,
    )

    assert (
        result.shared_up_layer_pairs[0]
        == (0, 1)
    )

    assert (
        result.shared_up_layer_pairs[1]
        == (2, 3)
    )

    assert (
        result.shared_up_layer_pairs[-1]
        == (56, 57)
    )

# ============================================================
# Sequential Pairing 消融模式
# ============================================================


def test_sequential_pair_function_exact_order():
    """
    Sequential Pairing 必须完全确定：

        E0-E1
        E2-E3
        ...
        E254-E255
    """

    pairs = (
        sequential_pair_routed_up_experts(
            layer_id=0,
        )
    )

    assert len(pairs) == 128

    assert pairs[0] == (0, 1)
    assert pairs[1] == (2, 3)
    assert pairs[2] == (4, 5)
    assert pairs[-1] == (254, 255)

    experts = [
        expert_id
        for pair in pairs
        for expert_id in pair
    ]

    assert experts == list(
        range(256)
    )


def test_sequential_mode_all_layers_exact_order():
    """
    完整 build_logical_planes() 在 sequential 模式下，
    58 层都必须采用相同的固定编号配对。
    """

    cubes = build_cubes()
    profile = build_fake_profile()

    result = build_logical_planes(
        cubes=cubes,
        profile=profile,
        pairing_mode=(
            PAIRING_MODE_SEQUENTIAL
        ),
        # 即使传 True，sequential 也不应执行 Local Search。
        improve_pairs=True,
        local_search_rounds=4,
    )

    expected_pairs = tuple(
        (expert_id, expert_id + 1)
        for expert_id
        in range(0, 256, 2)
    )

    assert (
        len(result.routed_up_pairs_by_layer)
        == 58
    )

    for layer_pairs in (
        result.routed_up_pairs_by_layer
    ):
        assert layer_pairs == expected_pairs


def test_sequential_mode_plane_counts_unchanged():
    """
    Pairing 策略变化不能改变空间结构数量。
    """

    cubes = build_cubes()
    profile = build_fake_profile()

    result = build_logical_planes(
        cubes=cubes,
        profile=profile,
        pairing_mode=(
            PAIRING_MODE_SEQUENTIAL
        ),
    )

    assert result.total_planes == 22359
    assert result.gate_down_plane_count == 14906
    assert result.routed_up_plane_count == 7424
    assert result.shared_up_plane_count == 29

    used = []

    for plane in result.planes:
        used.extend(plane.cube_ids)

    assert len(used) == 44718
    assert len(set(used)) == 44718
