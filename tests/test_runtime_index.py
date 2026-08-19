import copy

import pytest

from config import (
    ModelConfig,
)

from mapping.logical_weight import (
    MATRIX_DOWN,
    MATRIX_GATE,
    MATRIX_UP,
)

from scheduling.runtime_index import (
    RuntimeIndexError,
    build_runtime_index,
)


def build_test_mapping():
    """
    小型模型：

        2 Layer

    每层：

        2 Routed Expert
        1 Shared Expert

    每 Expert：

        gate/up/down

    共：

        2 × 3 × 3
        = 18 WeightCube

    Plane：

        gate/down:
            2 × 3 = 6

        routed up-up:
            2

        shared up-up:
            1

        共 9 Plane。
    """

    placements = []

    cube_id = 0

    # ========================================================
    # Helper
    # ========================================================

    def add(
        *,
        layer,
        expert,
        shared,
        matrix,
        plane,
        sc,
        z,
        slot,
    ):

        nonlocal cube_id

        placements.append(
            {
                "cube_id": cube_id,

                "layer_id": layer,

                "expert_id": expert,

                "is_shared": shared,

                "matrix_name": matrix,

                "logical_plane_id": (
                    plane
                ),

                "physical_plane_id": (
                    plane
                ),

                "slot_id": slot,

                "subcube_id": sc,

                "z": z,
            }
        )

        cube_id += 1

    # ========================================================
    # gate/down
    # ========================================================

    gate_data = {
        # layer, expert:
        #     plane, sc, z

        (0, 0): (0, 0, 0),
        (0, 1): (1, 1, 0),
        (0, 2): (2, 2, 0),

        (1, 0): (3, 1, 1),
        (1, 1): (4, 2, 1),
        (1, 2): (5, 3, 0),
    }

    slot_id = 0

    for (
        layer,
        expert,
    ), (
        plane,
        sc,
        z,
    ) in gate_data.items():

        shared = (
            expert == 2
        )

        add(
            layer=layer,
            expert=expert,
            shared=shared,
            matrix=MATRIX_GATE,
            plane=plane,
            sc=sc,
            z=z,
            slot=slot_id,
        )

        slot_id += 1

        add(
            layer=layer,
            expert=expert,
            shared=shared,
            matrix=MATRIX_DOWN,
            plane=plane,
            sc=sc,
            z=z,
            slot=slot_id,
        )

        slot_id += 1

    # ========================================================
    # Layer-0 Routed up：
    #
    # E0 + E1 -> Plane 6 -> SC3
    # ========================================================

    for expert in (
        0,
        1,
    ):

        add(
            layer=0,
            expert=expert,
            shared=False,
            matrix=MATRIX_UP,
            plane=6,
            sc=3,
            z=1,
            slot=slot_id,
        )

        slot_id += 1

    # ========================================================
    # Layer-1 Routed up：
    #
    # E0 + E1 -> Plane 7 -> SC0
    # ========================================================

    for expert in (
        0,
        1,
    ):

        add(
            layer=1,
            expert=expert,
            shared=False,
            matrix=MATRIX_UP,
            plane=7,
            sc=0,
            z=1,
            slot=slot_id,
        )

        slot_id += 1

    # ========================================================
    # Shared up：
    #
    # L0 E2 + L1 E2
    # -> Plane 8
    # -> SC0
    # ========================================================

    for layer in (
        0,
        1,
    ):

        add(
            layer=layer,
            expert=2,
            shared=True,
            matrix=MATRIX_UP,
            plane=8,
            sc=0,
            z=2,
            slot=slot_id,
        )

        slot_id += 1

    return {
        "mapping_version": 1,

        "model": {
            "num_logical_weight_cubes": (
                18
            ),

            "num_logical_planes": (
                9
            ),

            "shared_expert_enabled": (
                True
            ),
        },

        "spatial": {
            "num_subcubes": 4,
            "D": 3,

            "P": 9,
            "Q": 12,
        },

        "placements": (
            placements
        ),
    }


def build_test_config():

    return ModelConfig(
        num_moe_layers=2,

        routed_experts_per_layer=2,

        # Router 每次选 1 个，
        # 方便测试 active expert。
        experts_per_token=1,

        include_shared_expert=True,
    )


def test_build_runtime_index():

    index = (
        build_runtime_index(
            build_test_mapping(),

            model_config=(
                build_test_config()
            ),
        )
    )

    assert (
        index.num_layers
        == 2
    )

    assert (
        index.experts_per_layer
        == 3
    )

    assert (
        index.shared_expert_id
        == 2
    )

    assert (
        index.total_experts
        == 6
    )

    assert (
        index.total_matrices
        == 18
    )


def test_matrix_lookup():

    index = (
        build_runtime_index(
            build_test_mapping(),

            model_config=(
                build_test_config()
            ),
        )
    )

    expert = (
        index.expert(
            0,
            0,
        )
    )

    assert (
        expert.gate_subcube
        == 0
    )

    assert (
        expert.up_subcube
        == 3
    )

    assert (
        expert.down_subcube
        == 0
    )

    assert (
        index.matrix(
            0,
            0,
            MATRIX_GATE,
        ).cube_id
        == expert.gate.cube_id
    )


def test_gate_down_share_location():

    index = (
        build_runtime_index(
            build_test_mapping(),

            model_config=(
                build_test_config()
            ),
        )
    )

    for layer_id in range(2):

        for expert_id in range(3):

            expert = (
                index.expert(
                    layer_id,
                    expert_id,
                )
            )

            assert (
                expert.gate.subcube_id
                == expert.down.subcube_id
            )

            assert (
                expert.gate.z
                == expert.down.z
            )

            assert (
                expert.gate.physical_plane_id
                ==
                expert.down.physical_plane_id
            )


def test_gate_up_are_separated():

    index = (
        build_runtime_index(
            build_test_mapping(),

            model_config=(
                build_test_config()
            ),
        )
    )

    for layer_id in range(2):

        for expert_id in range(3):

            expert = (
                index.expert(
                    layer_id,
                    expert_id,
                )
            )

            assert (
                expert.gate_subcube
                != expert.up_subcube
            )


def test_resolve_active_experts_adds_shared():

    index = (
        build_runtime_index(
            build_test_mapping(),

            model_config=(
                build_test_config()
            ),
        )
    )

    active_ids = (
        index.resolve_active_expert_ids(
            layer_id=0,

            routed_expert_ids=(
                1,
            ),
        )
    )

    # Routed Expert-1
    # +
    # Shared Expert-2
    assert (
        active_ids
        == (
            1,
            2,
        )
    )


def test_wrong_top_k_rejected():

    index = (
        build_runtime_index(
            build_test_mapping(),

            model_config=(
                build_test_config()
            ),
        )
    )

    with pytest.raises(
        RuntimeIndexError
    ):

        index.resolve_active_expert_ids(
            layer_id=0,

            routed_expert_ids=(
                0,
                1,
            ),
        )


def test_invalid_gate_up_colocation_rejected():

    data = (
        build_test_mapping()
    )

    broken = copy.deepcopy(
        data
    )

    # Layer-0 E0 up 原本在 SC3。
    #
    # 改成它的 gate 所在 SC0。
    #
    # 同时为了避免先触发
    # Plane 内两个 up 不共址，
    # 将 Plane-6 两个 up 都改成 SC0。
    for record in (
        broken["placements"]
    ):

        if (
            record[
                "logical_plane_id"
            ]
            == 6
        ):

            record[
                "subcube_id"
            ] = 0

    with pytest.raises(
        RuntimeIndexError
    ):

        build_runtime_index(
            broken,

            model_config=(
                build_test_config()
            ),
        )