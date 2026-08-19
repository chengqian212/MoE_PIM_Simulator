import pytest


from config import (
    ModelConfig,
)

from mapping.logical_weight import (
    MATRIX_DOWN,
    MATRIX_GATE,
    MATRIX_UP,
)

from scheduling.token_scheduler import (
    TokenSchedulerError,
    schedule_token,
)

from scheduling.runtime_index import (
    RuntimeExpertLocation,
    RuntimeIndex,
    RuntimeLayerIndex,
    RuntimeMatrixLocation,
)


# ============================================================
# Location
# ============================================================


def make_location(
    *,
    layer_id,
    cube_id,
    expert_id,
    shared,
    matrix_name,
    plane_id,
    slot_id,
    sc,
    z,
):

    return RuntimeMatrixLocation(
        cube_id=cube_id,

        layer_id=layer_id,

        expert_id=expert_id,

        is_shared=shared,

        matrix_name=matrix_name,

        logical_plane_id=(
            plane_id
        ),

        physical_plane_id=(
            plane_id
        ),

        slot_id=slot_id,

        subcube_id=sc,

        z=z,
    )


# ============================================================
# Expert
# ============================================================


def make_expert(
    *,
    layer_id,
    expert_id,
    shared,
    cube_base,
    plane_base,
):
    """
    当前每层理想布局：

    Routed / Shared 都是：

        gate/down -> 一个 SC
        up        -> 另一个 SC

    Layer-0：
        Routed:
            gate/down SC0
            up        SC1

        Shared:
            gate/down SC2
            up        SC3

    Layer-1 使用同样 SC，
    但 WeightCube ID 不同。

    因此：
        Layer-1 会继承 Layer-0 active state，
        然后发生正常 WC switch。
    """

    if expert_id == 0:

        gate_sc = 0
        up_sc = 1

    else:

        gate_sc = 2
        up_sc = 3

    gate = make_location(
        layer_id=layer_id,

        cube_id=(
            cube_base
        ),

        expert_id=expert_id,

        shared=shared,

        matrix_name=(
            MATRIX_GATE
        ),

        plane_id=(
            plane_base
        ),

        slot_id=(
            cube_base
        ),

        sc=gate_sc,

        z=layer_id,
    )

    up = make_location(
        layer_id=layer_id,

        cube_id=(
            cube_base + 1
        ),

        expert_id=expert_id,

        shared=shared,

        matrix_name=(
            MATRIX_UP
        ),

        plane_id=(
            plane_base + 1
        ),

        slot_id=(
            cube_base + 1
        ),

        sc=up_sc,

        z=layer_id,
    )

    down = make_location(
        layer_id=layer_id,

        cube_id=(
            cube_base + 2
        ),

        expert_id=expert_id,

        shared=shared,

        matrix_name=(
            MATRIX_DOWN
        ),

        # gate/down 必须同 Plane
        plane_id=(
            plane_base
        ),

        slot_id=(
            cube_base + 2
        ),

        sc=gate_sc,

        z=layer_id,
    )

    return RuntimeExpertLocation(
        layer_id=layer_id,

        expert_id=expert_id,

        is_shared=shared,

        gate=gate,
        up=up,
        down=down,
    )


# ============================================================
# 两层测试 Index
# ============================================================


def build_test_index():

    config = (
        ModelConfig(
            num_moe_layers=2,

            routed_experts_per_layer=1,

            experts_per_token=1,

            include_shared_expert=True,
        )
    )

    # ========================================================
    # Layer 0
    # ========================================================

    layer_0 = (
        RuntimeLayerIndex(
            layer_id=0,

            experts=(
                make_expert(
                    layer_id=0,

                    expert_id=0,
                    shared=False,

                    cube_base=0,
                    plane_base=0,
                ),

                make_expert(
                    layer_id=0,

                    expert_id=1,
                    shared=True,

                    cube_base=3,
                    plane_base=2,
                ),
            ),
        )
    )

    # ========================================================
    # Layer 1
    # ========================================================

    layer_1 = (
        RuntimeLayerIndex(
            layer_id=1,

            experts=(
                make_expert(
                    layer_id=1,

                    expert_id=0,
                    shared=False,

                    cube_base=6,
                    plane_base=4,
                ),

                make_expert(
                    layer_id=1,

                    expert_id=1,
                    shared=True,

                    cube_base=9,
                    plane_base=6,
                ),
            ),
        )
    )

    return RuntimeIndex(
        model_config=config,

        num_subcubes=4,

        subcube_depth=10,

        layers=(
            layer_0,
            layer_1,
        ),
    )


# ============================================================
# Tests
# ============================================================


def test_two_layer_token_cycles():

    index = (
        build_test_index()
    )

    result = (
        schedule_token(
            index=index,

            routed_experts_by_layer=(
                (0,),
                (0,),
            ),
        )
    )

    # 每层：
    #
    # gate/up parallel：
    #     switch/activation 1
    #     compute 1
    #     => 2
    #
    # down：
    #     switch 1
    #     compute 1
    #     => 2
    #
    # 每层 4 cycle
    #
    # 两层严格串行：
    #
    # 4 + 4 = 8
    assert (
        result.total_cycles
        == 8
    )


def test_layer_timeline_is_sequential():

    index = (
        build_test_index()
    )

    result = (
        schedule_token(
            index=index,

            routed_experts_by_layer=(
                (0,),
                (0,),
            ),
        )
    )

    layer_0 = (
        result.layer(0)
    )

    layer_1 = (
        result.layer(1)
    )

    assert (
        layer_0.global_start_time
        == 0
    )

    assert (
        layer_0.global_finish_time
        == 4
    )

    assert (
        layer_1.global_start_time
        == 4
    )

    assert (
        layer_1.global_finish_time
        == 8
    )


def test_active_state_is_propagated():

    index = (
        build_test_index()
    )

    result = (
        schedule_token(
            index=index,

            routed_experts_by_layer=(
                (0,),
                (0,),
            ),
        )
    )

    layer_0 = (
        result.layer(0)
        .layer_result
    )

    layer_1 = (
        result.layer(1)
        .layer_result
    )

    assert (
        layer_1
        .initial_active_cube_by_subcube
        ==
        layer_0
        .final_active_cube_by_subcube
    )


def test_task_count():

    index = (
        build_test_index()
    )

    result = (
        schedule_token(
            index=index,

            routed_experts_by_layer=(
                (0,),
                (0,),
            ),
        )
    )

    # 每层：
    #
    # 1 Routed
    # +
    # 1 Shared
    #
    # × 3 matrices
    # = 6
    #
    # 两层：
    # = 12
    assert (
        result.total_tasks
        == 12
    )


def test_initial_activation_only_first_use():

    index = (
        build_test_index()
    )

    result = (
        schedule_token(
            index=index,

            routed_experts_by_layer=(
                (0,),
                (0,),
            ),
        )
    )

    layer_0 = (
        result.layer(0)
        .layer_result
    )

    layer_1 = (
        result.layer(1)
        .layer_result
    )

    # Layer-0：
    #
    # 4 个 SC 首次使用
    assert (
        layer_0
        .initial_activation_count
        == 4
    )

    # Layer-1：
    #
    # 4 个 SC 都已经有 active WC，
    # 所以不会再算 initial activation，
    # 而是 WeightCube switch。
    assert (
        layer_1
        .initial_activation_count
        == 0
    )


def test_second_layer_inherits_and_switches():

    index = (
        build_test_index()
    )

    result = (
        schedule_token(
            index=index,

            routed_experts_by_layer=(
                (0,),
                (0,),
            ),
        )
    )

    layer_1 = (
        result.layer(1)
        .layer_result
    )

    # Layer-1：
    #
    # 4 个 gate/up 都需要从上一层
    # 留下的 WeightCube 切换过去。
    #
    # 然后两个 gate -> down
    # 还各再切一次。
    #
    # 共 6 switches。
    assert (
        layer_1.switch_count
        == 6
    )


def test_wrong_layer_count_rejected():

    index = (
        build_test_index()
    )

    with pytest.raises(
        TokenSchedulerError
    ):

        schedule_token(
            index=index,

            routed_experts_by_layer=(
                (0,),
            ),
        )


def test_wrong_route_rejected():

    index = (
        build_test_index()
    )

    with pytest.raises(
        Exception
    ):

        schedule_token(
            index=index,

            routed_experts_by_layer=(
                # 当前 Top-K=1
                (0, 0),

                (0,),
            ),
        )