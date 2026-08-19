from config import (
    ExecutionRules,
    ModelConfig,
)

from mapping.logical_weight import (
    MATRIX_DOWN,
    MATRIX_GATE,
    MATRIX_UP,
)

from scheduling.runtime_index import (
    RuntimeExpertLocation,
    RuntimeIndex,
    RuntimeLayerIndex,
    RuntimeMatrixLocation,
)

from scheduling.token_scheduler import (
    schedule_token,
)

from scheduling.workload_evaluator import (
    fast_schedule_layer_cycles,
    fast_schedule_token_cycles,
    percentile_nearest_rank,
)


# ============================================================
# Helper
# ============================================================


def location(
    *,
    layer,
    cube,
    expert,
    shared,
    matrix,
    plane,
    slot,
    sc,
):

    return RuntimeMatrixLocation(
        cube_id=cube,

        layer_id=layer,

        expert_id=expert,

        is_shared=shared,

        matrix_name=matrix,

        logical_plane_id=plane,

        physical_plane_id=plane,

        slot_id=slot,

        subcube_id=sc,

        z=layer,
    )


def expert(
    *,
    layer,
    expert_id,
    shared,
    cube_base,
    plane_base,
    gate_sc,
    up_sc,
):

    gate = location(
        layer=layer,
        cube=cube_base,
        expert=expert_id,
        shared=shared,
        matrix=MATRIX_GATE,
        plane=plane_base,
        slot=cube_base,
        sc=gate_sc,
    )

    up = location(
        layer=layer,
        cube=cube_base + 1,
        expert=expert_id,
        shared=shared,
        matrix=MATRIX_UP,
        plane=plane_base + 1,
        slot=cube_base + 1,
        sc=up_sc,
    )

    down = location(
        layer=layer,
        cube=cube_base + 2,
        expert=expert_id,
        shared=shared,
        matrix=MATRIX_DOWN,
        plane=plane_base,
        slot=cube_base + 2,
        sc=gate_sc,
    )

    return RuntimeExpertLocation(
        layer_id=layer,

        expert_id=expert_id,

        is_shared=shared,

        gate=gate,
        up=up,
        down=down,
    )


# ============================================================
# 两层、有冲突的测试映射
# ============================================================


def build_index():

    config = ModelConfig(
        num_moe_layers=2,

        routed_experts_per_layer=2,

        experts_per_token=2,

        include_shared_expert=True,
    )

    # ========================================================
    # Layer 0
    #
    # E0 gate -> SC0, up -> SC1
    # E1 gate -> SC0, up -> SC2
    # Shared gate -> SC3, up -> SC1
    #
    # 故意制造：
    #
    # SC0 有两个 gate
    # SC1 有两个 up
    # ========================================================

    layer_0 = RuntimeLayerIndex(
        layer_id=0,

        experts=(
            expert(
                layer=0,
                expert_id=0,
                shared=False,
                cube_base=0,
                plane_base=0,
                gate_sc=0,
                up_sc=1,
            ),

            expert(
                layer=0,
                expert_id=1,
                shared=False,
                cube_base=3,
                plane_base=2,
                gate_sc=0,
                up_sc=2,
            ),

            expert(
                layer=0,
                expert_id=2,
                shared=True,
                cube_base=6,
                plane_base=4,
                gate_sc=3,
                up_sc=1,
            ),
        ),
    )

    # ========================================================
    # Layer 1
    # ========================================================

    layer_1 = RuntimeLayerIndex(
        layer_id=1,

        experts=(
            expert(
                layer=1,
                expert_id=0,
                shared=False,
                cube_base=9,
                plane_base=6,
                gate_sc=1,
                up_sc=0,
            ),

            expert(
                layer=1,
                expert_id=1,
                shared=False,
                cube_base=12,
                plane_base=8,
                gate_sc=1,
                up_sc=2,
            ),

            expert(
                layer=1,
                expert_id=2,
                shared=True,
                cube_base=15,
                plane_base=10,
                gate_sc=3,
                up_sc=0,
            ),
        ),
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


def test_fast_layer_matches_exact():

    index = build_index()

    route = (
        0,
        1,
    )

    exact = schedule_token(
        index=index,

        routed_experts_by_layer=(
            route,
            route,
        ),

        rules=ExecutionRules(),
    )

    fast_l0 = (
        fast_schedule_layer_cycles(
            index=index,

            layer_id=0,

            routed_expert_ids=route,
        )
    )

    assert (
        fast_l0
        ==
        exact
        .layer(0)
        .layer_result
        .total_cycles
    )


def test_fast_token_matches_exact():

    index = build_index()

    routes = (
        (0, 1),
        (0, 1),
    )

    exact = schedule_token(
        index=index,

        routed_experts_by_layer=(
            routes
        ),
    )

    fast = (
        fast_schedule_token_cycles(
            index=index,

            routed_experts_by_layer=(
                routes
            ),
        )
    )

    assert (
        fast
        == exact.total_cycles
    )


def test_route_order_change_still_matches():

    index = build_index()

    routes = (
        (1, 0),
        (0, 1),
    )

    exact = schedule_token(
        index=index,

        routed_experts_by_layer=(
            routes
        ),
    )

    fast = (
        fast_schedule_token_cycles(
            index=index,

            routed_experts_by_layer=(
                routes
            ),
        )
    )

    assert (
        fast
        == exact.total_cycles
    )


def test_percentile():

    values = (
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    )

    assert (
        percentile_nearest_rank(
            values,
            0.50,
        )
        == 50
    )

    assert (
        percentile_nearest_rank(
            values,
            0.95,
        )
        == 100
    )