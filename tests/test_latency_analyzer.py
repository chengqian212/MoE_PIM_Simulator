from dataclasses import dataclass


from config import (
    ModelConfig,
)

from mapping.logical_weight import (
    MATRIX_DOWN,
    MATRIX_GATE,
    MATRIX_UP,
)

from scheduling.latency_analyzer import (
    analyze_tokens,
    calculate_layer_collision_score,
    pearson_correlation,
)

from scheduling.runtime_index import (
    RuntimeExpertLocation,
    RuntimeIndex,
    RuntimeLayerIndex,
    RuntimeMatrixLocation,
)


# ============================================================
# Fake Token
# ============================================================


@dataclass(
    frozen=True,
)
class FakeToken:

    token_id: int

    routed_experts_by_layer: tuple

    category: str = "test"

    relative_file: str = "test.json"

    segment_index: int = 0

    token_index_in_segment: int = 0


# ============================================================
# Location
# ============================================================


def loc(
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


# ============================================================
# Expert
# ============================================================


def make_expert(
    *,
    layer,
    expert_id,
    shared,
    cube_base,
    plane_base,
    gate_sc,
    up_sc,
):

    return RuntimeExpertLocation(
        layer_id=layer,

        expert_id=expert_id,

        is_shared=shared,

        gate=loc(
            layer=layer,
            cube=cube_base,
            expert=expert_id,
            shared=shared,
            matrix=MATRIX_GATE,
            plane=plane_base,
            slot=cube_base,
            sc=gate_sc,
        ),

        up=loc(
            layer=layer,
            cube=cube_base + 1,
            expert=expert_id,
            shared=shared,
            matrix=MATRIX_UP,
            plane=plane_base + 1,
            slot=cube_base + 1,
            sc=up_sc,
        ),

        down=loc(
            layer=layer,
            cube=cube_base + 2,
            expert=expert_id,
            shared=shared,
            matrix=MATRIX_DOWN,
            plane=plane_base,
            slot=cube_base + 2,
            sc=gate_sc,
        ),
    )


# ============================================================
# 两层 Index
# ============================================================


def build_index():

    config = ModelConfig(
        num_moe_layers=2,

        routed_experts_per_layer=2,

        experts_per_token=1,

        include_shared_expert=True,
    )

    # Layer-0：
    #
    # Routed E0 / E1 gate 都在 SC0，
    # Shared gate SC2。
    #
    # Route 选哪个会影响冲突。
    layer_0 = RuntimeLayerIndex(
        layer_id=0,

        experts=(
            make_expert(
                layer=0,
                expert_id=0,
                shared=False,
                cube_base=0,
                plane_base=0,
                gate_sc=0,
                up_sc=1,
            ),

            make_expert(
                layer=0,
                expert_id=1,
                shared=False,
                cube_base=3,
                plane_base=2,
                gate_sc=0,
                up_sc=3,
            ),

            make_expert(
                layer=0,
                expert_id=2,
                shared=True,
                cube_base=6,
                plane_base=4,
                gate_sc=2,
                up_sc=1,
            ),
        ),
    )

    layer_1 = RuntimeLayerIndex(
        layer_id=1,

        experts=(
            make_expert(
                layer=1,
                expert_id=0,
                shared=False,
                cube_base=9,
                plane_base=6,
                gate_sc=0,
                up_sc=1,
            ),

            make_expert(
                layer=1,
                expert_id=1,
                shared=False,
                cube_base=12,
                plane_base=8,
                gate_sc=1,
                up_sc=2,
            ),

            make_expert(
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


def test_collision_score_non_negative():

    index = build_index()

    score = (
        calculate_layer_collision_score(
            index=index,

            layer_id=0,

            routed_expert_ids=(
                0,
            ),
        )
    )

    assert score >= 0


def test_pearson_perfect_positive():

    result = pearson_correlation(
        (
            1,
            2,
            3,
            4,
        ),

        (
            10,
            20,
            30,
            40,
        ),
    )

    assert result is not None

    assert abs(
        result - 1.0
    ) < 1e-9


def test_analyze_tokens():

    index = build_index()

    tokens = (
        FakeToken(
            token_id=0,

            routed_experts_by_layer=(
                (0,),
                (0,),
            ),
        ),

        FakeToken(
            token_id=1,

            routed_experts_by_layer=(
                (1,),
                (1,),
            ),

            token_index_in_segment=1,
        ),

        FakeToken(
            token_id=2,

            routed_experts_by_layer=(
                (0,),
                (1,),
            ),

            token_index_in_segment=2,
        ),
    )

    result = (
        analyze_tokens(
            index=index,

            tokens=tokens,

            exact_tokens=3,

            top_k_slow_tokens=2,

            verbose=False,
        )
    )

    assert (
        result.token_count
        == 3
    )

    assert (
        result.exact_token_count
        == 3
    )

    assert (
        len(
            result.layer_stats
        )
        == 2
    )

    assert (
        len(
            result.subcube_stats
        )
        == 4
    )

    assert (
        len(
            result.slow_tokens
        )
        == 2
    )

    assert (
        result.mean_token_cycles
        > 0
    )


def test_slow_tokens_are_descending():

    index = build_index()

    tokens = tuple(
        FakeToken(
            token_id=i,

            routed_experts_by_layer=(
                (
                    i % 2,
                ),
                (
                    (i + 1) % 2,
                ),
            ),

            token_index_in_segment=i,
        )

        for i in range(6)
    )

    result = (
        analyze_tokens(
            index=index,

            tokens=tokens,

            exact_tokens=0,

            top_k_slow_tokens=3,

            verbose=False,
        )
    )

    cycles = [
        token.total_cycles
        for token
        in result.slow_tokens
    ]

    assert (
        cycles
        == sorted(
            cycles,
            reverse=True,
        )
    )