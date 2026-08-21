from array import array

import pytest

from mapping.plane_pairer import (
    PAIRING_MODE_FREQUENCY_AWARE,
    PAIRING_MODE_GREEDY,
    PAIRING_MODE_OPTIMAL,
    PAIRING_MODE_RANDOM,
    PAIRING_MODE_SEQUENTIAL,
    PAIRING_MODE_TRACE_AWARE,
    PAIRING_MODES,
    frequency_aware_pair_routed_up_experts,
    greedy_pair_routed_up_experts,
    improve_routed_up_pairs,
    optimal_pair_routed_up_experts,
    random_pair_routed_up_experts,
    routed_up_pairing_cost,
    sequential_pair_routed_up_experts,
)
from mapping.trace_profile import (
    NUM_MOE_LAYERS,
    NUM_ROUTED_EXPERTS,
    TraceProfile,
)


def build_profile(
    *,
    frequency_layer0=None,
    cheap_sequential_pairs=False,
):
    if frequency_layer0 is None:
        frequency_layer0 = [1] * NUM_ROUTED_EXPERTS

    frequency = []
    for layer_id in range(NUM_MOE_LAYERS):
        if layer_id == 0:
            frequency.append(tuple(frequency_layer0))
        else:
            frequency.append(tuple([1] * NUM_ROUTED_EXPERTS))

    pair_size = NUM_ROUTED_EXPERTS * NUM_ROUTED_EXPERTS
    coactivation = []

    for layer_id in range(NUM_MOE_LAYERS):
        layer = array("Q", [0]) * pair_size

        # Layer-0 可以构造一个“顺序配对就是已知最优”的实例：
        # 所有边 cost=100，只有 (0,1),(2,3)... cost=0。
        if layer_id == 0 and cheap_sequential_pairs:
            for a in range(NUM_ROUTED_EXPERTS - 1):
                base = a * NUM_ROUTED_EXPERTS
                for b in range(a + 1, NUM_ROUTED_EXPERTS):
                    layer[base + b] = 100

            for a in range(0, NUM_ROUTED_EXPERTS, 2):
                layer[a * NUM_ROUTED_EXPERTS + a + 1] = 0

        coactivation.append(layer)

    return TraceProfile(
        file_count=1,
        trace_segment_count=1,
        skipped_segment_count=0,
        category_file_counts={"test": 1},
        frequency=tuple(frequency),
        coactivation=tuple(coactivation),
        token_count_by_layer=tuple([1] * NUM_MOE_LAYERS),
    )


def assert_perfect_pairing(pairs):
    assert len(pairs) == NUM_ROUTED_EXPERTS // 2
    flat = [expert for pair in pairs for expert in pair]
    assert len(flat) == NUM_ROUTED_EXPERTS
    assert len(set(flat)) == NUM_ROUTED_EXPERTS
    assert set(flat) == set(range(NUM_ROUTED_EXPERTS))
    assert all(a < b for a, b in pairs)


def test_all_pairing_modes_registered():
    assert set(PAIRING_MODES) == {
        PAIRING_MODE_SEQUENTIAL,
        PAIRING_MODE_RANDOM,
        PAIRING_MODE_FREQUENCY_AWARE,
        PAIRING_MODE_GREEDY,
        PAIRING_MODE_TRACE_AWARE,
        PAIRING_MODE_OPTIMAL,
    }


def test_random_pairing_is_reproducible():
    first = random_pair_routed_up_experts(layer_id=7, seed=42)
    second = random_pair_routed_up_experts(layer_id=7, seed=42)
    third = random_pair_routed_up_experts(layer_id=7, seed=43)

    assert first == second
    assert first != third
    assert_perfect_pairing(first)


def test_frequency_aware_pairs_hot_with_cold():
    # frequency 越大越热：E255 最热，E0 最冷。
    profile = build_profile(
        frequency_layer0=list(range(NUM_ROUTED_EXPERTS))
    )

    pairs = frequency_aware_pair_routed_up_experts(
        layer_id=0,
        profile=profile,
    )

    assert_perfect_pairing(pairs)
    assert pairs[0] == (0, 255)
    assert pairs[1] == (1, 254)
    assert pairs[-1] == (127, 128)


def test_greedy_and_local_search_are_valid():
    profile = build_profile(cheap_sequential_pairs=True)

    greedy = greedy_pair_routed_up_experts(
        layer_id=0,
        profile=profile,
    )
    improved = improve_routed_up_pairs(
        layer_id=0,
        pairs=greedy,
        profile=profile,
        max_rounds=2,
    )

    assert_perfect_pairing(greedy)
    assert_perfect_pairing(improved)

    greedy_cost = routed_up_pairing_cost(
        layer_id=0,
        pairs=greedy,
        profile=profile,
    )
    improved_cost = routed_up_pairing_cost(
        layer_id=0,
        pairs=improved,
        profile=profile,
    )

    assert improved_cost <= greedy_cost


def test_optimal_matching_reaches_known_zero_cost():
    profile = build_profile(cheap_sequential_pairs=True)

    try:
        pairs = optimal_pair_routed_up_experts(
            layer_id=0,
            profile=profile,
        )
    except Exception as exc:
        # 只有缺少 SciPy/NetworkX 时允许跳过，其他错误必须暴露。
        message = str(exc).lower()
        if "scipy" in message or "networkx" in message:
            pytest.skip(str(exc))
        raise

    assert_perfect_pairing(pairs)

    cost = routed_up_pairing_cost(
        layer_id=0,
        pairs=pairs,
        profile=profile,
    )

    assert cost == 0


def test_sequential_reference_still_unchanged():
    pairs = sequential_pair_routed_up_experts(layer_id=0)
    assert pairs[0] == (0, 1)
    assert pairs[-1] == (254, 255)
    assert_perfect_pairing(pairs)
