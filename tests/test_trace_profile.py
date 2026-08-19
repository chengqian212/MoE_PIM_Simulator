import json

import pytest

from mapping.trace_profile import (
    TraceProfileError,
    load_chinese_simpleqa_profile,
    project_layer_to_trace_layer,
    trace_layer_to_project_layer,
)


def make_segment(
    routes,
):
    """
    构造与真实 trace 相同结构的一个 segment。
    """

    segment = {
        "0": None,
        "1": None,
        "2": None,
    }

    for layer in range(
        3,
        61,
    ):
        segment[str(layer)] = routes

    return segment


def test_layer_mapping():

    assert (
        trace_layer_to_project_layer(3)
        == 0
    )

    assert (
        trace_layer_to_project_layer(60)
        == 57
    )

    assert (
        project_layer_to_trace_layer(0)
        == 3
    )

    assert (
        project_layer_to_trace_layer(57)
        == 60
    )


def test_profile_frequency_and_coactivation(
    tmp_path,
):

    category = (
        tmp_path
        / "工程、技术与应用科学"
    )

    category.mkdir()

    # 第一个 segment：
    # 两个 token
    segment_0 = make_segment(
        [
            [
                0, 1, 2, 3,
                4, 5, 6, 7,
            ],
            [
                8, 9, 10, 11,
                12, 13, 14, 15,
            ],
        ]
    )

    # 第二个 segment：
    # 一个 token
    segment_1 = make_segment(
        [
            [
                0, 8, 16, 24,
                32, 40, 48, 56,
            ]
        ]
    )

    path = (
        category
        / "test.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            [
                segment_0,
                segment_1,
            ],
            file,
        )

    profile = (
        load_chinese_simpleqa_profile(
            trace_root=tmp_path,
            strict=True,
            verbose=False,
        )
    )

    # 每层：
    #
    # segment0 = 2 token
    # segment1 = 1 token
    #
    # 共 3 token
    assert (
        profile.tokens_per_layer
        == 3
    )

    # Expert-0：
    #
    # token-1 中出现
    # token-3 中出现
    #
    # 因此 2 次
    assert (
        profile.frequency_count(
            0,
            0,
        )
        == 2
    )

    # Expert-1：
    # 只在 token-1 出现
    assert (
        profile.frequency_count(
            0,
            1,
        )
        == 1
    )

    # Expert-8：
    #
    # token-2
    # token-3
    #
    # 共 2 次
    assert (
        profile.frequency_count(
            0,
            8,
        )
        == 2
    )

    # E0 和 E1：
    # 只在 token-1 共激活
    assert (
        profile.coactivation_count(
            0,
            0,
            1,
        )
        == 1
    )

    # E0 和 E8：
    # 只在 token-3 共激活
    assert (
        profile.coactivation_count(
            0,
            0,
            8,
        )
        == 1
    )

    # E1 和 E8 从未同时出现
    assert (
        profile.coactivation_count(
            0,
            1,
            8,
        )
        == 0
    )


def test_all_58_layers_are_counted(
    tmp_path,
):

    category = (
        tmp_path
        / "中华文化"
    )

    category.mkdir()

    segment = make_segment(
        [
            [
                0, 1, 2, 3,
                4, 5, 6, 7,
            ]
        ]
    )

    path = category / "a.json"

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            [segment],
            file,
        )

    profile = (
        load_chinese_simpleqa_profile(
            trace_root=tmp_path,
            strict=True,
            verbose=False,
        )
    )

    assert (
        len(
            profile.token_count_by_layer
        )
        == 58
    )

    assert all(
        count == 1
        for count in
        profile.token_count_by_layer
    )


def test_invalid_top8_is_rejected(
    tmp_path,
):

    category = (
        tmp_path
        / "社会"
    )

    category.mkdir()

    # 错误：
    # 只有 7 个 Expert
    segment = make_segment(
        [
            [
                0, 1, 2, 3,
                4, 5, 6,
            ]
        ]
    )

    path = category / "bad.json"

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            [segment],
            file,
        )

    with pytest.raises(
        TraceProfileError
    ):
        load_chinese_simpleqa_profile(
            trace_root=tmp_path,
            strict=True,
            verbose=False,
        )