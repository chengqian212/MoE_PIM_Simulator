import json

from scheduling.trace_workload import (
    TraceWorkloadStats,
    iter_trace_tokens,
    scan_trace_workload,
)


# ============================================================
# Route
# ============================================================


def route(
    start: int,
):
    """
    构造保留顺序的 Top-8。

    特意不是升序，
    用来验证 trace_workload
    不会偷偷排序。
    """

    values = [
        start + 7,
        start + 0,
        start + 6,
        start + 1,
        start + 5,
        start + 2,
        start + 4,
        start + 3,
    ]

    return [
        value % 256
        for value
        in values
    ]


# ============================================================
# Fake Trace
# ============================================================


def build_fake_trace(
    tmp_path,
):

    root = (
        tmp_path
        / "Chinese-SimpleQA"
    )

    category = (
        root
        / "测试类别"
    )

    category.mkdir(
        parents=True
    )

    # ========================================================
    # Segment-0：
    #
    # 58 层
    # 每层 2 个 Token
    # ========================================================

    segment_0 = {
        "0": None,
        "1": None,
        "2": None,
    }

    for trace_layer in range(
        3,
        61,
    ):

        # 每层的 route 都稍微变化
        # 方便验证 Layer 对齐。
        segment_0[
            str(trace_layer)
        ] = [
            route(
                trace_layer
            ),

            route(
                trace_layer
                + 20
            ),
        ]

    # ========================================================
    # Segment-1：
    #
    # 故意让 Trace Layer-59 = null
    #
    # 整个 Segment 应被跳过。
    # ========================================================

    segment_1 = {
        "0": None,
        "1": None,
        "2": None,
    }

    for trace_layer in range(
        3,
        61,
    ):

        segment_1[
            str(trace_layer)
        ] = [
            route(
                trace_layer
                + 40
            )
        ]

    segment_1[
        "59"
    ] = None

    data = [
        segment_0,
        segment_1,
    ]

    path = (
        category
        / "0.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
        )

    return root


# ============================================================
# Tests
# ============================================================


def test_yields_complete_tokens(
    tmp_path,
):

    root = (
        build_fake_trace(
            tmp_path
        )
    )

    tokens = tuple(
        iter_trace_tokens(
            trace_root=root,

            verbose=False,
        )
    )

    # Segment-0 有两个 Token
    #
    # Segment-1 不完整，整段跳过
    assert (
        len(tokens)
        == 2
    )


def test_each_token_has_58_layers(
    tmp_path,
):

    root = (
        build_fake_trace(
            tmp_path
        )
    )

    tokens = tuple(
        iter_trace_tokens(
            trace_root=root,

            verbose=False,
        )
    )

    for token in tokens:

        assert (
            len(
                token.routed_experts_by_layer
            )
            == 58
        )

        for route_value in (
            token.routed_experts_by_layer
        ):

            assert (
                len(route_value)
                == 8
            )


def test_original_route_order_is_preserved(
    tmp_path,
):

    root = (
        build_fake_trace(
            tmp_path
        )
    )

    token = next(
        iter_trace_tokens(
            trace_root=root,

            verbose=False,
        )
    )

    # Project Layer-0
    # 对应 Trace Layer-3
    expected = tuple(
        route(3)
    )

    assert (
        token.route(0)
        == expected
    )

    # 不能变成 sorted(route)
    assert (
        token.route(0)
        != tuple(
            sorted(
                expected
            )
        )
    )


def test_layer_alignment(
    tmp_path,
):

    root = (
        build_fake_trace(
            tmp_path
        )
    )

    tokens = tuple(
        iter_trace_tokens(
            trace_root=root,

            verbose=False,
        )
    )

    token_0 = (
        tokens[0]
    )

    token_1 = (
        tokens[1]
    )

    # Project L0 = Trace L3
    assert (
        token_0.route(0)
        == tuple(
            route(3)
        )
    )

    assert (
        token_1.route(0)
        == tuple(
            route(23)
        )
    )

    # Project L57 = Trace L60
    assert (
        token_0.route(57)
        == tuple(
            route(60)
        )
    )

    assert (
        token_1.route(57)
        == tuple(
            route(80)
        )
    )


def test_incomplete_segment_is_skipped(
    tmp_path,
):

    root = (
        build_fake_trace(
            tmp_path
        )
    )

    stats = (
        scan_trace_workload(
            trace_root=root,

            verbose=False,
        )
    )

    assert (
        stats.trace_segment_count
        == 2
    )

    assert (
        stats.valid_segment_count
        == 1
    )

    assert (
        stats.skipped_segment_count
        == 1
    )

    assert (
        stats.yielded_token_count
        == 2
    )


def test_category_and_file_metadata(
    tmp_path,
):

    root = (
        build_fake_trace(
            tmp_path
        )
    )

    token = next(
        iter_trace_tokens(
            trace_root=root,

            verbose=False,
        )
    )

    assert (
        token.category
        == "测试类别"
    )

    assert (
        token.relative_file
        == str(
            (
                __import__(
                    "pathlib"
                )
                .Path(
                    "测试类别"
                )
                / "0.json"
            )
        )
    )

    assert (
        token.segment_index
        == 0
    )

    assert (
        token.token_index_in_segment
        == 0
    )


def test_max_tokens(
    tmp_path,
):

    root = (
        build_fake_trace(
            tmp_path
        )
    )

    stats = (
        TraceWorkloadStats()
    )

    tokens = tuple(
        iter_trace_tokens(
            trace_root=root,

            max_tokens=1,

            stats=stats,

            verbose=False,
        )
    )

    assert (
        len(tokens)
        == 1
    )

    assert (
        stats.yielded_token_count
        == 1
    )


def test_token_ids_are_continuous(
    tmp_path,
):

    root = (
        build_fake_trace(
            tmp_path
        )
    )

    tokens = tuple(
        iter_trace_tokens(
            trace_root=root,

            verbose=False,
        )
    )

    assert (
        [
            token.token_id
            for token
            in tokens
        ]
        == [
            0,
            1,
        ]
    )