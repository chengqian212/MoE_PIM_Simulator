"""
Prefill / Decode 正式评估链回归测试。

目的：

    防止后续修改 Mapping、Scheduler、WebUI 或评估脚本时，
    不小心破坏现在已经确认的 Prefill / Decode 语义。

重点固定以下规则：

1. Prefill Batch Size = 1 时
   必须退化为原单 Token Scheduler。

2. Prefill 总任务数：

       B × 58 × 9 × 3

   其中：
       58 层
       8 routed expert + 1 shared expert
       gate / up / down 三个矩阵

3. Decode workload 只能读取 segment1+，
   segment0 不能混入 Decode。

4. 每个 Decode Token：

       58 × 9 × 3 = 1566 tasks

5. Fast Decode Scheduler
   必须和 Exact token_scheduler
   在 total cycles 和 58 层 cycles 上完全一致。

6. Phase Summary
   必须能正确解析 Prefill + Decode 两阶段结果。

运行：

    python -m pytest tests/test_phase_evaluation.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest


from config import ExecutionRules

from scheduling.decode_fast_evaluator import (
    FastDecodeScheduler,
)

from scheduling.decode_workload import (
    DecodeWorkloadStats,
    iter_decode_tokens,
)

from scheduling.phase_evaluation_summary import (
    build_phase_summary,
)

from scheduling.prefill_scheduler import (
    schedule_prefill_batch,
)

from scheduling.runtime_index import (
    DEFAULT_MAPPING_PATH,
    load_runtime_index,
)

from scheduling.token_scheduler import (
    schedule_token,
)


# ============================================================
# 常量
# ============================================================


NUM_LAYERS = 58

ROUTED_EXPERTS_PER_TOKEN = 8

SHARED_EXPERTS_PER_TOKEN = 1

ACTIVE_EXPERTS_PER_TOKEN = (
    ROUTED_EXPERTS_PER_TOKEN
    +
    SHARED_EXPERTS_PER_TOKEN
)

MATRICES_PER_EXPERT = 3

TASKS_PER_TOKEN = (
    NUM_LAYERS
    *
    ACTIVE_EXPERTS_PER_TOKEN
    *
    MATRICES_PER_EXPERT
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(
    scope="module"
)
def runtime_index():

    mapping_path = (
        Path(
            DEFAULT_MAPPING_PATH
        )
        .resolve()
    )

    if not (
        mapping_path.exists()
    ):

        pytest.skip(
            f"Mapping 不存在："
            f"{mapping_path}"
        )

    return (
        load_runtime_index(
            mapping_path
        )
    )


@pytest.fixture(
    scope="module"
)
def first_decode_token():

    stats = (
        DecodeWorkloadStats()
    )

    iterator = (
        iter_decode_tokens(
            max_tokens=1,

            stats=(
                stats
            ),

            strict_singleton=True,

            verbose=False,
        )
    )

    try:

        token = next(
            iterator
        )

    except StopIteration:

        pytest.skip(
            "当前 Trace 中没有可用 Decode Token。"
        )

    return token


# ============================================================
# 1. Prefill B=1 == Token Scheduler
# ============================================================


def test_prefill_batch_size_1_matches_token_scheduler(
    runtime_index,
    first_decode_token,
):

    rules = (
        ExecutionRules()
    )

    routes = (
        first_decode_token
        .routed_experts_by_layer
    )

    # --------------------------------------------------------
    # 原单 Token Scheduler
    # --------------------------------------------------------

    token_result = (
        schedule_token(
            index=(
                runtime_index
            ),

            routed_experts_by_layer=(
                routes
            ),

            rules=(
                rules
            ),

            initial_active_cube_by_subcube=None,

            charge_initial_activation=True,
        )
    )

    # --------------------------------------------------------
    # Prefill Scheduler：
    #
    # shape:
    #     Token × Layer × Top8
    #
    # B = 1
    # --------------------------------------------------------

    prefill_result = (
        schedule_prefill_batch(
            index=(
                runtime_index
            ),

            routed_experts_by_token=(
                (
                    routes,
                )
            ),

            rules=(
                rules
            ),

            initial_active_cube_by_subcube=None,

            charge_initial_activation=True,
        )
    )

    # --------------------------------------------------------
    # 总周期完全一致
    # --------------------------------------------------------

    assert (
        prefill_result.total_cycles
        ==
        token_result.total_cycles
    )

    # --------------------------------------------------------
    # 总任务数完全一致
    # --------------------------------------------------------

    assert (
        prefill_result.total_tasks
        ==
        token_result.total_tasks
    )

    # --------------------------------------------------------
    # 58 层逐层周期完全一致
    # --------------------------------------------------------

    token_layer_cycles = tuple(
        layer.cycles

        for layer
        in token_result.layers
    )

    prefill_layer_cycles = tuple(
        layer.cycles

        for layer
        in prefill_result.layers
    )

    assert (
        prefill_layer_cycles
        ==
        token_layer_cycles
    )


# ============================================================
# 2. Prefill Task 数公式
# ============================================================


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        2,
        3,
    ],
)
def test_prefill_task_count_formula(
    runtime_index,
    first_decode_token,
    batch_size,
):

    routes = (
        first_decode_token
        .routed_experts_by_layer
    )

    routed_experts_by_token = tuple(
        routes

        for _ in range(
            batch_size
        )
    )

    result = (
        schedule_prefill_batch(
            index=(
                runtime_index
            ),

            routed_experts_by_token=(
                routed_experts_by_token
            ),

            rules=(
                ExecutionRules()
            ),

            initial_active_cube_by_subcube=None,

            charge_initial_activation=True,
        )
    )

    expected = (
        batch_size
        *
        TASKS_PER_TOKEN
    )

    assert (
        result.total_tasks
        ==
        expected
    )


# ============================================================
# 3. Decode workload 不允许 segment0
# ============================================================


def test_decode_workload_excludes_segment0():

    stats = (
        DecodeWorkloadStats()
    )

    tokens = list(
        iter_decode_tokens(
            max_tokens=100,

            stats=(
                stats
            ),

            strict_singleton=True,

            verbose=False,
        )
    )

    if not tokens:

        pytest.skip(
            "当前 Trace 中没有可用 Decode Token。"
        )

    # --------------------------------------------------------
    # Decode 必须全部来自 segment1+
    # --------------------------------------------------------

    assert all(
        token.segment_index
        >= 1

        for token
        in tokens
    )

    # --------------------------------------------------------
    # strict_singleton=True 下，
    # 每个 Decode Segment 只有 1 Token，
    # 因此 token_index_in_segment 必须为 0。
    # --------------------------------------------------------

    assert all(
        token.token_index_in_segment
        == 0

        for token
        in tokens
    )

    # --------------------------------------------------------
    # 不允许任何非 singleton Decode Segment
    # --------------------------------------------------------

    assert (
        stats
        .non_singleton_decode_segment_count
        ==
        0
    )


# ============================================================
# 4. Decode 每 Token = 1566 Tasks
# ============================================================


def test_decode_token_has_1566_tasks(
    runtime_index,
    first_decode_token,
):

    result = (
        schedule_token(
            index=(
                runtime_index
            ),

            routed_experts_by_layer=(
                first_decode_token
                .routed_experts_by_layer
            ),

            rules=(
                ExecutionRules()
            ),

            initial_active_cube_by_subcube=None,

            charge_initial_activation=True,
        )
    )

    assert (
        TASKS_PER_TOKEN
        ==
        1566
    )

    assert (
        result.total_tasks
        ==
        1566
    )


# ============================================================
# 5. FAST == EXACT
# ============================================================


def test_fast_decode_matches_exact_first_10_tokens(
    runtime_index,
):

    rules = (
        ExecutionRules()
    )

    fast_scheduler = (
        FastDecodeScheduler(
            index=(
                runtime_index
            ),

            rules=(
                rules
            ),

            cache_size=10000,
        )
    )

    stats = (
        DecodeWorkloadStats()
    )

    tokens = list(
        iter_decode_tokens(
            max_tokens=10,

            stats=(
                stats
            ),

            strict_singleton=True,

            verbose=False,
        )
    )

    if not tokens:

        pytest.skip(
            "当前 Trace 中没有可用 Decode Token。"
        )

    for token in tokens:

        routes = (
            token
            .routed_experts_by_layer
        )

        # ----------------------------------------------------
        # FAST
        # ----------------------------------------------------

        fast_result = (
            fast_scheduler
            .schedule_token(
                routes
            )
        )

        # ----------------------------------------------------
        # EXACT
        # ----------------------------------------------------

        exact_result = (
            schedule_token(
                index=(
                    runtime_index
                ),

                routed_experts_by_layer=(
                    routes
                ),

                rules=(
                    rules
                ),

                initial_active_cube_by_subcube=None,

                charge_initial_activation=True,
            )
        )

        exact_layer_cycles = tuple(
            execution.cycles

            for execution
            in exact_result.layers
        )

        # ----------------------------------------------------
        # Total cycles
        # ----------------------------------------------------

        assert (
            fast_result.total_cycles
            ==
            exact_result.total_cycles
        ), (
            f"Token-{token.token_id}: "
            f"FAST total="
            f"{fast_result.total_cycles}, "
            f"EXACT total="
            f"{exact_result.total_cycles}"
        )

        # ----------------------------------------------------
        # 58 Layer cycles
        # ----------------------------------------------------

        assert (
            fast_result.layer_cycles
            ==
            exact_layer_cycles
        ), (
            f"Token-{token.token_id}: "
            "FAST 与 EXACT 的 Layer cycles 不一致。"
        )


# ============================================================
# 6. Phase Summary Parser
# ============================================================


def test_phase_summary_parses_prefill_and_decode():

    # --------------------------------------------------------
    # 构造一个最小合法 Prefill JSON
    #
    # 字段名严格按照当前
    # prefill_evaluation.json / summary 结构。
    # --------------------------------------------------------

    prefill_payload = {
        "summary": {
            "batch_count": 2,

            "total_input_tokens": 20,

            "prompt_tokens": {
                "count": 2,
                "minimum": 8.0,
                "mean": 10.0,
                "p50": 10.0,
                "p95": 11.8,
                "p99": 11.96,
                "maximum": 12.0,
            },

            "total_cycles": {
                "count": 2,
                "minimum": 2000.0,
                "mean": 2500.0,
                "p50": 2500.0,
                "p95": 2950.0,
                "p99": 2990.0,
                "maximum": 3000.0,
            },

            "cycles_per_input_token": {
                "count": 2,
                "minimum": 200.0,
                "mean": 225.0,
                "p50": 225.0,
                "p95": 247.5,
                "p99": 249.5,
                "maximum": 250.0,
            },

            "global_cycles_per_input_token": 230.0,

            "global_input_tokens_per_cycle": (
                1.0
                /
                230.0
            ),

            "prompt_length_latency_pearson": 0.99,
        }
    }

    # --------------------------------------------------------
    # 构造一个最小合法 Decode JSON
    # --------------------------------------------------------

    decode_payload = {
        "summary": {
            "scheduler_mode": (
                "fast_exact-validated"
            ),

            "exact_checked_tokens": 100,

            "token_count": 3,

            "cycles_per_token": {
                "count": 3,
                "minimum": 500.0,
                "mean": 520.0,
                "p50": 520.0,
                "p95": 538.0,
                "p99": 539.6,
                "maximum": 540.0,
            },
        }
    }

    summary = (
        build_phase_summary(
            prefill_payload=(
                prefill_payload
            ),

            decode_payload=(
                decode_payload
            ),
        )
    )

    assert (
        summary.prefill.batch_count
        ==
        2
    )

    assert (
        summary.prefill.total_input_tokens
        ==
        20
    )

    assert (
        summary.prefill.latency_cycles.mean
        ==
        2500.0
    )

    assert (
        summary.decode.token_count
        ==
        3
    )

    assert (
        summary.decode.exact_checked_tokens
        ==
        100
    )

    assert (
        summary.decode.cycles_per_token.mean
        ==
        520.0
    )


# ============================================================
# 7. Summary 拒绝非法 Decode Scheduler
# ============================================================


def test_phase_summary_rejects_unvalidated_decode():

    prefill_payload = {
        "summary": {
            "batch_count": 1,

            "total_input_tokens": 10,

            "prompt_tokens": {
                "count": 1,
                "minimum": 10.0,
                "mean": 10.0,
                "p50": 10.0,
                "p95": 10.0,
                "p99": 10.0,
                "maximum": 10.0,
            },

            "total_cycles": {
                "count": 1,
                "minimum": 2000.0,
                "mean": 2000.0,
                "p50": 2000.0,
                "p95": 2000.0,
                "p99": 2000.0,
                "maximum": 2000.0,
            },

            "cycles_per_input_token": {
                "count": 1,
                "minimum": 200.0,
                "mean": 200.0,
                "p50": 200.0,
                "p95": 200.0,
                "p99": 200.0,
                "maximum": 200.0,
            },

            "global_cycles_per_input_token": 200.0,

            "global_input_tokens_per_cycle": 0.005,

            "prompt_length_latency_pearson": 0.0,
        }
    }

    decode_payload = {
        "summary": {
            "scheduler_mode": (
                "some_unvalidated_scheduler"
            ),

            "exact_checked_tokens": 0,

            "token_count": 1,

            "cycles_per_token": {
                "count": 1,
                "minimum": 500.0,
                "mean": 500.0,
                "p50": 500.0,
                "p95": 500.0,
                "p99": 500.0,
                "maximum": 500.0,
            },
        }
    }

    with pytest.raises(
        ValueError
    ):

        build_phase_summary(
            prefill_payload=(
                prefill_payload
            ),

            decode_payload=(
                decode_payload
            ),
        )
