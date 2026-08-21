"""Prefill SC 内任务选择策略。"""

from __future__ import annotations

PREFILL_MODE_NO_REUSE = "no_reuse"
PREFILL_MODE_SWITCH_AWARE = "switch_aware"
PREFILL_MODE_AGGRESSIVE_REUSE = "aggressive_reuse"
PREFILL_MODE_LARGEST_BATCH_REUSE = "largest_batch_reuse"

PREFILL_SCHEDULING_MODES = (
    PREFILL_MODE_NO_REUSE,
    PREFILL_MODE_SWITCH_AWARE,
    PREFILL_MODE_AGGRESSIVE_REUSE,
    PREFILL_MODE_LARGEST_BATCH_REUSE,
)


def normalize_prefill_scheduling_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in PREFILL_SCHEDULING_MODES:
        raise ValueError(
            "未知 Prefill scheduling_mode："
            f"{mode!r}；可选={PREFILL_SCHEDULING_MODES}。"
        )
    return value


def prefill_task_priority(
    *,
    ready_time: int,
    route_rank: int,
    token_index: int,
    matrix_priority: int,
    cube_id: int,
    active_cube_id: int | None,
    scheduling_mode: str,
) -> tuple[int, ...]:
    """返回 Exact Scheduler 的确定性优先级 tuple。"""

    mode = normalize_prefill_scheduling_mode(scheduling_mode)
    active_rank = 0 if active_cube_id == cube_id else 1

    if mode == PREFILL_MODE_NO_REUSE:
        return (
            ready_time,
            route_rank,
            token_index,
            matrix_priority,
            cube_id,
        )

    if mode == PREFILL_MODE_SWITCH_AWARE:
        return (
            ready_time,
            active_rank,
            route_rank,
            token_index,
            matrix_priority,
            cube_id,
        )

    # aggressive_reuse / largest_batch_reuse：
    # largest_batch_reuse 的“按 WC Ready 数量”需要看到同一 SC 的整个队列，
    # 因此由 prefill_layer_scheduler._select_ready_task() 单独处理。
    # 这里保留 aggressive 风格的任务级 fallback。
    return (
        active_rank,
        ready_time,
        route_rank,
        token_index,
        matrix_priority,
        cube_id,
    )
