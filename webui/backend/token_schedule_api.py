"""
完整 Token 的 58 层调度 API。

作用：

    单层：
        schedule_api.py

        Layer
          ↓
        Top-8 Routed Experts
          +
        Shared Expert 256
          ↓
        Layer Schedule


    本文件：

        Layer 0
          ↓
        Layer 1
          ↓
        ...
          ↓
        Layer 57
          ↓
        Token 总推理周期


当前 baseline：

1. 一共有 58 个 MoE Layer；
2. 每层 Router 激活 8 个 Routed Expert；
3. Shared Expert 256 始终激活；
4. 每个 Expert 有 gate / up / down 三个 Weight-Cube；
5. 不同 Sub-Cube 并行；
6. 同一 Sub-Cube 串行；
7. gate 和 up 可以并行；
8. down 等自己的 gate + up 完成；
9. 当前不做跨 Layer pipeline；
10. Layer 0 → Layer 57 顺序执行。
"""

from __future__ import annotations

from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
)

from pydantic import BaseModel

from webui.backend.schedule_api import (
    NUM_LAYERS,
    TOP_K,
    SHARED_EXPERT_ID,
    schedule_layer,
)


# ============================================================
# Router
# ============================================================


router = APIRouter(
    prefix="/api/token-schedule",
    tags=["Token Schedule"],
)


# ============================================================
# Request
# ============================================================


class TokenScheduleRequest(
    BaseModel
):
    """
    输入格式：

    {
        "routes": [
            [102, 125, ..., 140],   # Layer 0
            [ ... 8 experts ... ],  # Layer 1
            ...
            [ ... 8 experts ... ]   # Layer 57
        ]
    }

    routes 必须：

        len(routes) == 58

    并且：

        每层恰好 8 个 Routed Expert。
    """

    routes: list[list[int]]

    charge_initial_activation: bool = True

    include_tasks: bool = True


# ============================================================
# Route Validation
# ============================================================


def validate_token_routes(
    routes: list[list[int]],
) -> None:

    # --------------------------------------------------------
    # 必须 58 层
    # --------------------------------------------------------

    if (
        len(routes)
        != NUM_LAYERS
    ):

        raise ValueError(
            "一个完整 Token 必须包含 "
            f"{NUM_LAYERS} 个 MoE Layer，"
            f"当前收到 {len(routes)} 层。"
        )


    # --------------------------------------------------------
    # 每层检查
    # --------------------------------------------------------

    for (
        layer_id,
        routed_experts,
    ) in enumerate(
        routes
    ):

        if (
            len(routed_experts)
            != TOP_K
        ):

            raise ValueError(
                f"Layer {layer_id} "
                f"必须恰好包含 {TOP_K} 个 "
                "Routed Expert，"
                f"当前为 {len(routed_experts)} 个。"
            )


        if (
            len(
                set(
                    routed_experts
                )
            )
            != TOP_K
        ):

            raise ValueError(
                f"Layer {layer_id} "
                "的 Routed Expert "
                "存在重复 ID。"
            )


        for expert_id in (
            routed_experts
        ):

            if not (
                0
                <= expert_id
                <= 255
            ):

                raise ValueError(
                    f"Layer {layer_id} "
                    f"出现非法 Routed Expert："
                    f"{expert_id}。"
                    "合法范围为 0~255。"
                )


# ============================================================
# 将单层 task 的局部周期
# 转换成整个 Token 的全局周期
# ============================================================


def offset_task(
    task: dict[str, Any],
    layer_start_cycle: int,
) -> dict[str, Any]:

    result = dict(
        task
    )


    # --------------------------------------------------------
    # 这些字段属于时间坐标
    # --------------------------------------------------------

    for key in (
        "ready_time",
        "start_cycle",
        "compute_start_cycle",
        "end_cycle",
    ):

        value = result.get(
            key
        )


        if (
            value is not None
            and value >= 0
        ):

            result[
                key
            ] = (
                value
                + layer_start_cycle
            )


    # --------------------------------------------------------
    # 增加局部周期，
    # 方便前端同时显示：
    #
    # Token Cycle = 127
    # Layer Cycle = 4
    # --------------------------------------------------------

    result[
        "local_ready_time"
    ] = task.get(
        "ready_time"
    )


    result[
        "local_start_cycle"
    ] = task.get(
        "start_cycle"
    )


    result[
        "local_compute_start_cycle"
    ] = task.get(
        "compute_start_cycle"
    )


    result[
        "local_end_cycle"
    ] = task.get(
        "end_cycle"
    )


    return result


# ============================================================
# 完整 Token Scheduler
# ============================================================


def schedule_token(
    *,
    routes: list[list[int]],

    charge_initial_activation: bool = True,

    include_tasks: bool = True,
) -> dict[str, Any]:

    validate_token_routes(
        routes
    )


    # ========================================================
    # Token 全局时间
    # ========================================================

    token_cycle = 0


    # ========================================================
    # Layer 结果
    # ========================================================

    layer_results: list[
        dict[str, Any]
    ] = []


    # ========================================================
    # 全 Token Tasks
    # ========================================================

    all_tasks: list[
        dict[str, Any]
    ] = []


    # ========================================================
    # SC 全 Token 汇总
    # ========================================================

    total_sc_task_count = [
        0
        for _ in range(16)
    ]


    total_sc_switch_count = [
        0
        for _ in range(16)
    ]


    total_sc_busy_cycles = [
        0
        for _ in range(16)
    ]


    critical_count = [
        0
        for _ in range(16)
    ]


    # ========================================================
    # Layer 0 → Layer 57
    # ========================================================

    for layer_id in range(
        NUM_LAYERS
    ):

        routed_experts = (
            routes[
                layer_id
            ]
        )


        # ====================================================
        # 调用已经验证通过的单层 Scheduler
        # ====================================================

        layer_schedule = (
            schedule_layer(
                layer_id=layer_id,

                routed_expert_ids=(
                    routed_experts
                ),

                charge_initial_activation=(
                    charge_initial_activation
                ),
            )
        )


        layer_cycles = int(
            layer_schedule[
                "layer_cycles"
            ]
        )


        layer_start_cycle = (
            token_cycle
        )


        layer_end_cycle = (
            layer_start_cycle
            + layer_cycles
        )


        # ====================================================
        # Critical SC 统计
        # ====================================================

        critical_subcubes = (
            layer_schedule.get(
                "critical_subcubes",
                [],
            )
        )


        for sc in (
            critical_subcubes
        ):

            if (
                0
                <= sc
                < 16
            ):

                critical_count[
                    sc
                ] += 1


        # ====================================================
        # SC 统计
        # ====================================================

        for stat in (
            layer_schedule.get(
                "subcubes",
                [],
            )
        ):

            sc = int(
                stat[
                    "subcube_id"
                ]
            )


            if not (
                0
                <= sc
                < 16
            ):
                continue


            total_sc_task_count[
                sc
            ] += int(
                stat.get(
                    "task_count",
                    0,
                )
            )


            total_sc_switch_count[
                sc
            ] += int(
                stat.get(
                    "switch_count",
                    0,
                )
            )


            total_sc_busy_cycles[
                sc
            ] += int(
                stat.get(
                    "busy_cycles",
                    0,
                )
            )


        # ====================================================
        # Task 全局时间偏移
        # ====================================================

        global_tasks = []


        if include_tasks:

            for task in (
                layer_schedule.get(
                    "tasks",
                    [],
                )
            ):

                global_task = (
                    offset_task(
                        task,
                        layer_start_cycle,
                    )
                )


                global_tasks.append(
                    global_task
                )


                all_tasks.append(
                    global_task
                )


        # ====================================================
        # 保存这一层结果
        # ====================================================

        layer_result = {
            "layer_id":
                layer_id,

            "routed_expert_ids":
                list(
                    routed_experts
                ),

            "shared_expert_id":
                SHARED_EXPERT_ID,

            "start_cycle":
                layer_start_cycle,

            "end_cycle":
                layer_end_cycle,

            "layer_cycles":
                layer_cycles,

            "task_count":
                layer_schedule.get(
                    "task_count",
                    0,
                ),

            "active_subcube_count":
                layer_schedule.get(
                    "active_subcube_count",
                    0,
                ),

            "critical_subcubes":
                list(
                    critical_subcubes
                ),
        }


        if include_tasks:

            layer_result[
                "tasks"
            ] = global_tasks


        layer_results.append(
            layer_result
        )


        # ====================================================
        # 当前 Layer 完成，
        # 下一个 Layer 从这里开始
        # ====================================================

        token_cycle = (
            layer_end_cycle
        )


    # ========================================================
    # Token 总周期
    # ========================================================

    total_cycles = (
        token_cycle
    )


    # ========================================================
    # Layer latency
    # ========================================================

    layer_cycle_values = [

        layer[
            "layer_cycles"
        ]

        for layer
        in layer_results

    ]


    min_layer_cycles = min(
        layer_cycle_values
    )


    max_layer_cycles = max(
        layer_cycle_values
    )


    mean_layer_cycles = (
        sum(
            layer_cycle_values
        )
        /
        len(
            layer_cycle_values
        )
    )


    # ========================================================
    # 最慢 Layer
    # ========================================================

    slowest_layers = sorted(
        layer_results,

        key=lambda layer: (
            -layer[
                "layer_cycles"
            ],

            layer[
                "layer_id"
            ],
        ),
    )[:5]


    # ========================================================
    # SC 汇总
    # ========================================================

    subcube_summary = []


    for sc in range(16):

        subcube_summary.append(
            {
                "subcube_id":
                    sc,

                "task_count":
                    total_sc_task_count[
                        sc
                    ],

                "switch_count":
                    total_sc_switch_count[
                        sc
                    ],

                "busy_cycles":
                    total_sc_busy_cycles[
                        sc
                    ],

                "critical_layer_count":
                    critical_count[
                        sc
                    ],

                "critical_layer_rate":
                    (
                        critical_count[
                            sc
                        ]
                        /
                        NUM_LAYERS
                    ),
            }
        )


    # ========================================================
    # 返回
    # ========================================================

    result: dict[
        str,
        Any
    ] = {
        "total_cycles":
            total_cycles,

        "layer_count":
            NUM_LAYERS,

        "active_experts_per_layer":
            TOP_K + 1,

        "tasks_per_layer":
            (
                TOP_K + 1
            )
            * 3,

        "total_task_count":
            (
                NUM_LAYERS
                *
                (
                    TOP_K + 1
                )
                *
                3
            ),

        "shared_expert_id":
            SHARED_EXPERT_ID,

        "layer_cycle_stats": {
            "min":
                min_layer_cycles,

            "max":
                max_layer_cycles,

            "mean":
                mean_layer_cycles,
        },

        "slowest_layers": [
            {
                "layer_id":
                    layer[
                        "layer_id"
                    ],

                "cycles":
                    layer[
                        "layer_cycles"
                    ],

                "critical_subcubes":
                    layer[
                        "critical_subcubes"
                    ],
            }
            for layer
            in slowest_layers
        ],

        "layers":
            layer_results,

        "subcubes":
            subcube_summary,
    }


    if include_tasks:

        result[
            "tasks"
        ] = all_tasks


    return result


# ============================================================
# API
# ============================================================


@router.post(
    "/token"
)
def schedule_one_token(
    request: TokenScheduleRequest,
):

    try:

        return schedule_token(
            routes=(
                request.routes
            ),

            charge_initial_activation=(
                request
                .charge_initial_activation
            ),

            include_tasks=(
                request
                .include_tasks
            ),
        )


    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


    except RuntimeError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# Health
# ============================================================


@router.get(
    "/health"
)
def token_schedule_health():

    return {
        "status":
            "ok",

        "num_layers":
            NUM_LAYERS,

        "top_k":
            TOP_K,

        "shared_expert_id":
            SHARED_EXPERT_ID,

        "tasks_per_layer":
            (
                TOP_K + 1
            )
            * 3,

        "tasks_per_token":
            (
                NUM_LAYERS
                *
                (
                    TOP_K + 1
                )
                *
                3
            ),
    }


# ============================================================
# 命令行自测
#
# python -m webui.backend.token_schedule_api
# ============================================================


def main() -> None:

    print(
        "========== "
        "Token Schedule API Self Test "
        "=========="
    )


    # --------------------------------------------------------
    # 自测用路线
    #
    # 每层使用不同起点生成 8 个合法 Expert。
    #
    # 这里只为了验证 API，
    # 不是实际 Trace。
    # --------------------------------------------------------

    routes = []


    for layer_id in range(
        NUM_LAYERS
    ):

        start = (
            layer_id * 7
        ) % 256


        route = [

            (
                start
                + offset * 17
            )
            % 256

            for offset
            in range(
                TOP_K
            )

        ]


        # 理论上这里不会重复，
        # 仍然明确检查一下。
        if (
            len(
                set(
                    route
                )
            )
            != TOP_K
        ):

            raise RuntimeError(
                "Self-test route "
                "意外出现重复 Expert。"
            )


        routes.append(
            route
        )


    print(
        "Layers:",
        len(
            routes
        ),
    )


    print(
        "Layer-0 route:",
        routes[0],
    )


    print(
        "Layer-57 route:",
        routes[-1],
    )


    result = (
        schedule_token(
            routes=routes,

            charge_initial_activation=True,

            include_tasks=True,
        )
    )


    print(
        "\nTotal cycles:",
        result[
            "total_cycles"
        ],
    )


    print(
        "Layer count:",
        result[
            "layer_count"
        ],
    )


    print(
        "Total tasks:",
        result[
            "total_task_count"
        ],
    )


    print(
        "Returned tasks:",
        len(
            result[
                "tasks"
            ]
        ),
    )


    print(
        "\nLayer cycle stats:",
        result[
            "layer_cycle_stats"
        ],
    )


    print(
        "\nSlowest layers:"
    )


    for layer in (
        result[
            "slowest_layers"
        ]
    ):

        print(
            "  "
            f"Layer {layer['layer_id']:>2}: "
            f"{layer['cycles']} cycles, "
            "critical="
            f"{layer['critical_subcubes']}"
        )


    # --------------------------------------------------------
    # 必须 58 层
    # --------------------------------------------------------

    assert (
        result[
            "layer_count"
        ]
        == 58
    )


    # --------------------------------------------------------
    # 每层：
    #
    # 8 Routed + 1 Shared
    # ×
    # gate/up/down
    #
    # = 27
    #
    # Token：
    #
    # 58 × 27 = 1566
    # --------------------------------------------------------

    assert (
        result[
            "total_task_count"
        ]
        == 1566
    )


    assert (
        len(
            result[
                "tasks"
            ]
        )
        == 1566
    )


    # --------------------------------------------------------
    # 所有 Layer 必须连续
    # --------------------------------------------------------

    previous_end = 0


    for layer in (
        result[
            "layers"
        ]
    ):

        assert (
            layer[
                "start_cycle"
            ]
            == previous_end
        )


        assert (
            layer[
                "end_cycle"
            ]
            ==
            layer[
                "start_cycle"
            ]
            +
            layer[
                "layer_cycles"
            ]
        )


        previous_end = (
            layer[
                "end_cycle"
            ]
        )


    assert (
        previous_end
        ==
        result[
            "total_cycles"
        ]
    )


    print(
        "\nPASS"
    )


if __name__ == "__main__":

    main()