"""
MoE-PIM Web UI 单层调度 API。

这个文件只负责给 Web UI 提供可视化需要的单层调度结果。

输入：
    layer_id
    8 个 Routed Expert ID

自动加入：
    Shared Expert 256

输出：
    - 当前层总周期
    - 27 个 gate / up / down 的执行时间
    - 每个任务位于哪个 Sub-Cube
    - 是否发生 Weight-Cube switch
    - 16 个 Sub-Cube 的时间线
    - critical Sub-Cube

当前执行规则：

1. gate 和 up 初始都可以执行；
2. 某 Expert 的 down 等自己的 gate 和 up 都完成；
3. 不同 Sub-Cube 可以并行；
4. 同一 Sub-Cube 一次只能执行一个 Weight-Cube；
5. 切换到另一个 Weight-Cube：1 cycle；
6. Weight-Cube depth=1，因此计算：1 cycle；
7. 跨 Sub-Cube：0 cycle；
8. Shared Expert 256 始终激活。

注意：
    本文件是 Web UI 的单层可视化调度接口。
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
)

from pydantic import BaseModel


# ============================================================
# Router
# ============================================================


router = APIRouter(
    prefix="/api/schedule",
    tags=["Schedule"],
)


# ============================================================
# 项目路径
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


DEFAULT_MAPPING_PATH = (
    PROJECT_ROOT
    / "results"
    / "mappings"
    / "mapping_baseline_N4_H7168_W4096.json"
)


# ============================================================
# 当前执行规则
# ============================================================


NUM_LAYERS = 58

NUM_ROUTED_EXPERTS = 256

SHARED_EXPERT_ID = 256

TOP_K = 8

NUM_SUBCUBES = 16


SWITCH_CYCLES = 1

COMPUTE_CYCLES = 1


# ============================================================
# API Request
# ============================================================


class LayerScheduleRequest(
    BaseModel
):
    """
    前端请求：

    {
        "layer_id": 0,
        "routed_expert_ids": [
            102,
            125,
            149,
            186,
            206,
            215,
            217,
            140
        ]
    }
    """

    layer_id: int

    routed_expert_ids: list[int]

    charge_initial_activation: bool = True


# ============================================================
# 内部 Weight 数据
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class WeightLocation:

    cube_id: int

    layer_id: int

    expert_id: int

    is_shared: bool

    matrix_name: str

    subcube_id: int

    z: int

    physical_plane_id: int | None

    slot_id: int | None


# ============================================================
# Scheduler Task
# ============================================================


@dataclass(
    slots=True,
)
class RuntimeTask:

    task_id: str

    cube_id: int

    layer_id: int

    expert_id: int

    is_shared: bool

    route_rank: int

    matrix_name: str

    subcube_id: int

    z: int

    physical_plane_id: int | None

    slot_id: int | None

    ready_time: int = 0

    started: bool = False

    finished: bool = False

    start_cycle: int | None = None

    switch_cycles: int = 0

    compute_start_cycle: int | None = None

    end_cycle: int | None = None


# ============================================================
# JSON 工具
# ============================================================


def first_value(
    obj: dict[str, Any],
    *keys: str,
) -> Any:

    for key in keys:

        if (
            key in obj
            and
            obj[key] is not None
        ):
            return obj[key]

    return None


# ============================================================
# 在任意 JSON 结构中寻找 Placement
#
# 我们前面的 main.py 已经碰到过 Mapping JSON 层级不同的问题，
# 所以这里不要假定 placements 一定在某个固定 key。
# ============================================================


def looks_like_weight(
    obj: Any,
) -> bool:

    if not isinstance(
        obj,
        dict,
    ):
        return False


    has_identity = (
        first_value(
            obj,
            "cube_id",
            "weight_cube_id",
        )
        is not None
    )


    has_layer = (
        first_value(
            obj,
            "layer_id",
        )
        is not None
    )


    has_expert = (
        first_value(
            obj,
            "expert_id",
        )
        is not None
    )


    has_matrix = (
        first_value(
            obj,
            "matrix_name",
            "matrix",
        )
        is not None
    )


    has_sc = (
        first_value(
            obj,
            "subcube_id",
            "sc_id",
        )
        is not None
    )


    return (
        has_identity
        and has_layer
        and has_expert
        and has_matrix
        and has_sc
    )


def collect_weight_records(
    obj: Any,
    output: list[dict[str, Any]],
) -> None:
    """
    递归扫描 Mapping JSON。

    找到类似：

        {
            cube_id,
            layer_id,
            expert_id,
            matrix_name,
            subcube_id,
            ...
        }

    的记录。
    """

    if isinstance(
        obj,
        dict,
    ):

        if looks_like_weight(
            obj
        ):

            output.append(
                obj
            )

            # 已经识别成一个 Weight 后，
            # 不继续递归这个 dict，
            # 避免重复加入。
            return


        for value in obj.values():

            collect_weight_records(
                value,
                output,
            )


    elif isinstance(
        obj,
        list,
    ):

        for item in obj:

            collect_weight_records(
                item,
                output,
            )


# ============================================================
# Matrix Name
# ============================================================


def normalize_matrix_name(
    value: Any,
) -> str:

    name = str(
        value
    )


    if name in (
        "gate",
        "gate_proj",
    ):
        return "gate"


    if name in (
        "up",
        "up_proj",
    ):
        return "up"


    if name in (
        "down",
        "down_proj",
    ):
        return "down"


    return name


# ============================================================
# Normalize Weight
# ============================================================


def normalize_weight(
    raw: dict[str, Any],
) -> WeightLocation:

    cube_id = int(
        first_value(
            raw,
            "cube_id",
            "weight_cube_id",
        )
    )


    layer_id = int(
        first_value(
            raw,
            "layer_id",
        )
    )


    expert_id = int(
        first_value(
            raw,
            "expert_id",
        )
    )


    matrix_name = (
        normalize_matrix_name(
            first_value(
                raw,
                "matrix_name",
                "matrix",
            )
        )
    )


    subcube_id = int(
        first_value(
            raw,
            "subcube_id",
            "sc_id",
        )
    )


    z_value = first_value(
        raw,
        "z",
        "depth_index",
    )


    if z_value is None:
        z_value = 0


    physical_plane_id = (
        first_value(
            raw,
            "physical_plane_id",
            "plane_id",
        )
    )


    slot_id = first_value(
        raw,
        "slot_id",
    )


    return WeightLocation(
        cube_id=cube_id,

        layer_id=layer_id,

        expert_id=expert_id,

        is_shared=bool(
            raw.get(
                "is_shared",
                expert_id
                == SHARED_EXPERT_ID,
            )
        ),

        matrix_name=matrix_name,

        subcube_id=subcube_id,

        z=int(
            z_value
        ),

        physical_plane_id=(
            None
            if physical_plane_id is None
            else int(
                physical_plane_id
            )
        ),

        slot_id=(
            None
            if slot_id is None
            else int(
                slot_id
            )
        ),
    )


# ============================================================
# Mapping Cache
# ============================================================


class ScheduleMappingIndex:

    def __init__(
        self,
        path: Path,
    ) -> None:

        self.path = (
            path.resolve()
        )


        self.by_key: dict[
            tuple[
                int,
                int,
                str,
            ],
            WeightLocation,
        ] = {}


        self.load()


    def load(
        self,
    ) -> None:

        if not self.path.exists():

            raise RuntimeError(
                "找不到 Mapping 文件："
                f"{self.path}"
            )


        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )


        raw_records: list[
            dict[str, Any]
        ] = []


        collect_weight_records(
            data,
            raw_records,
        )


        if not raw_records:

            raise RuntimeError(
                "Mapping JSON 中没有找到 "
                "Weight-Cube Placement。"
            )


        index: dict[
            tuple[
                int,
                int,
                str,
            ],
            WeightLocation,
        ] = {}


        for raw in raw_records:

            weight = (
                normalize_weight(
                    raw
                )
            )


            if weight.matrix_name not in (
                "gate",
                "up",
                "down",
            ):
                continue


            key = (
                weight.layer_id,
                weight.expert_id,
                weight.matrix_name,
            )


            if key in index:

                old = index[
                    key
                ]


                if (
                    old.cube_id
                    == weight.cube_id
                ):
                    continue


                raise RuntimeError(
                    "发现重复的 "
                    "Layer/Expert/Matrix："
                    f"{key}；"
                    f"cube={old.cube_id}, "
                    f"{weight.cube_id}"
                )


            index[
                key
            ] = weight


        self.by_key = (
            index
        )


    def get(
        self,
        *,
        layer_id: int,
        expert_id: int,
        matrix_name: str,
    ) -> WeightLocation:

        key = (
            layer_id,
            expert_id,
            matrix_name,
        )


        if key not in self.by_key:

            raise RuntimeError(
                "Mapping 中找不到："
                f"Layer={layer_id}, "
                f"Expert={expert_id}, "
                f"Matrix={matrix_name}"
            )


        return self.by_key[
            key
        ]


# ============================================================
# 全局 Mapping
#
# FastAPI 启动时读取一次。
# ============================================================


MAPPING_INDEX = (
    ScheduleMappingIndex(
        DEFAULT_MAPPING_PATH
    )
)


# ============================================================
# 请求检查
# ============================================================


def validate_request(
    request: LayerScheduleRequest,
) -> None:

    if not (
        0
        <= request.layer_id
        < NUM_LAYERS
    ):

        raise ValueError(
            "layer_id 必须位于 0~57。"
        )


    routed = (
        request.routed_expert_ids
    )


    if (
        len(routed)
        != TOP_K
    ):

        raise ValueError(
            "必须提供恰好 8 个 "
            "Routed Expert。"
        )


    if (
        len(set(routed))
        != TOP_K
    ):

        raise ValueError(
            "Top-8 Expert "
            "不能包含重复 ID。"
        )


    for expert_id in routed:

        if not (
            0
            <= expert_id
            < NUM_ROUTED_EXPERTS
        ):

            raise ValueError(
                "Routed Expert ID "
                "必须位于 0~255，"
                f"当前为 {expert_id}。"
            )


# ============================================================
# 构建 27 个任务
# ============================================================


def build_tasks(
    *,
    layer_id: int,
    routed_expert_ids: list[int],
) -> list[RuntimeTask]:

    tasks: list[
        RuntimeTask
    ] = []


    active_experts = [
        (
            expert_id,
            False,
            rank,
        )
        for rank, expert_id
        in enumerate(
            routed_expert_ids,
            start=1,
        )
    ]


    active_experts.append(
        (
            SHARED_EXPERT_ID,
            True,
            TOP_K + 1,
        )
    )


    for (
        expert_id,
        is_shared,
        route_rank,
    ) in active_experts:

        for matrix_name in (
            "gate",
            "up",
            "down",
        ):

            location = (
                MAPPING_INDEX.get(
                    layer_id=layer_id,

                    expert_id=expert_id,

                    matrix_name=(
                        matrix_name
                    ),
                )
            )


            # gate / up 初始 ready。
            #
            # down 后面等依赖完成以后才设置 ready_time。
            initial_ready_time = (
                0
                if matrix_name
                in (
                    "gate",
                    "up",
                )
                else -1
            )


            tasks.append(
                RuntimeTask(
                    task_id=(
                        f"L{layer_id}"
                        f"-E{expert_id}"
                        f"-{matrix_name}"
                    ),

                    cube_id=(
                        location.cube_id
                    ),

                    layer_id=layer_id,

                    expert_id=expert_id,

                    is_shared=(
                        is_shared
                    ),

                    route_rank=(
                        route_rank
                    ),

                    matrix_name=(
                        matrix_name
                    ),

                    subcube_id=(
                        location.subcube_id
                    ),

                    z=(
                        location.z
                    ),

                    physical_plane_id=(
                        location
                        .physical_plane_id
                    ),

                    slot_id=(
                        location.slot_id
                    ),

                    ready_time=(
                        initial_ready_time
                    ),
                )
            )


    return tasks


# ============================================================
# Matrix Priority
# ============================================================


MATRIX_PRIORITY = {
    "gate": 0,
    "up": 1,
    "down": 2,
}


# ============================================================
# Ready 排序
#
# 对同一个 SC 有多个 ready task 时：
#
# 1. ready_time 越早
# 2. 当前已经激活的 cube 优先
# 3. Router rank 越靠前
# 4. gate -> up -> down
# 5. cube_id
# ============================================================


def task_priority(
    task: RuntimeTask,
    active_cube_id: int | None,
) -> tuple:

    same_active_cube = (
        active_cube_id
        is not None
        and
        task.cube_id
        == active_cube_id
    )


    return (
        task.ready_time,

        0
        if same_active_cube
        else 1,

        task.route_rank,

        MATRIX_PRIORITY.get(
            task.matrix_name,
            99,
        ),

        task.cube_id,
    )


# ============================================================
# 根据 gate / up 完成时间释放 down
# ============================================================


def release_down_tasks(
    tasks: list[RuntimeTask],
) -> None:

    by_expert: dict[
        int,
        dict[
            str,
            RuntimeTask,
        ],
    ] = {}


    for task in tasks:

        by_expert.setdefault(
            task.expert_id,
            {},
        )[
            task.matrix_name
        ] = task


    for expert_tasks in (
        by_expert.values()
    ):

        gate = (
            expert_tasks[
                "gate"
            ]
        )


        up = (
            expert_tasks[
                "up"
            ]
        )


        down = (
            expert_tasks[
                "down"
            ]
        )


        if (
            down.ready_time >= 0
        ):
            continue


        if (
            not gate.finished
            or
            not up.finished
        ):
            continue


        assert (
            gate.end_cycle
            is not None
        )


        assert (
            up.end_cycle
            is not None
        )


        down.ready_time = max(
            gate.end_cycle,
            up.end_cycle,
        )


# ============================================================
# 单层事件调度器
# ============================================================


def schedule_layer(
    *,
    layer_id: int,
    routed_expert_ids: list[int],
    charge_initial_activation: bool,
) -> dict[str, Any]:

    tasks = (
        build_tasks(
            layer_id=layer_id,

            routed_expert_ids=(
                routed_expert_ids
            ),
        )
    )


    # ========================================================
    # 当前每个 SC 激活的是哪个 Cube
    # ========================================================

    active_cube_by_sc: list[
        int | None
    ] = [
        None
        for _ in range(
            NUM_SUBCUBES
        )
    ]


    # ========================================================
    # running[sc] = task
    # ========================================================

    running: list[
        RuntimeTask | None
    ] = [
        None
        for _ in range(
            NUM_SUBCUBES
        )
    ]


    current_cycle = 0

    finished_count = 0


    # 防止代码错误造成死循环
    guard = 0


    while (
        finished_count
        < len(tasks)
    ):

        guard += 1


        if guard > 10000:

            raise RuntimeError(
                "Scheduler 超过最大迭代次数，"
                "可能出现死锁。"
            )


        # ====================================================
        # 1. 完成当前 cycle 已经结束的 Task
        # ====================================================

        for sc in range(
            NUM_SUBCUBES
        ):

            task = (
                running[sc]
            )


            if task is None:
                continue


            if (
                task.end_cycle
                != current_cycle
            ):
                continue


            task.finished = True

            running[
                sc
            ] = None

            active_cube_by_sc[
                sc
            ] = task.cube_id

            finished_count += 1


        # ====================================================
        # 2. gate + up 完成以后，
        #    对应 down 才 ready
        # ====================================================

        release_down_tasks(
            tasks
        )


        # ====================================================
        # 3. 所有空闲 SC 尝试启动一个任务
        # ====================================================

        started_something = False


        for sc in range(
            NUM_SUBCUBES
        ):

            if (
                running[sc]
                is not None
            ):
                continue


            ready_tasks = [

                task

                for task
                in tasks

                if (
                    task.subcube_id
                    == sc

                    and
                    not task.started

                    and
                    task.ready_time >= 0

                    and
                    task.ready_time
                    <= current_cycle
                )
            ]


            if not ready_tasks:
                continue


            active_cube = (
                active_cube_by_sc[
                    sc
                ]
            )


            ready_tasks.sort(
                key=lambda task:
                    task_priority(
                        task,
                        active_cube,
                    )
            )


            task = (
                ready_tasks[0]
            )


            # =================================================
            # Switch
            # =================================================

            if active_cube is None:

                switch_cycles = (
                    SWITCH_CYCLES
                    if charge_initial_activation
                    else 0
                )

            elif (
                active_cube
                != task.cube_id
            ):

                switch_cycles = (
                    SWITCH_CYCLES
                )

            else:

                switch_cycles = 0


            task.started = True

            task.start_cycle = (
                current_cycle
            )

            task.switch_cycles = (
                switch_cycles
            )

            task.compute_start_cycle = (
                current_cycle
                + switch_cycles
            )

            task.end_cycle = (
                task.compute_start_cycle
                + COMPUTE_CYCLES
            )


            running[
                sc
            ] = task


            started_something = True


        # ====================================================
        # 所有 Task 已完成
        # ====================================================

        if (
            finished_count
            == len(tasks)
        ):
            break


        # ====================================================
        # 4. 找下一次事件
        #
        # 可能是：
        #
        # - 某 Task 完成
        # - 某 ready_time 到达
        # ====================================================

        next_times: list[
            int
        ] = []


        for task in running:

            if (
                task is not None
                and
                task.end_cycle
                is not None
                and
                task.end_cycle
                > current_cycle
            ):

                next_times.append(
                    task.end_cycle
                )


        for task in tasks:

            if task.started:
                continue


            if (
                task.ready_time
                > current_cycle
            ):

                next_times.append(
                    task.ready_time
                )


        if next_times:

            current_cycle = min(
                next_times
            )

            continue


        # ====================================================
        # 如果刚启动任务，
        # 理论上上面一定有 end_cycle。
        # ====================================================

        if started_something:

            current_cycle += 1

            continue


        # ====================================================
        # 没有 Running、没有 Future Ready、
        # 但又没完成全部 Task：
        #
        # 说明依赖逻辑出错。
        # ====================================================

        unfinished = [

            task.task_id

            for task
            in tasks

            if not task.finished
        ]


        raise RuntimeError(
            "Scheduler Deadlock。"
            "未完成任务："
            f"{unfinished}"
        )


    # ========================================================
    # Layer Cycles
    # ========================================================

    layer_cycles = max(
        (
            task.end_cycle
            or 0
        )
        for task in tasks
    )


    # ========================================================
    # Task JSON
    # ========================================================

    task_json = []


    for task in sorted(
        tasks,
        key=lambda item: (
            item.start_cycle
            if item.start_cycle
            is not None
            else 10**9,

            item.subcube_id,

            item.route_rank,

            MATRIX_PRIORITY.get(
                item.matrix_name,
                99,
            ),
        ),
    ):

        task_json.append(
            {
                "task_id":
                    task.task_id,

                "cube_id":
                    task.cube_id,

                "layer_id":
                    task.layer_id,

                "expert_id":
                    task.expert_id,

                "is_shared":
                    task.is_shared,

                "route_rank":
                    (
                        None
                        if task.is_shared
                        else task.route_rank
                    ),

                "matrix_name":
                    task.matrix_name,

                "subcube_id":
                    task.subcube_id,

                "z":
                    task.z,

                "physical_plane_id":
                    task.physical_plane_id,

                "slot_id":
                    task.slot_id,

                "ready_time":
                    task.ready_time,

                "start_cycle":
                    task.start_cycle,

                "switch_cycles":
                    task.switch_cycles,

                "switched":
                    task.switch_cycles > 0,

                "compute_start_cycle":
                    task.compute_start_cycle,

                "end_cycle":
                    task.end_cycle,

                "duration":
                    (
                        (
                            task.end_cycle
                            - task.start_cycle
                        )
                        if (
                            task.end_cycle
                            is not None
                            and
                            task.start_cycle
                            is not None
                        )
                        else None
                    ),
            }
        )


    # ========================================================
    # 每个 SC 统计
    # ========================================================

    subcube_stats = []


    for sc in range(
        NUM_SUBCUBES
    ):

        sc_tasks = [

            item

            for item
            in task_json

            if (
                item[
                    "subcube_id"
                ]
                == sc
            )
        ]


        switch_count = sum(
            1
            for task
            in sc_tasks
            if task[
                "switched"
            ]
        )


        switch_cycle_count = sum(
            task[
                "switch_cycles"
            ]
            for task
            in sc_tasks
        )


        compute_cycle_count = (
            len(sc_tasks)
            * COMPUTE_CYCLES
        )


        busy_cycles = (
            switch_cycle_count
            + compute_cycle_count
        )


        finish_cycle = max(
            (
                task[
                    "end_cycle"
                ]
                or 0
            )
            for task
            in sc_tasks
        ) if sc_tasks else 0


        subcube_stats.append(
            {
                "subcube_id":
                    sc,

                "task_count":
                    len(
                        sc_tasks
                    ),

                "switch_count":
                    switch_count,

                "switch_cycles":
                    switch_cycle_count,

                "compute_cycles":
                    compute_cycle_count,

                "busy_cycles":
                    busy_cycles,

                "finish_cycle":
                    finish_cycle,

                "utilization":
                    (
                        busy_cycles
                        / layer_cycles
                        if layer_cycles > 0
                        else 0.0
                    ),

                "tasks":
                    sc_tasks,
            }
        )


    # ========================================================
    # Critical SC
    #
    # 最晚结束的 SC。
    # 可能不止一个。
    # ========================================================

    critical_subcubes = [

        stat[
            "subcube_id"
        ]

        for stat
        in subcube_stats

        if (
            stat[
                "finish_cycle"
            ]
            == layer_cycles

            and
            stat[
                "task_count"
            ]
            > 0
        )
    ]


    active_sc_count = sum(
        1
        for stat
        in subcube_stats
        if stat[
            "task_count"
        ]
        > 0
    )


    # ========================================================
    # 最终返回
    # ========================================================

    return {
        "layer_id":
            layer_id,

        "routed_expert_ids":
            list(
                routed_expert_ids
            ),

        "shared_expert_id":
            SHARED_EXPERT_ID,

        "active_expert_count":
            TOP_K + 1,

        "task_count":
            len(tasks),

        "active_subcube_count":
            active_sc_count,

        "layer_cycles":
            layer_cycles,

        "critical_subcubes":
            critical_subcubes,

        "rules": {
            "switch_cycles":
                SWITCH_CYCLES,

            "compute_cycles":
                COMPUTE_CYCLES,

            "cross_subcube_cycles":
                0,

            "charge_initial_activation":
                charge_initial_activation,
        },

        "tasks":
            task_json,

        "subcubes":
            subcube_stats,
    }


# ============================================================
# POST /api/schedule/layer
# ============================================================


@router.post(
    "/layer"
)
def schedule_one_layer(
    request: LayerScheduleRequest,
):

    try:

        validate_request(
            request
        )


        result = (
            schedule_layer(
                layer_id=(
                    request.layer_id
                ),

                routed_expert_ids=(
                    request
                    .routed_expert_ids
                ),

                charge_initial_activation=(
                    request
                    .charge_initial_activation
                ),
            )
        )


        return result


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
# GET /api/schedule/health
# ============================================================


@router.get(
    "/health"
)
def schedule_health():

    return {
        "status":
            "ok",

        "mapping_file":
            DEFAULT_MAPPING_PATH.name,

        "indexed_weight_count":
            len(
                MAPPING_INDEX.by_key
            ),

        "switch_cycles":
            SWITCH_CYCLES,

        "compute_cycles":
            COMPUTE_CYCLES,

        "shared_expert_id":
            SHARED_EXPERT_ID,
    }


# ============================================================
# 命令行自测
#
# python -m webui.backend.schedule_api
# ============================================================


def main() -> None:

    print(
        "========== "
        "Schedule API Self Test "
        "=========="
    )


    print(
        "Mapping:",
        DEFAULT_MAPPING_PATH,
    )


    print(
        "Indexed weights:",
        len(
            MAPPING_INDEX.by_key
        ),
    )


    example_route = [
        102,
        125,
        149,
        186,
        206,
        215,
        217,
        140,
    ]


    print(
        "\nExample:"
    )


    print(
        "Layer:",
        0,
    )


    print(
        "Route:",
        example_route,
    )


    result = (
        schedule_layer(
            layer_id=0,

            routed_expert_ids=(
                example_route
            ),

            charge_initial_activation=True,
        )
    )


    print(
        "\nLayer cycles:",
        result[
            "layer_cycles"
        ],
    )


    print(
        "Active SC:",
        result[
            "active_subcube_count"
        ],
    )


    print(
        "Critical SC:",
        result[
            "critical_subcubes"
        ],
    )


    print(
        "\nTasks:"
    )


    for task in result[
        "tasks"
    ]:

        print(
            "  "
            f"SC{task['subcube_id']:>2} "
            f"E{task['expert_id']:>3} "
            f"{task['matrix_name']:<4} "
            f"ready={task['ready_time']:<2} "
            f"start={task['start_cycle']:<2} "
            f"compute={task['compute_start_cycle']:<2} "
            f"end={task['end_cycle']:<2} "
            f"switch={task['switch_cycles']}"
        )


    print(
        "\nPASS"
    )


if __name__ == "__main__":

    main()