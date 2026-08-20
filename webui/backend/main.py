"""
MoE PIM WebUI Backend

第一版后端接口。

作用：

1. 读取第四步生成的最终 Mapping JSON；
2. 建立简单的运行时查询索引；
3. 给前端提供 REST API；
4. 暂时不负责 Token 调度模拟；
5. 后续再接 scheduling/ 中现有的调度器。

默认读取：

results/mappings/
mapping_baseline_N4_H7168_W4096.json

运行：

python -m uvicorn webui.backend.main:app --reload --port 8000

浏览器：

http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import re

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


from fastapi import (
    FastAPI,
    HTTPException,
    Query,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from webui.backend.trace_api import router as trace_router

from webui.backend.schedule_api import router as schedule_router

from webui.backend.token_schedule_api import router as token_schedule_router

from webui.backend.workload_api import router as workload_router

from webui.backend.request_api import router as request_router
# ============================================================
# 路径
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


DEFAULT_PHASE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "phase_evaluation_summary.json"
)


# ============================================================
# Phase Evaluation Summary
# ============================================================


def load_phase_summary() -> dict[str, Any]:
    """
    读取 Prefill / Decode 正式阶段评估汇总。

    这里每次请求都重新读取 JSON，便于重新运行评估后
    WebUI 无需重启后端即可看到最新结果。
    """

    if not DEFAULT_PHASE_SUMMARY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "找不到阶段评估文件："
                f"{DEFAULT_PHASE_SUMMARY_PATH}"
            ),
        )

    try:
        with DEFAULT_PHASE_SUMMARY_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw = json.load(file)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail="phase_evaluation_summary.json 不是合法 JSON。",
        ) from exc

    if not isinstance(raw, dict):
        raise HTTPException(
            status_code=500,
            detail="阶段评估 JSON 最外层必须是 dict。",
        )

    prefill = raw.get("prefill")
    decode = raw.get("decode")

    if not isinstance(prefill, dict) or not isinstance(decode, dict):
        raise HTTPException(
            status_code=500,
            detail="阶段评估 JSON 缺少 prefill 或 decode。",
        )

    # 不把 sources 中的本机绝对路径暴露给前端。
    return {
        "summary_version": raw.get("summary_version"),
        "scope": raw.get("scope", ""),
        "prefill": prefill,
        "decode": decode,
    }


# ============================================================
# FastAPI
# ============================================================


app = FastAPI(
    title="MoE PIM Simulator Web API",
    version="0.1.0",
    description=(
        "为 MoE PIM Cube 可视化和 Token "
        "推理模拟提供后端数据。"
    ),
)


# ============================================================
# CORS
#
# Vite 默认：
#     localhost:5173
# ============================================================


app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# Trace API
# ============================================================

app.include_router(
    trace_router
)
# ============================================================
# Schedule API
# ============================================================

app.include_router(
    schedule_router
)
# ============================================================
# 异常
# ============================================================

# ============================================================
# Full Token Schedule API
# ============================================================

app.include_router(
    token_schedule_router
)
class MappingDataError(
    RuntimeError
):
    """Mapping JSON 读取或解析错误。"""

# ============================================================
# Multi-Token Workload API
# ============================================================

app.include_router(
    workload_router
)

# ============================================================
# Stage-aware Request API
# ============================================================

app.include_router(
    request_router
)
# ============================================================
# JSON Helper
# ============================================================


def _nested_get(
    data: dict[str, Any],
    path: tuple[str, ...],
) -> Any:
    """
    按：

        ("binding", "placements")

    这种路径读取嵌套字段。

    不存在时返回 None。
    """

    current: Any = data

    for key in path:

        if not isinstance(
            current,
            dict,
        ):
            return None

        if key not in current:
            return None

        current = current[
            key
        ]

    return current


def _first_existing(
    data: dict[str, Any],
    paths: tuple[
        tuple[str, ...],
        ...
    ],
) -> Any:
    """
    尝试多个可能的 JSON 路径。

    因为后续 mapping JSON
    结构可能会继续微调。
    """

    for path in paths:

        value = _nested_get(
            data,
            path,
        )

        if value is not None:
            return value

    return None


# ============================================================
# Placement 读取
# ============================================================


def _find_placements(
    raw: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    找到最终 Weight-Cube Placement 列表。

    当前兼容几种可能的保存结构。
    """

    candidates = (
        ("placements",),

        (
            "weight_cube_placements",
        ),

        (
            "binding",
            "placements",
        ),

        (
            "physical_binding",
            "placements",
        ),

        (
            "mapping",
            "placements",
        ),

        (
            "result",
            "placements",
        ),
    )

    placements = (
        _first_existing(
            raw,
            candidates,
        )
    )

    if placements is None:

        raise MappingDataError(
            "在 Mapping JSON 中没有找到 "
            "placements。\n"
            "请检查第四步保存文件结构。"
        )

    if not isinstance(
        placements,
        list,
    ):

        raise MappingDataError(
            "Mapping JSON 中 placements "
            "必须是 list。"
        )

    if not placements:

        raise MappingDataError(
            "Mapping JSON 中 placements 为空。"
        )

    for index, item in enumerate(
        placements
    ):

        if not isinstance(
            item,
            dict,
        ):

            raise MappingDataError(
                f"placements[{index}] "
                "不是 dict。"
            )

    return placements


# ============================================================
# 字段读取
# ============================================================


def _int_field(
    item: dict[str, Any],
    *names: str,
    default: int | None = None,
) -> int | None:
    """
    从多个候选字段名中读取 int。
    """

    for name in names:

        if (
            name in item
            and
            item[
                name
            ] is not None
        ):

            return int(
                item[
                    name
                ]
            )

    return default


def _bool_field(
    item: dict[str, Any],
    *names: str,
    default: bool = False,
) -> bool:

    for name in names:

        if name in item:

            return bool(
                item[
                    name
                ]
            )

    return default


def _str_field(
    item: dict[str, Any],
    *names: str,
    default: str = "",
) -> str:

    for name in names:

        if (
            name in item
            and
            item[
                name
            ] is not None
        ):

            return str(
                item[
                    name
                ]
            )

    return default


# ============================================================
# 标准化 Placement
# ============================================================


def normalize_placement(
    item: dict[str, Any],
) -> dict[str, Any]:
    """
    将第四步的 Placement 统一整理成
    WebUI 使用的字段。

    Web 前端以后只依赖这个格式，
    不直接依赖原始 Mapping JSON。
    """

    cube_id = _int_field(
        item,
        "cube_id",
        "weight_cube_id",
    )

    layer_id = _int_field(
        item,
        "layer_id",
    )

    expert_id = _int_field(
        item,
        "expert_id",
    )

    subcube_id = _int_field(
        item,
        "subcube_id",
        "sc_id",
    )

    z = _int_field(
        item,
        "z",
        "depth_index",
    )

    if cube_id is None:
        raise MappingDataError(
            "Placement 缺少 cube_id。"
        )

    if layer_id is None:
        raise MappingDataError(
            f"Cube-{cube_id} "
            "缺少 layer_id。"
        )

    if expert_id is None:
        raise MappingDataError(
            f"Cube-{cube_id} "
            "缺少 expert_id。"
        )

    if subcube_id is None:
        raise MappingDataError(
            f"Cube-{cube_id} "
            "缺少 subcube_id。"
        )

    if z is None:
        raise MappingDataError(
            f"Cube-{cube_id} "
            "缺少 z。"
        )

    matrix_name = _str_field(
        item,
        "matrix_name",
        "matrix",
    )

    if not matrix_name:

        raise MappingDataError(
            f"Cube-{cube_id} "
            "缺少 matrix_name。"
        )

    return {
        # ====================================================
        # 逻辑身份
        # ====================================================

        "cube_id": cube_id,

        "layer_id": layer_id,

        "expert_id": expert_id,

        "is_shared": _bool_field(
            item,
            "is_shared",
            default=False,
        ),

        "matrix_name": matrix_name,

        # ====================================================
        # Logical Plane
        # ====================================================

        "logical_plane_id": (
            _int_field(
                item,
                "logical_plane_id",
            )
        ),

        # ====================================================
        # Physical
        # ====================================================

        "subcube_id": subcube_id,

        "z": z,

        "physical_plane_id": (
            _int_field(
                item,
                "physical_plane_id",
                "plane_id",
            )
        ),

        "slot_id": (
            _int_field(
                item,
                "slot_id",
            )
        ),

        # ====================================================
        # Position
        # ====================================================

        "x": _int_field(
            item,
            "x",
            default=0,
        ),

        "y": _int_field(
            item,
            "y",
            default=0,
        ),

        # ====================================================
        # Shape
        # ====================================================

        "logical_rows": (
            _int_field(
                item,
                "logical_rows",
                "rows",
            )
        ),

        "logical_cols": (
            _int_field(
                item,
                "logical_cols",
                "cols",
            )
        ),

        "slot_rows": (
            _int_field(
                item,
                "slot_rows",
                "placed_rows",
            )
        ),

        "slot_cols": (
            _int_field(
                item,
                "slot_cols",
                "placed_cols",
            )
        ),

        # ====================================================
        # Rotation
        # ====================================================

        "rotated": _bool_field(
            item,
            "rotated",
            "logical_rotation_required",
            default=False,
        ),
    }


# ============================================================
# Mapping Store
# ============================================================

def parse_hardware_from_mapping_filename(
    path: Path,
) -> dict[str, int]:
    """
    从 Mapping 文件名读取：

        mapping_baseline_N4_H7168_W4096.json

    得到：

        N = 4
        H = 7168
        W = 4096

    这是备用方案。

    正常情况下优先读取 JSON，
    JSON 没保存 H/W 时再使用文件名。
    """

    pattern = re.compile(
        r"N(?P<N>\d+)_"
        r"H(?P<H>\d+)_"
        r"W(?P<W>\d+)"
    )

    match = pattern.search(
        path.stem
    )

    if match is None:
        return {}

    return {
        "N": int(
            match.group("N")
        ),
        "H": int(
            match.group("H")
        ),
        "W": int(
            match.group("W")
        ),
    }


class MappingStore:
    """
    WebUI 使用的 Mapping 内存索引。

    加载一次 JSON 后建立：

        cube_id
            -> Weight

        subcube_id
            -> Weight[]

        (subcube_id, z)
            -> Weight[]

        layer_id
            -> Weight[]

    前端查询时不用每次扫描 44718 个矩阵。
    """

    def __init__(
        self,
        mapping_path: Path,
    ) -> None:

        self.mapping_path = (
            mapping_path
        )

        self.raw: dict[
            str,
            Any
        ] = {}

        self.placements: tuple[
            dict[str, Any],
            ...
        ] = ()

        self.by_cube: dict[
            int,
            dict[str, Any],
        ] = {}

        self.by_subcube: dict[
            int,
            list[
                dict[str, Any]
            ],
        ] = defaultdict(
            list
        )

        self.by_subcube_z: dict[
            tuple[int, int],
            list[
                dict[str, Any]
            ],
        ] = defaultdict(
            list
        )

        self.by_layer: dict[
            int,
            list[
                dict[str, Any]
            ],
        ] = defaultdict(
            list
        )

        self.load()

    # ========================================================
    # Load
    # ========================================================

    def load(
        self,
    ) -> None:

        if not (
            self.mapping_path.exists()
        ):

            raise MappingDataError(
                "找不到 Mapping 文件：\n"
                f"{self.mapping_path}"
            )

        with self.mapping_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            raw = json.load(
                file
            )

        if not isinstance(
            raw,
            dict,
        ):

            raise MappingDataError(
                "Mapping JSON 最外层必须是 dict。"
            )

        raw_placements = (
            _find_placements(
                raw
            )
        )

        normalized = tuple(
            normalize_placement(
                item
            )

            for item
            in raw_placements
        )

        # ====================================================
        # Reset
        # ====================================================

        self.raw = raw

        self.placements = normalized

        self.by_cube = {}

        self.by_subcube = defaultdict(
            list
        )

        self.by_subcube_z = defaultdict(
            list
        )

        self.by_layer = defaultdict(
            list
        )

        # ====================================================
        # Index
        # ====================================================

        for weight in normalized:

            cube_id = (
                weight[
                    "cube_id"
                ]
            )

            if cube_id in self.by_cube:

                raise MappingDataError(
                    f"重复 cube_id："
                    f"{cube_id}"
                )

            self.by_cube[
                cube_id
            ] = weight

            sc = weight[
                "subcube_id"
            ]

            z = weight[
                "z"
            ]

            layer_id = weight[
                "layer_id"
            ]

            self.by_subcube[
                sc
            ].append(
                weight
            )

            self.by_subcube_z[
                (
                    sc,
                    z,
                )
            ].append(
                weight
            )

            self.by_layer[
                layer_id
            ].append(
                weight
            )

        # ====================================================
        # Sort
        # ====================================================

        for values in (
            self.by_subcube.values()
        ):

            values.sort(
                key=lambda item: (
                    item[
                        "z"
                    ],
                    item[
                        "slot_id"
                    ]
                    if (
                        item[
                            "slot_id"
                        ]
                        is not None
                    )
                    else -1,
                )
            )

        for values in (
            self.by_layer.values()
        ):

            values.sort(
                key=lambda item: (
                    item[
                        "subcube_id"
                    ],
                    item[
                        "z"
                    ],
                    item[
                        "expert_id"
                    ],
                    item[
                        "matrix_name"
                    ],
                )
            )

    # ========================================================
    # Hardware Summary
    # ========================================================

    def hardware_summary(
        self,
    ) -> dict[str, Any]:

        hardware = (
            _first_existing(
                self.raw,

                (
                    (
                        "hardware",
                    ),

                    (
                        "subcube_mapping",
                        "hardware",
                    ),

                    (
                        "mapping",
                        "hardware",
                    ),
                ),
            )
        )

        if not isinstance(
            hardware,
            dict,
        ):

            hardware = {}

        subcube_ids = sorted(
            self.by_subcube
        )

        num_subcubes = (
            _int_field(
                hardware,
                "num_subcubes",
            )
        )

        if num_subcubes is None:

            num_subcubes = (
                max(
                    subcube_ids
                )
                + 1
                if subcube_ids
                else 0
            )

        filename_hardware = (
            parse_hardware_from_mapping_filename(
                self.mapping_path
            )
        )


        N = _int_field(
            hardware,
            "N",
        )

        if N is None:

            N = filename_hardware.get(
                "N"
            )


        H = _int_field(
            hardware,
            "H",
        )

        if H is None:

            H = filename_hardware.get(
                "H"
            )


        W = _int_field(
            hardware,
            "W",
        )

        if W is None:

            W = filename_hardware.get(
                "W"
            )

        D = _int_field(
            hardware,
            "D",
        )

        if D is None:

            max_z = max(
                (
                    weight[
                        "z"
                    ]

                    for weight
                    in self.placements
                ),

                default=-1,
            )

            D = max_z + 1

        used_planes = len(
            {
                (
                    weight[
                        "subcube_id"
                    ],
                    weight[
                        "z"
                    ],
                )

                for weight
                in self.placements
            }
        )

        total_plane_slots = (
            num_subcubes * D
            if (
                num_subcubes
                and D
            )
            else None
        )

        empty_planes = (
            total_plane_slots
            - used_planes

            if (
                total_plane_slots
                is not None
            )

            else None
        )

        return {
            "N": N,

            "H": H,

            "W": W,

            "D": D,

            "num_subcubes": (
                num_subcubes
            ),

            "used_planes": (
                used_planes
            ),

            "total_plane_slots": (
                total_plane_slots
            ),

            "empty_plane_slots": (
                empty_planes
            ),

            "weight_cube_count": (
                len(
                    self.placements
                )
            ),

            "layer_count": (
                len(
                    self.by_layer
                )
            ),
        }

    # ========================================================
    # SC Summary
    # ========================================================

    def subcube_summary(
        self,
        subcube_id: int,
    ) -> dict[str, Any]:

        weights = (
            self.by_subcube.get(
                subcube_id,
                []
            )
        )

        plane_ids = {
            weight[
                "z"
            ]

            for weight
            in weights
        }

        matrix_counts = Counter(
            weight[
                "matrix_name"
            ]

            for weight
            in weights
        )

        shared_count = sum(
            1

            for weight
            in weights

            if weight[
                "is_shared"
            ]
        )

        hardware = (
            self.hardware_summary()
        )

        D = (
            hardware[
                "D"
            ]
            or 0
        )

        used_planes = len(
            plane_ids
        )

        return {
            "subcube_id": (
                subcube_id
            ),

            "used_planes": (
                used_planes
            ),

            "depth_capacity": D,

            "empty_planes": (
                max(
                    D
                    - used_planes,
                    0,
                )
            ),

            "weight_cube_count": (
                len(
                    weights
                )
            ),

            "shared_weight_count": (
                shared_count
            ),

            "matrix_counts": (
                dict(
                    matrix_counts
                )
            ),
        }


# ============================================================
# 全局 Store
# ============================================================


store = MappingStore(
    DEFAULT_MAPPING_PATH
)


# ============================================================
# API
# ============================================================


@app.get(
    "/api/health"
)
def health() -> dict[str, Any]:

    return {
        "status": "ok",

        "mapping_file": (
            store.mapping_path.name
        ),
    }


# ============================================================
# Prefill / Decode 正式阶段评估
# ============================================================


@app.get(
    "/api/phase-summary"
)
def phase_summary() -> dict[str, Any]:
    return load_phase_summary()


# ============================================================
# 系统概况
# ============================================================


@app.get(
    "/api/system/summary"
)
def system_summary() -> dict[str, Any]:

    return {
        "mapping_file": (
            store.mapping_path.name
        ),

        "hardware": (
            store.hardware_summary()
        ),
    }


# ============================================================
# 16 个 Sub-Cube
# ============================================================


@app.get(
    "/api/subcubes"
)
def get_subcubes() -> dict[str, Any]:

    hardware = (
        store.hardware_summary()
    )

    num_subcubes = (
        hardware[
            "num_subcubes"
        ]
    )

    items = [
        store.subcube_summary(
            sc
        )

        for sc in range(
            num_subcubes
        )
    ]

    return {
        "count": (
            len(
                items
            )
        ),

        "items": items,
    }


# ============================================================
# 某个 SC 的 Plane
# ============================================================


@app.get(
    "/api/subcubes/{subcube_id}/planes"
)
def get_subcube_planes(
    subcube_id: int,

    layer_id: int | None = Query(
        default=None
    ),
) -> dict[str, Any]:

    hardware = (
        store.hardware_summary()
    )

    if not (
        0
        <= subcube_id
        < hardware[
            "num_subcubes"
        ]
    ):

        raise HTTPException(
            status_code=404,

            detail=(
                f"Sub-Cube "
                f"{subcube_id} "
                "不存在。"
            ),
        )

    plane_map: dict[
        int,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for weight in (
        store.by_subcube.get(
            subcube_id,
            []
        )
    ):

        if (
            layer_id is not None
            and
            weight[
                "layer_id"
            ]
            != layer_id
        ):
            continue

        plane_map[
            weight[
                "z"
            ]
        ].append(
            weight
        )

    items = []

    for z in sorted(
        plane_map
    ):

        weights = (
            plane_map[
                z
            ]
        )

        items.append(
            {
                "subcube_id": (
                    subcube_id
                ),

                "z": z,

                "weight_count": (
                    len(
                        weights
                    )
                ),

                "weights": weights,
            }
        )

    return {
        "subcube_id": (
            subcube_id
        ),

        "layer_filter": (
            layer_id
        ),

        "plane_count": (
            len(
                items
            )
        ),

        "items": items,
    }


# ============================================================
# 精确查看某个 Plane
# ============================================================


@app.get(
    "/api/subcubes/{subcube_id}/planes/{z}"
)
def get_plane(
    subcube_id: int,
    z: int,
) -> dict[str, Any]:

    hardware = (
        store.hardware_summary()
    )

    if not (
        0
        <= subcube_id
        < hardware[
            "num_subcubes"
        ]
    ):

        raise HTTPException(
            status_code=404,

            detail="Sub-Cube 不存在。",
        )

    D = hardware[
        "D"
    ]

    if (
        D is not None
        and
        not (
            0 <= z < D
        )
    ):

        raise HTTPException(
            status_code=404,

            detail=(
                f"z={z} 超出 "
                f"[0,{D - 1}]。"
            ),
        )

    weights = (
        store.by_subcube_z.get(
            (
                subcube_id,
                z,
            ),
            [],
        )
    )

    return {
        "subcube_id": (
            subcube_id
        ),

        "z": z,

        "empty": (
            len(
                weights
            )
            == 0
        ),

        "weight_count": (
            len(
                weights
            )
        ),

        "weights": weights,
    }


# ============================================================
# 查看某个模型 Layer
# ============================================================


@app.get(
    "/api/layers/{layer_id}"
)
def get_layer(
    layer_id: int,
) -> dict[str, Any]:

    if layer_id not in (
        store.by_layer
    ):

        raise HTTPException(
            status_code=404,

            detail=(
                f"Model Layer "
                f"{layer_id} "
                "不存在。"
            ),
        )

    weights = (
        store.by_layer[
            layer_id
        ]
    )

    per_subcube: dict[
        int,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for weight in weights:

        per_subcube[
            weight[
                "subcube_id"
            ]
        ].append(
            weight
        )

    subcubes = []

    for sc in sorted(
        per_subcube
    ):

        sc_weights = (
            per_subcube[
                sc
            ]
        )

        matrix_counts = Counter(
            weight[
                "matrix_name"
            ]

            for weight
            in sc_weights
        )

        subcubes.append(
            {
                "subcube_id": sc,

                "weight_count": (
                    len(
                        sc_weights
                    )
                ),

                "matrix_counts": (
                    dict(
                        matrix_counts
                    )
                ),

                "weights": (
                    sc_weights
                ),
            }
        )

    return {
        "layer_id": (
            layer_id
        ),

        "weight_count": (
            len(
                weights
            )
        ),

        "active_subcube_count": (
            len(
                subcubes
            )
        ),

        "subcubes": subcubes,
    }


# ============================================================
# 查看某个 Weight-Cube
# ============================================================


@app.get(
    "/api/weights/{cube_id}"
)
def get_weight(
    cube_id: int,
) -> dict[str, Any]:

    weight = (
        store.by_cube.get(
            cube_id
        )
    )

    if weight is None:

        raise HTTPException(
            status_code=404,

            detail=(
                f"Cube-{cube_id} "
                "不存在。"
            ),
        )

    return weight


# ============================================================
# Reload
#
# 调试时 Mapping JSON 重跑以后，
# 不用重启整个 Server。
# ============================================================


@app.post(
    "/api/reload"
)
def reload_mapping() -> dict[str, Any]:

    try:

        store.load()

    except Exception as exc:

        raise HTTPException(
            status_code=500,

            detail=str(
                exc
            ),
        ) from exc

    return {
        "status": "reloaded",

        "mapping_file": (
            store.mapping_path.name
        ),

        "hardware": (
            store.hardware_summary()
        ),
    }