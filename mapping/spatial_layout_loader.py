"""
第四步：读取前三步已经保存的匿名空间布局。

前三步 run_spatial_baseline.py 已经保存：

    results/
    ├── spatial_candidates.json
    └── layouts/
        └── <layout_id>.json

其中：

spatial_candidates.json：

    保存不同 N、H、W、D 的空间候选。

layouts/<layout_id>.json：

    保存匿名二维装箱结果：

        Plane
        PhysicalSlot
        x
        y
        slot_rows
        slot_cols
        orientation_swapped

本文件负责：

    JSON
        ↓
    ResolvedHardwareConfig
        +
    tuple[Plane, ...]

然后交给第四步：

    physical_binder.py

------------------------------------------------------------

重要：

本文件不会：

    重新运行 MaxRects
    修改 x/y
    修改 Plane
    修改 Slot 尺寸
    分配 layer/expert/matrix_name

只是把第三步已经保存的结果恢复成 Python 对象。

------------------------------------------------------------

当前第四步 Baseline 还有一个重要要求：

LogicalWeightCube 当前一个原始矩阵对应一个 Weight-Cube。

因此当前正式使用的空间模板必须满足：

    template.chunk_count == 1

也就是：

    一个 7168×2048 矩阵
        ->
    一个匿名 PhysicalSlot

当前 H=7168、W=4096 的最优方案正好满足这个要求。

以后如果要支持：

    一个矩阵切成多个 Weight-Cube

再扩展 LogicalWeightCube 的 chunk 身份即可。
"""

from __future__ import annotations

import argparse
import json

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any


from evaluation.hardware_resolver import (
    ResolvedHardwareConfig,
)

from packing.physical_slot import (
    PhysicalSlot,
)

from packing.plane import (
    Plane,
    create_empty_plane,
)


# ============================================================
# 默认路径
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


DEFAULT_SUMMARY_PATH = (
    DEFAULT_RESULTS_DIR
    / "spatial_candidates.json"
)


# ============================================================
# 异常
# ============================================================


class SpatialLayoutLoadError(ValueError):
    """读取前三步匿名布局失败。"""


# ============================================================
# 最终加载结果
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class SpatialLayoutBundle:
    """
    第三步空间结果恢复后的完整对象。

    后续第四步可以直接使用：

        bundle.hardware
        bundle.physical_planes
    """

    # ========================================================
    # Candidate 身份
    # ========================================================

    layout_id: str

    template_id: str

    spatial_rank: int

    # ========================================================
    # Hardware
    # ========================================================

    hardware: (
        ResolvedHardwareConfig
    )

    # ========================================================
    # Layout 基础信息
    # ========================================================

    matrix_count: int

    plane_count: int

    slot_count: int

    template_chunk_count: int

    # ========================================================
    # 第三步匿名物理布局
    # ========================================================

    physical_planes: tuple[
        Plane,
        ...
    ]

    # ========================================================
    # 常用属性
    # ========================================================

    @property
    def physical_slots(
        self,
    ) -> tuple[
        PhysicalSlot,
        ...
    ]:
        """
        返回全部 PhysicalSlot。
        """

        return tuple(
            slot
            for plane
            in self.physical_planes
            for slot
            in plane.slots
        )

    @property
    def empty_plane_slots(
        self,
    ) -> int:
        """
        Q - P
        """

        return (
            self.hardware
            .total_plane_slots
            - self.plane_count
        )


# ============================================================
# JSON 读取
# ============================================================


def _load_json(
    path: Path,
) -> Any:
    """
    安全读取 JSON。
    """

    if not path.exists():

        raise SpatialLayoutLoadError(
            f"文件不存在：{path}"
        )

    if not path.is_file():

        raise SpatialLayoutLoadError(
            f"路径不是文件：{path}"
        )

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(
                file
            )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise SpatialLayoutLoadError(
            f"无法读取 JSON：{path}"
        ) from exc


# ============================================================
# Candidate 读取
# ============================================================


def load_spatial_candidates(
    summary_path: Path | str = (
        DEFAULT_SUMMARY_PATH
    ),
) -> tuple[
    dict[str, Any],
    ...
]:
    """
    读取：

        results/spatial_candidates.json

    返回全部 candidate。
    """

    path = Path(
        summary_path
    ).resolve()

    data = _load_json(
        path
    )

    if not isinstance(
        data,
        dict,
    ):

        raise SpatialLayoutLoadError(
            "spatial_candidates.json "
            "最外层必须是 dict。"
        )

    candidates = data.get(
        "candidates"
    )

    if not isinstance(
        candidates,
        list,
    ):

        raise SpatialLayoutLoadError(
            "spatial_candidates.json "
            "缺少 candidates list。"
        )

    if not candidates:

        raise SpatialLayoutLoadError(
            "spatial_candidates.json "
            "中没有 Candidate。"
        )

    result = []

    required_fields = {
        "layout_id",
        "template_id",

        "N",
        "H",
        "W",
        "D",

        "num_subcubes",

        "P",
        "Q",

        "valid",
        "spatial_rank",
    }

    for index, candidate in enumerate(
        candidates
    ):

        if not isinstance(
            candidate,
            dict,
        ):

            raise SpatialLayoutLoadError(
                f"Candidate-{index} "
                "必须是 dict。"
            )

        missing = (
            required_fields
            - set(candidate)
        )

        if missing:

            raise SpatialLayoutLoadError(
                f"Candidate-{index} "
                "缺少字段："
                f"{sorted(missing)}。"
            )

        result.append(
            candidate
        )

    return tuple(
        result
    )


# ============================================================
# 选择空间 Candidate
# ============================================================


def select_spatial_candidate(
    candidates: tuple[
        dict[str, Any],
        ...
    ],
    *,
    spatial_rank: int | None = None,
    N: int | None = None,
    H: int | None = None,
    W: int | None = None,
    layout_id: str | None = None,
    require_valid: bool = True,
) -> dict[str, Any]:
    """
    从 spatial_candidates.json 中选一套硬件。

    可以通过：

        spatial_rank
        N
        H
        W
        layout_id

    任意组合筛选。

    如果所有条件都不指定：

        自动选择 spatial_rank 最小的合法方案。

    --------------------------------------------------------

    例如当前方案可以：

        N=4
        H=7168
        W=4096

    而 layout_id 不需要手工知道。
    """

    filtered = []

    for candidate in candidates:

        if (
            require_valid
            and not candidate["valid"]
        ):
            continue

        if (
            spatial_rank is not None
            and candidate["spatial_rank"]
            != spatial_rank
        ):
            continue

        if (
            N is not None
            and candidate["N"] != N
        ):
            continue

        if (
            H is not None
            and candidate["H"] != H
        ):
            continue

        if (
            W is not None
            and candidate["W"] != W
        ):
            continue

        if (
            layout_id is not None
            and candidate["layout_id"]
            != layout_id
        ):
            continue

        filtered.append(
            candidate
        )

    if not filtered:

        raise SpatialLayoutLoadError(
            "没有找到满足条件的空间 Candidate："
            f"rank={spatial_rank}, "
            f"N={N}, "
            f"H={H}, "
            f"W={W}, "
            f"layout_id={layout_id}。"
        )

    # ========================================================
    # 如果筛出多个：
    #
    # spatial_rank 最小者优先。
    # ========================================================

    filtered.sort(
        key=lambda candidate: (
            candidate[
                "spatial_rank"
            ],
            candidate["N"],
            candidate["H"],
            candidate["W"],
            candidate["D"],
            candidate["layout_id"],
        )
    )

    return filtered[0]


# ============================================================
# Candidate -> Hardware
# ============================================================


def candidate_to_hardware(
    candidate: dict[
        str,
        Any,
    ],
) -> ResolvedHardwareConfig:
    """
    从 spatial candidate 恢复：

        ResolvedHardwareConfig
    """

    hardware = (
        ResolvedHardwareConfig(
            N=int(
                candidate["N"]
            ),

            H=int(
                candidate["H"]
            ),

            W=int(
                candidate["W"]
            ),

            D=int(
                candidate["D"]
            ),
        )
    )

    P = int(
        candidate["P"]
    )

    Q = int(
        candidate["Q"]
    )

    # ========================================================
    # 检查 N²
    # ========================================================

    if (
        int(
            candidate[
                "num_subcubes"
            ]
        )
        != hardware.num_subcubes
    ):

        raise SpatialLayoutLoadError(
            "Candidate 中 num_subcubes "
            "与 N² 不一致。"
        )

    # ========================================================
    # 检查 Q
    # ========================================================

    if (
        Q
        != hardware.total_plane_slots
    ):

        raise SpatialLayoutLoadError(
            "Candidate 中 Q "
            "与 N²×D 不一致："
            f"json={Q}, "
            f"actual="
            f"{hardware.total_plane_slots}。"
        )

    # ========================================================
    # 检查 D 是否真的是最小需要深度
    # ========================================================

    expected_D = ceil(
        P
        / hardware.num_subcubes
    )

    if (
        hardware.D
        != expected_D
    ):

        raise SpatialLayoutLoadError(
            "Candidate 中 D "
            "不是 ceil(P/N²)："
            f"D={hardware.D}, "
            f"expected={expected_D}。"
        )

    if (
        P
        > hardware.total_plane_slots
    ):

        raise SpatialLayoutLoadError(
            "Candidate 中 P > Q。"
        )

    return hardware


# ============================================================
# 单个 PhysicalSlot 恢复
# ============================================================


def _slot_from_dict(
    *,
    data: dict[
        str,
        Any,
    ],
    expected_plane_id: int,
) -> PhysicalSlot:
    """
    从 JSON 恢复 PhysicalSlot。
    """

    required = {
        "slot_id",
        "plane_id",

        "x",
        "y",

        "slot_rows",
        "slot_cols",

        "orientation_swapped",
    }

    missing = (
        required
        - set(data)
    )

    if missing:

        raise SpatialLayoutLoadError(
            "PhysicalSlot JSON "
            "缺少字段："
            f"{sorted(missing)}。"
        )

    plane_id = int(
        data["plane_id"]
    )

    if (
        plane_id
        != expected_plane_id
    ):

        raise SpatialLayoutLoadError(
            "Slot 的 plane_id "
            "与所属 Plane 不一致："
            f"slot={plane_id}, "
            f"plane={expected_plane_id}。"
        )

    slot = PhysicalSlot(
        slot_id=int(
            data["slot_id"]
        ),

        plane_id=plane_id,

        x=int(
            data["x"]
        ),

        y=int(
            data["y"]
        ),

        slot_rows=int(
            data["slot_rows"]
        ),

        slot_cols=int(
            data["slot_cols"]
        ),

        orientation_swapped=bool(
            data[
                "orientation_swapped"
            ]
        ),
    )

    # ========================================================
    # 如果 JSON 保存了 size_key，
    # 顺便验证没有损坏。
    # ========================================================

    saved_size_key = (
        data.get(
            "size_key"
        )
    )

    if (
        saved_size_key
        is not None
    ):

        if (
            not isinstance(
                saved_size_key,
                list,
            )
            or len(
                saved_size_key
            ) != 2
        ):

            raise SpatialLayoutLoadError(
                f"Slot-{slot.slot_id} "
                "size_key 格式错误。"
            )

        expected = tuple(
            int(value)
            for value
            in saved_size_key
        )

        if (
            slot.size_key
            != expected
        ):

            raise SpatialLayoutLoadError(
                f"Slot-{slot.slot_id} "
                "size_key 与尺寸不一致："
                f"json={expected}, "
                f"actual={slot.size_key}。"
            )

    return slot


# ============================================================
# 单个 Plane 恢复
# ============================================================


def _plane_from_dict(
    *,
    data: dict[
        str,
        Any,
    ],
    expected_H: int,
    expected_W: int,
) -> Plane:
    """
    从保存的 JSON 恢复一张 Plane。
    """

    required = {
        "plane_id",
        "H",
        "W",
        "slots",
    }

    missing = (
        required
        - set(data)
    )

    if missing:

        raise SpatialLayoutLoadError(
            "Plane JSON 缺少字段："
            f"{sorted(missing)}。"
        )

    plane_id = int(
        data["plane_id"]
    )

    H = int(
        data["H"]
    )

    W = int(
        data["W"]
    )

    if (
        H != expected_H
        or W != expected_W
    ):

        raise SpatialLayoutLoadError(
            f"Plane-{plane_id} "
            "尺寸与 Layout 不一致："
            f"plane={H}×{W}, "
            f"layout="
            f"{expected_H}×{expected_W}。"
        )

    slots_data = (
        data["slots"]
    )

    if not isinstance(
        slots_data,
        list,
    ):

        raise SpatialLayoutLoadError(
            f"Plane-{plane_id} "
            "slots 必须是 list。"
        )

    # ========================================================
    # 建立空 Plane
    # ========================================================

    plane = (
        create_empty_plane(
            plane_id=plane_id,
            H=H,
            W=W,
        )
    )

    # ========================================================
    # 恢复 Slot
    # ========================================================

    for slot_data in slots_data:

        if not isinstance(
            slot_data,
            dict,
        ):

            raise SpatialLayoutLoadError(
                f"Plane-{plane_id} "
                "存在非 dict Slot。"
            )

        slot = _slot_from_dict(
            data=slot_data,
            expected_plane_id=(
                plane_id
            ),
        )

        # add_slot 会检查：
        #
        # 1. 越界；
        # 2. 重叠；
        # 3. plane_id。
        plane.add_slot(
            slot
        )

    # ========================================================
    # JSON 是最终布局。
    #
    # free_rectangles 是 MaxRects 运行时内部状态，
    # run_spatial_baseline.py 并没有保存它。
    #
    # 所以加载后明确置空，
    # 防止 create_empty_plane() 留下：
    #
    #     整个 Plane 仍然是 free
    #
    # 这种错误语义。
    # ========================================================

    plane.replace_free_rectangles(
        []
    )

    # ========================================================
    # 最终几何验证
    # ========================================================

    plane.validate_layout()

    # ========================================================
    # used_area
    # ========================================================

    if (
        "used_area"
        in data
    ):

        saved_used_area = int(
            data[
                "used_area"
            ]
        )

        if (
            plane.used_area
            != saved_used_area
        ):

            raise SpatialLayoutLoadError(
                f"Plane-{plane_id} "
                "used_area 不一致："
                f"json={saved_used_area}, "
                f"actual="
                f"{plane.used_area}。"
            )

    # ========================================================
    # unused_area
    # ========================================================

    if (
        "unused_area"
        in data
    ):

        saved_unused_area = int(
            data[
                "unused_area"
            ]
        )

        if (
            plane.unused_area
            != saved_unused_area
        ):

            raise SpatialLayoutLoadError(
                f"Plane-{plane_id} "
                "unused_area 不一致："
                f"json={saved_unused_area}, "
                f"actual="
                f"{plane.unused_area}。"
            )

    # ========================================================
    # signature
    # ========================================================

    saved_signature = (
        data.get(
            "signature"
        )
    )

    if (
        saved_signature
        is not None
    ):

        if not isinstance(
            saved_signature,
            list,
        ):

            raise SpatialLayoutLoadError(
                f"Plane-{plane_id} "
                "signature 必须是 list。"
            )

        normalized_signature = tuple(
            tuple(
                int(value)
                for value
                in size_key
            )

            for size_key
            in saved_signature
        )

        if (
            plane.signature()
            != normalized_signature
        ):

            raise SpatialLayoutLoadError(
                f"Plane-{plane_id} "
                "signature 不一致："
                f"json="
                f"{normalized_signature}, "
                f"actual="
                f"{plane.signature()}。"
            )

    return plane


# ============================================================
# 加载 layout JSON
# ============================================================


def load_physical_layout(
    *,
    layout_path: Path | str,

    expected_H: int | None = None,
    expected_W: int | None = None,
    expected_P: int | None = None,

    require_single_chunk: bool = True,
) -> tuple[
    tuple[
        Plane,
        ...
    ],
    dict[
        str,
        Any,
    ],
]:
    """
    加载单个：

        results/layouts/<layout_id>.json

    返回：

        (
            physical_planes,
            layout_metadata
        )
    """

    path = Path(
        layout_path
    ).resolve()

    data = _load_json(
        path
    )

    if not isinstance(
        data,
        dict,
    ):

        raise SpatialLayoutLoadError(
            "Layout JSON 最外层 "
            "必须是 dict。"
        )

    # ========================================================
    # Version
    # ========================================================

    version = data.get(
        "layout_version"
    )

    if version != 1:

        raise SpatialLayoutLoadError(
            "当前只支持 "
            "layout_version=1，"
            f"实际为 {version!r}。"
        )

    # ========================================================
    # 基础字段
    # ========================================================

    required = {
        "template",

        "H",
        "W",

        "matrix_count",

        "P",

        "slot_count",

        "planes",
    }

    missing = (
        required
        - set(data)
    )

    if missing:

        raise SpatialLayoutLoadError(
            "Layout JSON 缺少字段："
            f"{sorted(missing)}。"
        )

    H = int(
        data["H"]
    )

    W = int(
        data["W"]
    )

    P = int(
        data["P"]
    )

    matrix_count = int(
        data["matrix_count"]
    )

    slot_count = int(
        data["slot_count"]
    )

    # ========================================================
    # Candidate 与 Layout 尺寸一致
    # ========================================================

    if (
        expected_H is not None
        and H != expected_H
    ):

        raise SpatialLayoutLoadError(
            "Layout H 与 Candidate 不一致："
            f"layout={H}, "
            f"candidate={expected_H}。"
        )

    if (
        expected_W is not None
        and W != expected_W
    ):

        raise SpatialLayoutLoadError(
            "Layout W 与 Candidate 不一致："
            f"layout={W}, "
            f"candidate={expected_W}。"
        )

    if (
        expected_P is not None
        and P != expected_P
    ):

        raise SpatialLayoutLoadError(
            "Layout P 与 Candidate 不一致："
            f"layout={P}, "
            f"candidate={expected_P}。"
        )

    # ========================================================
    # Template
    # ========================================================

    template = data[
        "template"
    ]

    if not isinstance(
        template,
        dict,
    ):

        raise SpatialLayoutLoadError(
            "template 必须是 dict。"
        )

    template_id = (
        template.get(
            "template_id"
        )
    )

    chunk_count = (
        template.get(
            "chunk_count"
        )
    )

    if not isinstance(
        template_id,
        str,
    ):

        raise SpatialLayoutLoadError(
            "template_id 不存在或格式错误。"
        )

    if not isinstance(
        chunk_count,
        int,
    ):

        raise SpatialLayoutLoadError(
            "template.chunk_count "
            "不存在或格式错误。"
        )

    # ========================================================
    # 当前 Step4 只支持：
    #
    # 一个矩阵 -> 一个 LogicalWeightCube
    #
    # 所以要求 chunk_count == 1。
    # ========================================================

    if (
        require_single_chunk
        and chunk_count != 1
    ):

        raise SpatialLayoutLoadError(
            "当前第四步 LogicalWeightCube "
            "实现要求 "
            "template.chunk_count == 1，"
            f"当前模板为 {chunk_count}。\n"
            "请选择能够完整容纳 "
            "7168×2048 矩阵的空间方案。"
        )

    # ========================================================
    # Plane List
    # ========================================================

    planes_data = (
        data["planes"]
    )

    if not isinstance(
        planes_data,
        list,
    ):

        raise SpatialLayoutLoadError(
            "planes 必须是 list。"
        )

    if (
        len(planes_data)
        != P
    ):

        raise SpatialLayoutLoadError(
            "Layout 中 Plane 数量 "
            "与 P 不一致："
            f"len={len(planes_data)}, "
            f"P={P}。"
        )

    planes: list[
        Plane
    ] = []

    for plane_data in (
        planes_data
    ):

        if not isinstance(
            plane_data,
            dict,
        ):

            raise SpatialLayoutLoadError(
                "planes 中存在非 dict 项。"
            )

        plane = _plane_from_dict(
            data=plane_data,
            expected_H=H,
            expected_W=W,
        )

        planes.append(
            plane
        )

    # ========================================================
    # 按 plane_id 排序
    # ========================================================

    planes.sort(
        key=lambda plane: (
            plane.plane_id
        )
    )

    # ========================================================
    # Plane ID 唯一
    # ========================================================

    plane_ids = [
        plane.plane_id
        for plane
        in planes
    ]

    if (
        len(plane_ids)
        != len(
            set(plane_ids)
        )
    ):

        raise SpatialLayoutLoadError(
            "存在重复 plane_id。"
        )

    # ========================================================
    # Slot 数
    # ========================================================

    all_slots = [
        slot
        for plane
        in planes
        for slot
        in plane.slots
    ]

    if (
        len(all_slots)
        != slot_count
    ):

        raise SpatialLayoutLoadError(
            "实际 PhysicalSlot 数量 "
            "与 slot_count 不一致："
            f"actual={len(all_slots)}, "
            f"json={slot_count}。"
        )

    # ========================================================
    # Slot ID 唯一
    # ========================================================

    slot_ids = [
        slot.slot_id
        for slot
        in all_slots
    ]

    if (
        len(slot_ids)
        != len(
            set(slot_ids)
        )
    ):

        raise SpatialLayoutLoadError(
            "存在重复 slot_id。"
        )

    # ========================================================
    # 当前 single-chunk 模式下：
    #
    # 一个矩阵 -> 一个 Slot
    #
    # 所以：
    #
    # matrix_count == slot_count
    # ========================================================

    if (
        require_single_chunk
        and matrix_count != slot_count
    ):

        raise SpatialLayoutLoadError(
            "single-chunk 模式下 "
            "matrix_count 应等于 slot_count："
            f"matrix_count={matrix_count}, "
            f"slot_count={slot_count}。"
        )

    metadata = {
        "layout_version": 1,

        "template_id": (
            template_id
        ),

        "template_chunk_count": (
            chunk_count
        ),

        "H": H,
        "W": W,

        "matrix_count": (
            matrix_count
        ),

        "P": P,

        "slot_count": (
            slot_count
        ),
    }

    return (
        tuple(planes),
        metadata,
    )


# ============================================================
# 完整入口
# ============================================================


def load_spatial_layout_bundle(
    *,
    results_dir: Path | str = (
        DEFAULT_RESULTS_DIR
    ),

    spatial_rank: int | None = None,

    N: int | None = None,

    H: int | None = None,

    W: int | None = None,

    layout_id: str | None = None,

    expected_matrix_count: (
        int | None
    ) = None,

    require_single_chunk: bool = True,
) -> SpatialLayoutBundle:
    """
    第四步最常用入口。

    自动读取：

        results/spatial_candidates.json

    选择 Candidate 后，再读取：

        results/layouts/<layout_id>.json

    最终返回：

        SpatialLayoutBundle
    """

    root = Path(
        results_dir
    ).resolve()

    summary_path = (
        root
        / "spatial_candidates.json"
    )

    candidates = (
        load_spatial_candidates(
            summary_path
        )
    )

    candidate = (
        select_spatial_candidate(
            candidates,

            spatial_rank=(
                spatial_rank
            ),

            N=N,
            H=H,
            W=W,

            layout_id=(
                layout_id
            ),

            require_valid=True,
        )
    )

    hardware = (
        candidate_to_hardware(
            candidate
        )
    )

    chosen_layout_id = str(
        candidate[
            "layout_id"
        ]
    )

    layout_path = (
        root
        / "layouts"
        / f"{chosen_layout_id}.json"
    )

    (
        physical_planes,
        metadata,
    ) = load_physical_layout(
        layout_path=(
            layout_path
        ),

        expected_H=(
            hardware.H
        ),

        expected_W=(
            hardware.W
        ),

        expected_P=int(
            candidate["P"]
        ),

        require_single_chunk=(
            require_single_chunk
        ),
    )

    # ========================================================
    # template_id 必须一致
    # ========================================================

    if (
        metadata[
            "template_id"
        ]
        != candidate[
            "template_id"
        ]
    ):

        raise SpatialLayoutLoadError(
            "Candidate 与 Layout "
            "template_id 不一致："
            f"candidate="
            f"{candidate['template_id']}, "
            f"layout="
            f"{metadata['template_id']}。"
        )

    # ========================================================
    # 如果第四步知道理论矩阵数，
    # 可以在这里提前发现：
    #
    # 旧的 44544 layout
    # 与
    # 新的 44718 Shared Expert 模型
    #
    # 不一致。
    # ========================================================

    if (
        expected_matrix_count
        is not None
    ):

        if (
            metadata[
                "matrix_count"
            ]
            != expected_matrix_count
        ):

            raise SpatialLayoutLoadError(
                "空间 Layout 的 matrix_count "
                "与当前逻辑模型不一致："
                f"layout="
                f"{metadata['matrix_count']}, "
                f"expected="
                f"{expected_matrix_count}。\n"
                "如果你刚刚加入 Shared Expert，"
                "请确认已经重新运行过 "
                "run_spatial_baseline.py --full。"
            )

    bundle = (
        SpatialLayoutBundle(
            layout_id=(
                chosen_layout_id
            ),

            template_id=str(
                candidate[
                    "template_id"
                ]
            ),

            spatial_rank=int(
                candidate[
                    "spatial_rank"
                ]
            ),

            hardware=hardware,

            matrix_count=int(
                metadata[
                    "matrix_count"
                ]
            ),

            plane_count=int(
                metadata["P"]
            ),

            slot_count=int(
                metadata[
                    "slot_count"
                ]
            ),

            template_chunk_count=int(
                metadata[
                    "template_chunk_count"
                ]
            ),

            physical_planes=(
                physical_planes
            ),
        )
    )

    # ========================================================
    # 最后检查 Q >= P
    # ========================================================

    if (
        bundle.plane_count
        > bundle.hardware
        .total_plane_slots
    ):

        raise SpatialLayoutLoadError(
            "恢复后出现 P > Q。"
        )

    return bundle


# ============================================================
# 输出
# ============================================================


def print_spatial_layout_bundle(
    bundle: SpatialLayoutBundle,
) -> None:
    """
    打印恢复结果。
    """

    print(
        "\n"
        "========== Loaded Spatial Layout =========="
    )

    print(
        f"Spatial Rank："
        f"{bundle.spatial_rank}"
    )

    print(
        f"Layout ID："
        f"{bundle.layout_id}"
    )

    print(
        f"Template："
        f"{bundle.template_id}"
    )

    print(
        f"Template Chunk Count："
        f"{bundle.template_chunk_count}"
    )

    print(
        f"N："
        f"{bundle.hardware.N}"
    )

    print(
        f"H×W："
        f"{bundle.hardware.H}"
        "×"
        f"{bundle.hardware.W}"
    )

    print(
        f"D："
        f"{bundle.hardware.D}"
    )

    print(
        f"Sub-Cubes："
        f"{bundle.hardware.num_subcubes}"
    )

    print(
        f"Matrix Count："
        f"{bundle.matrix_count}"
    )

    print(
        f"Physical Slots："
        f"{bundle.slot_count}"
    )

    print(
        f"P："
        f"{bundle.plane_count}"
    )

    print(
        f"Q："
        f"{bundle.hardware.total_plane_slots}"
    )

    print(
        f"Empty Plane Slots："
        f"{bundle.empty_plane_slots}"
    )


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "读取前三步已经保存的 "
                "匿名 Spatial Layout。"
            )
        )
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help=(
            "指定 spatial_rank。"
            "不指定则选择满足其他条件的 "
            "最优合法 Candidate。"
        ),
    )

    parser.add_argument(
        "--N",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--H",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--W",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--layout-id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--expected-matrices",
        type=int,
        default=None,
    )

    args = (
        parser.parse_args()
    )

    bundle = (
        load_spatial_layout_bundle(
            results_dir=(
                args.results_dir
            ),

            spatial_rank=(
                args.rank
            ),

            N=args.N,
            H=args.H,
            W=args.W,

            layout_id=(
                args.layout_id
            ),

            expected_matrix_count=(
                args.expected_matrices
            ),
        )
    )

    print_spatial_layout_bundle(
        bundle
    )


if __name__ == "__main__":
    main()