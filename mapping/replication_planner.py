"""
Expert Weight Replication Planner.

Purpose
-------
Build an *independent* replication overlay on top of the frozen formal Mapping.

This file DOES NOT modify:
    - the base mapping JSON,
    - Pairing,
    - Sub-Cube Mapping,
    - Physical Binding.

It only uses Q-P spare hardware plane slots to place extra copies of selected
Routed Expert weights.

Current replica unit
--------------------
The current physical plane can hold two 7168x2048 Weight-Cubes.

For two Routed Experts from the same layer:
    Expert-A gate + down -> 1 plane
    Expert-B gate + down -> 1 plane
    Expert-A up + Expert-B up -> 1 plane

Therefore one replicated Routed Expert pair costs 3 spare planes and creates
two complete Expert replicas.

Selection signal
----------------
Use PROFILE subset only.

For every Prefill batch / layer / routed expert:

    duplicate_pressure += max(0, hits_in_this_batch - 1)

This directly measures the condition that replication can help:
multiple input tokens in one Prefill batch need the same Expert.

The Evaluation subset must never be used to choose replicas.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mapping.logical_weight import MATRIX_DOWN, MATRIX_GATE, MATRIX_UP
from mapping.trace_profile import (
    DEFAULT_TRACE_ROOT,
    NUM_MOE_LAYERS,
    NUM_ROUTED_EXPERTS,
)
from mapping.trace_split import TRACE_SUBSETS
from scheduling.prefill_workload import (
    PrefillWorkloadStats,
    iter_prefill_batches,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_PATH = (
    PROJECT_ROOT
    / "results"
    / "mappings"
    / "mapping_baseline_N4_H7168_W4096.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "replication"
    / "replication_plan.json"
)

PROFILE_SUBSET = "profile"
REPLICATION_POLICY = "profile_prefill_duplicate_pressure_routed_pair"


class ReplicationPlannerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExpertPressure:
    layer_id: int
    expert_id: int
    total_hits: int
    duplicate_pressure: int
    duplicate_batches: int
    max_hits_in_batch: int


@dataclass(frozen=True, slots=True)
class PairCandidate:
    layer_id: int
    expert_a: int
    expert_b: int
    duplicate_pressure: int
    total_hits: int
    duplicate_batches: int
    max_hits_in_batch: int
    base_subcubes_a: tuple[int, int]
    base_subcubes_b: tuple[int, int]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise ReplicationPlannerError(f"文件不存在：{path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ReplicationPlannerError(f"JSON 顶层必须是对象：{path}")
    return data


def _safe_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _validate_mapping(mapping: dict) -> None:
    required = {
        "model",
        "trace",
        "spatial",
        "pairing",
        "subcube_mapping",
        "placements",
    }
    missing = required - set(mapping)
    if missing:
        raise ReplicationPlannerError(
            f"Mapping 缺少字段：{sorted(missing)}"
        )

    spatial = mapping["spatial"]
    sub = mapping["subcube_mapping"]

    for key in ("D", "num_subcubes", "P", "Q", "empty_plane_slots"):
        if key not in spatial:
            raise ReplicationPlannerError(f"spatial 缺少 {key}")

    plane_counts = sub.get("plane_counts")
    if not isinstance(plane_counts, list):
        raise ReplicationPlannerError(
            "subcube_mapping.plane_counts 必须是 list。"
        )

    if len(plane_counts) != int(spatial["num_subcubes"]):
        raise ReplicationPlannerError(
            "plane_counts 长度与 num_subcubes 不一致。"
        )

    D = int(spatial["D"])
    if any((not isinstance(x, int)) or x < 0 or x > D for x in plane_counts):
        raise ReplicationPlannerError("plane_counts 中存在非法值。")

    empty = sum(D - int(x) for x in plane_counts)
    if empty != int(spatial["empty_plane_slots"]):
        raise ReplicationPlannerError(
            "Mapping 的 empty_plane_slots 与 D-plane_counts 不一致："
            f"derived={empty}, json={spatial['empty_plane_slots']}。"
        )


def _build_base_location_index(
    mapping: dict,
) -> dict[tuple[int, int, str], dict]:
    result: dict[tuple[int, int, str], dict] = {}

    for row in mapping["placements"]:
        key = (
            int(row["layer_id"]),
            int(row["expert_id"]),
            str(row["matrix_name"]),
        )
        if key in result:
            raise ReplicationPlannerError(
                f"Mapping 中出现重复矩阵位置：{key}"
            )
        result[key] = row

    return result


def _expert_base_subcubes(
    *,
    location_index: dict[tuple[int, int, str], dict],
    layer_id: int,
    expert_id: int,
) -> tuple[int, int]:
    try:
        gate = location_index[(layer_id, expert_id, MATRIX_GATE)]
        up = location_index[(layer_id, expert_id, MATRIX_UP)]
        down = location_index[(layer_id, expert_id, MATRIX_DOWN)]
    except KeyError as exc:
        raise ReplicationPlannerError(
            f"找不到 L{layer_id}/E{expert_id} 的完整 gate/up/down。"
        ) from exc

    gate_sc = int(gate["subcube_id"])
    up_sc = int(up["subcube_id"])
    down_sc = int(down["subcube_id"])

    if gate_sc != down_sc:
        raise ReplicationPlannerError(
            f"L{layer_id}/E{expert_id} gate/down 不在同一 SC。"
        )
    if gate_sc == up_sc:
        raise ReplicationPlannerError(
            f"L{layer_id}/E{expert_id} gate/up 未分离。"
        )

    return gate_sc, up_sc


def collect_profile_duplicate_pressure(
    *,
    trace_root: Path,
    trace_manifest: Path,
    trace_subset: str,
    max_batches: int | None,
    verbose: bool,
) -> tuple[
    dict[tuple[int, int], ExpertPressure],
    int,
    int,
]:
    if trace_subset != PROFILE_SUBSET:
        raise ReplicationPlannerError(
            "Replication Planner 只能使用 profile subset 选择副本，"
            f"当前收到 subset={trace_subset!r}。"
        )

    total_hits = [
        [0] * NUM_ROUTED_EXPERTS
        for _ in range(NUM_MOE_LAYERS)
    ]
    duplicate_pressure = [
        [0] * NUM_ROUTED_EXPERTS
        for _ in range(NUM_MOE_LAYERS)
    ]
    duplicate_batches = [
        [0] * NUM_ROUTED_EXPERTS
        for _ in range(NUM_MOE_LAYERS)
    ]
    max_hits = [
        [0] * NUM_ROUTED_EXPERTS
        for _ in range(NUM_MOE_LAYERS)
    ]

    stats = PrefillWorkloadStats()
    batch_count = 0
    token_count = 0

    for batch in iter_prefill_batches(
        trace_root=trace_root,
        trace_manifest=trace_manifest,
        trace_subset=trace_subset,
        max_files=None,
        max_batches=max_batches,
        stats=stats,
        verbose=False,
    ):
        batch_count += 1
        token_count += batch.token_count

        # batch.routed_experts_by_token:
        # token -> layer -> routed Top-K
        for layer_id in range(NUM_MOE_LAYERS):
            counts: Counter[int] = Counter()

            for token_routes in batch.routed_experts_by_token:
                route = token_routes[layer_id]
                for expert_id in route:
                    if not 0 <= expert_id < NUM_ROUTED_EXPERTS:
                        raise ReplicationPlannerError(
                            "Profile routed route 中出现非法 Expert ID："
                            f"layer={layer_id}, expert={expert_id}。"
                        )
                    counts[expert_id] += 1

            for expert_id, hits in counts.items():
                total_hits[layer_id][expert_id] += hits
                if hits > 1:
                    duplicate_pressure[layer_id][expert_id] += hits - 1
                    duplicate_batches[layer_id][expert_id] += 1
                if hits > max_hits[layer_id][expert_id]:
                    max_hits[layer_id][expert_id] = hits

        if verbose and (batch_count == 1 or batch_count % 200 == 0):
            print(
                "[ReplicationProfile] "
                f"batches={batch_count}, tokens={token_count}"
            )

    if batch_count <= 0:
        raise ReplicationPlannerError(
            "Profile subset 中没有 Prefill Candidate。"
        )

    result: dict[tuple[int, int], ExpertPressure] = {}
    for layer_id in range(NUM_MOE_LAYERS):
        for expert_id in range(NUM_ROUTED_EXPERTS):
            result[(layer_id, expert_id)] = ExpertPressure(
                layer_id=layer_id,
                expert_id=expert_id,
                total_hits=total_hits[layer_id][expert_id],
                duplicate_pressure=duplicate_pressure[layer_id][expert_id],
                duplicate_batches=duplicate_batches[layer_id][expert_id],
                max_hits_in_batch=max_hits[layer_id][expert_id],
            )

    return result, batch_count, token_count


def choose_replica_target_subcubes(
    *,
    D: int,
    plane_counts: list[int],
) -> tuple[int, int, int]:
    """
    One replicated routed pair needs:
        2 gate/down planes on gd_sc
        1 up+up plane on up_sc

    Search ordered (gd_sc, up_sc), gd_sc != up_sc.

    Returns:
        (gate_down_target_sc, up_target_sc, max_pair_units)
    """
    free = [D - count for count in plane_counts]

    candidates: list[tuple[tuple[int, ...], int, int, int]] = []

    for gd_sc in range(len(free)):
        for up_sc in range(len(free)):
            if gd_sc == up_sc:
                continue

            units = min(
                free[gd_sc] // 2,
                free[up_sc],
            )
            if units <= 0:
                continue

            # Maximize pair units first.
            # Then prefer more residual total capacity and deterministic IDs.
            used_gd = 2 * units
            used_up = units
            residual = (
                free[gd_sc] - used_gd
                + free[up_sc] - used_up
            )

            score = (
                -units,
                -free[gd_sc],
                -free[up_sc],
                -residual,
                gd_sc,
                up_sc,
            )
            candidates.append((score, gd_sc, up_sc, units))

    if not candidates:
        raise ReplicationPlannerError(
            "现有 Q-P 空间不足以形成一个完整 Routed Expert Pair 副本："
            "至少需要某个 SC 有 2 个空 Plane，另一个不同 SC 有 1 个空 Plane。"
        )

    _score, gd_sc, up_sc, units = min(
        candidates,
        key=lambda item: item[0],
    )
    return gd_sc, up_sc, units


def _iter_existing_routed_pairs(
    mapping: dict,
) -> Iterable[tuple[int, int, int]]:
    rows = mapping["pairing"].get("routed_up_pairs_by_layer")
    if not isinstance(rows, list):
        raise ReplicationPlannerError(
            "Mapping 缺少 pairing.routed_up_pairs_by_layer。"
        )

    for layer_row in rows:
        layer_id = int(layer_row["layer_id"])
        pairs = layer_row.get("pairs")
        if not isinstance(pairs, list):
            raise ReplicationPlannerError(
                f"Layer-{layer_id} routed up pairs 非法。"
            )

        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ReplicationPlannerError(
                    f"Layer-{layer_id} 存在非法 routed up pair：{pair!r}"
                )
            a, b = int(pair[0]), int(pair[1])
            if not (
                0 <= a < NUM_ROUTED_EXPERTS
                and 0 <= b < NUM_ROUTED_EXPERTS
                and a != b
            ):
                raise ReplicationPlannerError(
                    f"Layer-{layer_id} Routed Pair Expert ID 非法：{pair!r}"
                )
            yield layer_id, a, b


def rank_pair_candidates(
    *,
    mapping: dict,
    pressures: dict[tuple[int, int], ExpertPressure],
    gate_down_target_sc: int,
    up_target_sc: int,
    require_base_disjoint: bool,
) -> list[PairCandidate]:
    location_index = _build_base_location_index(mapping)
    target_scs = {gate_down_target_sc, up_target_sc}

    candidates: list[PairCandidate] = []

    for layer_id, expert_a, expert_b in _iter_existing_routed_pairs(mapping):
        base_a = _expert_base_subcubes(
            location_index=location_index,
            layer_id=layer_id,
            expert_id=expert_a,
        )
        base_b = _expert_base_subcubes(
            location_index=location_index,
            layer_id=layer_id,
            expert_id=expert_b,
        )

        if require_base_disjoint:
            if (
                target_scs.intersection(base_a)
                or target_scs.intersection(base_b)
            ):
                continue

        pa = pressures[(layer_id, expert_a)]
        pb = pressures[(layer_id, expert_b)]

        candidates.append(
            PairCandidate(
                layer_id=layer_id,
                expert_a=expert_a,
                expert_b=expert_b,
                duplicate_pressure=(
                    pa.duplicate_pressure
                    + pb.duplicate_pressure
                ),
                total_hits=pa.total_hits + pb.total_hits,
                duplicate_batches=(
                    pa.duplicate_batches
                    + pb.duplicate_batches
                ),
                max_hits_in_batch=max(
                    pa.max_hits_in_batch,
                    pb.max_hits_in_batch,
                ),
                base_subcubes_a=base_a,
                base_subcubes_b=base_b,
            )
        )

    candidates.sort(
        key=lambda row: (
            -row.duplicate_pressure,
            -row.duplicate_batches,
            -row.total_hits,
            -row.max_hits_in_batch,
            row.layer_id,
            row.expert_a,
            row.expert_b,
        )
    )
    return candidates


def _next_ids(mapping: dict) -> tuple[int, int, int]:
    placements = mapping["placements"]
    if not placements:
        raise ReplicationPlannerError("Mapping placements 为空。")

    max_cube = max(int(row["cube_id"]) for row in placements)
    max_plane = max(int(row["physical_plane_id"]) for row in placements)
    max_slot = max(int(row["slot_id"]) for row in placements)

    return max_cube + 1, max_plane + 1, max_slot + 1


def _slot_payload(
    *,
    virtual_cube_id: int,
    virtual_plane_id: int,
    virtual_slot_id: int,
    subcube_id: int,
    z: int,
    slot_index: int,
    matrix_name: str,
) -> dict:
    if slot_index not in (0, 1):
        raise ReplicationPlannerError("slot_index 必须为 0 或 1。")

    # Current formal H=7168, W=4096 template:
    # slot-0: y=0, slot-1: y=2048
    return {
        "virtual_cube_id": virtual_cube_id,
        "virtual_plane_id": virtual_plane_id,
        "virtual_slot_id": virtual_slot_id,
        "subcube_id": subcube_id,
        "z": z,
        "slot_index": slot_index,
        "x": 0,
        "y": 0 if slot_index == 0 else 2048,
        "slot_rows": 7168,
        "slot_cols": 2048,
        "matrix_name": matrix_name,
        "logical_cube_rotated": matrix_name == MATRIX_DOWN,
    }


def build_replication_plan(
    *,
    mapping_path: Path,
    trace_root: Path,
    trace_manifest: Path,
    trace_subset: str,
    max_batches: int | None,
    max_pair_units: int | None,
    require_base_disjoint: bool,
    verbose: bool,
) -> dict:
    mapping = _load_json(mapping_path)
    _validate_mapping(mapping)

    spatial = mapping["spatial"]
    sub = mapping["subcube_mapping"]

    D = int(spatial["D"])
    plane_counts = [int(x) for x in sub["plane_counts"]]
    free_before = [D - x for x in plane_counts]

    gd_sc, up_sc, capacity_pair_units = choose_replica_target_subcubes(
        D=D,
        plane_counts=plane_counts,
    )

    if max_pair_units is None:
        pair_units = capacity_pair_units
    else:
        if max_pair_units <= 0:
            raise ReplicationPlannerError("--max-pair-units 必须大于 0。")
        pair_units = min(max_pair_units, capacity_pair_units)

    pressures, profile_batches, profile_tokens = (
        collect_profile_duplicate_pressure(
            trace_root=trace_root,
            trace_manifest=trace_manifest,
            trace_subset=trace_subset,
            max_batches=max_batches,
            verbose=verbose,
        )
    )

    candidates = rank_pair_candidates(
        mapping=mapping,
        pressures=pressures,
        gate_down_target_sc=gd_sc,
        up_target_sc=up_sc,
        require_base_disjoint=require_base_disjoint,
    )

    if len(candidates) < pair_units:
        raise ReplicationPlannerError(
            "满足当前副本放置约束的 Routed Expert Pair 不足："
            f"need={pair_units}, available={len(candidates)}。"
        )

    selected = candidates[:pair_units]

    next_cube_id, next_plane_id, next_slot_id = _next_ids(mapping)

    next_z = {
        sc: plane_counts[sc]
        for sc in range(len(plane_counts))
    }

    replicas: list[dict] = []
    selected_pairs_payload: list[dict] = []

    for pair_rank, candidate in enumerate(selected, start=1):
        layer_id = candidate.layer_id
        experts = (candidate.expert_a, candidate.expert_b)

        # 2 gate/down planes on gd_sc.
        gd_rows: dict[int, tuple[dict, dict]] = {}
        for expert_id in experts:
            plane_id = next_plane_id
            next_plane_id += 1
            z = next_z[gd_sc]
            next_z[gd_sc] += 1

            gate = _slot_payload(
                virtual_cube_id=next_cube_id,
                virtual_plane_id=plane_id,
                virtual_slot_id=next_slot_id,
                subcube_id=gd_sc,
                z=z,
                slot_index=0,
                matrix_name=MATRIX_GATE,
            )
            next_cube_id += 1
            next_slot_id += 1

            down = _slot_payload(
                virtual_cube_id=next_cube_id,
                virtual_plane_id=plane_id,
                virtual_slot_id=next_slot_id,
                subcube_id=gd_sc,
                z=z,
                slot_index=1,
                matrix_name=MATRIX_DOWN,
            )
            next_cube_id += 1
            next_slot_id += 1

            gd_rows[expert_id] = (gate, down)

        # One up+up plane on up_sc.
        up_plane_id = next_plane_id
        next_plane_id += 1
        up_z = next_z[up_sc]
        next_z[up_sc] += 1

        up_rows: dict[int, dict] = {}
        for slot_index, expert_id in enumerate(experts):
            up = _slot_payload(
                virtual_cube_id=next_cube_id,
                virtual_plane_id=up_plane_id,
                virtual_slot_id=next_slot_id,
                subcube_id=up_sc,
                z=up_z,
                slot_index=slot_index,
                matrix_name=MATRIX_UP,
            )
            next_cube_id += 1
            next_slot_id += 1
            up_rows[expert_id] = up

        expert_payloads = []
        for expert_id in experts:
            pressure = pressures[(layer_id, expert_id)]
            base_gd, base_up = _expert_base_subcubes(
                location_index=_build_base_location_index(mapping),
                layer_id=layer_id,
                expert_id=expert_id,
            )
            gate, down = gd_rows[expert_id]
            up = up_rows[expert_id]

            replica = {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "replica_id": 1,
                "selection_stats": {
                    "total_hits": pressure.total_hits,
                    "duplicate_pressure": pressure.duplicate_pressure,
                    "duplicate_batches": pressure.duplicate_batches,
                    "max_hits_in_batch": pressure.max_hits_in_batch,
                },
                "base_copy": {
                    "gate_down_subcube_id": base_gd,
                    "up_subcube_id": base_up,
                },
                "replica_copy": {
                    "gate": gate,
                    "up": up,
                    "down": down,
                },
            }
            replicas.append(replica)
            expert_payloads.append(replica)

        selected_pairs_payload.append(
            {
                "rank": pair_rank,
                "layer_id": layer_id,
                "experts": list(experts),
                "pair_duplicate_pressure": candidate.duplicate_pressure,
                "pair_total_hits": candidate.total_hits,
                "pair_duplicate_batches": candidate.duplicate_batches,
                "pair_max_hits_in_batch": candidate.max_hits_in_batch,
                "members": expert_payloads,
            }
        )

    used_plane_counts = [0] * len(plane_counts)
    used_plane_counts[gd_sc] = 2 * len(selected)
    used_plane_counts[up_sc] = len(selected)

    free_after = [
        free_before[sc] - used_plane_counts[sc]
        for sc in range(len(plane_counts))
    ]
    if any(x < 0 for x in free_after):
        raise ReplicationPlannerError(
            "内部错误：Replica 使用量超过空闲 Plane 容量。"
        )

    plan = {
        "replication_version": 1,
        "policy": {
            "name": REPLICATION_POLICY,
            "trace_manifest": _safe_relative_or_absolute(trace_manifest),
            "trace_subset": trace_subset,
            "profile_batch_count": profile_batches,
            "profile_input_token_count": profile_tokens,
            "routed_experts_only": True,
            "selection_metric": "sum_batch_max_0_hits_minus_1",
            "reuse_existing_routed_up_pairs": True,
            "require_base_copy_disjoint_from_replica_target_scs": (
                require_base_disjoint
            ),
        },
        "base_mapping": {
            "path": _safe_relative_or_absolute(mapping_path),
            "mapping_version": mapping.get("mapping_version"),
            "layout_id": spatial.get("layout_id"),
            "mapping_mode": sub.get("mode"),
            "pairing_mode": mapping.get("pairing", {}).get("mode"),
        },
        "hardware": {
            "N": int(spatial["N"]),
            "num_subcubes": int(spatial["num_subcubes"]),
            "H": int(spatial["H"]),
            "W": int(spatial["W"]),
            "D": D,
            "P": int(spatial["P"]),
            "Q": int(spatial["Q"]),
            "empty_plane_slots_before": sum(free_before),
            "plane_counts_before": plane_counts,
            "free_plane_counts_before": free_before,
            "replica_gate_down_subcube_id": gd_sc,
            "replica_up_subcube_id": up_sc,
            "capacity_pair_units": capacity_pair_units,
            "selected_pair_units": len(selected),
            "replicated_expert_count": 2 * len(selected),
            "replica_planes_used": 3 * len(selected),
            "free_plane_counts_after": free_after,
            "empty_plane_slots_after": sum(free_after),
        },
        "selected_pairs": selected_pairs_payload,
        "replicas": replicas,
        "top_candidate_pairs": [
            {
                "rank": rank,
                "layer_id": row.layer_id,
                "experts": [row.expert_a, row.expert_b],
                "duplicate_pressure": row.duplicate_pressure,
                "total_hits": row.total_hits,
                "duplicate_batches": row.duplicate_batches,
                "max_hits_in_batch": row.max_hits_in_batch,
                "base_subcubes": [
                    list(row.base_subcubes_a),
                    list(row.base_subcubes_b),
                ],
            }
            for rank, row in enumerate(candidates[:20], start=1)
        ],
    }

    return plan


def validate_replication_plan(plan: dict) -> None:
    hw = plan["hardware"]
    nsc = int(hw["num_subcubes"])
    D = int(hw["D"])
    before = [int(x) for x in hw["plane_counts_before"]]
    free_after = [int(x) for x in hw["free_plane_counts_after"]]

    if len(before) != nsc or len(free_after) != nsc:
        raise ReplicationPlannerError(
            "Replication plan 的 Sub-Cube 数量错误。"
        )

    gd_sc = int(hw["replica_gate_down_subcube_id"])
    up_sc = int(hw["replica_up_subcube_id"])
    if gd_sc == up_sc:
        raise ReplicationPlannerError(
            "Replica gate/down 与 up 必须位于不同 Sub-Cube。"
        )

    used_coordinates: set[tuple[int, int]] = set()
    used_cubes: set[int] = set()
    used_slots: set[int] = set()

    for replica in plan["replicas"]:
        copy = replica["replica_copy"]
        gate = copy["gate"]
        up = copy["up"]
        down = copy["down"]

        if gate["subcube_id"] != down["subcube_id"]:
            raise ReplicationPlannerError("Replica gate/down 未共址。")
        if gate["z"] != down["z"]:
            raise ReplicationPlannerError("Replica gate/down z 不一致。")
        if gate["subcube_id"] == up["subcube_id"]:
            raise ReplicationPlannerError("Replica gate/up 未分离。")

        for matrix in (gate, up, down):
            sc = int(matrix["subcube_id"])
            z = int(matrix["z"])
            if not (0 <= sc < nsc and 0 <= z < D):
                raise ReplicationPlannerError(
                    f"Replica 坐标越界：SC={sc}, z={z}。"
                )
            cube_id = int(matrix["virtual_cube_id"])
            slot_id = int(matrix["virtual_slot_id"])
            if cube_id in used_cubes:
                raise ReplicationPlannerError(
                    f"重复 virtual_cube_id={cube_id}。"
                )
            if slot_id in used_slots:
                raise ReplicationPlannerError(
                    f"重复 virtual_slot_id={slot_id}。"
                )
            used_cubes.add(cube_id)
            used_slots.add(slot_id)

        used_coordinates.add((int(gate["subcube_id"]), int(gate["z"])))
        used_coordinates.add((int(up["subcube_id"]), int(up["z"])))

    expected_used_planes = int(hw["replica_planes_used"])
    if len(used_coordinates) != expected_used_planes:
        raise ReplicationPlannerError(
            "Replica 实际使用 Plane 坐标数与统计不一致："
            f"actual={len(used_coordinates)}, expected={expected_used_planes}。"
        )

    if any(x < 0 for x in free_after):
        raise ReplicationPlannerError("Replica 后出现负空闲容量。")

    if sum(free_after) != int(hw["empty_plane_slots_after"]):
        raise ReplicationPlannerError(
            "empty_plane_slots_after 统计错误。"
        )


def save_replication_plan(
    *,
    plan: dict,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)


def print_replication_plan_summary(plan: dict) -> None:
    hw = plan["hardware"]

    print("\n========== Expert Weight Replication Plan ==========")
    print(f"Policy：{plan['policy']['name']}")
    print(f"Profile Subset：{plan['policy']['trace_subset']}")
    print(
        "Profile Batches / Tokens："
        f"{plan['policy']['profile_batch_count']} / "
        f"{plan['policy']['profile_input_token_count']}"
    )
    print(
        "Spare Planes Before："
        f"{hw['empty_plane_slots_before']}"
    )
    print(
        "Replica Target："
        f"gate+down -> SC-{hw['replica_gate_down_subcube_id']}, "
        f"up -> SC-{hw['replica_up_subcube_id']}"
    )
    print(
        "Capacity Pair Units："
        f"{hw['capacity_pair_units']}"
    )
    print(
        "Selected Pair Units："
        f"{hw['selected_pair_units']}"
    )
    print(
        "Replicated Experts："
        f"{hw['replicated_expert_count']}"
    )
    print(
        "Replica Planes Used："
        f"{hw['replica_planes_used']}"
    )
    print(
        "Spare Planes After："
        f"{hw['empty_plane_slots_after']}"
    )

    print("\nSelected Routed Expert Pairs：")
    for row in plan["selected_pairs"]:
        print(
            f"  #{row['rank']}: "
            f"L{row['layer_id']} "
            f"E{row['experts'][0]} + E{row['experts'][1]} | "
            f"duplicate_pressure={row['pair_duplicate_pressure']}, "
            f"hits={row['pair_total_hits']}, "
            f"duplicate_batches={row['pair_duplicate_batches']}, "
            f"max_hits/batch={row['pair_max_hits_in_batch']}"
        )

    print("\nFree Plane Counts After：")
    for sc, free in enumerate(hw["free_plane_counts_after"]):
        if free > 0:
            print(f"  SC-{sc}: {free}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "利用 Q-P 空闲 Plane，在冻结的正式 Mapping 上生成 "
            "Routed Expert Weight Replication overlay。"
        )
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
    )
    parser.add_argument(
        "--trace-manifest",
        type=Path,
        required=True,
        help="正式实验必须传 80/20 split manifest。",
    )
    parser.add_argument(
        "--trace-subset",
        choices=TRACE_SUBSETS,
        default=PROFILE_SUBSET,
        help="Planner 强制只允许 profile。",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="仅用于 smoke；正式生成不要设置。",
    )
    parser.add_argument(
        "--max-pair-units",
        type=int,
        default=None,
        help=(
            "最多复制多少个 routed up-pair；"
            "1 pair = 2 Experts = 3 spare planes。"
            "默认自动用到当前硬件允许的最大值。"
        ),
    )
    parser.add_argument(
        "--allow-base-overlap",
        action="store_true",
        help=(
            "允许原始 Expert 的 gate/up 使用 Replica 目标 SC。"
            "正式实验不建议开启；默认要求原始副本与两个 Replica SC 完全分离。"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    mapping = args.mapping.resolve()
    trace_root = args.trace_root.resolve()
    trace_manifest = args.trace_manifest.resolve()
    output = args.output.resolve()

    plan = build_replication_plan(
        mapping_path=mapping,
        trace_root=trace_root,
        trace_manifest=trace_manifest,
        trace_subset=args.trace_subset,
        max_batches=args.max_batches,
        max_pair_units=args.max_pair_units,
        require_base_disjoint=not args.allow_base_overlap,
        verbose=not args.quiet,
    )
    validate_replication_plan(plan)
    save_replication_plan(plan=plan, output=output)

    print_replication_plan_summary(plan)
    print(f"\nSaved：{output}")


if __name__ == "__main__":
    main()