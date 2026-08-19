import json

from mapping.spatial_layout_loader import (
    load_spatial_layout_bundle,
)


def build_fake_results(
    tmp_path,
):

    results_dir = (
        tmp_path
        / "results"
    )

    layouts_dir = (
        results_dir
        / "layouts"
    )

    layouts_dir.mkdir(
        parents=True
    )

    layout_id = (
        "H7168_W4096_test"
    )

    # ========================================================
    # spatial_candidates.json
    # ========================================================

    summary = {
        "candidates": [
            {
                "layout_id": (
                    layout_id
                ),

                "template_id": (
                    "test-template"
                ),

                "N": 2,

                "H": 7168,
                "W": 4096,

                "D": 1,

                "num_subcubes": 4,

                "P_lower": 2,

                "P": 2,

                "Q": 4,

                "valid": True,

                "spatial_rank": 1,
            }
        ]
    }

    with (
        results_dir
        / "spatial_candidates.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
        )

    # ========================================================
    # layout
    # ========================================================

    slot_area = (
        7168
        * 2048
    )

    plane_area = (
        7168
        * 4096
    )

    planes = []

    next_slot_id = 0

    for plane_id in range(2):

        slots = []

        for y in (
            0,
            2048,
        ):

            slots.append(
                {
                    "slot_id": (
                        next_slot_id
                    ),

                    "plane_id": (
                        plane_id
                    ),

                    "x": 0,
                    "y": y,

                    "slot_rows": (
                        7168
                    ),

                    "slot_cols": (
                        2048
                    ),

                    "size_key": [
                        2048,
                        7168,
                    ],

                    "orientation_swapped": (
                        False
                    ),
                }
            )

            next_slot_id += 1

        planes.append(
            {
                "plane_id": (
                    plane_id
                ),

                "H": 7168,
                "W": 4096,

                "used_area": (
                    2 * slot_area
                ),

                "unused_area": (
                    plane_area
                    - 2 * slot_area
                ),

                "utilization": 1.0,

                "signature": [
                    [
                        2048,
                        7168,
                    ],
                    [
                        2048,
                        7168,
                    ],
                ],

                "slots": slots,
            }
        )

    layout = {
        "layout_version": 1,

        "template": {
            "template_id": (
                "test-template"
            ),

            "base_rows": 7168,
            "base_cols": 2048,

            "orientation_mode": (
                "normal"
            ),

            "chunk_count": 1,

            "size_histogram": {
                "2048x7168": 1
            },

            "chunks": [
                {
                    "template_chunk_id": 0,

                    "row_start": 0,
                    "col_start": 0,

                    "rows": 7168,
                    "cols": 2048,

                    "size_key": [
                        2048,
                        7168,
                    ],
                }
            ],
        },

        "H": 7168,
        "W": 4096,

        "matrix_count": 4,

        "P": 2,

        "slot_count": 4,

        "total_weight_area": (
            4
            * slot_area
        ),

        "packing_utilization": (
            1.0
        ),

        "internal_fragmentation": (
            0
        ),

        "slot_size_histogram": {
            "2048x7168": 4
        },

        "planes": planes,
    }

    with (
        layouts_dir
        / f"{layout_id}.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            layout,
            file,
        )

    return results_dir


def test_load_bundle(
    tmp_path,
):

    results_dir = (
        build_fake_results(
            tmp_path
        )
    )

    bundle = (
        load_spatial_layout_bundle(
            results_dir=(
                results_dir
            ),

            expected_matrix_count=4,
        )
    )

    assert (
        bundle.layout_id
        == "H7168_W4096_test"
    )

    assert (
        bundle.hardware.N
        == 2
    )

    assert (
        bundle.hardware.H
        == 7168
    )

    assert (
        bundle.hardware.W
        == 4096
    )

    assert (
        bundle.hardware.D
        == 1
    )

    assert (
        bundle.plane_count
        == 2
    )

    assert (
        bundle.slot_count
        == 4
    )

    assert (
        len(
            bundle.physical_planes
        )
        == 2
    )

    assert (
        len(
            bundle.physical_slots
        )
        == 4
    )


def test_plane_signature(
    tmp_path,
):

    results_dir = (
        build_fake_results(
            tmp_path
        )
    )

    bundle = (
        load_spatial_layout_bundle(
            results_dir=(
                results_dir
            )
        )
    )

    for plane in (
        bundle.physical_planes
    ):

        assert (
            plane.signature()
            == (
                (
                    2048,
                    7168,
                ),
                (
                    2048,
                    7168,
                ),
            )
        )


def test_free_rectangles_are_not_restored(
    tmp_path,
):

    results_dir = (
        build_fake_results(
            tmp_path
        )
    )

    bundle = (
        load_spatial_layout_bundle(
            results_dir=(
                results_dir
            )
        )
    )

    # JSON 只保存最终空间布局，
    # 没有保存 MaxRects 中间状态。
    for plane in (
        bundle.physical_planes
    ):

        assert (
            len(
                plane.free_rectangles
            )
            == 0
        )


def test_empty_plane_slots(
    tmp_path,
):

    results_dir = (
        build_fake_results(
            tmp_path
        )
    )

    bundle = (
        load_spatial_layout_bundle(
            results_dir=(
                results_dir
            )
        )
    )

    # N=2
    # D=1
    #
    # Q=4
    # P=2
    assert (
        bundle.empty_plane_slots
        == 2
    )


def test_select_specific_hardware(
    tmp_path,
):

    results_dir = (
        build_fake_results(
            tmp_path
        )
    )

    bundle = (
        load_spatial_layout_bundle(
            results_dir=(
                results_dir
            ),

            N=2,
            H=7168,
            W=4096,
        )
    )

    assert (
        bundle.hardware.N
        == 2
    )

    assert (
        bundle.hardware.H
        == 7168
    )

    assert (
        bundle.hardware.W
        == 4096
    )