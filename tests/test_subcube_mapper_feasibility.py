"""所有受约束 Mapping 共用的 up 阶段容量可行性保护测试。"""

from mapping.subcube_mapper import _candidate_preserves_up_feasibility


def test_feasibility_guard_rejects_using_the_only_future_legal_slot():
    # 3 个 SC，每个还剩 1 个槽位。
    # 当前 Plane 之后还剩 1 张 Plane，它同时禁止 SC1、SC2，
    # 所以未来那张 Plane 唯一能去 SC0。
    plane_counts = [1, 1, 1]
    D = 2

    remaining_forbidden_single = [0, 1, 1]
    remaining_forbidden_pair = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
    ]

    # 如果当前 Plane 占掉 SC0，未来 Plane 就无处可放。
    assert not _candidate_preserves_up_feasibility(
        candidate_sc=0,
        plane_counts=plane_counts,
        D=D,
        remaining_plane_count=1,
        remaining_forbidden_single=remaining_forbidden_single,
        remaining_forbidden_pair=remaining_forbidden_pair,
    )

    # 当前 Plane 放 SC2，则 SC0 留给未来受限 Plane，可行。
    assert _candidate_preserves_up_feasibility(
        candidate_sc=2,
        plane_counts=plane_counts,
        D=D,
        remaining_plane_count=1,
        remaining_forbidden_single=remaining_forbidden_single,
        remaining_forbidden_pair=remaining_forbidden_pair,
    )


def test_feasibility_guard_checks_total_capacity():
    # 放完当前 Plane 后只剩 1 个空槽，但未来还有 2 张 Plane。
    assert not _candidate_preserves_up_feasibility(
        candidate_sc=0,
        plane_counts=[2, 2],
        D=3,
        remaining_plane_count=2,
        remaining_forbidden_single=[0, 0],
        remaining_forbidden_pair=[[0, 0], [0, 0]],
    )
