from config import (
    ExecutionRules,
    ModelConfig,
)

from mapping.logical_weight import (
    MATRIX_DOWN,
    MATRIX_GATE,
    MATRIX_UP,
)

from scheduling.layer_scheduler import (
    schedule_layer,
)

from scheduling.runtime_index import (
    RuntimeExpertLocation,
    RuntimeIndex,
    RuntimeLayerIndex,
    RuntimeMatrixLocation,
)


# ============================================================
# Helper
# ============================================================


def make_location(
    *,
    cube_id,
    expert_id,
    shared,
    matrix_name,
    plane_id,
    slot_id,
    sc,
    z,
):

    return RuntimeMatrixLocation(
        cube_id=cube_id,

        layer_id=0,

        expert_id=expert_id,

        is_shared=shared,

        matrix_name=matrix_name,

        logical_plane_id=(
            plane_id
        ),

        physical_plane_id=(
            plane_id
        ),

        slot_id=slot_id,

        subcube_id=sc,

        z=z,
    )


# ============================================================
# 理想情况：
#
# Routed E0：
#     gate/down -> SC0
#     up        -> SC1
#
# Shared E1：
#     gate/down -> SC2
#     up        -> SC3
#
# 4 个 pre task 可以完全并行。
# ============================================================


def build_ideal_index():

    config = ModelConfig(
        num_moe_layers=1,

        routed_experts_per_layer=1,

        experts_per_token=1,

        include_shared_expert=True,
    )

    # Routed E0
    gate_0 = make_location(
        cube_id=0,
        expert_id=0,
        shared=False,
        matrix_name=MATRIX_GATE,
        plane_id=0,
        slot_id=0,
        sc=0,
        z=0,
    )

    down_0 = make_location(
        cube_id=2,
        expert_id=0,
        shared=False,
        matrix_name=MATRIX_DOWN,
        plane_id=0,
        slot_id=1,
        sc=0,
        z=0,
    )

    up_0 = make_location(
        cube_id=1,
        expert_id=0,
        shared=False,
        matrix_name=MATRIX_UP,
        plane_id=2,
        slot_id=4,
        sc=1,
        z=0,
    )

    routed = RuntimeExpertLocation(
        layer_id=0,
        expert_id=0,
        is_shared=False,

        gate=gate_0,
        up=up_0,
        down=down_0,
    )

    # Shared E1
    gate_1 = make_location(
        cube_id=3,
        expert_id=1,
        shared=True,
        matrix_name=MATRIX_GATE,
        plane_id=1,
        slot_id=2,
        sc=2,
        z=0,
    )

    down_1 = make_location(
        cube_id=5,
        expert_id=1,
        shared=True,
        matrix_name=MATRIX_DOWN,
        plane_id=1,
        slot_id=3,
        sc=2,
        z=0,
    )

    up_1 = make_location(
        cube_id=4,
        expert_id=1,
        shared=True,
        matrix_name=MATRIX_UP,
        plane_id=3,
        slot_id=5,
        sc=3,
        z=0,
    )

    shared = RuntimeExpertLocation(
        layer_id=0,
        expert_id=1,
        is_shared=True,

        gate=gate_1,
        up=up_1,
        down=down_1,
    )

    layer = RuntimeLayerIndex(
        layer_id=0,

        experts=(
            routed,
            shared,
        ),
    )

    return RuntimeIndex(
        model_config=config,

        num_subcubes=4,

        subcube_depth=10,

        layers=(
            layer,
        ),
    )


# ============================================================
# Conflict 情况：
#
# Routed gate/down
# Shared gate/down
#
# 都在 SC0。
# ============================================================


def build_conflict_index():

    config = ModelConfig(
        num_moe_layers=1,

        routed_experts_per_layer=1,

        experts_per_token=1,

        include_shared_expert=True,
    )

    routed = RuntimeExpertLocation(
        layer_id=0,

        expert_id=0,

        is_shared=False,

        gate=make_location(
            cube_id=0,
            expert_id=0,
            shared=False,
            matrix_name=MATRIX_GATE,
            plane_id=0,
            slot_id=0,
            sc=0,
            z=0,
        ),

        up=make_location(
            cube_id=1,
            expert_id=0,
            shared=False,
            matrix_name=MATRIX_UP,
            plane_id=2,
            slot_id=4,
            sc=1,
            z=0,
        ),

        down=make_location(
            cube_id=2,
            expert_id=0,
            shared=False,
            matrix_name=MATRIX_DOWN,
            plane_id=0,
            slot_id=1,
            sc=0,
            z=0,
        ),
    )

    shared = RuntimeExpertLocation(
        layer_id=0,

        expert_id=1,

        is_shared=True,

        gate=make_location(
            cube_id=3,
            expert_id=1,
            shared=True,
            matrix_name=MATRIX_GATE,
            plane_id=1,
            slot_id=2,
            sc=0,
            z=1,
        ),

        up=make_location(
            cube_id=4,
            expert_id=1,
            shared=True,
            matrix_name=MATRIX_UP,
            plane_id=3,
            slot_id=5,
            sc=2,
            z=0,
        ),

        down=make_location(
            cube_id=5,
            expert_id=1,
            shared=True,
            matrix_name=MATRIX_DOWN,
            plane_id=1,
            slot_id=3,
            sc=0,
            z=1,
        ),
    )

    return RuntimeIndex(
        model_config=config,

        num_subcubes=3,

        subcube_depth=10,

        layers=(
            RuntimeLayerIndex(
                layer_id=0,

                experts=(
                    routed,
                    shared,
                ),
            ),
        ),
    )


# ============================================================
# Tests
# ============================================================


def test_ideal_layer_cycles():

    index = build_ideal_index()

    result = schedule_layer(
        index=index,

        layer_id=0,

        routed_expert_ids=(
            0,
        ),

        rules=(
            ExecutionRules()
        ),
    )

    # t=0~2:
    #
    # 4 个 gate/up 完全并行
    #
    # t=2~4:
    #
    # 两个 down 完全并行
    assert (
        result.total_cycles
        == 4
    )


def test_task_count():

    index = build_ideal_index()

    result = schedule_layer(
        index=index,

        layer_id=0,

        routed_expert_ids=(
            0,
        ),
    )

    # 1 Routed + 1 Shared
    #
    # 每个 Expert 3 个矩阵
    assert (
        result.task_count
        == 6
    )


def test_down_dependency():

    index = build_ideal_index()

    result = schedule_layer(
        index=index,

        layer_id=0,

        routed_expert_ids=(
            0,
        ),
    )

    for expert_id in (
        0,
        1,
    ):

        gate = result.task(
            expert_id,
            MATRIX_GATE,
        )

        up = result.task(
            expert_id,
            MATRIX_UP,
        )

        down = result.task(
            expert_id,
            MATRIX_DOWN,
        )

        assert (
            down.ready_time
            == max(
                gate.finish_time,
                up.finish_time,
            )
        )

        assert (
            down.dispatch_time
            >= down.ready_time
        )


def test_ideal_switch_counts():

    index = build_ideal_index()

    result = schedule_layer(
        index=index,

        layer_id=0,

        routed_expert_ids=(
            0,
        ),
    )

    # 首先：
    #
    # SC0 gate
    # SC1 up
    # SC2 shared gate
    # SC3 shared up
    #
    # 4 次首次 activation。
    assert (
        result.initial_activation_count
        == 4
    )

    # 之后：
    #
    # SC0 gate -> down
    # SC2 gate -> down
    #
    # 两次真正的 WC switch。
    assert (
        result.switch_count
        == 2
    )


def test_conflict_increases_cycles():

    ideal = schedule_layer(
        index=build_ideal_index(),

        layer_id=0,

        routed_expert_ids=(
            0,
        ),
    )

    conflict = schedule_layer(
        index=build_conflict_index(),

        layer_id=0,

        routed_expert_ids=(
            0,
        ),
    )

    assert (
        conflict.total_cycles
        > ideal.total_cycles
    )

    assert (
        conflict.total_cycles
        == 8
    )


def test_same_subcube_tasks_never_overlap():

    index = build_conflict_index()

    result = schedule_layer(
        index=index,

        layer_id=0,

        routed_expert_ids=(
            0,
        ),
    )

    for sc in range(
        index.num_subcubes
    ):

        tasks = sorted(
            (
                task

                for task
                in result.tasks

                if (
                    task.subcube_id
                    == sc
                )
            ),

            key=lambda task: (
                task.dispatch_time
            ),
        )

        for (
            first,
            second,
        ) in zip(
            tasks,
            tasks[1:],
        ):

            assert (
                first.finish_time
                <= second.dispatch_time
            )


def test_initial_activation_cost_can_be_disabled():

    index = build_ideal_index()

    result = schedule_layer(
        index=index,

        layer_id=0,

        routed_expert_ids=(
            0,
        ),

        charge_initial_activation=False,
    )

    # gate/up：
    #
    # 只有 compute = 1
    #
    # down：
    #
    # switch 1 + compute 1
    #
    # 所以：
    #
    # 1 + 2 = 3
    assert (
        result.total_cycles
        == 3
    )

    assert (
        result.initial_activation_count
        == 0
    )