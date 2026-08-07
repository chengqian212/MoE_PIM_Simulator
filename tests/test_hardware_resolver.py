# tests/test_hardware_resolver.py

from math import ceil

import pytest

from evaluation.hardware_resolver import (
    HardwareResolutionError,
    calculate_required_depth,
    resolve_hardware,
    resolve_all_n,
    filter_valid_hardware,
)

from evaluation.spatial_metrics import (
    calculate_plane_lower_bound,
    evaluate_spatial_metrics,
)

from partition.partition_generator import (
    generate_original_template,
)

from packing.anonymous_packer import (
    pack_anonymous_blocks,
)


# ============================================================
# D = ceil(P / N^2)
# ============================================================


@pytest.mark.parametrize(
    "P,N,expected_D",
    [
        (1, 2, 1),
        (4, 2, 1),
        (5, 2, 2),
        (16, 2, 4),
        (17, 2, 5),

        (1, 3, 1),
        (9, 3, 1),
        (10, 3, 2),
        (17, 3, 2),
        (18, 3, 2),
        (19, 3, 3),

        (1, 4, 1),
        (16, 4, 1),
        (17, 4, 2),
        (31, 4, 2),
        (32, 4, 2),
        (33, 4, 3),
    ],
)
def test_calculate_required_depth(
    P,
    N,
    expected_D,
):
    """
    验证核心公式：

        D = ceil(P / N^2)
    """

    actual_D = (
        calculate_required_depth(
            used_plane_count=P,
            N=N,
        )
    )

    assert actual_D == expected_D


# ============================================================
# P=17 示例
# ============================================================


def test_p17_all_n():
    """
    P=17 时：

        N=2:
            D=5
            Q=20

        N=3:
            D=2
            Q=18

        N=4:
            D=2
            Q=32
    """

    P = 17

    H = 4096
    W = 4096

    # 只要 S <= 所有配置容量即可。
    S = (
        17
        * H
        * W
    )

    results = resolve_all_n(
        H=H,
        W=W,
        used_plane_count=P,
        total_weight_area=S,
        n_values=(2, 3, 4),
    )

    by_n = {
        result.hardware.N: result
        for result in results
    }

    assert by_n[2].hardware.D == 5
    assert by_n[2].total_plane_slots == 20

    assert by_n[3].hardware.D == 2
    assert by_n[3].total_plane_slots == 18

    assert by_n[4].hardware.D == 2
    assert by_n[4].total_plane_slots == 32


# ============================================================
# Q >= P
# ============================================================


@pytest.mark.parametrize(
    "P,N",
    [
        (1, 2),
        (4, 2),
        (5, 2),
        (17, 2),
        (17, 3),
        (17, 4),
        (100, 2),
        (100, 3),
        (100, 4),
    ],
)
def test_total_plane_slots_always_cover_p(
    P,
    N,
):
    """
    由于：

        D = ceil(P/N²)

    必须始终：

        Q = N²D >= P
    """

    H = 4096
    W = 4096

    S = P * H * W

    result = resolve_hardware(
        N=N,
        H=H,
        W=W,
        used_plane_count=P,
        total_weight_area=S,
    )

    assert (
        result.total_plane_slots
        >= P
    )


# ============================================================
# Q-P
# ============================================================


def test_empty_plane_slots():
    """
    P=17, N=4：

        D=2
        Q=32
        Q-P=15
    """

    H = 4096
    W = 4096

    result = resolve_hardware(
        N=4,
        H=H,
        W=W,
        used_plane_count=17,
        total_weight_area=(
            17 * H * W
        ),
    )

    assert result.hardware.D == 2

    assert (
        result.total_plane_slots
        == 32
    )

    assert (
        result.empty_plane_slots
        == 15
    )

    assert (
        result.empty_plane_capacity
        == 15 * H * W
    )


# ============================================================
# C = N² D H W
# ============================================================


def test_total_capacity_formula():
    """
    检查总容量公式：

        C = N² × D × H × W
    """

    N = 3
    H = 8192
    W = 4096
    P = 20

    S = (
        P
        * H
        * W
    )

    result = resolve_hardware(
        N=N,
        H=H,
        W=W,
        used_plane_count=P,
        total_weight_area=S,
    )

    D = ceil(
        P / (N * N)
    )

    expected_capacity = (
        N
        * N
        * D
        * H
        * W
    )

    assert (
        result.hardware.D
        == D
    )

    assert (
        result.total_capacity
        == expected_capacity
    )


# ============================================================
# 合法容量范围
# ============================================================


def test_valid_capacity():
    """
    构造：

        S <= C <= 2S

    应 valid=True。
    """

    H = 4096
    W = 4096

    P = 17
    N = 3

    # N=3:
    #
    # D=2
    # Q=18
    # C=18HW
    #
    # 令 S=17HW：
    #
    # S < C < 2S

    S = (
        17
        * H
        * W
    )

    result = resolve_hardware(
        N=N,
        H=H,
        W=W,
        used_plane_count=P,
        total_weight_area=S,
    )

    assert result.enough_capacity
    assert result.within_double_capacity
    assert result.valid


def test_invalid_over_double_capacity():
    """
    如果：

        C > 2S

    则必须判定非法。

    P=17, N=4：

        Q=32
        C=32HW

    如果：
        S=15HW

    那么：
        2S=30HW

    所以：
        C > 2S
    """

    H = 4096
    W = 4096

    result = resolve_hardware(
        N=4,
        H=H,
        W=W,
        used_plane_count=17,
        total_weight_area=(
            15 * H * W
        ),
    )

    assert (
        result.within_double_capacity
        is False
    )

    assert result.valid is False


# ============================================================
# filter_valid_hardware
# ============================================================


def test_filter_valid_hardware():
    """
    使用 P=17，S=17HW。

    N=2：
        Q=20
        合法

    N=3：
        Q=18
        合法

    N=4：
        Q=32
        C/S=32/17 < 2
        仍合法

    因此三个都应保留。
    """

    H = 4096
    W = 4096
    P = 17

    S = (
        17
        * H
        * W
    )

    results = resolve_all_n(
        H=H,
        W=W,
        used_plane_count=P,
        total_weight_area=S,
    )

    valid_results = (
        filter_valid_hardware(
            results
        )
    )

    assert len(valid_results) == 3


# ============================================================
# 输入非法值
# ============================================================


@pytest.mark.parametrize(
    "P,N",
    [
        (0, 2),
        (-1, 2),
        (10, 1),
        (10, 5),
    ],
)
def test_invalid_depth_inputs(
    P,
    N,
):
    """
    非法 P 或 N 必须抛异常。
    """

    with pytest.raises(
        HardwareResolutionError
    ):
        calculate_required_depth(
            used_plane_count=P,
            N=N,
        )


# ============================================================
# P_lower
# ============================================================


def test_plane_lower_bound():
    """
    如果：

        S = 10.5 个 Plane 的面积

    那么：

        P_lower = 11
    """

    H = 4096
    W = 4096

    plane_area = H * W

    S = (
        10 * plane_area
        + plane_area // 2
    )

    P_lower = (
        calculate_plane_lower_bound(
            total_weight_area=S,
            H=H,
            W=W,
        )
    )

    assert P_lower == 11


# ============================================================
# 与真实 PackingResult 联合测试
# ============================================================


def test_hardware_with_real_packing_result():
    """
    真正执行：

        Partition
        ↓
        Anonymous Packing
        ↓
        P
        ↓
        Hardware Resolution
        ↓
        Spatial Metrics

    测试前三步模块能完整串通。
    """

    H = 4096
    W = 4096

    matrix_count = 8

    template = (
        generate_original_template(
            matrix_rows=7168,
            matrix_cols=2048,
            H=H,
            W=W,
        )
    )

    packing = (
        pack_anonymous_blocks(
            template=template,
            matrix_count=matrix_count,
            H=H,
            W=W,
            allow_rotation=True,
        )
    )

    S = (
        matrix_count
        * 7168
        * 2048
    )

    assert (
        packing.total_block_area
        == S
    )

    hardware_results = (
        resolve_all_n(
            H=H,
            W=W,
            used_plane_count=(
                packing.plane_count
            ),
            total_weight_area=S,
        )
    )

    for hardware_result in (
        hardware_results
    ):

        metrics = (
            evaluate_spatial_metrics(
                packing=packing,
                hardware_result=(
                    hardware_result
                ),
            )
        )

        assert (
            metrics.used_plane_count
            == packing.plane_count
        )

        assert (
            metrics.total_plane_slots
            >= metrics.used_plane_count
        )

        assert (
            metrics.total_capacity
            >= S
        )


# ============================================================
# 空间浪费分解公式
# ============================================================


def test_space_waste_decomposition():
    """
    验证核心公式：

        C-S
        =
        (P*H*W-S)
        +
        (Q-P)*H*W
    """

    H = 4096
    W = 4096

    matrix_count = 10

    template = (
        generate_original_template(
            matrix_rows=7168,
            matrix_cols=2048,
            H=H,
            W=W,
        )
    )

    packing = (
        pack_anonymous_blocks(
            template=template,
            matrix_count=matrix_count,
            H=H,
            W=W,
            allow_rotation=True,
        )
    )

    S = packing.total_block_area
    P = packing.plane_count

    result = resolve_hardware(
        N=3,
        H=H,
        W=W,
        used_plane_count=P,
        total_weight_area=S,
    )

    metrics = evaluate_spatial_metrics(
        packing=packing,
        hardware_result=result,
    )

    left = (
        metrics.total_capacity
        - S
    )

    right = (
        metrics.internal_fragmentation
        + metrics.empty_plane_capacity
    )

    assert left == right


# ============================================================
# packing utilization >= hardware utilization
# ============================================================


def test_packing_utilization_not_lower_than_hardware_utilization():
    """
    二维装箱只考虑实际使用的 P 个 Plane：

        U_packing = S/(PHW)

    整体硬件还包括 Q-P 个完整空 Plane：

        U_hardware = S/(QHW)

    因为：

        Q >= P

    所以必须：

        U_packing >= U_hardware
    """

    H = 4096
    W = 4096

    matrix_count = 10

    template = (
        generate_original_template(
            matrix_rows=7168,
            matrix_cols=2048,
            H=H,
            W=W,
        )
    )

    packing = pack_anonymous_blocks(
        template=template,
        matrix_count=matrix_count,
        H=H,
        W=W,
        allow_rotation=True,
    )

    result = resolve_hardware(
        N=4,
        H=H,
        W=W,
        used_plane_count=(
            packing.plane_count
        ),
        total_weight_area=(
            packing.total_block_area
        ),
    )

    metrics = evaluate_spatial_metrics(
        packing=packing,
        hardware_result=result,
    )

    assert (
        metrics.packing_utilization
        >= metrics.hardware_utilization
    )


# ============================================================
# Q=P 时两个利用率应该相等
# ============================================================


def test_utilization_equal_when_q_equals_p():
    """
    如果刚好：

        Q = P

    则没有完整空 Plane。

    所以：

        U_packing = U_hardware
    """

    H = 4096
    W = 4096

    # 人工用 P=9，N=3：
    #
    # D=1
    # Q=9

    P = 9
    N = 3

    # 为了测试公式，设 S=8个Plane面积。
    S = (
        8
        * H
        * W
    )

    result = resolve_hardware(
        N=N,
        H=H,
        W=W,
        used_plane_count=P,
        total_weight_area=S,
    )

    assert (
        result.total_plane_slots
        == P
    )

    packing_utilization = (
        S
        / (
            P
            * H
            * W
        )
    )

    assert (
        packing_utilization
        == result.hardware_utilization
    )


# ============================================================
# D 是被动变量
# ============================================================


def test_depth_is_minimum_required():
    """
    检查当前 resolver 得到的一定是最小 D。

    如果 D>1：

        使用 D-1 时容量的 Plane 数必须 < P。
    """

    P = 37
    N = 3

    D = calculate_required_depth(
        used_plane_count=P,
        N=N,
    )

    assert (
        N * N * D
        >= P
    )

    if D > 1:

        assert (
            N
            * N
            * (D - 1)
            < P
        )