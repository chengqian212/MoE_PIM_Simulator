"""
四组 Mapping 消融实验自动汇总。

当前四组实验：

1. Naive
   Sequential Pairing
   +
   Round-Robin Mapping

2. Pairing Only
   Trace-aware Pairing
   +
   Round-Robin Mapping

3. Mapping Only
   Sequential Pairing
   +
   Trace-aware Mapping

4. Full
   Trace-aware Pairing
   +
   Trace-aware Mapping


输入：

    results/mappings/
        mapping_baseline_N4_H7168_W4096.json
        mapping_mapping_only_N4_H7168_W4096.json
        mapping_pairing_only_N4_H7168_W4096.json
        mapping_naive_N4_H7168_W4096.json

    results/
        phase_evaluation_summary.json

    results/experiments/
        mapping_only/phase_evaluation_summary.json
        pairing_only/phase_evaluation_summary.json
        naive/phase_evaluation_summary.json


输出：

    results/experiments/ablation_summary.json


指标：

    Pairing Cost
    Mapping Conflict Cost

    Prefill Mean Latency
    Prefill Mean Cycles / Input Token
    Prefill Global Cycles / Input Token

    Decode Mean Cycles / Token
    Decode P50
    Decode P95
    Decode P99
    Decode Max

同时计算：

    相对 Naive 的性能提升百分比

注意：

    latency / cycles 指标越低越好。

因此：

    improvement =
        (Naive - Current)
        / Naive
        * 100%

正数：
    当前方案优于 Naive。

负数：
    当前方案反而更慢。
"""

from __future__ import annotations

import argparse
import json

from dataclasses import (
    asdict,
    dataclass,
)

from pathlib import Path
from typing import Any


# ============================================================
# 路径
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


MAPPING_DIR = (
    RESULTS_DIR
    / "mappings"
)


EXPERIMENT_DIR = (
    RESULTS_DIR
    / "experiments"
)


DEFAULT_FULL_MAPPING = (
    MAPPING_DIR
    / "mapping_baseline_N4_H7168_W4096.json"
)


DEFAULT_MAPPING_ONLY_MAPPING = (
    MAPPING_DIR
    / "mapping_mapping_only_N4_H7168_W4096.json"
)


DEFAULT_PAIRING_ONLY_MAPPING = (
    MAPPING_DIR
    / "mapping_pairing_only_N4_H7168_W4096.json"
)


DEFAULT_NAIVE_MAPPING = (
    MAPPING_DIR
    / "mapping_naive_N4_H7168_W4096.json"
)


DEFAULT_FULL_SUMMARY = (
    RESULTS_DIR
    / "phase_evaluation_summary.json"
)


DEFAULT_MAPPING_ONLY_SUMMARY = (
    EXPERIMENT_DIR
    / "mapping_only"
    / "phase_evaluation_summary.json"
)


DEFAULT_PAIRING_ONLY_SUMMARY = (
    EXPERIMENT_DIR
    / "pairing_only"
    / "phase_evaluation_summary.json"
)


DEFAULT_NAIVE_SUMMARY = (
    EXPERIMENT_DIR
    / "naive"
    / "phase_evaluation_summary.json"
)


DEFAULT_OUTPUT_PATH = (
    EXPERIMENT_DIR
    / "ablation_summary.json"
)


# ============================================================
# 异常
# ============================================================


class AblationSummaryError(
    ValueError
):
    """消融结果汇总失败。"""


# ============================================================
# 数据结构
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class AblationMetrics:

    # --------------------------------------------------------
    # 实验身份
    # --------------------------------------------------------

    experiment: str

    pairing_mode: str

    mapping_mode: str

    # --------------------------------------------------------
    # Mapping 中间指标
    # --------------------------------------------------------

    pairing_cost: int

    mapping_conflict_cost: int

    pre_conflict_cost: int

    down_conflict_cost: int

    # --------------------------------------------------------
    # Prefill
    # --------------------------------------------------------

    prefill_mean_latency: float

    prefill_p50_latency: float

    prefill_p95_latency: float

    prefill_p99_latency: float

    prefill_max_latency: float

    prefill_mean_cycles_per_input_token: float

    prefill_global_cycles_per_input_token: float

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    decode_mean_cycles_per_token: float

    decode_p50_cycles_per_token: float

    decode_p95_cycles_per_token: float

    decode_p99_cycles_per_token: float

    decode_max_cycles_per_token: float


@dataclass(
    frozen=True,
    slots=True,
)
class ImprovementMetrics:

    experiment: str

    # --------------------------------------------------------
    # 相对 Naive 的提升
    #
    # 正数 = 更快
    # 负数 = 更慢
    # --------------------------------------------------------

    prefill_mean_improvement_percent: float

    prefill_cycles_per_input_token_improvement_percent: float

    decode_mean_improvement_percent: float

    decode_p95_improvement_percent: float

    # --------------------------------------------------------
    # 中间 cost 降低比例
    #
    # 正数 = Cost 更低
    # --------------------------------------------------------

    pairing_cost_reduction_percent: float

    mapping_conflict_cost_reduction_percent: float


# ============================================================
# JSON
# ============================================================


def load_json(
    path: Path | str,
) -> dict[str, Any]:

    file_path = (
        Path(path)
        .resolve()
    )

    if not file_path.exists():

        raise AblationSummaryError(
            f"文件不存在：{file_path}"
        )

    try:

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise AblationSummaryError(
            f"无法读取 JSON：{file_path}"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):

        raise AblationSummaryError(
            f"JSON 最外层必须是 dict："
            f"{file_path}"
        )

    return data


# ============================================================
# 安全字段读取
# ============================================================


def require_dict(
    obj: dict,
    key: str,
    *,
    context: str,
) -> dict:

    value = obj.get(
        key
    )

    if not isinstance(
        value,
        dict,
    ):

        raise AblationSummaryError(
            f"{context} 缺少 dict 字段："
            f"{key}"
        )

    return value


def require_number(
    obj: dict,
    key: str,
    *,
    context: str,
) -> float:

    value = obj.get(
        key
    )

    if (
        not isinstance(
            value,
            (
                int,
                float,
            ),
        )
        or
        isinstance(
            value,
            bool,
        )
    ):

        raise AblationSummaryError(
            f"{context} 字段 "
            f"{key} 不是数值。"
        )

    return float(
        value
    )


def require_int(
    obj: dict,
    key: str,
    *,
    context: str,
) -> int:

    value = obj.get(
        key
    )

    if (
        not isinstance(
            value,
            int,
        )
        or
        isinstance(
            value,
            bool,
        )
    ):

        raise AblationSummaryError(
            f"{context} 字段 "
            f"{key} 不是整数。"
        )

    return int(
        value
    )


# ============================================================
# Mapping Cost
# ============================================================


def parse_mapping_metrics(
    payload: dict,
) -> tuple[
    int,
    int,
    int,
    int,
]:

    pairing = (
        require_dict(
            payload,
            "pairing",
            context="Mapping JSON",
        )
    )

    mapping = (
        require_dict(
            payload,
            "subcube_mapping",
            context="Mapping JSON",
        )
    )

    pairing_cost = (
        require_int(
            pairing,
            "total_routed_up_coactivation_cost",
            context="pairing",
        )
    )

    pre_conflict = (
        require_int(
            mapping,
            "pre_conflict_cost",
            context="subcube_mapping",
        )
    )

    down_conflict = (
        require_int(
            mapping,
            "down_conflict_cost",
            context="subcube_mapping",
        )
    )

    total_conflict = (
        require_int(
            mapping,
            "total_conflict_cost",
            context="subcube_mapping",
        )
    )

    # --------------------------------------------------------
    # 一致性检查
    # --------------------------------------------------------

    if (
        pre_conflict
        + down_conflict
        != total_conflict
    ):

        raise AblationSummaryError(
            "Mapping Conflict Cost 不一致："
            f"pre={pre_conflict}, "
            f"down={down_conflict}, "
            f"total={total_conflict}"
        )

    return (
        pairing_cost,
        total_conflict,
        pre_conflict,
        down_conflict,
    )


# ============================================================
# Phase Summary
# ============================================================


def parse_phase_metrics(
    payload: dict,
) -> dict[str, float]:

    prefill = (
        require_dict(
            payload,
            "prefill",
            context="Phase Summary",
        )
    )

    decode = (
        require_dict(
            payload,
            "decode",
            context="Phase Summary",
        )
    )

    # ========================================================
    # Prefill
    # ========================================================

    latency = (
        require_dict(
            prefill,
            "latency_cycles",
            context="Prefill",
        )
    )

    cycles_per_input_token = (
        require_dict(
            prefill,
            "cycles_per_input_token",
            context="Prefill",
        )
    )

    # ========================================================
    # Decode
    # ========================================================

    decode_cycles = (
        require_dict(
            decode,
            "cycles_per_token",
            context="Decode",
        )
    )

    return {
        # ----------------------------------------------------
        # Prefill
        # ----------------------------------------------------

        "prefill_mean_latency":
            require_number(
                latency,
                "mean",
                context="Prefill latency",
            ),

        "prefill_p50_latency":
            require_number(
                latency,
                "p50",
                context="Prefill latency",
            ),

        "prefill_p95_latency":
            require_number(
                latency,
                "p95",
                context="Prefill latency",
            ),

        "prefill_p99_latency":
            require_number(
                latency,
                "p99",
                context="Prefill latency",
            ),

        "prefill_max_latency":
            require_number(
                latency,
                "maximum",
                context="Prefill latency",
            ),

        "prefill_mean_cycles_per_input_token":
            require_number(
                cycles_per_input_token,
                "mean",
                context=(
                    "Prefill "
                    "cycles_per_input_token"
                ),
            ),

        "prefill_global_cycles_per_input_token":
            require_number(
                prefill,
                "global_cycles_per_input_token",
                context="Prefill",
            ),

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        "decode_mean_cycles_per_token":
            require_number(
                decode_cycles,
                "mean",
                context="Decode cycles",
            ),

        "decode_p50_cycles_per_token":
            require_number(
                decode_cycles,
                "p50",
                context="Decode cycles",
            ),

        "decode_p95_cycles_per_token":
            require_number(
                decode_cycles,
                "p95",
                context="Decode cycles",
            ),

        "decode_p99_cycles_per_token":
            require_number(
                decode_cycles,
                "p99",
                context="Decode cycles",
            ),

        "decode_max_cycles_per_token":
            require_number(
                decode_cycles,
                "maximum",
                context="Decode cycles",
            ),
    }


# ============================================================
# 单个实验
# ============================================================


def build_experiment_metrics(
    *,
    experiment: str,

    pairing_mode: str,

    mapping_mode: str,

    mapping_path: Path,

    phase_summary_path: Path,
) -> AblationMetrics:

    mapping_payload = (
        load_json(
            mapping_path
        )
    )

    phase_payload = (
        load_json(
            phase_summary_path
        )
    )

    (
        pairing_cost,
        mapping_conflict_cost,
        pre_conflict_cost,
        down_conflict_cost,
    ) = (
        parse_mapping_metrics(
            mapping_payload
        )
    )

    phase = (
        parse_phase_metrics(
            phase_payload
        )
    )

    return AblationMetrics(
        experiment=(
            experiment
        ),

        pairing_mode=(
            pairing_mode
        ),

        mapping_mode=(
            mapping_mode
        ),

        pairing_cost=(
            pairing_cost
        ),

        mapping_conflict_cost=(
            mapping_conflict_cost
        ),

        pre_conflict_cost=(
            pre_conflict_cost
        ),

        down_conflict_cost=(
            down_conflict_cost
        ),

        prefill_mean_latency=(
            phase[
                "prefill_mean_latency"
            ]
        ),

        prefill_p50_latency=(
            phase[
                "prefill_p50_latency"
            ]
        ),

        prefill_p95_latency=(
            phase[
                "prefill_p95_latency"
            ]
        ),

        prefill_p99_latency=(
            phase[
                "prefill_p99_latency"
            ]
        ),

        prefill_max_latency=(
            phase[
                "prefill_max_latency"
            ]
        ),

        prefill_mean_cycles_per_input_token=(
            phase[
                "prefill_mean_cycles_per_input_token"
            ]
        ),

        prefill_global_cycles_per_input_token=(
            phase[
                "prefill_global_cycles_per_input_token"
            ]
        ),

        decode_mean_cycles_per_token=(
            phase[
                "decode_mean_cycles_per_token"
            ]
        ),

        decode_p50_cycles_per_token=(
            phase[
                "decode_p50_cycles_per_token"
            ]
        ),

        decode_p95_cycles_per_token=(
            phase[
                "decode_p95_cycles_per_token"
            ]
        ),

        decode_p99_cycles_per_token=(
            phase[
                "decode_p99_cycles_per_token"
            ]
        ),

        decode_max_cycles_per_token=(
            phase[
                "decode_max_cycles_per_token"
            ]
        ),
    )


# ============================================================
# Improvement
# ============================================================


def reduction_percent(
    *,
    baseline: float,
    current: float,
) -> float:
    """
    指标越低越好时的改善率。

    baseline = 100
    current  = 80

    improvement = 20%
    """

    if baseline == 0:

        return 0.0

    return (
        (
            baseline
            - current
        )
        /
        baseline
        *
        100.0
    )


def build_improvement(
    *,
    naive: AblationMetrics,
    current: AblationMetrics,
) -> ImprovementMetrics:

    return ImprovementMetrics(
        experiment=(
            current.experiment
        ),

        prefill_mean_improvement_percent=(
            reduction_percent(
                baseline=(
                    naive
                    .prefill_mean_latency
                ),

                current=(
                    current
                    .prefill_mean_latency
                ),
            )
        ),

        prefill_cycles_per_input_token_improvement_percent=(
            reduction_percent(
                baseline=(
                    naive
                    .prefill_mean_cycles_per_input_token
                ),

                current=(
                    current
                    .prefill_mean_cycles_per_input_token
                ),
            )
        ),

        decode_mean_improvement_percent=(
            reduction_percent(
                baseline=(
                    naive
                    .decode_mean_cycles_per_token
                ),

                current=(
                    current
                    .decode_mean_cycles_per_token
                ),
            )
        ),

        decode_p95_improvement_percent=(
            reduction_percent(
                baseline=(
                    naive
                    .decode_p95_cycles_per_token
                ),

                current=(
                    current
                    .decode_p95_cycles_per_token
                ),
            )
        ),

        pairing_cost_reduction_percent=(
            reduction_percent(
                baseline=(
                    naive
                    .pairing_cost
                ),

                current=(
                    current
                    .pairing_cost
                ),
            )
        ),

        mapping_conflict_cost_reduction_percent=(
            reduction_percent(
                baseline=(
                    naive
                    .mapping_conflict_cost
                ),

                current=(
                    current
                    .mapping_conflict_cost
                ),
            )
        ),
    )


# ============================================================
# 打印
# ============================================================


def print_metrics_table(
    experiments: list[
        AblationMetrics
    ],
) -> None:

    print(
        "\n"
        "=============================================================="
    )

    print(
        "                 MoE Mapping Ablation"
    )

    print(
        "=============================================================="
    )

    header = (
        f"{'Experiment':<15}"
        f"{'Pairing':<14}"
        f"{'Mapping':<14}"
        f"{'Prefill':>12}"
        f"{'Decode':>12}"
        f"{'P95':>10}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for item in experiments:

        print(
            f"{item.experiment:<15}"
            f"{item.pairing_mode:<14}"
            f"{item.mapping_mode:<14}"
            f"{item.prefill_mean_latency:>12.2f}"
            f"{item.decode_mean_cycles_per_token:>12.2f}"
            f"{item.decode_p95_cycles_per_token:>10.2f}"
        )


def print_cost_table(
    experiments: list[
        AblationMetrics
    ],
) -> None:

    print(
        "\n"
        "Intermediate Cost Metrics"
    )

    header = (
        f"{'Experiment':<15}"
        f"{'Pairing Cost':>18}"
        f"{'Mapping Conflict':>22}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for item in experiments:

        print(
            f"{item.experiment:<15}"
            f"{item.pairing_cost:>18,d}"
            f"{item.mapping_conflict_cost:>22,d}"
        )

def print_improvement_table(
    improvements: list[
        ImprovementMetrics
    ],
) -> None:

    print(
        "\n"
        "Improvement vs Naive"
    )

    print(
        f"{'Experiment':<14}"
        f"{'Prefill':>10}"
        f"{'Decode':>10}"
        f"{'P95':>10}"
    )

    print("-" * 44)

    for item in improvements:

        print(
            f"{item.experiment:<14}"
            f"{item.prefill_mean_improvement_percent:>9.2f}%"
            f"{item.decode_mean_improvement_percent:>9.2f}%"
            f"{item.decode_p95_improvement_percent:>9.2f}%"
        )

    print(
        "\nCost Reduction vs Naive"
    )

    print(
        f"{'Experiment':<14}"
        f"{'Pairing':>12}"
        f"{'Mapping':>12}"
    )

    print("-" * 38)

    for item in improvements:

        print(
            f"{item.experiment:<14}"
            f"{item.pairing_cost_reduction_percent:>11.2f}%"
            f"{item.mapping_conflict_cost_reduction_percent:>11.2f}%"
        )
# ============================================================
# 自动核心结论
# ============================================================


def print_core_findings(
    *,
    naive: AblationMetrics,
    pairing_only: AblationMetrics,
    mapping_only: AblationMetrics,
    full: AblationMetrics,
) -> None:

    mapping_prefill = (
        reduction_percent(
            baseline=(
                naive
                .prefill_mean_latency
            ),

            current=(
                mapping_only
                .prefill_mean_latency
            ),
        )
    )

    mapping_decode = (
        reduction_percent(
            baseline=(
                naive
                .decode_mean_cycles_per_token
            ),

            current=(
                mapping_only
                .decode_mean_cycles_per_token
            ),
        )
    )

    pairing_prefill = (
        reduction_percent(
            baseline=(
                naive
                .prefill_mean_latency
            ),

            current=(
                pairing_only
                .prefill_mean_latency
            ),
        )
    )

    pairing_decode = (
        reduction_percent(
            baseline=(
                naive
                .decode_mean_cycles_per_token
            ),

            current=(
                pairing_only
                .decode_mean_cycles_per_token
            ),
        )
    )

    full_prefill = (
        reduction_percent(
            baseline=(
                naive
                .prefill_mean_latency
            ),

            current=(
                full
                .prefill_mean_latency
            ),
        )
    )

    full_decode = (
        reduction_percent(
            baseline=(
                naive
                .decode_mean_cycles_per_token
            ),

            current=(
                full
                .decode_mean_cycles_per_token
            ),
        )
    )

    pairing_cost_reduction = (
        reduction_percent(
            baseline=(
                naive
                .pairing_cost
            ),

            current=(
                pairing_only
                .pairing_cost
            ),
        )
    )

    print(
        "\n"
        "=============================================================="
    )

    print(
        "Core Findings"
    )

    print(
        "=============================================================="
    )

    print(
        "\n[Mapping Contribution]"
    )

    print(
        "Naive -> Mapping Only"
    )

    print(
        f"  Prefill Improvement："
        f"{mapping_prefill:.2f}%"
    )

    print(
        f"  Decode Improvement："
        f"{mapping_decode:.2f}%"
    )

    print(
        "\n[Pairing Contribution]"
    )

    print(
        "Naive -> Pairing Only"
    )

    print(
        f"  Prefill Improvement："
        f"{pairing_prefill:.2f}%"
    )

    print(
        f"  Decode Improvement："
        f"{pairing_decode:.2f}%"
    )

    print(
        f"  Pairing Cost Reduction："
        f"{pairing_cost_reduction:.2f}%"
    )

    print(
        "\n[Full Improvement]"
    )

    print(
        "Naive -> Full"
    )

    print(
        f"  Prefill Improvement："
        f"{full_prefill:.2f}%"
    )

    print(
        f"  Decode Improvement："
        f"{full_decode:.2f}%"
    )


# ============================================================
# 保存
# ============================================================


def save_summary(
    *,
    output_path: Path,

    experiments: list[
        AblationMetrics
    ],

    improvements: list[
        ImprovementMetrics
    ],

    source_paths: dict[
        str,
        dict[
            str,
            Path,
        ],
    ],
) -> Path:

    output_path = (
        output_path.resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "ablation_version": 1,

        "scope": (
            "MoE Expert Prefill / Decode "
            "mapping ablation"
        ),

        "experiments": [
            asdict(
                item
            )
            for item
            in experiments
        ],

        "improvements_vs_naive": [
            asdict(
                item
            )
            for item
            in improvements
        ],

        "sources": {
            name: {
                key: str(
                    path.resolve()
                )
                for key, path
                in paths.items()
            }
            for name, paths
            in source_paths.items()
        },
    }

    try:

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

    except OSError as exc:

        raise AblationSummaryError(
            f"无法保存：{output_path}"
        ) from exc

    return output_path


# ============================================================
# Main
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "汇总 MoE Mapping 四组消融实验。"
            )
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    args = (
        parser.parse_args()
    )

    # ========================================================
    # 四组路径
    # ========================================================

    source_paths = {
        "naive": {
            "mapping": (
                DEFAULT_NAIVE_MAPPING
            ),

            "summary": (
                DEFAULT_NAIVE_SUMMARY
            ),
        },

        "pairing_only": {
            "mapping": (
                DEFAULT_PAIRING_ONLY_MAPPING
            ),

            "summary": (
                DEFAULT_PAIRING_ONLY_SUMMARY
            ),
        },

        "mapping_only": {
            "mapping": (
                DEFAULT_MAPPING_ONLY_MAPPING
            ),

            "summary": (
                DEFAULT_MAPPING_ONLY_SUMMARY
            ),
        },

        "full": {
            "mapping": (
                DEFAULT_FULL_MAPPING
            ),

            "summary": (
                DEFAULT_FULL_SUMMARY
            ),
        },
    }

    # ========================================================
    # Build
    # ========================================================

    naive = (
        build_experiment_metrics(
            experiment="Naive",

            pairing_mode="Sequential",

            mapping_mode="Round-Robin",

            mapping_path=(
                source_paths[
                    "naive"
                ][
                    "mapping"
                ]
            ),

            phase_summary_path=(
                source_paths[
                    "naive"
                ][
                    "summary"
                ]
            ),
        )
    )

    pairing_only = (
        build_experiment_metrics(
            experiment="Pairing Only",

            pairing_mode="Trace-aware",

            mapping_mode="Round-Robin",

            mapping_path=(
                source_paths[
                    "pairing_only"
                ][
                    "mapping"
                ]
            ),

            phase_summary_path=(
                source_paths[
                    "pairing_only"
                ][
                    "summary"
                ]
            ),
        )
    )

    mapping_only = (
        build_experiment_metrics(
            experiment="Mapping Only",

            pairing_mode="Sequential",

            mapping_mode="Trace-aware",

            mapping_path=(
                source_paths[
                    "mapping_only"
                ][
                    "mapping"
                ]
            ),

            phase_summary_path=(
                source_paths[
                    "mapping_only"
                ][
                    "summary"
                ]
            ),
        )
    )

    full = (
        build_experiment_metrics(
            experiment="Full",

            pairing_mode="Trace-aware",

            mapping_mode="Trace-aware",

            mapping_path=(
                source_paths[
                    "full"
                ][
                    "mapping"
                ]
            ),

            phase_summary_path=(
                source_paths[
                    "full"
                ][
                    "summary"
                ]
            ),
        )
    )

    experiments = [
        naive,
        pairing_only,
        mapping_only,
        full,
    ]

    # ========================================================
    # Improvements
    # ========================================================

    improvements = [
        build_improvement(
            naive=naive,
            current=item,
        )
        for item
        in experiments
    ]

    # ========================================================
    # Print
    # ========================================================

    print_metrics_table(
        experiments
    )

    print_cost_table(
        experiments
    )

    print_improvement_table(
        improvements
    )

    print_core_findings(
        naive=naive,

        pairing_only=(
            pairing_only
        ),

        mapping_only=(
            mapping_only
        ),

        full=full,
    )

    # ========================================================
    # Save
    # ========================================================

    saved = (
        save_summary(
            output_path=(
                args.output
            ),

            experiments=(
                experiments
            ),

            improvements=(
                improvements
            ),

            source_paths=(
                source_paths
            ),
        )
    )

    print(
        "\nSaved："
        f"{saved}"
    )


if __name__ == "__main__":
    main()