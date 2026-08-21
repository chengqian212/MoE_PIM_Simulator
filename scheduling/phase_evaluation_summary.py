"""
Prefill + Decode 统一汇总。

输入：

    results/prefill/prefill_evaluation.json
    results/decode/decode_fast_evaluation.json

输出：

    results/phase_evaluation_summary.json

作用：

    统一整理当前 Mapping + 当前 Scheduler 下：

        1. Prefill Baseline
        2. Decode Baseline

方便后续：

    - WebUI 展示
    - 周报 / 汇报
    - 后续优化前后的统一对比
    - 自动读取核心指标

------------------------------------------------------------

注意指标范围：

Prefill：
    MoE Expert Prefill only
    不等于完整 TTFT

Decode：
    MoE Expert Decode only
    不等于完整 TPOT

------------------------------------------------------------

当前正式数据口径：

Prefill：
    每个 JSON 的 segment0

Decode：
    每个 JSON 的 segment1+

------------------------------------------------------------

当前正式结果应来自：

Prefill：
    exact prefill_scheduler

Decode：
    fast_exact-validated
    并已用 exact token_scheduler 做前若干 Token 校验
"""

from __future__ import annotations

import argparse
import json

from dataclasses import (
    asdict,
    dataclass,
)

from pathlib import Path

from mapping.trace_split import (
    EVALUATION_SUBSET,
    TRACE_SUBSETS,
    manifest_protocol_summary,
)


# ============================================================
# 默认路径
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


DEFAULT_PREFILL_PATH = (
    PROJECT_ROOT
    / "results"
    / "prefill"
    / "prefill_evaluation.json"
)


DEFAULT_DECODE_PATH = (
    PROJECT_ROOT
    / "results"
    / "decode"
    / "decode_fast_evaluation.json"
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "phase_evaluation_summary.json"
)


# ============================================================
# 异常
# ============================================================


class PhaseEvaluationSummaryError(
    ValueError
):
    pass


# ============================================================
# 数据结构
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class DistributionSummary:

    count: int

    minimum: float

    mean: float

    p50: float

    p95: float

    p99: float

    maximum: float


@dataclass(
    frozen=True,
    slots=True,
)
class PrefillPhaseSummary:

    batch_count: int

    total_input_tokens: int

    prompt_length: DistributionSummary

    latency_cycles: DistributionSummary

    cycles_per_input_token: DistributionSummary

    global_cycles_per_input_token: float

    global_input_tokens_per_cycle: float

    prompt_length_latency_pearson: float


@dataclass(
    frozen=True,
    slots=True,
)
class DecodePhaseSummary:

    token_count: int

    scheduler_mode: str

    exact_checked_tokens: int

    cycles_per_token: DistributionSummary


@dataclass(
    frozen=True,
    slots=True,
)
class PhaseEvaluationSummary:

    summary_version: int

    scope: str

    prefill: PrefillPhaseSummary

    decode: DecodePhaseSummary


# ============================================================
# JSON 读取
# ============================================================


def load_json(
    path: Path | str,
) -> dict:

    file_path = (
        Path(
            path
        )
        .resolve()
    )

    if not (
        file_path.exists()
    ):

        raise PhaseEvaluationSummaryError(
            f"文件不存在：{file_path}"
        )

    try:

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = (
                json.load(
                    file
                )
            )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise PhaseEvaluationSummaryError(
            f"无法读取 JSON：{file_path}"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):

        raise PhaseEvaluationSummaryError(
            f"{file_path} "
            "JSON 最外层必须是 dict。"
        )

    return data


# ============================================================
# 通用字段读取
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

        raise PhaseEvaluationSummaryError(
            f"{context}: "
            f"缺少 dict 字段 `{key}`。"
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

        raise PhaseEvaluationSummaryError(
            f"{context}: "
            f"字段 `{key}` 不是数值。"
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

        raise PhaseEvaluationSummaryError(
            f"{context}: "
            f"字段 `{key}` 不是整数。"
        )

    return value


def require_string(
    obj: dict,
    key: str,
    *,
    context: str,
) -> str:

    value = obj.get(
        key
    )

    if not isinstance(
        value,
        str,
    ):

        raise PhaseEvaluationSummaryError(
            f"{context}: "
            f"字段 `{key}` 不是字符串。"
        )

    return value


# ============================================================
# Distribution
# ============================================================


def parse_distribution(
    obj: dict,
    *,
    context: str,
) -> DistributionSummary:

    return DistributionSummary(
        count=(
            int(
                require_number(
                    obj,
                    "count",
                    context=context,
                )
            )
        ),

        minimum=(
            require_number(
                obj,
                "minimum",
                context=context,
            )
        ),

        mean=(
            require_number(
                obj,
                "mean",
                context=context,
            )
        ),

        p50=(
            require_number(
                obj,
                "p50",
                context=context,
            )
        ),

        p95=(
            require_number(
                obj,
                "p95",
                context=context,
            )
        ),

        p99=(
            require_number(
                obj,
                "p99",
                context=context,
            )
        ),

        maximum=(
            require_number(
                obj,
                "maximum",
                context=context,
            )
        ),
    )


# ============================================================
# Prefill
# ============================================================


def parse_prefill_summary(
    payload: dict,
) -> PrefillPhaseSummary:

    summary = (
        require_dict(
            payload,
            "summary",
            context=(
                "Prefill JSON"
            ),
        )
    )

    prompt_tokens = (
        require_dict(
            summary,
            "prompt_tokens",
            context=(
                "Prefill summary"
            ),
        )
    )

    total_cycles = (
        require_dict(
            summary,
            "total_cycles",
            context=(
                "Prefill summary"
            ),
        )
    )

    cycles_per_input_token = (
        require_dict(
            summary,
            "cycles_per_input_token",
            context=(
                "Prefill summary"
            ),
        )
    )

    result = (
        PrefillPhaseSummary(
            batch_count=(
                require_int(
                    summary,
                    "batch_count",
                    context=(
                        "Prefill summary"
                    ),
                )
            ),

            total_input_tokens=(
                require_int(
                    summary,
                    "total_input_tokens",
                    context=(
                        "Prefill summary"
                    ),
                )
            ),

            prompt_length=(
                parse_distribution(
                    prompt_tokens,
                    context=(
                        "Prefill prompt_tokens"
                    ),
                )
            ),

            latency_cycles=(
                parse_distribution(
                    total_cycles,
                    context=(
                        "Prefill total_cycles"
                    ),
                )
            ),

            cycles_per_input_token=(
                parse_distribution(
                    cycles_per_input_token,
                    context=(
                        "Prefill cycles_per_input_token"
                    ),
                )
            ),

            global_cycles_per_input_token=(
                require_number(
                    summary,
                    "global_cycles_per_input_token",
                    context=(
                        "Prefill summary"
                    ),
                )
            ),

            global_input_tokens_per_cycle=(
                require_number(
                    summary,
                    "global_input_tokens_per_cycle",
                    context=(
                        "Prefill summary"
                    ),
                )
            ),

            prompt_length_latency_pearson=(
                require_number(
                    summary,
                    "prompt_length_latency_pearson",
                    context=(
                        "Prefill summary"
                    ),
                )
            ),
        )
    )

    # ========================================================
    # 基本一致性
    # ========================================================

    if (
        result
        .prompt_length
        .count
        != result
        .batch_count
    ):

        raise PhaseEvaluationSummaryError(
            "Prefill："
            "Prompt Length Count "
            "与 batch_count 不一致。"
        )

    if (
        result
        .latency_cycles
        .count
        != result
        .batch_count
    ):

        raise PhaseEvaluationSummaryError(
            "Prefill："
            "Latency Count "
            "与 batch_count 不一致。"
        )

    return result


# ============================================================
# Decode
# ============================================================


def parse_decode_summary(
    payload: dict,
) -> DecodePhaseSummary:

    summary = (
        require_dict(
            payload,
            "summary",
            context=(
                "Decode JSON"
            ),
        )
    )

    cycles_per_token = (
        require_dict(
            summary,
            "cycles_per_token",
            context=(
                "Decode summary"
            ),
        )
    )

    scheduler_mode = (
        require_string(
            summary,
            "scheduler_mode",
            context=(
                "Decode summary"
            ),
        )
    )

    result = (
        DecodePhaseSummary(
            token_count=(
                require_int(
                    summary,
                    "token_count",
                    context=(
                        "Decode summary"
                    ),
                )
            ),

            scheduler_mode=(
                scheduler_mode
            ),

            exact_checked_tokens=(
                require_int(
                    summary,
                    "exact_checked_tokens",
                    context=(
                        "Decode summary"
                    ),
                )
            ),

            cycles_per_token=(
                parse_distribution(
                    cycles_per_token,
                    context=(
                        "Decode cycles_per_token"
                    ),
                )
            ),
        )
    )

    if (
        result
        .cycles_per_token
        .count
        != result
        .token_count
    ):

        raise PhaseEvaluationSummaryError(
            "Decode："
            "Cycles Count "
            "与 token_count 不一致。"
        )

    if (
        result.scheduler_mode
        != "fast_exact-validated"
    ):

        raise PhaseEvaluationSummaryError(
            "Decode scheduler_mode "
            "不是 fast_exact-validated："
            f"{result.scheduler_mode}"
        )

    return result


# ============================================================
# 构建总结果
# ============================================================


def build_phase_summary(
    *,
    prefill_payload: dict,
    decode_payload: dict,
) -> PhaseEvaluationSummary:

    prefill = (
        parse_prefill_summary(
            prefill_payload
        )
    )

    decode = (
        parse_decode_summary(
            decode_payload
        )
    )

    return PhaseEvaluationSummary(
        summary_version=1,

        scope=(
            "MoE Expert phase evaluation only; "
            "Prefill is not full TTFT, "
            "Decode is not full TPOT"
        ),

        prefill=(
            prefill
        ),

        decode=(
            decode
        ),
    )


# ============================================================
# Print
# ============================================================


def print_phase_summary(
    summary: PhaseEvaluationSummary,
) -> None:

    prefill = (
        summary.prefill
    )

    decode = (
        summary.decode
    )

    print(
        "\n"
        "========== MoE Phase Evaluation Summary =========="
    )

    # ========================================================
    # Prefill
    # ========================================================

    print(
        "\n[Prefill]"
    )

    print(
        f"Batches："
        f"{prefill.batch_count}"
    )

    print(
        f"Total Input Tokens："
        f"{prefill.total_input_tokens}"
    )

    print(
        "Prompt Length："
        f"mean="
        f"{prefill.prompt_length.mean:.2f}, "
        f"p50="
        f"{prefill.prompt_length.p50:.2f}, "
        f"p95="
        f"{prefill.prompt_length.p95:.2f}, "
        f"p99="
        f"{prefill.prompt_length.p99:.2f}, "
        f"max="
        f"{prefill.prompt_length.maximum:.2f}"
    )

    print(
        "MoE Prefill Latency："
        f"mean="
        f"{prefill.latency_cycles.mean:.2f}, "
        f"p50="
        f"{prefill.latency_cycles.p50:.2f}, "
        f"p95="
        f"{prefill.latency_cycles.p95:.2f}, "
        f"p99="
        f"{prefill.latency_cycles.p99:.2f}, "
        f"max="
        f"{prefill.latency_cycles.maximum:.2f}"
    )

    print(
        "Mean Cycles / Input Token："
        f"{prefill.cycles_per_input_token.mean:.4f}"
    )

    print(
        "Global Weighted Cycles / Input Token："
        f"{prefill.global_cycles_per_input_token:.4f}"
    )

    print(
        "Prompt Length vs Latency Pearson："
        f"{prefill.prompt_length_latency_pearson:.4f}"
    )

    # ========================================================
    # Decode
    # ========================================================

    print(
        "\n[Decode]"
    )

    print(
        f"Tokens："
        f"{decode.token_count}"
    )

    print(
        f"Scheduler："
        f"{decode.scheduler_mode}"
    )

    print(
        "FAST == EXACT Checked Tokens："
        f"{decode.exact_checked_tokens}"
    )

    print(
        "MoE Decode Cycles / Token："
        f"mean="
        f"{decode.cycles_per_token.mean:.2f}, "
        f"p50="
        f"{decode.cycles_per_token.p50:.2f}, "
        f"p95="
        f"{decode.cycles_per_token.p95:.2f}, "
        f"p99="
        f"{decode.cycles_per_token.p99:.2f}, "
        f"max="
        f"{decode.cycles_per_token.maximum:.2f}"
    )

    # ========================================================
    # 最核心两行
    # ========================================================

    print(
        "\n[Core Metrics]"
    )

    print(
        "MoE Prefill Mean Latency："
        f"{prefill.latency_cycles.mean:.2f} cycles"
    )

    print(
        "MoE Decode Mean Cycles / Token："
        f"{decode.cycles_per_token.mean:.2f} cycles/token"
    )

    print(
        "\nScope："
        "以上仅为 MoE Expert 部分，"
        "不包含 Attention / KV Cache 等。"
    )


# ============================================================
# 保存
# ============================================================


def save_phase_summary(
    *,
    output_path: Path | str,

    summary: PhaseEvaluationSummary,

    prefill_path: Path | str,

    decode_path: Path | str,

    trace_manifest: Path | str | None = None,

    evaluation_subset: str = EVALUATION_SUBSET,
) -> Path:

    output = (
        Path(
            output_path
        )
        .resolve()
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "summary_version": (
            summary.summary_version
        ),

        "scope": (
            summary.scope
        ),

        "protocol": (
            manifest_protocol_summary(
                manifest_path=trace_manifest,
                evaluation_subset=evaluation_subset,
            )
            if trace_manifest is not None
            else {
                "manifest": None,
                "evaluation_subset": "all",
            }
        ),

        "sources": {
            "prefill": str(
                Path(
                    prefill_path
                ).resolve()
            ),

            "decode": str(
                Path(
                    decode_path
                ).resolve()
            ),
        },

        "prefill": (
            asdict(
                summary.prefill
            )
        ),

        "decode": (
            asdict(
                summary.decode
            )
        ),
    }

    try:

        with output.open(
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

        raise PhaseEvaluationSummaryError(
            f"无法保存：{output}"
        ) from exc

    return output


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "合并 Prefill 和 Decode "
                "正式评估结果。"
            )
        )
    )

    parser.add_argument(
        "--prefill",
        type=Path,
        default=(
            DEFAULT_PREFILL_PATH
        ),
    )

    parser.add_argument(
        "--decode",
        type=Path,
        default=(
            DEFAULT_DECODE_PATH
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--trace-manifest",
        type=Path,
        default=None,
        help="正式实验的 Profile/Held-out split manifest。",
    )

    parser.add_argument(
        "--evaluation-subset",
        choices=TRACE_SUBSETS,
        default=EVALUATION_SUBSET,
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    prefill_payload = (
        load_json(
            args.prefill
        )
    )

    decode_payload = (
        load_json(
            args.decode
        )
    )

    summary = (
        build_phase_summary(
            prefill_payload=(
                prefill_payload
            ),

            decode_payload=(
                decode_payload
            ),
        )
    )

    print_phase_summary(
        summary
    )

    if not (
        args.no_save
    ):

        saved = (
            save_phase_summary(
                output_path=(
                    args.output
                ),

                summary=(
                    summary
                ),

                prefill_path=(
                    args.prefill
                ),

                decode_path=(
                    args.decode
                ),

                trace_manifest=args.trace_manifest,
                evaluation_subset=args.evaluation_subset,
            )
        )

        print(
            "\nSaved："
            f"{saved}"
        )


if __name__ == "__main__":
    main()
