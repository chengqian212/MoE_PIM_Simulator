"""
一键运行 Prefill + Decode 正式评估。

作用：

    把当前已经验证完成的三步串起来：

        1. scheduling.prefill_evaluator
        2. scheduling.decode_fast_evaluator
        3. scheduling.phase_evaluation_summary

以后：

    - 换新的 Mapping
    - 修改调度策略
    - 重新做 Baseline
    - 做后续优化版本

都可以从这个统一入口重新生成一套
Prefill + Decode 结果。

------------------------------------------------------------

默认正式全量：

    Prefill:
        全部 2020 个 Prefill Batch

    Decode:
        全部纯 Decode Token
        默认 exact-check 前 100 个 Token

    Summary:
        自动合并两阶段结果

------------------------------------------------------------

快速检查：

    python -m scheduling.run_phase_evaluation --smoke

等价于大致：

    Prefill:
        10 Batch

    Decode:
        1000 Token
        前 100 Token FAST == EXACT

------------------------------------------------------------

已有结果直接复用：

    python -m scheduling.run_phase_evaluation --reuse-existing

若 Prefill / Decode JSON 已存在，
则跳过对应耗时评估，
直接生成 / 更新总 Summary。

------------------------------------------------------------

指标范围：

Prefill:
    MoE Expert Prefill only
    不是完整 TTFT

Decode:
    MoE Expert Decode only
    不是完整 TPOT
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from dataclasses import dataclass
from pathlib import Path


# ============================================================
# 路径
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


DEFAULT_MAPPING_PATH = (
    PROJECT_ROOT
    / "results"
    / "mappings"
    / "mapping_baseline_N4_H7168_W4096.json"
)


DEFAULT_PREFILL_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "prefill"
    / "prefill_evaluation.json"
)


DEFAULT_DECODE_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "decode"
    / "decode_fast_evaluation.json"
)


DEFAULT_SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "phase_evaluation_summary.json"
)


# ============================================================
# 异常
# ============================================================


class PhaseEvaluationRunnerError(
    RuntimeError
):
    pass


# ============================================================
# Stage 结果
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class StageRunResult:

    name: str

    skipped: bool

    elapsed_seconds: float

    output_path: Path | None


# ============================================================
# 工具函数
# ============================================================


def format_seconds(
    seconds: float,
) -> str:
    """
    只用于显示已完成耗时，
    不做未来运行时间预测。
    """

    if seconds < 60:

        return (
            f"{seconds:.2f}s"
        )

    minutes = int(
        seconds
        // 60
    )

    remain = (
        seconds
        - minutes * 60
    )

    if minutes < 60:

        return (
            f"{minutes}m "
            f"{remain:.1f}s"
        )

    hours = (
        minutes
        // 60
    )

    minutes = (
        minutes
        % 60
    )

    return (
        f"{hours}h "
        f"{minutes}m "
        f"{remain:.1f}s"
    )


def run_command(
    *,
    name: str,
    command: list[str],
    output_path: Path | None,
) -> StageRunResult:

    print(
        "\n"
        "========================================"
    )

    print(
        f"[Run] {name}"
    )

    print(
        "Command："
    )

    print(
        "  "
        +
        " ".join(
            command
        )
    )

    print(
        "========================================"
    )

    start = (
        time.perf_counter()
    )

    completed = (
        subprocess.run(
            command,

            cwd=(
                PROJECT_ROOT
            ),

            check=False,
        )
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if (
        completed.returncode
        != 0
    ):

        raise PhaseEvaluationRunnerError(
            f"{name} 运行失败，"
            f"returncode="
            f"{completed.returncode}。"
        )

    if (
        output_path
        is not None
        and
        not output_path.exists()
    ):

        raise PhaseEvaluationRunnerError(
            f"{name} 已结束，"
            "但没有找到预期输出文件："
            f"{output_path}"
        )

    print(
        f"\n[Done] {name} "
        f"elapsed="
        f"{format_seconds(elapsed)}"
    )

    return (
        StageRunResult(
            name=name,

            skipped=False,

            elapsed_seconds=(
                elapsed
            ),

            output_path=(
                output_path
            ),
        )
    )


def skip_stage(
    *,
    name: str,
    output_path: Path,
) -> StageRunResult:

    print(
        f"\n[Reuse] {name}"
    )

    print(
        f"  Existing："
        f"{output_path}"
    )

    return (
        StageRunResult(
            name=name,

            skipped=True,

            elapsed_seconds=0.0,

            output_path=(
                output_path
            ),
        )
    )


# ============================================================
# Prefill
# ============================================================


def run_prefill(
    *,
    python_executable: str,

    mapping: Path,

    output_path: Path,

    smoke: bool,

    prefill_max_batches: int | None,

    reuse_existing: bool,

    progress_every: int,
) -> StageRunResult:

    if (
        reuse_existing
        and
        output_path.exists()
    ):

        return (
            skip_stage(
                name=(
                    "Prefill Evaluation"
                ),

                output_path=(
                    output_path
                ),
            )
        )

    command = [
        python_executable,
        "-m",
        "scheduling.prefill_evaluator",

        "--mapping",
        str(
            mapping
        ),

        "--output",
        str(
            output_path
        ),

        "--progress-every",
        str(
            progress_every
        ),
    ]

    # ========================================================
    # max-batches 优先级：
    #
    # 显式参数 > smoke > 全量
    # ========================================================

    if (
        prefill_max_batches
        is not None
    ):

        command.extend(
            [
                "--max-batches",
                str(
                    prefill_max_batches
                ),
            ]
        )

    elif smoke:

        command.extend(
            [
                "--max-batches",
                "10",
            ]
        )

    return (
        run_command(
            name=(
                "Prefill Evaluation"
            ),

            command=(
                command
            ),

            output_path=(
                output_path
            ),
        )
    )


# ============================================================
# Decode
# ============================================================


def run_decode(
    *,
    python_executable: str,

    mapping: Path,

    output_path: Path,

    smoke: bool,

    decode_max_tokens: int | None,

    exact_check: int,

    cache_size: int,

    reuse_existing: bool,

    progress_every: int,
) -> StageRunResult:

    if (
        reuse_existing
        and
        output_path.exists()
    ):

        return (
            skip_stage(
                name=(
                    "Decode Fast Evaluation"
                ),

                output_path=(
                    output_path
                ),
            )
        )

    command = [
        python_executable,
        "-m",
        "scheduling.decode_fast_evaluator",

        "--mapping",
        str(
            mapping
        ),

        "--output",
        str(
            output_path
        ),

        "--exact-check",
        str(
            exact_check
        ),

        "--cache-size",
        str(
            cache_size
        ),

        "--progress-every",
        str(
            progress_every
        ),
    ]

    # ========================================================
    # max-tokens 优先级：
    #
    # 显式参数 > smoke > 全量
    # ========================================================

    if (
        decode_max_tokens
        is not None
    ):

        command.extend(
            [
                "--max-tokens",
                str(
                    decode_max_tokens
                ),
            ]
        )

    elif smoke:

        command.extend(
            [
                "--max-tokens",
                "1000",
            ]
        )

    return (
        run_command(
            name=(
                "Decode Fast Evaluation"
            ),

            command=(
                command
            ),

            output_path=(
                output_path
            ),
        )
    )


# ============================================================
# Summary
# ============================================================


def run_summary(
    *,
    python_executable: str,

    prefill_path: Path,

    decode_path: Path,

    output_path: Path,
) -> StageRunResult:

    if not (
        prefill_path.exists()
    ):

        raise PhaseEvaluationRunnerError(
            "生成 Summary 前缺少 Prefill 结果："
            f"{prefill_path}"
        )

    if not (
        decode_path.exists()
    ):

        raise PhaseEvaluationRunnerError(
            "生成 Summary 前缺少 Decode 结果："
            f"{decode_path}"
        )

    command = [
        python_executable,
        "-m",
        "scheduling.phase_evaluation_summary",

        "--prefill",
        str(
            prefill_path
        ),

        "--decode",
        str(
            decode_path
        ),

        "--output",
        str(
            output_path
        ),
    ]

    return (
        run_command(
            name=(
                "Phase Summary"
            ),

            command=(
                command
            ),

            output_path=(
                output_path
            ),
        )
    )


# ============================================================
# 打印最终状态
# ============================================================


def print_final_status(
    results: list[
        StageRunResult
    ],
) -> None:

    print(
        "\n"
        "========================================"
    )

    print(
        "MoE Phase Evaluation Pipeline Finished"
    )

    print(
        "========================================"
    )

    total_elapsed = sum(
        result.elapsed_seconds

        for result
        in results
    )

    for result in (
        results
    ):

        state = (
            "REUSED"
            if result.skipped
            else "DONE"
        )

        print(
            f"{result.name}："
            f"{state}"
        )

        if not (
            result.skipped
        ):

            print(
                "  elapsed="
                f"{format_seconds(result.elapsed_seconds)}"
            )

        if (
            result.output_path
            is not None
        ):

            print(
                "  output="
                f"{result.output_path}"
            )

    print(
        "\nExecuted Stages Total Elapsed："
        f"{format_seconds(total_elapsed)}"
    )


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "一键运行 MoE Prefill + Decode "
                "正式评估与统一汇总。"
            )
        )
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=(
            DEFAULT_MAPPING_PATH
        ),
    )

    parser.add_argument(
        "--prefill-output",
        type=Path,
        default=(
            DEFAULT_PREFILL_OUTPUT
        ),
    )

    parser.add_argument(
        "--decode-output",
        type=Path,
        default=(
            DEFAULT_DECODE_OUTPUT
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=(
            DEFAULT_SUMMARY_OUTPUT
        ),
    )

    # ========================================================
    # 模式
    # ========================================================

    parser.add_argument(
        "--smoke",
        action="store_true",

        help=(
            "快速检查："
            "Prefill 10 Batch + "
            "Decode 1000 Token。"
        ),
    )

    parser.add_argument(
        "--reuse-existing",
        action="store_true",

        help=(
            "若阶段结果 JSON 已存在，"
            "跳过该阶段并直接复用。"
        ),
    )

    parser.add_argument(
        "--prefill-only",
        action="store_true",
    )

    parser.add_argument(
        "--decode-only",
        action="store_true",
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
    )

    # ========================================================
    # 显式限制
    # ========================================================

    parser.add_argument(
        "--prefill-max-batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--decode-max-tokens",
        type=int,
        default=None,
    )

    # ========================================================
    # Decode Fast
    # ========================================================

    parser.add_argument(
        "--exact-check",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--cache-size",
        type=int,
        default=200000,
    )

    # ========================================================
    # Progress
    # ========================================================

    parser.add_argument(
        "--prefill-progress-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--decode-progress-every",
        type=int,
        default=10000,
    )

    args = (
        parser.parse_args()
    )

    # ========================================================
    # 参数检查
    # ========================================================

    exclusive_modes = sum(
        [
            bool(
                args.prefill_only
            ),
            bool(
                args.decode_only
            ),
            bool(
                args.summary_only
            ),
        ]
    )

    if (
        exclusive_modes
        > 1
    ):

        raise PhaseEvaluationRunnerError(
            "--prefill-only / "
            "--decode-only / "
            "--summary-only "
            "最多只能指定一个。"
        )

    if (
        args.prefill_max_batches
        is not None
        and
        args.prefill_max_batches
        <= 0
    ):

        raise PhaseEvaluationRunnerError(
            "--prefill-max-batches "
            "必须大于 0。"
        )

    if (
        args.decode_max_tokens
        is not None
        and
        args.decode_max_tokens
        <= 0
    ):

        raise PhaseEvaluationRunnerError(
            "--decode-max-tokens "
            "必须大于 0。"
        )

    if (
        args.exact_check
        < 0
    ):

        raise PhaseEvaluationRunnerError(
            "--exact-check "
            "不能小于 0。"
        )

    # ========================================================
    # Resolve
    # ========================================================

    mapping = (
        args.mapping
        .resolve()
    )

    prefill_output = (
        args.prefill_output
        .resolve()
    )

    decode_output = (
        args.decode_output
        .resolve()
    )

    summary_output = (
        args.summary_output
        .resolve()
    )

    if not (
        mapping.exists()
    ):

        raise PhaseEvaluationRunnerError(
            f"Mapping 不存在：{mapping}"
        )

    python_executable = (
        sys.executable
    )

    results: list[
        StageRunResult
    ] = []

    # ========================================================
    # summary-only
    # ========================================================

    if (
        args.summary_only
    ):

        results.append(
            run_summary(
                python_executable=(
                    python_executable
                ),

                prefill_path=(
                    prefill_output
                ),

                decode_path=(
                    decode_output
                ),

                output_path=(
                    summary_output
                ),
            )
        )

        print_final_status(
            results
        )

        return

    # ========================================================
    # prefill-only
    # ========================================================

    if (
        args.prefill_only
    ):

        results.append(
            run_prefill(
                python_executable=(
                    python_executable
                ),

                mapping=(
                    mapping
                ),

                output_path=(
                    prefill_output
                ),

                smoke=(
                    args.smoke
                ),

                prefill_max_batches=(
                    args
                    .prefill_max_batches
                ),

                reuse_existing=(
                    args
                    .reuse_existing
                ),

                progress_every=(
                    args
                    .prefill_progress_every
                ),
            )
        )

        print_final_status(
            results
        )

        return

    # ========================================================
    # decode-only
    # ========================================================

    if (
        args.decode_only
    ):

        results.append(
            run_decode(
                python_executable=(
                    python_executable
                ),

                mapping=(
                    mapping
                ),

                output_path=(
                    decode_output
                ),

                smoke=(
                    args.smoke
                ),

                decode_max_tokens=(
                    args
                    .decode_max_tokens
                ),

                exact_check=(
                    args.exact_check
                ),

                cache_size=(
                    args.cache_size
                ),

                reuse_existing=(
                    args
                    .reuse_existing
                ),

                progress_every=(
                    args
                    .decode_progress_every
                ),
            )
        )

        print_final_status(
            results
        )

        return

    # ========================================================
    # 默认：完整 Pipeline
    # ========================================================

    results.append(
        run_prefill(
            python_executable=(
                python_executable
            ),

            mapping=(
                mapping
            ),

            output_path=(
                prefill_output
            ),

            smoke=(
                args.smoke
            ),

            prefill_max_batches=(
                args
                .prefill_max_batches
            ),

            reuse_existing=(
                args
                .reuse_existing
            ),

            progress_every=(
                args
                .prefill_progress_every
            ),
        )
    )

    results.append(
        run_decode(
            python_executable=(
                python_executable
            ),

            mapping=(
                mapping
            ),

            output_path=(
                decode_output
            ),

            smoke=(
                args.smoke
            ),

            decode_max_tokens=(
                args
                .decode_max_tokens
            ),

            exact_check=(
                args.exact_check
            ),

            cache_size=(
                args.cache_size
            ),

            reuse_existing=(
                args
                .reuse_existing
            ),

            progress_every=(
                args
                .decode_progress_every
            ),
        )
    )

    results.append(
        run_summary(
            python_executable=(
                python_executable
            ),

            prefill_path=(
                prefill_output
            ),

            decode_path=(
                decode_output
            ),

            output_path=(
                summary_output
            ),
        )
    )

    print_final_status(
        results
    )


if __name__ == "__main__":
    main()
