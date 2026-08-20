"""
WebUI Experiment Worker。

由 experiment_api.py 通过独立 Python 进程启动。

Smoke：
    严格限制到前 10 个完整 JSON Request。
    Prefill：Exact Prefill Evaluator，max_files=10。
    Decode：Exact Continuous Request State Evaluator，max_files=10。

Full：
    Prefill：全部 2020 Request，Exact Prefill Evaluator。
    Decode：正式 Fast Exact-validated 全量路径。

每次实验输出到：

    results/webui_experiments/<job_id>/

不会覆盖当前正式 Baseline JSON。
"""

from __future__ import annotations

import argparse
import inspect
import json
import traceback

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mapping.trace_profile import DEFAULT_TRACE_ROOT
from scheduling.runtime_index import load_runtime_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results" / "webui_experiments"

SMOKE_REQUEST_COUNT = 10
FORMAL_FULL_REQUEST_COUNT = 2020


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> Path:
    return RESULT_ROOT / job_id


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_status(job_id: str) -> dict[str, Any]:
    path = _status_path(job_id)
    if not path.exists():
        return {"job_id": job_id}

    return json.loads(path.read_text(encoding="utf-8"))


def _update_status(
    job_id: str,
    *,
    state: str | None = None,
    stage: str | None = None,
    message: str | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    finished: bool = False,
) -> None:
    payload = _read_status(job_id)

    if state is not None:
        payload["state"] = state

    if stage is not None:
        payload["stage"] = stage

    if message is not None:
        payload["message"] = message

    if result is not None:
        payload["result"] = result

    if error is not None:
        payload["error"] = error

    if payload.get("started_at") is None and state == "running":
        payload["started_at"] = _utc_now()

    if finished:
        payload["finished_at"] = _utc_now()

    _write_json_atomic(_status_path(job_id), payload)


def _scalar_summary(value: Any) -> dict[str, float | int]:
    return {
        "count": int(value.count),
        "minimum": float(value.minimum),
        "mean": float(value.mean),
        "p50": float(value.p50),
        "p95": float(value.p95),
        "p99": float(value.p99),
        "maximum": float(value.maximum),
    }


def _prefill_result_summary(summary: Any) -> dict[str, Any]:
    return {
        "mode": "exact_prefill",
        "batch_count": int(summary.batch_count),
        "total_input_tokens": int(summary.total_input_tokens),
        "prompt_tokens": _scalar_summary(summary.prompt_tokens),
        "latency_cycles": _scalar_summary(summary.total_cycles),
        "cycles_per_input_token": _scalar_summary(
            summary.cycles_per_input_token
        ),
        "global_cycles_per_input_token": float(
            summary.global_cycles_per_input_token
        ),
        "prompt_length_latency_pearson": float(
            summary.prompt_length_latency_pearson
        ),
    }


def _decode_exact_result_summary(summary: Any) -> dict[str, Any]:
    return {
        "mode": "exact_continuous_request_state",
        "state_mode": str(summary.state_mode),
        "request_count": int(summary.request_count),
        "token_count": int(summary.token_count),
        "tasks_per_token": int(summary.total_tasks_per_token),
        "cycles_per_token": _scalar_summary(summary.cycles_per_token),
    }


def _decode_fast_result_summary(summary: Any, stats: Any) -> dict[str, Any]:
    return {
        "mode": "fast_exact_validated",
        "scheduler_mode": str(summary.scheduler_mode),
        "exact_checked_tokens": int(summary.exact_checked_tokens),
        "request_count": int(stats.processed_file_count),
        "token_count": int(summary.token_count),
        "cycles_per_token": _scalar_summary(summary.cycles_per_token),
        "semantic_note": (
            "Full Decode 使用当前正式 Fast Exact-validated 路径。"
            "当前硬件规则下已验证其 latency 与连续 Request State 一致；"
            "若以后修改 initial activation / switch 代价或跨层 WC 复用规则，"
            "应重新验证或切回 Exact Continuous 模式。"
        ),
    }


def run_prefill(
    *,
    job_id: str,
    index: Any,
    mapping_path: Path,
    scope: str,
) -> dict[str, Any]:
    from scheduling.prefill_evaluator import (
        evaluate_prefill_workload,
        save_prefill_evaluation,
    )

    request_limit = (
        SMOKE_REQUEST_COUNT
        if scope == "smoke"
        else None
    )

    _update_status(
        job_id,
        state="running",
        stage="Prefill 正式评估",
        message=(
            "正在运行前 10 个完整 Request 的 Exact Prefill。"
            if scope == "smoke"
            else "正在运行全部 Request 的 Exact Prefill。"
        ),
    )

    print("\n[WebUI Experiment] Prefill start", flush=True)
    print(f"scope={scope}, max_files={request_limit}", flush=True)

    summary, records = evaluate_prefill_workload(
        index=index,
        trace_root=DEFAULT_TRACE_ROOT,
        max_files=request_limit,
        max_batches=None,
        charge_initial_activation=True,
        progress_every=(1 if scope == "smoke" else 50),
        verbose=True,
    )

    output_path = _job_dir(job_id) / "prefill_evaluation.json"

    save_prefill_evaluation(
        output_path=output_path,
        summary=summary,
        records=records,
        mapping_path=mapping_path,
        trace_root=DEFAULT_TRACE_ROOT,
        charge_initial_activation=True,
    )

    print(
        "[WebUI Experiment] Prefill done: "
        f"batches={summary.batch_count}, "
        f"mean={summary.total_cycles.mean:.2f}",
        flush=True,
    )

    return _prefill_result_summary(summary)


def run_decode_smoke_exact(
    *,
    job_id: str,
    index: Any,
    mapping_path: Path,
) -> dict[str, Any]:
    # 注意：必须使用连续请求状态版的 exact evaluator。
    from scheduling import decode_evaluator

    evaluate_fn = decode_evaluator.evaluate_decode_workload

    if "continuous_request_state" not in inspect.signature(
        evaluate_fn
    ).parameters:
        raise RuntimeError(
            "当前 scheduling/decode_evaluator.py 仍是旧冷启动版本，"
            "缺少 continuous_request_state 参数。"
            "请先使用已经完成的连续请求状态版 Decode Evaluator，"
            "避免 WebUI 产生错误实验口径。"
        )

    _update_status(
        job_id,
        state="running",
        stage="Decode Exact 连续请求评估",
        message=(
            "正在运行与 Prefill 相同的前 10 个完整 Request。"
            "每个 Request 会先执行自己的 segment0 Prefill 生成 SC active state，"
            "再按 segment1+ 连续执行 Decode。"
        ),
    )

    print("\n[WebUI Experiment] Decode smoke exact start", flush=True)
    print("scope=smoke, max_files=10, continuous_request_state=True", flush=True)

    summary, records, stats = evaluate_fn(
        index=index,
        trace_root=DEFAULT_TRACE_ROOT,
        max_files=SMOKE_REQUEST_COUNT,
        max_tokens=None,
        charge_initial_activation=True,
        progress_every=100,
        top_slowest_tokens=10,
        continuous_request_state=True,
        verbose=True,
    )

    output_path = _job_dir(job_id) / "decode_evaluation.json"

    decode_evaluator.save_result(
        output_path,
        summary=summary,
        records=records,
        stats=stats,
        mapping=mapping_path,
        trace_root=DEFAULT_TRACE_ROOT,
        charge_initial_activation=True,
    )

    print(
        "[WebUI Experiment] Decode smoke exact done: "
        f"requests={summary.request_count}, "
        f"tokens={summary.token_count}, "
        f"mean={summary.cycles_per_token.mean:.2f}",
        flush=True,
    )

    return _decode_exact_result_summary(summary)


def run_decode_full_fast(
    *,
    job_id: str,
    index: Any,
    mapping_path: Path,
) -> dict[str, Any]:
    from scheduling.decode_fast_evaluator import (
        evaluate_decode_fast,
        save_result,
    )

    _update_status(
        job_id,
        state="running",
        stage="Decode Full Fast 正式评估",
        message=(
            "正在运行全部纯 Decode Token。"
            "使用正式 Fast Scheduler，并对前 100 Token 做 FAST == EXACT 校验。"
        ),
    )

    print("\n[WebUI Experiment] Decode full fast start", flush=True)

    summary, records, stats = evaluate_decode_fast(
        index=index,
        trace_root=DEFAULT_TRACE_ROOT,
        max_files=None,
        max_tokens=None,
        exact_check=100,
        cache_size=200000,
        progress_every=10000,
        top_slowest_tokens=10,
        verbose=True,
    )

    output_path = _job_dir(job_id) / "decode_fast_evaluation.json"

    save_result(
        output_path=output_path,
        summary=summary,
        records=records,
        workload_stats=stats,
        mapping=mapping_path,
        trace_root=DEFAULT_TRACE_ROOT,
    )

    print(
        "[WebUI Experiment] Decode full fast done: "
        f"requests={stats.processed_file_count}, "
        f"tokens={summary.token_count}, "
        f"mean={summary.cycles_per_token.mean:.2f}",
        flush=True,
    )

    return _decode_fast_result_summary(summary, stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WebUI Prefill / Decode 实验 Worker。"
    )

    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--scope",
        choices=("smoke", "full"),
        required=True,
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        required=True,
    )

    # 算法 ID 由 experiment_api 的注册表校验后传入。
    # 当前 worker 仍只执行已经实现的 Baseline 分支；
    # 这些参数会写入结果，后续增加真实算法时在此处分派。
    parser.add_argument("--partition-algorithm", required=True)
    parser.add_argument("--placement-algorithm", required=True)
    parser.add_argument("--plane-pairing-algorithm", required=True)
    parser.add_argument("--mapping-algorithm", required=True)
    parser.add_argument("--prefill-scheduler", required=True)
    parser.add_argument("--decode-scheduler", required=True)

    parser.add_argument("--run-prefill", action="store_true")
    parser.add_argument("--run-decode", action="store_true")

    args = parser.parse_args()

    if not args.run_prefill and not args.run_decode:
        raise RuntimeError("至少运行 Prefill 或 Decode 之一。")

    job_id = args.job_id
    mapping_path = args.mapping.resolve()

    if not mapping_path.exists():
        raise RuntimeError(f"Mapping 不存在：{mapping_path}")

    algorithms = {
        "partition": args.partition_algorithm,
        "placement": args.placement_algorithm,
        "plane_pairing": args.plane_pairing_algorithm,
        "mapping": args.mapping_algorithm,
        "prefill_scheduler": args.prefill_scheduler,
        "decode_scheduler": args.decode_scheduler,
    }

    result: dict[str, Any] = {
        "scope": args.scope,
        "request_limit": (
            SMOKE_REQUEST_COUNT
            if args.scope == "smoke"
            else FORMAL_FULL_REQUEST_COUNT
        ),
        "algorithms": algorithms,
        "mapping_file": mapping_path.name,
        "metric_scope": (
            "MoE Expert phase only; Prefill is not full TTFT; "
            "Decode is not full TPOT"
        ),
    }

    try:
        _update_status(
            job_id,
            state="running",
            stage="初始化",
            message="正在加载当前 Mapping 与 Runtime Index。",
        )

        print("========== WebUI MoE Experiment =========", flush=True)
        print(f"job_id={job_id}", flush=True)
        print(f"scope={args.scope}", flush=True)
        print(f"mapping={mapping_path}", flush=True)
        print("algorithms=", json.dumps(algorithms, ensure_ascii=False), flush=True)

        # 当前所有 enabled 选项都复用现有 Baseline Mapping / Scheduler。
        # 未来某个算法真正实现后，可在这里根据 algorithms 分派：
        #   重新生成空间布局 / Mapping -> load_runtime_index -> evaluator。
        index = load_runtime_index(mapping_path)

        if args.run_prefill:
            result["prefill"] = run_prefill(
                job_id=job_id,
                index=index,
                mapping_path=mapping_path,
                scope=args.scope,
            )

        if args.run_decode:
            if args.scope == "smoke":
                result["decode"] = run_decode_smoke_exact(
                    job_id=job_id,
                    index=index,
                    mapping_path=mapping_path,
                )
            else:
                result["decode"] = run_decode_full_fast(
                    job_id=job_id,
                    index=index,
                    mapping_path=mapping_path,
                )

        result_path = _job_dir(job_id) / "result_summary.json"
        _write_json_atomic(result_path, result)

        _update_status(
            job_id,
            state="completed",
            stage="完成",
            message="Prefill / Decode 实验已完成，结果已保存到独立实验目录。",
            result=result,
            finished=True,
        )

        print("\n[WebUI Experiment] completed", flush=True)

    except Exception as exc:
        print("\n[WebUI Experiment] FAILED", flush=True)
        traceback.print_exc()

        _update_status(
            job_id,
            state="failed",
            stage="运行失败",
            message=str(exc),
            error=str(exc),
            finished=True,
        )
        raise


if __name__ == "__main__":
    main()
