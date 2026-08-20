"""
WebUI 正式实验运行 API。

目标：

1. WebUI 提交“算法配置 + Workload 配置”；
2. 真正的 Prefill / Decode 计算继续复用 scheduling/ 中的正式 evaluator；
3. 每次实验写入独立目录，不覆盖当前 Baseline 结果；
4. Smoke 模式严格按“完整 Request”限制到同一批 JSON 文件；
5. 同一时间只允许一个 WebUI 实验任务；
6. 算法通过统一注册表暴露给前端，后续新增算法无需重做页面结构。

重要：
当前项目源码中真正可运行的仍是当前 Baseline 算法链。
规划中的 Naive / Pairing-only / Mapping-only 等选项会返回给前端，
但 implemented=False，前端显示为“待接入”且不可运行。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_ROOT = PROJECT_ROOT / "results" / "webui_experiments"

DEFAULT_MAPPING_PATH = (
    PROJECT_ROOT
    / "results"
    / "mappings"
    / "mapping_baseline_N4_H7168_W4096.json"
)

router = APIRouter(
    prefix="/api/experiments",
    tags=["experiments"],
)


# ============================================================
# 算法注册表
# ============================================================
#
# 前端只依赖这个注册表，不把算法名称写死在页面里。
# 新算法真正实现以后，只需要：
#   1. 在这里增加/启用 option；
#   2. 在 experiment_worker / 对应 pipeline 中实现分支；
# 即可直接出现在 WebUI 下拉框中。
# ============================================================

ALGORITHM_REGISTRY: dict[str, dict[str, Any]] = {
    "partition": {
        "label": "矩阵切分 / Matrix Partition",
        "stage": "空间规划",
        "options": [
            {
                "id": "anonymous_template_baseline",
                "label": "匿名切分模板 / Baseline",
                "description": "使用当前第二步生成的匿名几何切分模板。",
                "implemented": True,
            },
        ],
    },
    "placement": {
        "label": "空间放置 / Placement",
        "stage": "空间规划",
        "options": [
            {
                "id": "maxrects_bssf",
                "label": "MaxRects-BSSF",
                "description": "面积降序 + MaxRects-BSSF + 允许匿名块旋转。",
                "implemented": True,
            },
        ],
    },
    "plane_pairing": {
        "label": "Plane 配对 / Plane Pairing",
        "stage": "逻辑映射",
        "options": [
            {
                "id": "trace_greedy_local_search",
                "label": "Trace-aware Greedy + Local Search",
                "description": "当前正式方法：Greedy Initialization + Local Search。",
                "implemented": True,
            },
            {
                "id": "naive_pairing",
                "label": "Naive Pairing / 不看 Trace",
                "description": "规划中的消融选项：配对阶段不使用 Trace 信息。",
                "implemented": False,
            },
        ],
    },
    "mapping": {
        "label": "Sub-Cube 映射 / Mapping",
        "stage": "逻辑映射",
        "options": [
            {
                "id": "trace_aware_mapping",
                "label": "Trace-aware Mapping / 当前正式方法",
                "description": "使用访问频率、共激活与负载信息进行 Sub-Cube 映射。",
                "implemented": True,
            },
            {
                "id": "naive_mapping",
                "label": "Naive Mapping / 不看 Trace",
                "description": "规划中的消融选项：映射阶段不使用 Trace 信息。",
                "implemented": False,
            },
        ],
    },
    "prefill_scheduler": {
        "label": "Prefill 调度器 / Scheduler",
        "stage": "推理调度",
        "options": [
            {
                "id": "exact_batch_scheduler",
                "label": "Exact Batch Scheduler",
                "description": "多 Token Batch 逐层精确调度。",
                "implemented": True,
            },
        ],
    },
    "decode_scheduler": {
        "label": "Decode 调度器 / Scheduler",
        "stage": "推理调度",
        "options": [
            {
                "id": "formal_auto",
                "label": "Formal Auto / 正式模式",
                "description": "Smoke=Exact Continuous；Full=Fast Exact-validated。",
                "implemented": True,
            },
        ],
    },
}

DEFAULT_ALGORITHMS: dict[str, str] = {
    "partition": "anonymous_template_baseline",
    "placement": "maxrects_bssf",
    "plane_pairing": "trace_greedy_local_search",
    "mapping": "trace_aware_mapping",
    "prefill_scheduler": "exact_batch_scheduler",
    "decode_scheduler": "formal_auto",
}


# ============================================================
# 内存中的子进程表
# ============================================================

_PROCESS_LOCK = threading.Lock()
_PROCESSES: dict[str, subprocess.Popen] = {}


# ============================================================
# 请求模型
# ============================================================


class AlgorithmSelection(BaseModel):
    partition: str = DEFAULT_ALGORITHMS["partition"]
    placement: str = DEFAULT_ALGORITHMS["placement"]
    plane_pairing: str = DEFAULT_ALGORITHMS["plane_pairing"]
    mapping: str = DEFAULT_ALGORITHMS["mapping"]
    prefill_scheduler: str = DEFAULT_ALGORITHMS["prefill_scheduler"]
    decode_scheduler: str = DEFAULT_ALGORITHMS["decode_scheduler"]


class ExperimentRunRequest(BaseModel):
    scope: Literal["smoke", "full"] = "smoke"
    run_prefill: bool = True
    run_decode: bool = True
    algorithms: AlgorithmSelection = Field(default_factory=AlgorithmSelection)


# ============================================================
# Helper
# ============================================================


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_job_id(job_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise HTTPException(
            status_code=400,
            detail="非法 experiment job_id。",
        )
    return job_id


def _job_dir(job_id: str) -> Path:
    _validate_job_id(job_id)
    return RESULT_ROOT / job_id


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _log_path(job_id: str) -> Path:
    return _job_dir(job_id) / "experiment.log"


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
        raise HTTPException(
            status_code=404,
            detail=f"实验任务不存在：{job_id}",
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"无法读取实验状态：{exc}",
        ) from exc

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=500,
            detail="实验 status.json 格式错误。",
        )

    return data


def _tail_log(job_id: str, max_lines: int = 14) -> list[str]:
    path = _log_path(job_id)

    if not path.exists():
        return []

    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return []

    useful = [line for line in lines if line.strip()]
    return useful[-max_lines:]


def _cleanup_finished_processes() -> None:
    with _PROCESS_LOCK:
        finished = [
            job_id
            for job_id, process in _PROCESSES.items()
            if process.poll() is not None
        ]

        for job_id in finished:
            _PROCESSES.pop(job_id, None)


def _active_job_id() -> str | None:
    _cleanup_finished_processes()

    with _PROCESS_LOCK:
        for job_id, process in _PROCESSES.items():
            if process.poll() is None:
                return job_id

    return None


def _status_with_runtime(job_id: str) -> dict[str, Any]:
    status = _read_status(job_id)
    status["log_tail"] = _tail_log(job_id)

    with _PROCESS_LOCK:
        process = _PROCESSES.get(job_id)

    if (
        process is not None
        and process.poll() is not None
        and status.get("state") in {"queued", "running"}
    ):
        status.update(
            {
                "state": "failed",
                "stage": "异常结束",
                "message": (
                    "实验子进程已结束，但没有生成完整完成状态。"
                    f" returncode={process.returncode}"
                ),
                "finished_at": _utc_now(),
            }
        )
        _write_json_atomic(_status_path(job_id), status)

    return status


def _algorithms_to_dict(selection: AlgorithmSelection) -> dict[str, str]:
    if hasattr(selection, "model_dump"):
        return selection.model_dump()
    return selection.dict()


def _validate_algorithm_selection(selection: dict[str, str]) -> None:
    """只允许注册表中存在且已经 implemented 的算法真正运行。"""

    for group_id, selected_id in selection.items():
        group = ALGORITHM_REGISTRY.get(group_id)
        if group is None:
            raise HTTPException(
                status_code=400,
                detail=f"未知算法组：{group_id}",
            )

        option = next(
            (
                item
                for item in group["options"]
                if item["id"] == selected_id
            ),
            None,
        )

        if option is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"算法组 {group_id} 中不存在选项：{selected_id}"
                ),
            )

        if not option.get("implemented", False):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"算法尚未接入实验执行链：{option['label']}。"
                    "请先实现对应算法后再启用。"
                ),
            )


# ============================================================
# API
# ============================================================


@router.get("/algorithms")
def experiment_algorithms() -> dict[str, Any]:
    """返回实验页面使用的算法注册表。"""

    return {
        "defaults": DEFAULT_ALGORITHMS,
        "groups": ALGORITHM_REGISTRY,
        "note": (
            "implemented=true 的选项可以运行；"
            "implemented=false 的选项只用于显示后续计划。"
        ),
    }


@router.get("/status")
def experiment_service_status() -> dict[str, Any]:
    active = _active_job_id()

    return {
        "busy": active is not None,
        "active_job_id": active,
        "result_root": "results/webui_experiments",
        "smoke_request_count": 10,
        "formal_full_request_count": 2020,
        "algorithms": {
            "defaults": DEFAULT_ALGORITHMS,
            "groups": ALGORITHM_REGISTRY,
        },
    }


@router.post("/run")
def run_experiment(
    request: ExperimentRunRequest,
) -> dict[str, Any]:
    """启动一次“算法配置 + Workload 配置”的实验。"""

    if not request.run_prefill and not request.run_decode:
        raise HTTPException(
            status_code=400,
            detail="Prefill / Decode 至少选择一个阶段。",
        )

    selected_algorithms = _algorithms_to_dict(request.algorithms)
    _validate_algorithm_selection(selected_algorithms)

    if not DEFAULT_MAPPING_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "找不到当前 Mapping："
                f"{DEFAULT_MAPPING_PATH}"
            ),
        )

    active = _active_job_id()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "已有实验正在运行："
                f"{active}。请等待其结束或先取消。"
            ),
        )

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"exp_{stamp}_{uuid.uuid4().hex[:8]}"
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "job_id": job_id,
        "scope": request.scope,
        "run_prefill": request.run_prefill,
        "run_decode": request.run_decode,
        "algorithms": selected_algorithms,
        "mapping_file": DEFAULT_MAPPING_PATH.name,
        "created_at": _utc_now(),
    }

    _write_json_atomic(job_dir / "config.json", config)

    initial_status = {
        **config,
        "state": "queued",
        "stage": "等待启动",
        "message": "实验任务已创建。",
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    }
    _write_json_atomic(_status_path(job_id), initial_status)

    command = [
        sys.executable,
        "-u",
        "-m",
        "webui.backend.experiment_worker",
        "--job-id",
        job_id,
        "--scope",
        request.scope,
        "--mapping",
        str(DEFAULT_MAPPING_PATH),
        "--partition-algorithm",
        selected_algorithms["partition"],
        "--placement-algorithm",
        selected_algorithms["placement"],
        "--plane-pairing-algorithm",
        selected_algorithms["plane_pairing"],
        "--mapping-algorithm",
        selected_algorithms["mapping"],
        "--prefill-scheduler",
        selected_algorithms["prefill_scheduler"],
        "--decode-scheduler",
        selected_algorithms["decode_scheduler"],
    ]

    if request.run_prefill:
        command.append("--run-prefill")

    if request.run_decode:
        command.append("--run-decode")

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    creationflags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW

    log_file = _log_path(job_id).open(
        "a",
        encoding="utf-8",
        buffering=1,
    )

    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
    except Exception as exc:
        log_file.close()

        failed = {
            **initial_status,
            "state": "failed",
            "stage": "启动失败",
            "message": str(exc),
            "finished_at": _utc_now(),
            "error": str(exc),
        }
        _write_json_atomic(_status_path(job_id), failed)

        raise HTTPException(
            status_code=500,
            detail=f"无法启动实验进程：{exc}",
        ) from exc

    log_file.close()

    with _PROCESS_LOCK:
        _PROCESSES[job_id] = process

    return _status_with_runtime(job_id)


@router.get("/{job_id}")
def get_experiment(job_id: str) -> dict[str, Any]:
    return _status_with_runtime(job_id)


@router.post("/{job_id}/cancel")
def cancel_experiment(job_id: str) -> dict[str, Any]:
    _validate_job_id(job_id)

    with _PROCESS_LOCK:
        process = _PROCESSES.get(job_id)

    if process is None or process.poll() is not None:
        status = _read_status(job_id)

        if status.get("state") in {"completed", "failed", "cancelled"}:
            return _status_with_runtime(job_id)

        raise HTTPException(
            status_code=409,
            detail="该实验当前没有可取消的活动进程。",
        )

    process.terminate()

    status = _read_status(job_id)
    status.update(
        {
            "state": "cancelled",
            "stage": "已取消",
            "message": "实验已由 WebUI 取消。",
            "finished_at": _utc_now(),
        }
    )
    _write_json_atomic(_status_path(job_id), status)

    return _status_with_runtime(job_id)
