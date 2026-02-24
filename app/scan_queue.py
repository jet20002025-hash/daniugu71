# 扫描任务队列：支持多用户同时提交，各自结果不冲突
import json
import os
import time
from typing import Any, Dict, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
SCAN_QUEUE_DIR = os.path.join(RESULTS_DIR, "scan_queue")
USER_RESULTS_DIR = os.path.join(RESULTS_DIR, "by_user")
USER_STATUS_DIR = os.path.join(RESULTS_DIR, "user_status")


def _ensure_dirs() -> None:
    os.makedirs(SCAN_QUEUE_DIR, exist_ok=True)
    os.makedirs(USER_RESULTS_DIR, exist_ok=True)
    os.makedirs(USER_STATUS_DIR, exist_ok=True)


def push_job(user_id: int, config: Dict[str, Any]) -> str:
    """将扫描任务加入队列，返回 job_id。"""
    _ensure_dirs()
    job_id = f"{user_id}_{int(time.time() * 1000)}"
    path = os.path.join(SCAN_QUEUE_DIR, f"{job_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"user_id": user_id, "config": config}, f, ensure_ascii=False)
    return job_id


def get_user_status(user_id: int) -> Dict[str, Any]:
    """读取该用户当前扫描状态（用于 /status）。"""
    _ensure_dirs()
    path = os.path.join(USER_STATUS_DIR, f"{user_id}.json")
    if not os.path.exists(path):
        return {"running": False, "progress": 0, "total": 0, "message": "空闲", "error": None, "source": "", "last_run": None}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("source") == "gpt":
            data["source"] = "本地"
        return data
    except Exception:
        return {"running": False, "progress": 0, "total": 0, "message": "空闲", "error": None, "source": "", "last_run": None}


def save_user_status(user_id: int, state: Dict[str, Any]) -> None:
    """写入该用户扫描状态（worker 调用）。"""
    _ensure_dirs()
    path = os.path.join(USER_STATUS_DIR, f"{user_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def user_result_dir(user_id: int) -> str:
    """该用户结果目录。"""
    d = os.path.join(USER_RESULTS_DIR, str(user_id))
    os.makedirs(d, exist_ok=True)
    return d


def list_pending_jobs() -> list:
    """列出未处理的 job 文件（仅 .json，不含 .processing）。"""
    _ensure_dirs()
    out = []
    for name in os.listdir(SCAN_QUEUE_DIR):
        if name.endswith(".json") and not name.endswith(".processing"):
            out.append(os.path.join(SCAN_QUEUE_DIR, name))
    return sorted(out)


def has_pending_job(user_id: int) -> bool:
    """该用户是否还有未处理的队列任务。"""
    if not os.path.isdir(SCAN_QUEUE_DIR):
        return False
    prefix = f"{user_id}_"
    for name in os.listdir(SCAN_QUEUE_DIR):
        if name.startswith(prefix) and name.endswith(".json") and not name.endswith(".processing"):
            return True
    return False
