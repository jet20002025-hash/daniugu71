"""K 线全量/批量更新与其它任务协调：锁存在且 PID 仍存活时，扫描队列 worker 不取任务。"""
import os

from app.paths import PROJECT_DIR

_LOCK_PATH = os.path.join(PROJECT_DIR, "data", ".kline_heavy_update")


def lock_file_path() -> str:
    return _LOCK_PATH


def is_heavy_kline_running() -> bool:
    if not os.path.isfile(_LOCK_PATH):
        return False
    try:
        with open(_LOCK_PATH, "r", encoding="utf-8") as f:
            s = (f.read() or "").strip()
        pid = int(s) if s else 0
    except (OSError, ValueError):
        return True
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        try:
            os.remove(_LOCK_PATH)
        except OSError:
            pass
        return False


def acquire_heavy_kline() -> None:
    os.makedirs(os.path.dirname(_LOCK_PATH), exist_ok=True)
    with open(_LOCK_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def release_heavy_kline() -> None:
    try:
        os.remove(_LOCK_PATH)
    except OSError:
        pass
