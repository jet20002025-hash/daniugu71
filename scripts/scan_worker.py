#!/usr/bin/env python3
"""
扫描队列 worker：从 data/results/scan_queue 取任务并执行，结果写入对应用户目录。
可多进程运行（如 4 个进程）以实现约 4 个任务并发，满足约 100 人排队、各自结果不冲突。
用法：python scripts/scan_worker.py [--workers 4]
"""
import argparse
import importlib.util
import json
import os
import sys
import time

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.scan_queue import list_pending_jobs, SCAN_QUEUE_DIR
from app.scanner import ScanConfig

# run_mode3_scan 在项目根目录的 app.py 中，而 import app 会加载 app 包，故按文件加载
_app_py = os.path.join(ROOT, "app.py")
_spec = importlib.util.spec_from_file_location("app_main", _app_py)
_app_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_app_main)
run_mode3_scan = _app_main.run_mode3_scan


def _claim_job():
    """原子抢占一个待处理任务，返回 (processing_path, payload) 或 (None, None)。"""
    for path in list_pending_jobs():
        processing = path + ".processing"
        try:
            os.rename(path, processing)
            with open(processing, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return processing, payload
        except (OSError, json.JSONDecodeError):
            try:
                if os.path.exists(processing):
                    os.rename(processing, path)
            except OSError:
                pass
            continue
    return None, None


def run_one():
    processing, payload = _claim_job()
    if not processing:
        return False
    user_id = payload.get("user_id")
    cfg = payload.get("config") or {}
    if user_id is None:
        try:
            os.remove(processing)
        except OSError:
            pass
        return True
    try:
        cap = cfg.get("max_market_cap")
        config = ScanConfig(
            min_score=int(cfg.get("min_score", 70)),
            max_results=int(cfg.get("max_results", 200)),
            workers=int(cfg.get("workers", 6)),
            max_market_cap=cap,
        )
        mode = payload.get("mode", "mode3")
        run_mode3_scan(
            config,
            cutoff_date=payload.get("cutoff_date"),
            start_date=payload.get("start_date"),
            data_source=payload.get("data_source", "gpt"),
            remote_provider=payload.get("remote_provider", "eastmoney"),
            prefer_local=bool(payload.get("prefer_local")),
            avoid_big_candle=(mode == "mode3_avoid"),
            prefer_upper_shadow=(mode in ("mode3_upper", "mode3_upper_near")),
            require_upper_shadow=(mode == "mode3_upper_strict"),
            require_vol_ratio=(mode == "mode3_upper_near"),
            require_close_gap=(mode == "mode3_upper_near"),
            mode4_filters=(mode == "mode4"),
            model_tag_override="mode3ok" if mode == "mode3ok" else None,
            use_startup_modes_data=True,
            use_71x_standard=(mode in ("mode3", "mode9")),
            use_mode8=(mode == "mode8"),
            use_mode9=(mode == "mode9"),
            user_id=user_id,
            throttle_free_user=payload.get("throttle_free_user", True),
        )
    finally:
        try:
            os.remove(processing)
        except OSError:
            pass
    return True


def main():
    parser = argparse.ArgumentParser(description="Scan queue worker")
    parser.add_argument("--workers", type=int, default=4, help="并发 worker 数（本进程内线程）")
    parser.add_argument("--interval", type=float, default=1.0, help="无任务时轮询间隔秒")
    args = parser.parse_args()
    if not os.path.isdir(SCAN_QUEUE_DIR):
        os.makedirs(SCAN_QUEUE_DIR, exist_ok=True)
    n = max(1, args.workers)
    import threading
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            if run_one():
                continue
            stop.wait(timeout=args.interval)

    ths = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
    for t in ths:
        t.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        stop.set()
    for t in ths:
        t.join(timeout=2)


if __name__ == "__main__":
    main()
