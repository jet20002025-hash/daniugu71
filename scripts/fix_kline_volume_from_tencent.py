#!/usr/bin/env python3
"""用腾讯日 K 接口校正本地缓存近 N 个交易日的成交量（手）。

仅当缓存与接口偏差明显（含 ×100/÷100）时才写入，避免无谓改动。
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import requests

from app.paths import GPT_DATA_DIR
from app.tencent import fetch_kline, read_cached_kline, write_cached_kline


def _volume_matches_cache(cv: float, av: float, tol: float = 0.02) -> bool:
    if av <= 0:
        return True
    if abs(cv - av) <= max(1.0, av * tol):
        return True
    for scaled in (cv / 100.0, cv * 100.0):
        if abs(scaled - av) <= max(1.0, av * tol):
            return False
    return abs(cv - av) > max(1.0, av * tol)


def fix_one(
    path: str,
    fetch_count: int,
    session: Optional[requests.Session] = None,
) -> dict | None:
    code = os.path.splitext(os.path.basename(path))[0]
    cached = read_cached_kline(path)
    if not cached:
        return None

    api_rows = fetch_kline(code, count=fetch_count, session=session)
    if not api_rows:
        return None
    api_vol = {r.date[:10]: float(r.volume) for r in api_rows if float(r.volume) > 0}
    if not api_vol:
        return None

    changed = 0
    for row in cached:
        d = row.date[:10]
        av = api_vol.get(d)
        if av is None:
            continue
        cv = float(row.volume or 0)
        if _volume_matches_cache(cv, av):
            continue
        row.volume = av
        changed += 1

    if changed == 0:
        return None
    write_cached_kline(path, cached)
    return {"code": code, "rows": changed}


def main() -> int:
    ap = argparse.ArgumentParser(description="用腾讯接口校正缓存成交量")
    ap.add_argument("--cache-dir", default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"))
    ap.add_argument("--days", type=int, default=12, help="拉取最近 N 根 K 线对照")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--delay", type=float, default=0.03, help="每请求间隔秒")
    ap.add_argument("--codes", default="", help="仅处理指定代码，逗号分隔")
    args = ap.parse_args()

    if args.codes.strip():
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
        files = [os.path.join(args.cache_dir, f"{c}.csv") for c in codes]
    else:
        files = sorted(
            os.path.join(args.cache_dir, fn)
            for fn in os.listdir(args.cache_dir)
            if fn.endswith(".csv")
        )

    fetch_count = max(args.days + 3, 15)
    lock = threading.Lock()
    last_req = [0.0]
    changed: List[dict] = []
    failed = 0

    def worker(path: str) -> dict | None:
        with lock:
            now = time.time()
            wait = args.delay - (now - last_req[0])
            if wait > 0:
                time.sleep(wait)
            last_req[0] = time.time()
        sess = requests.Session()
        sess.trust_env = False
        try:
            return fix_one(path, fetch_count, session=sess)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(worker, p): p for p in files}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                st = fut.result()
            except Exception:
                failed += 1
                continue
            if st:
                changed.append(st)
            if done % 500 == 0:
                print(f"进度 {done}/{len(files)}  已修正 {len(changed)} 只")

    print(f"完成：扫描 {len(files)} 只，修正 {len(changed)} 只，失败 {failed} 只")
    for st in sorted(changed, key=lambda x: x["code"])[:40]:
        print(f"  {st['code']}  {st['rows']} 行")
    if len(changed) > 40:
        print(f"  ... 另有 {len(changed) - 40} 只")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
