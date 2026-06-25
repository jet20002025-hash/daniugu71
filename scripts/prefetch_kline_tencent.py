#!/usr/bin/env python3
"""回补本地日 K 历史深度（默认与腾讯前复权缓存合并，多源轮流拉取）。

全量回补建议单线程 + 适当 pause，避免腾讯接口 501 限流：
  python3 scripts/prefetch_kline_tencent.py --count 1500 --min-rows 1200 --workers 1 --pause 0.25
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import netease, sina, tencent
from app.eastmoney import (
    KlineRow,
    fetch_kline as eastmoney_fetch_kline,
    kline_is_fresh,
    read_cached_kline,
    stock_items_from_list_csv,
    write_cached_kline,
)


def _merge_rows(cached: List[KlineRow], new_rows: List[KlineRow]) -> List[KlineRow]:
    if not cached:
        return list(new_rows)
    if not new_rows:
        return list(cached)
    by_date: dict[str, KlineRow] = {r.date[:10]: r for r in cached}
    for r in new_rows:
        by_date[r.date[:10]] = r
    return [by_date[d] for d in sorted(by_date)]


def _from_sina(rows: List[sina.KlineRow]) -> List[KlineRow]:
    return [
        KlineRow(
            date=r.date,
            open=r.open,
            close=r.close,
            high=r.high,
            low=r.low,
            volume=r.volume,
            amount=r.amount,
            amplitude=r.amplitude,
            pct_chg=r.pct_chg,
            chg=r.chg,
            turnover=r.turnover,
        )
        for r in rows
    ]


def _from_tencent(rows: List[tencent.KlineRow]) -> List[KlineRow]:
    return [
        KlineRow(
            date=r.date,
            open=r.open,
            close=r.close,
            high=r.high,
            low=r.low,
            volume=r.volume,
            amount=r.amount,
            amplitude=r.amplitude,
            pct_chg=r.pct_chg,
            chg=r.chg,
            turnover=r.turnover,
        )
        for r in rows
    ]


def _fetch_one(
    item,
    count: int,
    source: str,
    session: requests.Session,
) -> Tuple[Optional[List[KlineRow]], Optional[str]]:
    if source == "sina":
        rows = sina.fetch_kline(item.code, count=count, session=session)
        return (_from_sina(rows) if rows else None, "新浪" if rows else None)
    if source == "tencent":
        rows = tencent.fetch_kline(item.code, count=count, session=session)
        return (_from_tencent(rows) if rows else None, "腾讯" if rows else None)
    if source == "netease":
        rows = netease.fetch_kline(item.code, count=count, session=session)
        if not rows:
            return None, None
        return (
            [
                KlineRow(
                    date=r.date,
                    open=r.open,
                    close=r.close,
                    high=r.high,
                    low=r.low,
                    volume=r.volume,
                    amount=r.amount,
                    amplitude=r.amplitude,
                    pct_chg=r.pct_chg,
                    chg=r.chg,
                    turnover=r.turnover,
                )
                for r in rows
            ],
            "网易",
        )
    if source == "eastmoney":
        rows = eastmoney_fetch_kline(item.secid, count=count, session=session)
        return (rows if rows else None, "东财" if rows else None)

    merged: List[KlineRow] = []
    used: List[str] = []
    for name, fetcher in [
        ("东财", lambda: eastmoney_fetch_kline(item.secid, count=count, session=session)),
        ("新浪", lambda: _from_sina(sina.fetch_kline(item.code, count=count, session=session))),
        ("网易", lambda: netease.fetch_kline(item.code, count=count, session=session)),
        ("腾讯", lambda: _from_tencent(tencent.fetch_kline(item.code, count=count, session=session))),
    ]:
        try:
            rows = fetcher()
            if rows:
                merged = _merge_rows(merged, rows)
                used.append(name)
        except Exception:
            pass
    if merged:
        return merged, "+".join(used)
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="回补 K 线历史深度（多源，与缓存合并）")
    parser.add_argument(
        "--stock-list",
        default=os.path.join(
            os.environ.get("GPT_DATA_DIR", os.path.join(ROOT, "data", "gpt")),
            "stock_list.csv",
        ),
        help="股票列表 CSV",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(
            os.environ.get("GPT_DATA_DIR", os.path.join(ROOT, "data", "gpt")),
            "kline_cache_tencent",
        ),
        help="K线缓存目录",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1500,
        help="每只股票拉取K线条数（全量回补用1500；新浪单请求最多1023）",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1000,
        help="缓存已有不少于该根数则跳过（新浪单次最多约1023；要1500根请单线程慢速拉腾讯）",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "eastmoney", "sina", "netease", "tencent"],
        default="auto",
        help="数据源；auto=东财→新浪→网易→腾讯",
    )
    parser.add_argument("--workers", type=int, default=1, help="并发数；回补建议 1")
    parser.add_argument(
        "--pause",
        type=float,
        default=0.25,
        help="每只股票请求后休眠秒数（防限流）",
    )
    parser.add_argument("--force", action="store_true", help="强制拉取，不跳过已有足够历史的股票")
    parser.add_argument("--limit", type=int, default=0, help="限制更新数量，0=全部")
    parser.add_argument(
        "--retry-rounds",
        type=int,
        default=1,
        help="首轮失败后对仍不足 min-rows 的股票再试几轮",
    )
    args = parser.parse_args()
    from app.kline_resource_lock import acquire_heavy_kline, release_heavy_kline

    acquire_heavy_kline()
    try:
        _prefetch_main_impl(args)
    finally:
        release_heavy_kline()


def _prefetch_main_impl(args) -> None:
    stock_list = stock_items_from_list_csv(args.stock_list)
    if not stock_list:
        raise RuntimeError("股票列表为空")

    if args.limit > 0:
        stock_list = stock_list[: args.limit]

    total = len(stock_list)
    print(f"待扫描: {total} 只")
    print(f"缓存目录: {args.cache_dir}")
    print(
        f"目标: count={args.count}, min_rows={args.min_rows}, "
        f"source={args.source}, workers={args.workers}, pause={args.pause}s"
    )

    def needs_update(code: str) -> Tuple[bool, Optional[List[KlineRow]]]:
        path = os.path.join(args.cache_dir, f"{code}.csv")
        cached = read_cached_kline(path)
        if args.force:
            return True, cached
        if cached and len(cached) >= args.min_rows and kline_is_fresh(cached):
            return False, cached
        if cached and len(cached) >= args.min_rows:
            return False, cached
        return True, cached

    to_update = []
    skipped = 0
    for item in stock_list:
        need, cached = needs_update(item.code)
        if need:
            to_update.append((item, cached))
        else:
            skipped += 1

    print(f"需回补: {len(to_update)} 只，已满足跳过: {skipped} 只")
    if not to_update:
        print("无需回补")
        return

    ok = 0
    fail = 0
    failed_items: list = []

    def worker(item, cached) -> Tuple[bool, str]:
        session = requests.Session()
        session.trust_env = False
        path = os.path.join(args.cache_dir, f"{item.code}.csv")
        try:
            rows, src = _fetch_one(item, args.count, args.source, session)
            if rows:
                merged = _merge_rows(cached or [], rows)
                write_cached_kline(path, merged)
                if args.pause:
                    time.sleep(args.pause)
                return True, src or ""
            if cached:
                return True, "保留缓存"
            return False, "多源均无数据"
        except Exception as exc:
            if cached:
                return True, f"保留缓存({exc})"
            return False, str(exc)

    def run_batch(batch, round_no: int) -> Tuple[int, int, list]:
        b_ok = 0
        b_fail = 0
        b_failed: list = []
        workers = max(1, args.workers)
        if workers <= 1:
            for item, cached in batch:
                success, info = worker(item, cached)
                if success:
                    b_ok += 1
                else:
                    b_fail += 1
                    b_failed.append((item, cached))
                done = b_ok + b_fail
                if done % 200 == 0 or done == len(batch):
                    print(f"  轮次{round_no}: {done}/{len(batch)} 成功 {b_ok} 失败 {b_fail}")
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(worker, item, cached): (item, cached)
                    for item, cached in batch
                }
                for future in as_completed(futures):
                    item, cached = futures[future]
                    success, info = future.result()
                    if success:
                        b_ok += 1
                    else:
                        b_fail += 1
                        b_failed.append((item, cached))
                    done = b_ok + b_fail
                    if done % 200 == 0 or done == len(batch):
                        print(f"  轮次{round_no}: {done}/{len(batch)} 成功 {b_ok} 失败 {b_fail}")
        return b_ok, b_fail, b_failed

    batch = list(to_update)
    for round_no in range(1, args.retry_rounds + 2):
        if not batch:
            break
        if round_no > 1:
            print(f"重试第 {round_no - 1} 轮: {len(batch)} 只（pause 加倍）")
            old_pause = args.pause
            args.pause = max(args.pause * 2, 0.5)
            args.workers = 1
            b_ok, b_fail, batch = run_batch(batch, round_no)
            args.pause = old_pause
        else:
            b_ok, b_fail, batch = run_batch(batch, round_no)
        ok += b_ok
        fail += b_fail
        failed_items = batch

    print("回补完成")
    print(f"成功: {ok}  失败: {fail}  跳过: {skipped}")
    if failed_items:
        print(f"仍失败 {len(failed_items)} 只，可稍后单线程重跑：")
        print(
            "  python3 scripts/prefetch_kline_tencent.py "
            f"--count {args.count} --min-rows {args.min_rows} "
            "--workers 1 --pause 0.4 --source auto"
        )


if __name__ == "__main__":
    main()
