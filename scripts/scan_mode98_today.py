#!/usr/bin/env python3
"""基于 mode98 筛选指定日期（默认今日）的个股，按得分降序输出。

mode98：日线、周线、月线 KDJ 均为 (9,3,3)，且 K、D、J 三线均严格小于 20（可调）。

用法:
  python3 scripts/scan_mode98_today.py
  python3 scripts/scan_mode98_today.py --date 2026-03-02
  python3 scripts/scan_mode98_today.py --min-score 55 --kdj-threshold 20
"""

from __future__ import annotations

import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import tencent
from app.eastmoney import list_cached_stocks_flat, load_stock_list_csv
from app.paths import GPT_DATA_DIR
from app.scanner import ScanConfig, scan_with_mode3


CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
STOCK_LIST_CSV = os.path.join(GPT_DATA_DIR, "stock_list.csv")
KLINE_COUNT = 300


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="mode98：日/周/月 KDJ(9,3,3) 三线均低于阈值")
    ap.add_argument("--date", default=date.today().strftime("%Y-%m-%d"), help="信号日，默认今天")
    ap.add_argument("--min-score", type=int, default=55, help="最低得分，默认 55")
    ap.add_argument("--top", type=int, default=30, help="输出前 N 名，默认 30")
    ap.add_argument("--kdj-threshold", type=float, default=20.0, help="K/D/J 上限（不含），默认 20")
    ap.add_argument("--kdj-n", type=int, default=9, help="KDJ 周期 N，默认 9")
    ap.add_argument("--kdj-m1", type=int, default=3, help="K 平滑 M1，默认 3")
    ap.add_argument("--kdj-m2", type=int, default=3, help="D 平滑 M2，默认 3")
    ap.add_argument(
        "--allow-refresh",
        action="store_true",
        help="允许从网络补全 K 线；默认仅用本地缓存",
    )
    args = ap.parse_args()

    prefer_local = not args.allow_refresh
    target_date = args.date
    min_score = int(args.min_score or 0)
    top_n = max(1, int(args.top or 30))

    name_map = load_stock_list_csv(STOCK_LIST_CSV) if os.path.exists(STOCK_LIST_CSV) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if not stock_list:
        print("股票列表为空，请确认缓存目录:", CACHE_DIR)
        raise SystemExit(1)

    config = ScanConfig(
        min_score=min_score,
        max_results=50000,
        max_market_cap=None,
        mode98_kdj_threshold=float(args.kdj_threshold),
        mode98_kdj_n=int(args.kdj_n),
        mode98_kdj_m1=int(args.kdj_m1),
        mode98_kdj_m2=int(args.kdj_m2),
    )

    def kline_loader(item):
        return tencent.get_kline_cached(
            item.code,
            cache_dir=CACHE_DIR,
            count=KLINE_COUNT,
            min_len=220,
            prefer_local=prefer_local,
        )

    results = list(
        scan_with_mode3(
            stock_list,
            config,
            CACHE_DIR,
            progress_cb=None,
            kline_loader=kline_loader,
            start_date=target_date,
            cutoff_date=target_date,
            use_mode98=True,
            use_71x_standard=True,
        )
    )

    rows_out = []
    for r in results:
        code = r.code.zfill(6) if len(r.code) < 6 else r.code
        rows_out.append((code, r.name or code, int(r.score)))
    rows_out.sort(key=lambda x: (-x[2], x[0]))

    print(
        "mode98 信号日: %s  KDJ(%d,%d,%d) K/D/J均<%s  最低分: %d  命中: %d 只\n"
        % (
            target_date,
            config.mode98_kdj_n,
            config.mode98_kdj_m1,
            config.mode98_kdj_m2,
            config.mode98_kdj_threshold,
            min_score,
            len(rows_out),
        )
    )
    if not rows_out:
        print("当日无满足条件的个股。可尝试降低 --min-score 或放宽 --kdj-threshold")
        return

    print("Top%d:" % top_n)
    print("代码      名称            得分")
    print("-" * 36)
    for code, name, score in rows_out[:top_n]:
        print("%s  %-14s  %s" % (code, (name or "")[:14], score))


if __name__ == "__main__":
    main()
