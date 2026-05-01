#!/usr/bin/env python3
"""mode93：低位放量涨停后回调到涨停日低点附近，筛选指定日期个股并按得分降序输出。

形态（按你的定义量化）：
- 低位：近3天出现过去120日最低点（以 low 判定）
- 次日：成交量放大≥3倍 且 涨停
- 之后：股价慢慢回调，到涨停日最低价A附近（close ∈ [0.99A, 1.02A]）
- 信号日：回到上述区间的当天（距离涨停日≤20天）

用法:
  python3 scripts/scan_mode93_today.py
  python3 scripts/scan_mode93_today.py --date 2026-04-30 --min-score 70 --top 10
"""

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
KLINE_COUNT = 320


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="mode93 筛选指定日期个股（低位放量涨停回调模型）")
    ap.add_argument("--date", default=date.today().strftime("%Y-%m-%d"), help="信号日，默认今天")
    ap.add_argument("--min-score", type=int, default=70, help="最低得分，默认70")
    ap.add_argument("--top", type=int, default=10, help="输出前N名，默认10")
    ap.add_argument("--lookback", type=int, default=20, help="涨停事件回溯天数（默认20）")
    ap.add_argument("--low-window", type=int, default=120, help="低位窗口（默认120）")
    ap.add_argument("--low-recent", type=int, default=10, help="低位近N天（默认10）")
    ap.add_argument("--vol-mult", type=float, default=3.0, help="次日放量倍数（默认3）")
    ap.add_argument("--pull-min", type=float, default=0.95, help="回调区间下沿（默认0.95*A）")
    ap.add_argument("--pull-max", type=float, default=1.05, help="回调区间上沿（默认1.05*A）")
    ap.add_argument("--pull-days", type=int, default=20, help="回调最长天数（默认20）")
    ap.add_argument(
        "--allow-refresh",
        action="store_true",
        help="允许从网络补全 K 线；默认仅用本地缓存",
    )
    args = ap.parse_args()

    prefer_local = not args.allow_refresh
    target_date = args.date
    min_score = int(args.min_score or 0)
    top_n = max(1, int(args.top or 10))

    name_map = load_stock_list_csv(STOCK_LIST_CSV) if os.path.exists(STOCK_LIST_CSV) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if not stock_list:
        print("股票列表为空，请确认缓存目录:", CACHE_DIR)
        raise SystemExit(1)

    config = ScanConfig(min_score=min_score, max_results=50000, max_market_cap=None)
    config.mode93_lookback_days = int(args.lookback or 20)
    config.mode93_low_window = int(args.low_window or 120)
    config.mode93_low_recent_days = int(args.low_recent or 10)
    config.mode93_vol_mult = float(args.vol_mult or 3.0)
    config.mode93_pullback_min = float(args.pull_min or 0.95)
    config.mode93_pullback_max = float(args.pull_max or 1.05)
    config.mode93_pullback_max_days = int(args.pull_days or 20)

    def kline_loader(item):
        return tencent.get_kline_cached(
            item.code,
            cache_dir=CACHE_DIR,
            count=KLINE_COUNT,
            min_len=260,
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
            use_mode93=True,
            use_71x_standard=True,
        )
    )

    rows = []
    for r in results:
        code = r.code.zfill(6) if len(r.code) < 6 else r.code
        rows.append((code, r.name or code, int(r.score)))
    rows.sort(key=lambda x: (-x[2], x[0]))

    print("mode93 信号日: %s  最低分: %d  命中: %d 只\n" % (target_date, min_score, len(rows)))
    if not rows:
        print("当日无满足条件的个股。可尝试: --min-score 60 或更换 --date")
        return

    print("Top%d:" % top_n)
    print("代码      名称            得分")
    print("-" * 36)
    for code, name, score in rows[:top_n]:
        print("%s  %-14s  %s" % (code, (name or "")[:14], score))


if __name__ == "__main__":
    main()

