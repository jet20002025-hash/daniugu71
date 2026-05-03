#!/usr/bin/env python3
"""mode32（3+2）：实体首板后 5 日整理，信号日 = 首板后第 6 个交易日。

规则摘要（与 scanner 内一致，参数见 ScanConfig.mode32_*）：
- 首板日前约 60 日横盘（高低区间/均价 ≤ 阈值）
- 首板：实体涨停，剔除近似一字板、T 字板，ST 不参与
- 次日：实体偏小、成交量 ≥ 首板量
- 第 2～3 日：量能梯形递减，收盘贴近首板最高价
- 第 4～5 日：小实体、量能低迷
- 信号日：收盘不低于首板实体中轴；五日最低价不破中轴过多

用法:
  python3 scripts/scan_mode32_today.py --date 2026-04-30
  python3 scripts/scan_mode32_today.py --date 2026-04-30 --min-score 65
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
KLINE_COUNT = 320


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="mode32（实体首板 3+2）筛选指定信号日")
    ap.add_argument("--date", default=date.today().strftime("%Y-%m-%d"), help="信号日（首板后第6个交易日），默认今天")
    ap.add_argument("--min-score", type=int, default=62, help="最低得分，默认 62（mode32 分数区间约 62～92）")
    ap.add_argument("--top", type=int, default=40, help="打印前 N 名，默认 40")
    ap.add_argument(
        "--allow-refresh",
        action="store_true",
        help="允许联网补 K 线；默认仅用本地缓存",
    )
    args = ap.parse_args()

    prefer_local = not args.allow_refresh
    target_date = args.date
    min_score = int(args.min_score or 0)
    top_n = max(1, int(args.top or 40))

    name_map = load_stock_list_csv(STOCK_LIST_CSV) if os.path.exists(STOCK_LIST_CSV) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if not stock_list:
        print("股票列表为空:", CACHE_DIR)
        raise SystemExit(1)

    config = ScanConfig(min_score=min_score, max_results=50000, max_market_cap=None)

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
            use_mode32=True,
            use_71x_standard=True,
        )
    )

    rows_out = []
    for r in results:
        code = r.code.zfill(6) if len(r.code) < 6 else r.code
        m = r.metrics or {}
        ld = m.get("mode32_limit_date") or ""
        rows_out.append((code, r.name or code, int(r.score), str(ld)))
    rows_out.sort(key=lambda x: (-x[2], x[0]))

    print("mode32 信号日: %s  最低分: %d  命中: %d 只\n" % (target_date, min_score, len(rows_out)))
    if not rows_out:
        print("当日无命中。可调低 --min-score 或确认该日是否为「某实体首板后的第 6 个交易日」。")
        return

    print("Top%d（首板日 mode32_limit_date）:" % top_n)
    print("%s  %-14s  %s  %s" % ("代码", "名称", "得分", "首板日"))
    print("-" * 44)
    for code, name, score, ld in rows_out[:top_n]:
        print("%s  %-14s  %s  %s" % (code, (name or "")[:14], score, ld))


if __name__ == "__main__":
    main()
