#!/usr/bin/env python3
"""mode98 区间扫描：寻找信号日收盘价最低的个股（中文说明）。

在指定日期范围内运行 mode98（日/周/月 KDJ 9,3,3，K/D/J 均低于阈值），
对每条命中记录取「信号日收盘价」，按价格从低到高排序，便于筛低价股。

用法:
  python3 scripts/mode98_period_lowest_close.py --start 2026-04-01 --end 2026-04-30 --top 50
  python3 scripts/mode98_period_lowest_close.py --start 2026-04-01 --end 2026-04-30 --excel out.xlsx
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

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


def _signal_day_close(
    klines: List[Any],
    signal_date: str,
) -> Optional[float]:
    sd = str(signal_date).strip()[:10]
    for row in klines:
        if str(row.date)[:10] == sd:
            return float(row.close)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="mode98 区间扫描：按信号日收盘价从低到高榜单")
    ap.add_argument("--start", required=True, help="区间起始日 YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="区间结束日 YYYY-MM-DD（含）")
    ap.add_argument("--top", type=int, default=50, help="输出前 N 条，默认 50")
    ap.add_argument("--min-score", type=int, default=50, help="mode98 最低得分，默认 50")
    ap.add_argument("--kdj-threshold", type=float, default=20.0, help="K/D/J 严格上限，默认 20")
    ap.add_argument(
        "--allow-refresh",
        action="store_true",
        help="允许联网补 K 线；默认仅用本地缓存",
    )
    ap.add_argument("--csv", default="", metavar="PATH", help="可选：导出 CSV")
    ap.add_argument("--excel", default="", metavar="PATH", help="可选：导出 xlsx（需 pandas）")
    args = ap.parse_args()

    prefer_local = not args.allow_refresh
    name_map = load_stock_list_csv(STOCK_LIST_CSV) if os.path.exists(STOCK_LIST_CSV) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if not stock_list:
        print("股票列表为空:", CACHE_DIR)
        raise SystemExit(1)

    config = ScanConfig(
        min_score=int(args.min_score),
        max_results=500_000,
        max_market_cap=None,
        mode98_kdj_threshold=float(args.kdj_threshold),
    )

    code_to_item = {item.code.zfill(6): item for item in stock_list}
    for it in stock_list:
        code_to_item.setdefault(it.code, it)

    def kline_loader(item):
        return tencent.get_kline_cached(
            item.code,
            cache_dir=CACHE_DIR,
            count=KLINE_COUNT,
            min_len=220,
            prefer_local=prefer_local,
        )

    print(
        f"mode98 扫描 {args.start}～{args.end}（寻找信号日收盘价最低个股）…",
        flush=True,
    )
    results = list(
        scan_with_mode3(
            stock_list,
            config,
            CACHE_DIR,
            progress_cb=None,
            kline_loader=kline_loader,
            start_date=args.start,
            cutoff_date=args.end,
            use_mode98=True,
            use_71x_standard=True,
        )
    )

    enriched: List[Tuple[float, str, str, str, int]] = []
    for r in results:
        sig = str((r.metrics or {}).get("signal_date") or "").strip()[:10]
        if not sig:
            continue
        code = str(r.code).strip().zfill(6)
        item = code_to_item.get(code) or code_to_item.get(r.code)
        if not item:
            continue
        kl = kline_loader(item)
        if not kl:
            continue
        c = _signal_day_close(kl, sig)
        if c is None or c <= 0:
            continue
        enriched.append((c, sig, code, r.name or code, int(r.score)))

    enriched.sort(key=lambda x: (x[0], x[1], x[2]))
    top_n = max(1, int(args.top))

    print(f"\n共命中 {len(results)} 条记录，有效收盘价 {len(enriched)} 条；按信号日收盘价升序 Top{top_n}：\n")
    print(f"{'信号日':12} {'代码':8} {'名称':14} {'信号日收盘':>10} {'得分':>6}")
    print("-" * 56)
    for c, sig, code, name, sc in enriched[:top_n]:
        print(f"{sig:12} {code:8} {name[:14]:14} {c:10.2f} {sc:6d}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["signal_date", "code", "name", "signal_close", "score"])
            for c, sig, code, name, sc in enriched:
                w.writerow([sig, code, name, f"{c:.4f}", sc])
        print(f"\n已写 CSV: {args.csv}")

    if args.excel:
        try:
            import pandas as pd
        except ImportError:
            print("未安装 pandas，跳过 Excel。可: pip install pandas openpyxl")
        else:
            df = pd.DataFrame(
                [
                    {"信号日": sig, "代码": code, "名称": name, "信号日收盘": c, "得分": sc}
                    for c, sig, code, name, sc in enriched
                ]
            )
            df.to_excel(args.excel, index=False)
            print(f"已写 Excel: {args.excel}")


if __name__ == "__main__":
    main()
