#!/usr/bin/env python3
"""mode34 电科模版观察池：按观察日区间逐日全市场扫描。

用法:
  python3 scripts/scan_mode34_watch_period.py --start 2026-05-01 --end 2026-05-31
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.eastmoney import list_cached_stocks_flat, load_stock_list_csv, read_cached_kline_by_code
from app.mode34_bottom_break_pullback import mode34_kw_from_scan_config
from app.paths import GPT_DATA_DIR
from app.scanner import ScanConfig
from scripts.scan_mode34_watch_prebuy import _scan_watch

CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
STOCK_LIST = os.path.join(GPT_DATA_DIR, "stock_list.csv")


def _trade_days(start_ymd: str, end_ymd: str) -> List[str]:
    ref = read_cached_kline_by_code(CACHE_DIR, "000001")
    if not ref:
        return []
    return sorted({r.date[:10] for r in ref if start_ymd <= r.date[:10] <= end_ymd})


def main() -> None:
    ap = argparse.ArgumentParser(description="mode34 电科模版观察池区间扫描")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--skip-st", action="store_true", default=True)
    ap.add_argument("--skip-bj", action="store_true", default=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    start_ymd = args.start.strip()[:10]
    end_ymd = args.end.strip()[:10]
    trade_days = _trade_days(start_ymd, end_ymd)
    if not trade_days:
        print("区间内无交易日")
        sys.exit(1)

    kw = mode34_kw_from_scan_config(ScanConfig())
    name_map = load_stock_list_csv(STOCK_LIST) if os.path.exists(STOCK_LIST) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)

    hits: List[Dict[str, Any]] = []
    for d in trade_days:
        day_hits = _scan_watch(
            stock_list,
            name_map,
            d,
            kw,
            skip_st=args.skip_st,
            skip_bj=args.skip_bj,
        )
        hits.extend(day_hits)
        if day_hits:
            print(f"  {d}: {len(day_hits)} 只", flush=True)

    hits.sort(key=lambda x: (x["watch_date"], -x["watch_score"], x["code"]))
    ym = start_ymd[:7].replace("-", "_")
    out_path = args.out.strip() or os.path.join(
        GPT_DATA_DIR, "results", f"mode34_watch_dianke_{ym}.csv"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if hits:
        fields = list(hits[0].keys())
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(hits)

    codes = {h["code"] for h in hits}
    print(f"\n观察池 {start_ymd}～{end_ymd}  命中 {len(hits)} 条  去重 {len(codes)} 只 → {out_path}")


if __name__ == "__main__":
    main()
