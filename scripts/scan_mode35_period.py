#!/usr/bin/env python3
"""mode35（前高压顶洗盘突破）区间扫描：A 类突破日。

用法:
  python3 scripts/scan_mode35_period.py --start 2026-05-01 --end 2026-05-31
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
from app.mode35_prior_high_breakout import (
    MODE35_DISPLAY_NAME,
    mode35_kw_from_scan_config,
    mode35_signal_metrics,
    score_mode35_prior_high_breakout,
)
from app.paths import GPT_DATA_DIR
from app.scanner import ScanConfig, _is_st

CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
STOCK_LIST = os.path.join(GPT_DATA_DIR, "stock_list.csv")


def _trade_days(start_ymd: str, end_ymd: str) -> List[str]:
    ref = read_cached_kline_by_code(CACHE_DIR, "000001")
    if not ref:
        return []
    return sorted({r.date[:10] for r in ref if start_ymd <= r.date[:10] <= end_ymd})


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{MODE35_DISPLAY_NAME} 区间扫描")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--min-score", type=int, default=70)
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

    kw = mode35_kw_from_scan_config(ScanConfig(min_score=args.min_score))
    name_map = load_stock_list_csv(STOCK_LIST) if os.path.exists(STOCK_LIST) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    hits: List[Dict[str, Any]] = []

    for n, item in enumerate(stock_list):
        code = item.code.zfill(6)
        name = (item.name or name_map.get(code, code) or "").strip()
        if args.skip_st and _is_st(name):
            continue
        if args.skip_bj and code.startswith("920"):
            continue
        rows = read_cached_kline_by_code(CACHE_DIR, code)
        if not rows or len(rows) < 120:
            continue
        idx_map = {r.date[:10]: i for i, r in enumerate(rows)}

        for d in trade_days:
            idx = idx_map.get(d)
            if idx is None:
                continue
            score = score_mode35_prior_high_breakout(rows, idx, code, name, **kw)
            if score < args.min_score:
                continue
            m = mode35_signal_metrics(rows, idx, code, name, **kw)
            if not m:
                continue
            hits.append(
                {
                    "list_date": m.get("signal_date", d),
                    "signal_date": m.get("signal_date", d),
                    "event_type": "突破",
                    "code": code,
                    "name": name,
                    "score": score,
                    **{k: m[k] for k in m if k not in ("mode35_score", "anchor_date_idx", "event_type")},
                }
            )
        if (n + 1) % 1000 == 0:
            print(f"进度 {n+1}/{len(stock_list)}  命中 {len(hits)}", flush=True)

    hits.sort(key=lambda x: (x["signal_date"], -int(x["score"]), x["code"]))
    out_path = args.out.strip() or os.path.join(
        GPT_DATA_DIR,
        "results",
        f"mode35_breakout_{start_ymd[:7].replace('-', '_')}.csv",
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if hits:
        fields = list(hits[0].keys())
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(hits)

    by_day: Dict[str, int] = {}
    for h in hits:
        by_day[h["signal_date"]] = by_day.get(h["signal_date"], 0) + 1
    print(f"\n{MODE35_DISPLAY_NAME} {start_ymd}～{end_ymd}  共 {len(hits)} 条 → {out_path}")
    for d in sorted(by_day.keys()):
        print(f"  {d}: {by_day[d]} 只")


if __name__ == "__main__":
    main()
