#!/usr/bin/env python3
"""mode38 区间扫描：大牛股关键位回踩。

用法:
  python3 scripts/scan_mode38_period.py --start 2026-06-01 --end 2026-06-17
  python3 scripts/scan_mode38_period.py --code 603929 --start 2025-11-01 --end 2026-06-17
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
from app.mode38_bull_ma_pullback import (
    MODE38_DISPLAY_NAME,
    dedupe_mode38_hits,
    mode38_kw_from_scan_config,
    mode38_signal_metrics,
    score_mode38_bull_ma_pullback,
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
    ap = argparse.ArgumentParser(description=f"{MODE38_DISPLAY_NAME} 区间扫描")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--code", default="", help="仅扫单股")
    ap.add_argument("--min-score", type=int, default=60)
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

    kw = mode38_kw_from_scan_config(ScanConfig(min_score=args.min_score))
    name_map = load_stock_list_csv(STOCK_LIST) if os.path.exists(STOCK_LIST) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if args.code.strip():
        oc = args.code.strip().zfill(6)
        stock_list = [s for s in stock_list if s.code.zfill(6) == oc]
    hits: List[Dict[str, Any]] = []

    for item in stock_list:
        code = item.code.zfill(6)
        name = (item.name or name_map.get(code, code) or "").strip()
        if args.skip_st and _is_st(name):
            continue
        if args.skip_bj and code.startswith("920"):
            continue
        rows = read_cached_kline_by_code(CACHE_DIR, code)
        if not rows or len(rows) < 150:
            continue
        idx_map = {r.date[:10]: i for i, r in enumerate(rows)}

        for d in trade_days:
            idx = idx_map.get(d)
            if idx is None:
                continue
            score = score_mode38_bull_ma_pullback(rows, idx, code, name, **kw)
            if score < args.min_score:
                continue
            m = mode38_signal_metrics(rows, idx, code, name, **kw)
            if not m:
                continue
            hits.append(
                {
                    "date": d,
                    "code": code,
                    "name": name,
                    "score": score,
                    **m,
                }
            )

    hits = dedupe_mode38_hits(hits)

    out_dir = os.path.join(GPT_DATA_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out.strip() or os.path.join(
        out_dir,
        f"mode38_bull_ma_pullback_{start_ymd[:7].replace('-', '_')}.csv",
    )
    fields = [
        "date",
        "code",
        "name",
        "score",
        "peak_date",
        "rally_pct",
        "pullback_pct",
        "support_ma",
        "support_ma_val",
        "low_dist_ma_pct",
        "close_dist_ma_pct",
        "pct_chg",
        "vol_shrink_ratio",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for h in sorted(hits, key=lambda x: (-x["score"], x["date"])):
            w.writerow(h)

    print(f"{MODE38_DISPLAY_NAME}: {len(hits)} 条 → {out_path}")
    for h in sorted(hits, key=lambda x: (-x["score"], x["date"]))[:20]:
        print(
            f"  {h['date']} {h['code']} {h['name']} score={h['score']} "
            f"MA{h['support_ma']} 回撤{h['pullback_pct']:.1f}% 前涨{h['rally_pct']:.0f}%"
        )


if __name__ == "__main__":
    main()
