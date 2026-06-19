#!/usr/bin/env python3
"""mode39 区间扫描：大阳锚点回踩再升。

用法:
  python3 scripts/scan_mode39_period.py --start 2026-04-01 --end 2026-06-18
  python3 scripts/scan_mode39_period.py --code 300790 --start 2026-04-01 --end 2026-06-18
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
from app.mode39_bull_anchor_pullback import (
    MODE39_DISPLAY_NAME,
    dedupe_mode39_hits,
    mode39_kw_from_scan_config,
    mode39_signal_metrics,
    score_mode39_bull_anchor_pullback,
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
    ap = argparse.ArgumentParser(description=f"{MODE39_DISPLAY_NAME} 区间扫描")
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

    kw = mode39_kw_from_scan_config(ScanConfig(min_score=args.min_score))
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
            score = score_mode39_bull_anchor_pullback(rows, idx, code, name, **kw)
            if score < args.min_score:
                continue
            m = mode39_signal_metrics(rows, idx, code, name, **kw)
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

    hits = dedupe_mode39_hits(hits)

    out_dir = os.path.join(GPT_DATA_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out.strip() or os.path.join(
        out_dir,
        f"mode39_bull_anchor_pullback_{start_ymd[:7].replace('-', '_')}.csv",
    )
    fields = [
        "date",
        "code",
        "name",
        "score",
        "signal_style",
        "anchor_date",
        "anchor_close",
        "peak_date",
        "rally_pct",
        "pullback_peak_pct",
        "anchor_dist_pct",
        "signal_date",
        "exec_buy_date",
        "exec_buy_open",
        "buy_trigger_above",
        "pct_chg",
        "lower_shadow_ratio",
        "vol_shrink_ratio",
        "low",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for h in sorted(hits, key=lambda x: (-x["score"], x["date"])):
            w.writerow(h)

    print(f"{MODE39_DISPLAY_NAME}: {len(hits)} 条 → {out_path}")
    for h in sorted(hits, key=lambda x: (-x["score"], x["date"]))[:20]:
        print(
            f"  {h['date']} {h['code']} {h['name']} score={h['score']} "
            f"{h['signal_style']} 锚点{h['anchor_date']} 买点{h['exec_buy_date']}"
        )


if __name__ == "__main__":
    main()
