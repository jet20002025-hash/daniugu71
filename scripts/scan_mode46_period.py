#!/usr/bin/env python3
"""mode46 区间扫描：前高附近二次攻击。

用法:
  python3 scripts/scan_mode46_period.py --code 000021 --start 2026-06-01 --end 2026-06-26
  python3 scripts/scan_mode46_period.py --high-tech --start 2026-06-01 --end 2026-06-26
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
from app.mode46_prior_high_retest import (
    MODE46_DISPLAY_NAME,
    dedupe_mode46_hits,
    mode46_kw_from_scan_config,
    mode46_signal_metrics,
    score_mode46_prior_high_retest,
)
from app.paths import GPT_DATA_DIR
from app.high_tech_universe import HIGH_TECH_STOCK_LIST_CSV, high_tech_code_set
from app.scanner import ScanConfig, _is_st

CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
STOCK_LIST = os.path.join(GPT_DATA_DIR, "stock_list.csv")


def _trade_days(start_ymd: str, end_ymd: str) -> List[str]:
    ref = read_cached_kline_by_code(CACHE_DIR, "000001")
    if not ref:
        return []
    return sorted({r.date[:10] for r in ref if start_ymd <= r.date[:10] <= end_ymd})


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{MODE46_DISPLAY_NAME} 区间扫描")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--code", default="", help="仅扫单股")
    ap.add_argument("--high-tech", action="store_true")
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

    kw = mode46_kw_from_scan_config(ScanConfig(min_score=args.min_score))
    stock_list_path = HIGH_TECH_STOCK_LIST_CSV if args.high_tech else STOCK_LIST
    name_map = load_stock_list_csv(stock_list_path) if os.path.exists(stock_list_path) else {}
    if args.high_tech:
        ht = high_tech_code_set()
        if not ht:
            print("高科技池为空")
            sys.exit(1)
        print(f"高科技板块扫描: {len(ht)} 只")
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if args.high_tech:
        stock_list = [s for s in stock_list if s.code.zfill(6) in ht]
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
            score = score_mode46_prior_high_retest(rows, idx, code, name, **kw)
            if score < args.min_score:
                continue
            m = mode46_signal_metrics(rows, idx, code, name, **kw)
            if not m:
                continue
            hits.append(
                {
                    "code": code,
                    "name": name,
                    "signal_date": m.get("signal_date", d),
                    "score": score,
                    "peak_date": m.get("peak_date"),
                    "prior_high": m.get("prior_high"),
                    "trough_date": m.get("trough_date"),
                    "pullback_pct": m.get("pullback_pct"),
                    "high_dist_pct": m.get("high_dist_pct"),
                    "close_dist_pct": m.get("close_dist_pct"),
                    "rebound_from_trough_pct": m.get("rebound_from_trough_pct"),
                    "vol_ratio": m.get("vol_ratio"),
                    "exec_buy_date": m.get("exec_buy_date"),
                    "buy_trigger_above": m.get("buy_trigger_above"),
                }
            )

    hits = dedupe_mode46_hits(hits)
    hits.sort(key=lambda x: (-int(x.get("score", 0) or 0), str(x.get("signal_date", ""))))

    out_dir = os.path.join(GPT_DATA_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out.strip() or os.path.join(
        out_dir,
        f"mode46_prior_high_retest_{start_ymd.replace('-', '_')}_{end_ymd.replace('-', '_')}.csv",
    )
    fields = [
        "code",
        "name",
        "signal_date",
        "score",
        "peak_date",
        "prior_high",
        "trough_date",
        "pullback_pct",
        "high_dist_pct",
        "close_dist_pct",
        "rebound_from_trough_pct",
        "vol_ratio",
        "exec_buy_date",
        "buy_trigger_above",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for h in hits:
            w.writerow(h)

    print(f"{MODE46_DISPLAY_NAME}: {len(hits)} 条 → {out_path}")
    for h in hits[:20]:
        print(
            f"  {h['code']} {h['name']} {h['signal_date']} score={h['score']} "
            f"前高{h['peak_date']} 距高{h['high_dist_pct']}%"
        )


if __name__ == "__main__":
    main()
