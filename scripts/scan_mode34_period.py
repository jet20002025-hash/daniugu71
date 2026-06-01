#!/usr/bin/env python3
"""mode34 电科严格模版：预案买点日（信号日）区间扫描。

信号日 = 预案日（如 5/25），观察 = 上一交易日（如 5/22），执行买 = 信号日尾盘（同 5/25）。

用法:
  python3 scripts/scan_mode34_period.py --start 2026-05-01 --end 2026-05-31
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
from app.mode34_bottom_break_pullback import (
    mode34_prebuy_signal_metrics,
    score_mode34_prebuy_signal,
)
from app.paths import GPT_DATA_DIR
from app.scanner import ScanConfig, _is_st
from app.mode34_bottom_break_pullback import mode34_kw_from_scan_config

CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
STOCK_LIST = os.path.join(GPT_DATA_DIR, "stock_list.csv")


def _trade_days(start_ymd: str, end_ymd: str) -> List[str]:
    ref = read_cached_kline_by_code(CACHE_DIR, "000001")
    if not ref:
        return []
    return sorted({r.date[:10] for r in ref if start_ymd <= r.date[:10] <= end_ymd})


def main() -> None:
    ap = argparse.ArgumentParser(description="mode34 严格模版区间扫描（预案买点=信号日）")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--min-score", type=int, default=72)
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

    cfg = ScanConfig(min_score=args.min_score)
    kw = mode34_kw_from_scan_config(cfg)
    name_map = load_stock_list_csv(STOCK_LIST) if os.path.exists(STOCK_LIST) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    day_set = set(trade_days)

    hits: List[Dict[str, Any]] = []
    for n, item in enumerate(stock_list):
        code = item.code.zfill(6)
        name = (item.name or name_map.get(code, code) or "").strip()
        if args.skip_st and _is_st(name):
            continue
        if args.skip_bj and code.startswith("920"):
            continue
        rows = read_cached_kline_by_code(CACHE_DIR, code)
        if not rows or len(rows) < 80:
            continue
        idx_map = {r.date[:10]: i for i, r in enumerate(rows)}
        for sig in trade_days:
            idx = idx_map.get(sig)
            if idx is None:
                continue
            score = score_mode34_prebuy_signal(rows, idx, code, name, **kw)
            if score < args.min_score:
                continue
            m = mode34_prebuy_signal_metrics(rows, idx, code, name, **kw)
            if not m:
                continue
            hits.append(
                {
                    "signal_date": m.get("signal_date", sig),
                    "watch_date": m.get("watch_date", ""),
                    "exec_buy_date": m.get("exec_buy_date", ""),
                    "confirm_date": m.get("confirm_date", ""),
                    "code": code,
                    "name": name,
                    "score": score,
                    "advice": m.get("advice", ""),
                    "close": m.get("close"),
                    "pct_chg": m.get("pct_chg"),
                    "bottom_pos_pct": m.get("bottom_pos_pct"),
                    "base_date": m.get("mode34_base_date"),
                    "peak_date": m.get("mode34_peak_date"),
                    "surge_rise_pct": m.get("surge_rise_pct"),
                    "pullback_dd_pct": m.get("pullback_dd_pct"),
                    "pullback_days": m.get("pullback_days"),
                    "vol_ratio": m.get("vol_ratio"),
                    "buy_trigger_above": m.get("buy_trigger_above"),
                }
            )
        if (n + 1) % 1000 == 0:
            print(f"进度 {n+1}/{len(stock_list)}  命中 {len(hits)}", flush=True)

    hits.sort(key=lambda x: (x["signal_date"], -x["score"], x["code"]))
    ym = start_ymd[:7].replace("-", "_")
    out_path = args.out.strip() or os.path.join(
        GPT_DATA_DIR, "results", f"mode34_strict_{ym}.csv"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fields = list(hits[0].keys()) if hits else []
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(hits)

    by_day: Dict[str, int] = {}
    for h in hits:
        by_day[h["signal_date"]] = by_day.get(h["signal_date"], 0) + 1
    print(f"\nmode34 {start_ymd}～{end_ymd}  命中 {len(hits)} 条 → {out_path}")
    for d in trade_days:
        c = by_day.get(d, 0)
        if c:
            print(f"  {d}: {c} 只")


if __name__ == "__main__":
    main()
