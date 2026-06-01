#!/usr/bin/env python3
"""mode34 电科严格模版：区间扫描（观察日 + 信号日盘中买）。

电科范例：5/22 观察入池 → 5/25 盘中突破昨高买入（非 26 日）。

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
    match_mode34_watchlist,
    mode34_kw_from_scan_config,
    mode34_prebuy_signal_metrics,
    score_mode34_prebuy_signal,
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
    ap = argparse.ArgumentParser(description="mode34 区间扫描（观察+信号日盘中买）")
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

    kw = mode34_kw_from_scan_config(ScanConfig(min_score=args.min_score))
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
        if not rows or len(rows) < 80:
            continue
        idx_map = {r.date[:10]: i for i, r in enumerate(rows)}

        for d in trade_days:
            idx = idx_map.get(d)
            if idx is None:
                continue
            w = match_mode34_watchlist(rows, idx, code, name, **kw)
            if w:
                hits.append(
                    {
                        "event_type": "观察",
                        "list_date": w.get("watch_date", d),
                        "signal_date": w.get("signal_date", ""),
                        "watch_date": w.get("watch_date", ""),
                        "exec_buy_date": w.get("exec_buy_date", ""),
                        "confirm_date": "",
                        "buy_mode": w.get("buy_mode", "watch"),
                        "code": code,
                        "name": name,
                        "score": w.get("watch_score", 0),
                        "advice": "列入观察池",
                        "action": f"待信号日{w.get('signal_date', '')}盘中买",
                        "close": w.get("close"),
                        "pct_chg": w.get("pct_chg"),
                        "bottom_pos_pct": w.get("bottom_pos_pct"),
                        "base_date": w.get("base_date"),
                        "peak_date": w.get("peak_date"),
                        "surge_rise_pct": w.get("surge_rise_pct"),
                        "pullback_dd_pct": w.get("pullback_dd_pct"),
                        "pullback_days": w.get("pullback_days"),
                        "vol_ratio": w.get("vol_ratio"),
                        "buy_trigger_above": "",
                    }
                )

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
                    "event_type": "信号",
                    "list_date": m.get("signal_date", sig),
                    "signal_date": m.get("signal_date", sig),
                    "watch_date": m.get("watch_date", ""),
                    "exec_buy_date": m.get("exec_buy_date", ""),
                    "confirm_date": m.get("confirm_date", ""),
                    "buy_mode": m.get("buy_mode", "intraday"),
                    "code": code,
                    "name": name,
                    "score": score,
                    "advice": m.get("advice", ""),
                    "action": m.get("action", ""),
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

    hits.sort(key=lambda x: (x["list_date"], x["event_type"], -int(x["score"]), x["code"]))
    ym = start_ymd[:7].replace("-", "_")
    out_path = args.out.strip() or os.path.join(
        GPT_DATA_DIR, "results", f"mode34_strict_{ym}.csv"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if hits:
        fields = list(hits[0].keys())
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(hits)

    by_watch: Dict[str, int] = {}
    by_sig: Dict[str, int] = {}
    for h in hits:
        if h["event_type"] == "观察":
            by_watch[h["list_date"]] = by_watch.get(h["list_date"], 0) + 1
        else:
            by_sig[h["list_date"]] = by_sig.get(h["list_date"], 0) + 1
    print(f"\nmode34 {start_ymd}～{end_ymd}  共 {len(hits)} 条 → {out_path}")
    print("观察入池:")
    for d in trade_days:
        if by_watch.get(d):
            print(f"  {d}: {by_watch[d]} 只")
    print("信号日(盘中买):")
    for d in trade_days:
        if by_sig.get(d):
            print(f"  {d}: {by_sig[d]} 只")


if __name__ == "__main__":
    main()
