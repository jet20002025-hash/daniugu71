#!/usr/bin/env python3
"""mode34 电科严格模版：扫描指定日期预案买点（信号日）。

用法:
  python3 scripts/scan_mode34_today.py
  python3 scripts/scan_mode34_today.py --date 2026-05-26 --min-score 60
  python3 scripts/scan_mode34_today.py --code 600850 --date 2026-05-26
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.eastmoney import list_cached_stocks_flat, load_stock_list_csv, read_cached_kline_by_code
from app.mode34_bottom_break_pullback import (
    mode34_kw_from_scan_config,
    mode34_prebuy_signal_metrics,
    score_mode34_prebuy_signal,
)
from app.paths import GPT_DATA_DIR
from app.scanner import ScanConfig, _is_st

CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
STOCK_LIST = os.path.join(GPT_DATA_DIR, "stock_list.csv")


def _find_date_index(rows, ymd: str) -> Optional[int]:
    ymd = ymd.strip()[:10]
    for i, r in enumerate(rows):
        if str(r.date)[:10] == ymd:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="mode34 严格模版（预案买点=信号日）")
    ap.add_argument("--date", default="", help="信号日（预案日），默认缓存最新交易日")
    ap.add_argument("--min-score", type=int, default=72)
    ap.add_argument("--code", default="", help="仅测单股")
    ap.add_argument("--skip-st", action="store_true", default=True)
    ap.add_argument("--skip-bj", action="store_true", default=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = ScanConfig(min_score=args.min_score)
    kw = mode34_kw_from_scan_config(cfg)
    name_map = load_stock_list_csv(STOCK_LIST) if os.path.exists(STOCK_LIST) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if args.code.strip():
        oc = args.code.strip().zfill(6)
        stock_list = [s for s in stock_list if s.code.zfill(6) == oc]
        if not stock_list:
            stock_list = [type("X", (), {"code": oc, "name": name_map.get(oc, oc)})()]

    signal_date = args.date.strip()[:10]
    if not signal_date:
        ref = read_cached_kline_by_code(CACHE_DIR, "000001")
        signal_date = ref[-1].date[:10] if ref else date.today().isoformat()

    hits: List[Tuple[int, Dict[str, Any]]] = []
    for n, item in enumerate(stock_list):
        code = item.code.zfill(6)
        name = (item.name or name_map.get(code, code) or "").strip()
        if args.skip_st and _is_st(name):
            continue
        if args.skip_bj and code.startswith("920"):
            continue
        rows = read_cached_kline_by_code(CACHE_DIR, code)
        if not rows:
            continue
        idx = _find_date_index(rows, signal_date)
        if idx is None:
            continue
        score = score_mode34_prebuy_signal(rows, idx, code, name, **kw)
        if score < args.min_score:
            continue
        m = mode34_prebuy_signal_metrics(rows, idx, code, name, **kw)
        if not m:
            continue
        hits.append(
            (
                score,
                {
                    "signal_date": m.get("signal_date", signal_date),
                    "watch_date": m.get("watch_date", ""),
                    "exec_buy_date": m.get("exec_buy_date", ""),
                    "confirm_date": m.get("confirm_date", ""),
                    "code": code,
                    "name": name,
                    "score": score,
                    "advice": m.get("advice", ""),
                    **{k: m[k] for k in m if not str(k).startswith("mode34_")},
                    "base_date": m.get("mode34_base_date"),
                    "peak_date": m.get("mode34_peak_date"),
                },
            )
        )
        if (n + 1) % 1500 == 0:
            print(f"进度 {n+1}/{len(stock_list)}  命中 {len(hits)}", flush=True)

    hits.sort(key=lambda x: (-x[0], x[1]["code"]))
    rows_out = [h[1] for h in hits]

    out = args.out or os.path.join(
        GPT_DATA_DIR,
        "results",
        f"mode34_bottom_break_pullback_{signal_date.replace('-', '')}.csv",
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if rows_out:
        fields = list(rows_out[0].keys())
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_out)

    print(f"\nmode34 信号日 {signal_date}  命中 {len(rows_out)} 只 → {out}\n")
    print(f"{'代码':<8}{'名称':<10}{'收':>7}{'信号涨%':>7}{'底部%':>6}{'突破涨%':>7}{'回踩%':>6}{'回踩日':>5}{'分':>4}")
    for r in rows_out[:40]:
        print(
            f"{r['code']:<8}{(r['name'] or '')[:10]:<10}{r['close']:7.2f}"
            f"{r.get('pct_chg',0):7.2f}{r.get('bottom_pos_pct',0):6.1f}"
            f"{r.get('surge_rise_pct',0):7.1f}{r.get('pullback_dd_pct',0):6.1f}"
            f"{int(r.get('pullback_days',0)):5d}{r['score']:4d}"
        )


if __name__ == "__main__":
    main()
