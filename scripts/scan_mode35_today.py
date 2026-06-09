#!/usr/bin/env python3
"""mode35（前高压顶洗盘突破）A 类：扫描指定日期突破信号。

用法:
  python3 scripts/scan_mode35_today.py
  python3 scripts/scan_mode35_today.py --date 2026-05-14 --min-score 70
  python3 scripts/scan_mode35_today.py --code 300221 --date 2026-05-14
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date
from typing import Any, Dict, List, Tuple

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


def main() -> None:
    ap = argparse.ArgumentParser(description=MODE35_DISPLAY_NAME)
    ap.add_argument("--date", default="", help="信号日，默认缓存最新交易日")
    ap.add_argument("--min-score", type=int, default=70)
    ap.add_argument("--code", default="", help="仅测单股")
    ap.add_argument("--skip-st", action="store_true", default=True)
    ap.add_argument("--skip-bj", action="store_true", default=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = ScanConfig(min_score=args.min_score)
    kw = mode35_kw_from_scan_config(cfg)
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
        idx = next((i for i, r in enumerate(rows) if str(r.date)[:10] == signal_date), None)
        if idx is None:
            continue
        score = score_mode35_prior_high_breakout(rows, idx, code, name, **kw)
        if score < args.min_score:
            continue
        m = mode35_signal_metrics(rows, idx, code, name, **kw)
        if not m:
            continue
        hits.append(
            (
                score,
                {
                    "signal_date": m.get("signal_date", signal_date),
                    "code": code,
                    "name": name,
                    "score": score,
                    **{k: m[k] for k in m if k not in ("mode35_score", "anchor_date_idx")},
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
        f"mode35_breakout_{signal_date.replace('-', '')}.csv",
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if rows_out:
        fields = list(rows_out[0].keys())
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_out)

    print(f"\n{MODE35_DISPLAY_NAME} {signal_date}  命中 {len(rows_out)} 只 → {out}\n")
    hdr = f"{'代码':<8}{'名称':<10}{'收':>7}{'涨%':>6}{'前高':>7}{'前高日':>10}{'压顶日':>5}{'量比':>5}{'分':>4}"
    print(hdr)
    for r in rows_out[:40]:
        print(
            f"{r['code']:<8}{(r['name'] or '')[:10]:<10}{r['close']:7.2f}"
            f"{r.get('pct_chg', 0):6.2f}{r.get('prior_high', 0):7.2f}"
            f"{r.get('anchor_date', ''):>10}{int(r.get('under_days', 0)):5d}"
            f"{r.get('vol_ratio', 0):5.2f}{r['score']:4d}"
        )


if __name__ == "__main__":
    main()
