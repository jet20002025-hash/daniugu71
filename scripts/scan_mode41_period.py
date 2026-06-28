#!/usr/bin/env python3
"""mode41 区间扫描：周线关键位回踩缩量。

用法:
  python3 scripts/scan_mode41_period.py --start 2026-06-01 --end 2026-06-24
  python3 scripts/scan_mode41_period.py --support-ma 20 --start 2026-06-01 --end 2026-06-24
  python3 scripts/scan_mode41_period.py --code 603929 --start 2025-11-01 --end 2026-06-24
  python3 scripts/scan_mode41_period.py --high-tech --start 2026-06-01 --end 2026-06-24
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
from app.mode41_weekly_ma_pullback import (
    MODE41_DISPLAY_NAME,
    dedupe_mode41_hits,
    mode41_kw_from_scan_config,
    mode41_signal_metrics,
    score_mode41_weekly_ma_pullback,
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
    ap = argparse.ArgumentParser(description=f"{MODE41_DISPLAY_NAME} 区间扫描")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--code", default="", help="仅扫单股")
    ap.add_argument(
        "--support-ma",
        type=int,
        default=0,
        help="仅回踩指定周均线（如 20 表示周 MA20）；0 为自动选最深关键位",
    )
    ap.add_argument("--high-tech", action="store_true", help="仅在高科技板块股票池内扫描")
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

    kw = mode41_kw_from_scan_config(ScanConfig(min_score=args.min_score))
    if args.support_ma > 0:
        kw["support_ma_only"] = int(args.support_ma)

    stock_list_path = HIGH_TECH_STOCK_LIST_CSV if args.high_tech else STOCK_LIST
    name_map = load_stock_list_csv(stock_list_path) if os.path.exists(stock_list_path) else {}
    if args.high_tech:
        ht = high_tech_code_set()
        if not ht:
            print("高科技池为空，请先运行: python3 scripts/build_high_tech_universe.py")
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
        if not rows or len(rows) < 320:
            continue
        idx_map = {r.date[:10]: i for i, r in enumerate(rows)}

        for d in trade_days:
            idx = idx_map.get(d)
            if idx is None:
                continue
            score = score_mode41_weekly_ma_pullback(rows, idx, code, name, **kw)
            if score < args.min_score:
                continue
            m = mode41_signal_metrics(rows, idx, code, name, **kw)
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

    hits = dedupe_mode41_hits(hits)

    out_dir = os.path.join(GPT_DATA_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    suffix = ""
    if args.support_ma > 0:
        suffix += f"_ma{args.support_ma}"
    if args.high_tech:
        suffix += "_high_tech"
    out_path = args.out.strip() or os.path.join(
        out_dir,
        f"mode41{suffix}_{start_ymd.replace('-', '_')}_{end_ymd.replace('-', '_')}.csv",
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
        "vol_vs_min_pct",
        "week_chg_pct",
        "exec_buy_date",
        "exec_buy_open",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for h in hits:
            w.writerow(h)

    print(f"{MODE41_DISPLAY_NAME}: {len(hits)} 条 → {out_path}")
    for h in hits[:30]:
        print(
            f"  {h['date']} {h['code']} {h['name']} score={h['score']} "
            f"周MA{h['support_ma']} dist={h['low_dist_ma_pct']}% vol+{h['vol_vs_min_pct']}%"
        )
    if len(hits) > 30:
        print(f"  ... 共 {len(hits)} 条")


if __name__ == "__main__":
    main()
