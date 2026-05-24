#!/usr/bin/env python3
"""
mode底部支撑

底部起量大阳（锚点）→ 拉升 → 回调至起量位获支撑 → 再拉升 → 再回调至支撑（抄底买点）。
参考样本：斯达半导 603290（2025-08-07 起量，后续多次回踩支撑）。

用法:
  python3 scripts/scan_bottom_support.py --date 2025-11-24 --skip-st
  python3 scripts/scan_bottom_support.py --code 603290 --date 2025-11-24
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.eastmoney import list_cached_stocks_flat, load_stock_list_csv, read_cached_kline_by_code
from app.paths import GPT_DATA_DIR
from app.scanner import KlineRow, _is_st, _match_mode_bottom_support
from app import tencent

CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
STOCK_LIST_CSV = os.path.join(GPT_DATA_DIR, "stock_list.csv")
KLINE_COUNT = 400


def _find_date_index(rows: List[KlineRow], ymd: str) -> Optional[int]:
    ymd = str(ymd).strip()[:10]
    for i, r in enumerate(rows):
        if str(r.date)[:10] == ymd:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="mode底部支撑")
    ap.add_argument("--date", default="", help="信号日，默认各股缓存最后一根")
    ap.add_argument("--code", default="", help="仅测单股")
    ap.add_argument("--anchor-days-min", type=int, default=30)
    ap.add_argument("--anchor-days-max", type=int, default=200)
    ap.add_argument("--low-lookback", type=int, default=60)
    ap.add_argument("--bottom-pos-max", type=float, default=0.50)
    ap.add_argument("--anchor-vol-mult", type=float, default=2.0)
    ap.add_argument("--anchor-vol-ma", type=int, default=20)
    ap.add_argument("--big-pct-min", type=float, default=5.0)
    ap.add_argument("--body-ratio-min", type=float, default=0.55)
    ap.add_argument("--min-rally-pct", type=float, default=0.15)
    ap.add_argument("--support-near-max", type=float, default=0.15)
    ap.add_argument("--support-break-min", type=float, default=0.97)
    ap.add_argument("--test-tol", type=float, default=0.15)
    ap.add_argument("--min-support-tests", type=int, default=1)
    ap.add_argument("--bounce-days", type=int, default=5)
    ap.add_argument("--weekly-vol-mult", type=float, default=1.5)
    ap.add_argument("--skip-st", action="store_true")
    ap.add_argument("--allow-refresh", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    prefer_local = not args.allow_refresh
    target = str(args.date).strip()[:10] if args.date else ""
    only_code = str(args.code).strip()

    kw = dict(
        anchor_days_min=int(args.anchor_days_min),
        anchor_days_max=int(args.anchor_days_max),
        low_lookback=int(args.low_lookback),
        bottom_pos_max=float(args.bottom_pos_max),
        anchor_vol_mult=float(args.anchor_vol_mult),
        anchor_vol_ma=int(args.anchor_vol_ma),
        big_pct_min=float(args.big_pct_min),
        body_ratio_min=float(args.body_ratio_min),
        min_rally_pct=float(args.min_rally_pct),
        support_near_max=float(args.support_near_max),
        support_break_min=float(args.support_break_min),
        test_tol=float(args.test_tol),
        min_support_tests=int(args.min_support_tests),
        bounce_days=int(args.bounce_days),
        weekly_vol_mult=float(args.weekly_vol_mult),
    )

    name_map = load_stock_list_csv(STOCK_LIST_CSV) if os.path.exists(STOCK_LIST_CSV) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if only_code:
        oc = only_code.zfill(6)
        stock_list = [s for s in stock_list if s.code.zfill(6) == oc]
        if not stock_list:
            stock_list = [type("X", (), {"code": oc, "name": name_map.get(oc, oc)})()]
    if not stock_list:
        print("股票列表为空:", CACHE_DIR)
        sys.exit(1)

    hits: List[Tuple[str, str, Dict[str, float], str, str]] = []
    min_len = max(220, int(args.anchor_days_max) + int(args.low_lookback) + 5)

    for item in stock_list:
        if args.skip_st and _is_st(item.name or ""):
            continue
        if prefer_local:
            rows = read_cached_kline_by_code(CACHE_DIR, item.code)
        else:
            rows = tencent.get_kline_cached(
                item.code,
                cache_dir=CACHE_DIR,
                count=KLINE_COUNT,
                min_len=min_len,
                prefer_local=False,
            )
        if not rows or len(rows) < min_len:
            continue
        if target:
            idx = _find_date_index(rows, target)
            if idx is None:
                continue
            sig_date = target
        else:
            idx = len(rows) - 1
            sig_date = str(rows[idx].date)[:10]

        m = _match_mode_bottom_support(rows, idx, item.code, item.name or "", **kw)
        if m is None:
            continue
        i_a = int(m["anchor_date_idx"])
        anchor_date = str(rows[i_a].date)[:10]
        code = item.code.zfill(6) if len(item.code) < 6 else item.code
        hits.append((code, (item.name or "").strip(), m, sig_date, anchor_date))

    hits.sort(
        key=lambda x: (
            x[2]["low_dist_pct"],
            -x[2]["support_tests"],
            -x[2]["max_rally_pct"],
            x[0],
        )
    )

    out_path = args.out.strip() or os.path.join(GPT_DATA_DIR, "results", "bottom_support.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "signal_date",
                "code",
                "name",
                "anchor_date",
                "support",
                "phase_days",
                "support_tests",
                "max_rally_pct",
                "low_dist_pct",
                "close_dist_pct",
                "close",
                "pct_chg",
                "anchor_vol_ratio",
            ]
        )
        for code, name, m, sig_date, anchor_date in hits:
            w.writerow(
                [
                    sig_date,
                    code,
                    name,
                    anchor_date,
                    f"{m['support']:.2f}",
                    int(m["phase_days"]),
                    int(m["support_tests"]),
                    f"{m['max_rally_pct']:.1f}",
                    f"{m['low_dist_pct']:.1f}",
                    f"{m['close_dist_pct']:.1f}",
                    f"{m['close']:.2f}",
                    f"{m['pct_chg']:.2f}",
                    f"{m['anchor_vol_ratio']:.2f}",
                ]
            )

    print(f"命中 {len(hits)} 只，已写入 {out_path}")
    for code, name, m, sig_date, anchor_date in hits[:40]:
        print(
            f"  {sig_date} {code} {name} 锚点{anchor_date} 支撑{m['support']:.2f} "
            f"距+{m['low_dist_pct']:.1f}% 验证{int(m['support_tests'])}次 "
            f"最大拉升+{m['max_rally_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
