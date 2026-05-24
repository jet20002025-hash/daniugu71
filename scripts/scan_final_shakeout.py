#!/usr/bin/env python3
"""
mode最后震仓

起量→箱体→最后洗盘→反包/突破买点。
参考样本：金利华电 300069（2026-03-24 反包、2026-03-25 突破）。

用法:
  python3 scripts/scan_final_shakeout.py --date 2026-03-24 --skip-st
  python3 scripts/scan_final_shakeout.py --start 2026-04-01 --end 2026-04-30 --skip-st
  python3 scripts/scan_final_shakeout.py --code 300069 --date 2026-03-25
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
from app.scanner import KlineRow, _is_st, _match_mode_final_shakeout
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
    ap = argparse.ArgumentParser(description="mode最后震仓")
    ap.add_argument("--date", default="", help="单日信号日")
    ap.add_argument("--start", default="", help="区间起始（含），与 --end 联用扫整月")
    ap.add_argument("--end", default="", help="区间结束（含）")
    ap.add_argument("--code", default="", help="仅测单股")
    ap.add_argument("--phase-days-min", type=int, default=30)
    ap.add_argument("--phase-days-max", type=int, default=90)
    ap.add_argument("--anchor-vol-mult", type=float, default=1.5)
    ap.add_argument("--min-rally-pct", type=float, default=0.10)
    ap.add_argument("--consolid-days", type=int, default=20)
    ap.add_argument("--consolid-amp-max", type=float, default=0.15)
    ap.add_argument("--peak-lookback", type=int, default=15)
    ap.add_argument("--shakeout-days-min", type=int, default=3)
    ap.add_argument("--shakeout-days-max", type=int, default=7)
    ap.add_argument("--shakeout-drop-min", type=float, default=0.10)
    ap.add_argument("--shakeout-drop-max", type=float, default=0.22)
    ap.add_argument("--phase-low-lookback", type=int, default=90)
    ap.add_argument("--phase-low-break-min", type=float, default=0.95)
    ap.add_argument("--shakeout-vol-min", type=float, default=0.6)
    ap.add_argument("--shakeout-vol-max", type=float, default=2.5)
    ap.add_argument("--ma60-slope-days", type=int, default=20)
    ap.add_argument("--reversal-pct-min", type=float, default=8.0)
    ap.add_argument("--reversal-vol-min", type=float, default=1.5)
    ap.add_argument("--reversal-low-tol", type=float, default=0.05)
    ap.add_argument("--breakout-pct-min", type=float, default=15.0)
    ap.add_argument("--breakout-pct-min-main", type=float, default=9.0)
    ap.add_argument("--breakout-vol-min", type=float, default=3.0)
    ap.add_argument("--body-ratio-min", type=float, default=0.55)
    ap.add_argument("--skip-st", action="store_true")
    ap.add_argument("--allow-refresh", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    prefer_local = not args.allow_refresh
    target = str(args.date).strip()[:10] if args.date else ""
    start_ymd = str(args.start).strip()[:10] if args.start else ""
    end_ymd = str(args.end).strip()[:10] if args.end else ""
    only_code = str(args.code).strip()

    kw = dict(
        phase_days_min=int(args.phase_days_min),
        phase_days_max=int(args.phase_days_max),
        anchor_vol_mult=float(args.anchor_vol_mult),
        min_rally_pct=float(args.min_rally_pct),
        consolid_days=int(args.consolid_days),
        consolid_amp_max=float(args.consolid_amp_max),
        peak_lookback=int(args.peak_lookback),
        shakeout_days_min=int(args.shakeout_days_min),
        shakeout_days_max=int(args.shakeout_days_max),
        shakeout_drop_min=float(args.shakeout_drop_min),
        shakeout_drop_max=float(args.shakeout_drop_max),
        phase_low_lookback=int(args.phase_low_lookback),
        phase_low_break_min=float(args.phase_low_break_min),
        shakeout_vol_min=float(args.shakeout_vol_min),
        shakeout_vol_max=float(args.shakeout_vol_max),
        ma60_slope_days=int(args.ma60_slope_days),
        reversal_pct_min=float(args.reversal_pct_min),
        reversal_vol_min=float(args.reversal_vol_min),
        reversal_low_tol=float(args.reversal_low_tol),
        breakout_pct_min=float(args.breakout_pct_min),
        breakout_pct_min_main=float(args.breakout_pct_min_main),
        breakout_vol_min=float(args.breakout_vol_min),
        body_ratio_min=float(args.body_ratio_min),
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

    hits: List[Tuple[str, str, Dict[str, float], str, str, str, str]] = []
    min_len = max(120, int(args.phase_days_max) + int(args.phase_low_lookback) + 10)

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

        if start_ymd and end_ymd:
            check_indices = [
                i
                for i, r in enumerate(rows)
                if start_ymd <= str(r.date)[:10] <= end_ymd
            ]
        elif target:
            idx = _find_date_index(rows, target)
            check_indices = [idx] if idx is not None else []
        else:
            check_indices = [len(rows) - 1]

        for idx in check_indices:
            if idx is None or idx < min_len:
                continue
            sig_date = str(rows[idx].date)[:10]
            m = _match_mode_final_shakeout(rows, idx, item.code, item.name or "", **kw)
            if m is None:
                continue
            i_a = int(m["anchor_date_idx"])
            i_t = int(m["trough_date_idx"])
            i_p = int(m["peak_date_idx"])
            anchor_date = str(rows[i_a].date)[:10]
            trough_date = str(rows[i_t].date)[:10]
            peak_date = str(rows[i_p].date)[:10]
            sig_type = "突破" if int(m["signal_type"]) == 1 else "反包"
            code = item.code.zfill(6) if len(item.code) < 6 else item.code
            hits.append((code, (item.name or "").strip(), m, sig_date, anchor_date, trough_date, peak_date, sig_type))

    hits.sort(
        key=lambda x: (
            x[3],
            -x[2]["signal_type"],
            -x[2]["pct_chg"],
            -x[2]["vol_ratio"],
            x[0],
        )
    )

    default_name = "final_shakeout.csv"
    if start_ymd and end_ymd:
        default_name = f"final_shakeout_{start_ymd[:7].replace('-', '_')}.csv"
    out_path = args.out.strip() or os.path.join(GPT_DATA_DIR, "results", default_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "signal_date",
                "code",
                "name",
                "signal_type",
                "anchor_date",
                "peak_date",
                "trough_date",
                "anchor_support",
                "peak_high",
                "trough_low",
                "shakeout_drop_pct",
                "rally_from_anchor_pct",
                "phase_days",
                "shakeout_days",
                "vol_ratio",
                "pct_chg",
                "body_ratio",
                "close",
            ]
        )
        for code, name, m, sig_date, anchor_date, trough_date, peak_date, sig_type in hits:
            w.writerow(
                [
                    sig_date,
                    code,
                    name,
                    sig_type,
                    anchor_date,
                    peak_date,
                    trough_date,
                    f"{m['anchor_support']:.2f}",
                    f"{m['peak_high']:.2f}",
                    f"{m['trough_low']:.2f}",
                    f"{m['shakeout_drop_pct']:.1f}",
                    f"{m['rally_from_anchor_pct']:.1f}",
                    int(m["phase_days"]),
                    int(m["shakeout_days"]),
                    f"{m['vol_ratio']:.2f}",
                    f"{m['pct_chg']:.2f}",
                    f"{m['body_ratio']:.2f}",
                    f"{m['close']:.2f}",
                ]
            )

    print(f"命中 {len(hits)} 只，已写入 {out_path}")
    for code, name, m, sig_date, anchor_date, trough_date, peak_date, sig_type in hits[:40]:
        print(
            f"  {sig_date} {code} {name} [{sig_type}] 锚点{anchor_date} 峰{peak_date} 低{trough_date} "
            f"洗盘-{m['shakeout_drop_pct']:.1f}% vr={m['vol_ratio']:.2f} pct={m['pct_chg']:.2f}%"
        )


if __name__ == "__main__":
    main()
