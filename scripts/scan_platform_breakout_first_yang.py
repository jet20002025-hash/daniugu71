#!/usr/bin/env python3
"""
mode平台突破首阳

约 3 个月（45～95 交易日）吸筹/震仓/整理后，信号日为突破平台的第一根放量大阳线（买点）：

- **阶段低点 L**：在信号日前 45～95 日内寻最低低点，自 L 收盘涨幅 20%～55%
- **末段整理**：信号前 consolid_days 日振幅/均价 <= consolid_amp_max（默认 20%）
- **突破**：当日最高价 > 近 breakout_lookback 日最高（默认 60）
- **大阳线**：收阳；涨幅 >= big_pct_min；实体/振幅 >= body_ratio_min
- **放量**：量 >= vol_mult × max(昨量, vol_ma 日均量)
- **首阳**：前 big_yang_gap 日无同标准大阳线

参考样本：京源环保 688096（2026-05-08）、斯达半导 603290（2026-05-13）

用法:
  python3 scripts/scan_platform_breakout_first_yang.py --date 2026-05-15 --skip-st
  python3 scripts/scan_platform_breakout_first_yang.py --date 2026-05-08
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
from app.scanner import KlineRow, _is_st, _match_mode_platform_breakout_first_yang
from app import tencent

CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
STOCK_LIST_CSV = os.path.join(GPT_DATA_DIR, "stock_list.csv")
KLINE_COUNT = 320


def _find_date_index(rows: List[KlineRow], ymd: str) -> Optional[int]:
    ymd = str(ymd).strip()[:10]
    for i, r in enumerate(rows):
        if str(r.date)[:10] == ymd:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="mode平台突破首阳")
    ap.add_argument("--date", default="", help="信号日，默认各股缓存最后一根")
    ap.add_argument("--phase-days-min", type=int, default=45)
    ap.add_argument("--phase-days-max", type=int, default=95)
    ap.add_argument("--rise-min", type=float, default=0.20, help="自阶段低点最低涨幅(比例)")
    ap.add_argument("--rise-max", type=float, default=0.55, help="自阶段低点最高涨幅(比例)")
    ap.add_argument("--consolid-days", type=int, default=20)
    ap.add_argument("--consolid-amp-max", type=float, default=0.20)
    ap.add_argument("--breakout-lookback", type=int, default=60)
    ap.add_argument("--big-pct-min", type=float, default=7.0)
    ap.add_argument("--body-ratio-min", type=float, default=0.55)
    ap.add_argument("--vol-mult", type=float, default=1.25)
    ap.add_argument("--vol-ma", type=int, default=20)
    ap.add_argument("--big-yang-gap", type=int, default=15)
    ap.add_argument("--skip-st", action="store_true")
    ap.add_argument("--allow-refresh", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    prefer_local = not args.allow_refresh
    target = str(args.date).strip()[:10] if args.date else ""

    kw = dict(
        phase_days_min=int(args.phase_days_min),
        phase_days_max=int(args.phase_days_max),
        rise_from_low_min=float(args.rise_min),
        rise_from_low_max=float(args.rise_max),
        consolid_days=int(args.consolid_days),
        consolid_amp_max=float(args.consolid_amp_max),
        breakout_lookback=int(args.breakout_lookback),
        big_pct_min=float(args.big_pct_min),
        body_ratio_min=float(args.body_ratio_min),
        vol_mult=float(args.vol_mult),
        vol_ma=int(args.vol_ma),
        big_yang_gap=int(args.big_yang_gap),
    )

    name_map = load_stock_list_csv(STOCK_LIST_CSV) if os.path.exists(STOCK_LIST_CSV) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if not stock_list:
        print("股票列表为空:", CACHE_DIR)
        sys.exit(1)

    hits: List[Tuple[str, str, Dict[str, float], str, str]] = []

    min_len = max(120, int(args.phase_days_max) + int(args.breakout_lookback) + 5)

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

        m = _match_mode_platform_breakout_first_yang(
            rows, idx, item.code, item.name or "", **kw
        )
        if m is None:
            continue
        i_low = int(m["low_date_idx"])
        low_date = str(rows[i_low].date)[:10]
        code = item.code.zfill(6) if len(item.code) < 6 else item.code
        hits.append((code, (item.name or "").strip(), m, sig_date, low_date))

    hits.sort(
        key=lambda x: (
            -x[2]["breakout_pct"],
            -x[2]["vol_ratio"],
            -x[2]["pct_chg"],
            x[0],
        )
    )

    out_path = args.out.strip() or os.path.join(
        GPT_DATA_DIR, "results", "platform_breakout_first_yang.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "signal_date",
                "code",
                "name",
                "low_date",
                "phase_days",
                "close",
                "pct_chg",
                "vol_ratio",
                "rise_from_low_pct",
                "consolid_amp_pct",
                "prior_high",
                "breakout_pct",
                "body_ratio",
                "wash_big_yang_cnt",
            ]
        )
        for code, name, m, sig_date, low_date in hits:
            w.writerow(
                [
                    sig_date,
                    code,
                    name,
                    low_date,
                    int(m["phase_days"]),
                    f"{m['close']:.4f}",
                    f"{m['pct_chg']:.2f}",
                    f"{m['vol_ratio']:.2f}",
                    f"{m['rise_from_low_pct']:.2f}",
                    f"{m['consolid_amp_pct']:.2f}",
                    f"{m['prior_high']:.4f}",
                    f"{m['breakout_pct']:.2f}",
                    f"{m['body_ratio']:.3f}",
                    int(m["wash_big_yang_cnt"]),
                ]
            )

    ref = target or "各股缓存最新交易日"
    print(f"mode平台突破首阳  信号日: {ref}")
    print(
        f"  震仓 {args.phase_days_min}～{args.phase_days_max} 日  "
        f"自低涨幅 {args.rise_min:.0%}～{args.rise_max:.0%}  "
        f"前{args.consolid_days}日振幅≤{args.consolid_amp_max:.0%}  "
        f"突破{args.breakout_lookback}日高  大阳≥{args.big_pct_min}%  "
        f"量≥{args.vol_mult}×"
    )
    print(f"命中 {len(hits)} 只\n")
    print("代码      名称            低日期      震仓  突破%  量比  涨幅%  自低%")
    print("-" * 72)
    for code, name, m, sig_date, low_date in hits[:50]:
        print(
            f"{code}  {(name or '')[:10]:<10}  {low_date}  "
            f"{int(m['phase_days']):3d}  {m['breakout_pct']:5.1f}  "
            f"{m['vol_ratio']:4.1f}  {m['pct_chg']:5.1f}  {m['rise_from_low_pct']:5.1f}"
        )
    if len(hits) > 50:
        print(f"... 共 {len(hits)} 只，见 CSV")
    print(f"\n已写入: {out_path}")


if __name__ == "__main__":
    main()
