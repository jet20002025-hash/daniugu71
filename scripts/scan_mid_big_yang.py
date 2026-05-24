#!/usr/bin/env python3
"""
mode中位大阳线

主力介入大阳（锚点）→ 吸筹/洗盘/震仓 → 突破大阳线买点（参考埃科光电 688610）：

- **锚点**：信号前 30～90 日内最早一根放量大阳线（量 >= anchor_vol_mult × 基准量）
- **震仓周期**：锚点至信号日；自锚点收盘涨幅 25%～120%（中位，非底部平台）
- **末段整理**：信号前 consolid_days 日振幅/均价 <= consolid_amp_max（默认 35%）
- **突破**：当日最高严格突破近 60 日高（breakout_min=1.0）；100 日高比 >= 1.0
- **大阳线 + 放量 + 量比/上影/前5日涨幅** 等质量过滤

与「平台突破首阳」区别：以主力大阳为锚点（非阶段最低），允许自锚点已有较大涨幅，不要求贴顶首阳。

用法:
  python3 scripts/scan_mid_big_yang.py --date 2026-04-13 --skip-st
  python3 scripts/scan_mid_big_yang.py --start 2026-05-01 --end 2026-05-31 --skip-st
  python3 scripts/scan_mid_big_yang.py --code 688610 --date 2026-04-13
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
from app.scanner import KlineRow, _is_st, _match_mode_mid_big_yang
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
    ap = argparse.ArgumentParser(description="mode中位大阳线")
    ap.add_argument("--date", default="", help="单日信号日")
    ap.add_argument("--start", default="", help="区间起始（含），与 --end 联用")
    ap.add_argument("--end", default="", help="区间结束（含）")
    ap.add_argument("--code", default="", help="仅测单股代码")
    ap.add_argument("--anchor-days-min", type=int, default=30)
    ap.add_argument("--anchor-days-max", type=int, default=200)
    ap.add_argument("--anchor-vol-mult", type=float, default=1.5)
    ap.add_argument("--rise-min", type=float, default=0.20, help="自锚点最低涨幅(比例)")
    ap.add_argument("--rise-max", type=float, default=1.20, help="自锚点最高涨幅(比例)")
    ap.add_argument("--consolid-days", type=int, default=20)
    ap.add_argument("--consolid-amp-max", type=float, default=0.35)
    ap.add_argument("--breakout-lookback", type=int, default=60)
    ap.add_argument("--breakout-min", type=float, default=1.0)
    ap.add_argument("--high100-lookback", type=int, default=100)
    ap.add_argument("--high100-min", type=float, default=1.0)
    ap.add_argument("--tight-consolid-amp-max", type=float, default=0.15)
    ap.add_argument("--tight-vol-ratio-min", type=float, default=1.8)
    ap.add_argument("--tight-rise-min", type=float, default=0.10)
    ap.add_argument("--tight-high100-min", type=float, default=0.985)
    ap.add_argument("--big-pct-min", type=float, default=7.0)
    ap.add_argument("--big-pct-min-main", type=float, default=4.5)
    ap.add_argument("--body-ratio-min", type=float, default=0.55)
    ap.add_argument("--vol-mult", type=float, default=1.25)
    ap.add_argument("--vol-ma", type=int, default=20)
    ap.add_argument("--vol-ratio-max", type=float, default=5.0)
    ap.add_argument("--upper-ratio-max", type=float, default=0.40)
    ap.add_argument("--close-break60-min", type=float, default=1.0)
    ap.add_argument("--pre-rise5-min", type=float, default=-0.05)
    ap.add_argument("--pre-rise5-max", type=float, default=0.15, help="信号前5日涨幅上限(<=)，排除连板追高")
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
        anchor_days_min=int(args.anchor_days_min),
        anchor_days_max=int(args.anchor_days_max),
        anchor_vol_mult=float(args.anchor_vol_mult),
        rise_from_anchor_min=float(args.rise_min),
        rise_from_anchor_max=float(args.rise_max),
        consolid_days=int(args.consolid_days),
        consolid_amp_max=float(args.consolid_amp_max),
        breakout_lookback=int(args.breakout_lookback),
        breakout_min=float(args.breakout_min),
        high100_lookback=int(args.high100_lookback),
        high100_min=float(args.high100_min),
        tight_consolid_amp_max=float(args.tight_consolid_amp_max),
        tight_vol_ratio_min=float(args.tight_vol_ratio_min),
        tight_rise_from_anchor_min=float(args.tight_rise_min),
        tight_high100_min=float(args.tight_high100_min),
        big_pct_min=float(args.big_pct_min),
        big_pct_min_main=float(args.big_pct_min_main),
        body_ratio_min=float(args.body_ratio_min),
        vol_mult=float(args.vol_mult),
        vol_ma=int(args.vol_ma),
        vol_ratio_max=float(args.vol_ratio_max),
        upper_ratio_max=float(args.upper_ratio_max),
        close_break60_min=float(args.close_break60_min),
        pre_rise5_min=float(args.pre_rise5_min),
        pre_rise5_max=float(args.pre_rise5_max),
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
    min_len = max(120, int(args.anchor_days_max) + int(args.high100_lookback) + 5)

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
            m = _match_mode_mid_big_yang(rows, idx, item.code, item.name or "", **kw)
            if m is None:
                continue
            i_a = int(m["anchor_date_idx"])
            anchor_date = str(rows[i_a].date)[:10]
            code = item.code.zfill(6) if len(item.code) < 6 else item.code
            hits.append((code, (item.name or "").strip(), m, sig_date, anchor_date))

    hits.sort(
        key=lambda x: (
            x[3],
            -x[2]["breakout_pct"],
            -x[2]["vol_ratio"],
            -x[2]["pct_chg"],
            x[0],
        )
    )

    default_name = "mid_big_yang.csv"
    if start_ymd and end_ymd:
        default_name = f"mid_big_yang_{start_ymd[:7].replace('-', '_')}.csv"
    out_path = args.out.strip() or os.path.join(GPT_DATA_DIR, "results", default_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "signal_date",
                "code",
                "name",
                "anchor_date",
                "phase_days",
                "close",
                "pct_chg",
                "vol_ratio",
                "rise_from_anchor_pct",
                "consolid_amp_pct",
                "breakout_pct",
                "high100_ratio",
                "close_break60",
                "upper_ratio",
                "pre_rise5_pct",
            ]
        )
        for code, name, m, sig_date, anchor_date in hits:
            w.writerow(
                [
                    sig_date,
                    code,
                    name,
                    anchor_date,
                    int(m["phase_days"]),
                    f"{m['close']:.2f}",
                    f"{m['pct_chg']:.2f}",
                    f"{m['vol_ratio']:.2f}",
                    f"{m['rise_from_anchor_pct']:.1f}",
                    f"{m['consolid_amp_pct']:.1f}",
                    f"{m['breakout_pct']:.1f}",
                    f"{m['high100_ratio']:.3f}",
                    f"{m['close_break60']:.3f}",
                    f"{m['upper_ratio']:.3f}",
                    f"{m['pre_rise5_pct']:.1f}",
                ]
            )

    print(f"命中 {len(hits)} 只，已写入 {out_path}")
    for code, name, m, sig_date, anchor_date in hits[:30]:
        print(
            f"  {sig_date} {code} {name} 锚点{anchor_date} "
            f"震仓{int(m['phase_days'])}日 自锚点+{m['rise_from_anchor_pct']:.1f}% "
            f"突破+{m['breakout_pct']:.1f}% 量比{m['vol_ratio']:.2f} 涨{m['pct_chg']:.1f}%"
        )


if __name__ == "__main__":
    main()
