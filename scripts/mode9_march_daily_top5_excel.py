#!/usr/bin/env python3
"""
mode9（与网页默认「最新模型」一致）+ 71 倍标准：
自当年 3 月 1 日起至指定结束日，按交易日取每日排序后 top5，
导出 Excel；涨幅口径与 analyze_mode9_march_ge80 一致：
**买入价 = T+1（buy_date）开盘价**，**卖出价 = 本地缓存该股的最后一根收盘价**。

注意：`scan_with_mode3` 在 `start_date` 模式下最后会 `return out[:max_results]`，
故扫描时须将 `max_results` 设得足够大（本脚本默认 2_000_000），否则会丢尾部日期。
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.eastmoney import list_cached_stocks_flat, load_stock_list_csv, read_cached_kline_by_code
from app.paths import GPT_DATA_DIR as _GPT_ENV
from app.scanner import ScanConfig, mode3_sort_tuple, scan_with_mode3


def _resolve_gpt_data_dir() -> str:
    """若 GPT_DATA_DIR 指向无效路径（如本机误设服务器 /data/gpt），回退到项目 data/gpt。"""
    fallback = os.path.join(ROOT, "data", "gpt")
    for base in (_GPT_ENV, fallback):
        kdir = os.path.join(base, "kline_cache_tencent")
        if not os.path.isdir(kdir):
            continue
        try:
            if any(name.endswith(".csv") for name in os.listdir(kdir)):
                return base
        except OSError:
            continue
    return fallback


GPT_DATA_DIR = _resolve_gpt_data_dir()
if GPT_DATA_DIR.rstrip("/") != _GPT_ENV.rstrip("/"):
    print(
        f"提示: 环境变量 GPT_DATA_DIR={_GPT_ENV!r} 下无有效 K 线，已改用 {GPT_DATA_DIR!r}",
        file=sys.stderr,
    )
CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
STOCK_LIST_CSV = os.path.join(GPT_DATA_DIR, "stock_list.csv")
OUT_DIR = os.path.join(ROOT, "data", "results")


def _default_march_to_today() -> tuple[str, str]:
    today = date.today()
    start = date(today.year, 3, 1)
    if today < start:
        start = date(today.year - 1, 3, 1)
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def _year_range(y: int) -> tuple[str, str]:
    """自然年 YYYY-01-01 ~ YYYY-12-31（无 K 线的未来日期自然无信号）。"""
    return f"{y}-01-01", f"{y}-12-31"


def _find_date_index(rows, d: str) -> Optional[int]:
    ds = d[:10]
    for i, r in enumerate(rows):
        if (r.date[:10] if r.date else "") == ds:
            return i
    return None


def _forward_returns(rows, buy_date: str) -> dict:
    bi = _find_date_index(rows, buy_date)
    out = {
        "ret_to_last_pct": np.nan,
        "ret_5d_pct": np.nan,
        "ret_10d_pct": np.nan,
        "hold_bars": np.nan,
        "cache_last_date": "",
    }
    if not rows:
        return out
    out["cache_last_date"] = str(rows[-1].date or "")[:10]
    if bi is None or bi >= len(rows):
        return out
    o = float(rows[bi].open)
    if o <= 0:
        return out
    last_i = len(rows) - 1
    out["hold_bars"] = float(last_i - bi)
    last_c = float(rows[last_i].close)
    out["ret_to_last_pct"] = (last_c - o) / o * 100.0
    for n, key in ((5, "ret_5d_pct"), (10, "ret_10d_pct")):
        j = min(bi + n, last_i)
        if j > bi:
            out[key] = (float(rows[j].close) - o) / o * 100.0
    return out


def main() -> None:
    s0, e0 = _default_march_to_today()
    ap = argparse.ArgumentParser(
        description="mode9 区间扫描后按交易日取每日 topN，导出 Excel（与网页 mode9+71 倍一致）"
    )
    ap.add_argument(
        "--year",
        type=int,
        default=0,
        help="若 >0，则信号区间为该自然年全年（覆盖默认的 --start/--end，除非显式传入起止）",
    )
    ap.add_argument("--start", default="", help="信号日区间起（默认：三月起至今；与 --year 二选一）")
    ap.add_argument("--end", default="", help="信号日区间止（含）")
    ap.add_argument("--min-score", type=int, default=70, help="最低分（与网页默认一致可 70）")
    ap.add_argument("--top-n", type=int, default=5, help="每日取前 N 只")
    ap.add_argument("--limit", type=int, default=0, help="仅扫描前 N 只股票，0=全市场")
    ap.add_argument(
        "--out",
        default="",
        help="输出 xlsx，默认 data/results/mode9_march_daily_top5_to_today.xlsx",
    )
    args = ap.parse_args()

    if args.year and args.year > 0:
        start_e, end_e = _year_range(args.year)
        args.start = args.start or start_e
        args.end = args.end or end_e
    else:
        if not args.start:
            args.start = s0
        if not args.end:
            args.end = e0

    name_map = load_stock_list_csv(STOCK_LIST_CSV) if os.path.exists(STOCK_LIST_CSV) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if not stock_list:
        raise SystemExit(f"本地股票缓存为空：{CACHE_DIR}")
    if args.limit:
        stock_list = stock_list[: args.limit]

    if args.out:
        out_xlsx = args.out
    elif args.year and args.year > 0:
        out_xlsx = os.path.join(OUT_DIR, f"mode9_{args.year}_daily_top{args.top_n}.xlsx")
    else:
        out_xlsx = os.path.join(OUT_DIR, "mode9_march_daily_top5_to_today.xlsx")
    os.makedirs(os.path.dirname(out_xlsx) or ".", exist_ok=True)

    cfg = ScanConfig(
        min_score=args.min_score,
        max_results=2_000_000,
        max_market_cap=None,
    )

    def _loader(item):
        return read_cached_kline_by_code(CACHE_DIR, item.code)

    print(
        f"mode9 扫描 信号日 {args.start} ~ {args.end}，min_score>={args.min_score}，"
        f"股票数 {len(stock_list)}，扫描结束后按日取 top{args.top_n} …"
    )
    results = scan_with_mode3(
        stock_list,
        cfg,
        CACHE_DIR,
        progress_cb=None,
        kline_loader=_loader,
        start_date=args.start,
        cutoff_date=args.end,
        use_mode9=True,
        use_71x_standard=True,
    )
    print(f"扫描返回记录数: {len(results)}")

    by_date: dict[str, list] = defaultdict(list)
    for r in results:
        m = r.metrics or {}
        sig = str(m.get("signal_date") or "").strip()
        if not sig or sig < args.start or sig > args.end:
            continue
        by_date[sig].append(r)

    rows_out = []
    for d in sorted(by_date.keys()):
        lst = by_date[d]
        lst.sort(key=lambda x: mode3_sort_tuple(x, prefer_upper_shadow=False))
        for rank, r in enumerate(lst[: args.top_n], start=1):
            m = r.metrics or {}
            buy = str(m.get("buy_date") or "").strip()
            code = r.code.zfill(6) if len(r.code) < 6 else r.code
            krows = read_cached_kline_by_code(CACHE_DIR, code)
            fr = _forward_returns(krows, buy) if buy else _forward_returns(krows or [], "")
            rows_out.append(
                {
                    "signal_date": d,
                    "day_rank": rank,
                    "code": code,
                    "name": r.name or code,
                    "score": int(r.score),
                    "buy_point_score": int(m.get("buy_point_score") or 0),
                    "buy_date": buy,
                    "vol_ratio": float(m.get("vol_ratio") or 0),
                    "ret20_signal": float(m.get("ret20") or 0),
                    "ret5_signal": float(m.get("ret5") or 0),
                    "has_limit_up_6d": int(m.get("has_limit_up_6d") or 0),
                    "limitup_shrink_vol": int(m.get("limitup_shrink_vol") or 0),
                    **fr,
                }
            )

    df = pd.DataFrame(rows_out)
    meta = pd.DataFrame(
        [
            {
                "项": "模型",
                "值": "mode9 + use_71x_standard；板块热度仅涨停行业家数/TopN（无行业指数涨跌幅加分）",
            },
            {"项": "信号日区间", "值": f"{args.start} ~ {args.end}"},
            {"项": "自然年", "值": str(args.year) if args.year else "(未指定 --year)"},
            {"项": "每日条数", "值": args.top_n},
            {"项": "最低分", "值": args.min_score},
            {
                "项": "涨幅口径",
                "值": "T+1日开盘价 -> 该股缓存最后一根收盘价（%）",
            },
            {"项": "生成日", "值": date.today().strftime("%Y-%m-%d")},
            {"项": "扫描股票数", "值": len(stock_list)},
            {"项": "扫描返回行数", "值": len(results)},
        ]
    )

    daily_summary = pd.DataFrame()
    if not df.empty and df["ret_to_last_pct"].notna().any():
        daily_summary = (
            df.dropna(subset=["ret_to_last_pct"])
            .groupby("signal_date", as_index=False)
            .agg(
                n=("code", "count"),
                mean_ret_to_last=("ret_to_last_pct", "mean"),
                median_ret_to_last=("ret_to_last_pct", "median"),
            )
        )

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="每日top5清单")
        meta.to_excel(w, index=False, sheet_name="说明")
        if not daily_summary.empty:
            daily_summary.to_excel(w, index=False, sheet_name="按日汇总涨幅")

    print(f"交易日数（有信号）: {len(by_date)}，输出行数: {len(df)}")
    print(f"已保存: {out_xlsx}")


if __name__ == "__main__":
    main()
