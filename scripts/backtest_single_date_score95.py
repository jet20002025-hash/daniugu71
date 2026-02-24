#!/usr/bin/env python3
"""指定信号日、最低分、不限市值，筛选个股并对每只做单笔回测（次日开盘买、10%止损、破MA10卖）。"""
import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from app.eastmoney import stock_items_from_list_csv, read_cached_kline_by_code
from app.paths import GPT_DATA_DIR

from scripts.backtest_mode3_2023_2024 import _find_exit, _moving_mean
from scripts.score_mode3_date import _load_market_caps, get_results_for_date


def _next_trading_day(date_str: str) -> Optional[str]:
    d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    for _ in range(10):
        d += timedelta(days=1)
        if d.weekday() < 5:
            return d.strftime("%Y-%m-%d")
    return None


def _load_rows(cache_dir: str, code: str):
    return read_cached_kline_by_code(cache_dir, code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="指定日 95 分以上个股不限市值回测（次日开盘买、10%止损、破MA10卖）"
    )
    parser.add_argument("--date", default="2026-02-13", help="信号日期 YYYY-MM-DD")
    parser.add_argument("--min-score", type=int, default=95, help="最低分数")
    parser.add_argument("--cache-dir", default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"))
    parser.add_argument("--stock-list", default=os.path.join(GPT_DATA_DIR, "stock_list.csv"))
    parser.add_argument("--market-cap", default=os.path.join(GPT_DATA_DIR, "market_cap.csv"))
    parser.add_argument(
        "--output",
        default="data/results/backtest_20260213_score95_no_cap.xlsx",
        help="输出 Excel",
    )
    args = parser.parse_args()

    signal_date = args.date
    buy_date = _next_trading_day(signal_date)
    if not buy_date:
        raise SystemExit("无法计算次日交易日")

    stock_list = stock_items_from_list_csv(args.stock_list)
    if not stock_list:
        raise SystemExit("股票列表为空")
    market_caps = _load_market_caps(args.market_cap)
    results = get_results_for_date(
        signal_date,
        stock_list,
        market_caps,
        args.cache_dir,
        min_score=args.min_score,
        max_cap_yi=99999.0,
        no_cap_filter=True,
    )

    rows: List[dict] = []
    for r in results:
        if r["score"] < args.min_score:
            continue
        code = (r["code"] or "").strip().zfill(6)
        kline = _load_rows(args.cache_dir, code)
        if not kline or len(kline) < 80:
            continue
        dates = [x.date for x in kline]
        # 优先次日开盘买；若 K 线无次日则用信号日收盘买
        if buy_date in dates:
            buy_idx = dates.index(buy_date)
            entry_price = kline[buy_idx].open
        elif signal_date in dates:
            buy_idx = dates.index(signal_date)
            entry_price = kline[buy_idx].close
            buy_date = signal_date
        else:
            continue
        if entry_price <= 0:
            continue
        close = np.array([x.close for x in kline], dtype=float)
        ma = _moving_mean(close, 10)
        stop_price = entry_price * 0.9
        exit_idx, exit_price, reason = _find_exit(
            kline, buy_idx, stop_price, ma, end_date=None, ma_period=10
        )
        exit_date = kline[exit_idx].date
        ret_pct = (exit_price - entry_price) / entry_price * 100
        d1 = datetime.strptime(buy_date, "%Y-%m-%d").date()
        d2 = datetime.strptime(exit_date[:10], "%Y-%m-%d").date()
        hold_days = (d2 - d1).days
        reason_cn = {"stop_loss_10pct": "10%止损", "ma10_break": "破10日均线", "end": "持仓结束"}.get(reason, reason)
        rows.append({
            "信号日": signal_date,
            "代码": code,
            "名称": r.get("name", code),
            "得分": r["score"],
            "买入日": buy_date,
            "买入价": round(entry_price, 4),
            "卖出日": exit_date[:10],
            "卖出价": round(exit_price, 4),
            "卖出原因": reason_cn,
            "收益率%": round(ret_pct, 2),
            "持仓天数": hold_days,
        })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_excel(args.output, index=False)
    print(f"信号日 {signal_date} 得分>={args.min_score} 不限市值: {len(rows)} 只")
    print(f"已保存: {args.output}")
    if rows:
        win = sum(1 for x in rows if x["收益率%"] > 0)
        print(f"盈利笔数: {win}/{len(rows)}, 胜率: {win/len(rows)*100:.1f}%")


if __name__ == "__main__":
    main()
