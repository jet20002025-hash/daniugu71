#!/usr/bin/env python3
"""
基于最新 mode9、本地 K 线缓存，筛选指定日期得分>=给定阈值的个股，按分数降序输出 Excel。

默认：
- 目标日期 = 今天
- 最低得分 = 98
"""
import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.eastmoney import list_cached_stocks_flat, load_stock_list_csv
from app.paths import GPT_DATA_DIR
from app.scanner import _moving_mean, _score_mode9, _limit_rate
from scripts.backtest_startup_modes import _load_rows, _signals_mode3

CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
STOCK_LIST_CSV = os.path.join(GPT_DATA_DIR, "stock_list.csv")


def _has_limit_up_6d(rows, idx: int, code: str, name: str, lookback: int = 6) -> bool:
    """
    最近 lookback 个交易日内是否有涨停（仅判定有/无，不要求缩量）。
    涨停阈值与项目内一致：按 ST / 创业板 / 科创板 / 主板 的涨停幅度判定。
    """
    if idx < 1:
        return False
    rate = _limit_rate(code, name)
    limit_up = (rate * 100) - 0.5
    start = max(1, idx - lookback)
    for i in range(start, idx):
        if rows[i].pct_chg >= limit_up:
            return True
    return False


def get_results_mode9_for_date(target_date: str, stock_list: list, cache_dir: str, min_score: int):
    """指定日期用 mode9 筛选，返回 score>=min_score，按分数降序。"""
    cache_format = "code"
    results = []
    for item in stock_list:
        rows = _load_rows(cache_dir, cache_format, item.market, item.code)
        if not rows or len(rows) < 80:
            continue
        dates = [r.date[:10] if hasattr(r.date, "__getitem__") else str(r.date)[:10] for r in rows]
        close = np.array([r.close for r in rows], dtype=float)
        open_ = np.array([r.open for r in rows], dtype=float)
        high = np.array([r.high for r in rows], dtype=float)
        low = np.array([r.low for r in rows], dtype=float)
        volume = np.array([r.volume for r in rows], dtype=float)
        ma10 = _moving_mean(close, 10)
        ma20 = _moving_mean(close, 20)
        ma60 = _moving_mean(close, 60)
        vol20 = _moving_mean(volume, 20)
        ret20 = [None] * len(rows)
        for i in range(20, len(rows)):
            base = close[i - 20]
            ret20[i] = (close[i] - base) / base * 100 if base else None
        signals = _signals_mode3(
            rows, dates, close, open_, high, low, volume,
            ma10, ma20, ma60, vol20, ret20,
            target_date, target_date,
        )
        for idx in signals:
            if dates[idx] != target_date:
                continue
            score = _score_mode9(rows, idx, ma10, ma20, ma60, vol20, item.code, item.name)
            if score < min_score:
                continue
            code = item.code.zfill(6) if len(item.code) < 6 else item.code
            has_limit_up_6d = _has_limit_up_6d(rows, idx, item.code, item.name, lookback=6)
            results.append({
                "信号日": target_date,
                "代码": code,
                "名称": item.name or code,
                "得分": score,
                "当天收盘价": round(float(close[idx]), 2),
                "最早出现日期": target_date,
                "最近6个交易日有涨停": "是" if has_limit_up_6d else "否",
            })
    results.sort(key=lambda x: (-x["得分"], x["代码"]))
    return results


def main():
    parser = argparse.ArgumentParser(description="基于 mode9、本地缓存筛选指定日期高分个股")
    today_str = dt.date.today().strftime("%Y-%m-%d")
    parser.add_argument(
        "--date",
        dest="target_date",
        default=today_str,
        help="目标交易日期，格式 YYYY-MM-DD，默认今天",
    )
    parser.add_argument(
        "--min-score",
        dest="min_score",
        type=int,
        default=98,
        help="最低得分阈值，默认 98",
    )
    args = parser.parse_args()
    target_date = args.target_date
    min_score = args.min_score

    name_map = load_stock_list_csv(STOCK_LIST_CSV) if os.path.exists(STOCK_LIST_CSV) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if not stock_list:
        print("股票列表为空，请确认本地缓存目录:", CACHE_DIR)
        sys.exit(1)
    output_xlsx = os.path.join(
        ROOT,
        "data",
        "results",
        f"mode9_{target_date.replace('-', '')}_score{min_score}_plus.xlsx",
    )
    os.makedirs(os.path.dirname(output_xlsx) or ".", exist_ok=True)
    rows = get_results_mode9_for_date(target_date, stock_list, CACHE_DIR, min_score)
    df = pd.DataFrame(rows)
    df.to_excel(output_xlsx, index=False)
    print(f"{target_date} mode9 得分>={min_score}: {len(df)} 只")
    print(f"已保存: {output_xlsx}")
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
