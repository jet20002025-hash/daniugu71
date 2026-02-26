#!/usr/bin/env python3
"""
基于 mode9 对 2月13日、2月24日 筛选满分100的个股，按分数降序排列，输出 Excel。
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.eastmoney import list_cached_stocks_flat, load_stock_list_csv
from app.paths import GPT_DATA_DIR
from app.scanner import _moving_mean, _score_mode9
from scripts.backtest_startup_modes import _load_rows, _signals_mode3

CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
STOCK_LIST_CSV = os.path.join(GPT_DATA_DIR, "stock_list.csv")
OUTPUT_XLSX = os.path.join(ROOT, "data", "results", "score100_mode9_0213_0224.xlsx")
DATES = ["2026-02-13", "2026-02-24"]


def get_results_mode9_for_date(target_date: str, stock_list: list, cache_dir: str, min_score: int = 100):
    """指定日期用 mode9 筛选，返回 score>=min_score，按分数降序。"""
    cache_format = "code"
    results = []
    for item in stock_list:
        rows = _load_rows(cache_dir, cache_format, item.market, item.code)
        if not rows or len(rows) < 80:
            continue
        dates = [r.date[:10] if hasattr(r.date, '__getitem__') else str(r.date)[:10] for r in rows]
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
            score = _score_mode9(rows, idx, ma10, ma20, ma60, vol20)
            if score < min_score:
                continue
            code = item.code.zfill(6) if len(item.code) < 6 else item.code
            results.append({
                "信号日": target_date,
                "代码": code,
                "名称": item.name or code,
                "得分": score,
                "当天收盘价": round(float(close[idx]), 2),
            })
    results.sort(key=lambda x: (-x["得分"], x["代码"]))
    return results


def main():
    name_map = load_stock_list_csv(STOCK_LIST_CSV) if os.path.exists(STOCK_LIST_CSV) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if not stock_list:
        print("股票列表为空")
        sys.exit(1)
    os.makedirs(os.path.dirname(OUTPUT_XLSX) or ".", exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for day in DATES:
            rows = get_results_mode9_for_date(day, stock_list, CACHE_DIR, min_score=100)
            df = pd.DataFrame(rows)
            sheet_name = day.replace("-", "")[:8]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"{day} mode9 满分100: {len(df)} 只")
    print(f"已保存: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
