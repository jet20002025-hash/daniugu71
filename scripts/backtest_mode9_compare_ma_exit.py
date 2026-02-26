#!/usr/bin/env python3
"""对比 mode9 三年回测：5% 止损固定，破 MA5 / MA10 / MA20 三种卖点的收益与笔数。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from app.paths import GPT_DATA_DIR
from scripts.backtest_mode3_2023_2024 import run_backtest

CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
RES_DIR = os.path.join(ROOT, "data", "results")
INITIAL = 100_000.0
YEAR_CONFIG = [
    (2023, "2023-01-01", "2023-12-31"),
    (2024, "2024-01-01", "2024-12-31"),
    (2025, "2025-01-01", "2025-12-31"),
]


def main():
    results = []
    for ma_exit in (5, 10, 20):
        cash = INITIAL
        total_trades = 0
        for year, start, end in YEAR_CONFIG:
            path = os.path.join(RES_DIR, f"mode9_{year}_picks.csv")
            if not os.path.exists(path):
                print(f"缺少 {path}，请先运行 backtest_mode9_3year.py --refresh")
                return
            df = pd.read_csv(path)
            if "buy_point_score" not in df.columns:
                df["buy_point_score"] = 0
            trades, final_cash = run_backtest(
                df, CACHE_DIR, start, end,
                initial_cash=cash, stop_loss=0.05, ma_exit=ma_exit, use_stop_loss=True,
            )
            cash = final_cash
            total_trades += len(trades)
        ret = (cash / INITIAL - 1) * 100
        results.append({"卖点": f"破MA{ma_exit}", "三年总收益%": round(ret, 2), "交易笔数": total_trades, "期末": round(cash, 2)})

    print("止损固定 5%，卖点对比（同一批 mode9 选股）：")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
