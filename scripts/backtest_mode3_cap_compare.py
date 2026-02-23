#!/usr/bin/env python3
"""
按不同市值上限（150亿、300亿、1000亿）回测 2023/2024/2025，结果汇总到一个 Excel。
"""
import os
from typing import Dict, List

import pandas as pd

from app.paths import GPT_DATA_DIR

from scripts.backtest_mode3_2023_2024 import (
    _compute_stats,
    _filter_picks_by_cap,
    _load_market_caps,
    run_backtest,
)

CAPS = [150, 300, 1000]
YEARS = [
    (2023, "data/results/mode3_2023_picks.csv", "2023-01-01", "2023-12-31"),
    (2024, "data/results/mode3_2024_picks.csv", "2024-01-01", "2024-12-31"),
    (2025, "data/results/mode3_2025_picks.csv", "2025-01-01", "2025-12-31"),
]
CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
INITIAL_CASH = 100000.0
MARKET_CAP_PATH = os.path.join(GPT_DATA_DIR, "market_cap.csv")

# 指标 -> 汇总表列名
STAT_COL = {
    "期初资金(元)": "期初资金",
    "期末资金(元)": "期末资金",
    "总收益率(%)": "总收益率(%)",
    "交易次数": "交易次数",
    "盈利次数": "盈利次数",
    "亏损次数": "亏损次数",
    "胜率(%)": "胜率(%)",
    "平均单笔收益率(%)": "平均单笔收益率(%)",
    "平均持仓天数": "平均持仓天数",
    "最大单笔盈利(%)": "最大单笔盈利(%)",
    "最大单笔亏损(%)": "最大单笔亏损(%)",
    "最大回撤(%)": "最大回撤(%)",
}


def main() -> None:
    output_path = "data/results/mode3_backtest_市值150_300_1000亿_汇总.xlsx"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    market_caps = _load_market_caps(MARKET_CAP_PATH)
    if not market_caps:
        print("警告：未找到市值文件，将不进行市值过滤。")

    summary_rows: List[Dict] = []
    cap_sheets: Dict[str, pd.DataFrame] = {}  # 150亿 -> 指标 x 年份 表

    for cap in CAPS:
        cap_key = f"{cap}亿"
        year_data: Dict[int, List[Dict]] = {y[0]: [] for y in YEARS}

        for year, picks_path, start, end in YEARS:
            if not os.path.exists(picks_path):
                print(f"跳过 {year}：未找到 {picks_path}")
                continue
            picks = pd.read_csv(picks_path)
            if "score" not in picks.columns:
                picks["score"] = 100
            if cap > 0 and market_caps:
                picks = _filter_picks_by_cap(picks, market_caps, float(cap))
            trades, final_cash = run_backtest(
                picks,
                CACHE_DIR,
                start,
                end,
                initial_cash=INITIAL_CASH,
                stop_loss=0.10,
                ma_exit=10,
            )
            stats = _compute_stats(trades, INITIAL_CASH, final_cash, year)
            year_data[year] = stats

            # 汇总行
            row = {"市值上限(亿)": cap, "年份": year}
            for s in stats:
                idx = s["指标"]
                if idx in STAT_COL:
                    row[STAT_COL[idx]] = s["数值"]
            summary_rows.append(row)

            print(f"市值≤{cap}亿 {year}: 交易 {len(trades)} 笔, 期末 {final_cash:.2f}")

        # 按市值：指标为行、2023/2024/2025 为列
        df_cap = pd.DataFrame(year_data[2023])[["指标", "数值"]].rename(columns={"数值": 2023})
        for y in [2024, 2025]:
            if year_data[y]:
                df_y = pd.DataFrame(year_data[y])[["指标", "数值"]].rename(columns={"数值": y})
                df_cap = df_cap.merge(df_y, on="指标", how="outer")
        cap_sheets[cap_key] = df_cap

    # 写入 Excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="汇总")
        for cap_key, df in cap_sheets.items():
            df.to_excel(writer, index=False, sheet_name=cap_key)

    print("已写入:", output_path)


if __name__ == "__main__":
    main()
