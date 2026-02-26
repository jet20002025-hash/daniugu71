#!/usr/bin/env python3
"""
mode9 三年回测（仅盈利3%止盈）：每日买点分值最高、次日开盘买，盈利达 3% 即卖，不盈利则持有至期末。
不设止损。选股与卖出仅用当日及历史，禁止未来数据。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from app.paths import GPT_DATA_DIR
from scripts.backtest_mode3_2023_2024 import run_backtest, _compute_stats, TRADE_CN, REASON_CN

CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
RES_DIR = os.path.join(ROOT, "data", "results")
INITIAL_CASH = 100_000.0
YEAR_CONFIG = [
    (2023, "2023-01-01", "2023-12-31"),
    (2024, "2024-01-01", "2024-12-31"),
    (2025, "2025-01-01", "2025-12-31"),
]


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    all_trades = []
    cash = INITIAL_CASH
    stats_per_year = []

    for year, start, end in YEAR_CONFIG:
        path = os.path.join(RES_DIR, f"mode9_{year}_picks.csv")
        if not os.path.exists(path):
            print(f"缺少 {path}，请先运行 backtest_mode9_3year.py（会生成选股）")
            return
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "buy_point_score" not in df.columns:
            df["buy_point_score"] = 0
        trades, final_cash = run_backtest(
            df,
            CACHE_DIR,
            start,
            end,
            initial_cash=cash,
            stop_loss=0.05,
            ma_exit=20,
            use_stop_loss=False,
            take_profit_only=0.03,
        )
        ret_pct = (final_cash / cash - 1) * 100
        print(f"{year} 期初 {cash:.2f} 期末 {final_cash:.2f} 收益率 {ret_pct:.2f}% 交易 {len(trades)} 笔")
        stats_per_year.extend(_compute_stats(trades, cash, final_cash, year))
        for t in trades:
            t["year"] = year
            all_trades.append(t)
        cash = final_cash

    total_ret = (cash / INITIAL_CASH - 1) * 100
    print(f"\n三年复利 期初 {INITIAL_CASH:.2f} 期末 {cash:.2f} 总收益率 {total_ret:.2f}% （约 {cash/INITIAL_CASH:.2f} 倍）")

    out_xlsx = os.path.join(RES_DIR, "mode9_3year_take_profit3.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary = pd.DataFrame([
            {"指标": "期初资金(元)", "数值": INITIAL_CASH},
            {"指标": "期末资金(元)", "数值": round(cash, 2)},
            {"指标": "三年总收益率(%)", "数值": round(total_ret, 2)},
            {"指标": "总交易笔数", "数值": len(all_trades)},
        ])
        summary.to_excel(writer, index=False, sheet_name="三年汇总")
        if stats_per_year:
            pd.DataFrame(stats_per_year).to_excel(writer, index=False, sheet_name="分年统计")
        if all_trades:
            df_t = pd.DataFrame(all_trades)
            df_t["reason"] = df_t["reason"].map(lambda x: REASON_CN.get(x, x))
            cn_cols = [TRADE_CN[k] for k in TRADE_CN if k in df_t.columns]
            df_t.rename(columns=TRADE_CN)[cn_cols].to_excel(writer, index=False, sheet_name="交易明细")
    print(f"已保存: {out_xlsx}")


if __name__ == "__main__":
    main()
