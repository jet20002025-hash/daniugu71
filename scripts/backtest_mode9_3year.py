#!/usr/bin/env python3
"""
mode9 三年回测（2023/2024/2025）：每日买点分值最高个股、次日开盘买、5% 止损、破 MA20 止盈。
规则：10 万本金、复利、不限制市值；同分按买点分值取 top1；选股与卖出仅用当日及历史，禁止未来数据。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from app.eastmoney import list_cached_stocks_flat, load_stock_list_csv, read_cached_kline_by_code
from app.paths import GPT_DATA_DIR
from app.scanner import ScanConfig, scan_with_mode3
from scripts.backtest_mode3_2023_2024 import run_backtest, _compute_stats, TRADE_CN, REASON_CN

CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
STOCK_LIST_CSV = os.path.join(GPT_DATA_DIR, "stock_list.csv")
RES_DIR = os.path.join(ROOT, "data", "results")
INITIAL_CASH = 100_000.0


def _kline_loader(item):
    return read_cached_kline_by_code(CACHE_DIR, item.code)


def _generate_picks_for_year(year: int, limit: int = 0) -> pd.DataFrame:
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    name_map = load_stock_list_csv(STOCK_LIST_CSV) if os.path.exists(STOCK_LIST_CSV) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if not stock_list:
        raise SystemExit("本地股票缓存为空，请先更新 K 线。")
    if limit:
        stock_list = stock_list[:limit]
    config = ScanConfig(min_score=70, max_results=150000, max_market_cap=None)
    print(f"mode9 扫描 {year} …")
    results = scan_with_mode3(
        stock_list,
        config,
        CACHE_DIR,
        progress_cb=None,
        kline_loader=_kline_loader,
        cutoff_date=end,
        start_date=start,
        use_mode8=False,
        use_mode9=True,
        use_71x_standard=True,
    )
    rows = []
    for r in results:
        m = r.metrics or {}
        sig = m.get("signal_date") or ""
        buy = m.get("buy_date") or ""
        if not sig or not buy:
            continue
        buy_pt = (r.metrics or {}).get("buy_point_score")
        rows.append({
            "date": sig,
            "signal_date": sig,
            "buy_date": buy,
            "code": r.code.zfill(6) if len(r.code) < 6 else r.code,
            "name": r.name or r.code,
            "score": r.score,
            "buy_point_score": int(buy_pt) if buy_pt is not None else 0,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "score", "buy_point_score"], ascending=[True, False, False])
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser(description="mode9 三年回测：买点分值最高、次日开盘买、5%止损、破MA20止盈")
    ap.add_argument("--refresh", action="store_true", help="重新生成各年选股 CSV")
    ap.add_argument("--limit", type=int, default=0, help="仅扫描前 N 只股票（0=全部，调试用）")
    args = ap.parse_args()

    os.makedirs(RES_DIR, exist_ok=True)
    prefix = "mode9"
    year_config = [
        (2023, "2023-01-01", "2023-12-31"),
        (2024, "2024-01-01", "2024-12-31"),
        (2025, "2025-01-01", "2025-12-31"),
    ]
    all_trades = []
    cash = INITIAL_CASH
    stats_per_year = []

    for year, start, end in year_config:
        picks_path = os.path.join(RES_DIR, f"{prefix}_{year}_picks.csv")
        if args.refresh or not os.path.exists(picks_path):
            df = _generate_picks_for_year(year, limit=args.limit)
            if df.empty:
                print(f"{year} 无选股，跳过")
                continue
            df.to_csv(picks_path, index=False, encoding="utf-8-sig")
            print(f"已保存 {picks_path}，{len(df)} 条")
        df = pd.read_csv(picks_path)
        if df.empty:
            print(f"{year} 选股为空，跳过")
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
            use_stop_loss=True,
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

    out_xlsx = os.path.join(RES_DIR, f"{prefix}_3year_backtest.xlsx")
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
