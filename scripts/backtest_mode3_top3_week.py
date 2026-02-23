#!/usr/bin/env python3
"""
mode3 每日 top3 回测：最近一周，每日买 top3，次日开盘买入
- 10% 止损、破 MA20 卖出
- 每只分配 1/3 资金，最多 3 个仓位
"""
import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.eastmoney import read_cached_kline_by_code
from app.paths import GPT_DATA_DIR


def _parse_date(value: str) -> Optional[datetime.date]:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _min_lot(code: str) -> int:
    c = str(code).strip()
    if c.startswith("688"):
        return 1
    if c.startswith("300"):
        return 200
    return 100


def _load_rows(cache_dir: str, code: str):
    return read_cached_kline_by_code(cache_dir, code)


def _moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    res = np.full_like(values, np.nan, dtype=float)
    if len(values) < window:
        return res
    weights = np.ones(window, dtype=float) / window
    res[window - 1 :] = np.convolve(values, weights, mode="valid")
    return res


def _calc_shares(cash: float, price: float, min_lot: int) -> int:
    if price <= 0:
        return 0
    raw = int(cash / price)
    return max(0, (raw // min_lot) * min_lot)


def _find_exit(rows, buy_idx: int, stop_loss_price: float, ma: np.ndarray, end_date: Optional[str], ma_period: int = 20) -> Tuple[int, float, str]:
    reason_suffix = f"ma{ma_period}_break"
    exit_idx = buy_idx
    exit_price = rows[buy_idx].close
    reason = "end"
    for i in range(buy_idx + 1, len(rows)):
        if end_date and rows[i].date > end_date:
            break
        low, close = rows[i].low, rows[i].close
        if low <= stop_loss_price:
            exit_idx, exit_price, reason = i, stop_loss_price, "stop_loss_10pct"
            break
        if not np.isnan(ma[i]) and ma[i] > 0 and close < ma[i]:
            exit_idx, exit_price, reason = i, close, reason_suffix
            break
        exit_idx, exit_price, reason = i, close, "end"
    if end_date and rows[exit_idx].date > end_date:
        for j in range(exit_idx - 1, buy_idx - 1, -1):
            if rows[j].date <= end_date:
                exit_idx, exit_price, reason = j, rows[j].close, "end_date"
                break
    return exit_idx, exit_price, reason


def main() -> None:
    parser = argparse.ArgumentParser(description="mode3 最近一周每日 top3 回测")
    parser.add_argument("--picks-csv", default="data/results/mode3_top3_last_week.csv")
    parser.add_argument("--cache-dir", default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"))
    parser.add_argument("--initial-cash", type=float, default=100000)
    parser.add_argument("--stop-loss", type=float, default=0.10)
    parser.add_argument("--ma-exit", type=int, default=20)
    parser.add_argument("--output-xlsx", default="data/results/mode3_top3_last_week_backtest.xlsx")
    args = parser.parse_args()

    picks = pd.read_csv(args.picks_csv)
    if "date" not in picks.columns and "signal_date" in picks.columns:
        picks["date"] = picks["signal_date"]
    picks["date"] = picks["date"].astype(str)
    picks["buy_date"] = picks["buy_date"].astype(str)
    picks["code"] = picks["code"].astype(str).str.zfill(6)
    picks = picks.sort_values(["date", "score"], ascending=[True, False])
    daily_top3 = picks.groupby("date").apply(lambda g: g.head(3).to_dict("records"), include_groups=False).to_dict()

    trades: List[Dict] = []
    actions: List[Dict] = []
    cash = args.initial_cash
    positions: List[Dict] = []
    sold_today: set = set()

    for signal_date in sorted(daily_top3.keys()):
        candidates = daily_top3[signal_date]
        buy_date = str(candidates[0].get("buy_date", "")).strip() if candidates else ""

        # 检查并卖出到期仓位
        still_holding = []
        for pos in positions:
            pos_rows = _load_rows(args.cache_dir, pos["code"])
            if not pos_rows or len(pos_rows) < 80:
                still_holding.append(pos)
                continue
            dates = [r.date for r in pos_rows]
            buy_idx = next((i for i, d in enumerate(dates) if d == pos["buy_date"]), None)
            if buy_idx is None:
                still_holding.append(pos)
                continue
            close = np.array([r.close for r in pos_rows], dtype=float)
            ma = _moving_mean(close, args.ma_exit)
            stop_price = pos["buy_price"] * (1 - args.stop_loss)
            exit_idx, exit_price, reason = _find_exit(pos_rows, buy_idx, stop_price, ma, None, args.ma_exit)
            exit_date = pos_rows[exit_idx].date

            if exit_date <= buy_date:
                sell_value = pos["shares"] * exit_price
                cash += sell_value
                ret_pct = (exit_price - pos["buy_price"]) / pos["buy_price"] * 100
                trades.append({
                    "signal_date": pos["signal_date"], "code": pos["code"], "name": pos["name"],
                    "buy_date": pos["buy_date"], "buy_price": round(pos["buy_price"], 4),
                    "sell_date": exit_date, "sell_price": round(exit_price, 4), "reason": reason,
                    "return_pct": round(ret_pct, 2), "shares": pos["shares"], "cash_after": round(cash, 2),
                })
                actions.append({"date": exit_date, "action": "sell", "code": pos["code"], "name": pos["name"], "price": exit_price})
                sold_today.add(exit_date)
            else:
                still_holding.append(pos)
        positions = still_holding

        # 仅当无持仓时买入当日 top3
        if positions:
            continue

        cash_per_stock = cash / 3.0
        bought = 0
        for c in candidates[:3]:
            code = str(c.get("code", "")).strip().zfill(6)
            if not code:
                continue
            trade_date = str(c.get("buy_date", "")).strip()
            if not trade_date or (sold_today and trade_date in sold_today):
                continue
            kline = _load_rows(args.cache_dir, code)
            if not kline or len(kline) < 80 or trade_date not in [r.date for r in kline]:
                continue
            buy_idx = next(i for i, r in enumerate(kline) if r.date == trade_date)
            entry_price = kline[buy_idx].open
            if entry_price <= 0:
                continue
            min_lot = _min_lot(code)
            shares = _calc_shares(cash_per_stock, entry_price, min_lot)
            if shares < min_lot:
                continue
            cost = shares * entry_price
            cash -= cost
            close = np.array([r.close for r in kline], dtype=float)
            ma = _moving_mean(close, args.ma_exit)
            stop_price = entry_price * (1 - args.stop_loss)
            exit_idx, exit_price, reason = _find_exit(kline, buy_idx, stop_price, ma, None, args.ma_exit)
            exit_date = kline[exit_idx].date
            positions.append({
                "code": code, "name": c.get("name", code), "shares": shares, "buy_price": entry_price,
                "buy_date": trade_date, "signal_date": signal_date, "exit_date": exit_date,
                "exit_price": exit_price, "reason": reason,
            })
            actions.append({"date": trade_date, "action": "buy", "code": code, "name": c.get("name", code), "price": entry_price})
            bought += 1

    # 平仓剩余持仓
    for pos in positions:
        sell_value = pos["shares"] * pos["exit_price"]
        cash += sell_value
        ret_pct = (pos["exit_price"] - pos["buy_price"]) / pos["buy_price"] * 100
        trades.append({
            "signal_date": pos["signal_date"], "code": pos["code"], "name": pos["name"],
            "buy_date": pos["buy_date"], "buy_price": round(pos["buy_price"], 4),
            "sell_date": pos["exit_date"], "sell_price": round(pos["exit_price"], 4), "reason": pos["reason"],
            "return_pct": round(ret_pct, 2), "shares": pos["shares"], "cash_after": round(cash, 2),
        })
        actions.append({"date": pos["exit_date"], "action": "sell", "code": pos["code"], "name": pos["name"], "price": pos["exit_price"]})

    summary = {"start_cash": args.initial_cash, "end_cash": round(cash, 2), "return_pct": round((cash / args.initial_cash - 1) * 100, 2), "trades": len(trades)}
    top3_list = picks[picks["date"].isin(daily_top3.keys())].drop_duplicates(subset=["date", "code"])
    os.makedirs(os.path.dirname(args.output_xlsx), exist_ok=True)
    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl") as w:
        pd.DataFrame([summary]).to_excel(w, index=False, sheet_name="summary")
        top3_list.to_excel(w, index=False, sheet_name="daily_top3_list")
        pd.DataFrame(trades).to_excel(w, index=False, sheet_name="trades")
        pd.DataFrame(actions).sort_values("date").to_excel(w, index=False, sheet_name="actions")
    print("回测完成: 最近一周每日 top3")
    print("期初:", args.initial_cash, "期末:", round(cash, 2), "收益率:", round((cash / args.initial_cash - 1) * 100, 2), "%")
    print("交易次数:", len(trades), "输出:", args.output_xlsx)


if __name__ == "__main__":
    main()
