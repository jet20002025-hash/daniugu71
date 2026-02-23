#!/usr/bin/env python3
"""
71倍模型：当日多只满分100时，对比买高开 vs 买低开 的收益率
- 买高开：选 open/prev_close 最高的
- 买低开：选 open/prev_close 最低的
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
        return 200
    if c.startswith("300"):
        return 200
    if c.startswith(("8", "9")) and len(c) == 6:
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


def _find_exit(rows, buy_idx: int, stop_loss_price: float, ma: np.ndarray, end_date: Optional[str], ma_period: int = 10) -> Tuple[int, float, str]:
    exit_idx = buy_idx
    exit_price = rows[buy_idx].close
    reason = "end"
    for i in range(buy_idx + 1, len(rows)):
        if end_date and rows[i].date > end_date:
            break
        low, close = rows[i].low, rows[i].close
        if low <= stop_loss_price:
            exit_idx, exit_price, reason = i, stop_loss_price, "止损10%"
            break
        if not np.isnan(ma[i]) and ma[i] > 0 and close < ma[i]:
            exit_idx, exit_price, reason = i, close, f"破MA{ma_period}"
            break
        exit_idx, exit_price, reason = i, close, "end"
    if end_date and rows[exit_idx].date > end_date:
        for j in range(exit_idx - 1, buy_idx - 1, -1):
            if rows[j].date <= end_date:
                exit_idx, exit_price, reason = j, rows[j].close, "期末平仓"
                break
    return exit_idx, exit_price, reason


def _run_strategy(
    picks: pd.DataFrame,
    pick_func,
    cache_dir: str,
    start_date: str,
    end_date: str,
    initial_cash: float,
    stop_loss: float,
    ma_exit: int,
) -> Tuple[float, int, List[Dict]]:
    """pick_func(candidates, cache_dir) -> chosen row or None"""
    picks = picks.sort_values(["date", "score"], ascending=[True, False])
    daily_100 = picks[picks["score"] == 100].groupby("date").apply(lambda g: g.to_dict("records"), include_groups=False).to_dict()

    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    trades: List[Dict] = []
    cash = initial_cash
    position: Optional[Dict] = None
    sold_today: Optional[str] = None

    for signal_date in sorted(daily_100.keys()):
        if start_dt and _parse_date(signal_date) and _parse_date(signal_date) < start_dt:
            continue
        if end_dt and _parse_date(signal_date) and _parse_date(signal_date) > end_dt:
            continue

        candidates = daily_100[signal_date]
        if end_dt:
            candidates = [c for c in candidates if _parse_date(str(c.get("buy_date", ""))) and _parse_date(str(c.get("buy_date", ""))) <= end_dt]
        if start_dt:
            candidates = [c for c in candidates if _parse_date(str(c.get("buy_date", ""))) and _parse_date(str(c.get("buy_date", ""))) >= start_dt]
        if not candidates:
            continue

        row = pick_func(candidates, cache_dir, signal_date)
        if row is None:
            continue

        buy_date = str(row.get("buy_date", "")).strip()
        code = str(row.get("code", "")).strip().zfill(6)
        name = row.get("name", code)
        if not code or not buy_date:
            continue
        if sold_today and buy_date == sold_today:
            continue

        if position:
            pos_code = position["code"]
            pos_rows = _load_rows(cache_dir, pos_code)
            if not pos_rows or len(pos_rows) < 80:
                position = None
                continue
            dates = [r.date for r in pos_rows]
            buy_idx_pos = next((i for i, d in enumerate(dates) if d == position["buy_date"]), None)
            if buy_idx_pos is None:
                position = None
                continue
            close_arr = np.array([r.close for r in pos_rows], dtype=float)
            ma = _moving_mean(close_arr, ma_exit)
            stop_price = position["buy_price"] * (1 - stop_loss)
            exit_idx, exit_price, reason = _find_exit(pos_rows, buy_idx_pos, stop_price, ma, end_date, ma_exit)
            exit_date = pos_rows[exit_idx].date
            if exit_date > buy_date:
                continue
            sell_value = position["shares"] * exit_price
            cash += sell_value
            ret_pct = (exit_price - position["buy_price"]) / position["buy_price"] * 100
            hold_days = (_parse_date(exit_date) - _parse_date(position["buy_date"])).days if _parse_date(exit_date) and _parse_date(position["buy_date"]) else 0
            trades.append({
                "信号日": position["signal_date"], "代码": pos_code, "名称": position["name"],
                "买入日": position["buy_date"], "买入价": round(position["buy_price"], 4),
                "卖出日": exit_date, "卖出价": round(exit_price, 4), "卖出原因": reason,
                "收益率%": round(ret_pct, 2), "持仓天数": hold_days, "股数": position["shares"],
                "盈亏": round(sell_value - position["shares"] * position["buy_price"], 2), "卖出后资金": round(cash, 2),
            })
            sold_today = exit_date
            position = None
            if exit_date == buy_date:
                continue

        kline = _load_rows(cache_dir, code)
        if not kline or len(kline) < 80 or buy_date not in [r.date for r in kline]:
            continue
        dates = [r.date for r in kline]
        buy_idx = dates.index(buy_date)
        entry_price = kline[buy_idx].open
        if entry_price <= 0:
            continue
        min_lot = _min_lot(code)
        shares = _calc_shares(cash, entry_price, min_lot)
        if shares < min_lot:
            continue
        cash -= shares * entry_price
        close_arr = np.array([r.close for r in kline], dtype=float)
        ma = _moving_mean(close_arr, ma_exit)
        stop_price = entry_price * (1 - stop_loss)
        exit_idx, exit_price, reason = _find_exit(kline, buy_idx, stop_price, ma, end_date, ma_exit)
        position = {
            "code": code, "name": name, "shares": shares, "buy_price": entry_price,
            "buy_date": buy_date, "signal_date": signal_date,
            "exit_date": kline[exit_idx].date, "exit_price": exit_price, "reason": reason,
        }

    if position:
        sell_value = position["shares"] * position["exit_price"]
        cash += sell_value
        ret_pct = (position["exit_price"] - position["buy_price"]) / position["buy_price"] * 100
        hold_days = (_parse_date(position["exit_date"]) - _parse_date(position["buy_date"])).days if _parse_date(position["exit_date"]) and _parse_date(position["buy_date"]) else 0
        trades.append({
            "信号日": position["signal_date"], "代码": position["code"], "名称": position["name"],
            "买入日": position["buy_date"], "买入价": round(position["buy_price"], 4),
            "卖出日": position["exit_date"], "卖出价": round(position["exit_price"], 4), "卖出原因": position["reason"],
            "收益率%": round(ret_pct, 2), "持仓天数": hold_days, "股数": position["shares"],
            "盈亏": round(sell_value - position["shares"] * position["buy_price"], 2), "卖出后资金": round(cash, 2),
        })

    return_pct = (cash / initial_cash - 1) * 100
    return return_pct, len(trades), trades


def _pick_high_open(candidates: List[Dict], cache_dir: str, signal_date: str):
    """选 open/prev_close 最高的（高开）"""
    best = None
    best_ratio = -1.0
    for c in candidates:
        code = str(c.get("code", "")).strip().zfill(6)
        buy_date = str(c.get("buy_date", "")).strip()
        rows = _load_rows(cache_dir, code)
        if not rows or len(rows) < 80:
            continue
        dates = [r.date for r in rows]
        if buy_date not in dates or signal_date not in dates:
            continue
        buy_idx = dates.index(buy_date)
        sig_idx = dates.index(signal_date)
        prev_close = rows[sig_idx].close
        open_price = rows[buy_idx].open
        if prev_close <= 0 or open_price <= 0:
            continue
        ratio = open_price / prev_close
        if ratio > best_ratio:
            best_ratio = ratio
            best = c
            best["code"] = code
    return best


def _pick_low_open(candidates: List[Dict], cache_dir: str, signal_date: str):
    """选 open/prev_close 最低的（低开）"""
    best = None
    best_ratio = 999.0
    for c in candidates:
        code = str(c.get("code", "")).strip().zfill(6)
        buy_date = str(c.get("buy_date", "")).strip()
        rows = _load_rows(cache_dir, code)
        if not rows or len(rows) < 80:
            continue
        dates = [r.date for r in rows]
        if buy_date not in dates or signal_date not in dates:
            continue
        buy_idx = dates.index(buy_date)
        sig_idx = dates.index(signal_date)
        prev_close = rows[sig_idx].close
        open_price = rows[buy_idx].open
        if prev_close <= 0 or open_price <= 0:
            continue
        ratio = open_price / prev_close
        if ratio < best_ratio:
            best_ratio = ratio
            best = c
            best["code"] = code
    return best


def main():
    parser = argparse.ArgumentParser(description="71倍满分100：买高开 vs 买低开 收益率对比")
    parser.add_argument("--picks-csv", required=True, help="mode3选股CSV")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--cache-dir", default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"))
    parser.add_argument("--initial-cash", type=float, default=100000)
    parser.add_argument("--stop-loss", type=float, default=0.10)
    parser.add_argument("--ma-exit", type=int, default=10)
    parser.add_argument("--output-xlsx", default="data/results/high_open_vs_low_open.xlsx")
    args = parser.parse_args()

    picks = pd.read_csv(args.picks_csv)
    if "date" not in picks.columns and "signal_date" in picks.columns:
        picks["date"] = picks["signal_date"]
    picks["date"] = picks["date"].astype(str)
    picks["buy_date"] = picks["buy_date"].astype(str)
    picks["code"] = picks["code"].astype(str).str.zfill(6)
    picks = picks[picks["score"] == 100].copy()

    ret_high, n_high, trades_high = _run_strategy(
        picks, _pick_high_open, args.cache_dir,
        args.start_date, args.end_date,
        args.initial_cash, args.stop_loss, args.ma_exit,
    )
    ret_low, n_low, trades_low = _run_strategy(
        picks, _pick_low_open, args.cache_dir,
        args.start_date, args.end_date,
        args.initial_cash, args.stop_loss, args.ma_exit,
    )

    summary = pd.DataFrame([
        {"策略": "买高开", "收益率%": round(ret_high, 2), "交易次数": n_high},
        {"策略": "买低开", "收益率%": round(ret_low, 2), "交易次数": n_low},
    ])
    os.makedirs(os.path.dirname(args.output_xlsx), exist_ok=True)
    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="对比汇总")
        if trades_high:
            pd.DataFrame(trades_high).to_excel(writer, index=False, sheet_name="买高开明细")
        if trades_low:
            pd.DataFrame(trades_low).to_excel(writer, index=False, sheet_name="买低开明细")

    print("=" * 50)
    print(f"回测区间: {args.start_date} ~ {args.end_date}")
    print("策略: 当日多只满分100时")
    print("-" * 50)
    print(f"买高开(open/prev_close最高): 收益率 {ret_high:.2f}%  交易 {n_high} 次")
    print(f"买低开(open/prev_close最低): 收益率 {ret_low:.2f}%  交易 {n_low} 次")
    print("=" * 50)
    print("输出:", args.output_xlsx)


if __name__ == "__main__":
    main()
