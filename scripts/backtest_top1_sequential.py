import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.eastmoney import read_cached_kline, read_cached_kline_by_code


def _parse_date(value: str) -> Optional[datetime.date]:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _market_from_code(code: str) -> int:
    return 1 if str(code).startswith("6") else 0


def _load_rows(cache_dir: str, cache_format: str, market: int, code: str):
    if cache_format == "secid":
        path = os.path.join(cache_dir, f"{market}_{code}.csv")
        return read_cached_kline(path)
    return read_cached_kline_by_code(cache_dir, code)


def _moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    res = np.full_like(values, np.nan, dtype=float)
    if len(values) < window:
        return res
    weights = np.ones(window, dtype=float) / window
    res[window - 1 :] = np.convolve(values, weights, mode="valid")
    return res


def _find_exit(
    rows,
    buy_idx: int,
    stop_loss: float,
    take_profit: float,
    ma20: np.ndarray,
    end_date: Optional[str],
) -> Tuple[int, float, str]:
    exit_idx = len(rows) - 1
    exit_price = rows[exit_idx].close
    reason = "end"
    start_i = min(buy_idx + 1, len(rows) - 1)
    for i in range(start_i, len(rows)):
        if end_date:
            row_dt = rows[i].date
            if row_dt > end_date:
                break
        low = rows[i].low
        high = rows[i].high
        close = rows[i].close
        if low <= stop_loss:
            exit_idx = i
            exit_price = stop_loss
            reason = "stop_loss"
            break
        if high >= take_profit:
            exit_idx = i
            exit_price = take_profit
            reason = "take_profit"
            break
        if not np.isnan(ma20[i]) and close < ma20[i]:
            exit_idx = i
            exit_price = close
            reason = "ma20_break"
            break
    return exit_idx, exit_price, reason


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential top1 backtest with stop-loss/take-profit/MA20 exit.")
    parser.add_argument("--picks-csv", default="data/results/mode3_2025_top3.csv")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--cache-dir", default="data/gpt/kline_cache_tencent")
    parser.add_argument("--cache-format", choices=["code", "secid"], default="secid")
    parser.add_argument("--initial-cash", type=float, default=100000)
    parser.add_argument("--stop-loss", type=float, default=0.10)
    parser.add_argument("--take-profit", type=float, default=2.0)
    parser.add_argument("--output-xlsx", default="data/results/mode3_top1_seq_2025.xlsx")
    args = parser.parse_args()

    picks = pd.read_csv(args.picks_csv)
    picks["date"] = picks["date"].astype(str)
    picks["code"] = picks["code"].astype(str).str.zfill(6)
    picks = picks.sort_values("date")

    start_dt = _parse_date(args.start_date)
    end_dt = _parse_date(args.end_date)

    # take top1 per day (picks already ordered, so first row per date)
    top1 = picks.groupby("date", as_index=False).first()
    top1 = top1.sort_values("date")

    trades: List[Dict[str, object]] = []
    actions: List[Dict[str, object]] = []
    cash = args.initial_cash
    last_exit_date: Optional[str] = None

    for _, row in top1.iterrows():
        signal_date = row["date"]
        if start_dt and _parse_date(signal_date) and _parse_date(signal_date) < start_dt:
            continue
        if end_dt and _parse_date(signal_date) and _parse_date(signal_date) > end_dt:
            continue
        if last_exit_date and signal_date <= last_exit_date:
            continue

        code = row["code"]
        name = row.get("name", "")
        market = _market_from_code(code)
        rows = _load_rows(args.cache_dir, args.cache_format, market, code)
        if not rows or len(rows) < 80:
            continue

        dates = [r.date for r in rows]
        if signal_date not in dates:
            continue
        sig_idx = dates.index(signal_date)
        buy_idx = sig_idx + 1
        if buy_idx >= len(rows):
            continue

        buy_date = rows[buy_idx].date
        if start_dt and _parse_date(buy_date) and _parse_date(buy_date) < start_dt:
            continue
        if end_dt and _parse_date(buy_date) and _parse_date(buy_date) > end_dt:
            continue

        close = np.array([r.close for r in rows], dtype=float)
        ma20 = _moving_mean(close, 20)
        entry_price = rows[buy_idx].open
        if entry_price <= 0:
            continue
        stop_price = entry_price * (1 - args.stop_loss)
        target_price = entry_price * args.take_profit

        exit_idx, exit_price, reason = _find_exit(
            rows, buy_idx, stop_price, target_price, ma20, args.end_date
        )
        exit_date = rows[exit_idx].date

        shares = cash / entry_price
        cash = shares * exit_price
        ret_pct = (exit_price - entry_price) / entry_price * 100
        hold_days = (_parse_date(exit_date) - _parse_date(buy_date)).days if _parse_date(exit_date) and _parse_date(buy_date) else 0

        trades.append(
            {
                "signal_date": signal_date,
                "code": code,
                "name": name,
                "buy_date": buy_date,
                "buy_price": round(entry_price, 4),
                "sell_date": exit_date,
                "sell_price": round(exit_price, 4),
                "reason": reason,
                "return_pct": round(ret_pct, 2),
                "hold_days": hold_days,
                "cash_after": round(cash, 2),
            }
        )
        actions.append({"date": buy_date, "action": "buy", "code": code, "name": name, "price": entry_price})
        actions.append({"date": exit_date, "action": "sell", "code": code, "name": name, "price": exit_price})

        last_exit_date = exit_date

    summary = {
        "start_cash": args.initial_cash,
        "end_cash": round(cash, 2),
        "return_pct": round((cash / args.initial_cash - 1) * 100, 2),
        "trades": len(trades),
    }

    os.makedirs(os.path.dirname(args.output_xlsx), exist_ok=True)
    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="summary")
        pd.DataFrame(trades).to_excel(writer, index=False, sheet_name="trades")
        pd.DataFrame(actions).sort_values("date").to_excel(writer, index=False, sheet_name="actions")

    print("完成回测:", args.start_date, "~", args.end_date)
    print("期末资金:", round(cash, 2), "收益率:", round((cash / args.initial_cash - 1) * 100, 2), "%")
    print("输出:", args.output_xlsx)


if __name__ == "__main__":
    main()
