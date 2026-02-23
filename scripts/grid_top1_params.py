import argparse
import os
from datetime import datetime
from itertools import product
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
    dates: List[str],
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ma20: np.ndarray,
    buy_idx: int,
    stop_loss: float,
    take_profit: float,
    end_date: Optional[str],
) -> Tuple[int, float, str]:
    exit_idx = len(dates) - 1
    exit_price = close[exit_idx]
    reason = "end"
    stop_price = open_[buy_idx] * (1 - stop_loss)
    target_price = open_[buy_idx] * take_profit
    for i in range(buy_idx, len(dates)):
        if end_date and dates[i] > end_date:
            break
        if low[i] <= stop_price:
            return i, stop_price, "stop_loss"
        if high[i] >= target_price:
            return i, target_price, "take_profit"
        if not np.isnan(ma20[i]) and close[i] < ma20[i]:
            return i, close[i], "ma20_break"
    return exit_idx, exit_price, reason


def _prepare_price_maps(
    codes: List[str],
    cache_dir: str,
    cache_format: str,
) -> Dict[str, Dict[str, object]]:
    maps: Dict[str, Dict[str, object]] = {}
    for code in sorted(set(codes)):
        market = _market_from_code(code)
        rows = _load_rows(cache_dir, cache_format, market, code)
        if not rows or len(rows) < 80:
            continue
        dates = [r.date for r in rows]
        open_ = np.array([r.open for r in rows], dtype=float)
        high = np.array([r.high for r in rows], dtype=float)
        low = np.array([r.low for r in rows], dtype=float)
        close = np.array([r.close for r in rows], dtype=float)
        ma20 = _moving_mean(close, 20)
        maps[code] = {
            "dates": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "ma20": ma20,
        }
    return maps


def _simulate(
    top1: pd.DataFrame,
    price_maps: Dict[str, Dict[str, object]],
    start_date: str,
    end_date: str,
    initial_cash: float,
    stop_loss: float,
    take_profit: float,
) -> Tuple[float, List[Dict[str, object]]]:
    cash = initial_cash
    last_exit_date: Optional[str] = None
    trades: List[Dict[str, object]] = []

    for _, row in top1.iterrows():
        signal_date = str(row["date"])
        if start_date and signal_date < start_date:
            continue
        if end_date and signal_date > end_date:
            continue
        if last_exit_date and signal_date <= last_exit_date:
            continue

        code = str(row["code"]).zfill(6)
        name = row.get("name", "")
        data = price_maps.get(code)
        if not data:
            continue
        dates = data["dates"]
        if signal_date not in dates:
            continue
        sig_idx = dates.index(signal_date)
        buy_idx = sig_idx + 1
        if buy_idx >= len(dates):
            continue
        buy_date = dates[buy_idx]
        if buy_date > end_date:
            continue

        open_ = data["open"]
        if open_[buy_idx] <= 0:
            continue
        high = data["high"]
        low = data["low"]
        close = data["close"]
        ma20 = data["ma20"]

        exit_idx, exit_price, reason = _find_exit(
            dates,
            open_,
            high,
            low,
            close,
            ma20,
            buy_idx,
            stop_loss,
            take_profit,
            end_date,
        )

        entry_price = open_[buy_idx]
        shares = cash / entry_price
        cash = shares * exit_price
        ret_pct = (exit_price - entry_price) / entry_price * 100
        hold_days = (_parse_date(dates[exit_idx]) - _parse_date(buy_date)).days
        trades.append(
            {
                "signal_date": signal_date,
                "code": code,
                "name": name,
                "buy_date": buy_date,
                "buy_price": round(entry_price, 4),
                "sell_date": dates[exit_idx],
                "sell_price": round(exit_price, 4),
                "reason": reason,
                "return_pct": round(ret_pct, 2),
                "hold_days": hold_days,
                "cash_after": round(cash, 2),
            }
        )
        last_exit_date = dates[exit_idx]

    return cash, trades


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search for best Top1 sequential params.")
    parser.add_argument("--picks-csv", default="data/results/mode3_2025_top3.csv")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--cache-dir", default="data/gpt/kline_cache_tencent")
    parser.add_argument("--cache-format", choices=["code", "secid"], default="secid")
    parser.add_argument("--initial-cash", type=float, default=100000)
    parser.add_argument("--stop-losses", default="0.05,0.08,0.10,0.12,0.15")
    parser.add_argument("--take-profits", default="1.5,2.0,2.5")
    parser.add_argument("--output-xlsx", default="data/results/mode3_top1_grid_2025.xlsx")
    args = parser.parse_args()

    picks = pd.read_csv(args.picks_csv)
    picks["date"] = picks["date"].astype(str)
    picks["code"] = picks["code"].astype(str).str.zfill(6)
    picks = picks.sort_values("date")
    top1 = picks.groupby("date", as_index=False).first().sort_values("date")

    codes = list(top1["code"].unique())
    price_maps = _prepare_price_maps(codes, args.cache_dir, args.cache_format)

    stop_losses = [float(x) for x in args.stop_losses.split(",") if x.strip()]
    take_profits = [float(x) for x in args.take_profits.split(",") if x.strip()]

    results = []
    best = None
    best_trades: List[Dict[str, object]] = []

    for sl, tp in product(stop_losses, take_profits):
        cash, trades = _simulate(
            top1,
            price_maps,
            args.start_date,
            args.end_date,
            args.initial_cash,
            sl,
            tp,
        )
        ret_pct = (cash / args.initial_cash - 1) * 100
        results.append(
            {
                "stop_loss": sl,
                "take_profit": tp,
                "end_cash": round(cash, 2),
                "return_pct": round(ret_pct, 2),
                "trades": len(trades),
            }
        )
        if best is None or cash > best["end_cash"]:
            best = results[-1]
            best_trades = trades

    os.makedirs(os.path.dirname(args.output_xlsx), exist_ok=True)
    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl") as writer:
        pd.DataFrame(results).sort_values("end_cash", ascending=False).to_excel(
            writer, index=False, sheet_name="grid"
        )
        if best:
            pd.DataFrame([best]).to_excel(writer, index=False, sheet_name="best_summary")
        if best_trades:
            pd.DataFrame(best_trades).to_excel(writer, index=False, sheet_name="best_trades")

    if best:
        print("最佳参数:")
        print(best)
    print(f"输出: {args.output_xlsx}")


if __name__ == "__main__":
    main()
