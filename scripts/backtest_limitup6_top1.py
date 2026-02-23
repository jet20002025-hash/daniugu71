import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.eastmoney import read_cached_kline, stock_items_from_list_csv
from app.paths import GPT_DATA_DIR


def _parse_date(value: str) -> Optional[datetime.date]:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _load_rows(cache_dir: str, market: int, code: str):
    path = os.path.join(cache_dir, f"{market}_{code}.csv")
    return read_cached_kline(path)


def _is_st(name: str) -> bool:
    if not name:
        return False
    return "ST" in name or name.startswith("*ST") or name.startswith("退")


def _limit_rate(code: str, name: str) -> float:
    code = str(code)
    if _is_st(name):
        return 0.05
    if code.startswith(("30", "301", "688")):
        return 0.20
    if code.startswith(("8", "9")):
        return 0.30
    return 0.10


def _is_limit_up(prev_close: float, close: float, limit: float, tol: float = 0.002) -> bool:
    if prev_close <= 0:
        return False
    limit_price = prev_close * (1 + limit)
    return close >= limit_price * (1 - tol)


def _is_one_word(open_: float, high: float, low: float, close: float, prev_close: float, limit: float) -> bool:
    if prev_close <= 0:
        return False
    limit_price = prev_close * (1 + limit)
    if open_ < limit_price * 0.998:
        return False
    return abs(high - low) < 1e-6 and abs(close - open_) < 1e-6


def _build_candidates(
    stock_list,
    cache_dir: str,
    start_date: str,
    end_date: str,
    min_consec: int,
) -> Dict[str, List[Dict[str, object]]]:
    candidates: Dict[str, List[Dict[str, object]]] = {}
    for item in stock_list:
        rows = _load_rows(cache_dir, item.market, item.code)
        if not rows or len(rows) < 60:
            continue
        dates = [r.date for r in rows]
        limit = _limit_rate(item.code, item.name)
        consec = 0
        for i in range(1, len(rows) - 1):
            if dates[i] < start_date or dates[i] > end_date:
                # still update streak for future days
                prev_close = rows[i - 1].close
                if _is_limit_up(prev_close, rows[i].close, limit):
                    consec += 1
                else:
                    consec = 0
                continue

            prev_close = rows[i - 1].close
            if _is_limit_up(prev_close, rows[i].close, limit):
                consec += 1
            else:
                consec = 0
            if consec < min_consec:
                continue
            buy_idx = i + 1
            if buy_idx >= len(rows):
                continue
            buy_date = rows[buy_idx].date
            if buy_date < start_date or buy_date > end_date:
                continue
            # next board must be limit-up and not one-word at open
            prev_close2 = rows[buy_idx - 1].close
            if not _is_limit_up(prev_close2, rows[buy_idx].close, limit):
                continue
            if _is_one_word(
                rows[buy_idx].open,
                rows[buy_idx].high,
                rows[buy_idx].low,
                rows[buy_idx].close,
                prev_close2,
                limit,
            ):
                continue
            # must be buyable at open (not open at limit)
            if rows[buy_idx].open >= prev_close2 * (1 + limit) * 0.998:
                continue
            # rank by previous day amount (known before buy)
            amount = rows[i].amount
            if amount <= 0:
                amount = rows[i].close * rows[i].volume
            candidates.setdefault(buy_date, []).append(
                {
                    "code": item.code,
                    "name": item.name,
                    "buy_idx": buy_idx,
                    "signal_date": dates[i],
                    "amount": float(amount),
                }
            )
            consec = 0
    return candidates


def _find_exit(
    rows,
    buy_idx: int,
    limit: float,
    end_date: str,
) -> Tuple[int, float, str]:
    # T+1: start from next day
    for i in range(min(buy_idx + 1, len(rows) - 1), len(rows)):
        if rows[i].date > end_date:
            break
        prev_close = rows[i - 1].close
        if not _is_limit_up(prev_close, rows[i].close, limit):
            return i, rows[i].close, "first_non_limit"
    # fallback: last available close <= end_date
    for i in range(len(rows) - 1, buy_idx, -1):
        if rows[i].date <= end_date:
            return i, rows[i].close, "end"
    return buy_idx, rows[buy_idx].close, "end"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest 6th limit-up open buy, first non-limit close sell.")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--initial-cash", type=float, default=100000)
    parser.add_argument(
        "--consecutive",
        type=int,
        default=5,
        help="Consecutive limit-ups before buy (2 means buy 2nd board open).",
    )
    parser.add_argument("--cache-dir", default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"))
    parser.add_argument("--stock-list", default=os.path.join(GPT_DATA_DIR, "stock_list.csv"))
    parser.add_argument("--output-xlsx", default="data/results/limitup6_top1_2025.xlsx")
    args = parser.parse_args()

    stock_list = stock_items_from_list_csv(args.stock_list)
    if not stock_list:
        raise RuntimeError("股票列表为空")

    candidates = _build_candidates(
        stock_list, args.cache_dir, args.start_date, args.end_date, args.consecutive
    )
    all_dates = sorted(candidates.keys())

    cash = args.initial_cash
    last_exit_date: Optional[str] = None
    trades: List[Dict[str, object]] = []
    actions: List[Dict[str, object]] = []

    for date in all_dates:
        if last_exit_date and date <= last_exit_date:
            continue
        day_candidates = candidates.get(date, [])
        if not day_candidates:
            continue
        day_candidates.sort(key=lambda x: (-x["amount"], x["code"]))
        pick = day_candidates[0]

        code = pick["code"]
        name = pick["name"]
        signal_date = pick.get("signal_date", date)
        market = 1 if str(code).startswith("6") else 0
        rows = _load_rows(args.cache_dir, market, code)
        if not rows:
            continue
        buy_idx = pick["buy_idx"]
        if buy_idx >= len(rows):
            continue
        buy_date = rows[buy_idx].date
        if buy_date != date:
            continue
        entry = rows[buy_idx].open
        if entry <= 0:
            continue
        limit = _limit_rate(code, name)
        exit_idx, exit_price, reason = _find_exit(rows, buy_idx, limit, args.end_date)
        sell_date = rows[exit_idx].date
        shares = cash / entry
        cash = shares * exit_price
        ret_pct = (exit_price - entry) / entry * 100
        hold_days = (_parse_date(sell_date) - _parse_date(buy_date)).days
        trades.append(
            {
                "signal_date": signal_date,
                "code": code,
                "name": name,
                "buy_date": buy_date,
                "buy_price": round(entry, 4),
                "sell_date": sell_date,
                "sell_price": round(exit_price, 4),
                "reason": reason,
                "return_pct": round(ret_pct, 2),
                "hold_days": hold_days,
                "cash_after": round(cash, 2),
            }
        )
        actions.append({"date": buy_date, "action": "buy", "code": code, "name": name, "price": entry})
        actions.append({"date": sell_date, "action": "sell", "code": code, "name": name, "price": exit_price})
        last_exit_date = sell_date

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
