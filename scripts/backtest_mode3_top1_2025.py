#!/usr/bin/env python3
"""
mode3 选股回测：2025年，只买 top1，次日开盘买入
- 10% 止损
- 破 20 日均线卖出
- 主板 100 股起、创业板 200 股起、科创板任意
- 10 万起步，当天不能买卖（先卖后买）
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
    """主板100股、创业板200股、科创板任意(1股)"""
    c = str(code).strip()
    if c.startswith("688"):
        return 1
    if c.startswith("300"):
        return 200
    return 100  # 主板、北交所等


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
    """按最小交易单位计算可买股数"""
    if price <= 0:
        return 0
    raw = int(cash / price)
    return max(0, (raw // min_lot) * min_lot)


def _find_exit(
    rows,
    buy_idx: int,
    stop_loss_price: float,
    ma: np.ndarray,
    end_date: Optional[str],
    ma_period: int = 20,
) -> Tuple[int, float, str]:
    """找到卖出点：10%止损 或 破N日均线"""
    reason_suffix = f"ma{ma_period}_break"
    exit_idx = buy_idx
    exit_price = rows[buy_idx].close
    reason = "end"
    for i in range(buy_idx + 1, len(rows)):
        if end_date and rows[i].date > end_date:
            break
        low = rows[i].low
        high = rows[i].high
        close = rows[i].close
        if low <= stop_loss_price:
            exit_idx = i
            exit_price = stop_loss_price
            reason = "stop_loss_10pct"
            break
        if not np.isnan(ma[i]) and ma[i] > 0 and close < ma[i]:
            exit_idx = i
            exit_price = close
            reason = reason_suffix
            break
        exit_idx = i
        exit_price = close
        reason = "end"
    # 若超过 end_date，取 end_date 前最后一天
    if end_date and rows[exit_idx].date > end_date:
        for j in range(exit_idx - 1, buy_idx - 1, -1):
            if rows[j].date <= end_date:
                exit_idx = j
                exit_price = rows[j].close
                reason = "end_date"
                break
    return exit_idx, exit_price, reason


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mode3 top1 回测：次日开盘买、10%止损、破MA20卖、手数规则"
    )
    parser.add_argument("--picks-csv", default="data/results/mode3_2025_top3.csv")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"),
    )
    parser.add_argument("--initial-cash", type=float, default=100000)
    parser.add_argument("--stop-loss", type=float, default=0.10)
    parser.add_argument("--ma-exit", type=int, default=20, help="破N日均线卖出，默认20")
    parser.add_argument("--buy-at-close", action="store_true", help="信号日收盘价买入（默认次日开盘）")
    parser.add_argument("--output-xlsx", default=None, help="输出路径，默认根据ma-exit自动命名")
    args = parser.parse_args()
    if args.output_xlsx is None:
        suffix = "_close" if args.buy_at_close else ""
        args.output_xlsx = f"data/results/mode3_top1_backtest_2025_ma{args.ma_exit}{suffix}.xlsx"

    picks = pd.read_csv(args.picks_csv)
    if "date" not in picks.columns and "signal_date" in picks.columns:
        picks["date"] = picks["signal_date"]
    picks["date"] = picks["date"].astype(str)
    picks["buy_date"] = picks["buy_date"].astype(str)
    picks["code"] = picks["code"].astype(str).str.zfill(6)
    picks = picks.sort_values(["date", "score"], ascending=[True, False])
    # 每日按 score 排序，取前几个作为候选（top1 无数据时尝试 top2/top3）
    daily_candidates = picks.groupby("date").apply(
        lambda g: g.head(5).to_dict("records"), include_groups=False
    ).to_dict()

    start_dt = _parse_date(args.start_date)
    end_dt = _parse_date(args.end_date)

    trades: List[Dict] = []
    actions: List[Dict] = []
    cash = args.initial_cash
    position: Optional[Dict] = None  # {code, name, shares, buy_price, buy_date, signal_date}
    sold_today: Optional[str] = None  # 当天已卖出，不能买

    for signal_date in sorted(daily_candidates.keys()):
        if start_dt and _parse_date(signal_date) and _parse_date(signal_date) < start_dt:
            continue
        if end_dt and _parse_date(signal_date) and _parse_date(signal_date) > end_dt:
            continue

        candidates = daily_candidates[signal_date]
        # 只考虑交易日在区间内的（收盘买=信号日，开盘买=次日）
        def _trade_date(c):
            if args.buy_at_close:
                return str(c.get("date", c.get("signal_date", signal_date))).strip()
            return str(c.get("buy_date", "")).strip()
        if end_dt:
            candidates = [c for c in candidates if _parse_date(_trade_date(c)) and _parse_date(_trade_date(c)) <= end_dt]
        if start_dt:
            candidates = [c for c in candidates if _parse_date(_trade_date(c)) and _parse_date(_trade_date(c)) >= start_dt]
        row = None
        for c in candidates:
            sig_d = str(c.get("date", c.get("signal_date", signal_date))).strip()
            buy_date = str(c.get("buy_date", "")).strip()
            code = str(c.get("code", "")).strip().zfill(6)
            if not code:
                continue
            # 收盘价买入：用 signal_date 作为买入日；否则用 buy_date（次日）
            trade_date = sig_d if args.buy_at_close else buy_date
            if not trade_date:
                continue
            if sold_today and trade_date == sold_today:
                continue
            kline = _load_rows(args.cache_dir, code)
            if kline and len(kline) >= 80 and trade_date in [r.date for r in kline]:
                row = c
                row["code"] = code
                row["buy_date"] = trade_date
                row["signal_date"] = sig_d
                break
        if row is None:
            continue

        buy_date = row["buy_date"]  # 实际买入日（收盘买=信号日，开盘买=次日）
        code = row["code"]
        name = row.get("name", code)

        if sold_today and buy_date == sold_today:
            continue

        # 若有持仓，先检查是否应卖出
        if position:
            pos_code = position["code"]
            pos_rows = _load_rows(args.cache_dir, pos_code)
            if not pos_rows or len(pos_rows) < 80:
                position = None
                continue
            dates = [r.date for r in pos_rows]
            buy_idx_pos = None
            for i, d in enumerate(dates):
                if d == position["buy_date"]:
                    buy_idx_pos = i
                    break
            if buy_idx_pos is None:
                position = None
                continue
            close = np.array([r.close for r in pos_rows], dtype=float)
            ma = _moving_mean(close, args.ma_exit)
            stop_price = position["buy_price"] * (1 - args.stop_loss)
            exit_idx, exit_price, reason = _find_exit(
                pos_rows, buy_idx_pos, stop_price, ma, args.end_date, args.ma_exit
            )
            exit_date = pos_rows[exit_idx].date

            if exit_date > buy_date:
                # 持仓要持有到 exit_date 之后，无法在 buy_date 买新标的，跳过本信号
                continue

            # 执行卖出
            sell_value = position["shares"] * exit_price
            cash += sell_value
            ret_pct = (exit_price - position["buy_price"]) / position["buy_price"] * 100
            hold_days = (
                (_parse_date(exit_date) - _parse_date(position["buy_date"])).days
                if _parse_date(exit_date) and _parse_date(position["buy_date"])
                else 0
            )
            trades.append(
                {
                    "signal_date": position.get("signal_date", ""),
                    "code": str(pos_code).zfill(6),
                    "name": position["name"],
                    "buy_date": position["buy_date"],
                    "buy_price": round(position["buy_price"], 4),
                    "sell_date": exit_date,
                    "sell_price": round(exit_price, 4),
                    "reason": reason,
                    "return_pct": round(ret_pct, 2),
                    "hold_days": hold_days,
                    "shares": position["shares"],
                    "cash_after": round(cash, 2),
                }
            )
            actions.append(
                {"date": exit_date, "action": "sell", "code": pos_code, "name": position["name"], "price": exit_price}
            )
            sold_today = exit_date
            position = None
            if exit_date == buy_date:
                continue  # 当天已卖，不能买

        # 买入（持有，不在此处卖出）
        kline = _load_rows(args.cache_dir, code)
        if not kline or len(kline) < 80:
            continue
        dates = [r.date for r in kline]
        if buy_date not in dates:
            continue
        buy_idx = dates.index(buy_date)
        entry_price = kline[buy_idx].close if args.buy_at_close else kline[buy_idx].open
        if entry_price <= 0:
            continue

        min_lot = _min_lot(code)
        shares = _calc_shares(cash, entry_price, min_lot)
        if shares < min_lot:
            continue

        cost = shares * entry_price
        cash -= cost

        close = np.array([r.close for r in kline], dtype=float)
        ma = _moving_mean(close, args.ma_exit)
        stop_price = entry_price * (1 - args.stop_loss)
        exit_idx, exit_price, reason = _find_exit(
            kline, buy_idx, stop_price, ma, args.end_date, args.ma_exit
        )
        exit_date = kline[exit_idx].date

        position = {
            "code": str(code).zfill(6),
            "name": name,
            "shares": shares,
            "buy_price": entry_price,
            "buy_date": buy_date,
            "signal_date": signal_date,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "reason": reason,
        }
        actions.append({"date": buy_date, "action": "buy", "code": code, "name": name, "price": entry_price})

    # 若最后仍有持仓，平仓
    if position:
        sell_value = position["shares"] * position["exit_price"]
        cash += sell_value
        ret_pct = (position["exit_price"] - position["buy_price"]) / position["buy_price"] * 100
        hold_days = (
            (_parse_date(position["exit_date"]) - _parse_date(position["buy_date"])).days
            if _parse_date(position["exit_date"]) and _parse_date(position["buy_date"])
            else 0
        )
        trades.append(
            {
                "signal_date": position["signal_date"],
                "code": str(position["code"]).zfill(6),
                "name": position["name"],
                "buy_date": position["buy_date"],
                "buy_price": round(position["buy_price"], 4),
                "sell_date": position["exit_date"],
                "sell_price": round(position["exit_price"], 4),
                "reason": position["reason"],
                "return_pct": round(ret_pct, 2),
                "hold_days": hold_days,
                "shares": position["shares"],
                "cash_after": round(cash, 2),
            }
        )
        actions.append({
            "date": position["exit_date"],
            "action": "sell",
            "code": position["code"],
            "name": position["name"],
            "price": position["exit_price"],
        })

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
        df_actions = pd.DataFrame(actions)
        if not df_actions.empty and "date" in df_actions.columns:
            df_actions.sort_values("date").to_excel(writer, index=False, sheet_name="actions")
        else:
            df_actions.to_excel(writer, index=False, sheet_name="actions")

    print("回测完成:", args.start_date, "~", args.end_date)
    print("期初资金:", args.initial_cash, "期末资金:", round(cash, 2))
    print("收益率:", round((cash / args.initial_cash - 1) * 100, 2), "%")
    print("交易次数:", len(trades))
    print("输出:", args.output_xlsx)


if __name__ == "__main__":
    main()
