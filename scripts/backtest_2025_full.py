#!/usr/bin/env python3
"""
2025年 mode3 回测：10万本金，每日 top1，次日开盘买
- 10% 止损
- 破 MA10 止盈卖出
- 主板100股、创业板200股、科创200股、北交所200股
- T+1：卖出后才能买新，当天不能买卖
"""
import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.eastmoney import _market_from_code, read_cached_kline_by_code, read_cached_kline_by_market_code
from app.paths import GPT_DATA_DIR


def _parse_date(value: str) -> Optional[datetime.date]:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _min_lot(code: str) -> int:
    """主板100、创业板200、科创200、北交所200"""
    c = str(code).strip()
    if c.startswith("688"):  # 科创板
        return 200
    if c.startswith("300"):  # 创业板
        return 200
    if c.startswith(("8", "9")) and len(c) == 6:  # 北交所
        return 200
    return 100  # 主板


def _load_rows(cache_dir: str, code: str, use_secid: bool = False):
    if use_secid:
        market = _market_from_code(code)
        return read_cached_kline_by_market_code(cache_dir, market, code)
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


def _find_exit(
    rows,
    buy_idx: int,
    stop_loss_price: float,
    ma: np.ndarray,
    end_date: Optional[str],
    ma_period: int = 10,
) -> Tuple[int, float, str]:
    """卖出：10%止损 或 破MA10"""
    reason_suffix = f"破MA{ma_period}"
    exit_idx = buy_idx
    exit_price = rows[buy_idx].close
    reason = "end"
    for i in range(buy_idx + 1, len(rows)):
        if end_date and rows[i].date > end_date:
            break
        low = rows[i].low
        close = rows[i].close
        if low <= stop_loss_price:
            exit_idx = i
            exit_price = stop_loss_price
            reason = "止损10%"
            break
        if not np.isnan(ma[i]) and ma[i] > 0 and close < ma[i]:
            exit_idx = i
            exit_price = close
            reason = reason_suffix
            break
        exit_idx = i
        exit_price = close
        reason = "end"
    if end_date and rows[exit_idx].date > end_date:
        for j in range(exit_idx - 1, buy_idx - 1, -1):
            if rows[j].date <= end_date:
                exit_idx = j
                exit_price = rows[j].close
                reason = "期末平仓"
                break
    return exit_idx, exit_price, reason


def _detect_cache_format(cache_dir: str) -> bool:
    """True=secid(east), False=code(tencent)"""
    if not os.path.exists(cache_dir):
        return False
    for name in os.listdir(cache_dir):
        if name.endswith(".csv") and "_" in name[:-4]:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="2025年 mode3 回测：top1、10%止损、破MA10卖")
    parser.add_argument("--picks-csv", default="data/results/mode3_2025_all.csv")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--cache-dir", default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"))
    parser.add_argument("--initial-cash", type=float, default=100000)
    parser.add_argument("--stop-loss", type=float, default=0.10)
    parser.add_argument("--ma-exit", type=int, default=10)
    parser.add_argument("--output-xlsx", default="data/results/mode3_2025_backtest.xlsx")
    parser.add_argument("--min-score", type=int, default=None, help="仅回测评分>=此值的个股，如100表示满分")
    args = parser.parse_args()

    use_secid = _detect_cache_format(args.cache_dir)

    picks = pd.read_csv(args.picks_csv)
    if "date" not in picks.columns and "signal_date" in picks.columns:
        picks["date"] = picks["signal_date"]
    if args.min_score is not None and "score" in picks.columns:
        picks = picks[picks["score"] >= args.min_score].copy()
        print(f"已过滤：仅保留 score>={args.min_score} 的个股，共 {len(picks)} 条")
    picks["date"] = picks["date"].astype(str)
    picks["buy_date"] = picks["buy_date"].astype(str)
    picks["code"] = picks["code"].astype(str).str.zfill(6)
    picks = picks.sort_values(["date", "score"], ascending=[True, False])
    daily_candidates = picks.groupby("date").apply(
        lambda g: g.head(5).to_dict("records"), include_groups=False
    ).to_dict()

    start_dt = _parse_date(args.start_date)
    end_dt = _parse_date(args.end_date)

    trades: List[Dict] = []
    actions: List[Dict] = []
    cash = args.initial_cash
    position: Optional[Dict] = None
    sold_today: Optional[str] = None

    for signal_date in sorted(daily_candidates.keys()):
        if start_dt and _parse_date(signal_date) and _parse_date(signal_date) < start_dt:
            continue
        if end_dt and _parse_date(signal_date) and _parse_date(signal_date) > end_dt:
            continue

        candidates = daily_candidates[signal_date]
        def _trade_date(c):
            return str(c.get("buy_date", "")).strip()
        if end_dt:
            candidates = [c for c in candidates if _parse_date(_trade_date(c)) and _parse_date(_trade_date(c)) <= end_dt]
        if start_dt:
            candidates = [c for c in candidates if _parse_date(_trade_date(c)) and _parse_date(_trade_date(c)) >= start_dt]

        row = None
        for c in candidates:
            buy_date = str(c.get("buy_date", "")).strip()
            code = str(c.get("code", "")).strip().zfill(6)
            if not code or not buy_date:
                continue
            if sold_today and buy_date == sold_today:
                continue
            kline = _load_rows(args.cache_dir, code, use_secid)
            if kline and len(kline) >= 80 and buy_date in [r.date for r in kline]:
                row = c
                row["code"] = code
                row["buy_date"] = buy_date
                row["signal_date"] = signal_date
                break
        if row is None:
            continue

        buy_date = row["buy_date"]
        code = row["code"]
        name = row.get("name", code)

        if position:
            pos_code = position["code"]
            pos_rows = _load_rows(args.cache_dir, pos_code, use_secid)
            if not pos_rows or len(pos_rows) < 80:
                position = None
                continue
            dates = [r.date for r in pos_rows]
            buy_idx_pos = next((i for i, d in enumerate(dates) if d == position["buy_date"]), None)
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
                continue

            sell_value = position["shares"] * exit_price
            cash += sell_value
            ret_pct = (exit_price - position["buy_price"]) / position["buy_price"] * 100
            hold_days = (
                (_parse_date(exit_date) - _parse_date(position["buy_date"])).days
                if _parse_date(exit_date) and _parse_date(position["buy_date"])
                else 0
            )
            profit = sell_value - position["shares"] * position["buy_price"]
            trades.append({
                "信号日": position.get("signal_date", ""),
                "代码": str(pos_code).zfill(6),
                "名称": position["name"],
                "买入日": position["buy_date"],
                "买入价": round(position["buy_price"], 4),
                "卖出日": exit_date,
                "卖出价": round(exit_price, 4),
                "卖出原因": reason,
                "收益率%": round(ret_pct, 2),
                "持仓天数": hold_days,
                "股数": position["shares"],
                "盈亏": round(profit, 2),
                "卖出后资金": round(cash, 2),
            })
            actions.append({"日期": exit_date, "操作": "卖出", "代码": pos_code, "名称": position["name"], "价格": exit_price})
            sold_today = exit_date
            position = None
            if exit_date == buy_date:
                continue

        kline = _load_rows(args.cache_dir, code, use_secid)
        if not kline or len(kline) < 80:
            continue
        dates = [r.date for r in kline]
        if buy_date not in dates:
            continue
        buy_idx = dates.index(buy_date)
        entry_price = kline[buy_idx].open
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
        actions.append({"日期": buy_date, "操作": "买入", "代码": code, "名称": name, "价格": entry_price})

    if position:
        sell_value = position["shares"] * position["exit_price"]
        cash += sell_value
        ret_pct = (position["exit_price"] - position["buy_price"]) / position["buy_price"] * 100
        hold_days = (
            (_parse_date(position["exit_date"]) - _parse_date(position["buy_date"])).days
            if _parse_date(position["exit_date"]) and _parse_date(position["buy_date"])
            else 0
        )
        profit = sell_value - position["shares"] * position["buy_price"]
        trades.append({
            "信号日": position["signal_date"],
            "代码": str(position["code"]).zfill(6),
            "名称": position["name"],
            "买入日": position["buy_date"],
            "买入价": round(position["buy_price"], 4),
            "卖出日": position["exit_date"],
            "卖出价": round(position["exit_price"], 4),
            "卖出原因": position["reason"],
            "收益率%": round(ret_pct, 2),
            "持仓天数": hold_days,
            "股数": position["shares"],
            "盈亏": round(profit, 2),
            "卖出后资金": round(cash, 2),
        })
        actions.append({"日期": position["exit_date"], "操作": "卖出", "代码": position["code"], "名称": position["name"], "价格": position["exit_price"]})

    return_pct = (cash / args.initial_cash - 1) * 100
    summary = pd.DataFrame([{
        "期初资金": args.initial_cash,
        "期末资金": round(cash, 2),
        "收益率%": round(return_pct, 2),
        "交易次数": len(trades),
    }])

    os.makedirs(os.path.dirname(args.output_xlsx), exist_ok=True)
    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="收益汇总")
        if trades:
            pd.DataFrame(trades).to_excel(writer, index=False, sheet_name="交易明细")
        df_actions = pd.DataFrame(actions)
        if not df_actions.empty:
            df_actions.sort_values("日期").to_excel(writer, index=False, sheet_name="买卖时间")
        else:
            df_actions.to_excel(writer, index=False, sheet_name="买卖时间")

    print("回测完成:", args.start_date, "~", args.end_date)
    print("期初资金:", args.initial_cash, "  期末资金:", round(cash, 2))
    print("收益率:", round(return_pct, 2), "%")
    print("交易次数:", len(trades))
    print("输出:", args.output_xlsx)


if __name__ == "__main__":
    main()
