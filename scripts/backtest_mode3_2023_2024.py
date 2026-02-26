#!/usr/bin/env python3
"""
71倍模型(mode3) 2023、2024 年回测：次日开盘买、10%止损、破MA10卖（与「最新模型」一致）。
可选：150亿市值限制。输出：统计结果 + 详细交易记录表，表头中文。
"""
import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.eastmoney import read_cached_kline_by_code
from app.paths import GPT_DATA_DIR


def _load_market_caps(path: str) -> Dict[str, float]:
    if not path or not os.path.exists(path):
        return {}
    mapping: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = str(row.get("code", "")).strip().zfill(6)
            cap_value = row.get("total_cap") or row.get("market_cap")
            try:
                cap = float(cap_value) if cap_value else 0.0
            except Exception:
                continue
            if cap > 0:
                mapping[code] = cap
    return mapping


def _filter_picks_by_cap(
    picks: pd.DataFrame, market_caps: Dict[str, float], cap_limit_yi: float
) -> pd.DataFrame:
    """保留市值<=cap_limit_yi亿的标的。缺市值数据的剔除。"""
    if not market_caps or cap_limit_yi <= 0:
        return picks
    limit = cap_limit_yi * 1e8  # 亿 -> 元
    codes_ok = {c for c, v in market_caps.items() if v <= limit}
    picks = picks.copy()
    picks["code"] = picks["code"].astype(str).str.zfill(6)
    return picks[picks["code"].isin(codes_ok)].copy()


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


def _reason_stop_loss(stop_loss: float) -> str:
    if stop_loss and abs(stop_loss - 0.05) < 1e-6:
        return "stop_loss_5pct"
    return "stop_loss_10pct"


def _find_exit_take_profit_only(
    rows,
    buy_idx: int,
    buy_price: float,
    end_date: Optional[str],
    take_profit_pct: float = 0.03,
) -> Tuple[int, float, str]:
    """仅盈利止盈：达到 take_profit_pct（如 3%）即卖，不盈利则持有至 end_date；仅用当日及历史数据。"""
    target = buy_price * (1 + take_profit_pct)
    exit_idx = buy_idx
    exit_price = rows[buy_idx].close
    reason = "end"
    for i in range(buy_idx + 1, len(rows)):
        if end_date and rows[i].date > end_date:
            break
        high = rows[i].high
        close = rows[i].close
        if high >= target:
            exit_idx = i
            exit_price = target
            reason = "take_profit_3pct"
            break
        exit_idx = i
        exit_price = close
        reason = "end"
    if end_date and rows[exit_idx].date > end_date:
        for j in range(exit_idx - 1, buy_idx - 1, -1):
            if rows[j].date <= end_date:
                exit_idx = j
                exit_price = rows[j].close
                reason = "end_date"
                break
    return exit_idx, exit_price, reason


def _find_exit(
    rows,
    buy_idx: int,
    stop_loss_price: float,
    ma: np.ndarray,
    end_date: Optional[str],
    ma_period: int = 20,
    reason_stop_loss: str = "stop_loss_10pct",
) -> Tuple[int, float, str]:
    """卖出判断仅用当日及历史数据：逐日检查当日 low/close 与 MA，禁止使用未来数据。"""
    reason_suffix = f"ma{ma_period}_break"
    exit_idx = buy_idx
    exit_price = rows[buy_idx].close
    reason = "end"
    for i in range(buy_idx + 1, len(rows)):
        if end_date and rows[i].date > end_date:
            break
        low, close = rows[i].low, rows[i].close
        if low <= stop_loss_price:
            exit_idx = i
            exit_price = stop_loss_price
            reason = reason_stop_loss
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
                reason = "end_date"
                break
    return exit_idx, exit_price, reason


def run_backtest(
    picks: pd.DataFrame,
    cache_dir: str,
    start_date: str,
    end_date: str,
    initial_cash: float = 100000.0,
    stop_loss: float = 0.10,
    ma_exit: int = 10,
    buy_at_close: bool = False,
    use_stop_loss: bool = True,
    take_profit_only: Optional[float] = None,
) -> Tuple[List[Dict], float]:
    """执行单年回测，返回 (交易列表, 期末资金)。"""
    if "date" not in picks.columns and "signal_date" in picks.columns:
        picks = picks.copy()
        picks["date"] = picks["signal_date"]
    picks = picks.copy()
    picks["date"] = picks["date"].astype(str)
    picks["buy_date"] = picks["buy_date"].astype(str)
    picks["code"] = picks["code"].astype(str).str.zfill(6)
    sort_cols = ["date", "score"]
    ascending = [True, False]
    if "buy_point_score" in picks.columns:
        picks["buy_point_score"] = picks["buy_point_score"].fillna(0).astype(int)
        sort_cols.append("buy_point_score")
        ascending.append(False)
    picks = picks.sort_values(sort_cols, ascending=ascending)
    daily_candidates = picks.groupby("date").apply(
        lambda g: g.head(5).to_dict("records"), include_groups=False
    ).to_dict()

    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    trades: List[Dict] = []
    cash = initial_cash
    position: Optional[Dict] = None
    sold_today: Optional[str] = None

    for signal_date in sorted(daily_candidates.keys()):
        if start_dt and _parse_date(signal_date) and _parse_date(signal_date) < start_dt:
            continue
        if end_dt and _parse_date(signal_date) and _parse_date(signal_date) > end_dt:
            continue

        candidates = daily_candidates[signal_date]

        def _trade_date(c):
            return (
                str(c.get("date", c.get("signal_date", signal_date))).strip()
                if buy_at_close
                else str(c.get("buy_date", "")).strip()
            )

        if end_dt:
            candidates = [
                c
                for c in candidates
                if _parse_date(_trade_date(c)) and _parse_date(_trade_date(c)) <= end_dt
            ]
        if start_dt:
            candidates = [
                c
                for c in candidates
                if _parse_date(_trade_date(c)) and _parse_date(_trade_date(c)) >= start_dt
            ]
        row = None
        for c in candidates:
            sig_d = str(c.get("date", c.get("signal_date", signal_date))).strip()
            buy_date = str(c.get("buy_date", "")).strip()
            code = str(c.get("code", "")).strip().zfill(6)
            if not code:
                continue
            trade_date = sig_d if buy_at_close else buy_date
            if not trade_date or (sold_today and trade_date == sold_today):
                continue
            kline = _load_rows(cache_dir, code)
            if kline and len(kline) >= 80 and trade_date in [r.date for r in kline]:
                row = dict(c)
                row["code"] = code
                row["buy_date"] = trade_date
                row["signal_date"] = sig_d
                break
        if row is None:
            continue

        buy_date = row["buy_date"]
        code = row["code"]
        name = row.get("name", code)
        if sold_today and buy_date == sold_today:
            continue

        if position:
            pos_code = position["code"]
            pos_rows = _load_rows(cache_dir, pos_code)
            if not pos_rows or len(pos_rows) < 80:
                position = None
                continue
            dates = [r.date for r in pos_rows]
            buy_idx_pos = next(
                (i for i, d in enumerate(dates) if d == position["buy_date"]), None
            )
            if buy_idx_pos is None:
                position = None
                continue
            if take_profit_only is not None:
                exit_idx, exit_price, reason = _find_exit_take_profit_only(
                    pos_rows, buy_idx_pos, position["buy_price"], end_date, take_profit_only
                )
            else:
                close = np.array([r.close for r in pos_rows], dtype=float)
                ma = _moving_mean(close, ma_exit)
                stop_price = position["buy_price"] * (1 - stop_loss) if use_stop_loss else 0.0
                exit_idx, exit_price, reason = _find_exit(
                    pos_rows, buy_idx_pos, stop_price, ma, end_date, ma_exit,
                    reason_stop_loss=_reason_stop_loss(stop_loss),
                )
            exit_date = pos_rows[exit_idx].date
            if exit_date > buy_date:
                continue
            sell_value = position["shares"] * exit_price
            cash += sell_value
            ret_pct = (
                (exit_price - position["buy_price"]) / position["buy_price"] * 100
            )
            hold_days = (
                (
                    _parse_date(exit_date) - _parse_date(position["buy_date"])
                ).days
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
            sold_today = exit_date
            position = None
            if exit_date == buy_date:
                continue

        kline = _load_rows(cache_dir, code)
        if not kline or len(kline) < 80 or buy_date not in [r.date for r in kline]:
            continue
        dates = [r.date for r in kline]
        buy_idx = dates.index(buy_date)
        entry_price = (
            kline[buy_idx].close if buy_at_close else kline[buy_idx].open
        )
        if entry_price <= 0:
            continue
        min_lot = _min_lot(code)
        shares = _calc_shares(cash, entry_price, min_lot)
        if shares < min_lot:
            continue
        cash -= shares * entry_price
        if take_profit_only is not None:
            exit_idx, exit_price, reason = _find_exit_take_profit_only(
                kline, buy_idx, entry_price, end_date, take_profit_only
            )
        else:
            close = np.array([r.close for r in kline], dtype=float)
            ma = _moving_mean(close, ma_exit)
            sl_price = entry_price * (1 - stop_loss) if use_stop_loss else 0.0
            exit_idx, exit_price, reason = _find_exit(
                kline, buy_idx, sl_price, ma, end_date, ma_exit,
                reason_stop_loss=_reason_stop_loss(stop_loss),
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

    if position:
        cash += position["shares"] * position["exit_price"]
        ret_pct = (
            (position["exit_price"] - position["buy_price"])
            / position["buy_price"]
            * 100
        )
        hold_days = (
            (
                _parse_date(position["exit_date"])
                - _parse_date(position["buy_date"])
            ).days
            if _parse_date(position["exit_date"])
            and _parse_date(position["buy_date"])
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

    return trades, cash


def _compute_stats(
    trades: List[Dict], initial_cash: float, final_cash: float, year: int
) -> List[Dict]:
    """根据交易记录计算统计指标。"""
    if not trades:
        return [
            {"年份": year, "指标": "期初资金(元)", "数值": initial_cash},
            {"年份": year, "指标": "期末资金(元)", "数值": final_cash},
            {"年份": year, "指标": "总收益率(%)", "数值": 0},
            {"年份": year, "指标": "交易次数", "数值": 0},
            {"年份": year, "指标": "盈利次数", "数值": 0},
            {"年份": year, "指标": "亏损次数", "数值": 0},
            {"年份": year, "指标": "胜率(%)", "数值": 0},
            {"年份": year, "指标": "平均单笔收益率(%)", "数值": 0},
            {"年份": year, "指标": "平均持仓天数", "数值": 0},
            {"年份": year, "指标": "最大单笔盈利(%)", "数值": 0},
            {"年份": year, "指标": "最大单笔亏损(%)", "数值": 0},
            {"年份": year, "指标": "最大回撤(%)", "数值": 0},
        ]
    returns = [t["return_pct"] for t in trades]
    win_count = sum(1 for r in returns if r > 0)
    loss_count = sum(1 for r in returns if r <= 0)
    total_ret = (final_cash / initial_cash - 1) * 100
    avg_ret = sum(returns) / len(returns) if returns else 0
    hold_days = [t["hold_days"] for t in trades]
    avg_hold = sum(hold_days) / len(hold_days) if hold_days else 0
    max_win = max(returns) if returns else 0
    max_loss = min(returns) if returns else 0
    win_rate = win_count / len(trades) * 100 if trades else 0

    # 最大回撤：按资金曲线
    equity = [initial_cash]
    for t in trades:
        equity.append(t["cash_after"])
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        if peak > 0:
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd

    return [
        {"年份": year, "指标": "期初资金(元)", "数值": round(initial_cash, 2)},
        {"年份": year, "指标": "期末资金(元)", "数值": round(final_cash, 2)},
        {"年份": year, "指标": "总收益率(%)", "数值": round(total_ret, 2)},
        {"年份": year, "指标": "交易次数", "数值": len(trades)},
        {"年份": year, "指标": "盈利次数", "数值": win_count},
        {"年份": year, "指标": "亏损次数", "数值": loss_count},
        {"年份": year, "指标": "胜率(%)", "数值": round(win_rate, 2)},
        {"年份": year, "指标": "平均单笔收益率(%)", "数值": round(avg_ret, 2)},
        {"年份": year, "指标": "平均持仓天数", "数值": round(avg_hold, 2)},
        {"年份": year, "指标": "最大单笔盈利(%)", "数值": round(max_win, 2)},
        {"年份": year, "指标": "最大单笔亏损(%)", "数值": round(max_loss, 2)},
        {"年份": year, "指标": "最大回撤(%)", "数值": round(max_dd, 2)},
    ]


# 交易记录表 中英字段对应
TRADE_CN = {
    "signal_date": "信号日",
    "code": "代码",
    "name": "名称",
    "buy_date": "买入日",
    "buy_price": "买入价",
    "sell_date": "卖出日",
    "sell_price": "卖出价",
    "reason": "卖出原因",
    "return_pct": "收益率(%)",
    "hold_days": "持仓天数",
    "shares": "股数",
    "cash_after": "期末现金",
}

REASON_CN = {
    "stop_loss_10pct": "10%止损",
    "stop_loss_5pct": "5%止损",
    "take_profit_3pct": "盈利3%止盈",
    "ma20_break": "破20日均线",
    "ma10_break": "破10日均线",
    "ma5_break": "破5日均线",
    "end": "持仓结束",
    "end_date": "回测截止",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="71倍模型 2023/2024 回测，输出统计结果与中文表头交易记录"
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"),
    )
    parser.add_argument(
        "--picks-2023",
        default="data/results/mode3_2023_picks.csv",
        help="2023年选股CSV",
    )
    parser.add_argument(
        "--picks-2024",
        default="data/results/mode3_2024_picks.csv",
        help="2024年选股CSV",
    )
    parser.add_argument(
        "--picks-2025",
        default="data/results/mode3_2025_picks.csv",
        help="2025年选股CSV",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=None,
        help="只回测指定年份，如 --years 2025；默认 2023 2024 2025",
    )
    parser.add_argument("--initial-cash", type=float, default=100000)
    parser.add_argument("--stop-loss", type=float, default=0.10)
    parser.add_argument("--no-stop-loss", action="store_true", help="关闭10%%止损，仅破MA卖或期末平仓（用于对比收益差异）")
    parser.add_argument("--ma-exit", type=int, default=10, help="破N日均线卖出，默认10（与最新模型一致）")
    parser.add_argument(
        "--market-cap",
        default=os.path.join(GPT_DATA_DIR, "market_cap.csv"),
        help="市值缓存CSV",
    )
    parser.add_argument(
        "--max-cap",
        type=float,
        default=150.0,
        help="市值上限（亿），默认150；0表示不限制",
    )
    parser.add_argument(
        "--output",
        default="data/results/mode3_backtest_2023_2024.xlsx",
        help="输出Excel路径",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    market_caps = _load_market_caps(args.market_cap)
    if args.max_cap > 0 and not market_caps:
        print("警告：已设市值上限但未找到市值文件，将不进行市值过滤。")

    year_config = [
        (2023, args.picks_2023, "2023-01-01", "2023-12-31"),
        (2024, args.picks_2024, "2024-01-01", "2024-12-31"),
        (2025, args.picks_2025, "2025-01-01", "2025-12-31"),
    ]
    if args.years is not None:
        year_config = [c for c in year_config if c[0] in args.years]
    all_stats: List[Dict] = []
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        for year, picks_path, start, end in year_config:
            if not os.path.exists(picks_path):
                print(f"跳过 {year}：未找到 {picks_path}")
                continue
            picks = pd.read_csv(picks_path)
            if "score" not in picks.columns:
                picks["score"] = 100
            if args.max_cap > 0 and market_caps:
                n_before = len(picks)
                picks = _filter_picks_by_cap(picks, market_caps, args.max_cap)
                print(f"{year} 市值≤{args.max_cap}亿 过滤: {n_before} -> {len(picks)} 条")
            trades, final_cash = run_backtest(
                picks,
                args.cache_dir,
                start,
                end,
                initial_cash=args.initial_cash,
                stop_loss=args.stop_loss,
                ma_exit=args.ma_exit,
                use_stop_loss=not args.no_stop_loss,
            )
            stats = _compute_stats(
                trades, args.initial_cash, final_cash, year
            )
            all_stats.extend(stats)

            # 统计结果表：指标、数值两列
            stats_df = pd.DataFrame(stats)[["指标", "数值"]]
            stats_df.to_excel(writer, index=False, sheet_name=f"{year}统计结果")

            # 交易记录表：中文表头，卖出原因转中文
            if trades:
                df_trades = pd.DataFrame(trades)
                df_trades["reason"] = df_trades["reason"].map(
                    lambda x: REASON_CN.get(x, x)
                )
                # 按原始英文字段选出存在列，再映射为中文表头
                cn_cols = [TRADE_CN[k] for k in TRADE_CN if k in df_trades.columns]
                df_trades_cn = df_trades.rename(columns=TRADE_CN)
                df_trades_cn[cn_cols].to_excel(
                    writer, index=False, sheet_name=f"{year}交易记录"
                )
            else:
                pd.DataFrame(columns=list(TRADE_CN.values())).to_excel(
                    writer, index=False, sheet_name=f"{year}交易记录"
                )

            print(f"{year} 回测: 交易 {len(trades)} 笔, 期末资金 {final_cash:.2f}")

        # 汇总统计：两年并排
        if all_stats:
            summary_df = pd.DataFrame(all_stats)
            summary_pivot = summary_df.pivot(
                index="指标", columns="年份", values="数值"
            ).reset_index()
            summary_pivot.to_excel(
                writer, index=False, sheet_name="汇总统计"
            )

    print("已写入:", args.output)


if __name__ == "__main__":
    main()
