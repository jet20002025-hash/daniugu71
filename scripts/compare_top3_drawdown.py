import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.eastmoney import read_cached_kline, read_cached_kline_by_code


def _load_rows(cache_dir: str, cache_format: str, market: int, code: str):
    if cache_format == "secid":
        path = os.path.join(cache_dir, f"{market}_{code}.csv")
        return read_cached_kline(path)
    return read_cached_kline_by_code(cache_dir, code)


def _market_from_code(code: str) -> int:
    return 1 if str(code).startswith("6") else 0


def _build_price_maps(cache_dir: str, cache_format: str, codes: List[str]):
    maps = {}
    for code in sorted(set(codes)):
        market = _market_from_code(code)
        rows = _load_rows(cache_dir, cache_format, market, code)
        if not rows:
            continue
        dates = [r.date for r in rows]
        open_map = {r.date: float(r.open) for r in rows}
        close_map = {r.date: float(r.close) for r in rows}
        maps[code] = {"dates": dates, "open": open_map, "close": close_map}
    return maps


def _compute_equity(
    picks: pd.DataFrame,
    price_maps: Dict[str, Dict[str, object]],
    hold_days: int,
    label: str,
) -> Tuple[pd.DataFrame, float]:
    day_values: Dict[str, List[float]] = {}
    for _, row in picks.iterrows():
        code = str(row["code"]).zfill(6)
        buy_date = str(row.get("buy_date") or "")
        if not buy_date or code not in price_maps:
            continue
        data = price_maps[code]
        dates = data["dates"]
        open_map = data["open"]
        close_map = data["close"]
        if buy_date not in open_map:
            continue
        try:
            buy_idx = dates.index(buy_date)
        except ValueError:
            continue
        exit_idx = buy_idx + hold_days
        if exit_idx >= len(dates):
            continue
        entry = open_map[buy_date]
        if entry <= 0:
            continue
        for i in range(buy_idx, exit_idx + 1):
            d = dates[i]
            close = close_map.get(d)
            if close is None or close <= 0:
                continue
            day_values.setdefault(d, []).append(close / entry)

    all_days = sorted(day_values.keys())
    series = []
    last_value = 1.0
    for d in all_days:
        values = day_values.get(d, [])
        if values:
            value = float(np.mean(values))
            last_value = value
        else:
            value = last_value
        series.append({"date": d, f"{label}_value": value})

    df = pd.DataFrame(series)
    if df.empty:
        return df, 0.0
    peak = -1e9
    drawdowns = []
    for v in df[f"{label}_value"]:
        if v > peak:
            peak = v
        drawdowns.append((v - peak) / peak if peak > 0 else 0.0)
    df[f"{label}_drawdown"] = drawdowns
    max_dd = float(min(drawdowns)) if drawdowns else 0.0
    return df, max_dd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare max drawdown for rule vs ml Top3.")
    parser.add_argument("--rule-csv", required=True)
    parser.add_argument("--ml-csv", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-format", choices=["code", "secid"], default="secid")
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--output", default="data/results/mode3_top3_drawdown_compare.xlsx")
    args = parser.parse_args()

    rule = pd.read_csv(args.rule_csv)
    ml = pd.read_csv(args.ml_csv)
    for df in (rule, ml):
        df["code"] = df["code"].astype(str).str.zfill(6)

    codes = list(set(rule["code"]).union(set(ml["code"])))
    price_maps = _build_price_maps(args.cache_dir, args.cache_format, codes)

    rule_curve, rule_dd = _compute_equity(rule, price_maps, args.hold_days, "rule")
    ml_curve, ml_dd = _compute_equity(ml, price_maps, args.hold_days, "ml")

    curve = pd.merge(rule_curve, ml_curve, on="date", how="outer").sort_values("date")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        curve.to_excel(writer, index=False, sheet_name="curve")
        summary = pd.DataFrame(
            [
                {"model": "rule", "max_drawdown": rule_dd},
                {"model": "ml", "max_drawdown": ml_dd},
            ]
        )
        summary.to_excel(writer, index=False, sheet_name="summary")

    print("max_drawdown:")
    print(f"  rule: {rule_dd:.2%}")
    print(f"  ml:   {ml_dd:.2%}")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
