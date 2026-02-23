import argparse
import csv
import os
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.eastmoney import read_cached_kline, read_cached_kline_by_code, stock_items_from_list_csv
from app.ml_model import _detect_signals, MLConfig
from app.paths import GPT_DATA_DIR


def _max_high_in_window(rows, start_idx: int, end_idx: int) -> Tuple[float, Optional[str]]:
    max_high = -1.0
    max_date = None
    for i in range(start_idx, min(end_idx + 1, len(rows))):
        h = rows[i].high
        if h > max_high:
            max_high = h
            max_date = rows[i].date
    return max_high, max_date


def _rolling_max_with_index(values: List[float], window: int) -> Tuple[List[float], List[int]]:
    if window <= 0 or len(values) < window:
        return [], []
    dq: deque[int] = deque()
    max_vals: List[float] = []
    max_idx: List[int] = []
    for i, val in enumerate(values):
        while dq and values[dq[-1]] <= val:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - window:
            dq.popleft()
        if i >= window - 1:
            max_vals.append(values[dq[0]])
            max_idx.append(dq[0])
    return max_vals, max_idx


def _detect_cache_format(cache_dir: str) -> str:
    try:
        for name in os.listdir(cache_dir):
            if not name.endswith(".csv"):
                continue
            stem = name[:-4]
            if stem.startswith(("0_", "1_")):
                return "secid"
    except Exception:
        return "code"
    return "code"


def _load_rows(cache_dir: str, cache_format: str, market: int, code: str):
    if cache_format == "secid":
        path = os.path.join(cache_dir, f"{market}_{code}.csv")
        return read_cached_kline(path)
    return read_cached_kline_by_code(cache_dir, code)


def _parse_date(value: str) -> Optional[datetime.date]:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _in_range(date_str: str, start: Optional[str], end: Optional[str]) -> bool:
    if not start and not end:
        return True
    d = _parse_date(date_str)
    if d is None:
        return False
    if start:
        s = _parse_date(start)
        if s and d < s:
            return False
    if end:
        e = _parse_date(end)
        if e and d > e:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Find bull stocks that double within one month.")
    parser.add_argument(
        "--stock-list",
        default=os.path.join(GPT_DATA_DIR, "stock_list.csv"),
        help="Stock list CSV (code,name)",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"),
        help="Kline cache directory",
    )
    parser.add_argument(
        "--cache-format",
        choices=["auto", "code", "secid"],
        default="auto",
        help="Cache filename format (auto|code|secid)",
    )
    parser.add_argument("--hold-days", type=int, default=20, help="Window in trading days")
    parser.add_argument("--multiple", type=float, default=2.0, help="Target multiple")
    parser.add_argument(
        "--signal-type",
        choices=["aggressive", "relaxed"],
        default="relaxed",
        help="Signal detection mode",
    )
    parser.add_argument(
        "--mode",
        choices=["signal", "all"],
        default="signal",
        help="signal=use buy-point signals; all=scan every day without signals",
    )
    parser.add_argument("--year", type=int, default=None, help="Filter signal dates by year")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--buy-offset", type=int, default=1, help="Buy offset days from signal day")
    parser.add_argument(
        "--output",
        default="data/results/bull_signals.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    if args.year and not args.start_date and not args.end_date:
        args.start_date = f"{args.year}-01-01"
        args.end_date = f"{args.year}-12-31"

    cache_format = args.cache_format
    if cache_format == "auto":
        cache_format = _detect_cache_format(args.cache_dir)

    stock_list = stock_items_from_list_csv(args.stock_list)
    if not stock_list:
        raise RuntimeError("股票列表为空，无法查找大牛股。")

    config = MLConfig(signal_type=args.signal_type)
    results: List[Dict[str, object]] = []

    for item in stock_list:
        rows = _load_rows(args.cache_dir, cache_format, item.market, item.code)
        if not rows or len(rows) < 80:
            continue

        signals: List[Tuple[int, int, int]] = []
        if args.mode == "signal":
            signals = _detect_signals(rows, config.signal_type)
            if not signals:
                continue
        else:
            # No signal mode: treat every day in range as a "signal day"
            for i, row in enumerate(rows):
                if _in_range(row.date, args.start_date, args.end_date):
                    signals.append((i, i, i))
            if not signals:
                continue

        highs = [r.high for r in rows]
        window = args.hold_days + 1
        max_vals, max_idx = _rolling_max_with_index(highs, window)

        for signal_idx, shake_idx, stop_idx in signals:
            if args.mode == "signal" and not _in_range(
                rows[signal_idx].date, args.start_date, args.end_date
            ):
                continue
            buy_idx = signal_idx + max(args.buy_offset, 0)
            exit_idx = buy_idx + args.hold_days
            if buy_idx < 0 or exit_idx >= len(rows):
                continue
            buy_price = rows[buy_idx].open
            if buy_price <= 0:
                continue
            if buy_idx >= len(max_vals):
                continue
            max_high = max_vals[buy_idx]
            hit_idx = max_idx[buy_idx]
            hit_date = rows[hit_idx].date if 0 <= hit_idx < len(rows) else None
            if max_high <= 0:
                continue
            multiple = max_high / buy_price
            if multiple >= args.multiple:
                results.append(
                    {
                        "code": item.code,
                        "name": item.name,
                        "signal_date": rows[signal_idx].date,
                        "buy_date": rows[buy_idx].date,
                        "buy_price": round(buy_price, 4),
                        "max_high": round(max_high, 4),
                        "hit_date": hit_date or "",
                        "multiple": round(multiple, 4),
                    }
                )

    results.sort(key=lambda r: (-r["multiple"], r["signal_date"], r["code"]))

    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "code",
                "name",
                "signal_date",
                "buy_date",
                "buy_price",
                "max_high",
                "hit_date",
                "multiple",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"大牛股信号数量: {len(results)}")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
