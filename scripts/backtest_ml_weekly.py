import argparse
import csv
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from app.eastmoney import fetch_stock_list, get_kline_cached
from app.ml_model import MLConfig, load_model_bundle, _build_features, _detect_signals_aggressive


def _parse_date(value: str) -> datetime.date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _index_rows_by_date(rows) -> Dict[str, int]:
    return {row.date: idx for idx, row in enumerate(rows)}


def _next_trading_index(rows, idx: int, offset: int) -> int:
    target = idx + offset
    if target >= len(rows):
        return -1
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ML signals with 1-week returns.")
    parser.add_argument("--start-date", default="2026-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--hold-days", type=int, default=40, help="Holding days (trading days)")
    parser.add_argument("--top-n", type=int, default=10, help="Top N per day")
    parser.add_argument("--signal-lookback", type=int, default=0, help="Only signals on the same day if 0")
    parser.add_argument("--count", type=int, default=900, help="Kline rows to fetch per stock")
    args = parser.parse_args()

    start_date = _parse_date(args.start_date)

    model_path = "data/models/ml_model.pkl"
    meta_path = "data/models/ml_model_meta.json"
    model, meta = load_model_bundle(model_path, meta_path)
    if model is None or meta is None:
        raise RuntimeError("未找到ML模型，请先运行 scripts/train_ml.py")

    stock_list = fetch_stock_list()
    config = MLConfig(count=args.count)

    signals_by_date: Dict[str, List[Tuple[str, str, float, int]]] = {}
    rows_map: Dict[str, List] = {}
    index_map: Dict[str, Dict[str, int]] = {}

    for item in stock_list:
        rows = get_kline_cached(
            item.secid,
            cache_dir="data/kline_cache",
            count=config.count,
            max_age_days=config.cache_days,
            pause=0.0,
        )
        if not rows or len(rows) < config.min_history:
            continue
        rows_map[item.code] = rows
        index_map[item.code] = _index_rows_by_date(rows)

        signals = _detect_signals_aggressive(rows)
        for signal_idx, shake_idx, stop_idx in signals:
            signal_date = rows[signal_idx].date
            try:
                sd = _parse_date(signal_date)
            except Exception:
                continue
            if sd < start_date:
                continue

            feature_row = _build_features(
                rows=rows,
                signal_idx=signal_idx,
                shake_idx=shake_idx,
                stop_idx=stop_idx,
                market=item.market,
            )
            if feature_row is None:
                continue
            if not np.all(np.isfinite(feature_row)):
                continue
            proba = float(model.predict_proba([feature_row])[0][1])

            signals_by_date.setdefault(signal_date, []).append(
                (item.code, item.name, proba, signal_idx)
            )

    summary_rows = []
    detail_rows = []

    for signal_date, picks in sorted(signals_by_date.items()):
        sd = _parse_date(signal_date)
        if sd < start_date:
            continue

        # Only include signals on this day unless lookback > 0
        if args.signal_lookback > 0:
            pass

        picks_sorted = sorted(picks, key=lambda x: (-x[2], x[0]))[: args.top_n]

        returns = []
        for code, name, proba, signal_idx in picks_sorted:
            rows = rows_map.get(code)
            if not rows:
                continue
            buy_idx = _next_trading_index(rows, signal_idx, 1)
            exit_idx = _next_trading_index(rows, buy_idx, args.hold_days) if buy_idx != -1 else -1
            if buy_idx == -1 or exit_idx == -1:
                continue
            buy_price = rows[buy_idx].open
            exit_price = rows[exit_idx].close
            if buy_price <= 0:
                continue
            ret = (exit_price - buy_price) / buy_price * 100
            returns.append(ret)

            detail_rows.append(
                {
                    "signal_date": signal_date,
                    "code": code,
                    "name": name,
                    "ml_proba": round(proba * 100, 2),
                    "buy_date": rows[buy_idx].date,
                    "exit_date": rows[exit_idx].date,
                    "return_pct": round(ret, 2),
                }
            )

        if not returns:
            continue
        returns_np = np.array(returns, dtype=float)
        summary_rows.append(
            {
                "signal_date": signal_date,
                "count": len(returns),
                "avg_return_pct": round(float(np.mean(returns_np)), 2),
                "median_return_pct": round(float(np.median(returns_np)), 2),
                "win_rate": round(float(np.mean(returns_np > 0) * 100), 2),
            }
        )

    summary_path = "data/results/ml_backtest_summary.csv"
    detail_path = "data/results/ml_backtest_detail.csv"

    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["signal_date", "count", "avg_return_pct", "median_return_pct", "win_rate"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(detail_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "signal_date",
                "code",
                "name",
                "ml_proba",
                "buy_date",
                "exit_date",
                "return_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    print("回测完成:")
    print(f"  汇总: {summary_path}")
    print(f"  明细: {detail_path}")


if __name__ == "__main__":
    main()
