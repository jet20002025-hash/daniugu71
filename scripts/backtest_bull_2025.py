import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime
from statistics import median
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.eastmoney import read_cached_kline, read_cached_kline_by_code, stock_items_from_list_csv
from app.ml_model import (
    MLConfig,
    _build_features,
    _detect_signals,
    _passes_bull_strict,
    load_model_bundle,
)
from app.paths import GPT_DATA_DIR


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


def _max_high_in_window(rows, start_idx: int, end_idx: int) -> Tuple[float, Optional[str]]:
    max_high = -1.0
    max_date = None
    for i in range(start_idx, min(end_idx + 1, len(rows))):
        h = rows[i].high
        if h > max_high:
            max_high = h
            max_date = rows[i].date
    return max_high, max_date


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest bull model for 2025 with daily top-k.")
    parser.add_argument("--start-date", default="2025-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="2025-12-31", help="YYYY-MM-DD")
    parser.add_argument("--topk", type=int, default=5, help="Top N per day")
    parser.add_argument("--hold-days", type=int, default=20, help="Hold window in trading days")
    parser.add_argument("--multiple", type=float, default=2.0, help="Target multiple")
    parser.add_argument("--buy-offset", type=int, default=1, help="Buy offset from signal day")
    parser.add_argument("--min-score", type=int, default=None, help="Override min score")
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
    parser.add_argument(
        "--stock-list",
        default=os.path.join(GPT_DATA_DIR, "stock_list.csv"),
        help="Stock list CSV (code,name)",
    )
    parser.add_argument("--model", default="data/models/ml_bull.pkl", help="Model path")
    parser.add_argument("--meta", default="data/models/ml_bull_meta.json", help="Model meta path")
    parser.add_argument(
        "--output",
        default="data/results/bull_backtest_2025.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    model, meta = load_model_bundle(args.model, args.meta)
    if model is None or meta is None:
        raise RuntimeError("未找到牛股模型，请先训练。")

    meta_cfg = meta.get("config", {}) if isinstance(meta, dict) else {}
    config = MLConfig(
        signal_type=meta_cfg.get("signal_type", "relaxed"),
        min_history=meta_cfg.get("min_history", 80),
    )
    config.year_lookback = meta_cfg.get("year_lookback", config.year_lookback)
    config.year_return_limit = meta_cfg.get("year_return_limit", config.year_return_limit)

    bull_strict = bool(meta.get("strict_filter")) if isinstance(meta, dict) else False
    min_score = args.min_score
    if min_score is None:
        meta_min_score = meta.get("min_score") if isinstance(meta, dict) else None
        min_score = int(meta_min_score) if isinstance(meta_min_score, (int, float)) else 0

    feature_names = meta.get("feature_names") if isinstance(meta, dict) else None
    expected_len = len(feature_names) if feature_names else getattr(model, "n_features_in_", None)

    cache_format = args.cache_format
    if cache_format == "auto":
        cache_format = _detect_cache_format(args.cache_dir)

    stock_list = stock_items_from_list_csv(args.stock_list)
    if not stock_list:
        raise RuntimeError("股票列表为空。")

    picks_by_date: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for item in stock_list:
        rows = _load_rows(args.cache_dir, cache_format, item.market, item.code)
        if not rows or len(rows) < config.min_history:
            continue

        signals = _detect_signals(rows, config.signal_type)
        if not signals:
            continue

        for signal_idx, shake_idx, stop_idx in signals:
            signal_date = rows[signal_idx].date
            if not _in_range(signal_date, args.start_date, args.end_date):
                continue

            if signal_idx - config.year_lookback < 0:
                continue
            base = rows[signal_idx - config.year_lookback].close
            if base > 0:
                year_return = (rows[signal_idx].close - base) / base * 100
                if year_return >= config.year_return_limit:
                    continue

            feature_row = _build_features(
                rows=rows,
                signal_idx=signal_idx,
                shake_idx=shake_idx,
                stop_idx=stop_idx,
                market=item.market,
            )
            if feature_row is None or not np.all(np.isfinite(feature_row)):
                continue
            if bull_strict and not _passes_bull_strict(feature_row):
                continue

            if expected_len is not None:
                feature_row = feature_row[:expected_len]

            proba = float(model.predict_proba([feature_row])[0][1])
            score = int(round(proba * 100))
            if score < min_score:
                continue

            buy_idx = signal_idx + max(args.buy_offset, 0)
            exit_idx = buy_idx + args.hold_days
            if buy_idx < 0 or exit_idx >= len(rows):
                continue
            buy_price = rows[buy_idx].open
            if buy_price <= 0:
                continue
            max_high, hit_date = _max_high_in_window(rows, buy_idx, exit_idx)
            if max_high <= 0:
                continue
            multiple = max_high / buy_price
            label = 1 if multiple >= args.multiple else 0

            picks_by_date[signal_date].append(
                {
                    "date": signal_date,
                    "code": item.code,
                    "name": item.name,
                    "score": score,
                    "proba": round(proba, 6),
                    "buy_date": rows[buy_idx].date,
                    "buy_price": round(buy_price, 4),
                    "max_high": round(max_high, 4),
                    "hit_date": hit_date or "",
                    "multiple": round(multiple, 4),
                    "label": label,
                }
            )

    results: List[Dict[str, object]] = []
    daily_counts: List[int] = []
    total_pos = 0
    total_picks = 0

    for day, items in sorted(picks_by_date.items()):
        items.sort(key=lambda x: (-x["score"], -x["proba"]))
        picked = items[: args.topk] if args.topk > 0 else items
        if not picked:
            continue
        daily_counts.append(len(picked))
        total_picks += len(picked)
        total_pos += sum(int(x["label"]) for x in picked)
        results.extend(picked)

    precision = total_pos / total_picks if total_picks > 0 else 0.0
    days_with_picks = len(daily_counts)
    avg_picks = sum(daily_counts) / days_with_picks if days_with_picks else 0.0
    median_picks = median(daily_counts) if daily_counts else 0.0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "code",
                "name",
                "score",
                "proba",
                "buy_date",
                "buy_price",
                "max_high",
                "hit_date",
                "multiple",
                "label",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("回测完成")
    print(f"区间: {args.start_date} ~ {args.end_date}")
    print(f"模型阈值: {min_score}")
    print(f"每日TopK: {args.topk}")
    print(f"总选股数: {total_picks}")
    print(f"命中数: {total_pos}")
    print(f"精确率: {precision:.2%}")
    print(f"有结果的交易日: {days_with_picks}")
    print(f"平均每日选股: {avg_picks:.2f}")
    print(f"每日选股中位数: {median_picks:.2f}")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
