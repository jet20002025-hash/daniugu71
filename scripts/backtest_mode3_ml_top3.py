import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime
from statistics import mean, median
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.eastmoney import read_cached_kline, read_cached_kline_by_code, stock_items_from_list_csv
from app.paths import GPT_DATA_DIR, MARKET_CAP_PATH


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


def _moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    res = np.full_like(values, np.nan, dtype=float)
    if len(values) < window:
        return res
    weights = np.ones(window, dtype=float) / window
    res[window - 1 :] = np.convolve(values, weights, mode="valid")
    return res


def _load_rows(cache_dir: str, cache_format: str, market: int, code: str):
    if cache_format == "secid":
        path = os.path.join(cache_dir, f"{market}_{code}.csv")
        return read_cached_kline(path)
    return read_cached_kline_by_code(cache_dir, code)


def _calc_multiple(rows, buy_idx: int, hold_days: int, multiple: float) -> Tuple[int, float, Optional[str]]:
    exit_idx = buy_idx + hold_days
    if buy_idx < 0 or exit_idx >= len(rows):
        return 0, 0.0, None
    buy_price = rows[buy_idx].open
    if buy_price <= 0:
        return 0, 0.0, None
    max_high = max(r.high for r in rows[buy_idx : exit_idx + 1])
    if max_high <= 0:
        return 0, 0.0, None
    hit_date = None
    if max_high / buy_price >= multiple:
        for i in range(buy_idx, exit_idx + 1):
            if rows[i].high / buy_price >= multiple:
                hit_date = rows[i].date
                break
        return 1, max_high / buy_price, hit_date
    return 0, max_high / buy_price, hit_date


def _signals_mode3(
    rows,
    dates: List[str],
    close: np.ndarray,
    volume: np.ndarray,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    ret20: np.ndarray,
    start: Optional[str],
    end: Optional[str],
) -> List[int]:
    signals = []
    for i in range(60, len(rows)):
        if not _in_range(dates[i], start, end):
            continue
        if np.isnan(ma10[i]) or np.isnan(ma20[i]) or np.isnan(ma60[i]) or np.isnan(vol20[i]):
            continue
        if ret20[i] is not None and ret20[i] > 25:
            continue
        ma10_slope = ma10[i] - ma10[i - 3]
        ma20_slope = ma20[i] - ma20[i - 3]
        ma60_slope = ma60[i] - ma60[i - 3]
        if not (
            ma10[i] > ma20[i] > ma60[i]
            and ma10_slope > 0
            and ma20_slope > 0
            and ma60_slope > 0
        ):
            continue
        if close[i] < ma20[i]:
            continue
        if volume[i] < vol20[i] * 1.2:
            continue
        signals.append(i)
    return signals


def _build_features(
    rows,
    idx: int,
    close: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
) -> Optional[List[float]]:
    if np.isnan(ma10[idx]) or np.isnan(ma20[idx]) or np.isnan(ma60[idx]) or np.isnan(vol20[idx]):
        return None
    if ma20[idx] <= 0 or ma60[idx] <= 0 or vol20[idx] <= 0:
        return None
    o = open_[idx]
    c = close[idx]
    h = high[idx]
    l = low[idx]
    rng = h - l
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    body_ratio = body / rng if rng > 0 else 0.0
    upper_ratio = upper / rng if rng > 0 else 0.0
    lower_ratio = lower / rng if rng > 0 else 0.0
    ma10_gap = (ma10[idx] - ma20[idx]) / ma20[idx]
    ma20_gap = (ma20[idx] - ma60[idx]) / ma60[idx]
    close_gap = (c - ma20[idx]) / ma20[idx]
    vol_ratio = volume[idx] / vol20[idx]

    def _ret(period: int) -> float:
        if idx - period < 0:
            return 0.0
        base = close[idx - period]
        return (c - base) / base if base > 0 else 0.0

    ret5 = _ret(5)
    ret10 = _ret(10)
    ret20 = _ret(20)
    range_ratio = (rng / c) if c > 0 else 0.0

    return [
        c,
        rows[idx].pct_chg,
        vol_ratio,
        ma10_gap,
        ma20_gap,
        close_gap,
        ret5,
        ret10,
        ret20,
        body_ratio,
        upper_ratio,
        lower_ratio,
        range_ratio,
        upper_ratio * vol_ratio,
    ]


def _load_market_caps(path: str) -> Dict[str, float]:
    if not path or not os.path.exists(path):
        return {}
    mapping: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            return mapping
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = str(row.get("code", "")).strip()
            if code.isdigit() and len(code) < 6:
                code = code.zfill(6)
            if not code:
                continue
            try:
                cap = float(row.get("total_cap", 0) or 0)
            except Exception:
                continue
            if cap > 0:
                mapping[code] = cap
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest mode3 ML and report daily TopK.")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--multiple", type=float, default=2.0)
    parser.add_argument("--buy-offset", type=int, default=1)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--cache-dir", default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"))
    parser.add_argument("--cache-format", choices=["auto", "code", "secid"], default="auto")
    parser.add_argument("--stock-list", default=os.path.join(GPT_DATA_DIR, "stock_list.csv"))
    parser.add_argument("--model-path", default="data/models/mode3.pkl")
    parser.add_argument("--output", default="data/results/mode3_ml_top3_2025.csv")
    parser.add_argument("--output-xlsx", default="data/results/mode3_ml_top3_2025.xlsx")
    parser.add_argument("--max-market-cap", type=float, default=150.0, help="billion CNY, 0 to disable")
    args = parser.parse_args()

    cache_format = args.cache_format
    if cache_format == "auto":
        cache_format = "secid"

    stock_list = stock_items_from_list_csv(args.stock_list)
    if not stock_list:
        raise RuntimeError("股票列表为空")

    cap_limit = None if args.max_market_cap <= 0 else args.max_market_cap * 1e8
    market_caps = _load_market_caps(MARKET_CAP_PATH) if cap_limit else {}

    try:
        import joblib

        model = joblib.load(args.model_path)
    except Exception as exc:
        raise RuntimeError(f"无法加载模型: {exc}")

    daily_picks = defaultdict(list)
    stat = {"signals": 0, "hits": 0, "multiples": [], "hit_days": []}

    for item in stock_list:
        if cap_limit and market_caps:
            cap_value = market_caps.get(item.code)
            if cap_value is None or cap_value > cap_limit:
                continue
        rows = _load_rows(args.cache_dir, cache_format, item.market, item.code)
        if not rows or len(rows) < 80:
            continue
        dates = [r.date for r in rows]
        close = np.array([r.close for r in rows], dtype=float)
        open_ = np.array([r.open for r in rows], dtype=float)
        high = np.array([r.high for r in rows], dtype=float)
        low = np.array([r.low for r in rows], dtype=float)
        volume = np.array([r.volume for r in rows], dtype=float)

        ma10 = _moving_mean(close, 10)
        ma20 = _moving_mean(close, 20)
        ma60 = _moving_mean(close, 60)
        vol20 = _moving_mean(volume, 20)
        ret20 = [None] * len(rows)
        for i in range(20, len(rows)):
            base = close[i - 20]
            ret20[i] = (close[i] - base) / base * 100 if base else None

        signals = _signals_mode3(
            rows,
            dates,
            close,
            volume,
            ma10,
            ma20,
            ma60,
            vol20,
            ret20,
            args.start_date,
            args.end_date,
        )
        if not signals:
            continue

        for idx in signals:
            feats = _build_features(
                rows, idx, close, open_, high, low, volume, ma10, ma20, ma60, vol20
            )
            if feats is None:
                continue
            if not np.all(np.isfinite(feats)):
                continue
            proba = float(model.predict_proba([feats])[0][1])

            buy_idx = idx + max(args.buy_offset, 0)
            if buy_idx >= len(rows):
                continue
            label, mult, hit_date = _calc_multiple(rows, buy_idx, args.hold_days, args.multiple)
            stat["signals"] += 1
            stat["multiples"].append(mult)
            if label == 1 and hit_date:
                stat["hits"] += 1
                hit_days = (_parse_date(hit_date) - _parse_date(rows[buy_idx].date)).days
                stat["hit_days"].append(hit_days)

            daily_picks[dates[idx]].append(
                {
                    "date": dates[idx],
                    "code": item.code,
                    "name": item.name,
                    "buy_date": rows[buy_idx].date,
                    "proba": round(proba, 4),
                    "multiple": round(mult, 4),
                    "label": label,
                }
            )

    out_rows: List[Dict[str, object]] = []
    for day, items in daily_picks.items():
        items.sort(key=lambda x: (-x["proba"], -x["multiple"], x["code"]))
        for item in items[: args.topk]:
            out_rows.append(item)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["date", "code", "name", "buy_date", "proba", "multiple", "label"],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    try:
        import pandas as pd

        df = pd.DataFrame(out_rows)
        df.to_excel(args.output_xlsx, index=False)
    except Exception as exc:
        print(f"写入Excel失败: {exc}")

    precision = stat["hits"] / stat["signals"] if stat["signals"] else 0.0
    avg_mult = mean(stat["multiples"]) if stat["multiples"] else 0.0
    med_mult = median(stat["multiples"]) if stat["multiples"] else 0.0
    avg_hit_days = mean(stat["hit_days"]) if stat["hit_days"] else 0.0
    print("回测区间:", args.start_date, "~", args.end_date)
    print(
        f"mode3-ML: 信号数 {stat['signals']} 命中 {stat['hits']} 精确率 {precision:.2%} "
        f"平均倍数 {avg_mult:.2f} 中位倍数 {med_mult:.2f} 平均命中天数 {avg_hit_days:.1f}"
    )
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
