"""
71倍(mode3)信号上的 ML 训练。特征仅用信号日当日及历史（rows[0..idx]），禁止使用未来数据。
标签为持有期内的倍数（事后计算），仅作监督目标；不得用未来数据做特征或样本筛选。
"""
import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

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


def _calc_multiple(rows, buy_idx: int, hold_days: int) -> Tuple[float, Optional[str]]:
    exit_idx = buy_idx + hold_days
    if buy_idx < 0 or exit_idx >= len(rows):
        return 0.0, None
    buy_price = rows[buy_idx].open
    if buy_price <= 0:
        return 0.0, None
    max_high = max(r.high for r in rows[buy_idx : exit_idx + 1])
    if max_high <= 0:
        return 0.0, None
    hit_date = None
    for i in range(buy_idx, exit_idx + 1):
        if rows[i].high / buy_price >= 2.0:
            hit_date = rows[i].date
            break
    return max_high / buy_price, hit_date


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
    import csv

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
    parser = argparse.ArgumentParser(description="Train ML model on mode3 signals (2025).")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--buy-offset", type=int, default=1)
    parser.add_argument("--good-multiple", type=float, default=2.0)
    parser.add_argument("--bad-max", type=float, default=1.10)
    parser.add_argument("--cache-dir", default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"))
    parser.add_argument("--cache-format", choices=["auto", "code", "secid"], default="auto")
    parser.add_argument("--stock-list", default=os.path.join(GPT_DATA_DIR, "stock_list.csv"))
    parser.add_argument("--model-out", default="data/models/mode3.pkl")
    parser.add_argument("--meta-out", default="data/models/mode3_meta.json")
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

    X: List[List[float]] = []
    y: List[int] = []
    pos = neg = 0

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
            buy_idx = idx + max(args.buy_offset, 0)
            if buy_idx + args.hold_days >= len(rows):
                continue
            mult, _ = _calc_multiple(rows, buy_idx, args.hold_days)
            if mult >= args.good_multiple:
                label = 1
            elif mult <= args.bad_max:
                label = 0
            else:
                continue
            feats = _build_features(
                rows, idx, close, open_, high, low, volume, ma10, ma20, ma60, vol20
            )
            if feats is None:
                continue
            if not np.all(np.isfinite(feats)):
                continue
            X.append(feats)
            y.append(label)
            if label == 1:
                pos += 1
            else:
                neg += 1

    if not X:
        raise RuntimeError("没有构建出训练样本")

    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=int)
    print(f"样本量: {len(X_arr)} 正样本: {pos} 负样本: {neg}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    val_proba = clf.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_val, val_proba)
    except Exception:
        auc = 0.0
    print(f"验证集 AUC: {auc:.4f}")
    print("验证集分类报告:")
    print(classification_report(y_val, val_pred, digits=4))

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    import joblib

    joblib.dump(clf, args.model_out)

    meta = {
        "name": "mode3",
        "type": "mode3_ml",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_range": [args.start_date, args.end_date],
        "hold_days": args.hold_days,
        "buy_offset": args.buy_offset,
        "good_multiple": args.good_multiple,
        "bad_max": args.bad_max,
        "features": [
            "close",
            "pct_chg",
            "vol_ratio",
            "ma10_gap",
            "ma20_gap",
            "close_gap",
            "ret5",
            "ret10",
            "ret20",
            "body_ratio",
            "upper_ratio",
            "lower_ratio",
            "range_ratio",
            "upper_ratio_x_vol_ratio",
        ],
        "market_cap_max": cap_limit,
        "samples": len(X_arr),
        "pos_samples": pos,
        "neg_samples": neg,
        "val_auc": float(auc),
    }
    with open(args.meta_out, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print("模型已保存:")
    print(f"  {args.model_out}")
    print(f"  {args.meta_out}")


if __name__ == "__main__":
    main()
