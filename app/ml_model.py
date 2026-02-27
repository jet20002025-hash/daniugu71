import csv
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import joblib

from .eastmoney import KlineRow, StockItem, fetch_index_kline, get_kline_cached
from .scanner import ScanResult


FEATURE_NAMES = [
    "open",
    "close",
    "high",
    "low",
    "volume",
    "pct_chg",
    "amplitude",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "ma20_slope",
    "ma60_slope",
    "close_over_ma20",
    "close_over_ma60",
    "vol5",
    "vol20",
    "vol_ratio_5_20",
    "vol_over_20",
    "return_5",
    "return_10",
    "return_20",
    "high20_distance",
    "close_position",
    "shakeout_gap",
    "stop_gap",
    "market",
    "money",
    "money5",
    "money20",
    "money_ratio_5_20",
    "money_over_20",
    "up_volume_ratio_5",
    "up_days_10",
    "atr14",
    "vol_std_20",
    "body_ratio",
    "upper_shadow",
    "lower_shadow",
    "turnover",
    "turnover5",
    "turnover20",
    "turnover_ratio_5_20",
    "turnover_over_20",
    "money_flow",
    "money_flow5",
    "money_flow20",
    "money_flow_ratio_5_20",
]

FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}


@dataclass
class MLConfig:
    signal_type: str = "aggressive"
    signal_lookback: int = 5
    hold_days: int = 40
    return_threshold: float = 8.0
    index_excess: float = 3.0
    start_date: str = "2023-01-01"
    end_date: Optional[str] = None
    min_history: int = 80
    cache_days: int = 3650
    workers: int = 6
    count: int = 900
    max_results: int = 200
    min_score: int = 0
    max_per_day: int = 0
    bull_strict: bool = False
    entry_mode: str = "breakout"
    year_lookback: int = 240
    year_return_limit: float = 100.0
    max_market_cap: Optional[float] = 15_000_000_000.0


def _parse_date(value: str) -> datetime.date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _filter_rows_by_date(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[KlineRow]:
    if not rows:
        return []
    start = _parse_date(start_date) if start_date else None
    end = _parse_date(end_date) if end_date else None
    filtered: List[KlineRow] = []
    for row in rows:
        try:
            row_date = _parse_date(row.date)
        except Exception:
            continue
        if start and row_date < start:
            continue
        if end and row_date > end:
            continue
        filtered.append(row)
    return filtered


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.array([], dtype=float)
    weights = np.ones(window, dtype=float) / window
    return np.convolve(values, weights, mode="valid")


def _close_position(close: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    return (close - low) / (high - low)


def _pct_change(close: np.ndarray, period: int, idx: int) -> Optional[float]:
    if idx - period < 0:
        return None
    base = close[idx - period]
    if base == 0:
        return None
    return (close[idx] - base) / base * 100


def _return_10d(rows: List[KlineRow]) -> Optional[float]:
    if len(rows) <= 10:
        return None
    base = rows[-11].close
    if base == 0:
        return None
    return (rows[-1].close - base) / base * 100


def _load_index_kline_from_csv(path: Optional[str]) -> List[KlineRow]:
    if not path or not os.path.exists(path):
        return []
    rows: List[KlineRow] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append(
                    KlineRow(
                        date=row.get("date", ""),
                        open=float(row.get("open", 0) or 0),
                        close=float(row.get("close", 0) or 0),
                        high=float(row.get("high", 0) or 0),
                        low=float(row.get("low", 0) or 0),
                        volume=float(row.get("volume", 0) or 0),
                        amount=float(row.get("amount", 0) or 0),
                        amplitude=float(row.get("amplitude", 0) or 0),
                        pct_chg=float(row.get("pct_chg", 0) or 0),
                        chg=float(row.get("chg", 0) or 0),
                        turnover=float(row.get("turnover", 0) or 0),
                    )
                )
            except Exception:
                continue
    return rows


def _find_pullback_buy(rows: List[KlineRow], signal_idx: int, window: int = 7) -> Optional[int]:
    if signal_idx + 2 >= len(rows):
        return None
    close = np.array([r.close for r in rows], dtype=float)
    open_ = np.array([r.open for r in rows], dtype=float)
    low = np.array([r.low for r in rows], dtype=float)
    volume = np.array([r.volume for r in rows], dtype=float)

    last = min(len(rows) - 2, signal_idx + window)
    for i in range(signal_idx + 1, last + 1):
        if i < 20:
            continue
        ma10 = float(np.mean(close[i - 9 : i + 1]))
        ma20 = float(np.mean(close[i - 19 : i + 1]))
        vol20 = float(np.mean(volume[i - 19 : i + 1]))
        if vol20 <= 0:
            continue

        near_ma = close[i] <= ma20 * 1.02 or close[i] <= ma10 * 1.01
        vol_shrink = volume[i] <= vol20 * 0.8
        stop_rebound = close[i] >= open_[i] and low[i] >= low[i - 1] * 0.98

        if near_ma and vol_shrink and stop_rebound:
            buy_idx = i + 1
            if buy_idx < len(rows):
                return buy_idx
    return None


def _passes_bull_strict(feature_row: List[float]) -> bool:
    if not feature_row or len(feature_row) < len(FEATURE_NAMES):
        return False

    def _get(name: str) -> Optional[float]:
        idx = FEATURE_INDEX.get(name)
        if idx is None:
            return None
        val = feature_row[idx]
        if val is None or not np.isfinite(val):
            return None
        return float(val)

    vol_ratio = _get("vol_ratio_5_20")
    close_over_ma20 = _get("close_over_ma20")
    ma20_slope = _get("ma20_slope")
    ma60_slope = _get("ma60_slope")
    ret20 = _get("return_20")
    close_pos = _get("close_position")

    if vol_ratio is None or close_over_ma20 is None or ma20_slope is None:
        return False
    if ma60_slope is None or ret20 is None or close_pos is None:
        return False

    if vol_ratio < 2.0:
        return False
    if close_over_ma20 < 1.02:
        return False
    if ma20_slope <= 0:
        return False
    if ma60_slope <= 0:
        return False
    if ret20 < 10:
        return False
    if close_pos < 0.7:
        return False

    return True


def _detect_signals_aggressive(rows: List[KlineRow]) -> List[Tuple[int, int, int]]:
    if len(rows) < 80:
        return []

    close = np.array([r.close for r in rows], dtype=float)
    high = np.array([r.high for r in rows], dtype=float)
    low = np.array([r.low for r in rows], dtype=float)
    open_ = np.array([r.open for r in rows], dtype=float)
    volume = np.array([r.volume for r in rows], dtype=float)
    amount_raw = np.array([r.amount for r in rows], dtype=float)
    turnover = np.array([r.turnover for r in rows], dtype=float)
    pct_chg = np.array([r.pct_chg for r in rows], dtype=float)
    money = np.where(amount_raw > 0, amount_raw, close * volume)
    amplitude = np.array([r.amplitude for r in rows], dtype=float)

    signals: List[Tuple[int, int, int]] = []

    for i in range(60, len(rows)):
        if i < 3 or i < 20:
            continue

        vol20 = np.mean(volume[i - 19 : i + 1])
        if vol20 <= 0:
            continue

        # Start day (aggressive): close breaks recent 3-day high + volume expands
        recent_high = np.max(high[i - 3 : i])
        if close[i] < recent_high:
            continue
        if volume[i] < vol20 * 1.2:
            continue

        stop_idx = -1
        shake_idx = -1

        # Stop day within last 4 days
        for s in range(i - 4, i):
            if s < 1:
                continue
            vol20_s = np.mean(volume[s - 19 : s + 1]) if s >= 19 else None
            if vol20_s is None or vol20_s <= 0:
                continue
            if rows[s].close < rows[s].open:
                continue
            if rows[s].low < rows[s - 1].low * 0.98:
                continue
            if volume[s] > vol20_s:
                continue

            # Shakeout day within 10 days before stop day
            for q in range(s - 10, s):
                if q < 20:
                    continue
                vol20_q = np.mean(volume[q - 19 : q + 1])
                if vol20_q <= 0:
                    continue
                close_pos = _close_position(rows[q].close, rows[q].low, rows[q].high)
                amp = amplitude[q]
                if amp <= 0:
                    amp = (rows[q].high - rows[q].low) / max(rows[q].close, 1e-6) * 100
                if amp < 6:
                    continue
                if volume[q] < vol20_q * 1.5:
                    continue
                if close_pos > 0.35:
                    continue
                shake_idx = q
                break

            if shake_idx != -1:
                stop_idx = s
                break

        if stop_idx != -1 and shake_idx != -1:
            signals.append((i, shake_idx, stop_idx))

    return signals


def _detect_signals_relaxed(rows: List[KlineRow]) -> List[Tuple[int, int, int]]:
    if len(rows) < 80:
        return []

    close = np.array([r.close for r in rows], dtype=float)
    open_ = np.array([r.open for r in rows], dtype=float)
    high = np.array([r.high for r in rows], dtype=float)
    low = np.array([r.low for r in rows], dtype=float)
    volume = np.array([r.volume for r in rows], dtype=float)
    amount_raw = np.array([r.amount for r in rows], dtype=float)
    turnover = np.array([r.turnover for r in rows], dtype=float)
    pct_chg = np.array([r.pct_chg for r in rows], dtype=float)
    money = np.where(amount_raw > 0, amount_raw, close * volume)
    amplitude = np.array([r.amplitude for r in rows], dtype=float)

    signals: List[Tuple[int, int, int]] = []

    for i in range(60, len(rows)):
        vol20 = np.mean(volume[i - 19 : i + 1])
        if vol20 <= 0:
            continue

        recent_high = np.max(high[i - 5 : i])
        ma10_now = np.mean(close[i - 9 : i + 1])
        ma20_now = np.mean(close[i - 19 : i + 1])

        start_ok = False
        if close[i] >= recent_high * 0.995:
            start_ok = True
        if close[i] >= ma10_now and close[i] >= ma20_now and volume[i] >= vol20 * 1.05:
            start_ok = True
        if not start_ok:
            continue
        if volume[i] < vol20 * 1.05:
            continue

        stop_idx = -1
        shake_idx = -1

        # Stop day within last 7 days
        for s in range(i - 7, i):
            if s < 1:
                continue
            vol20_s = np.mean(volume[s - 19 : s + 1]) if s >= 19 else None
            if vol20_s is None or vol20_s <= 0:
                continue
            if rows[s].close < rows[s].open * 0.99:
                continue
            if rows[s].low < rows[s - 1].low * 0.97:
                continue
            if volume[s] > vol20_s * 1.2:
                continue

            # Shakeout day within 15 days before stop day
            for q in range(s - 15, s):
                if q < 20:
                    continue
                vol20_q = np.mean(volume[q - 19 : q + 1])
                if vol20_q <= 0:
                    continue
                close_pos = _close_position(rows[q].close, rows[q].low, rows[q].high)
                amp = amplitude[q]
                if amp <= 0:
                    amp = (rows[q].high - rows[q].low) / max(rows[q].close, 1e-6) * 100
                if amp < 4.5:
                    continue
                if volume[q] < vol20_q * 1.2:
                    continue
                if close_pos > 0.55:
                    continue
                shake_idx = q
                break

            if shake_idx != -1:
                stop_idx = s
                break

        if stop_idx != -1 and shake_idx != -1:
            signals.append((i, shake_idx, stop_idx))

    return signals


def _detect_signals(rows: List[KlineRow], signal_type: str) -> List[Tuple[int, int, int]]:
    if signal_type == "relaxed":
        return _detect_signals_relaxed(rows)
    return _detect_signals_aggressive(rows)


def _build_features(
    rows: List[KlineRow],
    signal_idx: int,
    shake_idx: int,
    stop_idx: int,
    market: int,
) -> Optional[List[float]]:
    """特征仅用 rows[0..signal_idx]（当日及历史），禁止使用 signal_idx 之后的数据。"""
    if signal_idx < 60:
        return None

    close = np.array([r.close for r in rows], dtype=float)
    open_ = np.array([r.open for r in rows], dtype=float)
    high = np.array([r.high for r in rows], dtype=float)
    low = np.array([r.low for r in rows], dtype=float)
    volume = np.array([r.volume for r in rows], dtype=float)
    amount_raw = np.array([r.amount for r in rows], dtype=float)
    turnover = np.array([r.turnover for r in rows], dtype=float)
    pct_chg = np.array([r.pct_chg for r in rows], dtype=float)
    money = np.where(amount_raw > 0, amount_raw, close * volume)

    ma5_slice = close[signal_idx - 4 : signal_idx + 1]
    ma10_slice = close[signal_idx - 9 : signal_idx + 1]
    ma20_slice = close[signal_idx - 19 : signal_idx + 1]
    ma60_slice = close[signal_idx - 59 : signal_idx + 1]

    ma20_prev_slice = close[signal_idx - 22 : signal_idx - 2]
    ma60_prev_slice = close[signal_idx - 62 : signal_idx - 2]

    vol5_slice = volume[signal_idx - 4 : signal_idx + 1]
    vol20_slice = volume[signal_idx - 19 : signal_idx + 1]
    money5_slice = money[signal_idx - 4 : signal_idx + 1]
    money20_slice = money[signal_idx - 19 : signal_idx + 1]
    turn5_slice = turnover[signal_idx - 4 : signal_idx + 1]
    turn20_slice = turnover[signal_idx - 19 : signal_idx + 1]

    if (
        ma5_slice.size == 0
        or ma10_slice.size == 0
        or ma20_slice.size == 0
        or ma60_slice.size == 0
        or ma20_prev_slice.size == 0
        or ma60_prev_slice.size == 0
        or vol5_slice.size == 0
        or vol20_slice.size == 0
        or money5_slice.size == 0
        or money20_slice.size == 0
        or turn5_slice.size == 0
        or turn20_slice.size == 0
    ):
        return None

    ma5 = float(np.mean(ma5_slice))
    ma10 = float(np.mean(ma10_slice))
    ma20 = float(np.mean(ma20_slice))
    ma60 = float(np.mean(ma60_slice))

    ma20_prev = float(np.mean(ma20_prev_slice))
    ma60_prev = float(np.mean(ma60_prev_slice))

    vol5 = float(np.mean(vol5_slice))
    vol20 = float(np.mean(vol20_slice))
    money5 = float(np.mean(money5_slice))
    money20 = float(np.mean(money20_slice))
    turn5 = float(np.mean(turn5_slice))
    turn20 = float(np.mean(turn20_slice))
    if vol20 <= 0:
        return None
    if money20 <= 0:
        return None

    turn_ratio_5_20 = turn5 / turn20 if turn20 > 0 else 0.0
    turn_over_20 = turnover[signal_idx] / turn20 if turn20 > 0 else 0.0

    flow_sign = np.where(pct_chg >= 0, 1.0, -1.0)
    money_flow = money * flow_sign
    money_flow5 = float(np.mean(money_flow[signal_idx - 4 : signal_idx + 1]))
    money_flow20 = float(np.mean(money_flow[signal_idx - 19 : signal_idx + 1]))
    money_flow_ratio = money_flow5 / max(abs(money_flow20), 1e-6)

    high20 = np.max(high[signal_idx - 19 : signal_idx + 1])
    high20_distance = (high20 - close[signal_idx]) / max(high20, 1e-6) * 100

    ret5 = _pct_change(close, 5, signal_idx)
    ret10 = _pct_change(close, 10, signal_idx)
    ret20 = _pct_change(close, 20, signal_idx)
    if ret5 is None or ret10 is None or ret20 is None:
        return None

    close_pos = _close_position(rows[signal_idx].close, rows[signal_idx].low, rows[signal_idx].high)

    up_volume = 0.0
    total_volume = float(np.sum(vol5_slice))
    up_days_10 = 0
    for i in range(signal_idx - 9, signal_idx + 1):
        if i < 0:
            continue
        if close[i] > open_[i]:
            up_days_10 += 1
    for i in range(signal_idx - 4, signal_idx + 1):
        if i < 0:
            continue
        if close[i] > open_[i]:
            up_volume += volume[i]
    up_volume_ratio = up_volume / max(total_volume, 1e-6)

    true_ranges = []
    for i in range(signal_idx - 13, signal_idx + 1):
        if i <= 0:
            continue
        prev_close = close[i - 1]
        tr = max(
            high[i] - low[i],
            abs(high[i] - prev_close),
            abs(low[i] - prev_close),
        )
        true_ranges.append(tr)
    atr14 = float(np.mean(true_ranges)) if true_ranges else 0.0

    pct_slice = pct_chg[signal_idx - 19 : signal_idx + 1]
    vol_std_20 = float(np.std(pct_slice)) if pct_slice.size > 0 else 0.0
    close_idx = float(close[signal_idx])
    open_idx = float(open_[signal_idx])
    high_idx = float(high[signal_idx])
    low_idx = float(low[signal_idx])
    body_ratio = abs(close_idx - open_idx) / max(close_idx, 1e-6)
    upper_shadow = (high_idx - max(open_idx, close_idx)) / max(close_idx, 1e-6)
    lower_shadow = (min(open_idx, close_idx) - low_idx) / max(close_idx, 1e-6)

    features = [
        float(rows[signal_idx].open),
        float(rows[signal_idx].close),
        float(rows[signal_idx].high),
        float(rows[signal_idx].low),
        float(rows[signal_idx].volume),
        float(rows[signal_idx].pct_chg),
        float(rows[signal_idx].amplitude),
        ma5,
        ma10,
        ma20,
        ma60,
        float(ma20 - ma20_prev),
        float(ma60 - ma60_prev),
        float(rows[signal_idx].close / max(ma20, 1e-6)),
        float(rows[signal_idx].close / max(ma60, 1e-6)),
        vol5,
        vol20,
        float(vol5 / max(vol20, 1e-6)),
        float(rows[signal_idx].volume / max(vol20, 1e-6)),
        float(ret5),
        float(ret10),
        float(ret20),
        float(high20_distance),
        float(close_pos),
        float(signal_idx - shake_idx),
        float(signal_idx - stop_idx),
        float(market),
        float(money[signal_idx]),
        float(money5),
        float(money20),
        float(money5 / max(money20, 1e-6)),
        float(money[signal_idx] / max(money20, 1e-6)),
        float(up_volume_ratio),
        float(up_days_10),
        float(atr14),
        float(vol_std_20),
        float(body_ratio),
        float(upper_shadow),
        float(lower_shadow),
        float(turnover[signal_idx]),
        float(turn5),
        float(turn20),
        float(turn_ratio_5_20),
        float(turn_over_20),
        float(money_flow[signal_idx]),
        float(money_flow5),
        float(money_flow20),
        float(money_flow_ratio),
    ]
    if not np.all(np.isfinite(features)):
        return None
    return features


def _index_close_map(index_rows: List[KlineRow]) -> Dict[str, float]:
    return {row.date: float(row.close) for row in index_rows}


def _label_sample(
    rows: List[KlineRow],
    index_close: Dict[str, float],
    signal_idx: int,
    hold_days: int,
    return_threshold: float,
    index_excess: float,
) -> Optional[int]:
    buy_idx = signal_idx + 1
    exit_idx = buy_idx + hold_days
    if exit_idx >= len(rows):
        return None
    buy_price = rows[buy_idx].open
    if buy_price <= 0:
        return None
    exit_price = rows[exit_idx].close
    stock_return = (exit_price - buy_price) / buy_price * 100

    buy_date = rows[buy_idx].date
    exit_date = rows[exit_idx].date
    index_buy = index_close.get(buy_date)
    index_exit = index_close.get(exit_date)
    if index_buy is None or index_exit is None:
        return None
    index_return = (index_exit - index_buy) / index_buy * 100 if index_buy != 0 else 0

    if stock_return >= return_threshold and (stock_return - index_return) >= index_excess:
        return 1
    return 0


def build_dataset(
    stock_list: List[StockItem],
    config: MLConfig,
    cache_dir: str,
    kline_loader: Optional[Callable[[StockItem], Optional[List[KlineRow]]]] = None,
    index_rows: Optional[List[KlineRow]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """构建训练集：特征仅用信号日及历史；标签为持有期收益/超额（事后），不得用未来数据做特征或样本筛选。"""
    if index_rows is None:
        index_rows = fetch_index_kline()
    index_rows = _filter_rows_by_date(index_rows, config.start_date, config.end_date)
    index_close = _index_close_map(index_rows)

    features: List[List[float]] = []
    labels: List[int] = []

    for item in stock_list:
        if kline_loader:
            rows = kline_loader(item)
        else:
            rows = get_kline_cached(
                item.secid,
                cache_dir=cache_dir,
                count=config.count,
                max_age_days=config.cache_days,
                pause=0.0,
            )
        if not rows:
            continue
        rows = _filter_rows_by_date(rows, config.start_date, config.end_date)
        if len(rows) < config.min_history:
            continue

        signals = _detect_signals(rows, config.signal_type)
        for signal_idx, shake_idx, stop_idx in signals:
            label = _label_sample(
                rows=rows,
                index_close=index_close,
                signal_idx=signal_idx,
                hold_days=config.hold_days,
                return_threshold=config.return_threshold,
                index_excess=config.index_excess,
            )
            if label is None:
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
            features.append(feature_row)
            labels.append(label)

    if not features:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,), dtype=int)

    return np.array(features, dtype=float), np.array(labels, dtype=int)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    weight_strategy: str = "balanced",
) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(random_state=42)
    if len(y) == 0:
        raise ValueError("样本为空，无法训练模型")
    if len(np.unique(y)) < 2:
        raise ValueError("样本仅单一类别，无法训练模型")
    sample_weight = None
    if weight_strategy == "balanced":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def save_model_bundle(
    model: GradientBoostingClassifier,
    config: MLConfig,
    model_path: str,
    meta_path: str,
) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    meta = {
        "feature_names": FEATURE_NAMES,
        "config": asdict(config),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


def load_model_bundle(model_path: str, meta_path: str):
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return model, meta


def scan_with_model(
    stock_list: List[StockItem],
    model,
    config: MLConfig,
    cache_dir: str,
    progress_cb: Optional[Callable[[], None]] = None,
    local_only: bool = False,
    kline_loader: Optional[Callable[[StockItem], Optional[List[KlineRow]]]] = None,
    prefer_local: bool = False,
    cutoff_date: Optional[str] = None,
    start_date: Optional[str] = None,
) -> List[ScanResult]:
    results: List[ScanResult] = []
    start_dt = _parse_date(start_date) if start_date else None
    end_dt = _parse_date(cutoff_date) if cutoff_date else None

    for item in stock_list:
        if progress_cb:
            progress_cb()
        try:
            if kline_loader:
                rows = kline_loader(item)
            else:
                rows = get_kline_cached(
                    item.secid,
                    cache_dir=cache_dir,
                    count=config.count,
                    max_age_days=config.cache_days,
                    pause=0.0,
                    local_only=local_only,
                    prefer_local=prefer_local,
                )
        except Exception:
            rows = None
        if not rows:
            continue
        if config.end_date:
            rows = _filter_rows_by_date(rows, None, config.end_date)

        if len(rows) < config.min_history:
            continue

        signals = _detect_signals(rows, config.signal_type)
        if not signals:
            continue

        if start_dt or end_dt:
            filtered = []
            for s in signals:
                try:
                    sd = _parse_date(rows[s[0]].date)
                except Exception:
                    continue
                if start_dt and sd < start_dt:
                    continue
                if end_dt and sd > end_dt:
                    continue
                filtered.append(s)
            signals = filtered
        elif cutoff_date:
            signals = [s for s in signals if rows[s[0]].date == cutoff_date]

        if not signals:
            continue

        use_all_signals = bool((start_dt or end_dt or cutoff_date) and config.max_per_day > 0)
        signals_to_use = signals if use_all_signals else [signals[-1]]

        for latest_idx, shake_idx, stop_idx in signals_to_use:
            if not start_dt and not end_dt and not cutoff_date:
                if len(rows) - 1 - latest_idx > config.signal_lookback:
                    continue

            if latest_idx - config.year_lookback < 0:
                continue
            base = rows[latest_idx - config.year_lookback].close
            if base > 0:
                year_return = (rows[latest_idx].close - base) / base * 100
                if year_return >= config.year_return_limit:
                    continue

            feature_row = _build_features(
                rows=rows,
                signal_idx=latest_idx,
                shake_idx=shake_idx,
                stop_idx=stop_idx,
                market=item.market,
            )
            if feature_row is None:
                continue
            if not np.all(np.isfinite(feature_row)):
                continue
            if config.bull_strict and not _passes_bull_strict(feature_row):
                continue

            proba = float(model.predict_proba([feature_row])[0][1])
            latest_close = float(rows[-1].close)
            latest_change = float(rows[-1].pct_chg)

            signal_date = rows[latest_idx].date
            if config.entry_mode == "pullback":
                buy_idx = _find_pullback_buy(rows, latest_idx, window=7)
                if buy_idx is None:
                    continue
            else:
                buy_idx = latest_idx + 1
            buy_date = rows[buy_idx].date if buy_idx < len(rows) else signal_date

            signal_label = "激进" if config.signal_type == "aggressive" else "宽松"
            entry_label = "回踩买点" if config.entry_mode == "pullback" else "量化买点"
            reasons = [
                f"ML概率 {proba * 100:.1f}%",
                f"信号日 {signal_date}",
                f"买入日 {buy_date} (T+1 开盘)",
                f"{entry_label}({signal_label})",
            ]

            metrics = {
                "ml_probability": proba,
                "signal_date": signal_date,
                "buy_date": buy_date,
            }

            results.append(
                ScanResult(
                    code=item.code,
                    name=item.name,
                    score=int(round(proba * 100)),
                    latest_close=latest_close,
                    change_pct=latest_change,
                    reasons=reasons,
                    metrics=metrics,
                )
            )

    results.sort(key=lambda r: (-r.score, -r.change_pct))

    if config.max_per_day > 0 and config.min_score > 0:
        results = [r for r in results if r.score >= config.min_score]

    if config.max_per_day > 0:
        grouped: Dict[str, List[ScanResult]] = {}
        for r in results:
            date_key = ""
            if isinstance(r.metrics, dict):
                date_key = str(r.metrics.get("signal_date") or "")
            grouped.setdefault(date_key, []).append(r)
        limited: List[ScanResult] = []
        for _, group in grouped.items():
            group.sort(key=lambda r: (-r.score, -r.change_pct))
            limited.extend(group[: config.max_per_day])
        results = limited
        results.sort(key=lambda r: (-r.score, -r.change_pct))

    return results[: config.max_results]
