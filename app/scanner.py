import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np

from .eastmoney import KlineRow, StockItem


@dataclass
class ScanResult:
    code: str
    name: str
    score: int
    latest_close: float
    change_pct: float
    reasons: List[str]
    metrics: Dict[str, float]


@dataclass
class ScanConfig:
    min_score: int = 70
    max_results: int = 200
    volume_ratio: float = 1.2
    near_high_pct: float = 3.0
    breakout_lookback: int = 20
    breakout_recent: int = 3
    year_lookback: int = 240
    year_return_limit: float = 500.0
    year_high_low_ratio_limit: float = 4.0  # 近一年最高/最低超4倍则排除
    cache_days: int = 2
    workers: int = 6
    weight_trend: float = 1.3
    weight_volume: float = 1.4
    weight_breakout: float = 1.0
    weight_strength: float = 1.0
    weight_risk: float = 1.0
    max_market_cap: Optional[float] = 15_000_000_000.0


def _normalize_code(code: str) -> str:
    value = str(code or "").strip()
    if value.isdigit() and len(value) < 6:
        return value.zfill(6)
    return value


def _is_st(name: str) -> bool:
    if not name:
        return True
    return "ST" in name or name.startswith("*ST") or name.startswith("退")


def _to_array(rows: List[KlineRow]) -> Dict[str, np.ndarray]:
    return {
        "close": np.array([r.close for r in rows], dtype=float),
        "high": np.array([r.high for r in rows], dtype=float),
        "low": np.array([r.low for r in rows], dtype=float),
        "volume": np.array([r.volume for r in rows], dtype=float),
        "pct": np.array([r.pct_chg for r in rows], dtype=float),
    }


def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    if len(data) < window:
        return np.array([])
    return np.convolve(data, np.ones(window) / window, mode="valid")


def _moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    res = np.full_like(values, np.nan, dtype=float)
    if len(values) < window:
        return res
    weights = np.ones(window, dtype=float) / window
    res[window - 1 :] = np.convolve(values, weights, mode="valid")
    return res


def _pct_change(a: np.ndarray, period: int) -> Optional[float]:
    if len(a) <= period:
        return None
    base = a[-period - 1]
    if base == 0:
        return None
    return (a[-1] - base) / base * 100


def score_stock(
    item: StockItem,
    rows: List[KlineRow],
    index_return_10d: Optional[float],
    return_percentile_10d: Optional[float],
    config: ScanConfig,
) -> Optional[ScanResult]:
    if _is_st(item.name):
        return None
    if len(rows) < 80:
        return None

    arr = _to_array(rows)
    close = arr["close"]
    high = arr["high"]
    low = arr["low"]
    volume = arr["volume"]

    ma20 = _rolling_mean(close, 20)
    ma60 = _rolling_mean(close, 60)
    if len(ma20) == 0 or len(ma60) == 0:
        return None

    latest_close = close[-1]
    latest_change = rows[-1].pct_chg

    if len(close) <= config.year_lookback:
        return None
    base = close[-config.year_lookback - 1]
    if base > 0:
        year_return = (latest_close - base) / base * 100
        if year_return >= config.year_return_limit:
            return None

    trend_score = 0
    volume_score = 0
    breakout_score = 0
    strength_score = 0
    risk_score = 0
    reasons: List[str] = []

    # Trend structure (30)
    ma20_now = ma20[-1]
    ma60_now = ma60[-1]
    ma20_slope = ma20[-1] - ma20[-4] if len(ma20) >= 4 else 0
    ma60_slope = ma60[-1] - ma60[-4] if len(ma60) >= 4 else 0

    if ma20_now > ma60_now and ma20_slope > 0 and ma60_slope > 0:
        trend_score += 10
        reasons.append("20/60均线多头且上行")

    if len(close) >= 3 and np.all(close[-3:] > ma20_now):
        trend_score += 10
        reasons.append("连续3日站上20日均线")

    high_20 = np.max(close[-20:])
    if high_20 > 0 and (high_20 - latest_close) / high_20 * 100 <= config.near_high_pct:
        trend_score += 10
        reasons.append("接近20日新高")

    # Volume & flow (25)
    vol5 = np.mean(volume[-5:])
    vol20 = np.mean(volume[-20:])
    if vol20 > 0 and vol5 >= vol20 * config.volume_ratio:
        volume_score += 10
        reasons.append("5日均量显著放大")

    if len(close) >= 11:
        up_mask = close[1:] >= close[:-1]
        vol_up = np.sum(volume[1:][up_mask])
        vol_down = np.sum(volume[1:][~up_mask])
        if vol_up > vol_down:
            volume_score += 10
            reasons.append("上涨日量能占优")

    if vol20 > 0 and np.max(volume[-3:]) >= vol20 * 1.8:
        volume_score += 5
        reasons.append("近3日出现明显放量")

    # Breakout / pattern (20)
    lookback = config.breakout_lookback
    recent = config.breakout_recent
    if len(close) > lookback + recent:
        previous_high = np.max(close[-(lookback + recent) : -recent])
        recent_high = np.max(close[-recent:])
        if recent_high >= previous_high:
            breakout_score += 10
            reasons.append("近期突破前高")

    if len(low) >= 5:
        if np.min(low[-3:]) >= ma20_now * 0.98 and volume[-1] <= vol20:
            breakout_score += 10
            reasons.append("回踩不破+量能收敛")

    # Strength (15)
    if return_percentile_10d is not None and return_percentile_10d >= 90:
        strength_score += 10
        reasons.append("10日涨幅位列前10%")

    if index_return_10d is not None:
        stock_return_10d = _pct_change(close, 10)
        if stock_return_10d is not None and stock_return_10d >= index_return_10d + 2:
            strength_score += 5
            reasons.append("强于指数")

    # Risk penalties
    # Long upper shadow on heavy volume
    body = abs(rows[-1].close - rows[-1].open)
    upper = rows[-1].high - max(rows[-1].close, rows[-1].open)
    if vol20 > 0 and upper > body * 2 and volume[-1] >= vol20 * 1.5:
        risk_score -= 5
        reasons.append("放量长上影扣分")

    if vol20 > 0 and np.mean(volume[-5:]) >= vol20 * 1.5:
        prev_high = np.max(close[-25:-5]) if len(close) >= 30 else np.max(close[:-5])
        if prev_high > 0 and np.max(close[-5:]) < prev_high:
            risk_score -= 5
            reasons.append("放量未创新高扣分")

    raw_score = trend_score + volume_score + breakout_score + strength_score + risk_score
    weighted_score = (
        trend_score * config.weight_trend
        + volume_score * config.weight_volume
        + breakout_score * config.weight_breakout
        + strength_score * config.weight_strength
        + risk_score * config.weight_risk
    )
    score = int(round(weighted_score))

    metrics = {
        "ma20": float(ma20_now),
        "ma60": float(ma60_now),
        "vol5": float(vol5),
        "vol20": float(vol20),
        "high20": float(high_20),
        "score_raw": float(raw_score),
        "score_weighted": float(score),
        "score_trend": float(trend_score),
        "score_volume": float(volume_score),
        "score_breakout": float(breakout_score),
        "score_strength": float(strength_score),
        "score_risk": float(risk_score),
        "weight_trend": float(config.weight_trend),
        "weight_volume": float(config.weight_volume),
        "weight_breakout": float(config.weight_breakout),
        "weight_strength": float(config.weight_strength),
        "weight_risk": float(config.weight_risk),
    }

    return ScanResult(
        code=item.code,
        name=item.name,
        score=int(score),
        latest_close=float(latest_close),
        change_pct=float(latest_change),
        reasons=reasons,
        metrics=metrics,
    )


def percentile_ranks(values: List[float]) -> Dict[int, float]:
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(arr))
    percentiles = ranks / (len(arr) - 1) * 100 if len(arr) > 1 else np.array([100.0])
    return {idx: float(percentiles[idx]) for idx in range(len(arr))}


def _parse_date(value: Optional[str]) -> Optional[datetime.date]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _mode3_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[int]:
    signals: List[int] = []
    if len(rows) < 60:
        return signals
    close = np.array([r.close for r in rows], dtype=float)
    volume = np.array([r.volume for r in rows], dtype=float)

    ma10 = _moving_mean(close, 10)
    ma20 = _moving_mean(close, 20)
    ma60 = _moving_mean(close, 60)
    vol20 = _moving_mean(volume, 20)

    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    for i in range(60, len(rows)):
        if start_dt or end_dt:
            try:
                row_dt = datetime.strptime(rows[i].date, "%Y-%m-%d").date()
            except Exception:
                continue
            if start_dt and row_dt < start_dt:
                continue
            if end_dt and row_dt > end_dt:
                continue

        if (
            np.isnan(ma10[i])
            or np.isnan(ma20[i])
            or np.isnan(ma60[i])
            or np.isnan(vol20[i])
        ):
            continue

        if i - 20 >= 0 and close[i - 20] > 0:
            ret20 = (close[i] - close[i - 20]) / close[i - 20] * 100
            if ret20 > 25:
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


def _limit_rate(code: str, name: str) -> float:
    """涨停/跌停幅度：ST 5%，科创/创业板 20%，其他 10%"""
    code = str(code or "")
    if _is_st(name or ""):
        return 0.05
    if code.startswith(("30", "301", "688")):
        return 0.20
    if code.startswith(("8", "9")):
        return 0.30
    return 0.10


def _has_limit_up_then_down(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    lookback: int = 5,
    min_consec_limit_up: int = 3,
) -> bool:
    """买点前 lookback 个交易日内：连续涨停(≥min_consec_limit_up天)后跌停 → True(需排除)"""
    if idx < lookback + 1:
        return False
    rate = _limit_rate(code, name)
    limit_up = (rate * 100) - 0.5
    limit_down = -(rate * 100) + 0.5
    conseq_limit_up = 0
    for j in range(lookback, 0, -1):
        i = idx - j
        p = rows[i].pct_chg
        if p >= limit_up:
            conseq_limit_up += 1
        elif p <= limit_down:
            if conseq_limit_up >= min_consec_limit_up:
                return True
            conseq_limit_up = 0
        else:
            conseq_limit_up = 0
    return False


def _close_below_ma20_today(
    close: np.ndarray,
    ma20: np.ndarray,
    idx: int,
) -> bool:
    """当天收盘破 MA20 → True(需排除)"""
    if idx < 0 or idx >= len(close):
        return False
    return not np.isnan(ma20[idx]) and ma20[idx] > 0 and close[idx] < ma20[idx]


def _score_mode3(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
) -> int:
    close_arr = np.array([r.close for r in rows], dtype=float)
    close = rows[idx].close
    volume = rows[idx].volume
    score = 60.0

    ma20_now = ma20[idx]
    ma60_now = ma60[idx]
    ma10_now = ma10[idx]
    vol20_now = vol20[idx]

    if ma20_now > 0:
        gap = (ma10_now - ma20_now) / ma20_now
        if gap >= 0.02:
            score += 10
        elif gap >= 0.01:
            score += 6
        elif gap >= 0.005:
            score += 3

    if ma60_now > 0:
        gap = (ma20_now - ma60_now) / ma60_now
        if gap >= 0.02:
            score += 10
        elif gap >= 0.01:
            score += 6
        elif gap >= 0.005:
            score += 3

    if vol20_now > 0:
        vol_ratio = volume / vol20_now
        if vol_ratio >= 1.6:
            score += 15
        elif vol_ratio >= 1.4:
            score += 10
        elif vol_ratio >= 1.2:
            score += 6

    if ma20_now > 0:
        close_gap = (close - ma20_now) / ma20_now
        if close_gap >= 0.03:
            score += 5
        elif close_gap >= 0.01:
            score += 3

    # 近3日涨幅超过20%则降分
    if idx >= 3:
        base_close = rows[idx - 3].close
        if base_close > 0:
            ret3 = (close - base_close) / base_close * 100
            if ret3 > 20:
                score -= 10
            elif ret3 > 15:
                score -= 5

    # 5日线拐头向下：今日MA5低于昨日MA5则降分
    if idx >= 5:
        ma5 = _moving_mean(close_arr, 5)
        if not (np.isnan(ma5[idx]) or np.isnan(ma5[idx - 1])) and ma5[idx] < ma5[idx - 1]:
            score -= 5

    return int(max(0, round(score)))


def scan_with_mode3(
    stock_list: List[StockItem],
    config: ScanConfig,
    cache_dir: str,
    progress_cb: Optional[Callable[[], None]] = None,
    local_only: bool = False,
    kline_loader: Optional[Callable[[StockItem], Optional[List[KlineRow]]]] = None,
    prefer_local: bool = False,
    cutoff_date: Optional[str] = None,
    start_date: Optional[str] = None,
    market_caps: Optional[Dict[str, float]] = None,
    avoid_big_candle: bool = False,
    big_candle_pct: float = 6.0,
    big_body_ratio: float = 0.6,
    prefer_upper_shadow: bool = False,
    require_upper_shadow: bool = False,
    upper_ratio_min: float = 0.30,
    upper_vol_min: float = 1.50,
    require_vol_ratio: bool = False,
    vol_ratio_min: float = 1.50,
    require_close_gap: bool = False,
    close_gap_max: float = 0.02,
    mode4_filters: bool = False,
    use_71x_standard: bool = False,
) -> List[ScanResult]:
    results: List[ScanResult] = []
    end_date = cutoff_date

    for item in stock_list:
        if progress_cb:
            progress_cb()
        cap_value = None
        if config.max_market_cap and market_caps is not None:
            cap_value = market_caps.get(_normalize_code(item.code))
            if cap_value is None:
                continue
            if cap_value > config.max_market_cap:
                continue
        try:
            if kline_loader:
                rows = kline_loader(item)
            else:
                from .eastmoney import get_kline_cached

                rows = get_kline_cached(
                    item.secid,
                    cache_dir=cache_dir,
                    count=max(260, config.year_lookback + 5),
                    max_age_days=config.cache_days,
                    pause=0.0,
                    local_only=local_only,
                    prefer_local=prefer_local,
                )
        except Exception:
            rows = None
        if not rows or len(rows) < 80:
            continue

        if end_date:
            filtered = []
            end_dt = _parse_date(end_date)
            if end_dt:
                for row in rows:
                    try:
                        row_dt = datetime.strptime(row.date, "%Y-%m-%d").date()
                    except Exception:
                        continue
                    if row_dt <= end_dt:
                        filtered.append(row)
                rows = filtered

        if len(rows) < config.year_lookback + 5:
            continue

        # 标准71倍模型（与脚本一致）不做一年高低价比过滤；否则排除 1年最高/最低>=4 倍
        if not use_71x_standard and len(rows) >= config.year_lookback:
            window = rows[-config.year_lookback:]
            max_high = max(r.high for r in window)
            min_low = min(r.low for r in window)
            if min_low > 0 and max_high / min_low >= config.year_high_low_ratio_limit:
                continue

        signals = _mode3_signals(rows, start_date, end_date)
        if cutoff_date and not start_date:
            signals = [s for s in signals if rows[s].date == cutoff_date]
        if not start_date and not cutoff_date and signals:
            signals = [signals[-1]]
        if not signals:
            continue

        close = np.array([r.close for r in rows], dtype=float)
        volume = np.array([r.volume for r in rows], dtype=float)
        ma10 = _moving_mean(close, 10)
        ma20 = _moving_mean(close, 20)
        ma60 = _moving_mean(close, 60)
        vol20 = _moving_mean(volume, 20)
        ret20 = np.full_like(close, np.nan, dtype=float)
        if len(close) > 20:
            for i in range(20, len(close)):
                base = close[i - 20]
                if base > 0:
                    ret20[i] = (close[i] - base) / base * 100

        for idx in signals:
            if np.isnan(ma20[idx]) or np.isnan(ma60[idx]) or np.isnan(vol20[idx]):
                continue

            if avoid_big_candle:
                o = rows[idx].open
                c = rows[idx].close
                h = rows[idx].high
                l = rows[idx].low
                rng = h - l
                body = abs(c - o)
                body_ratio = body / rng if rng > 0 else 0.0
                is_big_bull = (
                    c > o
                    and rows[idx].pct_chg >= big_candle_pct
                    and body_ratio >= big_body_ratio
                )
                if is_big_bull:
                    continue

            if mode4_filters:
                # 放宽：仅排除连续3天涨停后跌停（原2天过严），移除当天破MA20（mode3已要求close>=ma20）
                if _has_limit_up_then_down(rows, idx, item.code, item.name, lookback=5, min_consec_limit_up=3):
                    continue

            score = _score_mode3(rows, idx, ma10, ma20, ma60, vol20)
            if score < config.min_score:
                continue

            buy_idx = min(idx + 1, len(rows) - 1)
            signal_date = rows[idx].date
            buy_date = rows[buy_idx].date

            vol_ratio = volume[idx] / vol20[idx] if vol20[idx] > 0 else 0.0
            ma20_now = ma20[idx]
            ma60_now = ma60[idx]
            ma10_now = ma10[idx]
            ma20_gap = (ma10_now - ma20_now) / ma20_now if ma20_now > 0 else 0.0
            ma60_gap = (ma20_now - ma60_now) / ma60_now if ma60_now > 0 else 0.0
            close_gap = abs(close[idx] - ma20_now) / ma20_now if ma20_now > 0 else 0.0
            ret20_val = ret20[idx] if not np.isnan(ret20[idx]) else 0.0
            o = rows[idx].open
            c = rows[idx].close
            h = rows[idx].high
            l = rows[idx].low
            rng = h - l
            upper = h - max(o, c)
            upper_ratio = upper / rng if rng > 0 else 0.0
            upper_score = upper_ratio * vol_ratio
            if require_upper_shadow:
                if upper_ratio < upper_ratio_min or vol_ratio < upper_vol_min:
                    continue
            if require_vol_ratio and vol_ratio < vol_ratio_min:
                continue
            if require_close_gap and close_gap > close_gap_max:
                continue
            reasons = [
                "启动点 mode4" if mode4_filters else "启动点 mode3",
                f"信号日 {signal_date}",
                f"买入日 {buy_date} (T+1 开盘)",
                f"放量 {vol_ratio:.2f}x",
                f"MA10-20 {ma20_gap:.2%}",
                f"MA20-60 {ma60_gap:.2%}",
                f"距MA20 {close_gap:.2%}",
                f"20日涨幅 {ret20_val:.2f}%",
                f"上影占比 {upper_ratio:.2%}",
            ]

            results.append(
                ScanResult(
                    code=item.code,
                    name=item.name,
                    score=int(score),
                    latest_close=float(rows[-1].close),
                    change_pct=float(rows[-1].pct_chg),
                    reasons=reasons,
                    metrics={
                        "signal_date": signal_date,
                        "buy_date": buy_date,
                        "vol_ratio": float(vol_ratio),
                        "ma20_gap": float(ma20_gap),
                        "ma60_gap": float(ma60_gap),
                        "close_gap": float(close_gap),
                        "ret20": float(ret20_val),
                        "upper_ratio": float(upper_ratio),
                        "upper_score": float(upper_score),
                        "market_cap": float(cap_value) if cap_value is not None else None,
                    },
                )
            )

    def _mode3_sort_key(r: ScanResult):
        metrics = r.metrics or {}
        vol_ratio = float(metrics.get("vol_ratio", 0.0))
        ma20_gap = float(metrics.get("ma20_gap", 0.0))
        ma60_gap = float(metrics.get("ma60_gap", 0.0))
        close_gap = float(metrics.get("close_gap", 0.0))
        ret20_val = float(metrics.get("ret20", 0.0))
        upper_score = float(metrics.get("upper_score", 0.0))
        if prefer_upper_shadow:
            return (
                -r.score,
                -upper_score,
                close_gap,
                -vol_ratio,
                -(ma20_gap + ma60_gap),
                ret20_val,
                r.code,
            )
        return (-r.score, close_gap, -vol_ratio, -(ma20_gap + ma60_gap), ret20_val, r.code)

    results.sort(key=_mode3_sort_key)

    # 一段时间选股：每日取分数最高的3只，不受 max_results 限制
    if start_date:
        grouped: Dict[str, List[ScanResult]] = {}
        for r in results:
            sig = (r.metrics or {}).get("signal_date")
            if sig:
                grouped.setdefault(str(sig), []).append(r)
        out: List[ScanResult] = []
        for day in sorted(grouped.keys()):
            group = grouped[day]
            group.sort(key=_mode3_sort_key)
            out.extend(group[:3])
        return out

    return results[: config.max_results]


def serialize_results(results: List[ScanResult]) -> List[Dict[str, object]]:
    return [
        {
            **asdict(r),
            "reasons": ", ".join(r.reasons),
        }
        for r in results
    ]
