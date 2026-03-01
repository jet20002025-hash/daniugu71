import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np

from .eastmoney import KlineRow, StockItem
from .weekly_ma import (
    daily_to_weekly_with_last_index,
    daily_to_weekly_with_volume_and_last_index,
    _rolling_mean,
    weekly_convergence_value_series,
)


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
    mode8_n_bars: int = 60  # mode8 起算 K 线根数（70/80/90 等），仅 use_mode8 时生效
    mode10_conv_max: float = 1.0  # mode10 买点前周线拟合值上限（%），满足 拟合<conv_max 才出信号
    mode10_ma30_turn_weeks: int = 5  # mode10 信号周前 N 周内需存在 MA30 拐头向上
    mode11_accel_th: float = 2.5  # mode11 同 mode10
    mode11_body_ratio_max: float = 0.35  # mode11 拐点形态：实体/振幅上限
    mode11_vol_ratio_min: float = 1.5  # mode11 当周量 >= 该倍数 * 过去 N 周均量
    mode11_vol_weeks: int = 20  # mode11 均量回溯周数
    mode12_accel_th: float = 2.5  # mode12 同 mode10
    mode12_ma30_turn_weeks: int = 5  # mode12 信号周前 N 周内需存在周线 MA30 拐头向上
    mode88_d_min: float = 0.03  # 震仓周相对前周最小跌幅
    mode88_d_max: float = 0.15  # 震仓周相对前周最大跌幅
    mode88_r_min: float = 0.03  # 第3周相对第2周最小涨幅
    mode88_acc_L: int = 8
    mode88_acc_R: int = 20
    mode88_A_min: float = 15.0
    mode88_A_max: float = 55.0
    mode88_epsilon: float = 0.02
    mode88_wash_L: int = 2
    mode88_wash_R: int = 10
    mode88_R_rise: float = 8.0
    mode88_D_pull: float = 3.0
    mode88_K_vol: float = 1.0


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


def _ema_series(close: np.ndarray, n: int) -> np.ndarray:
    """指数移动平均，用于周线 MACD。"""
    out = np.full_like(close, np.nan, dtype=float)
    if len(close) < n or n <= 0:
        return out
    alpha = 2.0 / (n + 1)
    out[n - 1] = float(np.nanmean(close[:n]))
    for i in range(n, len(close)):
        if np.isnan(close[i]):
            continue
        if np.isnan(out[i - 1]):
            out[i] = close[i]
        else:
            out[i] = alpha * close[i] + (1.0 - alpha) * out[i - 1]
    return out


def _weekly_macd_dif_dea(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """周线 MACD：DIF = EMA(close,12)-EMA(close,26)，DEA = EMA(DIF,9)。
    返回 (dif_norm, dea_norm)：归一化值 = 原值 / 当周收盘价，便于跨股票、跨价位比较。"""
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    dif = np.full_like(close, np.nan, dtype=float)
    for i in range(slow - 1, len(close)):
        if not (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i])):
            dif[i] = ema_fast[i] - ema_slow[i]
    dea = np.full_like(close, np.nan, dtype=float)
    alpha_sig = 2.0 / (signal + 1)
    start = slow - 1
    if start < len(close) and not np.isnan(dif[start]):
        dea[start] = dif[start]
        for i in range(start + 1, len(close)):
            if np.isnan(dif[i]):
                continue
            dea[i] = alpha_sig * dif[i] + (1.0 - alpha_sig) * dea[i - 1]
    dif_norm = np.full_like(close, np.nan, dtype=float)
    dea_norm = np.full_like(close, np.nan, dtype=float)
    for i in range(len(close)):
        if close[i] > 0:
            if not np.isnan(dif[i]):
                dif_norm[i] = dif[i] / close[i]
            if not np.isnan(dea[i]):
                dea_norm[i] = dea[i] / close[i]
    return dif_norm, dea_norm


def _mode18_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[int]:
    """
    mode18 信号：周线 MACD 金叉买入。条件：DIF 上穿 DEA（金叉），且 DIF、DEA 均在 0 轴以上（MACD 值由负变正）。
    信号日 = 当周最后交易日，买点 = 下一交易日开盘。
    """
    if not rows or len(rows) < 200:
        return []
    weekly_bars, last_indices = daily_to_weekly_with_last_index(rows)
    if len(weekly_bars) < 35:
        return []
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    dif_norm, dea_norm = _weekly_macd_dif_dea(closes, 12, 26, 9)
    signal_indices = []
    for i in range(34, len(weekly_bars)):
        if np.isnan(dif_norm[i]) or np.isnan(dea_norm[i]) or np.isnan(dif_norm[i - 1]) or np.isnan(dea_norm[i - 1]):
            continue
        if dif_norm[i] <= 0 or dea_norm[i] <= 0:
            continue
        if not (dif_norm[i] > dea_norm[i] and dif_norm[i - 1] <= dea_norm[i - 1]):
            continue
        idx = last_indices[i]
        if idx >= len(rows):
            continue
        d = rows[idx].date
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        signal_indices.append(idx)
    return sorted(signal_indices)


def _score_mode18(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
) -> int:
    """mode18 评分：以当周归一化 DIF（DIF/收盘价）为权重，可比 across 股票，50～100。"""
    if idx < 0 or idx >= len(rows):
        return 50
    sub = rows[: idx + 1]
    if len(sub) < 200:
        return 50
    weekly_bars, _ = daily_to_weekly_with_last_index(sub)
    if len(weekly_bars) < 35:
        return 50
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    dif_norm, dea_norm = _weekly_macd_dif_dea(closes, 12, 26, 9)
    wi = len(weekly_bars) - 1
    if np.isnan(dif_norm[wi]) or dif_norm[wi] <= 0:
        return 50
    score = 50 + min(50, dif_norm[wi] * 2500)
    return int(max(50, min(100, round(score))))


def _mode88_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
    d_min: float = 0.03,
    d_max: float = 0.15,
    r_min: float = 0.03,
    acc_L: int = 8,
    acc_R: int = 20,
    A_min: float = 15.0,
    A_max: float = 55.0,
    epsilon: float = 0.02,
    wash_L: int = 2,
    wash_R: int = 10,
    R_rise: float = 8.0,
    D_pull: float = 3.0,
    K_vol: float = 1.0,
) -> List[int]:
    """
    mode88：吸筹 → 洗盘 → 震仓 → 拉升。信号日 = 震仓第 3 周最后交易日，买点 = 下一交易日开盘。
    仅当震仓三周形态成立且震仓前同时满足吸筹、洗盘时出信号。
    """
    if not rows or len(rows) < 260:
        return []
    weekly_bars, last_indices = daily_to_weekly_with_volume_and_last_index(rows)
    nw = len(weekly_bars)
    if nw < 25:
        return []
    # 周线: (week_key, open, high, low, close, volume)
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    highs = np.array([w[2] for w in weekly_bars], dtype=float)
    lows = np.array([w[3] for w in weekly_bars], dtype=float)
    vols = np.array([w[5] for w in weekly_bars], dtype=float)
    signal_indices = []
    for i in range(acc_R + wash_R + 2, nw):
        i1 = i - 2
        i2 = i - 1
        c1, c2, c3 = closes[i1], closes[i2], closes[i]
        v1, v2 = vols[i1], vols[i2]
        l2 = lows[i2]
        if c1 <= 0 or c2 <= 0 or c3 <= 0:
            continue
        if not (c2 < c1 and v2 < v1 and c3 > c2):
            continue
        drop_pct = (c1 - c2) / c1
        if drop_pct < d_min or drop_pct > d_max:
            continue
        if (c3 - c2) / c2 < r_min:
            continue
        shakeout_i = i2
        acc_start = shakeout_i - acc_R
        acc_end = shakeout_i - acc_L + 1
        if acc_start < 0 or acc_end <= acc_start + 2:
            continue
        acc_high = float(np.nanmax(highs[acc_start:acc_end]))
        acc_low = float(np.nanmin(lows[acc_start:acc_end]))
        acc_mid = (acc_high + acc_low) / 2
        if acc_mid <= 0:
            continue
        amplitude = (acc_high - acc_low) / acc_mid * 100
        if amplitude < A_min or amplitude > A_max:
            continue
        if l2 < acc_low * (1 - epsilon):
            continue
        wash_start = shakeout_i - wash_R
        wash_end = shakeout_i - wash_L + 1
        if wash_start < 0 or wash_end <= wash_start + 2:
            continue
        wash_high = float(np.nanmax(highs[wash_start:wash_end]))
        wash_low = float(np.nanmin(lows[wash_start:wash_end]))
        if wash_low <= 0:
            continue
        phase_rise = (wash_high - wash_low) / wash_low * 100
        if phase_rise < R_rise:
            continue
        close_before = closes[shakeout_i - 1]
        if wash_high <= 0:
            continue
        pullback = (wash_high - close_before) / wash_high * 100
        if pullback < D_pull:
            continue
        high_idx = wash_start + int(np.nanargmax(highs[wash_start:wash_end]))
        vol_up_slice = vols[wash_start : high_idx + 1]
        vol_down_slice = vols[high_idx : shakeout_i]
        if len(vol_up_slice) == 0 or len(vol_down_slice) == 0:
            continue
        v_up = float(np.nanmean(vol_up_slice))
        v_down = float(np.nanmean(vol_down_slice))
        if v_up <= 0 or v_down > v_up * K_vol:
            continue
        idx = last_indices[i]
        if idx >= len(rows):
            continue
        d = rows[idx].date
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        signal_indices.append(idx)
    return sorted(signal_indices)


def _score_mode88(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
) -> int:
    """
    mode88 评分：吸筹/洗盘/震仓形态 + 拉升周强度 + 周线 MA20/MA10/MACD + 日线偏离MA20。无上下限，按各项加减分汇总。
    特征：第3周涨幅/放量、震仓跌幅、不破前低缓冲；周线 MA20/MA10/MACD；日线收盘偏离日线MA20分档扣分（>15%扣5，>12%扣3，>8%扣2，>5%扣1）。
    """
    if idx < 0 or idx >= len(rows):
        return 50
    sub = rows[: idx + 1]
    if len(sub) < 260:
        return 50
    weekly_bars, _ = daily_to_weekly_with_volume_and_last_index(sub)
    nw = len(weekly_bars)
    if nw < 25:
        return 50
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    highs = np.array([w[2] for w in weekly_bars], dtype=float)
    lows = np.array([w[3] for w in weekly_bars], dtype=float)
    vols = np.array([w[5] for w in weekly_bars], dtype=float)
    i = nw - 1
    i2 = i - 1
    i1 = i - 2
    shakeout_i = i2
    c1, c2, c3 = closes[i1], closes[i2], closes[i]
    v1, v2, v3 = vols[i1], vols[i2], vols[i]
    l2 = lows[i2]
    # 第3周涨幅（拉升周强度，与后续收益正相关）
    week3_rise = (c3 - c2) / c2 * 100 if c2 > 0 else 0
    week3_volume_up = v3 > v2 if v2 > 0 else False
    shakeout_drop_pct = (c1 - c2) / c1 * 100 if c1 > 0 else 0

    acc_start = shakeout_i - 20
    acc_end = shakeout_i - 8 + 1
    if acc_start < 0 or acc_end <= acc_start + 2:
        return 50
    acc_high = float(np.nanmax(highs[acc_start:acc_end]))
    acc_low = float(np.nanmin(lows[acc_start:acc_end]))
    acc_mid = (acc_high + acc_low) / 2
    amplitude = (acc_high - acc_low) / acc_mid * 100 if acc_mid > 0 else 0
    # 震仓周最低相对吸筹低的缓冲（不破且留有余地加分）
    hold_buffer = (l2 - acc_low) / acc_low * 100 if acc_low > 0 else 0

    wash_start = shakeout_i - 10
    wash_end = shakeout_i - 2 + 1
    if wash_start < 0 or wash_end <= wash_start + 2:
        return 50
    wash_high = float(np.nanmax(highs[wash_start:wash_end]))
    wash_low = float(np.nanmin(lows[wash_start:wash_end]))
    phase_rise = (wash_high - wash_low) / wash_low * 100 if wash_low > 0 else 0
    close_before = closes[shakeout_i - 1]
    pullback = (wash_high - close_before) / wash_high * 100 if wash_high > 0 else 0
    high_idx = wash_start + int(np.nanargmax(highs[wash_start:wash_end]))
    v_up = float(np.nanmean(vols[wash_start : high_idx + 1]))
    v_down = float(np.nanmean(vols[high_idx : shakeout_i]))
    vol_ratio = (v_down / v_up) if v_up > 0 else 1.0

    # 周线 MA20 方向：信号周 MA20 与上周比较（与买点后收益正相关）
    ma20_weekly = np.full_like(closes, np.nan, dtype=float)
    for j in range(19, len(closes)):
        ma20_weekly[j] = float(np.mean(closes[j - 19 : j + 1]))
    ma20_up = (
        ma20_weekly[i] > ma20_weekly[i - 1]
        if (i >= 20 and not np.isnan(ma20_weekly[i]) and not np.isnan(ma20_weekly[i - 1]))
        else False
    )
    # 周线 MA10 方向：信号周 MA10 与上周比较，向下扣分
    ma10_weekly = np.full_like(closes, np.nan, dtype=float)
    for j in range(9, len(closes)):
        ma10_weekly[j] = float(np.mean(closes[j - 9 : j + 1]))
    ma10_up = (
        ma10_weekly[i] > ma10_weekly[i - 1]
        if (i >= 10 and not np.isnan(ma10_weekly[i]) and not np.isnan(ma10_weekly[i - 1]))
        else False
    )
    # 周线 MACD：信号周 DIF 与 DEA 比较，金叉加分、死叉扣分
    dif_norm, dea_norm = _weekly_macd_dif_dea(closes, 12, 26, 9)
    macd_golden = (
        dif_norm[i] > dea_norm[i]
        if (i >= 34 and not np.isnan(dif_norm[i]) and not np.isnan(dea_norm[i]))
        else None
    )
    # 日线偏离 MA20：信号日收盘相对日线 MA20 的偏离幅度，分档扣分（偏离越大扣分越多）
    day_close = float(rows[idx].close)
    ma20_day = float(ma20[idx]) if idx < len(ma20) and not np.isnan(ma20[idx]) and ma20[idx] > 0 else None
    if ma20_day is not None and ma20_day > 0:
        day_deviation_pct = abs(day_close - ma20_day) / ma20_day * 100
    else:
        day_deviation_pct = None

    base = 50
    # 吸筹振幅
    if 18 <= amplitude <= 45:
        base += 9
    elif 15 <= amplitude <= 55:
        base += 5
    # 洗盘阶段涨幅
    if phase_rise >= 15:
        base += 10
    elif phase_rise >= 8:
        base += 5
    # 洗盘回撤+缩量
    if pullback >= 5 and vol_ratio <= 0.85:
        base += 10
    elif pullback >= 3 and vol_ratio <= 1.0:
        base += 5
    # 第3周涨幅（与后续收益正相关，权重大）
    if week3_rise >= 8:
        base += 10
    elif week3_rise >= 6:
        base += 7
    elif week3_rise >= 5:
        base += 6
    elif week3_rise >= 4:
        base += 3
    elif week3_rise < 3.5 and not week3_volume_up:
        base -= 2
    # 第3周放量（放量拉升加分）
    if week3_volume_up:
        base += 5
    # 震仓跌幅适中加分、过深略扣
    if 5 <= shakeout_drop_pct <= 10:
        base += 3
    elif 4 <= shakeout_drop_pct <= 11:
        base += 1
    elif shakeout_drop_pct > 12:
        base -= 1
    # 不破前低且有缓冲
    if hold_buffer >= 1:
        base += 2
    elif hold_buffer >= 0:
        base += 1
    # 周线 MA20 向上（权重大，与买点后收益正相关）
    if ma20_up:
        base += 8
    else:
        base -= 3
    # 周线 MA10 向下扣分（权重较 MA20 小）
    if not ma10_up:
        base -= 1
    # 周线 MACD 金叉加分、死叉扣分
    if macd_golden is not None:
        if macd_golden:
            base += 4
        else:
            base -= 3
    # 日线偏离 MA20 分档扣分（追高或远离均线风险）
    if day_deviation_pct is not None:
        if day_deviation_pct > 15:
            base -= 5
        elif day_deviation_pct > 12:
            base -= 3
        elif day_deviation_pct > 8:
            base -= 2
        elif day_deviation_pct > 5:
            base -= 1
    return int(round(base))


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
    """
    找出所有满足 mode3 启动点的信号日下标。
    测算起点：从信号日当天往前至少 60 根 K 线参与计算（MA60、vol20 等），
    即从买点前约 60 个交易日开始数据就参与测算；第一个可能出信号的日期是第 61 根 K 线（下标 60）。
    """
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


def _mode9_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[int]:
    """mode9：与 mode3（71倍）完全一致的信号逻辑，复制一份便于独立调参或扩展。"""
    return _mode3_signals(rows, start_date, end_date)


def _mode8_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[int]:
    """
    mode8 信号（大牛股买点）：在 mode3 基础上放宽 20 日涨幅、增加 60 日涨幅过滤。
    - 20日涨幅 ≤ 50%（mode3 为 25%）；
    - 买点前60日涨幅 -15% ≤ ret60 ≤ 50%（需至少 60 根 K 线）。
    其余与 mode3 一致：MA10>MA20>MA60、收盘≥MA20、volume≥vol20×1.2。见 docs/mode8模型说明.md
    """
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
        # mode8: 20日涨幅 ≤ 50%（mode3 为 25%）
        if i - 20 >= 0 and close[i - 20] > 0:
            ret20 = (close[i] - close[i - 20]) / close[i - 20] * 100
            if ret20 > 50:
                continue
        # mode8: 买点前60日涨幅 -15% ≤ ret60 ≤ 50%
        if close[i - 60] > 0:
            ret60 = (close[i] - close[i - 60]) / close[i - 60] * 100
            if ret60 < -15 or ret60 > 50:
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


def _mode10_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
    conv_max: float = 1.0,
    ma30_turn_weeks: int = 5,
) -> List[int]:
    """
    mode10 信号：5 周内 MA30 拐头 + 当周 MA5>MA10>MA20 多头向上 + 买点前周线拟合 < conv_max（%）。
    返回满足条件的「当周最后交易日」在 rows 中的下标列表。
    """
    if not rows or len(rows) < 100:
        return []
    weekly_bars, last_indices = daily_to_weekly_with_last_index(rows)
    if len(weekly_bars) < 32:
        return []
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    ma5 = _moving_mean(closes, 5)
    ma10 = _moving_mean(closes, 10)
    ma20 = _moving_mean(closes, 20)
    ma30 = _moving_mean(closes, 30)
    conv = weekly_convergence_value_series(weekly_bars)
    if len(conv) == 0:
        return []
    signal_indices = []
    for i in range(30, len(weekly_bars)):
        if np.isnan(ma5[i]) or np.isnan(ma10[i]) or np.isnan(ma20[i]):
            continue
        if not (ma5[i] > ma10[i] > ma20[i]):
            continue
        if not _has_ma30_turn_in_weeks(weekly_bars, ma30, i, ma30_turn_weeks):
            continue
        conv_min = np.nanmin(conv[30:i]) if i > 30 else np.nan
        if np.isnan(conv_min) or conv_min >= conv_max:
            continue
        idx = last_indices[i]
        if idx >= len(rows):
            continue
        d = rows[idx].date
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        signal_indices.append(idx)
    return sorted(signal_indices)


def _has_ma30_turn_in_weeks(weekly_bars: List[tuple], ma30: np.ndarray, signal_week_i: int, within_weeks: int) -> bool:
    """信号周 signal_week_i 的前 within_weeks 周内是否存在周线 MA30 由下转上的拐点。"""
    if signal_week_i < 32 or within_weeks <= 0:
        return False
    lo = max(31, signal_week_i - within_weeks)
    hi = signal_week_i
    for j in range(lo, hi):
        if np.isnan(ma30[j]) or np.isnan(ma30[j - 1]) or np.isnan(ma30[j - 2]):
            continue
        if ma30[j] > ma30[j - 1] and ma30[j - 1] < ma30[j - 2]:
            return True
    return False


def _nearest_ma30_turn_weeks_before(ma30: np.ndarray, signal_week_i: int, within_weeks: int) -> Optional[int]:
    """信号周前 within_weeks 周内，距离信号周最近的拐点周数（1=前1周，2=前2周…），无则 None。"""
    if signal_week_i < 32 or within_weeks <= 0:
        return None
    lo = max(31, signal_week_i - within_weeks)
    for j in range(signal_week_i - 1, lo - 1, -1):
        if np.isnan(ma30[j]) or np.isnan(ma30[j - 1]) or np.isnan(ma30[j - 2]):
            continue
        if ma30[j] > ma30[j - 1] and ma30[j - 1] < ma30[j - 2]:
            return signal_week_i - j
    return None


def _mode12_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
    accel_th: float = 2.5,
    ma30_turn_weeks: int = 5,
) -> List[int]:
    """
    mode12 信号：mode10（周线 MA5 斜率突变）+ 5 周内存在周线 MA30 拐头向上。
    拐点 = MA30 由下转上（当周 > 上周 且 上周 < 上上周）。仅保留信号周前 ma30_turn_weeks 周内有拐点的信号。
    """
    if not rows or len(rows) < 100:
        return []
    weekly_bars, last_indices = daily_to_weekly_with_last_index(rows)
    if len(weekly_bars) < 32:
        return []
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    ma5 = _moving_mean(closes, 5)
    ma30 = _rolling_mean(closes, 30)
    slope = np.full_like(ma5, np.nan, dtype=float)
    for i in range(1, len(ma5)):
        if np.isnan(ma5[i]) or np.isnan(ma5[i - 1]) or ma5[i - 1] <= 0:
            continue
        slope[i] = (ma5[i] - ma5[i - 1]) / ma5[i - 1] * 100.0
    accel = np.full_like(slope, np.nan, dtype=float)
    for i in range(2, len(slope)):
        if np.isnan(slope[i]) or np.isnan(slope[i - 1]):
            continue
        accel[i] = slope[i] - slope[i - 1]
    signal_indices = []
    for i in range(2, len(accel)):
        if np.isnan(accel[i]) or accel[i] < accel_th:
            continue
        if not _has_ma30_turn_in_weeks(weekly_bars, ma30, i, ma30_turn_weeks):
            continue
        idx = last_indices[i]
        if idx >= len(rows):
            continue
        d = rows[idx].date
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        signal_indices.append(idx)
    return sorted(signal_indices)


def _mode11_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
    accel_th: float = 2.5,
    body_ratio_max: float = 0.35,
    vol_ratio_min: float = 1.5,
    vol_weeks: int = 20,
) -> List[int]:
    """
    mode11 信号：mode10（周线 MA5 斜率突变）+ 拐点形态过滤。
    - 当周 K 线小实体长影线：|close-open|/(high-low) < body_ratio_max；
    - 当周成交量放量：周量 >= vol_ratio_min * 过去 vol_weeks 周均量。
    返回满足条件的「当周最后交易日」在 rows 中的下标列表。
    """
    if not rows or len(rows) < 100:
        return []
    weekly_bars, last_indices = daily_to_weekly_with_volume_and_last_index(rows)
    if len(weekly_bars) < 18:
        return []
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    vols = np.array([w[5] for w in weekly_bars], dtype=float)
    ma5 = _moving_mean(closes, 5)
    slope = np.full_like(ma5, np.nan, dtype=float)
    for i in range(1, len(ma5)):
        if np.isnan(ma5[i]) or np.isnan(ma5[i - 1]) or ma5[i - 1] <= 0:
            continue
        slope[i] = (ma5[i] - ma5[i - 1]) / ma5[i - 1] * 100.0
    accel = np.full_like(slope, np.nan, dtype=float)
    for i in range(2, len(slope)):
        if np.isnan(slope[i]) or np.isnan(slope[i - 1]):
            continue
        accel[i] = slope[i] - slope[i - 1]
    signal_indices = []
    for i in range(2, len(accel)):
        if np.isnan(accel[i]) or accel[i] < accel_th:
            continue
        o, h, l, c = weekly_bars[i][1], weekly_bars[i][2], weekly_bars[i][3], weekly_bars[i][4]
        rng = h - l
        if rng > 0:
            body_ratio = abs(c - o) / rng
            if body_ratio >= body_ratio_max:
                continue
        else:
            continue
        lo = max(0, i - vol_weeks)
        vol_avg = np.mean(vols[lo:i]) if lo < i else vols[i]
        if vol_avg <= 0:
            continue
        if vols[i] < vol_ratio_min * vol_avg:
            continue
        idx = last_indices[i]
        if idx >= len(rows):
            continue
        d = rows[idx].date
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        signal_indices.append(idx)
    return sorted(signal_indices)


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


def _has_limit_up_6d(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    lookback: int = 6,
) -> bool:
    """
    买点日前 lookback 个交易日内是否出现过涨停（仅判定有/无，不要求缩量）。
    涨停阈值与 _limit_rate 一致：按 ST / 创业板 / 科创板 / 主板 的涨停幅度计算。
    """
    if idx < 1:
        return False
    rate = _limit_rate(code, name)
    limit_up = (rate * 100) - 0.5
    start = max(1, idx - lookback)
    for i in range(start, idx):
        if rows[i].pct_chg >= limit_up:
            return True
    return False


def _has_limit_up_then_shrink_volume(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    lookback: int = 6,
    next_vol_max_mult: float = 1.8,
) -> bool:
    """
    买点前 lookback 个交易日内出现涨停，且涨停后1个交易日成交量 < 涨停日成交量 * next_vol_max_mult → True(加分特征)。
    仅用当日及历史数据；涨停阈值按代码板块与 ST 规则计算。
    """
    if idx < 2:
        return False
    rate = _limit_rate(code, name)
    limit_up = (rate * 100) - 0.5
    start = max(1, idx - lookback)
    for i in range(start, idx):
        if rows[i].pct_chg < limit_up:
            continue
        if i + 1 >= len(rows):
            continue
        v0 = rows[i].volume
        v1 = rows[i + 1].volume
        if v0 > 0 and v1 < v0 * next_vol_max_mult:
            return True
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
    score = 40.0  # 基础分（原 60）

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


def _score_mode9(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
) -> int:
    """
    mode9 评分：在 mode3 基础上微调，使选股更准。
    基于「满分100 表现最好5只 vs 最差5只」买点前特征对比：
    - 收盘距MA20 过远(>8%)略降分，偏好温和突破；
    - MA20-MA60 开口大(趋势强)额外加分；
    - 当日实体占比大(阳线实在)加分；
    - 60日涨幅适中(15%～40%)略加分；
    - MA5 斜率过于陡峭(5日内MA5涨幅>15%)降分，避免短期冲得过猛（如金时科技 17.44%）。
    - 量能放大太快（多看3～5日）：近3日均量/再前3日>2 或 当日量/5日前量>2.8 降分，可能快到顶。
    - 均线整齐度（参考明阳电路 vs 神开股份）：当日 MA5>MA10>MA20 加分；近5日内均线交叉次数多则降分。
    """
    base = _score_mode3(rows, idx, ma10, ma20, ma60, vol20)
    close_arr = np.array([r.close for r in rows], dtype=float)
    volume = np.array([r.volume for r in rows], dtype=float)
    close = rows[idx].close
    ma20_now = ma20[idx]
    ma60_now = ma60[idx]
    if ma20_now <= 0:
        return base
    # 前一日收盘价跌破 MA10 且前一日最低价跌破 MA20：均线支撑弱，扣分（如中国电影 2月13日信号前一日）
    if idx >= 1:
        prev_close = rows[idx - 1].close
        prev_low = rows[idx - 1].low
        ma10_prev = ma10[idx - 1]
        ma20_prev = ma20[idx - 1]
        if (
            not (np.isnan(ma10_prev) or np.isnan(ma20_prev))
            and ma10_prev > 0
            and ma20_prev > 0
            and prev_close < ma10_prev
            and prev_low < ma20_prev
        ):
            base -= 4
            if breakdown is not None:
                breakdown.append(("前一日收盘破MA10且最低破MA20", -4))
    close_gap = (close - ma20_now) / ma20_now
    # 收盘距MA20 过远降分（最好组 10.34% vs 最差 12.53%，温和突破更优）
    if close_gap > 0.08:
        base -= 2
        if breakdown is not None:
            breakdown.append(("收盘距MA20过远(>8%)", -2))
    # 涨停后缩量：买点前6日内有涨停，且涨停次日量 < 涨停日量 * 1.8，加分（缩量不松、承接好）
    if code and _has_limit_up_then_shrink_volume(rows, idx, code, name, lookback=6, next_vol_max_mult=1.8):
        base += 2
        if breakdown is not None:
            breakdown.append(("涨停后缩量", 2))
    # MA5 斜率过于陡峭降分（最好5只 MA5斜率 7～14%，最差中金时科技 17.44%、亚康 16.32%）
    ma5 = None
    if idx >= 5:
        ma5 = _moving_mean(close_arr, 5)
        if not (np.isnan(ma5[idx]) or np.isnan(ma5[idx - 5]) or ma5[idx - 5] <= 0):
            ma5_slope_pct = (ma5[idx] - ma5[idx - 5]) / ma5[idx - 5] * 100
            if ma5_slope_pct > 15:
                base -= 2
                if breakdown is not None:
                    breakdown.append(("MA5斜率过陡(>15%)", -2))
        # 均线整齐度：当日 MA5>MA10>MA20 加分（明阳电路式完美多头）
        if ma5 is not None and not (np.isnan(ma5[idx]) or np.isnan(ma10[idx]) or np.isnan(ma20[idx])):
            if ma5[idx] > ma10[idx] > ma20[idx]:
                base += 2
                if breakdown is not None:
                    breakdown.append(("均线整齐MA5>MA10>MA20", 2))
        # 近5日内均线交叉次数多则降分（神开股份式乱序）
        if idx >= 6 and ma5 is not None:
            crosses = 0
            for i in range(idx - 4, idx + 1):
                if i <= 0:
                    continue
                if not (np.isnan(ma5[i]) or np.isnan(ma5[i - 1]) or np.isnan(ma10[i]) or np.isnan(ma10[i - 1])):
                    if (ma5[i] - ma10[i]) * (ma5[i - 1] - ma10[i - 1]) < 0:
                        crosses += 1
                if not (np.isnan(ma10[i]) or np.isnan(ma10[i - 1]) or np.isnan(ma20[i]) or np.isnan(ma20[i - 1])):
                    if (ma10[i] - ma20[i]) * (ma10[i - 1] - ma20[i - 1]) < 0:
                        crosses += 1
            if crosses >= 2:
                base -= 2
                if breakdown is not None:
                    breakdown.append(("近5日均线交叉多(>=2次)", -2))
        # MA5 与 MA10 粘连降分（如美利云 2月9日两线几乎贴在一起，趋势不清晰）
        if ma5 is not None and ma10[idx] > 0:
            # 当日粘连：|MA5-MA10|/MA10 < 1%
            gap_pct = abs(ma5[idx] - ma10[idx]) / ma10[idx]
            if gap_pct < 0.01:
                base -= 2
                if breakdown is not None:
                    breakdown.append(("MA5与MA10粘连(<1%)", -2))
            # 近5日内曾粘连（含当日）
            elif idx >= 5:
                for i in range(idx - 4, idx + 1):
                    if i < 0 or np.isnan(ma5[i]) or np.isnan(ma10[i]) or ma10[i] <= 0:
                        continue
                    if abs(ma5[i] - ma10[i]) / ma10[i] < 0.01:
                        base -= 2
                        if breakdown is not None:
                            breakdown.append(("近5日内MA5与MA10曾粘连", -2))
                        break
        # MA5 近期拐头向下、当日强行拐回降分（如兴民智通 2月12日 MA5 向下，2月13日拐回，形态不稳）
        if ma5 is not None and idx >= 3:
            today_up = not (np.isnan(ma5[idx]) or np.isnan(ma5[idx - 1])) and ma5[idx] > ma5[idx - 1]
            if today_up:
                # 昨日或前日 MA5 曾向下
                recent_down = False
                for i in range(1, 3):
                    if idx - i < 1:
                        break
                    if not (np.isnan(ma5[idx - i]) or np.isnan(ma5[idx - i - 1])) and ma5[idx - i] < ma5[idx - i - 1]:
                        recent_down = True
                        break
                if recent_down:
                    base -= 2
                    if breakdown is not None:
                        breakdown.append(("MA5近期拐头向下当日拐回", -2))
    # 当日收盘价低于 MA5 / 大阴线跌破 MA5：仅跌破幅度较大时扣分（轻微跌破不扣，避免误杀如招商轮船）
    if ma5 is not None and not np.isnan(ma5[idx]) and ma5[idx] > 0 and close < ma5[idx]:
        break_ma5_pct = (ma5[idx] - close) / ma5[idx] * 100  # 收盘低于 MA5 的幅度%
        open_ = rows[idx].open
        is_big_bear = (close < open_) and (getattr(rows[idx], "pct_chg", 0) <= -1.0)  # 阴线且跌幅≥1%
        # 仅当跌破超过 2% 才扣「收盘低于 MA5」；大阴线且跌破超过 2% 再加大扣分，超过 3% 更大扣分
        if break_ma5_pct >= 2.0:
            base -= 5  # 跌破 MA5 超 2 个点：扣分
            if breakdown is not None:
                breakdown.append(("当日收盘价跌破MA5超2%", -5))
        if is_big_bear and break_ma5_pct >= 3.0:
            base -= 15  # 大阴线且跌破 MA5 超 3 个点：大扣分
            if breakdown is not None:
                breakdown.append(("当日大阴线且跌破MA5超3%", -15))
        elif is_big_bear and break_ma5_pct >= 2.0:
            base -= 10  # 大阴线且跌破 MA5 超 2 个点
            if breakdown is not None:
                breakdown.append(("当日大阴线且跌破MA5超2%", -10))
    if ma60_now > 0:
        ma20_60_gap = (ma20[idx] - ma60_now) / ma60_now
        # 均线多头开口适中加分；开口过大（过度加速）不再额外加分，极端大时略降分
        if 0.03 <= ma20_60_gap <= 0.09:
            base += 2
            if breakdown is not None:
                breakdown.append(("MA20-MA60开口适中", 2))
        elif ma20_60_gap > 0.12:
            base -= 2
            if breakdown is not None:
                breakdown.append(("MA20-MA60开口过大(>12%)", -2))
    # 当日 K 线实体占比大加分（最好组 74.9% vs 最差 50%）
    rng = rows[idx].high - rows[idx].low
    if rng > 0:
        body = abs(close - rows[idx].open)
        if body / rng >= 0.6:
            base += 2
            if breakdown is not None:
                breakdown.append(("当日K线实体占比>=60%", 2))
    if idx >= 60 and close_arr[idx - 60] > 0:
        ret60 = (close - close_arr[idx - 60]) / close_arr[idx - 60] * 100
        # 60 日涨幅适中略加分；涨幅过大视为趋势已走出较长一段，适当降分
        if 15 <= ret60 <= 40:
            base += 1
            if breakdown is not None:
                breakdown.append(("60日涨幅15%~40%", 1))
        elif ret60 > 45:
            d = 2 if ret60 <= 55 else 4
            base -= d
            if breakdown is not None:
                breakdown.append((f"60日涨幅过大(>{45}%)", -d))
    # 突破质量：近20日（不含当日）最高价视为平台前高，当前收盘相对前高的位置
    if idx >= 21:
        high_arr = np.array([r.high for r in rows], dtype=float)
        prev_high_20 = float(np.nanmax(high_arr[idx - 20 : idx]))  # 前20根K线最高
        if prev_high_20 > 0:
            break_gap_pct = (close - prev_high_20) / prev_high_20 * 100
            if -3.0 <= break_gap_pct <= 1.0:
                base += 2  # 贴近前高或略低于前高，蓄势待发（如御银、金开新能）
                if breakdown is not None:
                    breakdown.append(("贴近前高蓄势", 2))
            elif 1.0 < break_gap_pct <= 6.0:
                base += 3  # 适度突破前高（如华盛昌、神马电力）
                if breakdown is not None:
                    breakdown.append(("适度突破前高", 3))
            elif break_gap_pct > 10.0:
                base -= 2  # 已远离前高，追高
                if breakdown is not None:
                    breakdown.append(("已远离前高追高", -2))
    # 量能放大太快（多看3～5日）：最好组 近3日/再前3日 1.27、最差组 2.13；当日/5日前 最好2.71、最差3.06
    if idx >= 6:
        vol_3d_recent = (volume[idx] + volume[idx - 1] + volume[idx - 2]) / 3.0
        vol_3d_older = (volume[idx - 3] + volume[idx - 4] + volume[idx - 5]) / 3.0
        if vol_3d_older > 0 and vol_3d_recent / vol_3d_older > 2.0:
            base -= 2  # 近3日量能相对再前3日陡升，可能快到顶
            if breakdown is not None:
                breakdown.append(("近3日量能/再前3日>2倍", -2))
    if idx >= 5 and volume[idx - 5] > 0:
        if volume[idx] / volume[idx - 5] > 2.8:
            base -= 2  # 5日内量能放大超过2.8倍，放大过快
            if breakdown is not None:
                breakdown.append(("当日量/5日前量>2.8倍", -2))
    # 买点前3日内爆量（如哈尔斯 2月12日量是2月11日的4倍以上）：前3日内任一天量>=前一日3倍则扣分；除非该日在近2个月最低价附近（底部放量可豁免）。按比例加重扣分。
    if idx >= 4:
        low_arr = np.array([r.low for r in rows], dtype=float)
        for i in range(idx - 3, idx):
            if i < 1 or volume[i - 1] <= 0:
                continue
            vol_ratio = volume[i] / volume[i - 1]
            if vol_ratio < 3.0:
                continue
            # 爆量日 i，是否在近2个月最低价附近（约40日）
            start = max(0, i - 39)
            min_low_40 = np.nanmin(low_arr[start : i + 1])
            if np.isnan(min_low_40) or min_low_40 <= 0:
                deduct = min(10, 4 + int((vol_ratio - 3) * 2))  # 3倍起扣4分，每多1倍多扣2分，上限10
                base -= deduct
                if breakdown is not None:
                    breakdown.append((f"买点前3日内爆量(约{vol_ratio:.1f}倍)", -deduct))
                break
            # 该日最低或收盘在最低价 5% 以内视为「最低价附近」
            near_bottom = (low_arr[i] <= min_low_40 * 1.05) or (close_arr[i] <= min_low_40 * 1.05)
            if not near_bottom:
                deduct = min(10, 4 + int((vol_ratio - 3) * 2))  # 3倍扣4分，4倍扣6分，4.67倍扣7分，上限10
                base -= deduct
                if breakdown is not None:
                    breakdown.append((f"买点前3日内爆量(约{vol_ratio:.1f}倍)", -deduct))
                break
    # 当日量/前一日量：小于2倍不扣（满分），>=2倍按比例扣分，比例越大扣越多
    if idx >= 1 and volume[idx - 1] > 0:
        vol_ratio_prev = volume[idx] / volume[idx - 1]
        if vol_ratio_prev > 2.0:
            deduct = min(10, max(2, int((vol_ratio_prev - 2) * 4)))
            base -= deduct
            if breakdown is not None:
                breakdown.append((f"当日量/前一日量(约{vol_ratio_prev:.1f}倍)", -deduct))
    return int(max(0, base))  # 不封顶，允许超过 100


def _score_mode8(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
) -> int:
    """
    mode8 评分（大牛股买点）：在 mode3 评分基础上，按买点前60日涨幅加分。
    - 0% ≤ ret60 ≤ 35%：+5 分；
    - -10% ≤ ret60 < 0%：+2 分。
    见 docs/mode8模型说明.md
    """
    base = _score_mode3(rows, idx, ma10, ma20, ma60, vol20)
    if idx < 60:
        return base
    close = np.array([r.close for r in rows], dtype=float)
    if close[idx - 60] <= 0:
        return base
    ret60 = (close[idx] - close[idx - 60]) / close[idx - 60] * 100
    if 0 <= ret60 <= 35:
        base += 5
    elif -10 <= ret60 < 0:
        base += 2
    return int(max(0, base))


def _score_mode10(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
) -> int:
    """mode10 评分：以买点前周线拟合最小值为权重，拟合越小分越高。50 + (1 - conv_min)*50，conv_min 为百分比（<1），上限 100。"""
    if idx < 0 or idx >= len(rows):
        return 50
    sub = rows[: idx + 1]
    if len(sub) < 100:
        return 50
    weekly_bars, _ = daily_to_weekly_with_last_index(sub)
    if len(weekly_bars) < 32:
        return 50
    conv = weekly_convergence_value_series(weekly_bars)
    if len(conv) == 0:
        return 50
    target_date = rows[idx].date
    try:
        target_yr, target_wk = datetime.strptime(target_date, "%Y-%m-%d").date().isocalendar()[:2]
    except Exception:
        return 50
    week_keys = [w[0] for w in weekly_bars]
    if (target_yr, target_wk) not in week_keys:
        return 50
    wi = week_keys.index((target_yr, target_wk))
    if wi <= 30:
        return 50
    conv_min = float(np.nanmin(conv[30:wi]))
    if np.isnan(conv_min) or conv_min < 0:
        return 50
    # 拟合值 conv_min 已是百分比（1.0=1%），越小分越高：conv_min=0 -> 100, conv_min>=1 -> 50
    score = 50 + (1.0 - min(1.0, conv_min)) * 50
    return int(max(50, min(100, round(score))))


def _score_mode11(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
) -> int:
    """mode11 评分：与 mode10 一致，以当周 MA5 斜率加速度 50 + min(50, accel*10)。"""
    return _score_mode10(rows, idx, ma10, ma20, ma60, vol20, code, name)


def _score_mode12(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
) -> int:
    """mode12 评分：mode10 基础分（加速度）+ 拐点距离加权。拐点距信号 1～5 周分别加 10/8/6/4/2 分，总分上限 100。"""
    base = _score_mode10(rows, idx, ma10, ma20, ma60, vol20, code, name)
    if idx < 0 or idx >= len(rows):
        return base
    sub = rows[: idx + 1]
    if len(sub) < 100:
        return base
    weekly_bars, _ = daily_to_weekly_with_last_index(sub)
    if len(weekly_bars) < 32:
        return base
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    ma30 = _rolling_mean(closes, 30)
    target_date = rows[idx].date
    try:
        target_yr, target_wk = datetime.strptime(target_date, "%Y-%m-%d").date().isocalendar()[:2]
    except Exception:
        return base
    week_keys = [w[0] for w in weekly_bars]
    if (target_yr, target_wk) not in week_keys:
        return base
    wi = week_keys.index((target_yr, target_wk))
    dist = _nearest_ma30_turn_weeks_before(ma30, wi, 5)
    bonus = 0
    if dist is not None and 1 <= dist <= 5:
        bonus = [10, 8, 6, 4, 2][dist - 1]
    return min(100, base + bonus)


def _buy_point_score(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
) -> int:
    """买点分值 0～100：衡量当日作为买入点的质量（放量、距MA20适中、均线多头、非追高等）。"""
    close = rows[idx].close
    volume = rows[idx].volume
    score = 50.0
    ma20_now = ma20[idx]
    ma60_now = ma60[idx]
    ma10_now = ma10[idx]
    vol20_now = vol20[idx]

    # 放量：量比越大买点越可靠
    if vol20_now > 0:
        vol_ratio = volume / vol20_now
        if vol_ratio >= 1.6:
            score += 15
        elif vol_ratio >= 1.4:
            score += 12
        elif vol_ratio >= 1.2:
            score += 8

    # 收盘相对 MA20：适中突破给高分，追高扣分
    if ma20_now > 0:
        close_gap = (close - ma20_now) / ma20_now
        if 0 <= close_gap <= 0.01:
            score += 15
        elif close_gap <= 0.03:
            score += 12
        elif close_gap <= 0.05:
            score += 5
        elif close_gap > 0.05:
            score -= 5

    # 均线多头排列
    if ma20_now > 0 and ma60_now > 0 and ma10_now > ma20_now and ma20_now > ma60_now:
        score += 10

    # 上影线适中（有试探但不过长）
    rng = rows[idx].high - rows[idx].low
    if rng > 0:
        upper = rows[idx].high - max(rows[idx].open, rows[idx].close)
        upper_ratio = upper / rng
        if 0.2 <= upper_ratio <= 0.5:
            score += 5

    # 近 3 日涨幅过大则扣分（追高）
    if idx >= 3 and rows[idx - 3].close > 0:
        ret3 = (close - rows[idx - 3].close) / rows[idx - 3].close * 100
        if ret3 > 20:
            score -= 15
        elif ret3 > 15:
            score -= 8

    return int(max(0, min(100, round(score))))


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
    use_mode8: bool = False,
    use_mode9: bool = False,
    use_mode10: bool = False,
    use_mode11: bool = False,
    use_mode12: bool = False,
    use_mode18: bool = False,
    use_mode88: bool = False,
) -> List[ScanResult]:
    """use_mode8/9/10/11/12/18/88；mode18 为周线 MACD 金叉；mode88 为吸筹→洗盘→震仓→拉升。"""
    results: List[ScanResult] = []
    mode8_n_bars = getattr(config, "mode8_n_bars", 60)
    mode10_conv_max = getattr(config, "mode10_conv_max", 1.0)
    mode10_ma30_turn_weeks = getattr(config, "mode10_ma30_turn_weeks", 5)
    mode11_accel_th = getattr(config, "mode11_accel_th", 2.5)
    mode11_body_ratio_max = getattr(config, "mode11_body_ratio_max", 0.35)
    mode11_vol_ratio_min = getattr(config, "mode11_vol_ratio_min", 1.5)
    mode11_vol_weeks = getattr(config, "mode11_vol_weeks", 20)
    mode12_accel_th = getattr(config, "mode12_accel_th", 2.5)
    mode12_ma30_turn_weeks = getattr(config, "mode12_ma30_turn_weeks", 5)
    if use_mode12:
        signal_fn = lambda rows, start, end: _mode12_signals(
            rows, start, end,
            accel_th=mode12_accel_th,
            ma30_turn_weeks=mode12_ma30_turn_weeks,
        )
        score_fn = _score_mode12
        mode_label = "mode12"
    elif use_mode11:
        signal_fn = lambda rows, start, end: _mode11_signals(
            rows, start, end,
            accel_th=mode11_accel_th,
            body_ratio_max=mode11_body_ratio_max,
            vol_ratio_min=mode11_vol_ratio_min,
            vol_weeks=mode11_vol_weeks,
        )
        score_fn = _score_mode11
        mode_label = "mode11"
    elif use_mode10:
        signal_fn = lambda rows, start, end: _mode10_signals(
            rows, start, end,
            conv_max=mode10_conv_max,
            ma30_turn_weeks=mode10_ma30_turn_weeks,
        )
        score_fn = _score_mode10
        mode_label = "mode10"
    elif use_mode18:
        signal_fn = _mode18_signals
        score_fn = _score_mode18
        mode_label = "mode18"
    elif use_mode88:
        mode88_d_min = getattr(config, "mode88_d_min", 0.03)
        mode88_d_max = getattr(config, "mode88_d_max", 0.15)
        mode88_r_min = getattr(config, "mode88_r_min", 0.03)
        mode88_acc_L = getattr(config, "mode88_acc_L", 8)
        mode88_acc_R = getattr(config, "mode88_acc_R", 20)
        mode88_A_min = getattr(config, "mode88_A_min", 15.0)
        mode88_A_max = getattr(config, "mode88_A_max", 55.0)
        mode88_epsilon = getattr(config, "mode88_epsilon", 0.02)
        mode88_wash_L = getattr(config, "mode88_wash_L", 2)
        mode88_wash_R = getattr(config, "mode88_wash_R", 10)
        mode88_R_rise = getattr(config, "mode88_R_rise", 8.0)
        mode88_D_pull = getattr(config, "mode88_D_pull", 3.0)
        mode88_K_vol = getattr(config, "mode88_K_vol", 1.0)
        signal_fn = lambda rows, start, end: _mode88_signals(
            rows, start, end,
            d_min=mode88_d_min, d_max=mode88_d_max, r_min=mode88_r_min,
            acc_L=mode88_acc_L, acc_R=mode88_acc_R,
            A_min=mode88_A_min, A_max=mode88_A_max, epsilon=mode88_epsilon,
            wash_L=mode88_wash_L, wash_R=mode88_wash_R,
            R_rise=mode88_R_rise, D_pull=mode88_D_pull, K_vol=mode88_K_vol,
        )
        score_fn = _score_mode88
        mode_label = "mode88"
    else:
        # mode3 / mode8 / mode9 为三套独立模型：信号上 mode8 与 mode3/mode9 不同，评分上三者均不同。见 docs/mode3_mode8_mode9_三者区别.md
        signal_fn = (
            _mode8_signals
            if use_mode8
            else (_mode9_signals if use_mode9 else _mode3_signals)
        )
        score_fn = _score_mode8 if use_mode8 else (_score_mode9 if use_mode9 else _score_mode3)
        mode_label = "mode8" if use_mode8 else ("mode9" if use_mode9 else ("mode4" if mode4_filters else "mode3"))
    end_date = cutoff_date

    for item in stock_list:
        if _is_st(item.name or ""):
            continue
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
        min_rows = max(80, mode8_n_bars) if use_mode8 else (100 if (use_mode10 or use_mode11 or use_mode12) else (260 if use_mode88 else (200 if use_mode18 else 80)))
        if not rows or len(rows) < min_rows:
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

        signals = signal_fn(rows, start_date, end_date)
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

            if (use_mode9 and score_fn is _score_mode9) or (use_mode8 and score_fn is _score_mode8) or (use_mode10 and score_fn is _score_mode10) or (use_mode11 and score_fn is _score_mode11) or (use_mode12 and score_fn is _score_mode12) or (use_mode18 and score_fn is _score_mode18) or (use_mode88 and score_fn is _score_mode88):
                score = score_fn(rows, idx, ma10, ma20, ma60, vol20, item.code, item.name)
            else:
                score = score_fn(rows, idx, ma10, ma20, ma60, vol20)
            if score < config.min_score:
                continue

            buy_point_score = _buy_point_score(rows, idx, ma10, ma20, ma60, vol20)
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
                f"启动点 {mode_label}",
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
                        "buy_point_score": int(buy_point_score),
                        "limitup_shrink_vol": int(
                            _has_limit_up_then_shrink_volume(rows, idx, item.code, item.name, lookback=6, next_vol_max_mult=1.8)
                        ),
                        "has_limit_up_6d": int(
                            _has_limit_up_6d(rows, idx, item.code, item.name, lookback=6)
                        ),
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
        buy_point_score = int(metrics.get("buy_point_score", 0))
        limitup_shrink_vol = int(metrics.get("limitup_shrink_vol", 0))
        has_limit_up_6d = int(metrics.get("has_limit_up_6d", 0))
        if prefer_upper_shadow:
            return (
                -r.score,
                -limitup_shrink_vol,
                -has_limit_up_6d,
                -buy_point_score,
                -upper_score,
                close_gap,
                -vol_ratio,
                -(ma20_gap + ma60_gap),
                ret20_val,
                r.code,
            )
        return (
            -r.score,
            -limitup_shrink_vol,
            -has_limit_up_6d,
            -buy_point_score,
            close_gap,
            -vol_ratio,
            -(ma20_gap + ma60_gap),
            ret20_val,
            r.code,
        )

    # 先按评分与买点/涨停特征排序
    results.sort(key=_mode3_sort_key)

    # 计算每个代码在本次扫描区间内的最早信号日，用于前端展示「最早出现日期」
    first_dates: Dict[str, str] = {}
    for r in results:
        metrics = r.metrics or {}
        sig = str(metrics.get("signal_date") or "").strip()
        if not sig:
            continue
        code = r.code
        if code not in first_dates or sig < first_dates[code]:
            first_dates[code] = sig
    for r in results:
        if not first_dates:
            break
        metrics = r.metrics or {}
        code = r.code
        if code in first_dates:
            metrics["first_signal_date"] = first_dates[code]
            r.metrics = metrics

    # 一段时间选股：每日取分数最高的 max_results 只（与前端「输出数量」一致）
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
            out.extend(group[: config.max_results])
        return out[: config.max_results]

    return results[: config.max_results]


def serialize_results(results: List[ScanResult]) -> List[Dict[str, object]]:
    return [
        {
            **asdict(r),
            "reasons": ", ".join(r.reasons),
            "buy_point_score": (r.metrics or {}).get("buy_point_score"),
            "first_signal_date": (r.metrics or {}).get("first_signal_date"),
            "has_limit_up_6d": (r.metrics or {}).get("has_limit_up_6d"),
        }
        for r in results
    ]
