"""周线均线粘合度：由日线聚合为周线，计算 5/10/20/30 周均线粘合度。

周线拟合值定义（供 mode10 等使用）：
  当周拟合值 = (max(MA5, MA10, MA20, MA30) - min(MA5, MA10, MA20, MA30)) / 当周收盘价 × 100
  - 单位：百分比（%）
  - 含义：四条周均线在该周的离散程度；值越小表示均线越粘合（越「拟合」）
  - 计算范围：需至少 30 周收盘价，前 29 周无 MA30 故无拟合值
"""
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np


def _week_key(date_str: str) -> tuple:
    """返回 (year, week) 用于按周分组。"""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (d.isocalendar().year, d.isocalendar().week)
    except Exception:
        return (0, 0)


def daily_to_weekly_closes(
    rows: List,
    get_date=lambda r: r.date,
    get_open=lambda r: r.open,
    get_high=lambda r: r.high,
    get_low=lambda r: r.low,
    get_close=lambda r: r.close,
) -> List[tuple]:
    """
    将日线按周聚合，返回 [(week_key, open, high, low, close), ...] 按周序排列。
    """
    by_week: dict = defaultdict(list)
    for r in rows:
        wk = _week_key(get_date(r))
        by_week[wk].append(
            (get_open(r), get_high(r), get_low(r), get_close(r))
        )
    result = []
    for wk in sorted(by_week.keys()):
        bars = by_week[wk]
        if not bars:
            continue
        opens, highs, lows, closes = zip(*bars)
        result.append((wk, opens[0], max(highs), min(lows), closes[-1]))
    return result


def daily_to_weekly_with_last_index(
    rows: List,
    get_date=lambda r: r.date,
    get_open=lambda r: r.open,
    get_high=lambda r: r.high,
    get_low=lambda r: r.low,
    get_close=lambda r: r.close,
) -> Tuple[List[tuple], List[int]]:
    """
    将日线按周聚合，返回 (周线列表, 每周最后一行在 rows 中的下标)。
    周线列表格式同 daily_to_weekly_closes：[(week_key, open, high, low, close), ...]。
    """
    by_week: dict = defaultdict(list)
    for i, r in enumerate(rows):
        wk = _week_key(get_date(r))
        by_week[wk].append(
            (i, get_open(r), get_high(r), get_low(r), get_close(r))
        )
    result = []
    last_indices = []
    for wk in sorted(by_week.keys()):
        bars = by_week[wk]
        indices = [b[0] for b in bars]
        opens = [b[1] for b in bars]
        highs = [b[2] for b in bars]
        lows = [b[3] for b in bars]
        closes = [b[4] for b in bars]
        result.append((wk, opens[0], max(highs), min(lows), closes[-1]))
        last_indices.append(max(indices))
    return result, last_indices


def daily_to_weekly_with_volume_and_last_index(
    rows: List,
    get_date=lambda r: r.date,
    get_open=lambda r: r.open,
    get_high=lambda r: r.high,
    get_low=lambda r: r.low,
    get_close=lambda r: r.close,
    get_volume=lambda r: getattr(r, "volume", 0),
) -> Tuple[List[tuple], List[int]]:
    """
    将日线按周聚合，返回 (周线列表, 每周最后一行在 rows 中的下标)。
    周线列表每项为 (week_key, open, high, low, close, volume_sum)。
    """
    by_week: dict = defaultdict(list)
    for i, r in enumerate(rows):
        wk = _week_key(get_date(r))
        by_week[wk].append(
            (i, get_open(r), get_high(r), get_low(r), get_close(r), get_volume(r))
        )
    result = []
    last_indices = []
    for wk in sorted(by_week.keys()):
        bars = by_week[wk]
        indices = [b[0] for b in bars]
        opens = [b[1] for b in bars]
        highs = [b[2] for b in bars]
        lows = [b[3] for b in bars]
        closes = [b[4] for b in bars]
        vol_sum = sum(b[5] for b in bars)
        result.append((wk, opens[0], max(highs), min(lows), closes[-1], vol_sum))
        last_indices.append(max(indices))
    return result, last_indices


def weekly_kdj(
    weekly_bars: List[tuple],
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    周线 KDJ（9,3,3）。
    weekly_bars 每项为 (week_key, open, high, low, close) 或至少 (..., high, low, close) 且 high=index2, low=index3, close=index4。
    返回 (K, D, J) 三个与周线等长的数组，前 n-1 根为 np.nan。
    公式：RSV = (C - Ln)/(Hn - Ln)*100，K = (1/m1)*RSV + ((m1-1)/m1)*K_prev，D = (1/m2)*K + ((m2-1)/m2)*D_prev，J = 3*K - 2*D；首值 K=D=50。
    """
    if not weekly_bars or len(weekly_bars) < n:
        return np.array([]), np.array([]), np.array([])
    highs = np.array([float(w[2]) for w in weekly_bars], dtype=float)
    lows = np.array([float(w[3]) for w in weekly_bars], dtype=float)
    closes = np.array([float(w[4]) for w in weekly_bars], dtype=float)
    length = len(closes)
    rsv = np.full(length, np.nan, dtype=float)
    for i in range(n - 1, length):
        h_n = np.max(highs[i - n + 1 : i + 1])
        l_n = np.min(lows[i - n + 1 : i + 1])
        if h_n > l_n:
            rsv[i] = (closes[i] - l_n) / (h_n - l_n) * 100.0
        else:
            rsv[i] = 50.0
    k = np.full(length, np.nan, dtype=float)
    d = np.full(length, np.nan, dtype=float)
    j = np.full(length, np.nan, dtype=float)
    w1 = 1.0 / max(1, m1)
    w1_prev = (max(1, m1) - 1.0) / max(1, m1)
    w2 = 1.0 / max(1, m2)
    w2_prev = (max(1, m2) - 1.0) / max(1, m2)
    k[n - 1] = w1_prev * 50.0 + w1 * rsv[n - 1]
    d[n - 1] = w2_prev * 50.0 + w2 * k[n - 1]
    j[n - 1] = 3.0 * k[n - 1] - 2.0 * d[n - 1]
    for i in range(n, length):
        if not np.isnan(rsv[i]):
            k[i] = w1_prev * k[i - 1] + w1 * rsv[i]
            d[i] = w2_prev * d[i - 1] + w2 * k[i]
            j[i] = 3.0 * k[i] - 2.0 * d[i]
    return k, d, j


def weekly_convergence_value_series(weekly_bars: List[tuple]) -> np.ndarray:
    """
    逐周计算「周线拟合值」（与 weekly_convergence 同一定义）。
    weekly_bars 每项为 (week_key, open, high, low, close)。
    返回与周线等长的数组，前 29 周为 np.nan（缺 MA30），之后为拟合值（%）。
    拟合值 = (max(MA5,MA10,MA20,MA30) - min(...)) / 当周收盘价 * 100。
    """
    if not weekly_bars or len(weekly_bars) < 30:
        return np.array([])
    closes = np.array([float(w[4]) for w in weekly_bars], dtype=float)
    ma5 = _rolling_mean(closes, 5)
    ma10 = _rolling_mean(closes, 10)
    ma20 = _rolling_mean(closes, 20)
    ma30 = _rolling_mean(closes, 30)
    n = len(closes)
    out = np.full(n, np.nan, dtype=float)
    for i in range(29, n):
        if closes[i] <= 0:
            continue
        v = [ma5[i], ma10[i], ma20[i], ma30[i]]
        if any(np.isnan(x) for x in v):
            continue
        rng = max(v) - min(v)
        out[i] = rng / closes[i] * 100.0
    return out


def weekly_convergence_value_series_ma20(weekly_bars: List[tuple]) -> np.ndarray:
    """
    逐周计算「周线拟合值」仅用 MA5、MA10、MA20（去掉 MA30）。
    拟合值 = (max(MA5,MA10,MA20) - min(MA5,MA10,MA20)) / 当周收盘价 * 100。
    返回与周线等长的数组，前 19 周为 np.nan（缺 MA20），之后为拟合值（%）。
    """
    if not weekly_bars or len(weekly_bars) < 20:
        return np.array([])
    closes = np.array([float(w[4]) for w in weekly_bars], dtype=float)
    ma5 = _rolling_mean(closes, 5)
    ma10 = _rolling_mean(closes, 10)
    ma20 = _rolling_mean(closes, 20)
    n = len(closes)
    out = np.full(n, np.nan, dtype=float)
    for i in range(19, n):
        if closes[i] <= 0:
            continue
        v = [ma5[i], ma10[i], ma20[i]]
        if any(np.isnan(x) for x in v):
            continue
        rng = max(v) - min(v)
        out[i] = rng / closes[i] * 100.0
    return out


def weekly_convergence(
    rows: List,
    get_date=lambda r: r.date,
    get_close=lambda r: r.close,
    get_high=lambda r: r.high,
    get_low=lambda r: r.low,
    get_open=lambda r: r.open,
    min_weeks: int = 30,
) -> Optional[float]:
    """
    计算 5/10/20/30 周线粘合度（即「周线拟合值」）的当前值（最后一周）。
    拟合值 = (max(MA5,MA10,MA20,MA30) - min(...)) / 当周收盘价 * 100。
    越小表示均线越粘合。
    若周线不足 min_weeks 周则返回 None（排序时视为无穷大）。
    """
    if not rows or len(rows) < 20:
        return None
    weekly = daily_to_weekly_closes(rows, get_date, get_open, get_high, get_low, get_close)
    if len(weekly) < min_weeks:
        return None
    closes = np.array([w[4] for w in weekly], dtype=float)
    ma5 = _rolling_mean(closes, 5)
    ma10 = _rolling_mean(closes, 10)
    ma20 = _rolling_mean(closes, 20)
    ma30 = _rolling_mean(closes, 30)
    n = len(closes)
    if np.isnan(ma30[n - 1]):
        return None
    last_close = closes[-1]
    if last_close <= 0:
        return None
    vals = [ma5[n - 1], ma10[n - 1], ma20[n - 1], ma30[n - 1]]
    if any(np.isnan(v) for v in vals):
        return None
    rng = max(vals) - min(vals)
    return float(rng / last_close * 100.0)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < window:
        return out
    for i in range(window - 1, len(arr)):
        out[i] = np.mean(arr[i - window + 1 : i + 1])
    return out
