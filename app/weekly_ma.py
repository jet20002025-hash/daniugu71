"""周线均线粘合度：由日线聚合为周线，计算 5/10/20/30 周均线粘合度。"""
from collections import defaultdict
from datetime import datetime
from typing import List, Optional

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
    计算 5/10/20/30 周线粘合度（百分比）。
    粘合度 = (max(MA5,MA10,MA20,MA30) - min(...)) / 当周收盘价 * 100。
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
