"""mode36 一阳穿多均线：放量阳线实体同时穿越多条均线。

样本：中捷精工 301072 @ 2026-06-03（一阳穿过 MA5/10/20/30/60/120）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.scanner import KlineRow, _is_st, _limit_rate, _modepbs_big_pct_threshold, _vol_ratio_at

MODE36_ID = "mode36"
MODE36_FULL_NAME = "一阳穿多均线"
MODE36_DISPLAY_NAME = f"{MODE36_ID}（{MODE36_FULL_NAME}）"
MODE36_ONE_LINE = "放量阳线开盘在下、收盘在上，一次穿越多条均线"

DEFAULT_MA_PERIODS: tuple[int, ...] = (5, 10, 20, 30, 60, 120)
MODE36_TANGLE_SPREAD_PCT = 6.0
MODE36_PRIOR_LOOKBACK = 20


def mode36_default_kw() -> Dict[str, Any]:
    return dict(
        ma_periods=list(DEFAULT_MA_PERIODS),
        min_ma_cross=6,
        pct_min=5.0,
        pct_min_main=4.0,
        pct_max=20.0,
        vol_mult=1.5,
        vol_ma=20,
        require_prev_below=True,
        body_ratio_min=0.55,
    )


def mode36_kw_from_scan_config(cfg: Any) -> Dict[str, Any]:
    base = mode36_default_kw()
    for k in base:
        ck = f"mode36_{k}"
        if hasattr(cfg, ck):
            base[k] = getattr(cfg, ck)
    if hasattr(cfg, "mode36_min_score"):
        base["min_score"] = int(getattr(cfg, "mode36_min_score", 60))
    return base


def _ma_series(closes: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(closes), np.nan)
    if len(closes) >= n:
        out[n - 1:] = np.convolve(closes, np.ones(n) / n, mode="valid")
    return out


def _ma_at(closes: np.ndarray, idx: int, n: int) -> float:
    if idx < n - 1:
        return float("nan")
    return float(np.mean(closes[idx - n + 1 : idx + 1]))


def _ma_slope_pct(closes: np.ndarray, idx: int, n: int, look: int) -> float:
    v0 = _ma_at(closes, idx - look, n) if idx >= look + n - 1 else float("nan")
    v1 = _ma_at(closes, idx, n)
    if np.isnan(v0) or v0 <= 0 or np.isnan(v1):
        return 0.0
    return (v1 / v0 - 1.0) * 100.0


def _ma_spread_pct(closes: np.ndarray, idx: int, ma_periods: Sequence[int]) -> float:
    if closes[idx] <= 0:
        return float("nan")
    mas = [_ma_at(closes, idx, n) for n in ma_periods if idx >= n - 1]
    mas = [v for v in mas if not np.isnan(v)]
    if not mas:
        return float("nan")
    return (max(mas) - min(mas)) / closes[idx] * 100.0


def _row_volume(rows: List[KlineRow], idx: int) -> float:
    r = rows[idx]
    return float(getattr(r, "volume", 0) or getattr(r, "vol", 0) or 0)


def _mode36_prior_features(
    rows: List[KlineRow],
    idx: int,
    m: Dict[str, Any],
    ma_periods: Sequence[int],
) -> Dict[str, float]:
    """信号日前均线纠缠、量能干涸、价格蓄势等特征（5 月复盘）。"""
    closes = np.array([float(r.close) for r in rows], dtype=float)
    highs = np.array([float(r.high) for r in rows], dtype=float)
    lows = np.array([float(r.low) for r in rows], dtype=float)
    vols = np.array([_row_volume(rows, j) for j in range(len(rows))], dtype=float)
    look = MODE36_PRIOR_LOOKBACK
    tangle_thr = MODE36_TANGLE_SPREAD_PCT
    max_p = max(ma_periods) if ma_periods else 120
    c = float(m.get("close", closes[idx]))
    h, l_ = float(m.get("high", highs[idx])), float(m.get("low", lows[idx]))

    ma_spread_pct = _ma_spread_pct(closes, idx, ma_periods)
    ma_spread_20d_ago = (
        _ma_spread_pct(closes, idx - look, ma_periods) if idx >= look + max_p else float("nan")
    )
    ma_spread_chg_20d = (
        ma_spread_pct - ma_spread_20d_ago
        if not np.isnan(ma_spread_pct) and not np.isnan(ma_spread_20d_ago)
        else 0.0
    )

    spreads10: List[float] = []
    for j in range(max(max_p - 1, idx - 9), idx + 1):
        sp = _ma_spread_pct(closes, j, ma_periods)
        if not np.isnan(sp):
            spreads10.append(sp)
    ma_spread_10d_mean = float(np.mean(spreads10)) if spreads10 else 0.0

    tangle_start = max(max_p - 1, idx - look + 1)
    tangle_days_20 = float(
        sum(
            1
            for j in range(tangle_start, idx + 1)
            if _ma_spread_pct(closes, j, ma_periods) < tangle_thr
        )
    )

    below_ma20_days_20 = 0.0
    below_ma60_days_20 = 0.0
    for j in range(max(60, idx - look + 1), idx):
        if closes[j] < _ma_at(closes, j, 20):
            below_ma20_days_20 += 1.0
        if closes[j] < _ma_at(closes, j, 60):
            below_ma60_days_20 += 1.0

    ma20_slope5d_pct = _ma_slope_pct(closes, idx, 20, 5)
    ma60_slope5d_pct = _ma_slope_pct(closes, idx, 60, 5)
    ma20_slope20d_pct = _ma_slope_pct(closes, idx, 20, 20)

    vma5 = float(np.mean(vols[max(0, idx - 5):idx])) if idx >= 5 else float(vols[idx])
    vma10 = float(np.mean(vols[max(0, idx - 10):idx])) if idx >= 10 else float(vols[idx])
    vma20 = float(np.mean(vols[max(0, idx - 20):idx])) if idx >= 20 else float(vols[idx])
    vma60 = float(np.mean(vols[max(0, idx - 60):idx])) if idx >= 60 else float(vols[idx])
    vol_vma5_vs_vma20 = vma5 / vma20 if vma20 > 0 else 0.0
    vol_dry_10_60 = vma10 / vma60 if vma60 > 0 else 1.0

    w20 = slice(max(0, idx - look), idx)
    w10 = slice(max(0, idx - 10), idx)
    if w20.stop > w20.start:
        hi20, lo20 = float(np.max(highs[w20])), float(np.min(lows[w20]))
        mean20 = float(np.mean(closes[w20]))
        amp20_pre_pct = (hi20 - lo20) / mean20 * 100.0 if mean20 > 0 else 0.0
        drawdown_20d_hi_pct = (c / hi20 - 1.0) * 100.0 if hi20 > 0 else 0.0
        seg_c = closes[w20]
        if len(seg_c) >= 5:
            slope = float(np.polyfit(range(len(seg_c)), seg_c, 1)[0])
            price_drift20d_pct = slope / float(np.mean(seg_c)) * 100.0 * len(seg_c)
        else:
            price_drift20d_pct = 0.0
    else:
        amp20_pre_pct = 0.0
        drawdown_20d_hi_pct = 0.0
        price_drift20d_pct = 0.0

    if w10.stop > w10.start:
        hi10, lo10 = float(np.max(highs[w10])), float(np.min(lows[w10]))
        mean10 = float(np.mean(closes[w10]))
        amp10_pre_pct = (hi10 - lo10) / mean10 * 100.0 if mean10 > 0 else 0.0
    else:
        amp10_pre_pct = 0.0

    rng = h - l_
    upper_shadow_ratio = (h - c) / rng if rng > 0 else 0.0

    w60 = slice(max(0, idx - 59), idx + 1)
    hi60, lo60 = float(np.max(highs[w60])), float(np.min(lows[w60]))
    pos60_pct = (c - lo60) / (hi60 - lo60) * 100.0 if hi60 > lo60 else 50.0

    spreads5: List[float] = []
    for j in range(max(max_p - 1, idx - 4), idx + 1):
        sp = _ma_spread_pct(closes, j, ma_periods)
        if not np.isnan(sp):
            spreads5.append(sp)
    spread_narrow_5d = bool(len(spreads5) >= 2 and spreads5[-1] < spreads5[0])

    return {
        "ma_spread_pct": float(ma_spread_pct) if not np.isnan(ma_spread_pct) else 0.0,
        "ma_spread_20d_ago_pct": float(ma_spread_20d_ago) if not np.isnan(ma_spread_20d_ago) else 0.0,
        "ma_spread_chg_20d_pct": float(ma_spread_chg_20d),
        "ma_spread_10d_mean_pct": float(ma_spread_10d_mean),
        "tangle_days_20": float(tangle_days_20),
        "below_ma20_days_20": float(below_ma20_days_20),
        "below_ma60_days_20": float(below_ma60_days_20),
        "ma20_slope5d_pct": float(ma20_slope5d_pct),
        "ma60_slope5d_pct": float(ma60_slope5d_pct),
        "ma20_slope20d_pct": float(ma20_slope20d_pct),
        "vol_vma5_vs_vma20": float(vol_vma5_vs_vma20),
        "vol_dry_10_60": float(vol_dry_10_60),
        "amp10_pre_pct": float(amp10_pre_pct),
        "amp20_pre_pct": float(amp20_pre_pct),
        "price_drift20d_pct": float(price_drift20d_pct),
        "drawdown_20d_hi_pct": float(drawdown_20d_hi_pct),
        "upper_shadow_ratio": float(upper_shadow_ratio),
        "pos60_pct": float(pos60_pct),
        "spread_narrow_5d": 1.0 if spread_narrow_5d else 0.0,
    }


def _day_pct(rows: List[KlineRow], i: int) -> float:
    raw = getattr(rows[i], "pct_chg", None)
    if raw is not None and float(raw) != 0:
        return float(raw)
    if i < 1:
        return 0.0
    prev = float(rows[i - 1].close)
    if prev <= 0:
        return 0.0
    return (float(rows[i].close) - prev) / prev * 100.0


def _pct_min(code: str, name: str, kw: Dict[str, Any]) -> float:
    if _limit_rate(code, name) < 0.15 and not _is_st(name or ""):
        return float(kw.get("pct_min_main", 4.0))
    return _modepbs_big_pct_threshold(
        code,
        name,
        big_pct_min=float(kw["pct_min"]),
        big_pct_min_main=float(kw.get("pct_min_main", 4.0)),
    )


def count_ma_crossed(
    rows: List[KlineRow],
    idx: int,
    ma_periods: Sequence[int],
) -> Dict[str, Any]:
    """统计信号日一阳穿越的均线（开盘在下、收盘在上，且均线落在 K 线高低区间内）。"""
    closes = np.array([float(r.close) for r in rows], dtype=float)
    r = rows[idx]
    o, c, h, l = float(r.open), float(r.close), float(r.high), float(r.low)
    prev_c = float(rows[idx - 1].close) if idx >= 1 else c

    crossed: List[int] = []
    in_range: List[int] = []
    prev_below: List[int] = []
    ma_vals: Dict[str, float] = {}

    for n in ma_periods:
        if idx < n:
            continue
        ma = _ma_series(closes, n)
        v = ma[idx]
        if np.isnan(v):
            continue
        ma_vals[f"ma{n}"] = float(v)
        if l <= v <= h:
            in_range.append(n)
        if o < v < c:
            crossed.append(n)
        if prev_c < v:
            prev_below.append(n)

    return {
        "crossed_ma_periods": crossed,
        "crossed_ma_count": float(len(crossed)),
        "in_range_ma_count": float(len(in_range)),
        "prev_below_ma_count": float(len(prev_below)),
        **ma_vals,
    }


def match_mode36_yang_cross_ma(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    kw = {**mode36_default_kw(), **kwargs}
    ma_periods = [int(x) for x in kw.get("ma_periods", DEFAULT_MA_PERIODS)]
    max_period = max(ma_periods) if ma_periods else 120
    if idx < max_period + 1 or idx >= len(rows):
        return None

    r = rows[idx]
    o, c, h, l = float(r.open), float(r.close), float(r.high), float(r.low)
    if c <= o:
        return None

    rng = h - l
    if rng <= 0:
        return None
    body_ratio = (c - o) / rng
    if body_ratio < float(kw.get("body_ratio_min", 0.55)):
        return None

    pct = _day_pct(rows, idx)
    pct_lim = _pct_min(code, name, kw)
    if pct < pct_lim or pct > float(kw.get("pct_max", 20.0)):
        return None

    vol_ratio = _vol_ratio_at(rows, idx, int(kw.get("vol_ma", 20)))
    if vol_ratio < float(kw.get("vol_mult", 1.5)):
        return None

    stat = count_ma_crossed(rows, idx, ma_periods)
    min_cross = int(kw.get("min_ma_cross", 6))
    if stat["crossed_ma_count"] < min_cross:
        return None

    if bool(kw.get("require_prev_below", True)):
        if stat["prev_below_ma_count"] < min_cross:
            return None

    return {
        "signal_date": str(rows[idx].date)[:10],
        "event_type": "一阳穿线",
        "close": c,
        "pct_chg": pct,
        "vol_ratio": vol_ratio,
        "body_ratio": body_ratio,
        "open": o,
        "high": h,
        "low": l,
        **stat,
    }


def score_mode36_yang_cross_ma(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> int:
    kw = {**mode36_default_kw(), **kwargs}
    m = match_mode36_yang_cross_ma(rows, idx, code, name, **kwargs)
    if not m:
        return 0

    ma_periods = [int(x) for x in kw.get("ma_periods", DEFAULT_MA_PERIODS)]
    q = _mode36_prior_features(rows, idx, m, ma_periods)
    n_cross = int(m["crossed_ma_count"])
    full_cross_setup = (
        n_cross >= int(kw.get("min_ma_cross", 6))
        and float(m.get("prev_below_ma_count", 0)) >= n_cross
    )

    # 当日穿线形态（满分约 72，为前期均线/量能质量留出评分空间）
    score = 36.0
    score += n_cross * 3.0
    score += min(8.0, float(m["vol_ratio"]) * 2.0)
    score += min(8.0, float(m["pct_chg"]) * 0.45)
    score += float(m["body_ratio"]) * 5.0
    if n_cross >= 7:
        score += 4.0
    if float(m.get("prev_below_ma_count", 0)) >= n_cross:
        score += 4.0

    # 信号日 K 线质量
    if q["upper_shadow_ratio"] <= 0.15:
        score += 8.0
    elif q["upper_shadow_ratio"] > 0.35:
        score -= 14.0
    elif q["upper_shadow_ratio"] > 0.25:
        score -= 10.0
    elif q["upper_shadow_ratio"] > 0.15:
        pen = 3.0 if full_cross_setup and float(m["body_ratio"]) >= 0.60 else 6.0
        score -= pen

    extended_chase = q["drawdown_20d_hi_pct"] > 3.0 and q["pos60_pct"] > 75.0

    # 前期均线纠缠与收窄（高位追涨时不给蓄势加分）
    if not extended_chase:
        if q["tangle_days_20"] >= 5.0:
            score += 10.0
        if q["tangle_days_20"] >= 10.0:
            score += 5.0
        if q["ma_spread_pct"] < 5.0:
            score += 8.0
        if q["ma_spread_chg_20d_pct"] < -2.0:
            score += 5.0
        if q["spread_narrow_5d"] >= 1.0:
            score += 4.0
        if q["ma20_slope5d_pct"] > 0.0:
            score += 8.0
        if q["ma20_slope20d_pct"] > 0.0:
            score += 5.0
        if q["vol_dry_10_60"] < 0.85:
            score += 10.0
        if q["vol_dry_10_60"] < 0.75:
            score += 5.0
        if q["amp10_pre_pct"] < 12.0:
            score += 8.0
        if -5.0 <= q["drawdown_20d_hi_pct"] <= 2.0:
            score += 8.0
        if q["below_ma60_days_20"] >= 12.0:
            score += 5.0

    if q["tangle_days_20"] < 3.0 and q["spread_narrow_5d"] < 1.0:
        score -= 16.0
    if q["ma_spread_pct"] > 6.0:
        score -= 8.0
    if q["ma20_slope5d_pct"] < -1.5:
        score -= 14.0
    elif q["ma20_slope5d_pct"] < -1.0:
        score -= 6.0
    if q["ma20_slope20d_pct"] < -2.0:
        score -= 10.0
    if q["vol_dry_10_60"] > 1.0:
        score -= 8.0 if full_cross_setup else 14.0
    elif q["vol_dry_10_60"] > 0.95:
        score -= 4.0 if full_cross_setup else 6.0
    if q["amp10_pre_pct"] > 18.0:
        score -= 10.0
    elif q["amp10_pre_pct"] > 14.0:
        score -= 4.0 if full_cross_setup else 10.0
    if q["drawdown_20d_hi_pct"] > 3.0:
        score -= 14.0
    elif q["drawdown_20d_hi_pct"] < -10.0:
        score -= 10.0
    if q["pos60_pct"] > 78.0:
        score -= 10.0
    if extended_chase:
        score -= 12.0
    if full_cross_setup:
        score += 10.0

    # 多指标共振劣化（5 月差样本：纠缠缺失 + 量未缩 + 上影/偏高）
    red_flags = 0
    if q["tangle_days_20"] < 3.0 and q["spread_narrow_5d"] < 1.0:
        red_flags += 1
    if q["ma_spread_pct"] > 6.0:
        red_flags += 1
    if q["ma20_slope5d_pct"] < -1.5:
        red_flags += 1
    if q["ma20_slope20d_pct"] < -2.0:
        red_flags += 1
    if q["vol_dry_10_60"] > 1.0:
        red_flags += 1
    if q["amp10_pre_pct"] > 14.0:
        red_flags += 1
    if q["drawdown_20d_hi_pct"] > 3.0 or q["drawdown_20d_hi_pct"] < -10.0:
        red_flags += 1
    if q["upper_shadow_ratio"] > 0.20:
        red_flags += 1
    if red_flags >= 2:
        penalty = min(20.0, (red_flags - 1) * 5.0)
        if full_cross_setup:
            penalty = min(penalty, 10.0)
        score -= penalty

    return int(min(99, max(0, round(score))))


def mode36_signal_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    kw = {**mode36_default_kw(), **kwargs}
    m = match_mode36_yang_cross_ma(rows, idx, code, name, **kwargs)
    if not m:
        return {}
    ma_periods = [int(x) for x in kw.get("ma_periods", DEFAULT_MA_PERIODS)]
    q = _mode36_prior_features(rows, idx, m, ma_periods)
    score = score_mode36_yang_cross_ma(rows, idx, code, name, **kwargs)
    crossed = m.get("crossed_ma_periods") or []
    return {
        **m,
        **q,
        "mode36_score": score,
        "crossed_ma_list": ",".join(str(x) for x in crossed),
    }
