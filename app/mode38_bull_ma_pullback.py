"""mode38 大牛股关键位回踩：前期大涨后回调，低点踩 MA10/20/30/60/120 关键均线。

样本：亚翔集成 603929（仅两买点）
  - 2026-03-12 回踩 MA60（二调低点）
  - 2026-06-02 回踩 MA120（三调低点）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.scanner import KlineRow, _is_st, _vol_ratio_at

MODE38_ID = "mode38"
MODE38_FULL_NAME = "大牛股关键位回踩"
MODE38_DISPLAY_NAME = f"{MODE38_ID}（{MODE38_FULL_NAME}）"
MODE38_ONE_LINE = "前期大涨后回调，低点踩 MA10/20/30/60/120 附近（±容差），不要求按均线档位最低回撤"

SUPPORT_MAS: tuple[int, ...] = (10, 20, 30, 60, 120)
MIN_PULLBACK_BY_MA: Dict[int, float] = {10: 10.0, 20: 12.0, 30: 16.0, 60: 22.0, 120: 28.0}


def mode38_default_kw() -> Dict[str, Any]:
    return dict(
        phase_lookback=120,
        peak_lookback=45,
        peak_end_offset=1,
        min_rally_pct=80.0,
        pullback_min_pct=3.0,
        pullback_max_pct=60.0,
        require_min_pullback_by_ma=False,
        min_pullback_ma10_pct=10.0,
        min_pullback_ma20_pct=12.0,
        min_pullback_ma30_pct=16.0,
        min_pullback_ma60_pct=22.0,
        min_pullback_ma120_pct=28.0,
        max_ma_dist_pct=5.0,
        max_close_above_ma_pct=8.0,
        ma_slope_days=10,
        min_ma120_slope_pct=0.0,
        max_break_ma120_pct=5.0,
        vol_ma=20,
        shrink_vol_max=1.0,
        bounce_pct_bonus=3.0,
        recent_spike_days=3,
        max_recent_day_pct=8.0,
        require_pullback_trough=True,
        require_trough_confirm=True,
        require_above_ma120_250=True,
        min_score=60,
        support_ma_only=0,
    )


def mode38_kw_from_scan_config(cfg: Any) -> Dict[str, Any]:
    base = mode38_default_kw()
    for k in base:
        ck = f"mode38_{k}"
        if hasattr(cfg, ck):
            base[k] = getattr(cfg, ck)
    if hasattr(cfg, "mode38_min_score"):
        base["min_score"] = int(getattr(cfg, "mode38_min_score", 60))
    return base


def _row_volume(rows: List[KlineRow], idx: int) -> float:
    r = rows[idx]
    return float(getattr(r, "volume", 0) or getattr(r, "vol", 0) or 0)


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


def _min_pullback_for_ma(sup_period: int, kw: Dict[str, Any]) -> float:
    key = f"min_pullback_ma{sup_period}_pct"
    if key in kw:
        return float(kw[key])
    return MIN_PULLBACK_BY_MA.get(sup_period, float(kw["pullback_min_pct"]))


def _support_at_period(
    closes: np.ndarray,
    idx: int,
    low: float,
    period: int,
    *,
    ma_slope_days: int,
    max_ma_dist_pct: float,
) -> Optional[Dict[str, float]]:
    if idx < period - 1:
        return None
    ma = _ma_at(closes, idx, period)
    if np.isnan(ma) or ma <= 0:
        return None
    slope = _ma_slope_pct(closes, idx, period, ma_slope_days)
    if slope <= 0:
        return None
    dist = (low / ma - 1.0) * 100.0
    if abs(dist) > max_ma_dist_pct:
        return None
    return {
        "support_ma": float(period),
        "support_ma_val": ma,
        "low_dist_ma_pct": dist,
        "support_ma_slope_pct": _ma_slope_pct(closes, idx, period, ma_slope_days),
    }


def _pick_support_ma(
    closes: np.ndarray,
    idx: int,
    low: float,
    *,
    ma_slope_days: int,
    max_ma_dist_pct: float,
) -> Optional[Dict[str, float]]:
    """在上升均线中，取离低点最近且周期最深的关键位。"""
    candidates: List[tuple[float, int, float, float]] = []
    for period in SUPPORT_MAS:
        if idx < period - 1:
            continue
        ma = _ma_at(closes, idx, period)
        if np.isnan(ma) or ma <= 0:
            continue
        slope = _ma_slope_pct(closes, idx, period, ma_slope_days)
        if slope <= 0:
            continue
        dist = (low / ma - 1.0) * 100.0
        if abs(dist) <= max_ma_dist_pct:
            candidates.append((abs(dist), period, ma, dist))
    if not candidates:
        return None
    # 同等距离优先更深均线（120>60>30>20>10）
    candidates.sort(key=lambda x: (x[0], -x[1]))
    _, period, ma, dist = candidates[0]
    return {
        "support_ma": float(period),
        "support_ma_val": ma,
        "low_dist_ma_pct": dist,
        "support_ma_slope_pct": _ma_slope_pct(closes, idx, period, ma_slope_days),
    }


def match_mode38_bull_ma_pullback(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    if _is_st(name or ""):
        return None
    kw = mode38_default_kw()
    kw.update({k: v for k, v in kwargs.items() if k in kw or k.startswith("mode38")})

    phase_lb = int(kw["phase_lookback"])
    peak_lb = int(kw["peak_lookback"])
    peak_end = int(kw["peak_end_offset"])
    min_len = max(phase_lb, 250) + 10
    if idx < min_len or idx >= len(rows):
        return None

    closes = np.array([float(r.close) for r in rows], dtype=float)
    highs = np.array([float(r.high) for r in rows], dtype=float)
    lows = np.array([float(r.low) for r in rows], dtype=float)
    vols = np.array([_row_volume(rows, j) for j in range(len(rows))], dtype=float)

    peak_start = max(0, idx - peak_lb)
    peak_end_i = max(peak_start, idx - peak_end)
    if peak_end_i <= peak_start:
        return None
    peak_slice = slice(peak_start, peak_end_i + 1)
    peak_i = peak_start + int(np.argmax(highs[peak_slice]))
    peak_high = float(highs[peak_i])
    if peak_high <= 0:
        return None

    phase_start = max(0, peak_i - phase_lb)
    phase_end = max(phase_start, peak_i - 8)
    if phase_end <= phase_start:
        return None
    phase_low = float(np.min(lows[phase_start : phase_end + 1]))
    if phase_low <= 0:
        return None

    rally_pct = (peak_high - phase_low) / phase_low * 100.0
    if rally_pct < float(kw["min_rally_pct"]):
        return None

    cur = rows[idx]
    cur_low = float(cur.low)
    cur_close = float(cur.close)
    cur_high = float(cur.high)
    cur_open = float(cur.open)
    pct = float(getattr(cur, "pct_chg", 0) or 0)

    spike_days = int(kw["recent_spike_days"])
    max_spike = float(kw["max_recent_day_pct"])
    if spike_days > 0:
        spike_start = max(0, idx - spike_days + 1)
        for j in range(spike_start, idx + 1):
            day_pct = float(getattr(rows[j], "pct_chg", 0) or 0)
            if day_pct > max_spike:
                return None

    pullback_pct = (1.0 - cur_low / peak_high) * 100.0
    if pullback_pct < float(kw["pullback_min_pct"]) or pullback_pct > float(kw["pullback_max_pct"]):
        return None

    ma120 = _ma_at(closes, idx, 120)
    ma250 = _ma_at(closes, idx, 250)
    if np.isnan(ma120) or ma120 <= 0 or np.isnan(ma250) or ma250 <= 0:
        return None
    if kw.get("require_above_ma120_250", True):
        if cur_close <= ma120 or cur_close <= ma250:
            return None
    ma120_slope = _ma_slope_pct(closes, idx, 120, int(kw["ma_slope_days"]))
    if ma120_slope < float(kw["min_ma120_slope_pct"]):
        return None
    if cur_low < ma120 * (1.0 - float(kw["max_break_ma120_pct"]) / 100.0):
        return None

    ma_slope_days = int(kw["ma_slope_days"])
    max_ma_dist = float(kw["max_ma_dist_pct"])
    support_ma_only = int(kw.get("support_ma_only", 0) or 0)
    if support_ma_only > 0:
        support = _support_at_period(
            closes,
            idx,
            cur_low,
            support_ma_only,
            ma_slope_days=ma_slope_days,
            max_ma_dist_pct=max_ma_dist,
        )
    else:
        support = _pick_support_ma(
            closes,
            idx,
            cur_low,
            ma_slope_days=ma_slope_days,
            max_ma_dist_pct=max_ma_dist,
        )
    if not support:
        return None

    sup_period = int(support["support_ma"])
    if kw.get("require_min_pullback_by_ma", False):
        if pullback_pct < _min_pullback_for_ma(sup_period, kw):
            return None

    if kw.get("require_pullback_trough", True):
        seg_low = float(np.min(lows[peak_i : idx + 1]))
        if cur_low > seg_low * 1.001:
            return None
        if kw.get("require_trough_confirm", True) and idx + 1 < len(rows):
            if float(lows[idx + 1]) <= cur_low:
                return None

    sup_ma = float(support["support_ma_val"])
    close_dist = (cur_close / sup_ma - 1.0) * 100.0
    if close_dist > float(kw["max_close_above_ma_pct"]):
        return None
    if cur_close < sup_ma * (1.0 - float(kw["max_break_ma120_pct"]) / 100.0):
        return None

    vol_ma = int(kw["vol_ma"])
    vol_ratio = _vol_ratio_at(rows, idx, vol_ma)
    rally_vol = float(np.mean(vols[max(phase_start, peak_i - 20) : peak_i + 1]))
    pull_vol = float(np.mean(vols[peak_i : idx + 1]))
    vol_shrink = pull_vol / rally_vol if rally_vol > 0 else 1.0

    rng = cur_high - cur_low
    lower_shadow = (min(cur_open, cur_close) - cur_low) / rng if rng > 0 else 0.0
    body_ratio = (cur_close - cur_open) / rng if rng > 0 else 0.0

    return {
        "phase_low": phase_low,
        "peak_high": peak_high,
        "peak_date": str(rows[peak_i].date)[:10],
        "rally_pct": rally_pct,
        "pullback_pct": pullback_pct,
        "support_ma": support["support_ma"],
        "support_ma_val": sup_ma,
        "low_dist_ma_pct": support["low_dist_ma_pct"],
        "close_dist_ma_pct": close_dist,
        "support_ma_slope_pct": support["support_ma_slope_pct"],
        "ma120_slope_pct": ma120_slope,
        "ma120": ma120,
        "ma250": ma250,
        "vol_ratio": vol_ratio,
        "vol_shrink_ratio": vol_shrink,
        "pct_chg": pct,
        "lower_shadow_ratio": lower_shadow,
        "body_ratio": body_ratio,
        "close": cur_close,
        "low": cur_low,
    }


def score_mode38_bull_ma_pullback(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> int:
    m = match_mode38_bull_ma_pullback(rows, idx, code, name, **kwargs)
    if not m:
        return 0
    score = 42.0
    score += min(18.0, float(m["rally_pct"]) * 0.08)
    pb = float(m["pullback_pct"])
    if 18.0 <= pb <= 28.0:
        score += 14.0
    elif 12.0 <= pb <= 35.0:
        score += 8.0
    dist = abs(float(m["low_dist_ma_pct"]))
    if dist <= 2.0:
        score += 16.0
    elif dist <= 4.0:
        score += 10.0
    else:
        score += 5.0
    sup = int(m["support_ma"])
    if sup >= 120:
        score += 12.0
    elif sup >= 60:
        score += 8.0
    elif sup >= 30:
        score += 6.0
    elif sup >= 20:
        score += 4.0
    else:
        score += 3.0
    if float(m["support_ma_slope_pct"]) > 3.0:
        score += 6.0
    if float(m["ma120_slope_pct"]) > 3.0:
        score += 6.0
    if float(m["vol_shrink_ratio"]) <= float(kwargs.get("shrink_vol_max", 1.0) or 1.0):
        score += 8.0
    if float(m["pct_chg"]) >= float(kwargs.get("bounce_pct_bonus", 3.0) or 3.0):
        score += 8.0
    elif float(m["pct_chg"]) > 0:
        score += 4.0
    if float(m["lower_shadow_ratio"]) >= 0.35:
        score += 6.0
    return int(min(100, max(0, round(score))))


def mode38_signal_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    m = match_mode38_bull_ma_pullback(rows, idx, code, name, **kwargs)
    if not m:
        return {}
    return {
        "peak_date": m["peak_date"],
        "phase_low": round(float(m["phase_low"]), 4),
        "peak_high": round(float(m["peak_high"]), 4),
        "rally_pct": round(float(m["rally_pct"]), 2),
        "pullback_pct": round(float(m["pullback_pct"]), 2),
        "support_ma": int(m["support_ma"]),
        "support_ma_val": round(float(m["support_ma_val"]), 4),
        "low_dist_ma_pct": round(float(m["low_dist_ma_pct"]), 2),
        "close_dist_ma_pct": round(float(m["close_dist_ma_pct"]), 2),
        "support_ma_slope_pct": round(float(m["support_ma_slope_pct"]), 2),
        "ma120_slope_pct": round(float(m["ma120_slope_pct"]), 2),
        "ma120": round(float(m["ma120"]), 4),
        "ma250": round(float(m["ma250"]), 4),
        "vol_ratio": round(float(m["vol_ratio"]), 2),
        "vol_shrink_ratio": round(float(m["vol_shrink_ratio"]), 2),
        "pct_chg": round(float(m["pct_chg"]), 2),
        "lower_shadow_ratio": round(float(m["lower_shadow_ratio"]), 2),
        "low": round(float(m["low"]), 4),
    }


def dedupe_mode38_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """同一股票、同一前高、同一支撑均线，只保留低点最低的一条（一波一个买点）。"""
    best: Dict[tuple, Dict[str, Any]] = {}
    for h in hits:
        key = (h.get("code", ""), h.get("peak_date", ""), int(h.get("support_ma", 0) or 0))
        low = float(h.get("low", h.get("support_ma_val", 0)) or 0)
        prev = best.get(key)
        if prev is None or low < float(prev.get("low", prev.get("support_ma_val", 0)) or 0):
            best[key] = h
    return sorted(best.values(), key=lambda x: (x.get("date", ""), x.get("code", "")))
