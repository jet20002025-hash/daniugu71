"""mode41 周线关键位回踩缩量：大牛股周线大涨后回调，周低点踩周均线 + 量能近5周最低附近。

基于 mode38 日线逻辑，改为周线口径：
  - 关键位：周 MA5/10/20/30/60
  - 量能：当周成交量在近 5 周最小值附近（默认 ≤ 最小值 × 1.15）
  - 买点：信号日次日开盘
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.scanner import KlineRow, _is_st
from app.weekly_ma import _week_key, daily_to_weekly_with_volume_and_last_index

MODE41_ID = "mode41"
MODE41_FULL_NAME = "周线关键位回踩缩量"
MODE41_DISPLAY_NAME = f"{MODE41_ID}（{MODE41_FULL_NAME}）"
MODE41_ONE_LINE = (
    "周线前期大涨后回调，周低点踩周 MA5/10/20/30/60，"
    "当周量能近5周最低附近；信号日次日开盘买"
)

SUPPORT_MAS: tuple[int, ...] = (5, 10, 20, 30, 60)
MIN_PULLBACK_BY_MA: Dict[int, float] = {
    5: 6.0,
    10: 8.0,
    20: 12.0,
    30: 16.0,
    60: 22.0,
}


def mode41_default_kw() -> Dict[str, Any]:
    return dict(
        phase_lookback_weeks=48,
        peak_lookback_weeks=20,
        peak_end_offset_weeks=1,
        min_rally_pct=80.0,
        pullback_min_pct=10.0,
        pullback_max_pct=40.0,
        min_pullback_ma5_pct=6.0,
        min_pullback_ma10_pct=8.0,
        min_pullback_ma20_pct=12.0,
        min_pullback_ma30_pct=16.0,
        min_pullback_ma60_pct=22.0,
        max_ma_dist_pct=8.0,
        max_close_above_ma_pct=10.0,
        ma_slope_weeks=4,
        min_ma30_slope_pct=0.0,
        max_break_ma30_pct=8.0,
        vol_lookback_weeks=5,
        max_vol_above_min_pct=15.0,
        require_vol_near_min=True,
        recent_spike_weeks=2,
        max_recent_week_pct=12.0,
        require_pullback_trough=True,
        require_trough_confirm=True,
        require_above_ma30_60=True,
        min_score=60,
        support_ma_only=0,
    )


def mode41_kw_from_scan_config(cfg: Any) -> Dict[str, Any]:
    base = mode41_default_kw()
    for k in base:
        ck = f"mode41_{k}"
        if hasattr(cfg, ck):
            base[k] = getattr(cfg, ck)
    if hasattr(cfg, "mode41_min_score"):
        base["min_score"] = int(getattr(cfg, "mode41_min_score", 60))
    return base


def _weekly_up_to(
    rows: List[KlineRow],
    idx: int,
) -> Tuple[List[tuple], int, List[int]]:
    sub = rows[: idx + 1]
    weekly, last_idx = daily_to_weekly_with_volume_and_last_index(sub)
    if not weekly:
        return weekly, -1, last_idx
    return weekly, len(weekly) - 1, last_idx


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
    w_idx: int,
    low: float,
    period: int,
    *,
    ma_slope_weeks: int,
    max_ma_dist_pct: float,
) -> Optional[Dict[str, float]]:
    if w_idx < period - 1:
        return None
    ma = _ma_at(closes, w_idx, period)
    if np.isnan(ma) or ma <= 0:
        return None
    slope = _ma_slope_pct(closes, w_idx, period, ma_slope_weeks)
    if slope <= 0:
        return None
    dist = (low / ma - 1.0) * 100.0
    if abs(dist) > max_ma_dist_pct:
        return None
    return {
        "support_ma": float(period),
        "support_ma_val": ma,
        "low_dist_ma_pct": dist,
        "support_ma_slope_pct": slope,
    }


def _pick_support_ma(
    closes: np.ndarray,
    w_idx: int,
    low: float,
    *,
    ma_slope_weeks: int,
    max_ma_dist_pct: float,
) -> Optional[Dict[str, float]]:
    candidates: List[tuple[float, int, float, float]] = []
    for period in SUPPORT_MAS:
        if w_idx < period - 1:
            continue
        ma = _ma_at(closes, w_idx, period)
        if np.isnan(ma) or ma <= 0:
            continue
        slope = _ma_slope_pct(closes, w_idx, period, ma_slope_weeks)
        if slope <= 0:
            continue
        dist = (low / ma - 1.0) * 100.0
        if abs(dist) <= max_ma_dist_pct:
            candidates.append((abs(dist), period, ma, dist))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], -x[1]))
    _, period, ma, dist = candidates[0]
    return {
        "support_ma": float(period),
        "support_ma_val": ma,
        "low_dist_ma_pct": dist,
        "support_ma_slope_pct": _ma_slope_pct(closes, w_idx, period, ma_slope_weeks),
    }


def _next_week_low_after(rows: List[KlineRow], idx: int) -> Optional[float]:
    wk_sig = _week_key(str(rows[idx].date)[:10])
    lows: List[float] = []
    for j in range(idx + 1, len(rows)):
        if _week_key(str(rows[j].date)[:10]) != wk_sig:
            lows.append(float(rows[j].low))
    if not lows:
        return None
    return float(min(lows))


def _week_complete_at(rows: List[KlineRow], idx: int, last_idx: List[int], w_idx: int) -> bool:
    if w_idx < 0 or w_idx >= len(last_idx):
        return False
    return idx >= last_idx[w_idx]


def _vol_idx_for_check(
    rows: List[KlineRow],
    idx: int,
    w_idx: int,
    last_idx: List[int],
) -> int:
    """未完成周用上一完整周做量能比较，避免周初单日量成为近5周最低。"""
    if _week_complete_at(rows, idx, last_idx, w_idx):
        return w_idx
    return w_idx - 1


def match_mode41_weekly_ma_pullback(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    if _is_st(name or ""):
        return None
    kw = mode41_default_kw()
    kw.update({k: v for k, v in kwargs.items() if k in kw or k.startswith("mode41")})

    weekly, w_idx, last_idx = _weekly_up_to(rows, idx)
    if w_idx < 0 or not weekly:
        return None

    min_weeks = max(
        int(kw["phase_lookback_weeks"]),
        max(SUPPORT_MAS),
        int(kw["vol_lookback_weeks"]),
    ) + 5
    if w_idx < min_weeks:
        return None

    closes = np.array([float(w[4]) for w in weekly], dtype=float)
    highs = np.array([float(w[2]) for w in weekly], dtype=float)
    lows = np.array([float(w[3]) for w in weekly], dtype=float)
    vols = np.array([float(w[5]) for w in weekly], dtype=float)
    opens = np.array([float(w[1]) for w in weekly], dtype=float)

    peak_lb = int(kw["peak_lookback_weeks"])
    peak_end = int(kw["peak_end_offset_weeks"])
    phase_lb = int(kw["phase_lookback_weeks"])

    peak_start = max(0, w_idx - peak_lb)
    peak_end_i = max(peak_start, w_idx - peak_end)
    if peak_end_i <= peak_start:
        return None
    peak_slice = slice(peak_start, peak_end_i + 1)
    peak_i = peak_start + int(np.argmax(highs[peak_slice]))
    peak_high = float(highs[peak_i])
    if peak_high <= 0:
        return None

    phase_start = max(0, peak_i - phase_lb)
    phase_end = max(phase_start, peak_i - 2)
    if phase_end <= phase_start:
        return None
    phase_low = float(np.min(lows[phase_start : phase_end + 1]))
    if phase_low <= 0:
        return None

    rally_pct = (peak_high - phase_low) / phase_low * 100.0
    if rally_pct < float(kw["min_rally_pct"]):
        return None

    cur_low = float(lows[w_idx])
    cur_close = float(closes[w_idx])
    cur_high = float(highs[w_idx])
    cur_open = float(opens[w_idx])
    week_chg_pct = (cur_close / float(closes[w_idx - 1]) - 1.0) * 100.0 if w_idx > 0 else 0.0

    spike_w = int(kw["recent_spike_weeks"])
    max_spike = float(kw["max_recent_week_pct"])
    if spike_w > 0 and w_idx > 0:
        spike_start = max(1, w_idx - spike_w + 1)
        for j in range(spike_start, w_idx + 1):
            w_chg = (closes[j] / closes[j - 1] - 1.0) * 100.0
            if w_chg > max_spike:
                return None

    pullback_pct = (1.0 - cur_low / peak_high) * 100.0
    if pullback_pct < float(kw["pullback_min_pct"]) or pullback_pct > float(kw["pullback_max_pct"]):
        return None

    ma30 = _ma_at(closes, w_idx, 30)
    ma60 = _ma_at(closes, w_idx, 60)
    if np.isnan(ma30) or ma30 <= 0 or np.isnan(ma60) or ma60 <= 0:
        return None
    if kw.get("require_above_ma30_60", True):
        if cur_close <= ma30 or cur_close <= ma60:
            return None
    ma30_slope = _ma_slope_pct(closes, w_idx, 30, int(kw["ma_slope_weeks"]))
    if ma30_slope < float(kw["min_ma30_slope_pct"]):
        return None
    if cur_low < ma30 * (1.0 - float(kw["max_break_ma30_pct"]) / 100.0):
        return None

    ma_slope_weeks = int(kw["ma_slope_weeks"])
    max_ma_dist = float(kw["max_ma_dist_pct"])
    support_ma_only = int(kw.get("support_ma_only", 0) or 0)
    if support_ma_only > 0:
        support = _support_at_period(
            closes,
            w_idx,
            cur_low,
            support_ma_only,
            ma_slope_weeks=ma_slope_weeks,
            max_ma_dist_pct=max_ma_dist,
        )
    else:
        support = _pick_support_ma(
            closes,
            w_idx,
            cur_low,
            ma_slope_weeks=ma_slope_weeks,
            max_ma_dist_pct=max_ma_dist,
        )
    if not support:
        return None

    sup_period = int(support["support_ma"])
    if pullback_pct < _min_pullback_for_ma(sup_period, kw):
        return None

    if kw.get("require_pullback_trough", True):
        seg_low = float(np.min(lows[peak_i : w_idx + 1]))
        if cur_low > seg_low * 1.001:
            return None
        if kw.get("require_trough_confirm", True):
            next_low = _next_week_low_after(rows, idx)
            if next_low is not None and next_low <= cur_low:
                return None

    sup_ma = float(support["support_ma_val"])
    close_dist = (cur_close / sup_ma - 1.0) * 100.0
    if close_dist > float(kw["max_close_above_ma_pct"]):
        return None
    if cur_close < sup_ma * (1.0 - float(kw["max_break_ma30_pct"]) / 100.0):
        return None

    vol_lb = int(kw["vol_lookback_weeks"])
    vol_idx = _vol_idx_for_check(rows, idx, w_idx, last_idx)
    if vol_idx < vol_lb - 1:
        return None
    vol_slice = vols[vol_idx - vol_lb + 1 : vol_idx + 1]
    min_vol = float(np.min(vol_slice))
    cur_vol = float(vols[vol_idx])
    vol_vs_min_pct = (cur_vol / min_vol - 1.0) * 100.0 if min_vol > 0 else 0.0
    if kw.get("require_vol_near_min", True):
        max_above = float(kw["max_vol_above_min_pct"]) / 100.0
        if cur_vol > min_vol * (1.0 + max_above):
            return None

    rng = cur_high - cur_low
    lower_shadow = (min(cur_open, cur_close) - cur_low) / rng if rng > 0 else 0.0
    body_ratio = (cur_close - cur_open) / rng if rng > 0 else 0.0

    signal_date = str(rows[idx].date)[:10]
    exec_buy_date = ""
    exec_buy_open = 0.0
    if idx + 1 < len(rows):
        exec_buy_date = str(rows[idx + 1].date)[:10]
        exec_buy_open = float(rows[idx + 1].open)

    return {
        "phase_low": phase_low,
        "peak_high": peak_high,
        "peak_date": str(rows[last_idx[peak_i]].date)[:10] if peak_i < len(last_idx) else str(weekly[peak_i][0]),
        "rally_pct": rally_pct,
        "pullback_pct": pullback_pct,
        "support_ma": support["support_ma"],
        "support_ma_val": sup_ma,
        "low_dist_ma_pct": support["low_dist_ma_pct"],
        "close_dist_ma_pct": close_dist,
        "support_ma_slope_pct": support["support_ma_slope_pct"],
        "ma30_slope_pct": ma30_slope,
        "ma30": ma30,
        "ma60": ma60,
        "vol_5w_min": min_vol,
        "vol_week_idx": vol_idx,
        "vol_vs_min_pct": vol_vs_min_pct,
        "week_chg_pct": week_chg_pct,
        "lower_shadow_ratio": lower_shadow,
        "body_ratio": body_ratio,
        "close": cur_close,
        "low": cur_low,
        "signal_date": signal_date,
        "exec_buy_date": exec_buy_date,
        "exec_buy_open": exec_buy_open,
        "week_idx": w_idx,
    }


def score_mode41_weekly_ma_pullback(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> int:
    m = match_mode41_weekly_ma_pullback(rows, idx, code, name, **kwargs)
    if not m:
        return 0
    score = 40.0
    score += min(18.0, float(m["rally_pct"]) * 0.07)
    pb = float(m["pullback_pct"])
    if 14.0 <= pb <= 28.0:
        score += 14.0
    elif 10.0 <= pb <= 35.0:
        score += 8.0
    dist = abs(float(m["low_dist_ma_pct"]))
    if dist <= 2.5:
        score += 16.0
    elif dist <= 5.0:
        score += 10.0
    else:
        score += 5.0
    sup = int(m["support_ma"])
    if sup >= 60:
        score += 12.0
    elif sup >= 30:
        score += 8.0
    elif sup >= 20:
        score += 6.0
    elif sup >= 10:
        score += 4.0
    else:
        score += 3.0
    if float(m["support_ma_slope_pct"]) > 2.0:
        score += 6.0
    if float(m["ma30_slope_pct"]) > 2.0:
        score += 6.0
    vv = float(m["vol_vs_min_pct"])
    if vv <= 3.0:
        score += 12.0
    elif vv <= float(kwargs.get("max_vol_above_min_pct", 15.0) or 15.0):
        score += 6.0
    if float(m["week_chg_pct"]) >= 3.0:
        score += 8.0
    elif float(m["week_chg_pct"]) > 0:
        score += 4.0
    if float(m["lower_shadow_ratio"]) >= 0.35:
        score += 6.0
    return int(min(100, max(0, round(score))))


def mode41_signal_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    m = match_mode41_weekly_ma_pullback(rows, idx, code, name, **kwargs)
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
        "ma30_slope_pct": round(float(m["ma30_slope_pct"]), 2),
        "ma30": round(float(m["ma30"]), 4),
        "ma60": round(float(m["ma60"]), 4),
        "vol_5w_min": round(float(m["vol_5w_min"]), 2),
        "vol_vs_min_pct": round(float(m["vol_vs_min_pct"]), 2),
        "week_chg_pct": round(float(m["week_chg_pct"]), 2),
        "lower_shadow_ratio": round(float(m["lower_shadow_ratio"]), 2),
        "low": round(float(m["low"]), 4),
        "signal_date": m["signal_date"],
        "exec_buy_date": m.get("exec_buy_date", ""),
        "exec_buy_open": round(float(m.get("exec_buy_open", 0) or 0), 4),
    }


def dedupe_mode41_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[tuple, Dict[str, Any]] = {}
    for h in hits:
        key = (h.get("code", ""), h.get("peak_date", ""), int(h.get("support_ma", 0) or 0))
        low = float(h.get("low", h.get("support_ma_val", 0)) or 0)
        prev = best.get(key)
        if prev is None or low < float(prev.get("low", prev.get("support_ma_val", 0)) or 0):
            best[key] = h
    return sorted(best.values(), key=lambda x: (x.get("date", ""), x.get("code", "")))
