"""mode42 周线短均线回踩缩量回升：大牛周线拉升后浅调，前周阴线极致缩量踩 MA5/10，当周转阳确认。

样本：
  - 精测电子 300567：2026-02-13（前周 02-06 阴探底，当周阳 +11.9%）
  - 新易盛 300502：2026-05-08（前周 04-30 阴，当周阳 +4.9%）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.scanner import KlineRow, _is_st
from app.weekly_ma import daily_to_weekly_with_volume_and_last_index

MODE42_ID = "mode42"
MODE42_FULL_NAME = "周线短均线回踩缩量回升"
MODE42_DISPLAY_NAME = f"{MODE42_ID}（{MODE42_FULL_NAME}）"
MODE42_ONE_LINE = (
    "周线大涨后浅中调，前周阴线极致缩量踩周 MA5/10，"
    "当周转阳线确认；信号日次日开盘买"
)

SHORT_MAS: tuple[int, ...] = (5, 10)


def mode42_default_kw() -> Dict[str, Any]:
    return dict(
        phase_lookback_weeks=48,
        peak_lookback_weeks=20,
        peak_end_offset_weeks=1,
        min_rally_pct=50.0,
        pullback_min_pct=8.0,
        pullback_max_pct=25.0,
        ma_touch_pct=8.0,
        max_close_above_touch_ma_pct=15.0,
        ma_slope_weeks=4,
        min_ma5_slope_pct=0.0,
        min_ma10_slope_pct=0.0,
        min_ma20_slope_pct=0.0,
        require_above_ma20=True,
        max_break_ma20_pct=8.0,
        vol_lookback_weeks=5,
        max_vol_above_min_pct=5.0,
        require_vol_near_min=True,
        require_vol_extreme=True,
        require_week_up=True,
        require_week_yang=True,
        require_prev_week_yin=True,
        min_week_chg_pct=0.0,
        recent_spike_weeks=2,
        max_recent_week_pct=15.0,
        require_pullback_trough=True,
        trough_tol_pct=2.0,
        support_ma_only=0,
        min_score=60,
    )


def mode42_kw_from_scan_config(cfg: Any) -> Dict[str, Any]:
    base = mode42_default_kw()
    for k in base:
        ck = f"mode42_{k}"
        if hasattr(cfg, ck):
            base[k] = getattr(cfg, ck)
    if hasattr(cfg, "mode42_min_score"):
        base["min_score"] = int(getattr(cfg, "mode42_min_score", 60))
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
    if _week_complete_at(rows, idx, last_idx, w_idx):
        return w_idx
    return w_idx - 1


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


def _pick_short_ma_touch(
    closes: np.ndarray,
    w_idx: int,
    low: float,
    *,
    ma_slope_weeks: int,
    ma_touch_pct: float,
    min_ma5_slope_pct: float,
    min_ma10_slope_pct: float,
) -> Optional[Dict[str, float]]:
    """低点触及上升中的周 MA5 或 MA10；可刺破 MA5 时优先认 MA5。"""
    touch = ma_touch_pct
    ma5 = _ma_at(closes, w_idx, 5)
    if w_idx >= 4 and not np.isnan(ma5) and ma5 > 0:
        slope5 = _ma_slope_pct(closes, w_idx, 5, ma_slope_weeks)
        if slope5 >= min_ma5_slope_pct:
            d5 = (low / ma5 - 1.0) * 100.0
            if abs(d5) <= touch:
                return {
                    "support_ma": 5.0,
                    "support_ma_val": ma5,
                    "low_dist_ma_pct": d5,
                    "support_ma_slope_pct": slope5,
                }

    candidates: List[tuple[float, int, float, float]] = []
    for period in SHORT_MAS:
        if w_idx < period - 1:
            continue
        ma = _ma_at(closes, w_idx, period)
        if np.isnan(ma) or ma <= 0:
            continue
        min_slope = min_ma5_slope_pct if period == 5 else min_ma10_slope_pct
        slope = _ma_slope_pct(closes, w_idx, period, ma_slope_weeks)
        if slope < min_slope:
            continue
        dist = (low / ma - 1.0) * 100.0
        if abs(dist) <= touch:
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


def match_mode42_weekly_short_ma_rebound(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    if _is_st(name or ""):
        return None
    kw = mode42_default_kw()
    kw.update({k: v for k, v in kwargs.items() if k in kw or k.startswith("mode42")})

    weekly, w_idx, last_idx = _weekly_up_to(rows, idx)
    if w_idx < 0 or not weekly:
        return None

    min_weeks = max(
        int(kw["phase_lookback_weeks"]),
        20,
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

    if w_idx < 1:
        return None

    prev_open = float(opens[w_idx - 1])
    prev_close = float(closes[w_idx - 1])
    prev_low = float(lows[w_idx - 1])
    prev_week_chg_pct = (
        (prev_close / float(closes[w_idx - 2]) - 1.0) * 100.0 if w_idx >= 2 else 0.0
    )

    if kw.get("require_prev_week_yin", True):
        if prev_close >= prev_open:
            return None

    if kw.get("require_week_yang", True):
        if cur_close <= cur_open:
            return None

    if kw.get("require_week_up", True):
        if week_chg_pct <= float(kw.get("min_week_chg_pct", 0.0)):
            return None

    # 阴线探底周低点用于回踩幅度与均线支撑判断
    probe_w_idx = w_idx - 1
    probe_low = prev_low

    spike_w = int(kw["recent_spike_weeks"])
    max_spike = float(kw["max_recent_week_pct"])
    if spike_w > 0 and w_idx > 0:
        spike_start = max(1, w_idx - spike_w + 1)
        for j in range(spike_start, w_idx + 1):
            w_chg = (closes[j] / closes[j - 1] - 1.0) * 100.0
            if w_chg > max_spike:
                return None

    pullback_pct = (1.0 - probe_low / peak_high) * 100.0
    if pullback_pct < float(kw["pullback_min_pct"]) or pullback_pct > float(kw["pullback_max_pct"]):
        return None

    ma20 = _ma_at(closes, w_idx, 20)
    if np.isnan(ma20) or ma20 <= 0:
        return None
    if kw.get("require_above_ma20", True) and cur_close <= ma20:
        return None
    ma20_slope = _ma_slope_pct(closes, w_idx, 20, int(kw["ma_slope_weeks"]))
    if ma20_slope < float(kw["min_ma20_slope_pct"]):
        return None
    if cur_low < ma20 * (1.0 - float(kw["max_break_ma20_pct"]) / 100.0):
        return None

    ma_slope_weeks = int(kw["ma_slope_weeks"])
    touch_pct = float(kw["ma_touch_pct"])
    support_ma_only = int(kw.get("support_ma_only", 0) or 0)
    if support_ma_only in SHORT_MAS:
        period = support_ma_only
        if probe_w_idx < period - 1:
            return None
        ma = _ma_at(closes, probe_w_idx, period)
        if np.isnan(ma) or ma <= 0:
            return None
        min_slope = float(kw["min_ma5_slope_pct"]) if period == 5 else float(kw["min_ma10_slope_pct"])
        slope = _ma_slope_pct(closes, probe_w_idx, period, ma_slope_weeks)
        if slope < min_slope:
            return None
        dist = (probe_low / ma - 1.0) * 100.0
        if abs(dist) > touch_pct:
            return None
        support = {
            "support_ma": float(period),
            "support_ma_val": ma,
            "low_dist_ma_pct": dist,
            "support_ma_slope_pct": slope,
        }
    else:
        support = _pick_short_ma_touch(
            closes,
            probe_w_idx,
            probe_low,
            ma_slope_weeks=ma_slope_weeks,
            ma_touch_pct=touch_pct,
            min_ma5_slope_pct=float(kw["min_ma5_slope_pct"]),
            min_ma10_slope_pct=float(kw["min_ma10_slope_pct"]),
        )
    if not support:
        return None

    if kw.get("require_pullback_trough", True):
        seg_low = float(np.min(lows[peak_i : w_idx + 1]))
        tol = float(kw.get("trough_tol_pct", 2.0)) / 100.0
        if probe_low > seg_low * (1.0 + tol):
            return None

    sup_ma = float(support["support_ma_val"])
    close_dist = (cur_close / sup_ma - 1.0) * 100.0
    if close_dist > float(kw["max_close_above_touch_ma_pct"]):
        return None
    if cur_close < sup_ma * (1.0 - float(kw["max_break_ma20_pct"]) / 100.0):
        return None

    vol_lb = int(kw["vol_lookback_weeks"])
    vol_idx = probe_w_idx
    if vol_idx < vol_lb - 1:
        return None
    vol_slice = vols[vol_idx - vol_lb + 1 : vol_idx + 1]
    min_vol = float(np.min(vol_slice))
    probe_vol = float(vols[probe_w_idx])
    max_above = float(kw["max_vol_above_min_pct"]) / 100.0
    vol_vs_min_pct = (probe_vol / min_vol - 1.0) * 100.0 if min_vol > 0 else 0.0
    if kw.get("require_vol_near_min", True) or kw.get("require_vol_extreme", True):
        if probe_vol > min_vol * (1.0 + max_above):
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
        "peak_date": str(rows[last_idx[peak_i]].date)[:10] if peak_i < len(last_idx) else "",
        "rally_pct": rally_pct,
        "pullback_pct": pullback_pct,
        "support_ma": support["support_ma"],
        "support_ma_val": sup_ma,
        "low_dist_ma_pct": support["low_dist_ma_pct"],
        "close_dist_ma_pct": close_dist,
        "support_ma_slope_pct": support["support_ma_slope_pct"],
        "ma20_slope_pct": ma20_slope,
        "ma20": ma20,
        "vol_5w_min": min_vol,
        "vol_week_idx": vol_idx,
        "vol_vs_min_pct": vol_vs_min_pct,
        "week_chg_pct": week_chg_pct,
        "prev_week_chg_pct": prev_week_chg_pct,
        "probe_week_date": str(rows[last_idx[probe_w_idx]].date)[:10],
        "probe_low": probe_low,
        "lower_shadow_ratio": lower_shadow,
        "body_ratio": body_ratio,
        "close": cur_close,
        "low": probe_low,
        "signal_date": signal_date,
        "exec_buy_date": exec_buy_date,
        "exec_buy_open": exec_buy_open,
        "week_idx": w_idx,
    }


def score_mode42_weekly_short_ma_rebound(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> int:
    m = match_mode42_weekly_short_ma_rebound(rows, idx, code, name, **kwargs)
    if not m:
        return 0
    score = 38.0
    score += min(16.0, float(m["rally_pct"]) * 0.05)
    pb = float(m["pullback_pct"])
    if 12.0 <= pb <= 20.0:
        score += 14.0
    elif 8.0 <= pb <= 25.0:
        score += 8.0
    dist = abs(float(m["low_dist_ma_pct"]))
    if dist <= 2.0:
        score += 18.0
    elif dist <= 5.0:
        score += 12.0
    else:
        score += 6.0
    sup = int(m["support_ma"])
    score += 6.0 if sup >= 10 else 4.0
    if float(m["support_ma_slope_pct"]) > 5.0:
        score += 8.0
    elif float(m["support_ma_slope_pct"]) > 0:
        score += 4.0
    if float(m["ma20_slope_pct"]) > 3.0:
        score += 6.0
    vv = float(m["vol_vs_min_pct"])
    if vv <= 1.0:
        score += 14.0
    elif vv <= float(kwargs.get("max_vol_above_min_pct", 5.0) or 5.0):
        score += 8.0
    wc = float(m["week_chg_pct"])
    if wc >= 4.0:
        score += 12.0
    elif wc > 0:
        score += 6.0
    if float(m["lower_shadow_ratio"]) >= 0.3:
        score += 6.0
    return int(min(100, max(0, round(score))))


def mode42_signal_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    m = match_mode42_weekly_short_ma_rebound(rows, idx, code, name, **kwargs)
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
        "ma20_slope_pct": round(float(m["ma20_slope_pct"]), 2),
        "ma20": round(float(m["ma20"]), 4),
        "vol_5w_min": round(float(m["vol_5w_min"]), 2),
        "vol_vs_min_pct": round(float(m["vol_vs_min_pct"]), 2),
        "week_chg_pct": round(float(m["week_chg_pct"]), 2),
        "prev_week_chg_pct": round(float(m.get("prev_week_chg_pct", 0) or 0), 2),
        "probe_week_date": m.get("probe_week_date", ""),
        "probe_low": round(float(m.get("probe_low", m.get("low", 0)) or 0), 4),
        "lower_shadow_ratio": round(float(m["lower_shadow_ratio"]), 2),
        "low": round(float(m["low"]), 4),
        "signal_date": m["signal_date"],
        "exec_buy_date": m.get("exec_buy_date", ""),
        "exec_buy_open": round(float(m.get("exec_buy_open", 0) or 0), 4),
    }


def dedupe_mode42_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[tuple, Dict[str, Any]] = {}
    for h in hits:
        key = (h.get("code", ""), h.get("peak_date", ""), int(h.get("support_ma", 0) or 0))
        low = float(h.get("low", h.get("support_ma_val", 0)) or 0)
        prev = best.get(key)
        if prev is None or low < float(prev.get("low", prev.get("support_ma_val", 0)) or 0):
            best[key] = h
    return sorted(best.values(), key=lambda x: (x.get("date", ""), x.get("code", "")))
