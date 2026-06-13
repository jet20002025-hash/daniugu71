"""mode37 跳空缺口支撑：向上跳空未回补，回踩缺口下沿/缺口区缩量后再度走强。

样本：蓝箭电子 301348
  - 缺口 2026-01-12/13（上沿~23.50，下沿~22.33）
  - 信号日 2026-04-07、2026-06-09（回踩缺口 + 日线 KDJ(8,2,2) J 上拐）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.scanner import KlineRow, _is_st, _modepbs_big_pct_threshold, _vol_ratio_at
from app.weekly_ma import weekly_kdj

MODE37_ID = "mode37"
MODE37_FULL_NAME = "跳空缺口支撑"
MODE37_DISPLAY_NAME = f"{MODE37_ID}（{MODE37_FULL_NAME}）"
MODE37_ONE_LINE = "向上跳空未有效回补，回踩缺口区缩量企稳/反包"


def mode37_default_kw() -> Dict[str, Any]:
    return dict(
        gap_lookback_min=5,
        gap_lookback_max=180,
        min_gap_size_pct=1.0,
        gap_day_pct_min=3.0,
        gap_day_pct_min_main=2.0,
        gap_vol_min=1.5,
        max_break_below_pct=0.0,
        zone_above_pct=3.0,
        vol_ma=20,
        shrink_vol_max=0.85,
        min_touch_dist_pct=0.0,
        max_touch_dist_pct=3.0,
        max_low_above_top_pct=2.0,
        max_close_dist_pct=5.0,
        max_close_above_top_pct=3.0,
        max_close_dist_pct_j_up=12.0,
        max_close_above_top_pct_j_up=12.0,
        min_score=60,
        require_bounce=False,
        bounce_pct_min=0.0,
        kdj_n=8,
        kdj_m1=2,
        kdj_m2=2,
        require_kdj_j_turn_up=True,
    )


def mode37_kw_from_scan_config(cfg: Any) -> Dict[str, Any]:
    base = mode37_default_kw()
    for k in base:
        ck = f"mode37_{k}"
        if hasattr(cfg, ck):
            base[k] = getattr(cfg, ck)
    if hasattr(cfg, "mode37_min_score"):
        base["min_score"] = int(getattr(cfg, "mode37_min_score", 60))
    return base


def _row_volume(rows: List[KlineRow], idx: int) -> float:
    r = rows[idx]
    return float(getattr(r, "volume", 0) or getattr(r, "vol", 0) or 0)


def _daily_kdj_series(
    rows: List[KlineRow],
    *,
    n: int,
    m1: int,
    m2: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bars = [(r.date, r.open, r.high, r.low, r.close) for r in rows]
    return weekly_kdj(bars, n=n, m1=m1, m2=m2)


def _kdj_j_turn_up(
    k: np.ndarray,
    d: np.ndarray,
    j: np.ndarray,
    idx: int,
) -> bool:
    """J 上拐：当日 J > 昨日 J（与周线回测「J 拐头向下」对称）。"""
    if idx < 1 or idx >= len(j):
        return False
    if np.isnan(j[idx]) or np.isnan(j[idx - 1]):
        return False
    return float(j[idx]) > float(j[idx - 1])


def _kdj_at_idx(
    k: np.ndarray,
    d: np.ndarray,
    j: np.ndarray,
    idx: int,
) -> Optional[Dict[str, float]]:
    if idx < 0 or idx >= len(j):
        return None
    if np.isnan(k[idx]) or np.isnan(d[idx]) or np.isnan(j[idx]):
        return None
    return {"kdj_k": float(k[idx]), "kdj_d": float(d[idx]), "kdj_j": float(j[idx])}


def _find_up_gaps(
    rows: List[KlineRow],
    code: str,
    name: str,
    *,
    gap_lookback_min: int,
    gap_lookback_max: int,
    min_gap_size_pct: float,
    gap_day_pct_min: float,
    gap_day_pct_min_main: float,
    gap_vol_min: float,
    vol_ma: int,
    max_break_below_pct: float,
    end_idx: int,
) -> List[Dict[str, Any]]:
    """在 end_idx 之前寻找仍有效的向上跳空（回看 gap_lookback_max 个交易日，未破下沿则有效）。"""
    gaps: List[Dict[str, Any]] = []
    start = max(2, end_idx - gap_lookback_max)
    pct_min = _modepbs_big_pct_threshold(
        code, name, big_pct_min=gap_day_pct_min, big_pct_min_main=gap_day_pct_min_main
    )
    brk_tol = max_break_below_pct / 100.0

    for i in range(start, end_idx):
        if i < 1:
            continue
        prev_h = float(rows[i - 1].high)
        cur_l = float(rows[i].low)
        if prev_h <= 0 or cur_l <= prev_h:
            continue
        gap_size = (cur_l - prev_h) / prev_h
        if gap_size * 100.0 < min_gap_size_pct:
            continue
        pct = float(getattr(rows[i], "pct_chg", 0) or 0)
        if pct < pct_min:
            continue
        vr = _vol_ratio_at(rows, i, vol_ma)
        if vr < gap_vol_min:
            continue
        gap_bottom = prev_h
        gap_top = cur_l
        age = end_idx - i
        if age < gap_lookback_min:
            continue

        broken = False
        for j in range(i + 1, end_idx + 1):
            if float(rows[j].low) < gap_bottom * (1.0 - brk_tol):
                broken = True
                break
        if broken:
            continue

        gaps.append(
            {
                "gap_idx": i,
                "gap_date": str(rows[i].date)[:10],
                "gap_bottom": gap_bottom,
                "gap_top": gap_top,
                "gap_size_pct": gap_size * 100.0,
                "gap_pct_chg": pct,
                "gap_vol_ratio": vr,
                "gap_age_days": age,
            }
        )
    return gaps


def _touch_at_support(
    rows: List[KlineRow],
    idx: int,
    gap: Dict[str, Any],
    *,
    zone_above_pct: float,
    max_touch_dist_pct: float,
    min_touch_dist_pct: float,
    max_low_above_top_pct: float,
    max_close_dist_pct: float,
    max_close_above_top_pct: float,
    j_turn_up: bool = False,
    max_close_dist_pct_j_up: float = 12.0,
    max_close_above_top_pct_j_up: float = 12.0,
) -> Optional[Dict[str, float]]:
    gap_bottom = float(gap["gap_bottom"])
    gap_top = float(gap["gap_top"])
    if gap_bottom <= 0:
        return None
    low = float(rows[idx].low)
    high = float(rows[idx].high)
    close = float(rows[idx].close)

    zone_top = gap_top * (1.0 + zone_above_pct / 100.0)
    low_cap = gap_top * (1.0 + max_low_above_top_pct / 100.0)
    close_dist_limit = (
        float(max_close_dist_pct_j_up) if j_turn_up else float(max_close_dist_pct)
    )
    close_above_limit = (
        float(max_close_above_top_pct_j_up) if j_turn_up else float(max_close_above_top_pct)
    )
    close_cap = gap_top * (1.0 + close_above_limit / 100.0)

    if low > zone_top or low > low_cap:
        return None

    dist_pct = (low - gap_bottom) / gap_bottom * 100.0
    in_gap = low <= gap_top
    if dist_pct < min_touch_dist_pct:
        return None
    # 低点已在缺口区间内：视为回踩缺口，不按距下沿百分比上限剔除（宽缺口上沿回踩）
    if not in_gap and dist_pct > max_touch_dist_pct:
        return None

    close_dist_pct = (close - gap_bottom) / gap_bottom * 100.0
    if close_dist_pct > close_dist_limit or close > close_cap:
        return None

    return {
        "touch_dist_pct": dist_pct,
        "close_dist_pct": close_dist_pct,
        "in_gap_zone": 1.0 if in_gap else 0.0,
        "low": low,
        "high": high,
        "close": close,
    }


def _count_zone_touches(
    rows: List[KlineRow],
    gap_idx: int,
    idx: int,
    gap_bottom: float,
    gap_top: float,
    zone_above_pct: float,
) -> int:
    zone_top = gap_top * (1.0 + zone_above_pct / 100.0)
    touches = 0
    for j in range(gap_idx + 1, idx + 1):
        low = float(rows[j].low)
        if gap_bottom <= low <= zone_top:
            touches += 1
    return touches


def match_mode37_gap_support(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    if _is_st(name):
        return None
    kw = mode37_default_kw()
    kw.update({k: v for k, v in kwargs.items() if k in kw or k.startswith("mode37")})

    need = max(
        int(kw["vol_ma"]) + 1,
        int(kw["gap_lookback_min"]) + 2,
        int(kw["kdj_n"]),
    )
    if idx < need or idx >= len(rows):
        return None

    kdj_n = int(kw["kdj_n"])
    kdj_m1 = int(kw["kdj_m1"])
    kdj_m2 = int(kw["kdj_m2"])
    require_j_up = bool(kw.get("require_kdj_j_turn_up", True))
    kd, dd, jd = _daily_kdj_series(rows, n=kdj_n, m1=kdj_m1, m2=kdj_m2)
    if require_j_up and not _kdj_j_turn_up(kd, dd, jd, idx):
        return None
    kdj_snap = _kdj_at_idx(kd, dd, jd, idx)
    j_turn_up = _kdj_j_turn_up(kd, dd, jd, idx)

    gaps = _find_up_gaps(
        rows,
        code,
        name,
        gap_lookback_min=int(kw["gap_lookback_min"]),
        gap_lookback_max=int(kw["gap_lookback_max"]),
        min_gap_size_pct=float(kw["min_gap_size_pct"]),
        gap_day_pct_min=float(kw["gap_day_pct_min"]),
        gap_day_pct_min_main=float(kw["gap_day_pct_min_main"]),
        gap_vol_min=float(kw["gap_vol_min"]),
        vol_ma=int(kw["vol_ma"]),
        max_break_below_pct=float(kw["max_break_below_pct"]),
        end_idx=idx - 1,
    )
    if not gaps:
        return None

    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    vol_ma = int(kw["vol_ma"])
    zone_above = float(kw["zone_above_pct"])
    max_dist = float(kw["max_touch_dist_pct"])
    min_dist = float(kw["min_touch_dist_pct"])
    max_low_above_top = float(kw["max_low_above_top_pct"])
    max_close_above_top = float(kw["max_close_above_top_pct"])
    require_bounce = bool(kw.get("require_bounce"))
    bounce_min = float(kw.get("bounce_pct_min", 0))

    pct_today = float(getattr(rows[idx], "pct_chg", 0) or 0)
    vr_today = _vol_ratio_at(rows, idx, vol_ma)

    for gap in gaps:
        touch = _touch_at_support(
            rows,
            idx,
            gap,
            zone_above_pct=zone_above,
            max_touch_dist_pct=max_dist,
            min_touch_dist_pct=min_dist,
            max_low_above_top_pct=max_low_above_top,
            max_close_dist_pct=float(kw["max_close_dist_pct"]),
            max_close_above_top_pct=max_close_above_top,
            j_turn_up=j_turn_up,
            max_close_dist_pct_j_up=float(kw["max_close_dist_pct_j_up"]),
            max_close_above_top_pct_j_up=float(kw["max_close_above_top_pct_j_up"]),
        )
        if touch is None:
            continue
        if require_bounce and pct_today < bounce_min:
            continue

        touches = _count_zone_touches(
            rows,
            int(gap["gap_idx"]),
            idx,
            float(gap["gap_bottom"]),
            float(gap["gap_top"]),
            zone_above,
        )

        sig_type = "support"
        if pct_today >= 5.0:
            sig_type = "strong_bounce"
        elif pct_today > 0:
            sig_type = "bounce"

        m = dict(gap)
        m.update(touch)
        m["touch_count"] = float(touches)
        m["vol_ratio"] = vr_today
        m["pct_chg"] = pct_today
        m["signal_type"] = sig_type
        m["signal_date"] = str(rows[idx].date)[:10]

        # 优先：更近的缺口、更多回踩次数、距离下沿更近
        pri = (
            touches * 10
            + float(gap["gap_vol_ratio"])
            + max(0.0, 5.0 - abs(touch["touch_dist_pct"]))
            - float(gap["gap_age_days"]) * 0.01
        )
        if pri > best_score:
            best_score = pri
            best = m

    if not best:
        return None
    if kdj_snap:
        best.update(kdj_snap)
    return best


def score_mode37_gap_support(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> int:
    m = match_mode37_gap_support(rows, idx, code, name, **kwargs)
    if not m:
        return 0
    score = 48.0
    score += min(15.0, float(m["gap_vol_ratio"]) * 4.0)
    score += min(12.0, float(m["gap_pct_chg"]) * 0.6)
    score += min(10.0, float(m["gap_size_pct"]) * 2.0)
    if float(m["vol_ratio"]) <= float(kwargs.get("shrink_vol_max", 0.85) or 0.85):
        score += 14.0
    elif float(m["vol_ratio"]) <= 1.0:
        score += 6.0
    tc = int(m.get("touch_count", 0))
    if tc >= 2:
        score += 12.0
    elif tc >= 1:
        score += 4.0
    pct = float(m["pct_chg"])
    if pct >= 7.0:
        score += 12.0
    elif pct > 0:
        score += 6.0
    if float(m.get("in_gap_zone", 0)) >= 1:
        score += 4.0
    dist = float(m["touch_dist_pct"])
    if 0 <= dist <= 3.0:
        score += 6.0
    return int(min(100, max(0, round(score))))


def mode37_signal_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    m = match_mode37_gap_support(rows, idx, code, name, **kwargs)
    if not m:
        return {}
    return {
        "gap_date": m["gap_date"],
        "gap_bottom": round(float(m["gap_bottom"]), 4),
        "gap_top": round(float(m["gap_top"]), 4),
        "gap_size_pct": round(float(m["gap_size_pct"]), 2),
        "gap_pct_chg": round(float(m["gap_pct_chg"]), 2),
        "gap_vol_ratio": round(float(m["gap_vol_ratio"]), 2),
        "gap_age_days": int(m["gap_age_days"]),
        "touch_dist_pct": round(float(m["touch_dist_pct"]), 2),
        "close_dist_pct": round(float(m.get("close_dist_pct", 0)), 2),
        "touch_count": int(m["touch_count"]),
        "vol_ratio": round(float(m["vol_ratio"]), 2),
        "pct_chg": round(float(m["pct_chg"]), 2),
        "signal_type": m["signal_type"],
        "kdj_k": round(float(m.get("kdj_k", 0)), 2),
        "kdj_d": round(float(m.get("kdj_d", 0)), 2),
        "kdj_j": round(float(m.get("kdj_j", 0)), 2),
    }
