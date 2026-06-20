"""mode39 大阳锚点回踩再升：放量大阳线作锚，回踩锚点收盘或阳线起点附近企稳后再攀升。

样本：宇瞳光学 300790
  - 2026-04-08 放量大阳锚点（收盘 28.42）
  - 2026-04-29 贴近锚点小十字企稳 → 买点 4/30 开盘
  - 2026-06-10 长下影十字探底 → 买点 6/11 开盘
样本：世名科技 300522
  - 2026-04-07 放量大阳锚点（低 10.99 / 开 11.06 / 收 11.94）
  - 2026-06-08 最低价踩锚点低企稳（低 10.93）→ 买点 6/09 开盘

买点（默认）：信号日收盘确认企稳后，**次日开盘价**买入。
激进买点：信号日最高价（盘中突破试仓，见 buy_trigger_above）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.scanner import KlineRow, _is_st, _limit_rate, _vol_ratio_at

MODE39_ID = "mode39"
MODE39_FULL_NAME = "大阳锚点回踩再升"
MODE39_DISPLAY_NAME = f"{MODE39_ID}（{MODE39_FULL_NAME}）"
MODE39_ONE_LINE = (
    "放量大阳作锚；锚点至信号日之间最低价不低于锚点日最低价×0.97；"
    "回踩锚点收盘、阳线起点或锚点最低价企稳；MA45 向上；信号日确认后次日开盘买"
)


def mode39_default_kw() -> Dict[str, Any]:
    return dict(
        anchor_lookback_min=5,
        anchor_lookback_max=120,
        big_pct_min=5.0,
        body_ratio_min=0.55,
        anchor_vol_mult=1.25,
        vol_ma=20,
        min_rally_pct=8.0,
        min_pullback_from_peak_pct=5.0,
        near_anchor_pct=3.0,
        near_anchor_open_pct=5.0,
        near_anchor_low_pct=3.0,
        anchor_low_floor=0.97,
        pullback_open_stabilize_pct_max=5.0,
        pullback_open_min_shadow=0.20,
        anchor_pick_max_dist_pct=15.0,
        stabilize_body_ratio_max=0.45,
        stabilize_pct_max=3.5,
        deep_pullback_from_peak_pct=15.0,
        deep_close_floor=0.72,
        long_shadow_ratio_min=0.55,
        long_shadow_max_above_anchor_pct=3.0,
        long_body_ratio_max=0.30,
        require_trough_confirm=True,
        shrink_vol_max=1.15,
        require_ma45_up=True,
        ma45_period=45,
        ma45_slope_days=10,
        min_ma45_slope_pct=0.0,
        min_score=60,
    )


def mode39_kw_from_scan_config(cfg: Any) -> Dict[str, Any]:
    base = mode39_default_kw()
    for k in base:
        ck = f"mode39_{k}"
        if hasattr(cfg, ck):
            base[k] = getattr(cfg, ck)
    if hasattr(cfg, "mode39_min_score"):
        base["min_score"] = int(getattr(cfg, "mode39_min_score", 60))
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


def _ma45_up_ok(closes: np.ndarray, idx: int, kw: Dict[str, Any]) -> tuple[bool, float, float]:
    period = int(kw.get("ma45_period", 45) or 45)
    look = int(kw.get("ma45_slope_days", 10) or 10)
    min_slope = float(kw.get("min_ma45_slope_pct", 0.0) or 0.0)
    ma_now = _ma_at(closes, idx, period)
    slope = _ma_slope_pct(closes, idx, period, look)
    if np.isnan(ma_now) or ma_now <= 0:
        return False, 0.0, slope
    if slope <= min_slope:
        return False, ma_now, slope
    return True, ma_now, slope


def _bar_ratios(r: KlineRow) -> tuple[float, float, float]:
    o, c, h, l_ = float(r.open), float(r.close), float(r.high), float(r.low)
    rng = h - l_
    if rng <= 0:
        return 0.0, 0.0, 0.0
    body = (c - o) / rng
    lower = (min(o, c) - l_) / rng
    upper = (h - max(o, c)) / rng
    return body, lower, upper


def _is_big_yang_anchor(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    *,
    big_pct_min: float,
    body_ratio_min: float,
    anchor_vol_mult: float,
    vol_ma: int,
) -> bool:
    r = rows[idx]
    o, c, h, l_ = float(r.open), float(r.close), float(r.high), float(r.low)
    if c <= o:
        return False
    pct = float(getattr(r, "pct_chg", 0.0) or 0.0)
    if pct < big_pct_min:
        return False
    if pct >= _limit_rate(code, name) * 100 - 0.6:
        return False
    rng = h - l_
    if rng <= 0 or (c - o) / rng < body_ratio_min:
        return False
    vr = _vol_ratio_at(rows, idx, vol_ma)
    if vr < anchor_vol_mult:
        return False
    return True


def _is_near_anchor_stabilize(
    r: KlineRow,
    *,
    stabilize_body_ratio_max: float,
    stabilize_pct_max: float,
) -> bool:
    body, lower, _ = _bar_ratios(r)
    pct = abs(float(getattr(r, "pct_chg", 0.0) or 0.0))
    rng = float(r.high) - float(r.low)
    if rng <= 0:
        return False
    if abs(body) > stabilize_body_ratio_max:
        return False
    if pct > stabilize_pct_max:
        return False
    return True


def _is_long_shadow_stabilize(
    r: KlineRow,
    *,
    long_shadow_ratio_min: float,
    long_body_ratio_max: float,
) -> bool:
    body, lower, _ = _bar_ratios(r)
    if lower < long_shadow_ratio_min:
        return False
    if abs(body) > long_body_ratio_max:
        return False
    return True


def _rally_lows_ok(
    rows: List[KlineRow],
    anchor_i: int,
    idx: int,
    anchor_low: float,
    kw: Dict[str, Any],
) -> bool:
    """锚点日与信号日之间（不含两端）最低价不低于锚点日最低价×anchor_low_floor（默认 0.97，即最多下浮 3%）。"""
    if anchor_low <= 0 or idx <= anchor_i + 1:
        return True
    lows = np.array([float(r.low) for r in rows], dtype=float)
    seg = lows[anchor_i + 1 : idx]
    if len(seg) == 0:
        return True
    floor_mult = float(kw.get("anchor_low_floor", 0.97) or 0.97)
    return float(np.min(seg)) >= anchor_low * floor_mult


def _anchor_near_dist_pct(rows: List[KlineRow], anchor_i: int, idx: int) -> float:
    """信号日价格相对锚点收盘/开盘/最低价的最近距离（%，取绝对值最小）。"""
    ar = rows[anchor_i]
    cur = rows[idx]
    ac = float(ar.close)
    ao = float(ar.open)
    al = float(ar.low)
    cc = float(cur.close)
    cl = float(cur.low)
    if ac <= 0 or ao <= 0 or al <= 0:
        return 999.0
    return min(
        abs((cc - ac) / ac * 100.0),
        abs((cc - ao) / ao * 100.0),
        abs((cl - al) / al * 100.0),
    )


def _anchor_signal_style_at(
    rows: List[KlineRow],
    anchor_i: int,
    idx: int,
    kw: Dict[str, Any],
) -> Optional[str]:
    """判断信号日相对指定锚点是否满足回踩企稳或长下影探底形态。"""
    if anchor_i >= idx:
        return None
    anchor_row = rows[anchor_i]
    anchor_open = float(anchor_row.open)
    anchor_close = float(anchor_row.close)
    anchor_low = float(anchor_row.low)
    if anchor_close <= 0 or anchor_open <= 0 or anchor_low <= 0:
        return None

    highs = np.array([float(r.high) for r in rows], dtype=float)
    lows = np.array([float(r.low) for r in rows], dtype=float)
    post_peak = float(np.max(highs[anchor_i : idx + 1]))
    if post_peak <= 0:
        return None

    cur = rows[idx]
    cur_low = float(cur.low)
    cur_close = float(cur.close)
    pct = float(getattr(cur, "pct_chg", 0.0) or 0.0)
    body, lower_shadow, _ = _bar_ratios(cur)

    pullback_peak_pct = (post_peak - cur_low) / post_peak * 100.0
    if pullback_peak_pct < float(kw["min_pullback_from_peak_pct"]):
        return None

    anchor_dist_close_pct = (cur_close - anchor_close) / anchor_close * 100.0
    anchor_dist_open_pct = (cur_close - anchor_open) / anchor_open * 100.0
    anchor_low_dist_pct = (cur_low - anchor_low) / anchor_low * 100.0

    if not _rally_lows_ok(rows, anchor_i, idx, anchor_low, kw):
        return None

    near_close = abs(anchor_dist_close_pct) <= float(kw["near_anchor_pct"])
    near_open = abs(anchor_dist_open_pct) <= float(kw.get("near_anchor_open_pct", 5.0) or 5.0)
    near_low = abs(anchor_low_dist_pct) <= float(kw.get("near_anchor_low_pct", 3.0) or 3.0)

    def _stab_for_pullback() -> bool:
        ok = _is_near_anchor_stabilize(
            cur,
            stabilize_body_ratio_max=float(kw["stabilize_body_ratio_max"]),
            stabilize_pct_max=float(kw.get("pullback_open_stabilize_pct_max", 5.0) or 5.0),
        )
        if ok:
            return True
        min_shadow = float(kw.get("pullback_open_min_shadow", 0.20) or 0.20)
        return (
            abs(pct) <= float(kw.get("pullback_open_stabilize_pct_max", 5.0) or 5.0)
            and lower_shadow >= min_shadow
        )

    stab_close = _is_near_anchor_stabilize(
        cur,
        stabilize_body_ratio_max=float(kw["stabilize_body_ratio_max"]),
        stabilize_pct_max=float(kw["stabilize_pct_max"]),
    )
    stab_pullback = _stab_for_pullback()

    if (
        (near_close and stab_close)
        or (near_open and not near_close and stab_pullback)
        or (near_low and not near_close and stab_pullback)
    ):
        if near_close:
            return "near_anchor_close"
        if near_low:
            return "near_anchor_low"
        return "near_anchor_open"

    deep_pb = float(kw["deep_pullback_from_peak_pct"])
    deep_floor = float(kw["deep_close_floor"])
    max_above = float(kw.get("long_shadow_max_above_anchor_pct", kw["near_anchor_pct"]))
    if (
        pullback_peak_pct >= deep_pb
        and cur_close >= anchor_low * deep_floor
        and anchor_dist_close_pct <= max_above
        and anchor_dist_open_pct <= max_above
        and _is_long_shadow_stabilize(
            cur,
            long_shadow_ratio_min=float(kw["long_shadow_ratio_min"]),
            long_body_ratio_max=float(kw["long_body_ratio_max"]),
        )
    ):
        return "long_shadow"
    return None


def _pick_anchor(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    kw: Dict[str, Any],
) -> Optional[tuple[int, float, float, float]]:
    """返回 (anchor_idx, anchor_close, post_peak, rally_pct)。"""
    lo = max(0, idx - int(kw["anchor_lookback_max"]))
    hi = idx - int(kw["anchor_lookback_min"])
    if hi < lo:
        return None
    highs = np.array([float(r.high) for r in rows], dtype=float)
    closes = np.array([float(r.close) for r in rows], dtype=float)
    min_rally = float(kw["min_rally_pct"])
    candidates: list[tuple[float, int, float, float, float]] = []
    for j in range(hi, lo - 1, -1):
        if not _is_big_yang_anchor(
            rows,
            j,
            code,
            name,
            big_pct_min=float(kw["big_pct_min"]),
            body_ratio_min=float(kw["body_ratio_min"]),
            anchor_vol_mult=float(kw["anchor_vol_mult"]),
            vol_ma=int(kw["vol_ma"]),
        ):
            continue
        anchor_close = float(closes[j])
        if anchor_close <= 0:
            continue
        post_peak = float(np.max(highs[j : idx + 1]))
        rally_pct = (post_peak - anchor_close) / anchor_close * 100.0
        if rally_pct < min_rally:
            continue
        near_dist = _anchor_near_dist_pct(rows, j, idx)
        candidates.append((near_dist, j, anchor_close, post_peak, rally_pct))

    if not candidates:
        return None
    max_pick_dist = float(kw.get("anchor_pick_max_dist_pct", 15.0) or 15.0)
    in_zone = [c for c in candidates if c[0] <= max_pick_dist]
    if not in_zone:
        return None

    recent_j = max(c[1] for c in in_zone)
    style_recent = _anchor_signal_style_at(rows, recent_j, idx, kw)
    styled: list[tuple[str, tuple[float, int, float, float, float]]] = []
    for c in in_zone:
        st = _anchor_signal_style_at(rows, c[1], idx, kw)
        if st:
            styled.append((st, c))

    if style_recent:
        pool = [x for x in styled if x[1][1] == recent_j]
        if not pool:
            pool = styled
    else:
        near_thr = float(kw["near_anchor_pct"])
        pool = [x for x in styled if x[0].startswith("near_anchor") and x[1][0] <= near_thr]
        if not pool:
            return None

    near_pool = [x for x in pool if x[0].startswith("near_anchor")]
    if near_pool:
        _, c = min(near_pool, key=lambda x: (x[1][0], -x[1][1]))
    else:
        long_pool = [x for x in pool if x[0] == "long_shadow"]
        if not long_pool:
            return None
        _, c = max(long_pool, key=lambda x: x[1][1])
    _, j, ac, peak, rally = c
    return j, ac, peak, rally


def match_mode39_bull_anchor_pullback(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    if _is_st(name or ""):
        return None
    kw = mode39_default_kw()
    kw.update({k: v for k, v in kwargs.items() if k in kw or k.startswith("mode39")})

    vol_ma = int(kw["vol_ma"])
    ma45_period = int(kw.get("ma45_period", 45) or 45)
    ma45_look = int(kw.get("ma45_slope_days", 10) or 10)
    min_len = int(kw["anchor_lookback_max"]) + max(vol_ma, ma45_period + ma45_look) + 5
    if idx < min_len or idx >= len(rows):
        return None

    closes = np.array([float(r.close) for r in rows], dtype=float)
    if kw.get("require_ma45_up", True):
        ok_ma45, ma45_val, ma45_slope = _ma45_up_ok(closes, idx, kw)
        if not ok_ma45:
            return None
    else:
        ma45_val = _ma_at(closes, idx, ma45_period)
        ma45_slope = _ma_slope_pct(closes, idx, ma45_period, ma45_look)

    picked = _pick_anchor(rows, idx, code, name, kw)
    if picked is None:
        return None
    anchor_i, anchor_close, post_peak, rally_pct = picked
    anchor_row = rows[anchor_i]
    anchor_open = float(anchor_row.open)
    anchor_low = float(anchor_row.low)
    anchor_date = str(anchor_row.date)[:10]

    cur = rows[idx]
    cur_low = float(cur.low)
    cur_close = float(cur.close)
    cur_high = float(cur.high)
    pct = float(getattr(cur, "pct_chg", 0.0) or 0.0)
    body, lower_shadow, _ = _bar_ratios(cur)

    if post_peak <= 0:
        return None
    pullback_peak_pct = (post_peak - cur_low) / post_peak * 100.0
    if pullback_peak_pct < float(kw["min_pullback_from_peak_pct"]):
        return None

    lows = np.array([float(r.low) for r in rows], dtype=float)
    vols = np.array([_row_volume(rows, j) for j in range(len(rows))], dtype=float)

    anchor_dist_close_pct = (cur_close - anchor_close) / anchor_close * 100.0
    anchor_dist_open_pct = (
        (cur_close - anchor_open) / anchor_open * 100.0 if anchor_open > 0 else 0.0
    )
    anchor_low_dist_pct = (
        (cur_low - anchor_low) / anchor_low * 100.0 if anchor_low > 0 else 0.0
    )
    anchor_dist_pct = anchor_dist_close_pct
    signal_style: Optional[str] = None

    near_close = abs(anchor_dist_close_pct) <= float(kw["near_anchor_pct"])
    near_open = abs(anchor_dist_open_pct) <= float(kw.get("near_anchor_open_pct", 5.0) or 5.0)
    near_low = abs(anchor_low_dist_pct) <= float(kw.get("near_anchor_low_pct", 3.0) or 3.0)

    if not _rally_lows_ok(rows, anchor_i, idx, anchor_low, kw):
        return None

    def _stab_for_pullback() -> bool:
        ok = _is_near_anchor_stabilize(
            cur,
            stabilize_body_ratio_max=float(kw["stabilize_body_ratio_max"]),
            stabilize_pct_max=float(kw.get("pullback_open_stabilize_pct_max", 5.0) or 5.0),
        )
        if ok:
            return True
        min_shadow = float(kw.get("pullback_open_min_shadow", 0.20) or 0.20)
        return (
            abs(pct) <= float(kw.get("pullback_open_stabilize_pct_max", 5.0) or 5.0)
            and lower_shadow >= min_shadow
        )

    stab_close = _is_near_anchor_stabilize(
        cur,
        stabilize_body_ratio_max=float(kw["stabilize_body_ratio_max"]),
        stabilize_pct_max=float(kw["stabilize_pct_max"]),
    )
    stab_pullback = _stab_for_pullback()

    near_ok = (
        (near_close and stab_close)
        or (near_open and not near_close and stab_pullback)
        or (near_low and not near_close and stab_pullback)
    )

    if near_ok:
        if near_close:
            signal_style = "near_anchor_close"
            anchor_dist_pct = anchor_dist_close_pct
        elif near_low:
            signal_style = "near_anchor_low"
            anchor_dist_pct = anchor_low_dist_pct
        else:
            signal_style = "near_anchor_open"
            anchor_dist_pct = anchor_dist_open_pct
    else:
        deep_pb = float(kw["deep_pullback_from_peak_pct"])
        deep_floor = float(kw["deep_close_floor"])
        max_above = float(kw.get("long_shadow_max_above_anchor_pct", kw["near_anchor_pct"]))
        if (
            pullback_peak_pct >= deep_pb
            and cur_close >= anchor_low * deep_floor
            and anchor_dist_close_pct <= max_above
            and anchor_dist_open_pct <= max_above
            and _is_long_shadow_stabilize(
                cur,
                long_shadow_ratio_min=float(kw["long_shadow_ratio_min"]),
                long_body_ratio_max=float(kw["long_body_ratio_max"]),
            )
        ):
            signal_style = "long_shadow"
        else:
            return None

    if kw.get("require_trough_confirm", True):
        if idx + 1 >= len(rows):
            return None
        if float(lows[idx + 1]) <= cur_low and float(closes[idx + 1]) <= cur_close:
            return None

    peak_i = anchor_i + int(np.argmax([float(r.high) for r in rows[anchor_i : idx + 1]]))
    rally_vol = float(np.mean(vols[max(anchor_i, peak_i - 15) : peak_i + 1]))
    pull_vol = float(np.mean(vols[peak_i : idx + 1]))
    vol_shrink = pull_vol / rally_vol if rally_vol > 0 else 1.0
    if vol_shrink > float(kw["shrink_vol_max"]):
        return None

    exec_i = idx + 1
    exec_date = str(rows[exec_i].date)[:10] if exec_i < len(rows) else ""
    exec_open = float(rows[exec_i].open) if exec_i < len(rows) else 0.0

    return {
        "anchor_date": anchor_date,
        "anchor_open": anchor_open,
        "anchor_close": anchor_close,
        "anchor_low": anchor_low,
        "anchor_dist_close_pct": anchor_dist_close_pct,
        "anchor_dist_open_pct": anchor_dist_open_pct,
        "anchor_low_dist_pct": anchor_low_dist_pct,
        "post_peak": post_peak,
        "peak_date": str(rows[peak_i].date)[:10],
        "rally_pct": rally_pct,
        "pullback_peak_pct": pullback_peak_pct,
        "anchor_dist_pct": anchor_dist_pct,
        "signal_style": signal_style,
        "pct_chg": pct,
        "body_ratio": body,
        "lower_shadow_ratio": lower_shadow,
        "close": cur_close,
        "low": cur_low,
        "high": cur_high,
        "vol_shrink_ratio": vol_shrink,
        "vol_ratio": _vol_ratio_at(rows, idx, vol_ma),
        "signal_date": str(cur.date)[:10],
        "exec_buy_date": exec_date,
        "exec_buy_open": exec_open,
        "buy_trigger_above": cur_high,
        "phase_days": float(idx - anchor_i),
        "ma45": ma45_val,
        "ma45_slope_pct": ma45_slope,
    }


def score_mode39_bull_anchor_pullback(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> int:
    m = match_mode39_bull_anchor_pullback(rows, idx, code, name, **kwargs)
    if not m:
        return 0
    score = 40.0
    score += min(16.0, float(m["rally_pct"]) * 0.5)
    pb = float(m["pullback_peak_pct"])
    if 8.0 <= pb <= 25.0:
        score += 10.0
    elif 5.0 <= pb <= 35.0:
        score += 6.0
    dist = min(
        abs(float(m.get("anchor_dist_close_pct", m["anchor_dist_pct"]) or 0)),
        abs(float(m.get("anchor_dist_open_pct", m["anchor_dist_pct"]) or 0)),
        abs(float(m.get("anchor_low_dist_pct", m["anchor_dist_pct"]) or 0)),
    )
    if dist <= 1.0:
        score += 16.0
    elif dist <= 2.0:
        score += 12.0
    elif dist <= float(kwargs.get("near_anchor_open_pct", 5.0) or 5.0):
        score += 8.0
    if m["signal_style"] == "long_shadow":
        score += min(12.0, float(m["lower_shadow_ratio"]) * 12.0)
    else:
        score += 6.0
    if float(m["vol_shrink_ratio"]) <= float(kwargs.get("shrink_vol_max", 1.15) or 1.15):
        score += 8.0
    if float(m["lower_shadow_ratio"]) >= 0.35:
        score += 6.0
    if float(m["pct_chg"]) > 0:
        score += 4.0
    if float(m.get("ma45_slope_pct", 0) or 0) > 3.0:
        score += 6.0
    elif float(m.get("ma45_slope_pct", 0) or 0) > 0:
        score += 3.0
    return int(min(100, max(0, round(score))))


def mode39_signal_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    m = match_mode39_bull_anchor_pullback(rows, idx, code, name, **kwargs)
    if not m:
        return {}
    return {
        "anchor_date": m["anchor_date"],
        "anchor_open": round(float(m.get("anchor_open", 0) or 0), 4),
        "anchor_close": round(float(m["anchor_close"]), 4),
        "peak_date": m["peak_date"],
        "rally_pct": round(float(m["rally_pct"]), 2),
        "pullback_peak_pct": round(float(m["pullback_peak_pct"]), 2),
        "anchor_dist_pct": round(float(m["anchor_dist_pct"]), 2),
        "anchor_dist_close_pct": round(float(m.get("anchor_dist_close_pct", m["anchor_dist_pct"])), 2),
        "anchor_dist_open_pct": round(float(m.get("anchor_dist_open_pct", 0) or 0), 2),
        "anchor_low_dist_pct": round(float(m.get("anchor_low_dist_pct", 0) or 0), 2),
        "signal_style": m["signal_style"],
        "pct_chg": round(float(m["pct_chg"]), 2),
        "body_ratio": round(float(m["body_ratio"]), 2),
        "lower_shadow_ratio": round(float(m["lower_shadow_ratio"]), 2),
        "vol_shrink_ratio": round(float(m["vol_shrink_ratio"]), 2),
        "low": round(float(m["low"]), 4),
        "signal_date": m["signal_date"],
        "exec_buy_date": m["exec_buy_date"],
        "exec_buy_open": round(float(m["exec_buy_open"]), 4),
        "buy_trigger_above": round(float(m["buy_trigger_above"]), 4),
        "ma45": round(float(m.get("ma45", 0) or 0), 4),
        "ma45_slope_pct": round(float(m.get("ma45_slope_pct", 0) or 0), 2),
    }


def dedupe_mode39_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[tuple, Dict[str, Any]] = {}
    for h in hits:
        key = (
            h.get("code", ""),
            h.get("anchor_date", ""),
            h.get("signal_style", ""),
        )
        low = float(h.get("low", 0) or 0)
        prev = best.get(key)
        if prev is None or low < float(prev.get("low", 0) or 0):
            best[key] = h
    return sorted(best.values(), key=lambda x: (x.get("date", ""), x.get("code", "")))
