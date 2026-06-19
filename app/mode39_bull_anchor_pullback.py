"""mode39 大阳锚点回踩再升：放量大阳线作锚，回踩锚点收盘附近企稳后再攀升。

样本：宇瞳光学 300790
  - 2026-04-08 放量大阳锚点（收盘 28.42）
  - 2026-04-29 贴近锚点小十字企稳 → 买点 4/30 开盘
  - 2026-06-10 长下影十字探底 → 买点 6/11 开盘

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
    "放量大阳作锚；拉升后回踩锚点收盘企稳（小阳/十字）或长下影探底；"
    "信号日确认后次日开盘买"
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
        anchor_close_floor=0.97,
        stabilize_body_ratio_max=0.45,
        stabilize_pct_max=3.5,
        deep_pullback_from_peak_pct=15.0,
        deep_close_floor=0.72,
        long_shadow_ratio_min=0.55,
        long_body_ratio_max=0.30,
        require_trough_confirm=True,
        shrink_vol_max=1.15,
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
    best: Optional[tuple[float, float, int, float, float, float]] = None
    min_rally = float(kw["min_rally_pct"])
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
        anchor_pct = float(getattr(rows[j], "pct_chg", 0.0) or 0.0)
        dist = abs(float(closes[idx]) - anchor_close) / anchor_close
        key = (-anchor_pct, dist, -j)
        if best is None or key < best[:3]:
            best = (key[0], key[1], j, anchor_close, post_peak, rally_pct)
    if best is None:
        return None
    _, _, j, ac, peak, rally = best
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
    min_len = int(kw["anchor_lookback_max"]) + vol_ma + 5
    if idx < min_len or idx >= len(rows):
        return None

    picked = _pick_anchor(rows, idx, code, name, kw)
    if picked is None:
        return None
    anchor_i, anchor_close, post_peak, rally_pct = picked
    anchor_low = float(rows[anchor_i].low)
    anchor_date = str(rows[anchor_i].date)[:10]

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

    closes = np.array([float(r.close) for r in rows], dtype=float)
    lows = np.array([float(r.low) for r in rows], dtype=float)
    vols = np.array([_row_volume(rows, j) for j in range(len(rows))], dtype=float)

    anchor_dist_pct = (cur_close - anchor_close) / anchor_close * 100.0
    signal_style: Optional[str] = None

    near_ok = (
        abs(anchor_dist_pct) <= float(kw["near_anchor_pct"])
        and _is_near_anchor_stabilize(
            cur,
            stabilize_body_ratio_max=float(kw["stabilize_body_ratio_max"]),
            stabilize_pct_max=float(kw["stabilize_pct_max"]),
        )
    )
    if near_ok:
        seg = closes[anchor_i + 1 : idx]
        if len(seg) > 0 and float(np.min(seg)) < anchor_close * float(kw["anchor_close_floor"]):
            near_ok = False

    if near_ok:
        signal_style = "near_anchor"
    else:
        deep_pb = float(kw["deep_pullback_from_peak_pct"])
        deep_floor = float(kw["deep_close_floor"])
        if (
            pullback_peak_pct >= deep_pb
            and cur_close >= anchor_close * deep_floor
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
        "anchor_close": anchor_close,
        "anchor_low": anchor_low,
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
    dist = abs(float(m["anchor_dist_pct"]))
    if dist <= 1.0:
        score += 16.0
    elif dist <= 2.0:
        score += 12.0
    elif dist <= float(kwargs.get("near_anchor_pct", 3.0) or 3.0):
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
        "anchor_close": round(float(m["anchor_close"]), 4),
        "peak_date": m["peak_date"],
        "rally_pct": round(float(m["rally_pct"]), 2),
        "pullback_peak_pct": round(float(m["pullback_peak_pct"]), 2),
        "anchor_dist_pct": round(float(m["anchor_dist_pct"]), 2),
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
