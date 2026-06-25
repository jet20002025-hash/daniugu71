"""mode40 新高回调踩60线回升：创阶段新高后连续回调7～10日，触及MA60止跌回升。

样本：中科飞测 688361
  - 2026-05-25 创60日新高（高 268.07）
  - 回调约10个交易日，2026-06-08 低点踩 MA60（低 188 / MA60≈186）
  - 2026-06-09 阳线回升 → 买点次日开盘
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from app.scanner import KlineRow, _is_st, _vol_ratio_at

MODE40_ID = "mode40"
MODE40_FULL_NAME = "新高回调踩60线回升"
MODE40_DISPLAY_NAME = f"{MODE40_ID}（{MODE40_FULL_NAME}）"
MODE40_ONE_LINE = (
    "阶段新高后回调7～10日，低点触及MA60（±容差）止跌，信号日阳线回升；"
    "均线与前高/回踩按除权跳空做前复权修正；次日开盘买"
)


def mode40_default_kw() -> Dict[str, Any]:
    return dict(
        peak_lookback=60,
        pullback_days_min=7,
        pullback_days_max=10,
        rebound_window=3,
        ma_period=60,
        ma_touch_pct=5.0,
        min_pullback_from_peak_pct=8.0,
        max_pullback_from_peak_pct=40.0,
        min_down_days_ratio=0.55,
        require_rebound_yang=True,
        require_close_above_ma=False,
        require_ma60_up=False,
        ma60_slope_days=10,
        min_ma60_slope_pct=0.0,
        vol_ma=20,
        min_score=60,
        ex_right_adj=True,
        ex_right_gap_pct=12.0,
    )


def mode40_kw_from_scan_config(cfg: Any) -> Dict[str, Any]:
    base = mode40_default_kw()
    for k in base:
        ck = f"mode40_{k}"
        if hasattr(cfg, ck):
            base[k] = getattr(cfg, ck)
    if hasattr(cfg, "mode40_min_score"):
        base["min_score"] = int(getattr(cfg, "mode40_min_score", 60))
    return base


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


def _down_days_ratio(closes: np.ndarray, start: int, end: int) -> float:
    """start..end 区间内收盘下跌日占比（不含 start）。"""
    if end <= start:
        return 0.0
    down = 0
    total = 0
    for j in range(start + 1, end + 1):
        if closes[j] < closes[j - 1]:
            down += 1
        total += 1
    return down / total if total else 0.0


def _price_arrays(
    rows: List[KlineRow],
    kw: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from app.kline_adjust import forward_adj_ohlc_rows

    if not kw.get("ex_right_adj", True):
        highs = np.array([float(r.high) for r in rows], dtype=float)
        lows = np.array([float(r.low) for r in rows], dtype=float)
        closes = np.array([float(r.close) for r in rows], dtype=float)
        return highs, lows, closes
    gap = float(kw.get("ex_right_gap_pct", 12.0) or 12.0) / 100.0
    _, adj_h, adj_l, adj_c, _ = forward_adj_ohlc_rows(rows, gap_pct=gap)
    return adj_h, adj_l, adj_c


def match_mode40_high_pullback_ma60_rebound(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    if _is_st(name or ""):
        return None
    kw = mode40_default_kw()
    kw.update({k: v for k, v in kwargs.items() if k in kw or k.startswith("mode40")})

    peak_lb = int(kw["peak_lookback"])
    pb_min = int(kw["pullback_days_min"])
    pb_max = int(kw["pullback_days_max"])
    rebound_win = int(kw["rebound_window"])
    ma_n = int(kw["ma_period"])
    touch_pct = float(kw["ma_touch_pct"])
    min_pb = float(kw["min_pullback_from_peak_pct"])
    max_pb = float(kw["max_pullback_from_peak_pct"])
    min_down_ratio = float(kw["min_down_days_ratio"])

    min_len = peak_lb + ma_n + pb_max + rebound_win + 5
    if idx < min_len or idx >= len(rows):
        return None

    adj_highs, adj_lows, adj_closes = _price_arrays(rows, kw)
    closes = adj_closes
    cur = rows[idx]
    cur_open = float(cur.open)
    cur_close = float(cur.close)
    cur_high = float(cur.high)
    cur_low = float(cur.low)
    adj_close = float(closes[idx])
    pct = float(getattr(cur, "pct_chg", 0.0) or 0.0)

    if kw.get("require_rebound_yang", True) and cur_close <= cur_open:
        return None

    ma60_now = _ma_at(closes, idx, ma_n)
    if kw.get("require_close_above_ma", False) and (
        np.isnan(ma60_now) or adj_close < ma60_now
    ):
        return None

    ma60_slope = _ma_slope_pct(closes, idx, ma_n, int(kw.get("ma60_slope_days", 10) or 10))
    if kw.get("require_ma60_up", False):
        if ma60_slope <= float(kw.get("min_ma60_slope_pct", 0.0) or 0.0):
            return None

    best: Optional[Dict[str, Any]] = None
    p_start = max(peak_lb, idx - pb_max - rebound_win)
    p_end = idx - pb_min

    for p in range(p_start, p_end + 1):
        peak_high = float(adj_highs[p])
        peak_high_raw = float(rows[p].high)
        prev_max = max(float(adj_highs[j]) for j in range(p - peak_lb, p))
        if peak_high < prev_max * 0.998:
            continue

        trough_i: Optional[int] = None
        trough_low = float("inf")
        trough_low_raw = float("inf")
        trough_ma_dist = 999.0

        for t in range(p + pb_min, min(p + pb_max + 1, idx)):
            low_t = float(adj_lows[t])
            low_t_raw = float(rows[t].low)
            ma_t = _ma_at(closes, t, ma_n)
            if np.isnan(ma_t) or ma_t <= 0:
                continue
            dist = abs(low_t - ma_t) / ma_t * 100.0
            if dist > touch_pct:
                continue
            pb_pct = (peak_high - low_t) / peak_high * 100.0 if peak_high > 0 else 0.0
            if pb_pct < min_pb or pb_pct > max_pb:
                continue
            if _down_days_ratio(closes, p, t) < min_down_ratio:
                continue
            if dist < trough_ma_dist or (dist == trough_ma_dist and low_t < trough_low):
                trough_ma_dist = dist
                trough_low = low_t
                trough_low_raw = low_t_raw
                trough_i = t

        if trough_i is None:
            continue
        t = trough_i
        if idx <= t or idx > t + rebound_win:
            continue
        if adj_close <= float(closes[t]):
            continue

        trough_close = float(closes[t])
        rebound_pct = (adj_close / trough_close - 1.0) * 100.0 if trough_close > 0 else 0.0
        pullback_pct = (peak_high - trough_low) / peak_high * 100.0 if peak_high > 0 else 0.0
        days_peak_to_trough = t - p

        cand = {
            "peak_date": str(rows[p].date)[:10],
            "peak_high": peak_high_raw,
            "trough_date": str(rows[t].date)[:10],
            "trough_low": trough_low_raw,
            "trough_close": float(rows[t].close),
            "ma60_at_trough": _ma_at(closes, t, ma_n),
            "ma_touch_dist_pct": trough_ma_dist,
            "pullback_days": days_peak_to_trough,
            "pullback_pct": pullback_pct,
            "rebound_pct": rebound_pct,
            "down_days_ratio": _down_days_ratio(closes, p, t),
            "pct_chg": pct,
            "close": cur_close,
            "low": cur_low,
            "high": cur_high,
            "signal_date": str(cur.date)[:10],
            "ma60": ma60_now,
            "ma60_slope_pct": ma60_slope,
            "vol_ratio": _vol_ratio_at(rows, idx, int(kw["vol_ma"])),
        }
        score_key = (-trough_ma_dist, -rebound_pct, -days_peak_to_trough)
        if best is None or score_key < (
            -best["ma_touch_dist_pct"],
            -best["rebound_pct"],
            -best["pullback_days"],
        ):
            best = cand

    if best is None:
        return None

    exec_i = idx + 1
    exec_date = ""
    exec_open = 0.0
    if exec_i < len(rows):
        exec_date = str(rows[exec_i].date)[:10]
        exec_open = float(rows[exec_i].open)

    best["exec_buy_date"] = exec_date
    best["exec_buy_open"] = exec_open
    best["buy_trigger_above"] = cur_high
    return best


def score_mode40_high_pullback_ma60_rebound(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> int:
    m = match_mode40_high_pullback_ma60_rebound(rows, idx, code, name, **kwargs)
    if not m:
        return 0
    score = 42.0
    dist = float(m["ma_touch_dist_pct"])
    if dist <= 1.0:
        score += 18.0
    elif dist <= 2.5:
        score += 14.0
    elif dist <= float(kwargs.get("ma_touch_pct", 5.0) or 5.0):
        score += 8.0
    pb = float(m["pullback_pct"])
    if 12.0 <= pb <= 28.0:
        score += 12.0
    elif 8.0 <= pb <= 35.0:
        score += 8.0
    rb = float(m["rebound_pct"])
    score += min(12.0, rb * 2.0)
    if float(m["down_days_ratio"]) >= 0.65:
        score += 6.0
    days = int(m["pullback_days"])
    if int(kwargs.get("pullback_days_min", 7)) <= days <= int(kwargs.get("pullback_days_max", 10)):
        score += 6.0
    if float(m["pct_chg"]) > 0:
        score += 4.0
    if float(m.get("ma60_slope_pct", 0) or 0) > 2.0:
        score += 4.0
    vr = float(m.get("vol_ratio", 0) or 0)
    if 1.0 <= vr <= 2.5:
        score += 4.0
    return int(min(100, max(0, round(score))))


def mode40_signal_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    m = match_mode40_high_pullback_ma60_rebound(rows, idx, code, name, **kwargs)
    if not m:
        return {}
    return {
        "peak_date": m["peak_date"],
        "peak_high": round(float(m["peak_high"]), 4),
        "trough_date": m["trough_date"],
        "trough_low": round(float(m["trough_low"]), 4),
        "ma_touch_dist_pct": round(float(m["ma_touch_dist_pct"]), 2),
        "pullback_days": int(m["pullback_days"]),
        "pullback_pct": round(float(m["pullback_pct"]), 2),
        "rebound_pct": round(float(m["rebound_pct"]), 2),
        "pct_chg": round(float(m["pct_chg"]), 2),
        "signal_date": m["signal_date"],
        "exec_buy_date": m.get("exec_buy_date", ""),
        "exec_buy_open": round(float(m.get("exec_buy_open", 0) or 0), 4),
        "buy_trigger_above": round(float(m.get("buy_trigger_above", 0) or 0), 4),
        "ma60": round(float(m.get("ma60", 0) or 0), 4),
        "ma60_slope_pct": round(float(m.get("ma60_slope_pct", 0) or 0), 2),
    }


def dedupe_mode40_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[tuple, Dict[str, Any]] = {}
    for h in hits:
        key = (
            h.get("code", ""),
            h.get("peak_date", ""),
            h.get("trough_date", ""),
        )
        score = int(h.get("score", 0) or 0)
        prev = best.get(key)
        if prev is None or score > int(prev.get("score", 0) or 0):
            best[key] = h
    return sorted(best.values(), key=lambda x: (x.get("date", ""), x.get("code", "")))
