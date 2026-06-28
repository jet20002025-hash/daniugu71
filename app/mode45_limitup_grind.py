"""mode45 涨停放量新高后缓升：启动日涨停/强阳放量创新高，随后数日横盘缓升、浅回调。

样本：国联股份 603613
  - 2025-08-21 强阳放量突破（+5.2%，量约 3×均量，创阶段新高）
  - 8/22～8/25 缓升不 deep 回调（收盘贴近前高，波动小）
  - 2025-08-27 再涨停（可选后续 leg，本模式买点在缓升段）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from app.scanner import KlineRow, _is_st, _limit_rate, _limit_up_day, _vol_ratio_at

MODE45_ID = "mode45"
MODE45_FULL_NAME = "涨停新高后缓升"
MODE45_DISPLAY_NAME = f"{MODE45_ID}（{MODE45_FULL_NAME}）"
MODE45_ONE_LINE = "涨停或强阳放量创新高后，数日横盘缓升、浅回调不破"


def mode45_default_kw() -> Dict[str, Any]:
    return dict(
        high_lookback=60,
        launch_vol_mult=1.8,
        surge_pct_min_main=4.5,
        surge_pct_min_growth=7.0,
        allow_limit_up_launch=True,
        allow_surge_launch=True,
        min_grind_days=2,
        max_grind_days=10,
        max_grind_pullback_pct=4.5,
        max_grind_low_pullback_pct=6.0,
        max_avg_abs_pct=2.5,
        max_grind_day_pct=8.0,
        min_grind_rise_pct=-2.0,
        max_grind_rise_pct=12.0,
        signal_near_high_pct=2.5,
        signal_not_limit_up=True,
        require_close_above_launch=True,
        launch_close_tol_pct=2.0,
        vol_ma=20,
        min_score=60,
    )


def mode45_kw_from_scan_config(cfg: Any) -> Dict[str, Any]:
    base = mode45_default_kw()
    for k in base:
        ck = f"mode45_{k}"
        if hasattr(cfg, ck):
            base[k] = getattr(cfg, ck)
    if hasattr(cfg, "mode45_min_score"):
        base["min_score"] = int(getattr(cfg, "mode45_min_score", 60))
    return base


def _row_volume(rows: List[KlineRow], idx: int) -> float:
    r = rows[idx]
    return float(getattr(r, "volume", 0) or getattr(r, "vol", 0) or 0)


def _day_pct(rows: List[KlineRow], idx: int) -> float:
    raw = getattr(rows[idx], "pct_chg", None)
    if raw is not None and float(raw) != 0:
        return float(raw)
    if idx < 1:
        return 0.0
    prev = float(rows[idx - 1].close)
    if prev <= 0:
        return 0.0
    return (float(rows[idx].close) - prev) / prev * 100.0


def _is_growth_board(code: str) -> bool:
    c = str(code or "")
    return c.startswith(("30", "301", "688"))


def _surge_pct_min(code: str, kw: Dict[str, Any]) -> float:
    if _is_growth_board(code):
        return float(kw.get("surge_pct_min_growth", 7.0) or 7.0)
    return float(kw.get("surge_pct_min_main", 4.5) or 4.5)


def _is_launch_day(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    kw: Dict[str, Any],
) -> bool:
    if bool(kw.get("allow_limit_up_launch", True)) and _limit_up_day(rows, idx, code, name):
        return True
    if not bool(kw.get("allow_surge_launch", True)):
        return False
    pct = _day_pct(rows, idx)
    return pct >= _surge_pct_min(code, kw)


def _is_new_high(
    highs: np.ndarray,
    idx: int,
    lookback: int,
    tol_pct: float = 0.2,
) -> bool:
    if idx <= 0:
        return False
    lo = max(0, idx - lookback)
    prev_max = float(np.max(highs[lo:idx]))
    if prev_max <= 0:
        return False
    return float(highs[idx]) >= prev_max * (1.0 - tol_pct / 100.0)


def _find_launch(
    rows: List[KlineRow],
    signal_idx: int,
    code: str,
    name: str,
    kw: Dict[str, Any],
) -> Optional[int]:
    min_g = int(kw.get("min_grind_days", 2) or 2)
    max_g = int(kw.get("max_grind_days", 10) or 10)
    vol_ma = int(kw.get("vol_ma", 20) or 20)
    vol_mult = float(kw.get("launch_vol_mult", 1.8) or 1.8)
    high_lb = int(kw.get("high_lookback", 60) or 60)
    highs = np.array([float(r.high) for r in rows], dtype=float)

    lo = max(high_lb, signal_idx - max_g)
    hi = signal_idx - min_g
    best: Optional[int] = None
    best_key = (-1.0, -1.0)

    for l in range(hi, lo - 1, -1):
        if not _is_launch_day(rows, l, code, name, kw):
            continue
        vr = _vol_ratio_at(rows, l, vol_ma)
        if vr < vol_mult:
            continue
        if not _is_new_high(highs, l, high_lb):
            continue
        pct = _day_pct(rows, l)
        key = (vr, pct)
        if best is None or key > best_key:
            best = l
            best_key = key
    return best


def _grind_ok(
    rows: List[KlineRow],
    launch_idx: int,
    signal_idx: int,
    kw: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    if signal_idx <= launch_idx:
        return None
    closes = np.array([float(r.close) for r in rows], dtype=float)
    lows = np.array([float(r.low) for r in rows], dtype=float)
    launch_close = closes[launch_idx]

    grind_slice = slice(launch_idx + 1, signal_idx + 1)
    g_closes = closes[grind_slice]
    g_lows = lows[grind_slice]
    if len(g_closes) < int(kw.get("min_grind_days", 2) or 2):
        return None

    peak_c = float(np.max(g_closes))
    min_c = float(np.min(g_closes))
    min_l = float(np.min(g_lows))
    if peak_c <= 0:
        return None

    max_pb = float(kw.get("max_grind_pullback_pct", 4.5) or 4.5)
    close_pb = (peak_c - min_c) / peak_c * 100.0
    if close_pb > max_pb:
        return None

    max_low_pb = float(kw.get("max_grind_low_pullback_pct", 6.0) or 6.0)
    low_pb = (peak_c - min_l) / peak_c * 100.0
    if low_pb > max_low_pb:
        return None

    sig_close = closes[signal_idx]
    rise_pct = (sig_close / launch_close - 1.0) * 100.0 if launch_close > 0 else 0.0
    if rise_pct < float(kw.get("min_grind_rise_pct", -2.0) or -2.0):
        return None
    if rise_pct > float(kw.get("max_grind_rise_pct", 12.0) or 12.0):
        return None

    if bool(kw.get("require_close_above_launch", True)):
        tol = float(kw.get("launch_close_tol_pct", 2.0) or 2.0)
        if sig_close < launch_close * (1.0 - tol / 100.0):
            return None

    abs_pcts: List[float] = []
    max_day = float(kw.get("max_grind_day_pct", 8.0) or 8.0)
    for j in range(launch_idx + 1, signal_idx + 1):
        p = abs(_day_pct(rows, j))
        if p > max_day:
            return None
        abs_pcts.append(p)
    avg_abs = float(np.mean(abs_pcts)) if abs_pcts else 0.0
    if avg_abs > float(kw.get("max_avg_abs_pct", 2.5) or 2.5):
        return None

    near_pct = float(kw.get("signal_near_high_pct", 2.5) or 2.5)
    if sig_close < peak_c * (1.0 - near_pct / 100.0):
        return None

    return {
        "grind_days": float(signal_idx - launch_idx),
        "grind_close_pullback_pct": close_pb,
        "grind_low_pullback_pct": low_pb,
        "grind_rise_pct": rise_pct,
        "grind_avg_abs_pct": avg_abs,
        "grind_peak_close": peak_c,
    }


def match_mode45_limitup_grind(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    if _is_st(name or ""):
        return None
    kw = mode45_default_kw()
    kw.update({k: v for k, v in kwargs.items() if k in kw or k.startswith("mode45")})

    high_lb = int(kw.get("high_lookback", 60) or 60)
    min_len = high_lb + int(kw.get("max_grind_days", 10) or 10) + 5
    if idx < min_len or idx >= len(rows):
        return None

    if bool(kw.get("signal_not_limit_up", True)) and _limit_up_day(rows, idx, code, name):
        return None

    launch_i = _find_launch(rows, idx, code, name, kw)
    if launch_i is None:
        return None

    grind = _grind_ok(rows, launch_i, idx, kw)
    if grind is None:
        return None

    launch_row = rows[launch_i]
    cur = rows[idx]
    vol_ma = int(kw.get("vol_ma", 20) or 20)
    launch_vr = _vol_ratio_at(rows, launch_i, vol_ma)
    launch_pct = _day_pct(rows, launch_i)

    exec_i = idx + 1
    exec_date = ""
    exec_open = 0.0
    if exec_i < len(rows):
        exec_date = str(rows[exec_i].date)[:10]
        exec_open = float(rows[exec_i].open)

    return {
        "signal_date": str(cur.date)[:10],
        "launch_date": str(launch_row.date)[:10],
        "launch_close": float(launch_row.close),
        "launch_pct": launch_pct,
        "launch_vol_ratio": launch_vr,
        "launch_limit_up": _limit_up_day(rows, launch_i, code, name),
        "close": float(cur.close),
        "pct_chg": _day_pct(rows, idx),
        "vol_ratio": _vol_ratio_at(rows, idx, vol_ma),
        **grind,
        "exec_buy_date": exec_date,
        "exec_buy_open": exec_open,
        "buy_mode": "next_open",
    }


def score_mode45_limitup_grind(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> int:
    m = match_mode45_limitup_grind(rows, idx, code, name, **kwargs)
    if not m:
        return 0
    kw = mode45_default_kw()
    kw.update({k: v for k, v in kwargs.items() if k in kw or k.startswith("mode45")})
    min_score = int(kw.get("min_score", 60) or 60)

    score = 62.0
    if m.get("launch_limit_up"):
        score += 8.0
    lvr = float(m.get("launch_vol_ratio") or 0.0)
    score += min(12.0, max(0.0, (lvr - 1.5) * 6.0))

    close_pb = float(m.get("grind_close_pullback_pct") or 0.0)
    score += max(0.0, min(10.0, (4.5 - close_pb) * 2.0))

    avg_abs = float(m.get("grind_avg_abs_pct") or 0.0)
    score += max(0.0, min(8.0, (2.5 - avg_abs) * 3.0))

    rise = float(m.get("grind_rise_pct") or 0.0)
    if 0.0 <= rise <= 6.0:
        score += 5.0
    elif -1.0 <= rise < 0.0:
        score += 2.0

    gdays = float(m.get("grind_days") or 0.0)
    if 3.0 <= gdays <= 6.0:
        score += 4.0
    elif 2.0 <= gdays < 3.0:
        score += 2.0

    return int(max(min_score, min(100, round(score))))


def mode45_signal_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    m = match_mode45_limitup_grind(rows, idx, code, name, **kwargs)
    return dict(m) if m else {}


def dedupe_mode45_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[tuple, Dict[str, Any]] = {}
    for h in hits:
        key = (str(h.get("code", "")).zfill(6), str(h.get("signal_date", ""))[:10])
        prev = best.get(key)
        if prev is None or int(h.get("score", 0) or 0) > int(prev.get("score", 0) or 0):
            best[key] = h
    out = list(best.values())
    out.sort(key=lambda x: (str(x.get("signal_date", "")), -int(x.get("score", 0) or 0)))
    return out
