"""mode46 前高附近二次攻击：前期高点回撤后再度上攻至前高附近（尚未有效突破）。

样本：
  - 深科技 000021 @ 2026-06-18（前高 5/27 高 47.03，回撤至 6/8 低 34.85，再攻前高）
  - 中科飞测 688361 @ 2026-06-18（前高 5/25 高 268.07，回撤至 6/8 低 188，再攻前高）

与 mode35 区别：mode35 要求放量突破前高；本模式为「贴近前高、收盘未有效突破」的二次攻击观察/试仓点。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from app.scanner import KlineRow, _is_st, _limit_rate

MODE46_ID = "mode46"
MODE46_FULL_NAME = "前高附近二次攻击"
MODE46_DISPLAY_NAME = f"{MODE46_ID}（{MODE46_FULL_NAME}）"
MODE46_ONE_LINE = (
    "阶段前高回撤后再上攻，信号日最高贴近前高、收盘未突破；"
    "收阳且在 MA20/MA60 之上；次日开盘或突破前高试仓"
)


def mode46_default_kw() -> Dict[str, Any]:
    return dict(
        peak_lookback=100,
        min_peak_age=8,
        max_peak_age=80,
        min_pullback_pct=12.0,
        max_pullback_pct=40.0,
        min_rebound_from_trough_pct=12.0,
        max_high_dist_pct=6.0,
        max_close_dist_pct=8.0,
        min_high_dist_pct=0.0,
        leak_tol_pct=1.0,
        require_yang=True,
        require_above_ma20=True,
        require_above_ma60=True,
        min_body_ratio=0.15,
        max_pct_chg=9.5,
        max_pct_chg_main=7.0,
        vol_ma=20,
        min_vol_ratio=0.0,
        max_vol_ratio=3.5,
        min_score=60,
    )


def mode46_kw_from_scan_config(cfg: Any) -> Dict[str, Any]:
    base = mode46_default_kw()
    for k in base:
        ck = f"mode46_{k}"
        if hasattr(cfg, ck):
            base[k] = getattr(cfg, ck)
    if hasattr(cfg, "mode46_min_score"):
        base["min_score"] = int(getattr(cfg, "mode46_min_score", 60))
    return base


def _robust_vol_ratio(rows: List[KlineRow], idx: int, n: int = 20) -> float:
    """量比：剔除窗口内极端放量日后计算，避免腾讯缓存偶发单位错误拉歪均值。"""
    if idx < n:
        return 0.0
    vols = [float(getattr(rows[j], "volume", 0) or 0) for j in range(idx - n, idx)]
    if not vols:
        return 0.0
    med = float(np.median(vols))
    if med <= 0:
        return 0.0
    cap = med * 5.0
    trimmed = [min(v, cap) for v in vols]
    base = float(np.mean(trimmed))
    cur = float(getattr(rows[idx], "volume", 0) or 0)
    return cur / base if base > 0 else 0.0


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


def _ma_at(closes: np.ndarray, idx: int, n: int) -> float:
    if idx < n - 1:
        return float("nan")
    return float(np.mean(closes[idx - n + 1 : idx + 1]))


def _find_prior_peak(
    rows: List[KlineRow],
    idx: int,
    kw: Dict[str, Any],
    highs: np.ndarray,
    lows: np.ndarray,
) -> Optional[Dict[str, Any]]:
    lookback = int(kw.get("peak_lookback", 100) or 100)
    min_age = int(kw.get("min_peak_age", 8) or 8)
    max_age = int(kw.get("max_peak_age", 80) or 80)
    if idx < lookback + min_age + 5:
        return None

    end = idx - min_age
    start = max(20, idx - lookback)
    if end <= start:
        return None

    seg = highs[start:end + 1]
    peak_rel = int(np.argmax(seg))
    peak_i = start + peak_rel
    prior_high = float(highs[peak_i])
    if prior_high <= 0:
        return None

    peak_age = idx - peak_i
    if peak_age < min_age or peak_age > max_age:
        return None

    trough_seg = lows[peak_i:idx + 1]
    trough_rel = int(np.argmin(trough_seg))
    trough_i = peak_i + trough_rel
    trough_low = float(lows[trough_i])
    if trough_i <= peak_i or trough_i >= idx:
        return None

    pullback_pct = (prior_high - trough_low) / prior_high * 100.0
    min_pb = float(kw.get("min_pullback_pct", 12.0) or 12.0)
    max_pb = float(kw.get("max_pullback_pct", 40.0) or 40.0)
    if pullback_pct < min_pb or pullback_pct > max_pb:
        return None

    leak_tol = float(kw.get("leak_tol_pct", 1.0) or 1.0) / 100.0
    for j in range(peak_i + 1, idx):
        if highs[j] > prior_high * (1.0 + leak_tol):
            return None

    rebound_pct = (
        (float(rows[idx].close) - trough_low) / trough_low * 100.0
        if trough_low > 0
        else 0.0
    )
    min_rb = float(kw.get("min_rebound_from_trough_pct", 12.0) or 12.0)
    if rebound_pct < min_rb:
        return None

    return {
        "peak_date": str(rows[peak_i].date)[:10],
        "peak_date_idx": float(peak_i),
        "prior_high": prior_high,
        "trough_date": str(rows[trough_i].date)[:10],
        "trough_low": trough_low,
        "pullback_pct": pullback_pct,
        "rebound_from_trough_pct": rebound_pct,
        "peak_age_days": float(peak_age),
    }


def match_mode46_prior_high_retest(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    if _is_st(name or ""):
        return None
    kw = mode46_default_kw()
    kw.update({k: v for k, v in kwargs.items() if k in kw or k.startswith("mode46")})

    lookback = int(kw.get("peak_lookback", 100) or 100)
    if idx < max(lookback, 80) or idx >= len(rows):
        return None

    closes = np.array([float(r.close) for r in rows], dtype=float)
    highs = np.array([float(r.high) for r in rows], dtype=float)
    lows = np.array([float(r.low) for r in rows], dtype=float)

    peak_info = _find_prior_peak(rows, idx, kw, highs, lows)
    if peak_info is None:
        return None

    prior_high = float(peak_info["prior_high"])
    r = rows[idx]
    o, c, h, l = float(r.open), float(r.close), float(r.high), float(r.low)
    pct = _day_pct(rows, idx)

    if kw.get("require_yang", True) and c <= o:
        return None

    max_pct = float(kw.get("max_pct_chg_main", 7.0) or 7.0)
    if _limit_rate(code, name) >= 0.15:
        max_pct = float(kw.get("max_pct_chg", 9.5) or 9.5)
    if pct > max_pct or pct <= 0:
        return None

    high_dist_pct = (prior_high - h) / prior_high * 100.0 if prior_high > 0 else 999.0
    close_dist_pct = (prior_high - c) / prior_high * 100.0 if prior_high > 0 else 999.0

    max_hd = float(kw.get("max_high_dist_pct", 6.0) or 6.0)
    max_cd = float(kw.get("max_close_dist_pct", 8.0) or 8.0)
    min_hd = float(kw.get("min_high_dist_pct", 0.0) or 0.0)

    if high_dist_pct > max_hd or high_dist_pct < min_hd:
        return None
    if close_dist_pct > max_cd or c >= prior_high:
        return None

    ma20 = _ma_at(closes, idx, 20)
    ma60 = _ma_at(closes, idx, 60)
    if kw.get("require_above_ma20", True) and (np.isnan(ma20) or c < ma20):
        return None
    if kw.get("require_above_ma60", True) and (np.isnan(ma60) or c < ma60):
        return None

    rng = h - l
    body_ratio = (c - o) / rng if rng > 0 else 0.0
    if body_ratio < float(kw.get("min_body_ratio", 0.15) or 0.15):
        return None

    vol_ratio = _robust_vol_ratio(rows, idx, int(kw.get("vol_ma", 20) or 20))
    if vol_ratio > float(kw.get("max_vol_ratio", 3.5) or 3.5):
        return None

    exec_i = idx + 1
    exec_date = ""
    exec_open = 0.0
    if exec_i < len(rows):
        exec_date = str(rows[exec_i].date)[:10]
        exec_open = float(rows[exec_i].open)

    return {
        **peak_info,
        "signal_date": str(r.date)[:10],
        "close": c,
        "high": h,
        "low": l,
        "pct_chg": pct,
        "high_dist_pct": high_dist_pct,
        "close_dist_pct": close_dist_pct,
        "vol_ratio": vol_ratio,
        "body_ratio": body_ratio,
        "ma20": float(ma20) if not np.isnan(ma20) else 0.0,
        "ma60": float(ma60) if not np.isnan(ma60) else 0.0,
        "exec_buy_date": exec_date,
        "exec_buy_open": exec_open,
        "buy_trigger_above": prior_high,
    }


def score_mode46_prior_high_retest(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> int:
    m = match_mode46_prior_high_retest(rows, idx, code, name, **kwargs)
    if not m:
        return 0

    score = 48.0
    hd = float(m["high_dist_pct"])
    cd = float(m["close_dist_pct"])
    if hd <= 2.0:
        score += 18.0
    elif hd <= 4.0:
        score += 14.0
    elif hd <= 6.0:
        score += 10.0
    if cd <= 4.0:
        score += 10.0
    elif cd <= 6.0:
        score += 6.0

    pb = float(m["pullback_pct"])
    if 18.0 <= pb <= 32.0:
        score += 10.0
    elif 12.0 <= pb <= 38.0:
        score += 6.0

    rb = float(m["rebound_from_trough_pct"])
    score += min(10.0, rb * 0.35)

    br = float(m.get("body_ratio", 0) or 0)
    if 0.25 <= br <= 0.75:
        score += 4.0

    vr = float(m.get("vol_ratio", 0) or 0)
    if 0.5 <= vr <= 1.5:
        score += 4.0

    age = float(m.get("peak_age_days", 0) or 0)
    if 12.0 <= age <= 25.0:
        score += 4.0

    # 贴近前高但收盘略远扣分
    if cd > hd + 3.0:
        score -= 6.0

    return int(min(100, max(0, round(score))))


def mode46_signal_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    m = match_mode46_prior_high_retest(rows, idx, code, name, **kwargs)
    if not m:
        return {}
    score = score_mode46_prior_high_retest(rows, idx, code, name, **kwargs)
    return {
        "peak_date": m["peak_date"],
        "prior_high": round(float(m["prior_high"]), 4),
        "trough_date": m["trough_date"],
        "trough_low": round(float(m["trough_low"]), 4),
        "pullback_pct": round(float(m["pullback_pct"]), 2),
        "rebound_from_trough_pct": round(float(m["rebound_from_trough_pct"]), 2),
        "high_dist_pct": round(float(m["high_dist_pct"]), 2),
        "close_dist_pct": round(float(m["close_dist_pct"]), 2),
        "pct_chg": round(float(m["pct_chg"]), 2),
        "vol_ratio": round(float(m["vol_ratio"]), 2),
        "signal_date": m["signal_date"],
        "exec_buy_date": m.get("exec_buy_date", ""),
        "exec_buy_open": round(float(m.get("exec_buy_open", 0) or 0), 4),
        "buy_trigger_above": round(float(m.get("buy_trigger_above", 0) or 0), 4),
        "mode46_score": score,
    }


def dedupe_mode46_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[tuple, Dict[str, Any]] = {}
    for h in hits:
        key = (h.get("code", ""), h.get("peak_date", ""))
        score = int(h.get("score", 0) or 0)
        prev = best.get(key)
        if prev is None or score > int(prev.get("score", 0) or 0):
            best[key] = h
    return sorted(best.values(), key=lambda x: (x.get("date", ""), x.get("code", "")))
