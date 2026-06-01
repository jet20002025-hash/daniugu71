"""mode34 底部突破回踩二波：底部强阳突破 → 缩量平台 → 二波确认。

参考样本：电科数字 600850（5/15 底 → 5/18～19 突破 → 5/21～25 回踩 → 5/26 二波）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from app.scanner import KlineRow, _is_big_yang_row_modepbs, _vol_ratio_at


def mode34_kw_from_scan_config(cfg: Any) -> Dict[str, Any]:
    """从 ScanConfig 合并 mode34 参数。"""
    base = mode34_default_kw()
    base.update(
        bottom_lookback=int(getattr(cfg, "mode34_bottom_lookback", 60)),
        bottom_pos_max=float(getattr(cfg, "mode34_bottom_pos_max", 0.30)),
        surge_cum_pct_min=float(getattr(cfg, "mode34_surge_cum_pct_min", 12.0)),
        surge_big_pct_min=float(getattr(cfg, "mode34_surge_big_pct_min", 7.0)),
        surge_big_pct_main=float(getattr(cfg, "mode34_surge_big_pct_main", 4.5)),
        pullback_days_min=int(getattr(cfg, "mode34_pullback_days_min", 2)),
        pullback_days_max=int(getattr(cfg, "mode34_pullback_days_max", 8)),
        pullback_dd_max=float(getattr(cfg, "mode34_pullback_dd_max", 0.20)),
        signal_pct_min=float(getattr(cfg, "mode34_signal_pct_min", 1.5)),
    )
    return base


def mode34_default_kw() -> Dict[str, Any]:
    return dict(
        bottom_lookback=60,
        bottom_pos_max=0.30,
        base_search_min=3,
        base_search_max=10,
        surge_search_max=18,
        surge_max_days=4,
        surge_cum_pct_min=12.0,
        surge_big_pct_min=7.0,
        surge_big_pct_main=4.5,
        surge_body_ratio_min=0.45,
        surge_vol_mult=1.35,
        vol_ma=20,
        pullback_days_min=2,
        pullback_days_max=8,
        pullback_dd_min=0.04,
        pullback_dd_max=0.20,
        pullback_vol_ratio_max=0.75,
        platform_break_tol=0.03,
        signal_pct_min=1.5,
        signal_body_ratio_min=0.25,
        signal_vol_mult=1.10,
        signal_vol_vs_pullback=1.12,
        signal_above_pull_high=True,
    )


def _day_pct(rows: List[KlineRow], i: int) -> float:
    if i < 1:
        return 0.0
    prev = float(rows[i - 1].close)
    if prev <= 0:
        return 0.0
    return (float(rows[i].close) - prev) / prev * 100.0


def _range_pos(rows: List[KlineRow], i: int, lookback: int) -> Optional[float]:
    if i < lookback - 1:
        return None
    highs = np.array([float(rows[j].high) for j in range(i - lookback + 1, i + 1)])
    lows = np.array([float(rows[j].low) for j in range(i - lookback + 1, i + 1)])
    c = float(rows[i].close)
    hi, lo = float(highs.max()), float(lows.min())
    if hi <= lo:
        return 0.5
    return (c - lo) / (hi - lo)


def match_mode34_bottom_break_pullback(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Optional[Dict[str, float]]:
    """信号日 idx 是否为「底部突破回踩二波」确认日。"""
    kw = {**mode34_default_kw(), **kwargs}
    n = len(rows)
    need = max(
        int(kw["bottom_lookback"]) + 5,
        int(kw["vol_ma"]) + 5,
        int(kw["surge_search_max"]) + int(kw["pullback_days_max"]) + 5,
    )
    if idx < need or idx >= n:
        return None

    r = rows[idx]
    o, c, h, l_ = float(r.open), float(r.close), float(r.high), float(r.low)
    if c <= o:
        return None
    sig_pct = _day_pct(rows, idx)
    if sig_pct < float(kw["signal_pct_min"]):
        return None
    rng_d = h - l_
    if rng_d <= 0 or (c - o) / rng_d < float(kw["signal_body_ratio_min"]):
        return None

    vol_arr = np.array([float(x.volume) for x in rows], dtype=float)
    close_arr = np.array([float(x.close) for x in rows], dtype=float)
    high_arr = np.array([float(x.high) for x in rows], dtype=float)
    low_arr = np.array([float(x.low) for x in rows], dtype=float)

    sig_vr = _vol_ratio_at(rows, idx, int(kw["vol_ma"]))
    if sig_vr < float(kw["signal_vol_mult"]):
        return None

    search_lo = max(need, idx - int(kw["surge_search_max"]))
    best: Optional[Dict[str, float]] = None

    for peak_i in range(idx - 1, search_lo - 1, -1):
        gap = idx - peak_i
        if gap < int(kw["pullback_days_min"]) or gap > int(kw["pullback_days_max"]):
            continue

        b_min = max(0, peak_i - int(kw["base_search_max"]))
        b_max = max(0, peak_i - int(kw["base_search_min"]))
        if b_max <= b_min:
            continue
        base_slice = range(b_min, b_max)
        base_i = min(base_slice, key=lambda j: float(close_arr[j]))
        base_close = float(close_arr[base_i])
        if base_close <= 0:
            continue

        peak_high = float(np.max(high_arr[base_i : peak_i + 1]))
        rise_pct = (peak_high - base_close) / base_close * 100.0
        if rise_pct < float(kw["surge_cum_pct_min"]):
            continue

        surge_days = 0
        big_days = 0
        peak_vol = 0.0
        for j in range(base_i + 1, peak_i + 1):
            if _is_big_yang_row_modepbs(
                rows[j],
                code,
                name,
                big_pct_min=float(kw["surge_big_pct_min"]),
                big_pct_min_main=float(kw["surge_big_pct_main"]),
                body_ratio_min=float(kw["surge_body_ratio_min"]),
                for_signal=True,
                allow_limit_up=True,
            ):
                big_days += 1
            if _day_pct(rows, j) >= float(kw["surge_big_pct_main"]) * 0.85:
                surge_days += 1
            peak_vol = max(peak_vol, float(vol_arr[j]))

        if big_days < 1 and surge_days < 2:
            continue
        if peak_vol <= 0:
            continue
        if peak_vol < float(kw["surge_vol_mult"]) * max(
            float(np.mean(vol_arr[max(0, peak_i - int(kw["vol_ma"])) : peak_i])),
            1e-9,
        ):
            continue

        pos = _range_pos(rows, base_i, int(kw["bottom_lookback"]))
        if pos is None or pos > float(kw["bottom_pos_max"]):
            continue

        pull_slice = range(peak_i + 1, idx + 1)
        pull_low = float(np.min(low_arr[list(pull_slice)]))
        pull_high_close = float(np.max(close_arr[peak_i:idx]))
        surge_floor = float(
            min(
                low_arr[base_i],
                low_arr[base_i + 1] if base_i + 1 <= peak_i else low_arr[base_i],
            )
        )
        tol = float(kw["platform_break_tol"])
        if pull_low < surge_floor * (1.0 - tol):
            continue

        dd = (peak_high - pull_low) / peak_high if peak_high > 0 else 0.0
        if dd < float(kw["pullback_dd_min"]) or dd > float(kw["pullback_dd_max"]):
            continue

        pull_vols = [float(vol_arr[j]) for j in pull_slice]
        pull_vol_avg = float(np.mean(pull_vols)) if pull_vols else 0.0
        if pull_vol_avg > peak_vol * float(kw["pullback_vol_ratio_max"]):
            continue
        if float(vol_arr[idx]) < pull_vol_avg * float(kw["signal_vol_vs_pullback"]):
            continue

        if kw["signal_above_pull_high"] and c <= pull_high_close * 0.998:
            continue

        peak_close = float(close_arr[peak_i])
        cand = {
            "close": c,
            "pct_chg": sig_pct,
            "vol_ratio": sig_vr,
            "body_ratio": (c - o) / rng_d if rng_d > 0 else 0.0,
            "base_date_idx": float(base_i),
            "base_close": base_close,
            "bottom_pos_pct": float(pos * 100.0),
            "peak_date_idx": float(peak_i),
            "peak_high": peak_high,
            "surge_rise_pct": rise_pct,
            "pullback_days": float(gap),
            "pullback_dd_pct": dd * 100.0,
            "pullback_vol_ratio": pull_vol_avg / peak_vol if peak_vol > 0 else 0.0,
            "surge_floor": surge_floor,
            "pull_low": pull_low,
            "big_surge_days": float(big_days),
            "rise_from_base_to_signal_pct": (c - base_close) / base_close * 100.0,
        }
        if best is None or cand["surge_rise_pct"] > best["surge_rise_pct"]:
            best = cand

    return best


def score_mode34_bottom_break_pullback(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> int:
    det = match_mode34_bottom_break_pullback(rows, idx, code, name, **kwargs)
    if not det:
        return 0
    score = 58.0
    score += min(12.0, float(det["surge_rise_pct"]) * 0.35)
    score += min(10.0, max(0.0, 12.0 - float(det["pullback_dd_pct"])) * 0.8)
    score += min(8.0, (1.0 - float(det["pullback_vol_ratio"])) * 12.0)
    score += min(10.0, float(det["vol_ratio"]) * 4.0)
    score += min(8.0, float(det["pct_chg"]) * 1.2)
    if float(det["bottom_pos_pct"]) <= 15.0:
        score += 8.0
    elif float(det["bottom_pos_pct"]) <= 25.0:
        score += 4.0
    if float(det["big_surge_days"]) >= 2:
        score += 5.0
    return int(min(99, round(score)))


def mode34_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    det = match_mode34_bottom_break_pullback(rows, idx, code, name, **kwargs)
    if not det:
        return {}
    base_i = int(det["base_date_idx"])
    peak_i = int(det["peak_date_idx"])
    return {
        **det,
        "mode34_base_date": str(rows[base_i].date)[:10],
        "mode34_peak_date": str(rows[peak_i].date)[:10],
        "mode34_score": score_mode34_bottom_break_pullback(rows, idx, code, name, **kwargs),
    }
