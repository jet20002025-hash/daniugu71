"""mode35 前高压顶洗盘突破（A 类）：前高锚点 → 压顶整理 → 放量突破。

参考样本：银禧科技 300221（2026-01-23 前高 13.09 → 2026-05-14 突破）。
仅输出 A 类突破信号日，不含 B 类确认。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from app.scanner import KlineRow, _is_st, _modepbs_big_pct_threshold, _vol_ratio_at

MODE35_ID = "mode35"
MODE35_FULL_NAME = "前高压顶洗盘突破"
MODE35_DISPLAY_NAME = f"{MODE35_ID}（{MODE35_FULL_NAME}）"
MODE35_ONE_LINE = "前高锚点→长压顶洗盘→A类放量突破（银禧模版）"


def mode35_default_kw() -> Dict[str, Any]:
    return dict(
        anchor_lookback=180,
        min_under_days=40,
        leak_tol=0.005,
        peak_near_tol=0.002,
        lookback_high_days=60,
        anchor_pct_min=7.0,
        anchor_pct_min_main=4.5,
        anchor_vol_mult=1.75,
        surge_vol_ma=20,
        pullback_dd_min=0.12,
        pullback_dd_max=0.45,
        consolid_days=15,
        consolid_amp_max=0.20,
        break_tol=0.003,
        pct_min=5.0,
        pct_max=15.0,
        pct_min_main=4.0,
        pct_max_main=9.5,
        gap_min=0.0,
        vol_mult=2.0,
        vol_ma=20,
        ma60_slope_days=20,
        ma60_rise_days=60,
        require_ma60_up=True,
        ma60_rise_min_pct=0.3,
        body_ratio_min=0.25,
    )


def mode35_kw_from_scan_config(cfg: Any) -> Dict[str, Any]:
    base = mode35_default_kw()
    for k in base:
        ck = f"mode35_{k}"
        if hasattr(cfg, ck):
            base[k] = getattr(cfg, ck)
    if hasattr(cfg, "mode35_min_score"):
        base["min_score"] = int(getattr(cfg, "mode35_min_score", 62))
    return base


def _day_pct(rows: List[KlineRow], i: int) -> float:
    raw = getattr(rows[i], "pct_chg", None)
    if raw is not None and float(raw) != 0:
        return float(raw)
    if i < 1:
        return 0.0
    prev = float(rows[i - 1].close)
    if prev <= 0:
        return 0.0
    return (float(rows[i].close) - prev) / prev * 100.0


def _lin_slope(y: np.ndarray, n: int) -> float:
    y = y[~np.isnan(y)]
    if len(y) < n:
        return float("nan")
    yy = y[-n:]
    x = np.arange(len(yy), dtype=float)
    return float(np.polyfit(x, yy, 1)[0])


def _pct_threshold(code: str, name: str, kw: Dict[str, Any], *, for_main: bool) -> float:
    if for_main:
        return float(kw.get("pct_min_main", 4.0))
    return _modepbs_big_pct_threshold(
        code,
        name,
        big_pct_min=float(kw["anchor_pct_min"]),
        big_pct_min_main=float(kw["anchor_pct_min_main"]),
    )


def _is_valid_anchor_day(
    rows: List[KlineRow],
    i: int,
    code: str,
    name: str,
    kw: Dict[str, Any],
    high_arr: np.ndarray,
) -> bool:
    r = rows[i]
    o, c = float(r.open), float(r.close)
    if c <= o:
        return False
    pct_min = _pct_threshold(code, name, kw, for_main=False)
    pct = _day_pct(rows, i)
    if pct < pct_min:
        return False
    vma = int(kw.get("surge_vol_ma", 20))
    vol_i = float(r.volume)
    vol_avg = float(np.mean([float(rows[j].volume) for j in range(max(0, i - vma), i)]))
    if vol_avg <= 0 or vol_i < float(kw["anchor_vol_mult"]) * vol_avg:
        return False
    lb = int(kw.get("lookback_high_days", 60))
    if i < lb:
        return False
    prev_max = float(np.max(high_arr[i - lb:i]))
    if high_arr[i] < prev_max * (1.0 - float(kw.get("peak_near_tol", 0.002))):
        return False
    return True


def _find_prior_high_anchor(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    kw: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    min_under = int(kw["min_under_days"])
    lookback = int(kw["anchor_lookback"])
    if idx < min_under + 70:
        return None

    start = max(60, idx - lookback)
    end = idx - min_under
    if end <= start:
        return None

    high_arr = np.array([float(r.high) for r in rows], dtype=float)
    low_arr = np.array([float(r.low) for r in rows], dtype=float)
    close_arr = np.array([float(r.close) for r in rows], dtype=float)

    candidates: List[int] = []
    window_max = float(np.max(high_arr[start:end + 1]))
    peak_tol = float(kw.get("peak_near_tol", 0.002))
    for i in range(start, end + 1):
        if high_arr[i] < window_max * (1.0 - peak_tol):
            continue
        if not _is_valid_anchor_day(rows, i, code, name, kw, high_arr):
            continue
        candidates.append(i)

    if not candidates:
        return None

    best_high = max(high_arr[i] for i in candidates)
    tied = [i for i in candidates if high_arr[i] >= best_high * (1.0 - peak_tol)]
    anchor_i = max(tied)

    prior_high = float(high_arr[anchor_i])
    leak_tol = float(kw["leak_tol"])
    for j in range(anchor_i + 1, idx):
        if high_arr[j] > prior_high * (1.0 + leak_tol):
            return None

    seg_low = float(np.min(low_arr[anchor_i:idx]))
    dd = (prior_high - seg_low) / prior_high if prior_high > 0 else 0.0
    if dd < float(kw["pullback_dd_min"]) or dd > float(kw["pullback_dd_max"]):
        return None

    consolid_days = int(kw.get("consolid_days", 15))
    if idx >= consolid_days:
        seg = close_arr[idx - consolid_days:idx]
        cmin, cmax = float(np.min(seg)), float(np.max(seg))
        cmid = float(np.mean(seg))
        if cmid > 0 and (cmax - cmin) / cmid > float(kw.get("consolid_amp_max", 0.20)):
            return None

    low_i = int(np.argmin(low_arr[anchor_i:idx]) + anchor_i)
    return {
        "anchor_date_idx": float(anchor_i),
        "anchor_date": str(rows[anchor_i].date)[:10],
        "prior_high": prior_high,
        "anchor_close": float(close_arr[anchor_i]),
        "anchor_pct": _day_pct(rows, anchor_i),
        "under_days": float(idx - anchor_i - 1),
        "pullback_low": seg_low,
        "pullback_low_date": str(rows[low_i].date)[:10],
        "pullback_dd_pct": dd * 100.0,
    }


def match_mode35_prior_high_breakout(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """A 类：信号日最高价突破 prior_high，压顶整理后放量收阳。"""
    kw = {**mode35_default_kw(), **kwargs}
    if idx < 80 or idx >= len(rows):
        return None

    anchor = _find_prior_high_anchor(rows, idx, code, name, kw)
    if anchor is None:
        return None

    prior_high = float(anchor["prior_high"])
    r = rows[idx]
    o, c, h, l = float(r.open), float(r.close), float(r.high), float(r.low)
    if c <= o:
        return None

    break_tol = float(kw["break_tol"])
    if h < prior_high * (1.0 + break_tol):
        return None

    pct_lim = _pct_threshold(code, name, kw, for_main=_limit_rate_main(code, name))
    pct_max = float(kw["pct_max_main"] if _limit_rate_main(code, name) else kw["pct_max"])
    pct = _day_pct(rows, idx)
    if pct < pct_lim or pct > pct_max:
        return None

    rng = h - l
    if rng <= 0:
        return None
    body_ratio = (c - o) / rng
    if body_ratio < float(kw.get("body_ratio_min", 0.25)):
        return None

    vol_ma = int(kw.get("vol_ma", 20))
    vol_ratio = _vol_ratio_at(rows, idx, vol_ma)
    if vol_ratio < float(kw["vol_mult"]):
        return None

    gap_pct = 0.0
    if idx >= 1 and float(rows[idx - 1].close) > 0:
        gap_pct = (o - float(rows[idx - 1].close)) / float(rows[idx - 1].close) * 100.0
    if float(kw.get("gap_min", 0)) > 0 and gap_pct < float(kw["gap_min"]):
        return None

    closes = np.array([float(x.close) for x in rows], dtype=float)
    ma60 = np.convolve(closes, np.ones(60) / 60, mode="valid")
    ma60_full = np.full(len(closes), np.nan)
    ma60_full[59:] = ma60
    ma60_now = ma60_full[idx]
    slope_n = int(kw.get("ma60_slope_days", 20))
    rise_n = int(kw.get("ma60_rise_days", 60))
    ma60_slope = _lin_slope(ma60_full[: idx + 1], slope_n)
    if bool(kw.get("require_ma60_up", True)):
        if np.isnan(ma60_now) or c < ma60_now:
            return None
        if idx >= rise_n and not np.isnan(ma60_full[idx - rise_n]):
            ma60_rise = (ma60_now - ma60_full[idx - rise_n]) / ma60_full[idx - rise_n] * 100.0
            if ma60_rise < float(kw.get("ma60_rise_min_pct", 0.3)):
                return None
        elif ma60_slope <= 0:
            return None

    break_high_pct = (h / prior_high - 1.0) * 100.0
    break_close_pct = (c / prior_high - 1.0) * 100.0
    dist_ma60 = (c - ma60_now) / ma60_now * 100.0 if ma60_now > 0 else 0.0

    return {
        **anchor,
        "signal_date": str(rows[idx].date)[:10],
        "event_type": "突破",
        "close": c,
        "pct_chg": pct,
        "vol_ratio": vol_ratio,
        "gap_pct": gap_pct,
        "body_ratio": body_ratio,
        "break_high_pct": break_high_pct,
        "break_close_pct": break_close_pct,
        "dist_ma60_pct": dist_ma60,
        "ma60": float(ma60_now) if not np.isnan(ma60_now) else 0.0,
        "ma60_slope20": ma60_slope,
        "close_above_prior_high": c >= prior_high,
    }


def _limit_rate_main(code: str, name: str) -> bool:
    from app.scanner import _limit_rate

    return _limit_rate(code, name) < 0.15 and not _is_st(name or "")


def _ma_series(closes: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(closes), np.nan)
    if len(closes) >= n:
        out[n - 1:] = np.convolve(closes, np.ones(n) / n, mode="valid")
    return out


def _mode35_quality_features(
    rows: List[KlineRow],
    idx: int,
    m: Dict[str, Any],
) -> Dict[str, float]:
    """突破日前量价/均线质量特征，用于加减分。"""
    anchor_i = int(m["anchor_date_idx"])
    prior_high = float(m["prior_high"])
    closes = np.array([float(r.close) for r in rows], dtype=float)
    highs = np.array([float(r.high) for r in rows], dtype=float)
    lows = np.array([float(r.low) for r in rows], dtype=float)
    vols = np.array([float(r.volume) for r in rows], dtype=float)

    seg_lows = lows[anchor_i + 1:idx]
    low_i = anchor_i + 1 + int(np.argmin(seg_lows)) if len(seg_lows) else anchor_i
    days_low_to_sig = float(idx - low_i)

    vol_anchor = vols[anchor_i]
    vol_pre15 = float(np.mean(vols[max(anchor_i + 1, idx - 15):idx])) if idx > anchor_i + 1 else 0.0
    vol_pre15_vs_anchor = vol_pre15 / vol_anchor if vol_anchor > 0 else 0.0

    near_high_days = float(
        sum(1 for j in range(anchor_i + 1, idx) if highs[j] > prior_high * 0.97)
    )

    if low_i > anchor_i:
        v_down = float(np.mean(vols[anchor_i + 1:low_i + 1]))
        v_up = float(np.mean(vols[low_i:idx])) if idx > low_i else 0.0
        vol_up_vs_down = v_up / v_down if v_down > 0 else 0.0
    else:
        vol_up_vs_down = 0.0

    ma20 = _ma_series(closes, 20)
    ma60 = _ma_series(closes, 60)
    ma20_ma60_spread_pct = 0.0
    ma20_rise20 = 0.0
    if not np.isnan(ma20[idx]) and not np.isnan(ma60[idx]) and closes[idx] > 0:
        ma20_ma60_spread_pct = (ma20[idx] - ma60[idx]) / closes[idx] * 100.0
    if idx >= 20 and not np.isnan(ma20[idx - 20]) and ma20[idx - 20] > 0:
        ma20_rise20 = (ma20[idx] - ma20[idx - 20]) / ma20[idx - 20] * 100.0

    under_len = idx - anchor_i - 1
    vol_under_second_half_vs_anchor = 0.0
    if under_len > 0 and vol_anchor > 0:
        mid = anchor_i + 1 + under_len // 2
        vol_under_second_half_vs_anchor = float(np.mean(vols[mid:idx])) / vol_anchor

    sig_r = rows[idx]
    o, c, h, l = (
        float(sig_r.open),
        float(sig_r.close),
        float(sig_r.high),
        float(sig_r.low),
    )
    close_pos_in_range = (c - l) / (h - l) if h > l else 0.5

    # 复盘用：信号后 3 日内是否收盘站稳前高；-1 表示尚无后续 K 线
    confirm_within_3d = -1.0
    if idx + 3 < len(rows):
        confirm_within_3d = (
            1.0
            if any(closes[j] >= prior_high for j in range(idx + 1, idx + 4))
            else 0.0
        )

    vol_sig = vols[idx]
    vol_sig_vs_anchor = vol_sig / vol_anchor if vol_anchor > 0 else 0.0
    vol_ratio_anchor = _vol_ratio_at(rows, anchor_i, 20)
    vol_ratio_sig = float(m.get("vol_ratio", 0))
    vol_ratio_expansion = (
        vol_ratio_sig / vol_ratio_anchor if vol_ratio_anchor > 0 else 0.0
    )

    return {
        "days_low_to_sig": days_low_to_sig,
        "vol_pre15_vs_anchor": vol_pre15_vs_anchor,
        "vol_under_second_half_vs_anchor": vol_under_second_half_vs_anchor,
        "near_high_days": near_high_days,
        "vol_up_vs_down": vol_up_vs_down,
        "ma20_ma60_spread_pct": ma20_ma60_spread_pct,
        "ma20_rise20": ma20_rise20,
        "close_pos_in_range": close_pos_in_range,
        "confirm_within_3d": confirm_within_3d,
        "vol_sig_vs_anchor": vol_sig_vs_anchor,
        "vol_ratio_anchor": vol_ratio_anchor,
        "vol_ratio_expansion": vol_ratio_expansion,
    }


def score_mode35_prior_high_breakout(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> int:
    m = match_mode35_prior_high_breakout(rows, idx, code, name, **kwargs)
    if not m:
        return 0

    q = _mode35_quality_features(rows, idx, m)

    score = 52.0
    under = float(m["under_days"])
    if 40.0 <= under <= 80.0:
        if under >= 60.0:
            score += 18.0
        else:
            score += 12.0
    elif under >= 30.0:
        score += 6.0

    dd = float(m["pullback_dd_pct"])
    if 20.0 <= dd <= 35.0:
        score += 12.0
    elif 15.0 <= dd <= 40.0:
        score += 8.0

    score += min(12.0, float(m["vol_ratio"]) * 3.0)
    score += min(8.0, float(m["pct_chg"]) * 0.8)
    if float(m["gap_pct"]) >= 2.0:
        score += 6.0
    if m.get("close_above_prior_high"):
        score += 8.0
    else:
        score += min(6.0, float(m["break_high_pct"]) * 2.0)

    br = float(m.get("body_ratio", 0))
    if 0.35 <= br <= 0.65:
        score += 4.0

    # 5 月样本复盘：突破日前量价/均线加减分
    if under > 80.0:
        score -= 18.0
    if q["vol_pre15_vs_anchor"] < 0.15:
        score -= 18.0
    elif q["vol_pre15_vs_anchor"] < 0.25:
        score -= 8.0
    if q["ma20_ma60_spread_pct"] < 0.0 and q["ma20_rise20"] < 3.0:
        score -= 15.0
    if q["near_high_days"] >= 4.0:
        score -= 12.0
    if q["vol_up_vs_down"] < 0.70:
        score -= 10.0
    if float(m["gap_pct"]) < 1.0:
        score -= 8.0
    if q["days_low_to_sig"] > 50.0:
        score -= 10.0

    # 4 月样本复盘：突破日质量与压顶末段量能
    if q["vol_under_second_half_vs_anchor"] < 0.20:
        score -= 12.0
    if q["close_pos_in_range"] < 0.50:
        score -= 12.0
    if q["confirm_within_3d"] == 0.0:
        score -= 15.0

    # 突破日量 vs 锚点(前期新高)日量
    if q["vol_sig_vs_anchor"] < 0.85:
        score -= 12.0
    if q["vol_ratio_expansion"] >= 2.0:
        score += 10.0

    if 25.0 <= q["days_low_to_sig"] <= 40.0 and q["ma20_rise20"] > 8.0:
        score += 12.0
    if q["vol_up_vs_down"] > 0.85:
        score += 10.0
    if float(m["gap_pct"]) >= 2.0 and m.get("close_above_prior_high"):
        score += 6.0

    return int(min(99, max(0, round(score))))


def mode35_signal_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    m = match_mode35_prior_high_breakout(rows, idx, code, name, **kwargs)
    if not m:
        return {}
    q = _mode35_quality_features(rows, idx, m)
    score = score_mode35_prior_high_breakout(rows, idx, code, name, **kwargs)
    return {**m, **q, "mode35_score": score}
