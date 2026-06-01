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


def _find_mode34_setup(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Optional[Dict[str, float]]:
    """截至 idx（含）识别突破+回踩结构，不要求当日为二波确认阳。"""
    kw = {**mode34_default_kw(), **kwargs}
    n = len(rows)
    need = max(
        int(kw["bottom_lookback"]) + 5,
        int(kw["vol_ma"]) + 5,
        int(kw["surge_search_max"]) + int(kw["pullback_days_max"]) + 5,
    )
    if idx < need or idx >= n:
        return None

    vol_arr = np.array([float(x.volume) for x in rows], dtype=float)
    close_arr = np.array([float(x.close) for x in rows], dtype=float)
    high_arr = np.array([float(x.high) for x in rows], dtype=float)
    low_arr = np.array([float(x.low) for x in rows], dtype=float)

    search_lo = max(need, idx - int(kw["surge_search_max"]))
    best: Optional[Dict[str, float]] = None
    best_peak_high = -1.0

    for peak_i in range(idx - 1, search_lo - 1, -1):
        gap = idx - peak_i
        if gap < 1 or gap > int(kw["pullback_days_max"]) + 1:
            continue

        b_min = max(0, peak_i - int(kw["base_search_max"]))
        b_max = max(0, peak_i - int(kw["base_search_min"]))
        if b_max <= b_min:
            continue
        base_i = min(range(b_min, b_max), key=lambda j: float(close_arr[j]))
        base_close = float(close_arr[base_i])
        if base_close <= 0:
            continue

        peak_high = float(np.max(high_arr[base_i : peak_i + 1]))
        rise_pct = (peak_high - base_close) / base_close * 100.0
        if rise_pct < float(kw["surge_cum_pct_min"]):
            continue

        big_days = 0
        surge_days = 0
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
        if not pull_slice:
            continue
        pull_low = float(np.min(low_arr[list(pull_slice)]))
        pull_high_close = float(np.max(close_arr[peak_i:idx]))
        pull_high = float(np.max(high_arr[peak_i:idx]))
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

        r = rows[idx]
        o, c, h, l_ = float(r.open), float(r.close), float(r.high), float(r.low)
        cand = {
            "close": c,
            "pct_chg": _day_pct(rows, idx),
            "vol_ratio": _vol_ratio_at(rows, idx, int(kw["vol_ma"])),
            "base_date_idx": float(base_i),
            "base_close": base_close,
            "bottom_pos_pct": float(pos * 100.0),
            "peak_date_idx": float(peak_i),
            "peak_high": peak_high,
            "peak_close": float(close_arr[peak_i]),
            "surge_rise_pct": rise_pct,
            "pullback_days": float(gap),
            "pullback_dd_pct": dd * 100.0,
            "pullback_vol_ratio": pull_vol_avg / peak_vol if peak_vol > 0 else 0.0,
            "surge_floor": surge_floor,
            "pull_low": pull_low,
            "pull_high_close": pull_high_close,
            "pull_high": pull_high,
            "big_surge_days": float(big_days),
            "dist_to_pull_high_pct": (pull_high_close - c) / pull_high_close * 100.0
            if pull_high_close > 0
            else 0.0,
        }
        ph = float(cand["peak_high"])
        gap_f = float(cand["pullback_days"])
        if best is None:
            best_peak_high = ph
            best = cand
            continue
        best_gap = float(best["pullback_days"])
        if gap_f > best_gap or (gap_f == best_gap and ph > best_peak_high):
            best_peak_high = ph
            best = cand

    return best


def match_mode34_bottom_break_pullback(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Optional[Dict[str, float]]:
    """信号日 idx 是否为「底部突破回踩二波」确认日。"""
    kw = {**mode34_default_kw(), **kwargs}
    det = _find_mode34_setup(rows, idx, code, name, **kwargs)
    if det is None:
        return None

    gap = float(det["pullback_days"])
    if gap < int(kw["pullback_days_min"]) or gap > int(kw["pullback_days_max"]):
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

    if float(det["vol_ratio"]) < float(kw["signal_vol_mult"]):
        return None

    vol_arr = np.array([float(x.volume) for x in rows], dtype=float)
    peak_i = int(det["peak_date_idx"])
    pull_vols = [float(vol_arr[j]) for j in range(peak_i + 1, idx + 1)]
    pull_vol_avg = float(np.mean(pull_vols)) if pull_vols else 0.0
    if float(vol_arr[idx]) < pull_vol_avg * float(kw["signal_vol_vs_pullback"]):
        return None

    if kw["signal_above_pull_high"] and c <= float(det["pull_high_close"]) * 0.998:
        return None

    base_close = float(det["base_close"])
    det["body_ratio"] = (c - o) / rng_d if rng_d > 0 else 0.0
    det["rise_from_base_to_signal_pct"] = (c - base_close) / base_close * 100.0
    return det


def match_mode34_watchlist(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """观察日：已在回踩平台，次日/后日可能出买点（如电科 5/22～5/24）。"""
    kw = {**mode34_default_kw(), **kwargs}
    if match_mode34_bottom_break_pullback(rows, idx, code, name, **kwargs):
        return None

    det = _find_mode34_setup(rows, idx, code, name, **kwargs)
    if det is None:
        return None

    gap = float(det["pullback_days"])
    if gap < 2 or gap > int(kw["pullback_days_max"]):
        return None

    vr = float(det["vol_ratio"])
    if vr > 1.0:
        return None
    if float(det["pullback_vol_ratio"]) > 0.82:
        return None
    if float(det["dist_to_pull_high_pct"]) < -1.0:
        return None

    peak_i = int(det["peak_date_idx"])
    base_i = int(det["base_date_idx"])
    return {
        **det,
        "watch_date": str(rows[idx].date)[:10],
        "base_date": str(rows[base_i].date)[:10],
        "peak_date": str(rows[peak_i].date)[:10],
        "watch_score": _score_mode34_watchlist(det),
    }


def _score_mode34_watchlist(det: Dict[str, float]) -> int:
    score = 50.0
    score += min(15.0, float(det["surge_rise_pct"]) * 0.4)
    score += min(12.0, max(0.0, 15.0 - float(det["pullback_dd_pct"])) * 0.7)
    score += min(10.0, (1.0 - float(det["pullback_vol_ratio"])) * 10.0)
    score += min(8.0, float(det["dist_to_pull_high_pct"]) * 1.5)
    if float(det["vol_ratio"]) <= 0.7:
        score += 8.0
    return int(min(99, round(score)))


def mode34_prebuy_advice(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """预案日：类似 5/25，收盘后/盘中辅助判断次日是否值得买（电科模版）。"""
    kw = {**mode34_default_kw(), **kwargs}
    det = _find_mode34_setup(rows, idx, code, name, **kwargs)
    if det is None:
        return None

    gap = float(det["pullback_days"])
    if gap < int(kw["pullback_days_min"]) or gap > int(kw["pullback_days_max"]):
        return None

    r = rows[idx]
    o, c, h, l_ = float(r.open), float(r.close), float(r.high), float(r.low)
    pct = _day_pct(rows, idx)
    rng = h - l_
    body_ratio = (c - o) / rng if rng > 0 else 0.0
    pull_high = float(det["pull_high"])
    pull_high_close = float(det["pull_high_close"])
    surge_floor = float(det["surge_floor"])

    signals: List[str] = []
    score = 40.0

    if c > o:
        score += 12.0
        signals.append("收阳")
    else:
        signals.append("收阴")
    if 0 < pct <= 6.0:
        score += 10.0
        signals.append(f"小涨{pct:.1f}%")
    elif pct <= 0:
        score -= 5.0
    if body_ratio <= 0.55 and c > o:
        score += 6.0
        signals.append("实体温和")
    if float(det["vol_ratio"]) <= 0.75:
        score += 12.0
        signals.append("缩量")
    elif float(det["vol_ratio"]) <= 0.9:
        score += 5.0
    if c <= pull_high_close * 1.02:
        score += 8.0
        signals.append("未突破平台")
    else:
        score -= 8.0
        signals.append("已破平台")
    if l_ >= surge_floor * 0.99:
        score += 8.0
        signals.append("不破铁底")
    else:
        score -= 15.0
        signals.append("破铁底")
    if idx >= 1 and c >= float(rows[idx - 1].close):
        score += 5.0
        signals.append("收>昨收")

    # 次日若高开突破：盘中参考价
    buy_trigger = round(h * 1.001, 2)
    stop_loss = round(min(float(det["pull_low"]), surge_floor) * 0.99, 2)
    advice_score = int(min(99, max(0, round(score))))

    if advice_score >= 72:
        advice = "偏多买入"
        action = "次日盘中突破昨高可试仓"
    elif advice_score >= 58:
        advice = "轻仓试探"
        action = "仅突破昨高且放量再跟"
    elif advice_score >= 45:
        advice = "继续观察"
        action = "等更明确阳线"
    else:
        advice = "放弃"
        action = "结构走弱"

    peak_i = int(det["peak_date_idx"])
    base_i = int(det["base_date_idx"])
    next_hit = (
        idx + 1 < len(rows)
        and match_mode34_bottom_break_pullback(rows, idx + 1, code, name, **kwargs) is not None
    )

    return {
        **det,
        "prebuy_date": str(rows[idx].date)[:10],
        "base_date": str(rows[base_i].date)[:10],
        "peak_date": str(rows[peak_i].date)[:10],
        "advice": advice,
        "action": action,
        "advice_score": advice_score,
        "signals": ",".join(signals),
        "buy_trigger_above": buy_trigger,
        "stop_below": stop_loss,
        "yesterday_high": round(h, 2),
        "next_day_mode34": next_hit,
    }


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
