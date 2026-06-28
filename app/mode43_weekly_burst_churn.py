"""mode43 周线爆量洗盘周：主升途中放量分歧换手，趋势未破后往往续涨。

样本：杭电股份 603618 2026-02-13 所在周（前4周+71%，量/5周均2.2x，振幅24%，周收仍>WMA10）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.scanner import KlineRow, _is_st
from app.weekly_ma import daily_to_weekly_with_volume_and_last_index

MODE43_ID = "mode43"
MODE43_FULL_NAME = "周线爆量洗盘周"
MODE43_DISPLAY_NAME = f"{MODE43_ID}（{MODE43_FULL_NAME}）"
MODE43_ONE_LINE = (
    "主升途中周线放量分歧（振幅大、涨跌有限），收盘在周MA10之上；"
    "信号周收盘确认，次日开盘买"
)


def mode43_default_kw() -> Dict[str, Any]:
    return dict(
        min_prior_4w_gain_pct=20.0,
        min_prior_8w_swing_pct=40.0,
        min_vol_vs_ma5=2.0,
        min_vol_vs_ma10=1.8,
        vol_ma5_weeks=5,
        vol_ma10_weeks=10,
        min_amplitude_pct=15.0,
        week_chg_min_pct=-12.0,
        week_chg_max_pct=12.0,
        require_above_wma10=True,
        min_wma10_slope_pct=0.0,
        wma10_slope_weeks=4,
        max_break_wma10_pct=5.0,
        require_vol_gt_prev_week=False,
        min_prior_week_gain_pct=0.0,
        max_week_vol_hand=15_000_000.0,
        min_weeks_history=16,
        min_score=60,
    )


def mode43_kw_from_scan_config(cfg: Any) -> Dict[str, Any]:
    base = mode43_default_kw()
    for k in base:
        ck = f"mode43_{k}"
        if hasattr(cfg, ck):
            base[k] = getattr(cfg, ck)
    if hasattr(cfg, "mode43_min_score"):
        base["min_score"] = int(getattr(cfg, "mode43_min_score", 60))
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


def _weekly_bundle(rows: List[KlineRow]) -> Tuple[List[tuple], List[int]]:
    return daily_to_weekly_with_volume_and_last_index(rows)


def _week_key_from_date(date_str: str) -> Tuple[int, int]:
    from app.weekly_ma import _week_key

    return _week_key(date_str)


def match_mode43_weekly_burst_churn(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    if _is_st(name or ""):
        return None
    kw = mode43_default_kw()
    kw.update({k: v for k, v in kwargs.items() if k in kw or k.startswith("mode43")})

    weekly, last_idx = _weekly_bundle(rows)
    if not weekly:
        return None

    w_idx: Optional[int] = None
    for i, li in enumerate(last_idx):
        if li == idx:
            w_idx = i
            break
    if w_idx is None:
        return None

    min_hist = int(kw["min_weeks_history"])
    if w_idx < min_hist:
        return None

    opens = np.array([float(w[1]) for w in weekly], dtype=float)
    highs = np.array([float(w[2]) for w in weekly], dtype=float)
    lows = np.array([float(w[3]) for w in weekly], dtype=float)
    closes = np.array([float(w[4]) for w in weekly], dtype=float)
    vols = np.array([float(w[5]) for w in weekly], dtype=float)

    cur_open = float(opens[w_idx])
    cur_high = float(highs[w_idx])
    cur_low = float(lows[w_idx])
    cur_close = float(closes[w_idx])
    cur_vol = float(vols[w_idx])

    max_vol = float(kw["max_week_vol_hand"])
    if cur_vol > max_vol or cur_vol <= 0:
        return None

    week_chg_pct = (cur_close / cur_open - 1.0) * 100.0 if cur_open > 0 else 0.0
    if week_chg_pct < float(kw["week_chg_min_pct"]) or week_chg_pct > float(kw["week_chg_max_pct"]):
        return None

    if cur_low <= 0:
        return None
    amplitude_pct = (cur_high - cur_low) / cur_low * 100.0
    if amplitude_pct < float(kw["min_amplitude_pct"]):
        return None

    wma10 = _ma_at(closes, w_idx, 10)
    if np.isnan(wma10) or wma10 <= 0:
        return None
    if kw.get("require_above_wma10", True):
        if cur_close <= wma10:
            return None
    if cur_low < wma10 * (1.0 - float(kw["max_break_wma10_pct"]) / 100.0):
        return None

    wma10_slope = _ma_slope_pct(
        closes, w_idx, 10, int(kw["wma10_slope_weeks"])
    )
    if wma10_slope < float(kw["min_wma10_slope_pct"]):
        return None

    prior4_ok = False
    if w_idx >= 4:
        prior4 = (cur_close / float(closes[w_idx - 4]) - 1.0) * 100.0
        prior4_ok = prior4 >= float(kw["min_prior_4w_gain_pct"])
    else:
        prior4 = 0.0

    swing_ok = False
    prior8_swing_pct = 0.0
    if w_idx >= 1:
        look = min(8, w_idx)
        seg_h = highs[w_idx - look : w_idx + 1]
        seg_l = lows[w_idx - look : w_idx + 1]
        mn = float(np.min(seg_l))
        mx = float(np.max(seg_h))
        if mn > 0:
            prior8_swing_pct = (mx / mn - 1.0) * 100.0
            swing_ok = prior8_swing_pct >= float(kw["min_prior_8w_swing_pct"])

    if not (prior4_ok or swing_ok):
        return None

    vma5_n = int(kw["vol_ma5_weeks"])
    vma10_n = int(kw["vol_ma10_weeks"])
    if w_idx < vma10_n - 1:
        return None
    vol_ma5 = float(np.mean(vols[w_idx - vma5_n + 1 : w_idx]))
    vol_ma10 = float(np.mean(vols[w_idx - vma10_n + 1 : w_idx]))
    if vol_ma5 <= 0 or vol_ma10 <= 0:
        return None
    vol_vs_ma5 = cur_vol / vol_ma5
    vol_vs_ma10 = cur_vol / vol_ma10
    if vol_vs_ma5 < float(kw["min_vol_vs_ma5"]) or vol_vs_ma10 < float(kw["min_vol_vs_ma10"]):
        return None

    prev_week_chg_pct = 0.0
    prev_week_vol = 0.0
    if w_idx >= 1:
        prev_week_chg_pct = (
            (float(closes[w_idx - 1]) / float(closes[w_idx - 2]) - 1.0) * 100.0
            if w_idx >= 2
            else 0.0
        )
        prev_week_vol = float(vols[w_idx - 1])
        if float(kw.get("min_prior_week_gain_pct", 0) or 0) > 0:
            if prev_week_chg_pct < float(kw["min_prior_week_gain_pct"]):
                return None
        if kw.get("require_vol_gt_prev_week", False) and prev_week_vol > 0:
            if cur_vol <= prev_week_vol:
                return None

    vol_gt_prev = prev_week_vol > 0 and cur_vol > prev_week_vol

    wma20 = _ma_at(closes, w_idx, 20)
    close_dist_wma10 = (cur_close / wma10 - 1.0) * 100.0

    signal_date = str(rows[idx].date)[:10]
    week_end_date = str(rows[last_idx[w_idx]].date)[:10]
    exec_buy_date = ""
    exec_buy_open = 0.0
    if idx + 1 < len(rows):
        exec_buy_date = str(rows[idx + 1].date)[:10]
        exec_buy_open = float(rows[idx + 1].open)

    return {
        "week_end_date": week_end_date,
        "prior_4w_gain_pct": prior4,
        "prior_8w_swing_pct": prior8_swing_pct,
        "week_chg_pct": week_chg_pct,
        "prev_week_chg_pct": prev_week_chg_pct,
        "amplitude_pct": amplitude_pct,
        "vol_week": cur_vol,
        "vol_vs_ma5": vol_vs_ma5,
        "vol_vs_ma10": vol_vs_ma10,
        "vol_gt_prev_week": vol_gt_prev,
        "wma10": wma10,
        "wma20": wma20 if not np.isnan(wma20) else 0.0,
        "wma10_slope_pct": wma10_slope,
        "close_dist_wma10_pct": close_dist_wma10,
        "close": cur_close,
        "low": cur_low,
        "high": cur_high,
        "signal_date": signal_date,
        "exec_buy_date": exec_buy_date,
        "exec_buy_open": exec_buy_open,
        "week_idx": w_idx,
    }


def score_mode43_weekly_burst_churn(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> int:
    m = match_mode43_weekly_burst_churn(rows, idx, code, name, **kwargs)
    if not m:
        return 0
    score = 32.0
    p4 = float(m["prior_4w_gain_pct"])
    if p4 >= 50.0:
        score += 18.0
    elif p4 >= 30.0:
        score += 12.0
    elif p4 >= 20.0:
        score += 6.0
    sw = float(m["prior_8w_swing_pct"])
    if sw >= 80.0:
        score += 10.0
    elif sw >= 50.0:
        score += 6.0
    vr5 = float(m["vol_vs_ma5"])
    if vr5 >= 3.0:
        score += 16.0
    elif vr5 >= 2.5:
        score += 12.0
    elif vr5 >= 2.0:
        score += 8.0
    amp = float(m["amplitude_pct"])
    if amp >= 22.0:
        score += 10.0
    elif amp >= 18.0:
        score += 6.0
    elif amp >= 15.0:
        score += 3.0
    if m.get("vol_gt_prev_week"):
        score += 8.0
    pw = float(m["prev_week_chg_pct"])
    if pw >= 25.0:
        score += 10.0
    elif pw >= 15.0:
        score += 6.0
    wc = abs(float(m["week_chg_pct"]))
    if wc <= 3.0:
        score += 8.0
    elif wc <= 8.0:
        score += 4.0
    if float(m["wma10_slope_pct"]) > 5.0:
        score += 6.0
    return int(min(100, max(0, round(score))))


def mode43_signal_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    m = match_mode43_weekly_burst_churn(rows, idx, code, name, **kwargs)
    if not m:
        return {}
    return {
        "week_end_date": m["week_end_date"],
        "prior_4w_gain_pct": round(float(m["prior_4w_gain_pct"]), 2),
        "prior_8w_swing_pct": round(float(m["prior_8w_swing_pct"]), 2),
        "week_chg_pct": round(float(m["week_chg_pct"]), 2),
        "prev_week_chg_pct": round(float(m["prev_week_chg_pct"]), 2),
        "amplitude_pct": round(float(m["amplitude_pct"]), 2),
        "vol_vs_ma5": round(float(m["vol_vs_ma5"]), 2),
        "vol_vs_ma10": round(float(m["vol_vs_ma10"]), 2),
        "vol_gt_prev_week": bool(m.get("vol_gt_prev_week")),
        "wma10": round(float(m["wma10"]), 4),
        "wma10_slope_pct": round(float(m["wma10_slope_pct"]), 2),
        "close_dist_wma10_pct": round(float(m["close_dist_wma10_pct"]), 2),
        "low": round(float(m["low"]), 4),
        "signal_date": m["signal_date"],
        "exec_buy_date": m.get("exec_buy_date", ""),
        "exec_buy_open": round(float(m.get("exec_buy_open", 0) or 0), 4),
    }


def dedupe_mode43_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[tuple, Dict[str, Any]] = {}
    for h in hits:
        key = (h.get("code", ""), h.get("week_end_date", h.get("date", "")))
        prev = best.get(key)
        if prev is None or float(h.get("score", 0) or 0) > float(prev.get("score", 0) or 0):
            best[key] = h
    return sorted(best.values(), key=lambda x: (x.get("date", ""), x.get("code", "")))
