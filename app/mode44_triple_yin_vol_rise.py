"""mode44 三连阴量价背离：连续三根阴线，成交量逐日放大（价跌量增）。

样本：国投中鲁 600962 @ 2026-06-24
  - 6/22～6/24 三连阴，量 67593 → 71530 → 87342 递增
  - 6/25 涨停反包 → 信号日收盘确认，次日开盘买
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from app.scanner import KlineRow, _is_st, _vol_ratio_at

MODE44_ID = "mode44"
MODE44_FULL_NAME = "三连阴量价背离"
MODE44_DISPLAY_NAME = f"{MODE44_ID}（{MODE44_FULL_NAME}）"
MODE44_ONE_LINE = "连续3日阴线且成交量逐日放大（价跌量增），第三日收盘在MA120之上"


def mode44_default_kw() -> Dict[str, Any]:
    return dict(
        streak_days=3,
        require_yin=True,
        require_close_down=True,
        require_vol_strict_inc=True,
        min_vol_step_ratio=1.02,
        min_cum_drop_pct=2.0,
        max_cum_drop_pct=18.0,
        max_single_day_drop_pct=12.0,
        prior_lookback=20,
        min_prior_rise_pct=0.0,
        ma_periods=(20, 60),
        ma_touch_pct=8.0,
        ma120_period=120,
        require_above_ma120=True,
        vol_ma=20,
        min_score=60,
    )


def mode44_kw_from_scan_config(cfg: Any) -> Dict[str, Any]:
    base = mode44_default_kw()
    for k in base:
        ck = f"mode44_{k}"
        if hasattr(cfg, ck):
            base[k] = getattr(cfg, ck)
    if hasattr(cfg, "mode44_min_score"):
        base["min_score"] = int(getattr(cfg, "mode44_min_score", 60))
    return base


def _row_volume(rows: List[KlineRow], idx: int) -> float:
    r = rows[idx]
    return float(getattr(r, "volume", 0) or getattr(r, "vol", 0) or 0)


def _is_yin(rows: List[KlineRow], idx: int) -> bool:
    r = rows[idx]
    return float(r.close) < float(r.open)


def _ma_at(closes: np.ndarray, idx: int, n: int) -> float:
    if idx < n - 1:
        return float("nan")
    return float(np.mean(closes[idx - n + 1 : idx + 1]))


def _near_ma_bonus(
    closes: np.ndarray,
    low: float,
    idx: int,
    ma_periods: tuple[int, ...],
    touch_pct: float,
) -> tuple[float, Optional[int], Optional[float]]:
    best_dist = 999.0
    best_period: Optional[int] = None
    best_ma: Optional[float] = None
    for n in ma_periods:
        ma = _ma_at(closes, idx, n)
        if np.isnan(ma) or ma <= 0:
            continue
        dist = abs(low / ma - 1.0) * 100.0
        if dist < best_dist:
            best_dist = dist
            best_period = n
            best_ma = ma
    if best_period is None or best_ma is None:
        return 0.0, None, None
    if best_dist <= touch_pct:
        return max(0.0, 12.0 - best_dist * 1.2), best_period, best_ma
    return 0.0, best_period, best_ma


def match_mode44_triple_yin_vol_rise(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    if _is_st(name or ""):
        return None
    kw = mode44_default_kw()
    kw.update({k: v for k, v in kwargs.items() if k in kw or k.startswith("mode44")})

    streak = int(kw["streak_days"])
    ma120_n = int(kw.get("ma120_period", 120) or 120)
    min_len = max(streak - 1, ma120_n - 1)
    if streak < 2 or idx < min_len or idx >= len(rows):
        return None

    start = idx - streak + 1
    closes = np.array([float(r.close) for r in rows], dtype=float)
    vols = np.array([_row_volume(rows, j) for j in range(len(rows))], dtype=float)

    day_metrics: List[Dict[str, Any]] = []
    for j in range(start, idx + 1):
        r = rows[j]
        o, c, h, l = float(r.open), float(r.close), float(r.high), float(r.low)
        pct = float(getattr(r, "pct_chg", 0) or 0)
        if pct == 0 and j > 0 and closes[j - 1] > 0:
            pct = (c / closes[j - 1] - 1.0) * 100.0

        if kw.get("require_yin", True) and c >= o:
            return None
        if kw.get("require_close_down", True) and j > start and c >= closes[j - 1]:
            return None

        max_drop = float(kw.get("max_single_day_drop_pct", 12.0) or 12.0)
        if j > 0 and pct < -max_drop:
            return None

        day_metrics.append(
            dict(
                date=str(r.date)[:10],
                open=o,
                close=c,
                high=h,
                low=l,
                volume=float(vols[j]),
                pct_chg=pct,
            )
        )

    min_step = float(kw.get("min_vol_step_ratio", 1.02) or 1.02)
    if kw.get("require_vol_strict_inc", True):
        for k in range(1, streak):
            v0, v1 = vols[start + k - 1], vols[start + k]
            if v0 <= 0 or v1 <= 0:
                return None
            if v1 < v0 * min_step:
                return None

    c0 = closes[start]
    c1 = closes[idx]
    if c0 <= 0:
        return None
    cum_drop = (c0 - c1) / c0 * 100.0
    min_drop = float(kw.get("min_cum_drop_pct", 2.0) or 2.0)
    max_drop = float(kw.get("max_cum_drop_pct", 18.0) or 18.0)
    if cum_drop < min_drop or cum_drop > max_drop:
        return None

    prior_lb = int(kw.get("prior_lookback", 20) or 20)
    min_prior = float(kw.get("min_prior_rise_pct", 0.0) or 0.0)
    if min_prior > 0 and start >= prior_lb:
        base = closes[start - prior_lb]
        if base > 0:
            prior_rise = (closes[start - 1] / base - 1.0) * 100.0
            if prior_rise < min_prior:
                return None
        else:
            prior_rise = 0.0
    else:
        prior_rise = 0.0
        if start >= prior_lb and closes[start - prior_lb] > 0:
            prior_rise = (closes[start - 1] / closes[start - prior_lb] - 1.0) * 100.0

    cur = rows[idx]
    cur_close = float(cur.close)
    cur_low = float(cur.low)

    if kw.get("require_above_ma120", True):
        ma120 = _ma_at(closes, idx, ma120_n)
        if np.isnan(ma120) or ma120 <= 0 or cur_close < ma120:
            return None
    else:
        ma120 = _ma_at(closes, idx, ma120_n)
        if np.isnan(ma120) or ma120 <= 0:
            ma120 = None

    ma_periods = tuple(int(x) for x in (kw.get("ma_periods") or (20, 60)))
    touch_pct = float(kw.get("ma_touch_pct", 8.0) or 8.0)
    ma_bonus, near_ma_period, near_ma_val = _near_ma_bonus(
        closes, cur_low, idx, ma_periods, touch_pct
    )

    v_start = float(vols[start])
    v_end = float(vols[idx])
    vol_ramp = v_end / v_start if v_start > 0 else 0.0
    vol_ratio = _vol_ratio_at(rows, idx, int(kw.get("vol_ma", 20) or 20))

    exec_i = idx + 1
    exec_date = ""
    exec_open = 0.0
    if exec_i < len(rows):
        exec_date = str(rows[exec_i].date)[:10]
        exec_open = float(rows[exec_i].open)

    return {
        "signal_date": str(cur.date)[:10],
        "streak_days": streak,
        "day1_date": day_metrics[0]["date"],
        "day2_date": day_metrics[1]["date"] if streak >= 2 else "",
        "day3_date": day_metrics[-1]["date"],
        "day1_vol": day_metrics[0]["volume"],
        "day2_vol": day_metrics[1]["volume"] if streak >= 2 else 0.0,
        "day3_vol": day_metrics[-1]["volume"],
        "vol_step1_ratio": day_metrics[1]["volume"] / day_metrics[0]["volume"]
        if streak >= 2 and day_metrics[0]["volume"] > 0
        else 0.0,
        "vol_step2_ratio": day_metrics[-1]["volume"] / day_metrics[1]["volume"]
        if streak >= 2 and day_metrics[1]["volume"] > 0
        else 0.0,
        "vol_ramp_total": vol_ramp,
        "cum_drop_pct": cum_drop,
        "prior_rise_pct": prior_rise,
        "vol_ratio": vol_ratio,
        "close": cur_close,
        "low": cur_low,
        "ma120": float(ma120) if ma120 is not None and not np.isnan(ma120) else None,
        "close_vs_ma120_pct": (cur_close / float(ma120) - 1.0) * 100.0
        if ma120 is not None and not np.isnan(ma120) and float(ma120) > 0
        else None,
        "near_ma_period": near_ma_period,
        "near_ma_val": near_ma_val,
        "near_ma_bonus": ma_bonus,
        "exec_buy_date": exec_date,
        "exec_buy_open": exec_open,
        "buy_mode": "next_open",
    }


def score_mode44_triple_yin_vol_rise(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> int:
    m = match_mode44_triple_yin_vol_rise(rows, idx, code, name, **kwargs)
    if not m:
        return 0
    kw = mode44_default_kw()
    kw.update({k: v for k, v in kwargs.items() if k in kw or k.startswith("mode44")})
    min_score = int(kw.get("min_score", 60) or 60)

    score = 62.0
    ramp = float(m.get("vol_ramp_total") or 0.0)
    score += min(15.0, max(0.0, (ramp - 1.0) * 40.0))

    step1 = float(m.get("vol_step1_ratio") or 0.0)
    step2 = float(m.get("vol_step2_ratio") or 0.0)
    score += min(8.0, max(0.0, (step1 - 1.0) * 25.0))
    score += min(10.0, max(0.0, (step2 - 1.0) * 30.0))

    drop = float(m.get("cum_drop_pct") or 0.0)
    if 3.0 <= drop <= 12.0:
        score += 6.0
    elif 2.0 <= drop < 3.0 or 12.0 < drop <= 15.0:
        score += 3.0

    prior = float(m.get("prior_rise_pct") or 0.0)
    if prior >= 15.0:
        score += 5.0
    elif prior >= 8.0:
        score += 3.0

    score += float(m.get("near_ma_bonus") or 0.0)

    vr = float(m.get("vol_ratio") or 0.0)
    if vr >= 1.5:
        score += min(6.0, (vr - 1.0) * 4.0)

    return int(max(min_score, min(100, round(score))))


def mode44_signal_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str = "",
    name: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    m = match_mode44_triple_yin_vol_rise(rows, idx, code, name, **kwargs)
    return dict(m) if m else {}


def dedupe_mode44_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """同股保留得分最高；同股同信号日去重。"""
    best: Dict[tuple, Dict[str, Any]] = {}
    for h in hits:
        key = (str(h.get("code", "")).zfill(6), str(h.get("signal_date", ""))[:10])
        prev = best.get(key)
        if prev is None or int(h.get("score", 0) or 0) > int(prev.get("score", 0) or 0):
            best[key] = h
    out = list(best.values())
    out.sort(key=lambda x: (str(x.get("signal_date", "")), -int(x.get("score", 0) or 0)))
    return out
