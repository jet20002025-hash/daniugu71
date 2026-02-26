"""个股评分：输入代码，输出 mode9 评分（当前主模型，由 71 倍/mode3 升级）"""
import csv
import os
from typing import Dict, Optional

import numpy as np

from .eastmoney import (
    _market_from_code,
    read_cached_kline_by_code,
    read_cached_kline_by_market_code,
)
from .paths import GPT_DATA_DIR
from .scanner import _mode3_signals, _moving_mean, _score_mode9


def _load_kline(code: str, cache_dir: str, use_secid: bool) -> tuple:
    code = str(code).strip().zfill(6)
    market = _market_from_code(code)
    if use_secid:
        rows = read_cached_kline_by_market_code(cache_dir, market, code)
    else:
        rows = read_cached_kline_by_code(cache_dir, code)
    name = code
    path = os.path.join(GPT_DATA_DIR, "stock_list.csv")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("code", "")).strip().zfill(6) == code:
                    name = str(row.get("name", code)).strip() or code
                    break
    return (rows or [], name)


def _breakdown(rows, idx, ma10, ma20, ma60, vol20) -> dict:
    close = rows[idx].close
    volume = rows[idx].volume
    ma20_now = ma20[idx]
    ma60_now = ma60[idx]
    ma10_now = ma10[idx]
    vol20_now = vol20[idx]
    b = {"base": 60}
    if ma20_now > 0:
        gap = (ma10_now - ma20_now) / ma20_now
        b["ma10_ma20"] = 10 if gap >= 0.02 else (6 if gap >= 0.01 else (3 if gap >= 0.005 else 0))
    if ma60_now > 0:
        gap = (ma20_now - ma60_now) / ma60_now
        b["ma20_ma60"] = 10 if gap >= 0.02 else (6 if gap >= 0.01 else (3 if gap >= 0.005 else 0))
    if vol20_now > 0:
        vr = volume / vol20_now
        b["vol_ratio"] = 15 if vr >= 1.6 else (10 if vr >= 1.4 else (6 if vr >= 1.2 else 0))
    if ma20_now > 0:
        cg = (close - ma20_now) / ma20_now
        b["close_gap"] = 5 if cg >= 0.03 else (3 if cg >= 0.01 else 0)
    if idx >= 3 and rows[idx - 3].close > 0:
        ret3 = (close - rows[idx - 3].close) / rows[idx - 3].close * 100
        b["ret3_penalty"] = -10 if ret3 > 20 else (-5 if ret3 > 15 else 0)
    else:
        b["ret3_penalty"] = 0
    return b


def score_stock(
    code: str,
    cutoff_date: Optional[str] = None,
    cache_dir: Optional[str] = None,
    use_secid: bool = True,
) -> Dict:
    code = str(code).strip().zfill(6)
    if cache_dir is None:
        cache_dir = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
    if use_secid:
        use_secid = False

    rows, name = _load_kline(code, cache_dir, use_secid)
    if not rows or len(rows) < 60:
        return {"code": code, "name": name, "score": None, "has_signal": False, "reason": "K线数据不足或未找到缓存", "breakdown": {}}

    close = np.array([r.close for r in rows], dtype=float)
    volume = np.array([r.volume for r in rows], dtype=float)
    ma10 = _moving_mean(close, 10)
    ma20 = _moving_mean(close, 20)
    ma60 = _moving_mean(close, 60)
    vol20 = _moving_mean(volume, 20)

    signals = _mode3_signals(rows, None, cutoff_date)
    if cutoff_date:
        signals = [i for i in signals if rows[i].date <= cutoff_date]

    if not signals:
        idx = len(rows) - 1
        raw_score = _score_mode9(rows, idx, ma10, ma20, ma60, vol20)
        breakdown = _breakdown(rows, idx, ma10, ma20, ma60, vol20)
        reasons = []
        if np.isnan(ma10[idx]) or np.isnan(ma20[idx]) or np.isnan(ma60[idx]):
            reasons.append("均线数据不足")
        elif not (ma10[idx] > ma20[idx] > ma60[idx]):
            reasons.append("均线未多头排列(MA10>MA20>MA60)")
        elif close[idx] < ma20[idx]:
            reasons.append("收盘价低于MA20")
        elif volume[idx] < vol20[idx] * 1.2:
            reasons.append(f"放量不足(需≥1.2x, 当前{volume[idx]/vol20[idx]:.2f}x)")
        elif idx >= 20 and close[idx - 20] > 0:
            ret20 = (close[idx] - close[idx - 20]) / close[idx - 20] * 100
            if ret20 > 25:
                reasons.append(f"20日涨幅超25%({ret20:.1f}%)")
        else:
            reasons.append("未满足启动点条件")
        return {"code": code, "name": name, "score": raw_score, "has_signal": False, "reason": "; ".join(reasons), "breakdown": breakdown, "latest_date": rows[idx].date}

    idx = signals[-1]
    year_lookback = 240
    year_high_low_ratio_limit = 4.0
    if idx >= year_lookback - 1:
        start = idx - year_lookback + 1
        max_high = max(r.high for r in rows[start : idx + 1])
        min_low = min(r.low for r in rows[start : idx + 1])
        if min_low > 0 and max_high / min_low >= year_high_low_ratio_limit:
            return {
                "code": code,
                "name": name,
                "score": _score_mode9(rows, idx, ma10, ma20, ma60, vol20),
                "has_signal": False,
                "reason": f"近一年最高/最低倍数{max_high/min_low:.2f}x超4倍，已排除",
                "breakdown": _breakdown(rows, idx, ma10, ma20, ma60, vol20),
                "latest_date": rows[idx].date,
            }
    score = _score_mode9(rows, idx, ma10, ma20, ma60, vol20)
    signal_date = rows[idx].date
    buy_idx = min(idx + 1, len(rows) - 1)
    buy_date = rows[buy_idx].date
    vol_ratio = volume[idx] / vol20[idx] if vol20[idx] > 0 else 0
    ma20_gap = (ma10[idx] - ma20[idx]) / ma20[idx] if ma20[idx] > 0 else 0
    ma60_gap = (ma20[idx] - ma60[idx]) / ma60[idx] if ma60[idx] > 0 else 0
    close_gap = (close[idx] - ma20[idx]) / ma20[idx] if ma20[idx] > 0 else 0
    ret20 = (close[idx] - close[idx - 20]) / close[idx - 20] * 100 if idx >= 20 and close[idx - 20] > 0 else None
    return {
        "code": code,
        "name": name,
        "score": score,
        "has_signal": True,
        "signal_date": signal_date,
        "buy_date": buy_date,
        "reason": "符合mode9启动点",
        "breakdown": _breakdown(rows, idx, ma10, ma20, ma60, vol20),
        "vol_ratio": round(vol_ratio, 2),
        "ma20_gap_pct": round(ma20_gap * 100, 2),
        "ma60_gap_pct": round(ma60_gap * 100, 2),
        "close_gap_pct": round(close_gap * 100, 2),
        "ret20_pct": round(ret20, 2) if ret20 is not None else None,
    }
