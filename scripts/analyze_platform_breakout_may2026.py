#!/usr/bin/env python3
"""对比 2026-05 mode平台突破首阳 牛股 vs 差股特征，输出过滤建议。"""
from __future__ import annotations

import os
import sys
from statistics import mean, median

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.eastmoney import (
    list_cached_stocks_flat,
    load_stock_list_csv,
    read_cached_kline_by_code,
)
from app.paths import GPT_DATA_DIR
from app.scanner import _is_st, _match_mode_platform_breakout_first_yang, _moving_mean

CACHE = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")

WINNERS = {
    ("300983", "2026-05-06"),
    ("301502", "2026-05-06"),
    ("000026", "2026-05-08"),
    ("688697", "2026-05-06"),
    ("300259", "2026-05-13"),
    ("002421", "2026-05-15"),
    ("002979", "2026-05-15"),
    ("688010", "2026-05-18"),
}
LOSER_CODES = {"603019", "002998", "002458", "603515", "000404", "603050"}

FEAT_KEYS = [
    "pre_rise5_pct",
    "pre20_pct",
    "pre10_pct",
    "upper_ratio",
    "close_break60",
    "body_ratio",
    "gap_open_pct",
    "dd_from_peak_pct",
    "rise_to_peak_pct",
    "consolid_amp5_pct",
    "above_ma20",
    "ma60_slope5_pct",
    "phase_days",
    "rise_from_low_pct",
    "consolid_amp_pct",
    "breakout_pct",
    "high100_ratio",
    "vol_ratio",
    "pct_chg",
    "wash_big_yang_cnt",
    "ret20",
    "ret40",
    "ret60",
]


def fwd_ret(rows, idx: int, days: int):
    if idx + days >= len(rows):
        return None
    c0 = float(rows[idx].close)
    c1 = float(rows[idx + days].close)
    return (c1 - c0) / c0 * 100 if c0 > 0 else None


def extra_feats(rows, idx: int, m: dict) -> dict:
    close = np.array([float(r.close) for r in rows], dtype=float)
    high = np.array([float(r.high) for r in rows], dtype=float)
    low = np.array([float(r.low) for r in rows], dtype=float)
    ma20 = _moving_mean(close, 20)
    ma60 = _moving_mean(close, 60)
    r = rows[idx]
    i_low = int(m["low_date_idx"])

    seg_h = high[i_low:idx]
    seg_l = low[i_low:idx]
    peak_i = i_low + int(np.argmax(seg_h))
    peak_h = float(high[peak_i])
    trough_after = float(np.min(seg_l[peak_i - i_low :])) if peak_i < idx else float(seg_l[-1])
    dd_from_peak = (peak_h - trough_after) / peak_h * 100 if peak_h > 0 else 0.0

    pre20 = (
        (close[idx - 1] - close[idx - 21]) / close[idx - 21] * 100
        if idx >= 21 and close[idx - 21] > 0
        else None
    )
    pre10 = (
        (close[idx - 1] - close[idx - 11]) / close[idx - 11] * 100
        if idx >= 11 and close[idx - 11] > 0
        else None
    )
    above_ma20 = (
        close[idx] / ma20[idx]
        if idx < len(ma20) and not np.isnan(ma20[idx]) and ma20[idx] > 0
        else None
    )
    ma60_up = None
    if idx >= 65 and not np.isnan(ma60[idx]) and not np.isnan(ma60[idx - 5]) and ma60[idx - 5] > 0:
        ma60_up = (ma60[idx] - ma60[idx - 5]) / ma60[idx - 5] * 100

    gap_pct = (
        (float(r.open) - close[idx - 1]) / close[idx - 1] * 100
        if idx >= 1 and close[idx - 1] > 0
        else None
    )
    rise_to_peak = (
        (peak_h - float(low[i_low])) / float(low[i_low]) * 100 if float(low[i_low]) > 0 else 0.0
    )
    if idx >= 5:
        seg5 = rows[idx - 5 : idx]
        m5 = float(np.mean([float(x.close) for x in seg5]))
        amp5 = (
            (max(float(x.high) for x in seg5) - min(float(x.low) for x in seg5)) / m5 * 100
            if m5 > 0
            else None
        )
    else:
        amp5 = None

    return {
        "pre_rise5_pct": float(m.get("pre_rise5_pct", 0)),
        "upper_ratio": float(m.get("upper_ratio", 0)) * 100,
        "close_break60": float(m.get("close_break60", 0)),
        "body_ratio": float(m.get("body_ratio", 0)),
        "pre20_pct": pre20,
        "pre10_pct": pre10,
        "dd_from_peak_pct": dd_from_peak,
        "rise_to_peak_pct": rise_to_peak,
        "gap_open_pct": gap_pct,
        "above_ma20": above_ma20,
        "ma60_slope5_pct": ma60_up,
        "consolid_amp5_pct": amp5,
        "phase_days": float(m["phase_days"]),
        "rise_from_low_pct": float(m["rise_from_low_pct"]),
        "consolid_amp_pct": float(m["consolid_amp_pct"]),
        "breakout_pct": float(m["breakout_pct"]),
        "high100_ratio": float(m["high100_ratio"]),
        "vol_ratio": float(m["vol_ratio"]),
        "pct_chg": float(m["pct_chg"]),
        "wash_big_yang_cnt": float(m["wash_big_yang_cnt"]),
        "ret20": fwd_ret(rows, idx, 20),
        "ret40": fwd_ret(rows, idx, 40),
        "ret60": fwd_ret(rows, idx, 60),
    }


def scan_may_hits():
    name_map = load_stock_list_csv(os.path.join(GPT_DATA_DIR, "stock_list.csv"))
    hits = []
    for item in list_cached_stocks_flat(CACHE, name_map=name_map):
        if _is_st(item.name or ""):
            continue
        rows = read_cached_kline_by_code(CACHE, item.code)
        if not rows or len(rows) < 150:
            continue
        code = item.code.zfill(6)
        for i, r in enumerate(rows):
            d = str(r.date)[:10]
            if not ("2026-05-01" <= d <= "2026-05-31"):
                continue
            m = _match_mode_platform_breakout_first_yang(rows, i, code, item.name or "")
            if m is None:
                continue
            hits.append((d, code, (item.name or "").strip(), extra_feats(rows, i, m)))
    return hits


def label_hit(d: str, code: str, loser_dates: dict) -> str:
    if (code, d) in WINNERS:
        return "win"
    if code in LOSER_CODES and loser_dates.get(code) == d:
        return "lose"
    return "other"


def summarize(group: list[dict], key: str):
    vals = [g[key] for g in group if g.get(key) is not None]
    if not vals:
        return None
    return mean(vals), min(vals), max(vals), median(vals)


def main():
    hits = scan_may_hits()
    loser_dates = {}
    for d, code, _, _ in hits:
        if code in LOSER_CODES:
            loser_dates[code] = d

    print(f"5月命中 {len(hits)} 条\n差股信号日:")
    for c in sorted(LOSER_CODES):
        print(f"  {c} {loser_dates.get(c, '未命中')}")

    labeled = []
    for d, code, name, f in hits:
        lab = label_hit(d, code, loser_dates)
        if lab in ("win", "lose"):
            labeled.append((lab, d, code, name, f))

    print(f"\n标注: 牛股 {sum(1 for x in labeled if x[0]=='win')}  差股 {sum(1 for x in labeled if x[0]=='lose')}")

    print("\n代码      名称        信号日      ret20  ret40  ret60  前5日% 前20日% 上影% 收盘/60 震仓回撤% 量比")
    print("-" * 95)
    for lab, d, code, name, f in sorted(labeled, key=lambda x: (x[0], x[1])):
        print(
            f"{lab:4s} {code} {(name or '')[:8]:8s} {d} "
            f"{f.get('ret20') or 0:6.1f} {f.get('ret40') or 0:6.1f} {f.get('ret60') or 0:6.1f} "
            f"{f['pre_rise5_pct']:5.1f} {f.get('pre20_pct') or 0:6.1f} {f['upper_ratio']:4.1f} "
            f"{f['close_break60']:5.3f} {f['dd_from_peak_pct']:6.1f} {f['vol_ratio']:4.1f}"
        )

    print("\n=== 特征均值对比 (win vs lose) ===")
    diffs = []
    for k in FEAT_KEYS:
        w = [x[4][k] for x in labeled if x[0] == "win" and x[4].get(k) is not None]
        l = [x[4][k] for x in labeled if x[0] == "lose" and x[4].get(k) is not None]
        if not w or not l:
            continue
        wm, lm = mean(w), mean(l)
        diffs.append((abs(wm - lm), k, wm, lm, min(w), max(w), min(l), max(l)))
    diffs.sort(reverse=True)
    for _, k, wm, lm, wmin, wmax, lmin, lmax in diffs[:15]:
        print(
            f"  {k:22s} win={wm:7.2f} [{wmin:6.1f},{wmax:6.1f}]  "
            f"lose={lm:7.2f} [{lmin:6.1f},{lmax:6.1f}]  Δ={wm-lm:+.2f}"
        )

    # 全月 other 收益
    for lab in ("win", "lose", "other"):
        sub = [f for d, c, n, f in hits if label_hit(d, c, loser_dates) == lab]
        for rd in (20, 40, 60):
            vals = [s[f"ret{rd}"] for s in sub if s.get(f"ret{rd}") is not None]
            if vals:
                print(f"{lab} ret{rd}: mean={mean(vals):.1f}% median={median(vals):.1f}% n={len(vals)}")

    # 试探过滤：哪些阈值能保留全部win、去掉部分lose
    print("\n=== 单条件过滤试探 (保留win去掉lose) ===")
    win_feats = [x[4] for x in labeled if x[0] == "win"]
    lose_feats = [x[4] for x in labeled if x[0] == "lose"]

    rules = [
        ("pre_rise5_pct <= 12", lambda f: f["pre_rise5_pct"] <= 12),
        ("pre_rise5_pct <= 10", lambda f: f["pre_rise5_pct"] <= 10),
        ("pre20_pct <= 15", lambda f: (f.get("pre20_pct") or 0) <= 15),
        ("pre20_pct <= 12", lambda f: (f.get("pre20_pct") or 0) <= 12),
        ("upper_ratio <= 25", lambda f: f["upper_ratio"] <= 25),
        ("upper_ratio <= 20", lambda f: f["upper_ratio"] <= 20),
        ("close_break60 >= 1.0", lambda f: f["close_break60"] >= 1.0),
        ("close_break60 >= 0.99", lambda f: f["close_break60"] >= 0.99),
        ("dd_from_peak_pct >= 8", lambda f: f["dd_from_peak_pct"] >= 8),
        ("dd_from_peak_pct >= 10", lambda f: f["dd_from_peak_pct"] >= 10),
        ("vol_ratio <= 3", lambda f: f["vol_ratio"] <= 3),
        ("vol_ratio <= 2.5", lambda f: f["vol_ratio"] <= 2.5),
        ("vol_ratio >= 1.5", lambda f: f["vol_ratio"] >= 1.5),
        ("rise_from_low_pct <= 45", lambda f: f["rise_from_low_pct"] <= 45),
        ("rise_from_low_pct <= 40", lambda f: f["rise_from_low_pct"] <= 40),
        ("wash_big_yang_cnt <= 4", lambda f: f["wash_big_yang_cnt"] <= 4),
        ("gap_open_pct <= 3", lambda f: (f.get("gap_open_pct") or 0) <= 3),
        ("gap_open_pct <= 5", lambda f: (f.get("gap_open_pct") or 0) <= 5),
        ("pre_rise5<=12 & close_br60>=0.99", lambda f: f["pre_rise5_pct"] <= 12 and f["close_break60"] >= 0.99),
        ("pre_rise5<=10 & vol<=3", lambda f: f["pre_rise5_pct"] <= 10 and f["vol_ratio"] <= 3),
        ("pre20<=15 & dd_peak>=8", lambda f: (f.get("pre20_pct") or 0) <= 15 and f["dd_from_peak_pct"] >= 8),
    ]
    for name, fn in rules:
        w_ok = sum(1 for f in win_feats if fn(f))
        l_ok = sum(1 for f in lose_feats if fn(f))
        print(f"  {name:35s}  win {w_ok}/{len(win_feats)}  lose {l_ok}/{len(lose_feats)}  drop_lose={len(lose_feats)-l_ok}")


if __name__ == "__main__":
    main()
