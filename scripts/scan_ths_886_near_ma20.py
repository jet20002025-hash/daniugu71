#!/usr/bin/env python3
"""
扫描同花顺 886 概念指数：最新日 K 收盘价在 MA20 附近。

判定：|close - MA20| / MA20 <= threshold（默认 ±2%）

输出：{GPT_DATA_DIR}/results/ths_886_near_ma20_{date}.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.paths import GPT_DATA_DIR
from app.ths_index_kline import ths_index_cache_dir


def _sma(closes: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(closes), np.nan, dtype=float)
    if len(closes) < window:
        return out
    w = np.ones(window, dtype=float) / float(window)
    out[window - 1 :] = np.convolve(closes, w, mode="valid")
    return out


def _load_meta(cache_dir: str) -> Dict[str, str]:
    path = os.path.join(cache_dir, "index_meta.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v or "") for k, v in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="886 指数日 K 在 MA20 附近")
    parser.add_argument(
        "--cache-dir",
        default=ths_index_cache_dir(GPT_DATA_DIR),
        help="886 指数 K 线目录",
    )
    parser.add_argument("--threshold", type=float, default=2.0, help="偏离 MA20 阈值(%)")
    parser.add_argument("--min-rows", type=int, default=25, help="最少 K 线根数")
    parser.add_argument("--out", default="", help="输出 CSV 路径")
    args = parser.parse_args()

    meta = _load_meta(args.cache_dir)
    hits: List[dict] = []

    for fn in sorted(os.listdir(args.cache_dir)):
        if not fn.endswith(".csv"):
            continue
        code = fn[:-4]
        path = os.path.join(args.cache_dir, fn)
        rows: List[dict] = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if len(rows) < args.min_rows:
            continue
        closes = np.array([float(r["close"]) for r in rows], dtype=float)
        ma20 = _sma(closes, 20)
        if np.isnan(ma20[-1]):
            continue
        close = closes[-1]
        ma = ma20[-1]
        dev = (close - ma) / ma * 100.0
        if abs(dev) > args.threshold:
            continue
        last = rows[-1]
        hits.append(
            {
                "code": code,
                "name": meta.get(code, ""),
                "date": last["date"][:10],
                "close": round(close, 2),
                "ma20": round(ma, 2),
                "dev_pct": round(dev, 2),
                "pct_chg": round(float(last.get("pct_chg") or 0), 2),
            }
        )

    hits.sort(key=lambda x: abs(x["dev_pct"]))

    out_path = args.out
    if not out_path:
        out_dir = os.path.join(GPT_DATA_DIR, "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"ths_886_near_ma20_{date.today():%Y_%m_%d}.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["code", "name", "date", "close", "ma20", "dev_pct", "pct_chg"],
        )
        writer.writeheader()
        writer.writerows(hits)

    print(f"阈值 ±{args.threshold}%  命中 {len(hits)} 只  -> {out_path}")
    for h in hits:
        print(
            f"{h['code']} {h['name']:18s} {h['date']} "
            f"close={h['close']} ma20={h['ma20']} dev={h['dev_pct']:+.2f}%"
        )


if __name__ == "__main__":
    main()
