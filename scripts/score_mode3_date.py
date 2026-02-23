#!/usr/bin/env python3
"""筛选指定日期 mode3 分数>阈值、市值<=上限的个股，输出分数。"""
import argparse
import csv
import os
from typing import Dict, List, Optional

import numpy as np

from app.eastmoney import stock_items_from_list_csv
from app.paths import GPT_DATA_DIR

from scripts.backtest_startup_modes import (
    _load_rows,
    _moving_mean,
    _score_mode3,
    _signals_mode3,
)


def _load_market_caps(path: str) -> Dict[str, float]:
    if not path or not os.path.exists(path):
        return {}
    mapping: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = str(row.get("code", "")).strip()
            if not code or len(code) < 6:
                code = code.zfill(6)
            cap_value = row.get("total_cap") or row.get("market_cap")
            try:
                cap = float(cap_value) if cap_value else 0.0
            except Exception:
                continue
            if cap > 0:
                mapping[code] = cap
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="筛选指定日期 mode3 高分个股（含市值过滤）")
    parser.add_argument(
        "--date",
        default="2026-02-13",
        help="信号日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=100,
        help="最低分数（>=此值，默认100）",
    )
    parser.add_argument(
        "--max-cap",
        type=float,
        default=150.0,
        help="市值上限（亿），默认150",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"),
        help="K线缓存目录",
    )
    parser.add_argument(
        "--stock-list",
        default=os.path.join(GPT_DATA_DIR, "stock_list.csv"),
        help="股票列表",
    )
    parser.add_argument(
        "--market-cap",
        default=os.path.join(GPT_DATA_DIR, "market_cap.csv"),
        help="市值缓存",
    )
    parser.add_argument(
        "--no-cap-filter",
        action="store_true",
        help="不启用市值过滤（用于调试）",
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出CSV路径（可选）",
    )
    args = parser.parse_args()

    target_date = args.date
    min_score = args.min_score
    cap_limit = args.max_cap * 1e8  # 亿 -> 元

    stock_list = stock_items_from_list_csv(args.stock_list)
    if not stock_list:
        raise RuntimeError("股票列表为空")

    market_caps = _load_market_caps(args.market_cap)
    cache_format = "code"
    if not os.path.exists(args.cache_dir):
        raise RuntimeError(f"缓存目录不存在: {args.cache_dir}")

    results: List[dict] = []
    for item in stock_list:
        rows = _load_rows(args.cache_dir, cache_format, item.market, item.code)
        if not rows or len(rows) < 80:
            continue
        dates = [r.date for r in rows]
        close = np.array([r.close for r in rows], dtype=float)
        open_ = np.array([r.open for r in rows], dtype=float)
        high = np.array([r.high for r in rows], dtype=float)
        low = np.array([r.low for r in rows], dtype=float)
        volume = np.array([r.volume for r in rows], dtype=float)
        ma10 = _moving_mean(close, 10)
        ma20 = _moving_mean(close, 20)
        ma60 = _moving_mean(close, 60)
        vol20 = _moving_mean(volume, 20)
        ret20 = [None] * len(rows)
        for i in range(20, len(rows)):
            base = close[i - 20]
            ret20[i] = (close[i] - base) / base * 100 if base else None

        signals = _signals_mode3(
            rows, dates, close, open_, high, low, volume,
            ma10, ma20, ma60, vol20, ret20,
            target_date, target_date,
        )
        for idx in signals:
            if dates[idx] != target_date:
                continue
            score = _score_mode3(close, volume, ma10, ma20, ma60, vol20, idx)
            if score < min_score:
                continue
            code = item.code
            code_norm = code.zfill(6) if len(code) < 6 else code
            cap_value = market_caps.get(code) or market_caps.get(code_norm)
            if not args.no_cap_filter:
                if cap_value is None:
                    continue  # 无市值数据则排除
                if cap_value > cap_limit:
                    continue
            cap_yi = cap_value / 1e8 if cap_value is not None else None
            # 同分排名用：放量比、贴近MA20、均线间距、20日涨幅、代码
            vol_ratio = volume[idx] / vol20[idx] if vol20[idx] > 0 else 0.0
            ma20_now, ma60_now = ma20[idx], ma60[idx]
            close_gap = (close[idx] - ma20_now) / ma20_now if ma20_now > 0 else 0.0
            ma20_gap = (ma10[idx] - ma20_now) / ma20_now if ma20_now > 0 else 0.0
            ma60_gap = (ma20_now - ma60_now) / ma60_now if ma60_now > 0 else 0.0
            ret20_val = ret20[idx] if ret20[idx] is not None else 0.0
            results.append({
                "date": target_date,
                "code": code,
                "name": item.name,
                "score": score,
                "market_cap_yi": round(cap_yi, 2) if cap_yi is not None else None,
                "close": round(close[idx], 2),
                "_vol_ratio": vol_ratio,
                "_close_gap": close_gap,
                "_ma20_gap": ma20_gap,
                "_ma60_gap": ma60_gap,
                "_ret20": ret20_val,
            })

    # 71倍模型同分排名：分数降序 → 贴近MA20 → 放量大 → 均线间距大 → 20日涨幅小 → 代码
    results.sort(
        key=lambda x: (
            -x["score"],
            x["_close_gap"],
            -x["_vol_ratio"],
            -(x["_ma20_gap"] + x["_ma60_gap"]),
            x["_ret20"],
            x["code"],
        )
    )

    print(f"\n{target_date} mode3 分数≥{min_score} 且市值≤{args.max_cap}亿 个股（共 {len(results)} 只）\n")
    print(f"{'代码':<8} {'名称':<12} {'分数':<6} {'市值(亿)':<10} {'收盘价'}")
    print("-" * 50)
    for r in results:
        cap_str = f"{r['market_cap_yi']:.2f}" if r["market_cap_yi"] is not None else "N/A"
        print(f"{r['code']:<8} {r['name']:<12} {r['score']:<6} {cap_str:<10} {r['close']}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_cols = ["date", "code", "name", "score", "market_cap_yi", "close"]
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\n已保存到 {args.output}")


if __name__ == "__main__":
    main()
