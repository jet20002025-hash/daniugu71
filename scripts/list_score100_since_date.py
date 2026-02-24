#!/usr/bin/env python3
"""基于最新 71 倍模型，列出指定起始日期以来所有满分 100 分个股，附信号日期。"""
import argparse
import csv
import os
from datetime import date, timedelta

from app.eastmoney import stock_items_from_list_csv
from app.paths import GPT_DATA_DIR

from scripts.score_mode3_date import _load_market_caps, get_results_for_date


def _trading_days(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2月7日以来所有 mode3 满分100分个股清单（附信号日期）"
    )
    parser.add_argument("--start", default="2026-02-07", help="起始信号日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期，默认今天")
    parser.add_argument("--max-cap", type=float, default=150.0, help="市值上限(亿)，默认150；0表示不限制")
    parser.add_argument("--cache-dir", default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"))
    parser.add_argument("--stock-list", default=os.path.join(GPT_DATA_DIR, "stock_list.csv"))
    parser.add_argument("--market-cap", default=os.path.join(GPT_DATA_DIR, "market_cap.csv"))
    parser.add_argument("--no-cap-filter", action="store_true", help="不启用市值过滤")
    parser.add_argument(
        "--output",
        default="data/results/mode3_score100_since_0207.csv",
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--output-xlsx",
        default="data/results/mode3_score100_since_0207.xlsx",
        help="同时输出 Excel 路径，设为空则不输出",
    )
    args = parser.parse_args()

    end_date = date.today() if args.end is None else date.fromisoformat(args.end)
    start_date = date.fromisoformat(args.start)
    max_cap = args.max_cap if args.max_cap > 0 else 0.0
    no_cap = args.no_cap_filter or (max_cap <= 0)

    stock_list = stock_items_from_list_csv(args.stock_list)
    if not stock_list:
        raise RuntimeError("股票列表为空")
    market_caps = _load_market_caps(args.market_cap)
    if not os.path.exists(args.cache_dir):
        raise RuntimeError(f"缓存目录不存在: {args.cache_dir}")

    rows: list = []
    for day in _trading_days(start_date, end_date):
        results = get_results_for_date(
            day,
            stock_list,
            market_caps,
            args.cache_dir,
            min_score=100,
            max_cap_yi=max_cap if max_cap > 0 else 99999.0,
            no_cap_filter=no_cap,
        )
        for r in results:
            if r["score"] != 100:
                continue
            rows.append({
                "信号日期": r["date"],
                "代码": r["code"],
                "名称": r["name"],
                "分数": r["score"],
                "市值亿": r.get("market_cap_yi"),
                "收盘价": r["close"],
            })
        if results and any(x["score"] == 100 for x in results):
            n100 = sum(1 for x in results if x["score"] == 100)
            print(f"{day} 满分100分: {n100} 只")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["信号日期", "代码", "名称", "分数", "市值亿", "收盘价"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n共 {len(rows)} 条满分100分记录，已保存: {args.output}")

    if args.output_xlsx and os.path.dirname(args.output_xlsx):
        os.makedirs(os.path.dirname(args.output_xlsx), exist_ok=True)
    if args.output_xlsx:
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_excel(args.output_xlsx, index=False)
            print(f"Excel 已保存: {args.output_xlsx}")
        except Exception as e:
            print(f"导出 Excel 失败: {e}")


if __name__ == "__main__":
    main()
