#!/usr/bin/env python3
"""2 月份以来每个交易日的 mode3 得分 100 分的 top1 个股，输出到 CSV。"""
import argparse
import csv
import os
from datetime import date, timedelta

from app.eastmoney import stock_items_from_list_csv
from app.paths import GPT_DATA_DIR

from scripts.score_mode3_date import _load_market_caps, get_results_for_date


def _trading_days(start: date, end: date):
    """生成 start 到 end 之间的交易日（简单按周一至周五）。"""
    d = start
    while d <= end:
        if d.weekday() < 5:  # 0=Mon .. 4=Fri
            yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2 月以来每日 mode3 得分 100 的 top1 个股"
    )
    parser.add_argument(
        "--start",
        default="2026-02-01",
        help="起始日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="结束日期 YYYY-MM-DD，默认今天",
    )
    parser.add_argument(
        "--max-cap",
        type=float,
        default=150.0,
        help="市值上限（亿），默认 150",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"),
        help="K 线缓存目录",
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
        help="不启用市值过滤",
    )
    parser.add_argument(
        "--output",
        default="data/results/mode3_feb_daily_top1_100.csv",
        help="输出 CSV 路径",
    )
    args = parser.parse_args()

    try:
        end_date = date.today() if args.end is None else date.fromisoformat(args.end)
    except ValueError:
        end_date = date.today()
    start_date = date.fromisoformat(args.start)

    stock_list = stock_items_from_list_csv(args.stock_list)
    if not stock_list:
        raise RuntimeError("股票列表为空")
    market_caps = _load_market_caps(args.market_cap)
    if not os.path.exists(args.cache_dir):
        raise RuntimeError(f"缓存目录不存在: {args.cache_dir}")

    out_cols = ["date", "code", "name", "score", "market_cap_yi", "close"]
    rows: list = []

    for day in _trading_days(start_date, end_date):
        results = get_results_for_date(
            day,
            stock_list,
            market_caps,
            args.cache_dir,
            min_score=100,
            max_cap_yi=args.max_cap,
            no_cap_filter=args.no_cap_filter,
        )
        score_100 = [r for r in results if r["score"] == 100]
        if not score_100:
            continue
        top1 = score_100[0]
        rows.append({
            "date": top1["date"],
            "code": top1["code"],
            "name": top1["name"],
            "score": top1["score"],
            "market_cap_yi": top1.get("market_cap_yi"),
            "close": top1["close"],
        })
        print(f"{day} top1 100分: {top1['code']} {top1['name']} 收盘 {top1['close']}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n共 {len(rows)} 个交易日有 100 分 top1，已保存到 {args.output}")


if __name__ == "__main__":
    main()
