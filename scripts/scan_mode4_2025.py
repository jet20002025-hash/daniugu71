#!/usr/bin/env python3
"""
生成 mode4 选股 CSV（2025年），供回测使用
"""
import argparse
import csv
import os

from app.eastmoney import read_cached_kline_by_market_code, stock_items_from_list_csv
from app.paths import GPT_DATA_DIR
from app.scanner import ScanConfig, scan_with_mode3


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 mode4 2025 选股 CSV")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"),
        help="与 mode3 71倍模型一致，使用 east 缓存",
    )
    parser.add_argument(
        "--stock-list",
        default=os.path.join(GPT_DATA_DIR, "stock_list.csv"),
    )
    parser.add_argument(
        "--output",
        default="data/results/mode4_2025_top3.csv",
    )
    parser.add_argument("--min-score", type=int, default=70)
    args = parser.parse_args()

    stock_list = stock_items_from_list_csv(args.stock_list)
    if not stock_list:
        raise RuntimeError("股票列表为空")

    config = ScanConfig(min_score=args.min_score, max_results=500)
    kline_loader = lambda item: read_cached_kline_by_market_code(args.cache_dir, item.market, item.code)

    results = scan_with_mode3(
        stock_list=stock_list,
        config=config,
        cache_dir=args.cache_dir,
        kline_loader=kline_loader,
        prefer_local=True,
        cutoff_date=args.end_date,
        start_date=args.start_date,
        mode4_filters=True,
    )

    out_rows = []
    for r in results:
        sig = (r.metrics or {}).get("signal_date")
        buy = (r.metrics or {}).get("buy_date")
        if sig and buy:
            out_rows.append({
                "date": sig,
                "code": r.code,
                "name": r.name,
                "mode": "mode4",
                "buy_date": buy,
                "multiple": 0,
                "label": 0,
                "score": r.score,
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "code", "name", "mode", "buy_date", "multiple", "label", "score"],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"mode4 2025 选股: {len(out_rows)} 条，输出 {args.output}")


if __name__ == "__main__":
    main()
