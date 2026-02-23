import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.eastmoney import StockItem, fetch_index_kline, fetch_stock_list, get_kline_cached
from app.scanner import ScanConfig, score_stock, percentile_ranks


def _parse_date(value: str) -> datetime.date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _filter_rows_by_date(rows, cutoff_date: Optional[datetime.date]):
    if cutoff_date is None:
        return rows
    filtered = []
    for row in rows:
        try:
            row_date = _parse_date(row.date)
        except Exception:
            continue
        if row_date <= cutoff_date:
            filtered.append(row)
    return filtered


def _return_10d(rows) -> Optional[float]:
    if len(rows) <= 10:
        return None
    base = rows[-11].close
    if base == 0:
        return None
    return (rows[-1].close - base) / base * 100


def _get_stock_item(stock_map: Dict[str, StockItem], code: str) -> StockItem:
    item = stock_map.get(code)
    if item:
        return item
    market = 1 if code.startswith("6") else 0
    return StockItem(code=code, name=code, market=market)


def _compute_percentiles(
    stock_list: List[StockItem],
    cutoff_date: Optional[datetime.date],
    cache_dir: str,
    count: int,
    cache_days: int,
    workers: int,
) -> Tuple[Dict[int, float], List[Optional[float]]]:
    returns_10d: List[Optional[float]] = [None] * len(stock_list)

    def worker(idx: int, item: StockItem) -> Tuple[int, Optional[float]]:
        rows = get_kline_cached(
            item.secid,
            cache_dir=cache_dir,
            count=count,
            max_age_days=cache_days,
            pause=0.0,
        )
        if not rows:
            return idx, None
        rows = _filter_rows_by_date(rows, cutoff_date)
        if len(rows) <= 10:
            return idx, None
        return idx, _return_10d(rows)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, idx, item) for idx, item in enumerate(stock_list)]
        for future in as_completed(futures):
            idx, value = future.result()
            returns_10d[idx] = value

    returns_clean = [r for r in returns_10d if r is not None]
    if not returns_clean:
        return {}, returns_10d

    full_values = [r if r is not None else float("-inf") for r in returns_10d]
    percentile_map = percentile_ranks(full_values)
    return percentile_map, returns_10d


def _print_result(label: str, result) -> None:
    print(f"\n{label}")
    print(f"代码: {result.code}  名称: {result.name}")
    print(f"得分(加权): {result.score}")
    print("分项得分:")
    print(f"  趋势: {result.metrics.get('score_trend')}")
    print(f"  量能: {result.metrics.get('score_volume')}")
    print(f"  突破: {result.metrics.get('score_breakout')}")
    print(f"  强度: {result.metrics.get('score_strength')}")
    print(f"  风险: {result.metrics.get('score_risk')}")
    print(f"原始总分: {result.metrics.get('score_raw')}")
    print("理由:", "；".join(result.reasons) if result.reasons else "无")
    print("关键指标:")
    print(f"  MA20: {result.metrics.get('ma20')}")
    print(f"  MA60: {result.metrics.get('ma60')}")
    print(f"  VOL5: {result.metrics.get('vol5')}")
    print(f"  VOL20: {result.metrics.get('vol20')}")
    print(f"  High20: {result.metrics.get('high20')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two stocks with the current scoring model.")
    parser.add_argument("--date", required=True, help="Cutoff date, e.g. 2025-12-31")
    parser.add_argument("--codes", nargs=2, required=True, help="Two stock codes, e.g. 002943 600592")
    parser.add_argument("--count", type=int, default=200, help="Kline count to fetch")
    parser.add_argument("--cache-days", type=int, default=3650, help="Cache max age days")
    parser.add_argument("--workers", type=int, default=8, help="Concurrency when building percentiles")
    parser.add_argument(
        "--full-universe",
        action="store_true",
        help="Compute 10-day return percentiles across all stocks (slow)",
    )
    args = parser.parse_args()

    cutoff_date = _parse_date(args.date)
    config = ScanConfig()

    stock_list = fetch_stock_list()
    stock_map = {item.code: item for item in stock_list}

    percentile_map: Dict[int, float] = {}
    code_to_index: Dict[str, int] = {item.code: idx for idx, item in enumerate(stock_list)}

    if args.full_universe:
        percentile_map, _ = _compute_percentiles(
            stock_list=stock_list,
            cutoff_date=cutoff_date,
            cache_dir="data/kline_cache",
            count=args.count,
            cache_days=args.cache_days,
            workers=args.workers,
        )

    index_rows = fetch_index_kline()
    index_rows = _filter_rows_by_date(index_rows, cutoff_date)
    index_return_10d = _return_10d(index_rows)

    for code in args.codes:
        item = _get_stock_item(stock_map, code)
        rows = get_kline_cached(
            item.secid,
            cache_dir="data/kline_cache",
            count=args.count,
            max_age_days=args.cache_days,
            pause=0.0,
        )
        if not rows:
            print(f"\n{code} 数据为空，无法评分。")
            continue
        rows = _filter_rows_by_date(rows, cutoff_date)
        if len(rows) < 80:
            print(f"\n{code} 数据不足 80 天，无法评分。")
            continue
        percentile = None
        if args.full_universe:
            idx = code_to_index.get(code)
            if idx is not None:
                percentile = percentile_map.get(idx)
        result = score_stock(
            item=item,
            rows=rows,
            index_return_10d=index_return_10d,
            return_percentile_10d=percentile,
            config=config,
        )
        if result is None:
            print(f"\n{code} 被过滤（可能是 ST 或数据不足）。")
            continue
        label = f"对比结果（截至 {args.date}）"
        _print_result(label, result)

    if not args.full_universe:
        print(
            "\n提示：未计算全市场10日涨幅分位数，因此“前10%”加分未生效。"
            "如需完整对比，可加 --full-universe。"
        )


if __name__ == "__main__":
    main()
