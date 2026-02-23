import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

import requests

from app import sina, tencent
from app.eastmoney import fetch_stock_list, get_kline_cached


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch all stock Kline data into local cache.")
    parser.add_argument("--count", type=int, default=200, help="Kline rows per stock")
    parser.add_argument("--workers", type=int, default=8, help="Concurrency")
    parser.add_argument("--cache-days", type=int, default=3650, help="Cache max age days")
    parser.add_argument("--pause", type=float, default=0.0, help="Pause seconds between requests")
    parser.add_argument("--max-pages", type=int, default=200, help="Max pages for stock list")
    parser.add_argument("--page-size", type=int, default=100, help="Page size for stock list")
    parser.add_argument(
        "--provider",
        choices=["eastmoney", "tencent", "sina"],
        default="eastmoney",
        help="Data provider for online update",
    )
    args = parser.parse_args()

    session = requests.Session()
    stock_list = fetch_stock_list(session=session, page_size=args.page_size, max_pages=args.max_pages)
    total = len(stock_list)
    if total == 0:
        raise RuntimeError("未获取到股票列表，无法预拉K线。")

    print(f"股票数量: {total}")

    def worker(idx: int, item) -> Tuple[int, bool, Optional[str]]:
        s = requests.Session()
        try:
            if args.provider == "tencent":
                rows = tencent.get_kline_cached(
                    item.code,
                    cache_dir="data/kline_cache_tencent",
                    count=args.count,
                    session=s,
                    max_age_days=args.cache_days,
                    pause=args.pause,
                    prefer_local=False,
                )
            elif args.provider == "sina":
                rows = sina.get_kline_cached(
                    item.code,
                    cache_dir="data/kline_cache_sina",
                    count=args.count,
                    session=s,
                    max_age_days=args.cache_days,
                    pause=args.pause,
                    prefer_local=False,
                )
            else:
                rows = get_kline_cached(
                    item.secid,
                    cache_dir="data/kline_cache",
                    count=args.count,
                    session=s,
                    max_age_days=args.cache_days,
                    pause=args.pause,
                    local_only=False,
                )
            return idx, bool(rows), None
        except Exception as exc:
            return idx, False, str(exc)

    ok = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, idx, item) for idx, item in enumerate(stock_list)]
        for future in as_completed(futures):
            idx, success, err = future.result()
            if success:
                ok += 1
            else:
                fail += 1
            done = ok + fail
            if done % 200 == 0 or done == total:
                print(f"进度: {done}/{total} 成功: {ok} 失败: {fail}")
            if err and done % 500 == 0:
                print(f"示例错误: {err}")

    print("预拉完成")
    print(f"成功: {ok} 失败: {fail}")


if __name__ == "__main__":
    main()
