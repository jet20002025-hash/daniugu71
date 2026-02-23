#!/usr/bin/env python3
"""更新 tencent K线缓存，补充最新数据（含2月13日等）"""
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import requests

from app.eastmoney import stock_items_from_list_csv
from app.paths import GPT_DATA_DIR
from app import tencent


def main() -> None:
    parser = argparse.ArgumentParser(description="更新 tencent K线缓存，补充最新数据")
    parser.add_argument(
        "--stock-list",
        default=os.path.join(GPT_DATA_DIR, "stock_list.csv"),
        help="股票列表 CSV",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"),
        help="K线缓存目录",
    )
    parser.add_argument("--count", type=int, default=500, help="每只股票拉取K线条数（足够覆盖到最新）")
    parser.add_argument("--workers", type=int, default=8, help="并发数")
    parser.add_argument("--max-age-days", type=int, default=0, help="缓存最大天数，0=强制刷新")
    parser.add_argument("--pause", type=float, default=0.05, help="请求间隔秒数")
    parser.add_argument("--limit", type=int, default=0, help="限制更新数量，0=全部")
    args = parser.parse_args()

    stock_list = stock_items_from_list_csv(args.stock_list)
    if not stock_list:
        raise RuntimeError("股票列表为空")

    codes = [item.code for item in stock_list]
    if args.limit > 0:
        codes = codes[: args.limit]

    total = len(codes)
    print(f"待更新: {total} 只")
    print(f"缓存目录: {args.cache_dir}")
    print(f"强制刷新: max_age_days={args.max_age_days}")

    def worker(idx: int, code: str) -> Tuple[int, bool, str]:
        session = requests.Session()
        try:
            rows = tencent.get_kline_cached(
                code,
                cache_dir=args.cache_dir,
                count=args.count,
                session=session,
                max_age_days=args.max_age_days,
                pause=args.pause,
                prefer_local=False,
            )
            return idx, bool(rows), ""
        except Exception as exc:
            return idx, False, str(exc)

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, idx, code) for idx, code in enumerate(codes)]
        for future in as_completed(futures):
            idx, success, err = future.result()
            if success:
                ok += 1
            else:
                fail += 1
            done = ok + fail
            if done % 500 == 0 or done == total:
                print(f"进度: {done}/{total}  成功: {ok}  失败: {fail}")
            if err and fail <= 3:
                print(f"  错误示例: {err}")

    print("更新完成")
    print(f"成功: {ok}  失败: {fail}")


if __name__ == "__main__":
    main()
