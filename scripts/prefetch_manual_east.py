import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import requests

from app.eastmoney import get_kline_cached, _market_from_code
from app.paths import GPT_DATA_DIR


def _load_codes(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    codes: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = str(row.get("code", "")).strip()
            if code:
                codes.append(code)
    return codes


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch Eastmoney kline for manual bull list.")
    parser.add_argument(
        "--manual-list",
        default="data/bull_manual_list.csv",
        help="Manual bull list CSV (code, name, date optional)",
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        default=None,
        help="直接指定股票代码，如 --codes 603496 000001",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"),
        help="Eastmoney cache dir (secid format)",
    )
    parser.add_argument("--count", type=int, default=1000, help="Kline rows per stock")
    parser.add_argument("--workers", type=int, default=6, help="Concurrency")
    parser.add_argument("--cache-days", type=int, default=3650, help="Cache max age days")
    parser.add_argument("--pause", type=float, default=0.0, help="Pause seconds between requests")
    args = parser.parse_args()

    if args.codes:
        codes = [str(c).strip().zfill(6) for c in args.codes]
    else:
        codes = _load_codes(args.manual_list)
    if not codes:
        raise RuntimeError("手工样本列表为空，无法预拉。")

    total = len(codes)
    print(f"手工样本数量: {total}")

    def worker(idx: int, code: str) -> Tuple[int, bool, str]:
        session = requests.Session()
        market = _market_from_code(code)
        secid = f"{market}.{code}"
        try:
            rows = get_kline_cached(
                secid,
                cache_dir=args.cache_dir,
                count=args.count,
                session=session,
                max_age_days=args.cache_days,
                pause=args.pause,
                local_only=False,
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
            if done % 10 == 0 or done == total:
                print(f"进度: {done}/{total} 成功: {ok} 失败: {fail}")
            if err and done % 25 == 0:
                print(f"示例错误: {err}")

    print("预拉完成")
    print(f"成功: {ok} 失败: {fail}")


if __name__ == "__main__":
    main()
