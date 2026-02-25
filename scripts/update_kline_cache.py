#!/usr/bin/env python3
"""
本地批量更新 K 线缓存并写入统一目录（与在线更新一致，东财/腾讯/新浪/本地共用 code.csv）。
已有完整历史时只拉最近约 10 根（含今天）并与缓存合并，不重拉全量；新股票或缓存不足时仍拉 300 根。
默认 12 线程并发 + 0.1s 限速，全量更新约 5～15 分钟（原单线程约 1 小时+）。
用法：
  python scripts/update_kline_cache.py                  # 默认 12 线程，轮流尝试多源
  python scripts/update_kline_cache.py --workers 1     # 单线程（慢，兼容旧行为）
  python scripts/update_kline_cache.py --source eastmoney --delay 0.05  # 单源可适当减小 delay
  python scripts/update_kline_cache.py --source sina --delay 0.15       # 新浪易限流建议 0.15
  python scripts/update_kline_cache.py --force --limit 100
"""
import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.paths import GPT_DATA_DIR as _GPT_DATA_DIR

# 若环境变量是占位路径（如 /path/to/...），改用项目 data/gpt
if _GPT_DATA_DIR.startswith("/path"):
    GPT_DATA_DIR = os.path.join(ROOT, "data", "gpt")
else:
    GPT_DATA_DIR = _GPT_DATA_DIR
from app.eastmoney import (
    fetch_kline as eastmoney_fetch_kline,
    write_cached_kline,
    read_cached_kline,
    kline_is_fresh,
    stock_items_from_list_csv,
    list_cached_stocks_flat,
    load_stock_list_csv,
    KlineRow as EKlineRow,
)
from app import sina
from app import netease
from app import tencent

# 可选：AKShare（pip install akshare），聚合多源，接口全挂时可作备选
try:
    import akshare as ak
    _HAS_AKSHARE = True
except ImportError:
    _HAS_AKSHARE = False


def _akshare_fetch_kline(code: str, count: int) -> list:
    """用 AKShare 拉取日 K，返回与 eastmoney.KlineRow 兼容的列表。未装 akshare 或失败返回 []。"""
    if not _HAS_AKSHARE:
        return []
    import datetime as dt
    end = dt.date.today()
    start = end - dt.timedelta(days=max(count * 2, 365))
    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
    except Exception:
        return []
    if df is None or df.empty:
        return []

    def get(row, *names):
        for n in names:
            if n in row.index:
                try:
                    return float(row[n])
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def date_str(row):
        for key in ("日期", "date"):
            if key in row.index:
                d = row[key]
                if hasattr(d, "strftime"):
                    return d.strftime("%Y-%m-%d")
                s = str(d).replace("/", "-")
                if len(s) >= 10:
                    return s[:10]
                if len(s) == 8:
                    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
                return s
        return ""

    rows = []
    for _, r in df.iterrows():
        ds = date_str(r)
        if not ds:
            continue
        rows.append(
            EKlineRow(
                date=ds,
                open=get(r, "开盘", "open"),
                close=get(r, "收盘", "close"),
                high=get(r, "最高", "高", "high"),
                low=get(r, "最低", "低", "low"),
                volume=get(r, "成交量", "volume"),
                amount=get(r, "成交额", "amount"),
                amplitude=get(r, "振幅", "amplitude"),
                pct_chg=get(r, "涨跌幅", "pct_chg"),
                chg=get(r, "涨跌额", "chg"),
                turnover=get(r, "换手率", "turnover"),
            )
        )
    return rows


class _RateLimiter:
    """限制请求间隔，多线程下保证每两次请求开始时间至少间隔 interval 秒。"""
    def __init__(self, interval: float):
        self.interval = interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.time()
            wait_time = self._last + self.interval - now
            if wait_time > 0:
                time.sleep(wait_time)
            self._last = time.time()


def main():
    choices = ["auto", "eastmoney", "sina", "netease", "tencent"]
    if _HAS_AKSHARE:
        choices.append("akshare")
    parser = argparse.ArgumentParser(description="更新本地 K 线缓存，写入统一 code.csv")
    parser.add_argument("--source", choices=choices, default="auto",
                        help="数据源：auto=轮流尝试多源（默认），或单选其一")
    parser.add_argument("--force", action="store_true", help="强制刷新，忽略已有今日数据")
    parser.add_argument("--delay", type=float, default=0.1, help="每只股票请求间隔秒（多线程时为全局限速），建议 0.05～0.15")
    parser.add_argument("--workers", type=int, default=12, help="并发请求数，1=单线程，12 约可缩短到 5～15 分钟")
    parser.add_argument("--limit", type=int, default=0, help="最多更新 N 只，0 表示全部")
    parser.add_argument("--list", choices=["csv", "cache"], default="csv",
                        help="股票列表来源：csv=stock_list.csv，cache=已有缓存目录")
    args = parser.parse_args()

    cache_dir = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
    os.makedirs(cache_dir, exist_ok=True)

    if args.list == "csv":
        stock_list_path = os.path.join(GPT_DATA_DIR, "stock_list.csv")
        if not os.path.exists(stock_list_path):
            print(f"未找到 {stock_list_path}，改用 cache 列表")
            stock_list = list_cached_stocks_flat(cache_dir, name_map=load_stock_list_csv(stock_list_path))
        else:
            stock_list = stock_items_from_list_csv(stock_list_path)
    else:
        name_map = load_stock_list_csv(os.path.join(GPT_DATA_DIR, "stock_list.csv"))
        stock_list = list_cached_stocks_flat(cache_dir, name_map=name_map)

    if not stock_list:
        print("股票列表为空，请先准备 stock_list.csv 或缓存目录下已有 csv")
        return 1

    if args.limit:
        stock_list = stock_list[: args.limit]
    total = len(stock_list)
    updated = 0
    skipped = 0
    failed = 0
    session = requests.Session()
    session.trust_env = False
    use_auto = args.source == "auto"
    use_sina = args.source == "sina"
    use_netease = args.source == "netease"
    use_tencent = args.source == "tencent"
    use_akshare = args.source == "akshare"
    auto_sources = "新浪→网易→东财→腾讯" + ("→AKShare" if _HAS_AKSHARE else "")
    print(f"数据源: {'轮流尝试 ' + auto_sources if use_auto else '新浪' if use_sina else '网易' if use_netease else '东财' if args.source == 'eastmoney' else '腾讯' if use_tencent else 'AKShare'}（已有缓存时仅拉最近约 10 根并合并，不重拉全量）")
    print(f"并发: {args.workers} 线程，请求间隔 {args.delay}s")

    def fetch_one(item, count, _session=None):
        s = _session if _session is not None else session
        if use_auto:
            for name, fetcher in [
                ("新浪", lambda: sina.fetch_kline(item.code, count=count, session=s)),
                ("网易", lambda: netease.fetch_kline(item.code, count=count, session=s)),
                ("东财", lambda: eastmoney_fetch_kline(item.secid, count=count, session=s)),
                ("腾讯", lambda: tencent.fetch_kline(item.code, count=count, session=s)),
            ] + ([("AKShare", lambda: _akshare_fetch_kline(item.code, count))] if _HAS_AKSHARE else []):
                try:
                    if _session is None:
                        time.sleep(args.delay * 0.3)
                    rows = fetcher()
                    if rows:
                        return rows, name
                except Exception:
                    pass
            return None, None
        try:
            if use_sina:
                rows = sina.fetch_kline(item.code, count=count, session=s)
            elif use_netease:
                rows = netease.fetch_kline(item.code, count=count, session=s)
            elif use_tencent:
                rows = tencent.fetch_kline(item.code, count=count, session=s)
            elif use_akshare:
                rows = _akshare_fetch_kline(item.code, count)
            else:
                rows = eastmoney_fetch_kline(item.secid, count=count, session=s)
            return (rows, "单源") if rows else (None, None)
        except Exception:
            return None, None

    def merge_tail(cached, new_rows):
        """在已有缓存后追加仅比最后日期新的 K 线（只更新今天/最近几根）。"""
        if not cached or not new_rows:
            return new_rows if not cached else cached
        last_date = cached[-1].date[:10]
        extra = [r for r in new_rows if r.date[:10] > last_date]
        if not extra:
            return cached
        return cached + extra

    # 先扫一遍：分出需要更新的 vs 跳过的（不请求接口）
    to_update = []
    for i, item in enumerate(stock_list):
        path = os.path.join(cache_dir, f"{item.code}.csv")
        cached = read_cached_kline(path)
        if not args.force and cached and kline_is_fresh(cached):
            skipped += 1
        else:
            count = 10 if (cached and len(cached) >= 100) else 300
            to_update.append((item, path, cached, count))
        if (i + 1) % 1000 == 0:
            print(f"扫描进度 {i + 1}/{total}，待更新 {len(to_update)} 只，跳过 {skipped} 只")

    num_to_update = len(to_update)
    if num_to_update == 0:
        print(f"完成：共 {total} 只，全部已是最新，跳过 {skipped} 只，缓存目录 {cache_dir}")
        return 0

    if args.workers <= 1:
        # 单线程：按原逻辑逐只请求并 sleep
        for i, (item, path, cached, count) in enumerate(to_update):
            rows, _ = fetch_one(item, count)
            if rows:
                if cached and len(cached) >= 100:
                    rows = merge_tail(cached, rows)
                write_cached_kline(path, rows)
                updated += 1
            else:
                failed += 1
                if failed <= 3:
                    print(f"  {item.code} 多源均无数据或失败")
            if (i + 1) % 200 == 0:
                print(f"进度 {i + 1}/{num_to_update}，已更新 {updated} 只，失败 {failed} 只")
            time.sleep(args.delay)
    else:
        # 多线程：速率限制 + 每线程独立 session
        rate_limiter = _RateLimiter(args.delay)
        counter_lock = threading.Lock()

        def do_one(task):
            item, path, cached, count = task
            rate_limiter.wait()
            my_session = requests.Session()
            my_session.trust_env = False
            rows, _ = fetch_one(item, count, _session=my_session)
            if rows:
                if cached and len(cached) >= 100:
                    rows = merge_tail(cached, rows)
                write_cached_kline(path, rows)
                with counter_lock:
                    nonlocal updated
                    updated += 1
                return "ok"
            else:
                with counter_lock:
                    nonlocal failed
                    failed += 1
                    current_failed = failed
                if current_failed <= 5:
                    print(f"  {item.code} 多源均无数据或失败")
                return "fail"

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(do_one, t): t for t in to_update}
            done = 0
            for f in as_completed(futures):
                done += 1
                if done % 200 == 0 or done == num_to_update:
                    print(f"进度 {done}/{num_to_update}，已更新 {updated} 只，失败 {failed} 只")
    print(f"完成：共 {total} 只，更新 {updated} 只，跳过 {skipped} 只，失败 {failed} 只，缓存目录 {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
