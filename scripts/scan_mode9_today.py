#!/usr/bin/env python3
"""基于改进 mode9（与网站 run_mode3_scan 一致）筛选指定日期的个股，按得分降序输出。

改进项与网站相同：MODE9_HOT_INDUSTRY_*、SECTOR_AK_CACHE_DIR、
SECTOR_FUND_FLOW_* 等环境变量会作用于扫描；scanner 要求 len(rows) >= year_lookback+5（默认 245），
故必须提供至少约 260 根 K 线，用 tencent.get_kline_cached(count=300) 拉取。

用法:
  python3 scripts/scan_mode9_today.py                      # 今日，min_score=80（默认）
  python3 scripts/scan_mode9_today.py --date 2026-03-02  # 指定日期
  python3 scripts/scan_mode9_today.py --min-score 70     # 降低最低分
"""
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.eastmoney import list_cached_stocks_flat, load_stock_list_csv
from app.paths import GPT_DATA_DIR
from app.scanner import ScanConfig, scan_with_mode3
from app import tencent

CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
STOCK_LIST_CSV = os.path.join(GPT_DATA_DIR, "stock_list.csv")
# 与网站一致：至少 260 根 K 线才能通过 year_lookback+5 检查
KLINE_COUNT = 300


def main():
    import argparse
    ap = argparse.ArgumentParser(description="mode9 筛选指定日期个股")
    ap.add_argument("--date", default=date.today().strftime("%Y-%m-%d"), help="信号日，默认今天")
    ap.add_argument("--min-score", type=int, default=80, help="最低得分，默认 80（≥80 高分池）")
    ap.add_argument(
        "--allow-refresh",
        action="store_true",
        help="允许从网络补全 K 线；默认仅用本地缓存，与服务器结果一致",
    )
    args = ap.parse_args()
    prefer_local = not args.allow_refresh

    target_date = args.date
    min_score = args.min_score

    name_map = load_stock_list_csv(STOCK_LIST_CSV) if os.path.exists(STOCK_LIST_CSV) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if not stock_list:
        print("股票列表为空，请确认缓存目录:", CACHE_DIR)
        sys.exit(1)

    config = ScanConfig(min_score=min_score, max_results=50000, max_market_cap=None)
    try:
        _v = os.environ.get("MODE9_HOT_INDUSTRY_BONUS", "").strip()
        if _v != "":
            config.mode9_hot_industry_bonus = int(_v)
    except ValueError:
        pass
    try:
        _v = os.environ.get("MODE9_HOT_INDUSTRY_TOP_N", "").strip()
        if _v != "":
            config.mode9_hot_industry_top_n = int(_v)
    except ValueError:
        pass
    _sector_ak = os.environ.get("SECTOR_AK_CACHE_DIR", "").strip() or None
    _fund_flow_max_pts = 5
    _fund_flow_yi_per_pt = 3.0
    try:
        _fund_flow_max_pts = int(os.environ.get("SECTOR_FUND_FLOW_MAX_POINTS", "5"))
    except ValueError:
        _fund_flow_max_pts = 0
    try:
        _fund_flow_yi_per_pt = float(os.environ.get("SECTOR_FUND_FLOW_YI_PER_POINT", "3"))
    except ValueError:
        _fund_flow_yi_per_pt = 3.0
    # 必须提供至少 year_lookback+5 根 K 线（默认 245），否则 scanner 会跳过该股
    def kline_loader(item):
        return tencent.get_kline_cached(
            item.code,
            cache_dir=CACHE_DIR,
            count=KLINE_COUNT,
            min_len=260,
            prefer_local=prefer_local,
        )

    results = list(
        scan_with_mode3(
            stock_list,
            config,
            CACHE_DIR,
            progress_cb=None,
            kline_loader=kline_loader,
            start_date=target_date,
            cutoff_date=target_date,
            use_mode8=False,
            use_mode9=True,
            use_71x_standard=True,
            sector_ak_cache_dir=_sector_ak,
            sector_fund_flow_max_points=_fund_flow_max_pts,
            sector_fund_flow_yi_per_point=_fund_flow_yi_per_pt,
        )
    )
    rows = []
    for r in results:
        code = r.code.zfill(6) if len(r.code) < 6 else r.code
        rows.append((code, r.name or code, r.score))
    rows.sort(key=lambda x: (-x[2], x[0]))

    print("mode9 信号日: %s  最低分: %d  共 %d 只\n" % (target_date, min_score, len(rows)))
    if not rows:
        print("当日无满足条件的个股。可尝试: --min-score 60 或更换 --date")
        return
    print("代码      名称            得分")
    print("-" * 36)
    for code, name, score in rows:
        print("%s  %-14s  %s" % (code, (name or "")[:14], score))


if __name__ == "__main__":
    main()
