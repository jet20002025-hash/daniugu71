#!/usr/bin/env python3
"""
个股评分系统：输入股票代码，输出 mode3 评分及明细
"""
import argparse
import os

from app.paths import GPT_DATA_DIR
from app.stock_score import score_stock


def main():
    parser = argparse.ArgumentParser(description="个股评分：输入代码，输出mode3评分")
    parser.add_argument("code", help="股票代码，如 000001 或 300766")
    parser.add_argument("--date", default=None, help="截止日期 YYYY-MM-DD，默认最新")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache", choices=["east", "tencent"], default="east")
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if not cache_dir:
        cache_dir = os.path.join(GPT_DATA_DIR, "kline_cache_east" if args.cache == "east" else "kline_cache_tencent")
    use_secid = args.cache == "east"

    result = score_stock(args.code, args.date, cache_dir, use_secid)
    code = result["code"]
    name = result.get("name", code)
    score = result["score"]
    has_signal = result["has_signal"]

    print(f"\n【{code} {name}】")
    print("-" * 40)
    if score is not None:
        print(f"评分: {score}")
        if has_signal:
            print(f"信号日: {result['signal_date']}  买入日: {result['buy_date']}")
            print(f"放量: {result.get('vol_ratio', '-')}x  MA10-20: {result.get('ma20_gap_pct', '-')}%  MA20-60: {result.get('ma60_gap_pct', '-')}%")
            print(f"距MA20: {result.get('close_gap_pct', '-')}%  20日涨幅: {result.get('ret20_pct', '-')}%")
        else:
            print(f"状态: 无启动点信号")
            print(f"原因: {result['reason']}")
            print(f"参考日期: {result.get('latest_date', '-')}")
        b = result.get("breakdown", {})
        if b:
            print("\n评分明细:")
            for k, v in b.items():
                print(f"  {k}: +{v}")
    else:
        print(f"无法评分: {result['reason']}")
    print()


if __name__ == "__main__":
    main()
