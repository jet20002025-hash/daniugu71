#!/usr/bin/env python3
"""用腾讯线上成交量覆盖缓存中对应日期（以行情为准校准最近 N 日）。

用法:
  python3 scripts/repair_kline_volume_from_api.py
  python3 scripts/repair_kline_volume_from_api.py --lookback 120 --workers 6
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import tencent
from app.kline_volume import tencent_volume_to_cache_lots
from app.paths import GPT_DATA_DIR

CSV_FIELDS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "pct_chg",
    "chg",
    "turnover",
]


def repair_one(
    code: str,
    cache_dir: str,
    lookback: int,
    delay: float,
) -> tuple[int, int, str | None]:
    path = os.path.join(cache_dir, f"{code}.csv")
    if not os.path.isfile(path):
        return 0, 0, "no_file"
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    if not rows:
        return 0, 0, "empty"

    session = requests.Session()
    session.trust_env = False
    time.sleep(delay)
    try:
        live = tencent.fetch_kline(code, count=lookback + 20, session=session)
    except Exception as exc:
        return 0, 0, f"fetch:{exc}"
    if not live:
        return 0, 0, "no_live"

    live_map = {r.date[:10]: float(r.volume) for r in live[-lookback:]}
    fixed = 0
    checked = 0
    for row in rows:
        d = row["date"][:10]
        if d not in live_map:
            continue
        checked += 1
        live_raw = live_map[d]
        old_v = float(row.get("volume") or 0)
        new_v = tencent_volume_to_cache_lots(live_raw, old_v)
        if abs(old_v - new_v) > max(1.0, max(old_v, new_v) * 1e-4):
            row["volume"] = new_v
            fixed += 1

    if fixed:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return fixed, checked, None


def main() -> int:
    ap = argparse.ArgumentParser(description="用腾讯线上 volume 校准缓存")
    ap.add_argument(
        "--cache-dir",
        default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"),
    )
    ap.add_argument("--lookback", type=int, default=120, help="校准最近 N 个交易日")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--delay", type=float, default=0.06)
    ap.add_argument("--code", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.code:
        codes = [args.code.strip()]
    else:
        codes = sorted(
            fn[:-4] for fn in os.listdir(args.cache_dir) if fn.endswith(".csv")
        )
    if args.limit > 0:
        codes = codes[: args.limit]

    print(f"校准 {len(codes)} 只  最近 {args.lookback} 日 volume（腾讯）")

    total_fixed = 0
    total_checked = 0
    errors: dict[str, int] = {}
    done = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {
            ex.submit(repair_one, c, args.cache_dir, args.lookback, args.delay): c
            for c in codes
        }
        for fut in as_completed(futs):
            done += 1
            fixed, checked, err = fut.result()
            total_fixed += fixed
            total_checked += checked
            if err:
                errors[err] = errors.get(err, 0) + 1
            if done % 500 == 0 or done == len(codes):
                print(
                    f"  进度 {done}/{len(codes)}  已改 {total_fixed:,} 行  错误 {sum(errors.values())}",
                    flush=True,
                )

    print("\n=== 完成 ===")
    print(f"比对行数: {total_checked:,}")
    print(f"覆盖修正: {total_fixed:,}")
    if errors:
        print(f"错误: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
