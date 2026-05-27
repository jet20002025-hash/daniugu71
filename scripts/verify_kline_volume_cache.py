#!/usr/bin/env python3
"""校验 K 线缓存成交量是否与腾讯线上一致（手）。

用法:
  python3 scripts/verify_kline_volume_cache.py
  python3 scripts/verify_kline_volume_cache.py --fix   # 不一致时用线上覆盖最近 N 日 volume
  python3 scripts/verify_kline_volume_cache.py --code 003036
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import tencent
from app.kline_volume import normalize_kline_volumes_inplace, tencent_volume_to_cache_lots
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


@dataclass
class Mismatch:
    code: str
    date: str
    cache_vol: float
    live_vol: float
    ratio: float


def _load_cache_rows(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def _cache_vol_map(rows: List[dict], *, apply_norm: bool) -> Dict[str, float]:
    class _R:
        def __init__(self, volume: float):
            self.volume = volume

    tmp = [_R(float(r.get("volume") or 0)) for r in rows]
    if apply_norm:
        normalize_kline_volumes_inplace(tmp)
    return {r["date"][:10]: float(tmp[i].volume) for i, r in enumerate(rows)}


def _compare_code(
    code: str,
    cache_dir: str,
    lookback: int,
    *,
    apply_norm: bool,
) -> Tuple[List[Mismatch], Optional[str]]:
    path = os.path.join(cache_dir, f"{code}.csv")
    if not os.path.isfile(path):
        return [], "no_file"
    rows = _load_cache_rows(path)
    if len(rows) < 5:
        return [], "short_cache"

    cache_map = _cache_vol_map(rows, apply_norm=apply_norm)
    session = requests.Session()
    session.trust_env = False
    try:
        live = tencent.fetch_kline(code, count=lookback + 10, session=session)
    except Exception as exc:
        return [], f"fetch_fail:{exc}"

    if not live:
        return [], "no_live"

    live_map = {r.date[:10]: float(r.volume) for r in live[-lookback:]}
    mismatches: List[Mismatch] = []
    for d, live_v in live_map.items():
        cache_v = cache_map.get(d)
        if cache_v is None:
            continue
        if live_v <= 0 and cache_v <= 0:
            continue
        aligned = tencent_volume_to_cache_lots(live_v, cache_v)
        if abs(cache_v - aligned) <= max(1.0, max(cache_v, aligned) * 1e-4):
            continue
        ratio = cache_v / aligned if aligned > 0 else float("inf")
        mismatches.append(
            Mismatch(
                code=code,
                date=d,
                cache_vol=cache_v,
                live_vol=aligned,
                ratio=ratio,
            )
        )
    return mismatches, None


def _fix_file(path: str, live_rows: List[tencent.KlineRow], lookback: int) -> int:
    """用线上最近 lookback 日 volume 覆盖缓存对应行。"""
    rows = _load_cache_rows(path)
    if not rows:
        return 0
    live_map = {r.date[:10]: float(r.volume) for r in live_rows[-lookback:]}
    fixed = 0
    for row in rows:
        d = row["date"][:10]
        if d not in live_map:
            continue
        new_v = live_map[d]
        old_v = float(row.get("volume") or 0)
        if abs(old_v - new_v) > max(1.0, new_v * 1e-6):
            row["volume"] = new_v
            fixed += 1
    if fixed:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return fixed


def main() -> int:
    ap = argparse.ArgumentParser(description="校验/修复缓存成交量与腾讯线上一致")
    ap.add_argument(
        "--cache-dir",
        default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"),
    )
    ap.add_argument("--lookback", type=int, default=60, help="比对最近 N 个交易日")
    ap.add_argument("--code", default="", help="只检查单码")
    ap.add_argument("--fix", action="store_true", help="用线上 volume 覆盖不一致行")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--delay", type=float, default=0.08)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--apply-norm",
        action="store_true",
        help="读入时再跑 volume 归一化（默认只校验磁盘原始值）",
    )
    args = ap.parse_args()

    cache_dir = args.cache_dir
    if not os.path.isdir(cache_dir):
        print(f"目录不存在: {cache_dir}")
        return 1

    if args.code:
        codes = [args.code.strip()]
    else:
        codes = sorted(fn[:-4] for fn in os.listdir(cache_dir) if fn.endswith(".csv"))
    if args.limit > 0:
        codes = codes[: args.limit]

    apply_norm = args.apply_norm
    mode = "读入归一化后" if apply_norm else "磁盘原始"
    print(f"校验 {len(codes)} 只  最近 {args.lookback} 日  模式={mode}  fix={args.fix}")

    all_mismatches: List[Mismatch] = []
    errors: Dict[str, int] = {}
    done = 0

    def task(code: str) -> Tuple[str, List[Mismatch], Optional[str]]:
        time.sleep(args.delay)
        mm, err = _compare_code(
            code, cache_dir, args.lookback, apply_norm=apply_norm
        )
        return code, mm, err

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(task, c): c for c in codes}
        for fut in as_completed(futs):
            done += 1
            code, mm, err = fut.result()
            if err:
                errors[err] = errors.get(err, 0) + 1
            else:
                all_mismatches.extend(mm)
            if done % 500 == 0 or done == len(codes):
                print(
                    f"  进度 {done}/{len(codes)}  不一致 {len(all_mismatches)}  错误 {sum(errors.values())}",
                    flush=True,
                )

    # 统计 ×100 / ÷100 特征
    times100 = sum(1 for m in all_mismatches if 80 <= m.ratio <= 120)
    times001 = sum(1 for m in all_mismatches if 0.008 <= m.ratio <= 0.012)

    print("\n=== 校验结果 ===")
    print(f"检查股票: {len(codes)}")
    print(f"成交量不一致: {len(all_mismatches)} 处")
    print(f"  约×100倍: {times100}  约÷100倍: {times001}")
    if errors:
        print(f"拉取/文件错误: {errors}")

    if all_mismatches:
        print("\n样例（前20）:")
        for m in all_mismatches[:20]:
            print(
                f"  {m.code} {m.date}  缓存={m.cache_vol:.0f}  线上={m.live_vol:.0f}  比={m.ratio:.2f}"
            )

    if args.fix and all_mismatches:
        by_code: Dict[str, List[Mismatch]] = {}
        for m in all_mismatches:
            by_code.setdefault(m.code, []).append(m)
        fixed_rows = 0
        fixed_files = 0
        session = requests.Session()
        session.trust_env = False
        for i, code in enumerate(sorted(by_code), 1):
            path = os.path.join(cache_dir, f"{code}.csv")
            try:
                live = tencent.fetch_kline(
                    code, count=args.lookback + 10, session=session
                )
            except Exception:
                continue
            n = _fix_file(path, live, args.lookback)
            if n:
                fixed_files += 1
                fixed_rows += n
            time.sleep(args.delay)
            if i % 200 == 0:
                print(f"  修复进度 {i}/{len(by_code)}", flush=True)
        print(f"\n已修复 {fixed_files} 文件 / {fixed_rows} 行（最近{args.lookback}日 volume）")

        # 复检
        print("\n--- 修复后复检（读入归一化）---")
        remain = 0
        for code in sorted(by_code):
            mm, _ = _compare_code(
                code, cache_dir, args.lookback, apply_norm=True
            )
            remain += len(mm)
        print(f"剩余不一致: {remain}")

    return 1 if all_mismatches and not args.fix else 0


if __name__ == "__main__":
    raise SystemExit(main())
