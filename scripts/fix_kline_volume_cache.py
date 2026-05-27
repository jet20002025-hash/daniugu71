#!/usr/bin/env python3
"""批量修正本地 K 线缓存中成交量「手×100」误存，写回为「手」。

默认处理 data/gpt/kline_cache_tencent 下全部 code.csv。
用法:
  python3 scripts/fix_kline_volume_cache.py              # 正式写回
  python3 scripts/fix_kline_volume_cache.py --dry-run    # 只统计不写盘
  python3 scripts/fix_kline_volume_cache.py --code 603989
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.kline_volume import normalize_kline_volumes_inplace
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
class FileStats:
    code: str
    rows: int
    fixed_rows: int
    changed: bool


class _VolRow:
    __slots__ = ("volume",)

    def __init__(self, volume: float):
        self.volume = volume


def _count_and_fix_volumes(volumes: List[float]) -> int:
    """原地修正 volumes，返回修正行数（与 read_cached_kline 同一套规则）。"""
    if len(volumes) < 5:
        return 0
    before = list(volumes)
    rows = [_VolRow(v) for v in volumes]
    normalize_kline_volumes_inplace(rows)
    for i, r in enumerate(rows):
        volumes[i] = float(r.volume)
    return sum(1 for a, b in zip(before, volumes) if abs(a - b) > 1e-6)


def process_file(path: str, *, dry_run: bool) -> Optional[FileStats]:
    code = os.path.splitext(os.path.basename(path))[0]
    rows: List[dict] = []
    volumes: List[float] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return None
        for row in reader:
            rows.append(row)
            try:
                volumes.append(float(row.get("volume") or 0))
            except (TypeError, ValueError):
                volumes.append(0.0)

    n = len(rows)
    if n == 0:
        return None

    fixed = _count_and_fix_volumes(volumes)
    if fixed == 0:
        return FileStats(code=code, rows=n, fixed_rows=0, changed=False)

    if not dry_run:
        for row, vol in zip(rows, volumes):
            row["volume"] = vol
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    return FileStats(code=code, rows=n, fixed_rows=fixed, changed=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="批量修正 K 线缓存成交量单位（手×100 → 手）")
    ap.add_argument(
        "--cache-dir",
        default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"),
        help="缓存目录，默认 kline_cache_tencent",
    )
    ap.add_argument("--code", default="", help="只处理指定代码，如 603989")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写回文件")
    ap.add_argument("--limit", type=int, default=0, help="最多处理 N 只，0=全部")
    args = ap.parse_args()

    cache_dir = args.cache_dir
    if not os.path.isdir(cache_dir):
        print(f"目录不存在: {cache_dir}")
        return 1

    if args.code:
        path = os.path.join(cache_dir, f"{args.code.strip()}.csv")
        files = [path] if os.path.isfile(path) else []
    else:
        files = sorted(
            os.path.join(cache_dir, fn)
            for fn in os.listdir(cache_dir)
            if fn.endswith(".csv")
        )

    if args.limit > 0:
        files = files[: args.limit]

    total_files = len(files)
    if total_files == 0:
        print("没有可处理的 csv 文件")
        return 1

    mode = "DRY-RUN" if args.dry_run else "WRITE"
    print(f"[{mode}] 目录: {cache_dir}  文件数: {total_files}")

    total_rows = 0
    total_fixed = 0
    changed_files = 0
    errors = 0

    for i, path in enumerate(files, 1):
        try:
            st = process_file(path, dry_run=args.dry_run)
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  失败 {path}: {exc}")
            continue
        if st is None:
            continue
        total_rows += st.rows
        total_fixed += st.fixed_rows
        if st.changed:
            changed_files += 1
        if (i % 500 == 0) or i == total_files:
            print(
                f"  进度 {i}/{total_files}  "
                f"已修正文件 {changed_files}  累计修正行 {total_fixed:,}",
                flush=True,
            )

    print("\n=== 完成 ===")
    print(f"处理文件: {total_files}")
    print(f"K 线总行: {total_rows:,}")
    print(f"修正行数: {total_fixed:,} ({100 * total_fixed / max(total_rows, 1):.2f}%)")
    print(f"涉及文件: {changed_files} ({100 * changed_files / max(total_files, 1):.1f}%)")
    if errors:
        print(f"失败: {errors}")
    if args.dry_run:
        print("（dry-run 未写盘；去掉 --dry-run 执行写回）")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
