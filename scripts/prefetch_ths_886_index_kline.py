#!/usr/bin/env python3
"""
下载同花顺 886 开头板块指数日 K，缓存到 data/gpt/kline_cache_ths_886/{code}.csv

用法:
  python3 scripts/prefetch_ths_886_index_kline.py
  python3 scripts/prefetch_ths_886_index_kline.py --discover-only
  python3 scripts/prefetch_ths_886_index_kline.py --codes 886070,886001 --start-date 20200101
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.paths import GPT_DATA_DIR
from app.ths_index_kline import (
    build_886_name_map,
    default_end_date,
    discover_886_codes,
    fetch_index_kline,
    ths_index_cache_dir,
    ths_index_meta_path,
    write_index_kline_csv,
)


def _disable_proxies() -> None:
    for k in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        os.environ.pop(k, None)


def _load_meta(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v}


def _save_meta(path: str, meta: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载同花顺 886 指数日 K")
    parser.add_argument(
        "--cache-dir",
        default=ths_index_cache_dir(GPT_DATA_DIR),
        help="K 线缓存目录",
    )
    parser.add_argument("--start-date", default="20200101")
    parser.add_argument("--end-date", default=default_end_date())
    parser.add_argument(
        "--codes",
        default="",
        help="仅下载指定代码，逗号分隔（默认自动发现全部 886）",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="仅扫描有效 886 代码并更新 index_meta.json",
    )
    parser.add_argument(
        "--skip-name-map",
        action="store_true",
        help="跳过概念页名称映射（扫描仍会做，但名称可能为空）",
    )
    parser.add_argument("--delay", type=float, default=0.05, help="请求间隔秒")
    parser.add_argument("--limit", type=int, default=0, help="最多下载只数（调试用）")
    args = parser.parse_args()

    _disable_proxies()
    session = requests.Session()

    meta_path = ths_index_meta_path(GPT_DATA_DIR)
    meta = _load_meta(meta_path)

    if args.codes.strip():
        codes = [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]
    else:
        print("扫描 886000–886999 有效指数代码…")
        codes = discover_886_codes(session, delay=args.delay)
        print(f"发现 {len(codes)} 只 886 指数")

    if not args.skip_name_map:
        print("拉取概念板块名称映射（clid → 名称）…")
        name_map = build_886_name_map(session, delay=args.delay)
        meta.update(name_map)
        print(f"名称映射 {len(name_map)} 条")

    for code in codes:
        if code not in meta:
            meta[code] = meta.get(code, "")

    _save_meta(meta_path, meta)

    if args.discover_only:
        print(f"已写入 {meta_path}，共 {len(codes)} 只")
        for code in codes:
            name = meta.get(code, "")
            print(f"  {code} {name}")
        return

    if args.limit > 0:
        codes = codes[: args.limit]

    ok, fail = 0, 0
    for i, code in enumerate(codes, 1):
        name = meta.get(code, "")
        label = f"{code} {name}".strip()
        out_path = os.path.join(args.cache_dir, f"{code}.csv")
        try:
            rows = fetch_index_kline(
                session,
                code,
                args.start_date,
                args.end_date,
                delay=args.delay,
            )
            if not rows:
                print(f"[{i}/{len(codes)}] {label} 无数据", file=sys.stderr)
                fail += 1
                continue
            write_index_kline_csv(out_path, rows)
            print(
                f"[{i}/{len(codes)}] {label} {len(rows)} 根 "
                f"{rows[0].date} ~ {rows[-1].date}"
            )
            ok += 1
        except Exception as e:
            print(f"[{i}/{len(codes)}] {label} 失败: {e}", file=sys.stderr)
            fail += 1
        time.sleep(args.delay)

    print(f"完成: 成功 {ok}，失败 {fail}，目录 {args.cache_dir}")


if __name__ == "__main__":
    main()
