#!/usr/bin/env python3
"""构建/更新「高科技板块」股票池（合并 high_tech_blocks.json 中的各子板块）。

用法:
  python3 scripts/build_high_tech_universe.py
  python3 scripts/build_high_tech_universe.py --add-block 半导体概念
  python3 scripts/build_high_tech_universe.py --list-blocks
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.high_tech_universe import (
    HIGH_TECH_BLOCKS_JSON,
    HIGH_TECH_STOCK_LIST_CSV,
    HIGH_TECH_UNIVERSE_JSON,
    HIGH_TECH_UNIVERSE_XLSX,
    add_block_name,
    build_high_tech_universe,
    load_blocks_config,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="构建高科技板块股票池")
    ap.add_argument("--add-block", default="", help="追加同花顺概念板块名称并重建")
    ap.add_argument("--list-blocks", action="store_true", help="列出已配置子板块")
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    if args.list_blocks:
        cfg = load_blocks_config()
        print(f"配置: {HIGH_TECH_BLOCKS_JSON}")
        for b in cfg.get("blocks", []):
            print(f"  - {b.get('name')} ({b.get('source')})")
        return

    if args.add_block.strip():
        add_block_name(args.add_block.strip())
        print(f"已追加板块: {args.add_block.strip()}")

    data = build_high_tech_universe(sleep_sec=args.sleep)
    print(f"高科技板块共 {data['total']} 只（{data['updated_at']}）")
    for b in data.get("blocks", []):
        err = b.get("error")
        if err:
            print(f"  ✗ {b.get('name')}: {err}")
        else:
            print(f"  ✓ {b.get('name')}: {b.get('count')} 只 clid={b.get('index_clid')}")
    print(f"JSON: {HIGH_TECH_UNIVERSE_JSON}")
    print(f"CSV:  {HIGH_TECH_STOCK_LIST_CSV}")
    print(f"XLSX: {HIGH_TECH_UNIVERSE_XLSX}")


if __name__ == "__main__":
    main()
