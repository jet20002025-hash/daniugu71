#!/usr/bin/env python3
"""拉取同花顺概念板块成分股。

用法:
  python3 scripts/fetch_ths_concept_stocks.py --name 国家大基金持股
  python3 scripts/fetch_ths_concept_stocks.py --code 307816 --out data/gpt/results/national_big_fund_stocks.xlsx
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.eastmoney import load_stock_list_csv
from app.paths import GPT_DATA_DIR
from app.ths_concept import fetch_ths_concept_by_name, fetch_ths_concept_constituents


def main() -> None:
    ap = argparse.ArgumentParser(description="拉取同花顺概念成分股")
    ap.add_argument("--name", default="", help="概念名称")
    ap.add_argument("--code", default="", help="概念行情页 code")
    ap.add_argument(
        "--stock-list",
        default=os.path.join(GPT_DATA_DIR, "stock_list.csv"),
    )
    ap.add_argument("--out", default="", help="输出 xlsx")
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    if args.code:
        board_code = str(args.code).strip()
        board_name = args.name.strip() or board_code
        codes, clid = fetch_ths_concept_constituents(board_code, sleep_sec=args.sleep)
    elif args.name:
        board_code, board_name, codes, clid = fetch_ths_concept_by_name(
            args.name, sleep_sec=args.sleep
        )
    else:
        ap.error("请指定 --name 或 --code")

    name_map = load_stock_list_csv(args.stock_list)
    rows = [
        {
            "code": c,
            "name": name_map.get(c, ""),
            "concept": board_name,
            "concept_code": board_code,
            "index_clid": clid or "",
        }
        for c in codes
    ]
    df = pd.DataFrame(rows)
    out = args.out.strip() or os.path.join(
        GPT_DATA_DIR, "results", f"ths_concept_{board_name}.xlsx"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_excel(out, index=False)
    txt = out.rsplit(".", 1)[0] + ".txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"{board_name}（同花顺概念 {board_code}，clid={clid}）共 {len(df)} 只\n")
        for _, r in df.iterrows():
            f.write(f"{r['code']} {r['name']}\n")

    print(f"{board_name}: {len(df)} 只 → {out}")


if __name__ == "__main__":
    main()
