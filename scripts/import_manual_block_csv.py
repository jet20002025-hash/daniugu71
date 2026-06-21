#!/usr/bin/env python3
"""将名称列表导入为 manual_blocks/{板块名}.csv（code,name）。"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import akshare as ak

from app.eastmoney import load_stock_list_csv
from app.paths import GPT_DATA_DIR

MANUAL_EXTRA = {
    "天海电子": "001365",
    "鸿仕达": "920125",
    "锐翔智能": "920178",
    "红板科技": "603459",
    "泰金新能": "688813",
    "ST章鼓": "002598",
    "ST瀚川": "688022",
    "*ST康佳A": "000016",
    "鼎熔岩": "301028",
    "*ST中迪": "000609",
    "ST得润": "002055",
    "ST长园": "002510",
    "盈新发展": "000620",
    "*ST亚振": "603389",
}


def _norm(s: str) -> str:
    return re.sub(r"^\*?ST", "", s.replace(" ", ""))


def _resolve_names(names: list[str], lst: dict[str, str], ak_map: dict[str, str]) -> list[tuple[str, str]]:
    name_to_codes: dict[str, list[str]] = {}
    for code, name in lst.items():
        key = name.replace(" ", "")
        name_to_codes.setdefault(key, []).append(code)

    out: list[tuple[str, str]] = []
    missing: list[str] = []
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        code = MANUAL_EXTRA.get(name)
        if not code:
            n = name.replace(" ", "")
            if n in name_to_codes:
                code = name_to_codes[n][0]
            elif _norm(n) in name_to_codes:
                code = name_to_codes[_norm(n)][0]
            elif n in ak_map:
                code = ak_map[n]
            elif _norm(n) in ak_map:
                code = ak_map[_norm(n)]
        if not code:
            missing.append(name)
            continue
        code = str(code).zfill(6)
        nm = lst.get(code, "") or ak_map.get(_norm(name), name)
        out.append((code, nm))

    if missing:
        raise SystemExit(f"未匹配名称: {missing}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="导入手工板块 CSV")
    ap.add_argument("--block", required=True, help="板块名，如 PCB概念")
    ap.add_argument("--names-file", required=True, help="每行一个股票名称的文本文件")
    ap.add_argument(
        "--out-dir",
        default=os.path.join(GPT_DATA_DIR, "manual_blocks"),
    )
    args = ap.parse_args()

    with open(args.names_file, "r", encoding="utf-8") as f:
        names = f.read().splitlines()

    lst = load_stock_list_csv(os.path.join(GPT_DATA_DIR, "stock_list.csv"))
    ak_df = ak.stock_info_a_code_name()
    ak_map = {str(r["name"]).replace(" ", ""): str(r["code"]).zfill(6) for _, r in ak_df.iterrows()}

    rows = _resolve_names(names, lst, ak_map)
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for code, nm in rows:
        if code in seen:
            continue
        seen.add(code)
        deduped.append((code, nm))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.block.strip()}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "name"])
        for code, nm in sorted(deduped):
            w.writerow([code, nm])

    print(f"已写入 {out_path}，共 {len(deduped)} 只")


if __name__ == "__main__":
    main()
