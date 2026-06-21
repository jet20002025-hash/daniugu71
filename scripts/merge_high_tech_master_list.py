#!/usr/bin/env python3
"""从名称文本合并为高科技汇总 CSV 并重建 universe。"""
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
from app.high_tech_universe import (
    HIGH_TECH_STOCK_LIST_CSV,
    HIGH_TECH_UNIVERSE_JSON,
    build_high_tech_universe,
    load_blocks_config,
    save_blocks_config,
)
from app.paths import GPT_DATA_DIR
from scripts.import_manual_block_csv import MANUAL_EXTRA, _norm, _resolve_names

MASTER_BLOCK = "高科技汇总"
MASTER_CSV = os.path.join(GPT_DATA_DIR, "manual_blocks", f"{MASTER_BLOCK}.csv")


def _clean_name_line(line: str) -> str:
    s = line.strip()
    if not s or s == "名称":
        return ""
    if s.startswith("wing:"):
        return ""
    if "以下是所有高科技个股" in s:
        s = re.sub(r"^.*以下是所有高科技个股[^，\n]*[，,]?\s*", "", s)
        s = re.sub(r"\s*wing:.*$", "", s).strip()
        if not s or s.startswith("wing:"):
            return ""
    if s.startswith("请加进去") or s.startswith("重复的去掉"):
        return ""
    # 去掉尾部 wing 时间戳残留
    s = re.sub(r"\s+wing:.*$", "", s).strip()
    return s


def parse_names_file(path: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = _clean_name_line(line)
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
    return names


def _alias_name(name: str) -> str:
    """XD/DR 等特殊简称 → 可匹配名称。"""
    aliases = {
        "XD世运电": "世运电路",
        "XD科强股": "科强股份",
        "XD聚和材": "聚和材料",
        "DR兴业股": "兴业股份",
        "DR中触媒": "中触媒",
        "XD税友股": "税友股份",
        "XD朗迪集": "朗迪集团",
        "XD拉普拉": "拉普拉斯",
        "XD恒运昌": "恒运昌",
        "XD赛恩斯": "赛恩斯",
        "XD厦门国": "厦门国贸",
        "XD北矿科": "北矿科技",
        "XD鸣志电": "鸣志电器",
        "XD旭升集": "旭升集团",
        "XD全柴动": "全柴动力",
        "XD万通液": "万通液压",
        "华虹宏力": "华虹公司",
        "XD巍华新": "巍华新材",
        "精进电动-W": "精进电动",
        "怡 亚 通": "怡亚通",
        "新 和 成": "新和成",
        "新 洋 丰": "新洋丰",
        "新洋丰": "新洋丰",
        "新 亚 制 程": "新亚制程",
        "新亚制程": "新亚制程",
        "XD中国瑞": "中国瑞林",
        "红 宝 丽": "红宝丽",
        "金 螳 螂": "金螳螂",
        "TCL中环": "TCL中环",
    }
    if name in aliases:
        return aliases[name]
    # 仅处理除权除息 XD/DR 前缀；勿剥离 ST/*ST（否则无法匹配）
    n = re.sub(r"^(XD|DR)", "", name)
    n = n.replace(" ", "")
    if n != name.replace(" ", ""):
        return n
    return name


def resolve_all(names: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    lst = load_stock_list_csv(os.path.join(GPT_DATA_DIR, "stock_list.csv"))
    ak_df = ak.stock_info_a_code_name()
    ak_map = {str(r["name"]).replace(" ", ""): str(r["code"]).zfill(6) for _, r in ak_df.iterrows()}
    # 全称映射
    for _, r in ak_df.iterrows():
        ak_map[str(r["name"]).strip()] = str(r["code"]).zfill(6)

    resolved: list[tuple[str, str]] = []
    missing: list[str] = []
    for raw in names:
        alias = _alias_name(raw)
        if raw in MANUAL_EXTRA:
            code = str(MANUAL_EXTRA[raw]).zfill(6)
            resolved.append((code, lst.get(code, raw)))
            continue
        if alias in MANUAL_EXTRA:
            code = str(MANUAL_EXTRA[alias]).zfill(6)
            resolved.append((code, lst.get(code, alias)))
            continue
        try:
            rows = _resolve_names([alias], lst, ak_map)
            if not rows and alias != raw:
                rows = _resolve_names([raw], lst, ak_map)
            if rows:
                code, nm = rows[0]
                resolved.append((code, nm or lst.get(code, alias)))
            else:
                missing.append(raw)
        except SystemExit:
            missing.append(raw)
    # dedupe by code
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for code, nm in resolved:
        if code in seen:
            continue
        seen.add(code)
        out.append((code, nm))
    return out, missing


def ensure_master_block_in_config() -> None:
    cfg = load_blocks_config()
    blocks = cfg.setdefault("blocks", [])
    for b in blocks:
        if str(b.get("name", "")).strip() == MASTER_BLOCK:
            b["source"] = "manual_csv"
            b["file"] = f"manual_blocks/{MASTER_BLOCK}.csv"
            save_blocks_config(cfg)
            return
    blocks.append(
        {
            "id": "high_tech_master",
            "name": MASTER_BLOCK,
            "source": "manual_csv",
            "file": f"manual_blocks/{MASTER_BLOCK}.csv",
        }
    )
    save_blocks_config(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="合并高科技主清单")
    ap.add_argument("--names-file", required=True)
    ap.add_argument("--rebuild", action="store_true", default=True)
    args = ap.parse_args()

    names = parse_names_file(args.names_file)
    print(f"解析名称 {len(names)} 条")
    rows, missing = resolve_all(names)
    print(f"匹配 {len(rows)} 只，未匹配 {len(missing)} 只")
    if missing:
        miss_path = args.names_file + ".missing.txt"
        with open(miss_path, "w", encoding="utf-8") as f:
            f.write("\n".join(missing))
        print(f"未匹配列表: {miss_path}")

    os.makedirs(os.path.dirname(MASTER_CSV), exist_ok=True)
    with open(MASTER_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "name"])
        for code, nm in sorted(rows):
            w.writerow([code, nm])
    print(f"已写入 {MASTER_CSV}")

    ensure_master_block_in_config()
    if args.rebuild:
        data = build_high_tech_universe()
        print(f"高科技板块合计 {data['total']} 只")
        print(f"CSV: {HIGH_TECH_STOCK_LIST_CSV}")
        print(f"JSON: {HIGH_TECH_UNIVERSE_JSON}")


if __name__ == "__main__":
    main()
