"""高科技板块股票池：合并用户指定的同花顺概念等板块成分股。"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from app.paths import GPT_DATA_DIR

HIGH_TECH_BLOCKS_JSON = os.path.join(GPT_DATA_DIR, "high_tech_blocks.json")
HIGH_TECH_UNIVERSE_JSON = os.path.join(GPT_DATA_DIR, "high_tech_universe.json")
HIGH_TECH_STOCK_LIST_CSV = os.path.join(GPT_DATA_DIR, "high_tech_stock_list.csv")
HIGH_TECH_UNIVERSE_XLSX = os.path.join(GPT_DATA_DIR, "results", "high_tech_universe.xlsx")
MANUAL_BLOCKS_DIR = os.path.join(GPT_DATA_DIR, "manual_blocks")


def default_blocks_config() -> Dict[str, Any]:
    return {
        "title": "高科技板块",
        "description": "由用户指定的同花顺概念等板块合并；寻股默认在此池内检索。",
        "blocks": [
            {
                "id": "national_big_fund",
                "name": "国家大基金持股",
                "source": "ths_concept",
            },
        ],
    }


def load_blocks_config(path: str = HIGH_TECH_BLOCKS_JSON) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return default_blocks_config()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return default_blocks_config()
    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        data["blocks"] = default_blocks_config()["blocks"]
    return data


def save_blocks_config(cfg: Dict[str, Any], path: str = HIGH_TECH_BLOCKS_JSON) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def add_block_name(name: str, *, source: str = "ths_concept") -> Dict[str, Any]:
    """向配置追加板块（按名称去重）。"""
    cfg = load_blocks_config()
    blocks = cfg.setdefault("blocks", [])
    key = name.strip()
    if not key:
        raise ValueError("板块名称不能为空")
    for b in blocks:
        if str(b.get("name", "")).strip() == key:
            return cfg
    blocks.append(
        {
            "id": key.replace(" ", "_")[:40],
            "name": key,
            "source": source,
        }
    )
    save_blocks_config(cfg)
    return cfg


def load_high_tech_universe(path: str = HIGH_TECH_UNIVERSE_JSON) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def high_tech_code_set(path: str = HIGH_TECH_UNIVERSE_JSON) -> Set[str]:
    data = load_high_tech_universe(path)
    if not data:
        return set()
    out: Set[str] = set()
    for row in data.get("stocks", []) or []:
        code = str(row.get("code", "")).strip().zfill(6)
        if code.isdigit() and len(code) == 6:
            out.add(code)
    return out


def filter_codes(codes: List[str], path: str = HIGH_TECH_UNIVERSE_JSON) -> List[str]:
    pool = high_tech_code_set(path)
    if not pool:
        return codes
    return [c for c in codes if str(c).zfill(6) in pool]


def in_high_tech_universe(code: str, path: str = HIGH_TECH_UNIVERSE_JSON) -> bool:
    return str(code).zfill(6) in high_tech_code_set(path)


def high_tech_universe_count(path: str = HIGH_TECH_UNIVERSE_JSON) -> int:
    data = load_high_tech_universe(path)
    if not data:
        return 0
    try:
        return int(data.get("total", 0) or 0)
    except (TypeError, ValueError):
        return 0


def filter_stock_items(items: List[Any], path: str = HIGH_TECH_UNIVERSE_JSON) -> List[Any]:
    """仅保留高科技板块池内的 StockItem 等带 code 属性的对象。"""
    pool = high_tech_code_set(path)
    if not pool:
        return list(items)
    out: List[Any] = []
    for item in items:
        code = str(getattr(item, "code", item)).zfill(6)
        if code in pool:
            out.append(item)
    return out


def load_manual_block_csv(path: str) -> List[str]:
    """读取手工板块 CSV（code,name），返回代码列表。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    codes: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = str(row.get("code", "")).strip().zfill(6)
            if code.isdigit() and len(code) == 6:
                codes.append(code)
    return sorted(set(codes))


def save_universe_artifacts(data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(HIGH_TECH_UNIVERSE_JSON) or ".", exist_ok=True)
    tmp = HIGH_TECH_UNIVERSE_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, HIGH_TECH_UNIVERSE_JSON)

    with open(HIGH_TECH_STOCK_LIST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "name"])
        writer.writeheader()
        for row in data.get("stocks", []):
            writer.writerow(
                {
                    "code": str(row.get("code", "")).zfill(6),
                    "name": str(row.get("name", "")).strip(),
                }
            )

    try:
        import pandas as pd

        os.makedirs(os.path.dirname(HIGH_TECH_UNIVERSE_XLSX) or ".", exist_ok=True)
        stocks = data.get("stocks", [])
        df = pd.DataFrame(stocks)
        blocks = data.get("blocks", [])
        with pd.ExcelWriter(HIGH_TECH_UNIVERSE_XLSX, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="汇总")
            for b in blocks:
                name = str(b.get("name", "板块"))
                sheet = name[:31]
                sub = df[df["from_blocks"].astype(str).str.contains(name, na=False, regex=False)]
                if not sub.empty:
                    sub.to_excel(w, index=False, sheet_name=sheet)
    except Exception:
        pass


def build_high_tech_universe(
    *,
    sleep_sec: float = 0.25,
    stock_list_path: Optional[str] = None,
) -> Dict[str, Any]:
    from app.eastmoney import load_stock_list_csv
    from app.ths_concept import fetch_ths_concept_by_name

    if stock_list_path is None:
        stock_list_path = os.path.join(GPT_DATA_DIR, "stock_list.csv")
    name_map = load_stock_list_csv(stock_list_path)

    cfg = load_blocks_config()
    blocks_cfg = cfg.get("blocks", [])
    code_to_blocks: Dict[str, Set[str]] = {}
    block_meta: List[Dict[str, Any]] = []

    for item in blocks_cfg:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "ths_concept")).strip()
        block_name = str(item.get("name", "")).strip()
        if not block_name:
            continue
        if source == "manual_csv":
            rel = str(item.get("file", f"manual_blocks/{block_name}.csv")).strip()
            csv_path = rel if os.path.isabs(rel) else os.path.join(GPT_DATA_DIR, rel)
            try:
                codes = load_manual_block_csv(csv_path)
                for c in codes:
                    code_to_blocks.setdefault(c, set()).add(block_name)
                block_meta.append(
                    {
                        "name": block_name,
                        "source": source,
                        "file": rel,
                        "count": len(codes),
                    }
                )
            except Exception as exc:
                block_meta.append(
                    {
                        "name": block_name,
                        "source": source,
                        "error": str(exc),
                        "count": 0,
                    }
                )
            continue
        if source != "ths_concept":
            block_meta.append(
                {
                    "name": block_name,
                    "source": source,
                    "error": f"unsupported source: {source}",
                    "count": 0,
                }
            )
            continue
        try:
            board_code, board_name, codes, clid = fetch_ths_concept_by_name(
                block_name, sleep_sec=sleep_sec
            )
            for c in codes:
                code_to_blocks.setdefault(c.zfill(6), set()).add(board_name)
            block_meta.append(
                {
                    "name": board_name,
                    "source": source,
                    "concept_code": board_code,
                    "index_clid": clid,
                    "count": len(codes),
                }
            )
        except Exception as exc:
            block_meta.append(
                {
                    "name": block_name,
                    "source": source,
                    "error": str(exc),
                    "count": 0,
                }
            )

    stocks: List[Dict[str, Any]] = []
    for code in sorted(code_to_blocks.keys()):
        blocks = sorted(code_to_blocks[code])
        stocks.append(
            {
                "code": code,
                "name": name_map.get(code, ""),
                "from_blocks": ";".join(blocks),
                "block_count": len(blocks),
            }
        )

    data = {
        "title": cfg.get("title", "高科技板块"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(stocks),
        "blocks": block_meta,
        "stocks": stocks,
    }
    save_universe_artifacts(data)
    return data
