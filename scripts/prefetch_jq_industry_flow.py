import argparse
import csv
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from jqdatasdk import auth, get_money_flow, is_auth, normalize_code


def _safe_industry_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def _load_industry_map(industry_dir: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not os.path.exists(industry_dir):
        return mapping
    for name in os.listdir(industry_dir):
        if not name.endswith(".txt"):
            continue
        code = name[:-4]
        path = os.path.join(industry_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                industry = handle.read().strip()
            if industry:
                mapping[code] = industry
        except Exception:
            continue
    return mapping


def _chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _guess_col(df, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        for cand in candidates:
            if cand.lower() in col.lower():
                return col
    return None


def _read_existing(path: str) -> Dict[str, float]:
    if not os.path.exists(path):
        return {}
    data: Dict[str, float] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                date = str(row.get("date", "")).strip()
                if not date:
                    continue
                try:
                    net = float(row.get("net", 0) or 0)
                except Exception:
                    net = 0.0
                data[date] = net
    except Exception:
        return {}
    return data


def _write_flow(path: str, data: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    items = sorted(data.items(), key=lambda x: x[0])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "net"])
        for date, net in items:
            writer.writerow([date, f"{net:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch JQData industry fund flow and merge.")
    parser.add_argument("--start-date", default="2025-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="2025-08-18", help="YYYY-MM-DD")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--industry-dir", default="data/akshare_cache/industry")
    parser.add_argument("--output-dir", default="data/akshare_cache/fund_flow")
    parser.add_argument("--merge", action="store_true", help="Merge into existing files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--limit", type=int, default=0, help="Limit stock count")
    args = parser.parse_args()

    user = os.getenv("JQDATA_USER")
    pwd = os.getenv("JQDATA_PASS")
    if not user or not pwd:
        raise RuntimeError("请先设置 JQDATA_USER 和 JQDATA_PASS 环境变量")
    auth(user, pwd)
    if not is_auth():
        raise RuntimeError("JQData 认证失败")

    code_to_ind = _load_industry_map(args.industry_dir)
    if not code_to_ind:
        raise RuntimeError("行业映射为空，请先生成 data/akshare_cache/industry")

    codes = list(code_to_ind.keys())
    if args.limit and args.limit > 0:
        codes = codes[: args.limit]

    sec_to_ind: Dict[str, str] = {}
    for code in codes:
        try:
            sec = normalize_code(code)
            sec_to_ind[sec] = code_to_ind[code]
        except Exception:
            continue

    securities = list(sec_to_ind.keys())
    if not securities:
        raise RuntimeError("未找到可用的证券代码")

    industry_flow: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    chunks = _chunk_list(securities, args.chunk_size)
    for idx, chunk in enumerate(chunks, start=1):
        df = get_money_flow(chunk, start_date=args.start_date, end_date=args.end_date)
        if df is None or df.empty:
            continue
        date_col = _guess_col(df, ["date", "日期"])
        sec_col = _guess_col(df, ["sec_code", "code", "证券代码"])
        net_col = _guess_col(df, ["net_amount_main", "net_amount", "主力净流入", "净流入"])
        if not date_col or not sec_col or not net_col:
            continue
        for _, row in df.iterrows():
            sec = str(row[sec_col]).strip()
            industry = sec_to_ind.get(sec)
            if not industry:
                continue
            date = str(row[date_col])[:10]
            try:
                net = float(row[net_col])
            except Exception:
                net = 0.0
            industry_flow[industry][date] += net
        if args.sleep:
            time.sleep(args.sleep)
        if idx % 10 == 0 or idx == len(chunks):
            print(f"进度 {idx}/{len(chunks)}")

    for industry, flow in industry_flow.items():
        safe = _safe_industry_name(industry)
        path = os.path.join(args.output_dir, f"{safe}.csv")
        if args.overwrite:
            merged = flow
        else:
            existing = _read_existing(path) if args.merge else {}
            for date, net in flow.items():
                if date not in existing:
                    existing[date] = net
            merged = existing
        _write_flow(path, merged)

    print(f"完成: 生成行业 {len(industry_flow)} 个资金流文件")


if __name__ == "__main__":
    main()
