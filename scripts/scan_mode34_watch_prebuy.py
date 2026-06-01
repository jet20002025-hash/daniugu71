#!/usr/bin/env python3
"""mode34 观察池 + 预案日买卖建议（电科模版：22 观察，25 盘中买）。

流程:
  1) 观察日 (--watch-date): 如 5/22 列入观察池，指向下一信号日
  2) 预案日 (--prebuy-date): 如 5/25 收阳缩量 → 当日盘中突破昨高买入

用法:
  # 5/24 入观察（电科实际 5/22 周五 / 5/25 周一前）
  python3 scripts/scan_mode34_watch_prebuy.py --watch-date 2026-05-22

  # 5/25 对观察池或全市场给买卖建议
  python3 scripts/scan_mode34_watch_prebuy.py --prebuy-date 2026-05-25

  # 同一脚本：先观察后预案（传两日）
  python3 scripts/scan_mode34_watch_prebuy.py --watch-date 2026-05-22 --prebuy-date 2026-05-25

  # 单股验证
  python3 scripts/scan_mode34_watch_prebuy.py --prebuy-date 2026-05-25 --code 600850
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.eastmoney import list_cached_stocks_flat, load_stock_list_csv, read_cached_kline_by_code
from app.mode34_bottom_break_pullback import (
    match_mode34_watchlist,
    mode34_prebuy_advice,
    mode34_kw_from_scan_config,
)
from app.paths import GPT_DATA_DIR
from app.scanner import ScanConfig, _is_st

CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
STOCK_LIST = os.path.join(GPT_DATA_DIR, "stock_list.csv")


def _find_idx(rows, ymd: str) -> Optional[int]:
    ymd = ymd.strip()[:10]
    for i, r in enumerate(rows):
        if r.date[:10] == ymd:
            return i
    return None


def _scan_watch(
    stock_list,
    name_map: dict,
    watch_date: str,
    kw: dict,
    *,
    skip_st: bool,
    skip_bj: bool,
    only_code: str = "",
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for item in stock_list:
        code = item.code.zfill(6)
        if only_code and code != only_code.zfill(6):
            continue
        name = (item.name or name_map.get(code, code) or "").strip()
        if skip_st and _is_st(name):
            continue
        if skip_bj and code.startswith("920"):
            continue
        rows = read_cached_kline_by_code(CACHE_DIR, code)
        if not rows:
            continue
        idx = _find_idx(rows, watch_date)
        if idx is None:
            continue
        rec = match_mode34_watchlist(rows, idx, code, name, **kw)
        if rec:
            rec["code"] = code
            rec["name"] = name
            hits.append(rec)
    hits.sort(key=lambda x: (-x["watch_score"], x["code"]))
    return hits


def _scan_prebuy(
    stock_list,
    name_map: dict,
    prebuy_date: str,
    kw: dict,
    *,
    skip_st: bool,
    skip_bj: bool,
    only_code: str = "",
    watch_codes: Optional[set] = None,
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for item in stock_list:
        code = item.code.zfill(6)
        if only_code and code != only_code.zfill(6):
            continue
        if watch_codes is not None and code not in watch_codes:
            continue
        name = (item.name or name_map.get(code, code) or "").strip()
        if skip_st and _is_st(name):
            continue
        if skip_bj and code.startswith("920"):
            continue
        rows = read_cached_kline_by_code(CACHE_DIR, code)
        if not rows:
            continue
        idx = _find_idx(rows, prebuy_date)
        if idx is None:
            continue
        rec = mode34_prebuy_advice(rows, idx, code, name, **kw)
        if rec:
            rec["code"] = code
            rec["name"] = name
            hits.append(rec)
    hits.sort(key=lambda x: (-x["advice_score"], x["code"]))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description="mode34 观察池 + 预案日买卖建议")
    ap.add_argument("--watch-date", default="", help="观察日（入池）")
    ap.add_argument("--prebuy-date", default="", help="预案日（买卖建议）")
    ap.add_argument("--code", default="", help="仅测单股")
    ap.add_argument("--from-watch-csv", default="", help="预案日仅扫描观察池 CSV 的 code 列")
    ap.add_argument("--skip-st", action="store_true", default=True)
    ap.add_argument("--skip-bj", action="store_true", default=True)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not args.watch_date and not args.prebuy_date:
        ref = read_cached_kline_by_code(CACHE_DIR, "000001")
        args.prebuy_date = ref[-1].date[:10] if ref else date.today().isoformat()

    cfg = ScanConfig()
    kw = mode34_kw_from_scan_config(cfg)
    name_map = load_stock_list_csv(STOCK_LIST) if os.path.exists(STOCK_LIST) else {}
    stock_list = list_cached_stocks_flat(CACHE_DIR, name_map=name_map)
    if args.code.strip():
        oc = args.code.strip().zfill(6)
        stock_list = [s for s in stock_list if s.code.zfill(6) == oc] or [
            type("X", (), {"code": oc, "name": name_map.get(oc, oc)})()
        ]

    watch_codes: Optional[set] = None
    watch_hits: List[Dict[str, Any]] = []

    if args.watch_date.strip():
        watch_hits = _scan_watch(
            stock_list,
            name_map,
            args.watch_date.strip()[:10],
            kw,
            skip_st=args.skip_st,
            skip_bj=args.skip_bj,
            only_code=args.code.strip(),
        )
        watch_codes = {h["code"] for h in watch_hits}
        out_w = args.out or os.path.join(
            GPT_DATA_DIR,
            "results",
            f"mode34_watch_{args.watch_date.replace('-', '')}.csv",
        )
        os.makedirs(os.path.dirname(out_w) or ".", exist_ok=True)
        if watch_hits:
            fields = list(watch_hits[0].keys())
            with open(out_w, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(watch_hits[: args.top])
        print(f"\n观察日 {args.watch_date}  入池 {len(watch_hits)} 只 → {out_w}")
        for r in watch_hits[:20]:
            print(
                f"  {r['code']} {(r['name'] or '')[:8]:8s} 分{r['watch_score']:3d} "
                f"回踩{r['pullback_days']:.0f}天 距平台高+{r['dist_to_pull_high_pct']:.1f}% "
                f"峰{r['peak_date']}"
            )

    if args.from_watch_csv.strip():
        path = args.from_watch_csv.strip()
        with open(path, encoding="utf-8-sig") as f:
            watch_codes = {row["code"].zfill(6) for row in csv.DictReader(f) if row.get("code")}

    if args.prebuy_date.strip():
        prebuy_hits = _scan_prebuy(
            stock_list,
            name_map,
            args.prebuy_date.strip()[:10],
            kw,
            skip_st=args.skip_st,
            skip_bj=args.skip_bj,
            only_code=args.code.strip(),
            watch_codes=watch_codes,
        )
        out_p = args.out if args.watch_date else (
            args.out
            or os.path.join(
                GPT_DATA_DIR,
                "results",
                f"mode34_prebuy_{args.prebuy_date.replace('-', '')}.csv",
            )
        )
        if not args.watch_date:
            os.makedirs(os.path.dirname(out_p) or ".", exist_ok=True)
            if prebuy_hits:
                fields = list(prebuy_hits[0].keys())
                with open(out_p, "w", encoding="utf-8-sig", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(prebuy_hits[: args.top])
        print(f"\n预案日 {args.prebuy_date}  建议 {len(prebuy_hits)} 只 → {out_p}")
        print(
            f"{'代码':<8}{'名称':<10}{'建议':<8}{'分':>4}{'触发>':>8}{'止损<':>8}"
            f"{'信号':<24}{'次日m34':>6}"
        )
        for r in prebuy_hits[:30]:
            print(
                f"{r['code']:<8}{(r['name'] or '')[:10]:<10}{r['advice']:<8}"
                f"{r['advice_score']:4d}{r['buy_trigger_above']:8.2f}{r['stop_below']:8.2f}"
                f"{(r.get('signals') or '')[:24]:<24}{'是' if r.get('next_day_mode34') else '否':>6}"
            )
            print(f"    → {r.get('action', '')}")


if __name__ == "__main__":
    main()
