"""
行业/板块趋势（信号日及以前，无未来数据）。

数据来源（与 scripts/backtest_startup_modes.py 一致）：
- 个股所属行业：`{cache_dir}/industry/{code}.txt` 文本一行，或按需用 akshare 拉取并写入；
- 行业板块指数日 K：`{cache_dir}/industry_hist/{safe_industry}.csv`（ak.stock_board_industry_hist_em）。

信号日当日指标：
- industry_ret5 / industry_ret10 / industry_ret20：行业指数收盘相对 N 日前收盘涨幅（%）。

可选「资金净流入排行」快照（仅展示/辅助，历史回测需自行按日落库）：
- `{cache_dir}/sector_flow_rank/{YYYY-MM-DD}.json`，由 scripts/fetch_sector_flow_snapshot.py 生成。
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

_SAFE = re.compile(r"[^\w\u4e00-\u9fff\-\.]+")


def _safe_industry_filename(industry: str) -> str:
    s = industry.replace("/", "_").replace(" ", "_").strip()
    return _SAFE.sub("_", s)[:120] or "unknown"


def load_industry_name(code: str, cache_dir: str) -> Optional[str]:
    """仅读缓存文件，不发起网络请求。"""
    path = os.path.join(cache_dir, "industry", f"{str(code).zfill(6)}.txt")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            v = f.read().strip()
            return v or None
    except OSError:
        return None


def load_industry_hist_rows(industry: str, cache_dir: str) -> Optional[List[Dict[str, str]]]:
    if not industry:
        return None
    safe = _safe_industry_filename(industry)
    path = os.path.join(cache_dir, "industry_hist", f"{safe}.csv")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return None


def _parse_hist_series(rows: List[Dict[str, str]]) -> Tuple[List[str], List[float]]:
    """返回 (dates YYYY-MM-DD, closes)."""
    dates: List[str] = []
    closes: List[float] = []
    for row in rows:
        d = row.get("日期") or row.get("date") or row.get("Date") or ""
        d = str(d).strip()[:10]
        if not d:
            continue
        close = row.get("收盘") or row.get("收盘价") or row.get("close")
        if close is None:
            continue
        try:
            c = float(str(close).replace(",", ""))
        except ValueError:
            continue
        if c <= 0:
            continue
        dates.append(d)
        closes.append(c)
    return dates, closes


def industry_close_returns_at_date(
    rows: List[Dict[str, str]], signal_date: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    在 signal_date 那根 K 上，用**仅截至该日**的收盘序列计算行业指数 ret5/10/20（%）。
    若历史不足则对应项为 None。
    """
    ds, cl = _parse_hist_series(rows)
    if not ds:
        return None, None, None
    sig = signal_date[:10]
    try:
        i = ds.index(sig)
    except ValueError:
        # 非交易日或缺 bar：取不超过 signal_date 的最后一根
        i = -1
        for j, d in enumerate(ds):
            if d <= sig:
                i = j
        if i < 0:
            return None, None, None

    def ret_n(n: int) -> Optional[float]:
        if i < n:
            return None
        base = cl[i - n]
        if base <= 0:
            return None
        return (cl[i] - base) / base * 100.0

    return ret_n(5), ret_n(10), ret_n(20)


def industry_rank_in_flow_snapshot(
    cache_dir: str, as_of_date: str, industry: Optional[str]
) -> Optional[int]:
    if not industry:
        return None
    path = os.path.join(cache_dir, "sector_flow_rank", f"{as_of_date[:10]}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    ranks = data.get("ranks") or data.get("items") or []
    for idx, item in enumerate(ranks, start=1):
        if not isinstance(item, dict):
            continue
        n = str(item.get("name") or item.get("板块名称") or item.get("行业名称") or "").strip()
        if n == industry.strip() or industry.strip() in n or n in industry.strip():
            r = item.get("rank")
            return int(r) if r is not None else idx
    return None


def metrics_for_signal(
    code: str,
    signal_date: str,
    cache_dir: str,
    hist_mem: Optional[Dict[str, Optional[List[Dict[str, str]]]]] = None,
) -> Dict[str, Any]:
    """
    聚合板块相关 metrics（无网络）。hist_mem: industry -> hist rows，扫描时复用。
    """
    out: Dict[str, Any] = {
        "industry": None,
        "industry_ret5": None,
        "industry_ret10": None,
        "industry_ret20": None,
        "sector_flow_rank": None,
    }
    industry = load_industry_name(code, cache_dir)
    if not industry:
        return out
    out["industry"] = industry

    mem = hist_mem if hist_mem is not None else {}
    if industry not in mem:
        mem[industry] = load_industry_hist_rows(industry, cache_dir)
    hist = mem[industry]
    if not hist:
        out["sector_flow_rank"] = industry_rank_in_flow_snapshot(cache_dir, signal_date, industry)
        return out

    r5, r10, r20 = industry_close_returns_at_date(hist, signal_date)
    out["industry_ret5"] = r5
    out["industry_ret10"] = r10
    out["industry_ret20"] = r20
    out["sector_flow_rank"] = industry_rank_in_flow_snapshot(cache_dir, signal_date, industry)
    return out


def _match_ths_industry_key(industry: str, by_industry: Dict[str, Any]) -> Optional[str]:
    """个股行业名与同花顺行业榜名称模糊匹配。"""
    if not industry or not by_industry:
        return None
    s = industry.strip()
    if s in by_industry:
        return s
    for k in by_industry:
        if not k:
            continue
        if s in k or k in s:
            return k
    return None


def merge_ths_flow_features(
    sector_sm: Dict[str, Any],
    signal_date: str,
    features_blob: Optional[Dict[str, Any]],
) -> None:
    """
    将同花顺多窗口资金特征并入 sector_sm（就地修改）。
    仅当 signal_date 与 features_blob['trade_date'] 同为 YYYY-MM-DD 时生效，避免区间回测误用「今天」的资金面。
    """
    if not features_blob or not sector_sm.get("industry"):
        return
    td = str(features_blob.get("trade_date") or "").strip()[:10]
    if not td or str(signal_date)[:10] != td:
        return
    by_ind = features_blob.get("by_industry")
    if not isinstance(by_ind, dict):
        return
    key = _match_ths_industry_key(str(sector_sm["industry"]), by_ind)
    if not key:
        return
    row = by_ind.get(key)
    if not isinstance(row, dict):
        return
    mapping = (
        ("rank_即时", "ths_flow_rank_1d"),
        ("rank_5日排行", "ths_flow_rank_5d"),
        ("rank_10日排行", "ths_flow_rank_10d"),
        ("rank_20日排行", "ths_flow_rank_20d"),
    )
    for src, dst in mapping:
        v = row.get(src)
        if v is not None:
            try:
                sector_sm[dst] = int(v)
            except (TypeError, ValueError):
                pass
    for w, nk in (("即时", "ths_flow_net_1d"), ("5日排行", "ths_flow_net_5d")):
        nt = row.get(f"net_{w}")
        if nt is not None:
            sector_sm[nk] = str(nt)
    mom = row.get("flow_momentum")
    if mom is not None:
        try:
            sector_sm["ths_flow_momentum"] = int(mom)
        except (TypeError, ValueError):
            pass


def sector_score_bonus(
    industry_ret5: Optional[float],
    cap: int = 5,
    per_pct: float = 3.0,
) -> int:
    """行业 5 日涨幅每 per_pct 个百分点 +1 分，上限 cap。"""
    if industry_ret5 is None or math.isnan(industry_ret5):
        return 0
    if per_pct <= 0:
        return 0
    b = int(max(0.0, industry_ret5) / per_pct)
    return int(min(cap, b))


def parse_ths_flow_net_yi(value: Any) -> Optional[float]:
    """
    将同花顺行业资金流「净额」解析为 float（单位：亿元）。
    支持数值、'12.34'、'1.2亿'、'500万' 等常见形式。
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    s = str(value).strip().replace(",", "")
    if not s or s in ("-", "—", "nan", "None"):
        return None
    mult = 1.0
    if "亿" in s:
        s = s.replace("亿", "").strip()
    elif "万" in s:
        s = s.replace("万", "").strip()
        mult = 1e-4
    try:
        return float(s) * mult
    except ValueError:
        return None


def sector_fund_flow_score_delta(
    net_yi: Optional[float],
    yi_per_point: float = 3.0,
    max_abs_points: int = 5,
) -> int:
    """
    板块（行业）资金净流入加分、净流出减分。
    - net_yi：净流入，亿元，正=流入、负=流出。
    - 每 |yi_per_point| 亿约对应 1 分，四舍五入；总分变绝对值不超过 max_abs_points。
    """
    if net_yi is None or yi_per_point <= 0 or max_abs_points <= 0:
        return 0
    step = net_yi / yi_per_point
    if math.isnan(step) or math.isinf(step):
        return 0
    delta = int(round(step))
    if delta > max_abs_points:
        return max_abs_points
    if delta < -max_abs_points:
        return -max_abs_points
    return delta
