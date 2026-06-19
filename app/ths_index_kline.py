"""
同花顺板块指数日 K（886 系列概念指数等）。

指数在行情页 clid / 软件代码为 886xxx；K 线接口：
  https://d.10jqka.com.cn/v4/line/bk_{code}/01/{year}.js
"""
from __future__ import annotations

import csv
import os
import re
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import requests
from akshare.utils import demjson
from bs4 import BeautifulSoup

from app.eastmoney import KlineRow
from app.ths_industry import _ths_request_headers


def ths_index_cache_dir(base: str) -> str:
    return os.path.join(base, "kline_cache_ths_886")


def ths_index_meta_path(base: str) -> str:
    return os.path.join(ths_index_cache_dir(base), "index_meta.json")


def discover_886_codes(
    session: requests.Session,
    *,
    start: int = 886000,
    end: int = 887000,
    probe_year: int | None = None,
    delay: float = 0.03,
) -> List[str]:
    """扫描 886000–886999，探测哪些代码有日 K。"""
    if probe_year is None:
        probe_year = datetime.now().year
    headers = _ths_request_headers()
    headers.setdefault("Referer", "http://q.10jqka.com.cn")
    out: List[str] = []
    for num in range(start, end):
        code = f"{num:06d}"
        url = f"https://d.10jqka.com.cn/v4/line/bk_{code}/01/{probe_year}.js"
        try:
            resp = session.get(url, headers=headers, timeout=12)
        except requests.RequestException:
            if delay:
                time.sleep(delay)
            continue
        text = resp.text or ""
        if resp.status_code == 200 and '"data"' in text and "null" not in text[:80]:
            out.append(code)
        if delay:
            time.sleep(delay)
    return out


def build_886_name_map(
    session: requests.Session,
    *,
    delay: float = 0.05,
) -> Dict[str, str]:
    """从同花顺概念板块详情页 clid 字段汇总 886 代码与名称。"""
    import akshare as ak

    headers = _ths_request_headers()
    headers.setdefault("Referer", "http://q.10jqka.com.cn/gn/")
    concepts = ak.stock_board_concept_name_ths()
    out: Dict[str, str] = {}
    for _, row in concepts.iterrows():
        page_code = str(row["code"]).strip()
        name = str(row["name"]).strip()
        url = f"https://q.10jqka.com.cn/gn/detail/code/{page_code}/"
        try:
            resp = session.get(url, headers=headers, timeout=20)
        except requests.RequestException:
            if delay:
                time.sleep(delay)
            continue
        soup = BeautifulSoup(resp.text, features="lxml")
        inp = soup.find("input", id="clid")
        if inp and inp.get("value"):
            clid = str(inp["value"]).strip()
            if clid.startswith("886") and name:
                out[clid] = name
        if delay:
            time.sleep(delay)
    return out


def _parse_kline_js(text: str) -> List[Tuple[str, float, float, float, float, float, float]]:
    """解析同花顺 v4 line JS，返回 (date, open, high, low, close, volume, amount)。"""
    if not text or "{" not in text:
        return []
    payload = demjson.decode(text[text.find("{") : -1])
    raw = payload.get("data")
    if not raw:
        return []
    rows: List[Tuple[str, float, float, float, float, float, float]] = []
    for line in str(raw).split(";"):
        parts = line.split(",")
        if len(parts) < 7:
            continue
        d = parts[0].strip()
        if not re.fullmatch(r"\d{8}", d):
            continue
        try:
            rows.append(
                (
                    f"{d[:4]}-{d[4:6]}-{d[6:8]}",
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                )
            )
        except (TypeError, ValueError):
            continue
    return rows


def fetch_index_kline(
    session: requests.Session,
    code: str,
    start_date: str,
    end_date: str,
    *,
    delay: float = 0.05,
) -> List[KlineRow]:
    """拉取单只同花顺指数日 K，并补齐涨跌幅等字段。"""
    headers = _ths_request_headers()
    headers.setdefault("Referer", "http://q.10jqka.com.cn")
    begin_year = int(start_date[:4])
    end_year = int(end_date[:4])
    merged: Dict[str, Tuple[str, float, float, float, float, float, float]] = {}
    for year in range(begin_year, end_year + 1):
        url = f"https://d.10jqka.com.cn/v4/line/bk_{code}/01/{year}.js"
        try:
            resp = session.get(url, headers=headers, timeout=20)
        except requests.RequestException:
            if delay:
                time.sleep(delay)
            continue
        for row in _parse_kline_js(resp.text or ""):
            merged[row[0]] = row
        if delay:
            time.sleep(delay)

    if not merged:
        return []

    dates = sorted(merged.keys())
    start_key = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    end_key = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
    dates = [d for d in dates if start_key <= d <= end_key]
    out: List[KlineRow] = []
    prev_close: Optional[float] = None
    for d in dates:
        _, op, hi, lo, cl, vol, amt = merged[d]
        if prev_close is None:
            pct = 0.0
            chg = 0.0
            amp = 0.0
        else:
            chg = cl - prev_close
            pct = (chg / prev_close * 100.0) if prev_close else 0.0
            amp = ((hi - lo) / prev_close * 100.0) if prev_close else 0.0
        out.append(
            KlineRow(
                date=d,
                open=op,
                close=cl,
                high=hi,
                low=lo,
                volume=vol,
                amount=amt,
                amplitude=amp,
                pct_chg=pct,
                chg=chg,
                turnover=0.0,
            )
        )
        prev_close = cl
    return out


def write_index_kline_csv(path: str, rows: List[KlineRow]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
        )
        for row in rows:
            writer.writerow(
                [
                    row.date,
                    row.open,
                    row.close,
                    row.high,
                    row.low,
                    row.volume,
                    row.amount,
                    row.amplitude,
                    row.pct_chg,
                    row.chg,
                    row.turnover,
                ]
            )


def default_end_date() -> str:
    return date.today().strftime("%Y%m%d")
