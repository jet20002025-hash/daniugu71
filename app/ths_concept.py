"""同花顺概念板块成分股（行情中心 gn 分页详情页）。"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import requests

from app.ths_industry import (
    _codes_from_detail_html,
    _page_count_from_html,
    _ths_request_headers,
)


def find_ths_concept_board(name: str) -> Tuple[str, str]:
    """按名称查找同花顺概念，返回 (board_code, board_name)。"""
    import akshare as ak

    boards = ak.stock_board_concept_name_ths()
    if boards is None or boards.empty:
        raise RuntimeError("未获取到同花顺概念列表")
    key = name.strip()
    hit = boards[boards["name"].astype(str).str.strip() == key]
    if hit.empty:
        mask = boards["name"].astype(str).str.contains(key, na=False)
        hit = boards[mask]
    if hit.empty:
        raise RuntimeError(f"未找到同花顺概念板块: {name}")
    row = hit.iloc[0]
    return str(row["code"]).strip(), str(row["name"]).strip()


def fetch_ths_concept_constituents(
    board_code: str,
    *,
    sleep_sec: float = 0.25,
) -> Tuple[List[str], Optional[str]]:
    """返回 (成分股代码列表, 指数 clid 如 885893)。"""
    bc = str(board_code).strip()
    headers_base = {"Referer": "http://q.10jqka.com.cn/gn/"}

    def fetch_page(page: int) -> str:
        headers = _ths_request_headers()
        headers.update(headers_base)
        if page <= 1:
            url = f"http://q.10jqka.com.cn/gn/detail/code/{bc}/"
        else:
            url = f"http://q.10jqka.com.cn/gn/detail/code/{bc}/page/{page}/"
        resp = requests.get(url, headers=headers, timeout=45)
        resp.encoding = resp.apparent_encoding or "gbk"
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} {url}")
        return resp.text

    html = fetch_page(1)
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    inp = soup.find("input", id="clid")
    clid = str(inp.get("value")).strip() if inp and inp.get("value") else None

    pages = _page_count_from_html(html)
    codes: List[str] = []
    for page in range(1, pages + 1):
        if page > 1:
            time.sleep(sleep_sec)
            html = fetch_page(page)
        codes.extend(_codes_from_detail_html(html))

    return sorted(set(codes)), clid


def fetch_ths_concept_by_name(
    name: str,
    *,
    sleep_sec: float = 0.25,
) -> Tuple[str, str, List[str], Optional[str]]:
    """返回 (board_code, board_name, codes, clid)。"""
    board_code, board_name = find_ths_concept_board(name)
    codes, clid = fetch_ths_concept_constituents(board_code, sleep_sec=sleep_sec)
    return board_code, board_name, codes, clid
