"""
同花顺行业：通过行情中心行业详情页（非 ajax）分页拉取成分股，构建 股票代码 -> 行业名称 映射。

说明：
- 分页 URL：首页 .../detail/code/{board_code}/ ，第 N 页 .../detail/code/{board_code}/page/N/
- 行业板块列表仅来自 akshare.stock_board_industry_name_ths()；拉不到则不再做本地/HTML 兜底，可自行联网查行业后手写
  ``{cache}/industry/{code}.txt`` 或使用 ``--source em``。
- 请求需携带 akshare 内置 ths.js 生成的 v / hexin-v，与同花顺其他接口一致
"""
from __future__ import annotations

import json
import os
import re
import time
from io import StringIO
from typing import Any, Dict, Optional, Tuple

_PAGE_INFO_RE = re.compile(r'class="page_info"[^>]*>(\d+)\s*/\s*(\d+)')


def ths_map_path(cache_base: str) -> str:
    return os.path.join(cache_base, "ths_code_to_industry.json")


def load_ths_code_industry_map(cache_base: str) -> Optional[Dict[str, str]]:
    p = ths_map_path(cache_base)
    if not os.path.isfile(p) or os.path.getsize(p) < 4:
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    out: Dict[str, str] = {}
    for k, v in data.items():
        if v is None:
            continue
        ks = str(k).zfill(6)
        vs = str(v).strip()
        if ks.isdigit() and len(ks) == 6 and vs:
            out[ks] = vs
    return out or None


def save_ths_code_industry_map(cache_base: str, m: Dict[str, str]) -> None:
    """写入映射；拒绝空字典，避免覆盖已有数据。"""
    if not m:
        raise ValueError("refuse to save empty ths_code_to_industry map")
    os.makedirs(cache_base, exist_ok=True)
    p = ths_map_path(cache_base)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, p)


def _ths_request_headers() -> Dict[str, str]:
    from pathlib import Path

    from py_mini_racer import MiniRacer

    from akshare.datasets import get_ths_js

    js_path = Path(get_ths_js("ths.js"))
    js = MiniRacer()
    js.eval(js_path.read_text(encoding="utf-8"))
    v = js.call("v")
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Cookie": f"v={v}",
        "hexin-v": v,
        "Referer": "http://q.10jqka.com.cn/thshy/",
    }


def _page_count_from_html(html: str) -> int:
    m = _PAGE_INFO_RE.search(html)
    if not m:
        return 1
    try:
        return max(1, int(m.group(2)))
    except ValueError:
        return 1


def _codes_from_detail_html(html: str) -> list[str]:
    from bs4 import BeautifulSoup
    import pandas as pd

    soup = BeautifulSoup(html, "lxml")
    tbl = soup.find("table", class_="m-table")
    if tbl is None:
        return []
    try:
        df = pd.read_html(StringIO(str(tbl)))[0]
    except ValueError:
        return []
    col = None
    for c in df.columns:
        if str(c).strip() == "代码":
            col = c
            break
    if col is None:
        return []
    out: list[str] = []
    for x in df[col].tolist():
        s = str(x).strip().replace(".0", "")
        if s.isdigit():
            out.append(s.zfill(6))
    return out


def fetch_ths_industry_detail_html(board_code: str, page: int) -> str:
    """
    拉取行业详情页 HTML。每次请求使用新的 hexin-v，并对 401/403/429 做短暂重试（同花顺会按 token 限流）。
    """
    import requests

    bc = str(board_code).strip()
    if page <= 1:
        url = f"http://q.10jqka.com.cn/thshy/detail/code/{bc}/"
    else:
        url = f"http://q.10jqka.com.cn/thshy/detail/code/{bc}/page/{page}/"

    last_exc: Optional[Exception] = None
    for attempt in range(3):
        headers = _ths_request_headers()
        try:
            r = requests.get(url, headers=headers, timeout=45)
            if r.status_code == 200:
                enc = r.encoding or "utf-8"
                if enc.lower() in ("iso-8859-1", "ascii"):
                    r.encoding = r.apparent_encoding or "gbk"
                return r.text
            if r.status_code in (401, 403, 429):
                last_exc = RuntimeError(f"HTTP {r.status_code} for {url}")
                time.sleep(0.35 + attempt * 0.4)
                continue
            r.raise_for_status()
        except Exception as e:
            last_exc = e
            time.sleep(0.35 + attempt * 0.35)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"THS detail failed after retries: {url}")


def build_ths_code_industry_map(
    *,
    sleep_sec: float = 0.25,
    max_boards: int = 0,
    progress_every: int = 1,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    遍历同花顺全部行业板块，拉取成分股，返回 (code -> 行业名, stats)。
    max_boards>0 时仅处理前 N 个板块（调试用）。
    """
    import akshare as ak

    boards = None
    for attempt in range(4):
        try:
            boards = ak.stock_board_industry_name_ths()
            if boards is not None and not getattr(boards, "empty", True):
                break
        except Exception:
            boards = None
        time.sleep(0.8 + attempt * 0.4)

    if boards is None or getattr(boards, "empty", True):
        return {}, {
            "error": "empty industry list (stock_board_industry_name_ths failed); "
            "可改用 --source em，或自行联网查行业后写入 akshare_cache/industry/{code}.txt",
        }

    name_by_code: Dict[str, str] = {}
    for _, row in boards.iterrows():
        name = str(row.get("name", "")).strip()
        code = str(row.get("code", "")).strip()
        if name and code:
            name_by_code[code] = name

    codes_order = list(name_by_code.keys())
    if max_boards > 0:
        codes_order = codes_order[:max_boards]

    out: Dict[str, str] = {}
    dup: Dict[str, Tuple[str, str]] = {}
    errors: list[str] = []

    for bi, bc in enumerate(codes_order, 1):
        ind_name = name_by_code.get(bc, bc)
        try:
            html1 = fetch_ths_industry_detail_html(bc, 1)
        except Exception as e:
            errors.append(f"{bc}:{e!r}")
            if sleep_sec:
                time.sleep(sleep_sec)
            continue

        n_pages = _page_count_from_html(html1)
        pages_html = [html1]
        if sleep_sec:
            time.sleep(sleep_sec)

        for pg in range(2, n_pages + 1):
            try:
                pages_html.append(fetch_ths_industry_detail_html(bc, pg))
            except Exception as e:
                errors.append(f"{bc} page{pg}:{e!r}")
                break
            if sleep_sec:
                time.sleep(sleep_sec)

        for html in pages_html:
            for c in _codes_from_detail_html(html):
                if c in out and out[c] != ind_name:
                    dup[c] = (out[c], ind_name)
                else:
                    out[c] = ind_name

        if progress_every and bi % progress_every == 0:
            print(f"  THS 行业映射 {bi}/{len(codes_order)} 板块 {ind_name!r} 已收录股票 {len(out)} 只")

    stats: Dict[str, Any] = {
        "boards": len(codes_order),
        "stocks": len(out),
        "duplicate_assignments": len(dup),
        "errors": errors[:50],
        "error_count": len(errors),
    }
    return out, stats


def fetch_ths_industry_name_for_symbol(symbol: str, cache_base: str) -> Optional[str]:
    """从预构建映射读取同花顺行业名；无映射文件或代码不存在则 None。"""
    m = load_ths_code_industry_map(cache_base)
    if not m:
        return None
    return m.get(str(symbol).zfill(6))
