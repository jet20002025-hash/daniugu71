import csv
import datetime as dt
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

CLIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


@dataclass
class StockItem:
    code: str
    name: str
    market: int  # 0=SZ, 1=SH

    @property
    def secid(self) -> str:
        return f"{self.market}.{self.code}"


@dataclass
class KlineRow:
    date: str
    open: float
    close: float
    high: float
    low: float
    volume: float
    amount: float
    amplitude: float
    pct_chg: float
    chg: float
    turnover: float


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _market_from_code(code: str) -> int:
    if code.startswith("6"):
        return 1
    return 0


def fetch_stock_list(
    session: Optional[requests.Session] = None,
    page_size: int = 100,
    max_pages: int = 200,
) -> List[StockItem]:
    session = session or requests.Session()
    items: List[StockItem] = []
    seen = set()
    for page in range(1, max_pages + 1):
        params = {
            "pn": page,
            "pz": page_size,
            "po": 1,
            "np": 1,
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": 2,
            "invt": 2,
            "fid": "f3",
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
            "fields": "f12,f14,f13",
        }
        resp = session.get(CLIST_URL, params=params, timeout=15)
        resp.raise_for_status()
        try:
            data = resp.json() or {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data_block = data.get("data", {})
        if not isinstance(data_block, dict):
            data_block = {}
        rows = data_block.get("diff", []) or []
        if not rows:
            break
        added = 0
        for row in rows:
            code = str(row.get("f12", "")).strip()
            name = str(row.get("f14", "")).strip()
            market = row.get("f13", None)
            if not code or code in seen:
                continue
            if market not in (0, 1):
                market = _market_from_code(code)
            items.append(StockItem(code=code, name=name, market=int(market)))
            seen.add(code)
            added += 1
        if added == 0:
            break
    return items


def stock_items_from_list_csv(path: str) -> List[StockItem]:
    if not os.path.exists(path):
        return []
    items: List[StockItem] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = str(row.get("code", "")).strip()
            name = str(row.get("name", "")).strip()
            if not code or code in seen:
                continue
            market = _market_from_code(code)
            items.append(StockItem(code=code, name=name or code, market=market))
            seen.add(code)
    return items


def list_cached_stocks(cache_dir: str) -> List[StockItem]:
    if not os.path.exists(cache_dir):
        return []
    items: List[StockItem] = []
    seen = set()
    for name in os.listdir(cache_dir):
        if not name.endswith(".csv"):
            continue
        base = name[:-4]
        if "_" not in base:
            continue
        prefix, code = base.split("_", 1)
        if not code or code in seen:
            continue
        market = _market_from_code(code)
        try:
            market = int(prefix)
        except Exception:
            market = _market_from_code(code)
        items.append(StockItem(code=code, name=code, market=market))
        seen.add(code)
    return items


def load_stock_list_csv(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    mapping = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = str(row.get("code", "")).strip()
            name = str(row.get("name", "")).strip()
            if code and name:
                mapping[code] = name
    return mapping


def list_cached_stocks_flat(cache_dir: str, name_map: Optional[dict] = None) -> List[StockItem]:
    if not os.path.exists(cache_dir):
        return []
    items: List[StockItem] = []
    seen = set()
    name_map = name_map or {}
    for name in os.listdir(cache_dir):
        if not name.endswith(".csv"):
            continue
        code = name[:-4]
        if not code or code in seen:
            continue
        if not code.isdigit():
            continue
        market = _market_from_code(code)
        items.append(StockItem(code=code, name=name_map.get(code, code), market=market))
        seen.add(code)
    return items


_SCAN_LIST_CACHE_PATH = ".scan_list_flat.json"


def list_cached_stocks_flat_cached(
    cache_dir: str,
    name_map: Optional[dict] = None,
    max_age_sec: int = 600,
) -> List[StockItem]:
    """与 list_cached_stocks_flat 相同，但用文件缓存结果，避免每次 listdir 数千文件导致启动慢。"""
    import json
    cache_path = os.path.join(cache_dir, _SCAN_LIST_CACHE_PATH)
    try:
        if os.path.exists(cache_path):
            age = time.time() - os.path.getmtime(cache_path)
            if age <= max_age_sec:
                with open(cache_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                return [StockItem(code=x["code"], name=x.get("name", x["code"]), market=int(x["market"])) for x in raw]
    except Exception:
        pass
    items = list_cached_stocks_flat(cache_dir, name_map)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"code": i.code, "name": i.name, "market": i.market} for i in items],
                f,
                ensure_ascii=False,
            )
    except Exception:
        pass
    return items


def list_cached_stocks_secid(cache_dir: str, name_map: Optional[dict] = None) -> List[StockItem]:
    if not os.path.exists(cache_dir):
        return []
    items: List[StockItem] = []
    seen = set()
    name_map = name_map or {}
    for name in os.listdir(cache_dir):
        if not name.endswith(".csv"):
            continue
        stem = name[:-4]
        if "_" not in stem:
            continue
        prefix, code = stem.split("_", 1)
        if not code or code in seen or not code.isdigit():
            continue
        try:
            market = int(prefix)
        except Exception:
            market = _market_from_code(code)
        items.append(StockItem(code=code, name=name_map.get(code, code), market=market))
        seen.add(code)
    return items


def read_cached_kline_by_market_code(
    cache_dir: str,
    market: int,
    code: str,
) -> Optional[List[KlineRow]]:
    cache_path = os.path.join(cache_dir, f"{market}_{code}.csv")
    return read_cached_kline(cache_path)


def read_cached_kline_by_code(cache_dir: str, code: str) -> Optional[List[KlineRow]]:
    cache_path = os.path.join(cache_dir, f"{code}.csv")
    return read_cached_kline(cache_path)


def fetch_kline(
    secid: str,
    count: int = 120,
    session: Optional[requests.Session] = None,
) -> List[KlineRow]:
    session = session or requests.Session()
    end_date = int(dt.date.today().strftime("%Y%m%d"))
    params = {
        "secid": secid,
        "klt": 101,
        "fqt": 1,
        "beg": 0,
        "end": end_date,
        "lmt": count,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
    }
    resp = session.get(KLINE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("data", {})
    klines = data.get("klines", []) or []
    rows = []
    for item in klines:
        parts = item.split(",")
        if len(parts) < 11:
            continue
        rows.append(
            KlineRow(
                date=parts[0],
                open=_safe_float(parts[1]),
                close=_safe_float(parts[2]),
                high=_safe_float(parts[3]),
                low=_safe_float(parts[4]),
                volume=_safe_float(parts[5]),
                amount=_safe_float(parts[6]),
                amplitude=_safe_float(parts[7]),
                pct_chg=_safe_float(parts[8]),
                chg=_safe_float(parts[9]),
                turnover=_safe_float(parts[10]),
            )
        )
    return rows


def read_cached_kline(cache_path: str) -> Optional[List[KlineRow]]:
    if not os.path.exists(cache_path):
        return None
    rows: List[KlineRow] = []
    with open(cache_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                KlineRow(
                    date=row["date"],
                    open=_safe_float(row["open"]),
                    close=_safe_float(row["close"]),
                    high=_safe_float(row["high"]),
                    low=_safe_float(row["low"]),
                    volume=_safe_float(row["volume"]),
                    amount=_safe_float(row["amount"]),
                    amplitude=_safe_float(row["amplitude"]),
                    pct_chg=_safe_float(row["pct_chg"]),
                    chg=_safe_float(row["chg"]),
                    turnover=_safe_float(row["turnover"]),
                )
            )
    return rows


def write_cached_kline(cache_path: str, rows: List[KlineRow]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", newline="", encoding="utf-8") as handle:
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


def kline_is_fresh(rows: List[KlineRow], max_age_days: int = 2) -> bool:
    """缓存视为新鲜仅当已包含「今天」的 K 线，否则会一直用旧缓存拿不到当日数据。"""
    if not rows:
        return False
    try:
        last_date = dt.datetime.strptime(rows[-1].date[:10], "%Y-%m-%d").date()
    except Exception:
        return False
    today = dt.date.today()
    return last_date >= today


def get_kline_cached(
    secid: str,
    cache_dir: str,
    count: int = 120,
    session: Optional[requests.Session] = None,
    max_age_days: int = 2,
    pause: float = 0.0,
    local_only: bool = False,
    prefer_local: bool = False,
) -> Optional[List[KlineRow]]:
    cache_path = os.path.join(cache_dir, f"{secid.replace('.', '_')}.csv")
    cached = read_cached_kline(cache_path)
    if cached and (local_only or prefer_local or kline_is_fresh(cached, max_age_days=max_age_days)):
        return cached
    if local_only:
        return cached

    rows = fetch_kline(secid=secid, count=count, session=session)
    if rows:
        write_cached_kline(cache_path, rows)
    if pause:
        time.sleep(pause)
    return rows


def fetch_index_kline(session: Optional[requests.Session] = None) -> List[KlineRow]:
    return fetch_kline("1.000001", count=120, session=session)
