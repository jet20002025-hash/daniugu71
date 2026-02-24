import csv
import datetime as dt
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

KLINE_URL = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
REFERER = "http://finance.sina.com.cn"


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


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _prefix(code: str) -> str:
    if code.startswith("6"):
        return "sh"
    if code.startswith("8") or code.startswith("4"):
        return "bj"
    return "sz"


def fetch_kline(
    code: str,
    count: int = 120,
    session: Optional[requests.Session] = None,
) -> List[KlineRow]:
    session = session or requests.Session()
    session.trust_env = False
    session.headers.update({"Referer": REFERER})

    symbol = f"{_prefix(code)}{code}"
    datalen = min(max(1, count), 1023)
    params = {
        "symbol": symbol,
        "scale": 240,
        "ma": "no",
        "datalen": datalen,
    }

    resp = session.get(KLINE_URL, params=params, timeout=20)
    if resp.status_code != 200:
        return []
    payload = resp.json()
    if not isinstance(payload, list):
        return []

    rows: List[KlineRow] = []
    prev_close = None
    for item in payload:
        if not isinstance(item, dict):
            continue
        date = item.get("day") or item.get("date")
        open_ = _safe_float(item.get("open"))
        high = _safe_float(item.get("high"))
        low = _safe_float(item.get("low"))
        close = _safe_float(item.get("close"))
        volume = _safe_float(item.get("volume"))
        amount = _safe_float(item.get("amount")) if item.get("amount") is not None else 0.0

        if prev_close:
            chg = close - prev_close
            pct = (chg / prev_close * 100) if prev_close else 0.0
        else:
            chg = close - open_
            pct = (chg / open_ * 100) if open_ else 0.0

        amplitude = (high - low) / close * 100 if close else 0.0
        rows.append(
            KlineRow(
                date=date,
                open=open_,
                close=close,
                high=high,
                low=low,
                volume=volume,
                amount=amount,
                amplitude=amplitude,
                pct_chg=pct,
                chg=chg,
                turnover=0.0,
            )
        )
        prev_close = close
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
    code: str,
    cache_dir: str,
    count: int = 120,
    session: Optional[requests.Session] = None,
    max_age_days: int = 2,
    pause: float = 0.0,
    prefer_local: bool = False,
    min_len: Optional[int] = None,
) -> Optional[List[KlineRow]]:
    cache_path = os.path.join(cache_dir, f"{code}.csv")
    cached = read_cached_kline(cache_path)
    if cached:
        if min_len is not None and len(cached) < min_len:
            cached = None
        elif prefer_local or kline_is_fresh(cached, max_age_days=max_age_days):
            return cached

    rows = fetch_kline(code=code, count=count, session=session)
    if rows:
        write_cached_kline(cache_path, rows)
    if pause:
        time.sleep(pause)
    return rows


def fetch_index_kline(session: Optional[requests.Session] = None) -> List[KlineRow]:
    return fetch_kline("000001", count=1023, session=session)
