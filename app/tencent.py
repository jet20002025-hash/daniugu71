import csv
import datetime as dt
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"


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
    symbol = f"{_prefix(code)}{code}"
    end_date = dt.date.today().strftime("%Y-%m-%d")
    params = {"param": f"{symbol},day,2020-01-01,{end_date},{count},"}
    resp = session.get(KLINE_URL, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        return []
    data_obj = payload.get("data", {})
    if not isinstance(data_obj, dict):
        return []
    data = data_obj.get(symbol, {})
    if not isinstance(data, dict):
        return []
    rows = data.get("qfqday") or data.get("day") or []
    result: List[KlineRow] = []
    prev_close = None
    for item in rows:
        if len(item) < 6:
            continue
        date = item[0]
        open_ = _safe_float(item[1])
        close = _safe_float(item[2])
        high = _safe_float(item[3])
        low = _safe_float(item[4])
        volume = _safe_float(item[5])
        amount = _safe_float(item[6]) if len(item) > 6 else 0.0
        chg = close - (prev_close if prev_close else open_)
        pct = (chg / prev_close * 100) if prev_close else ((close - open_) / open_ * 100 if open_ else 0.0)
        amplitude = (high - low) / close * 100 if close else 0.0
        result.append(
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
    return result


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
    return fetch_kline("000001", count=120, session=session)
