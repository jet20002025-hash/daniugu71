"""
网易财经 K 线数据（日线）。接口：quotes.money.163.com/service/chddata.html
code：上海 0+6位（如 0600519），深圳 1+6位（如 1000001）
"""
import csv
import datetime as dt
import io
import os
from typing import List, Optional

import requests

KLINE_URL = "http://quotes.money.163.com/service/chddata.html"


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _netease_code(code: str) -> str:
    """网易 code：上海 0+代码，深圳 1+代码。"""
    if code.startswith("6"):
        return "0" + code
    return "1" + code


# 与项目内 KlineRow 一致，从 eastmoney 读
from app.eastmoney import KlineRow


def fetch_kline(
    code: str,
    count: int = 300,
    session: Optional[requests.Session] = None,
) -> List[KlineRow]:
    session = session or requests.Session()
    session.trust_env = False
    netease_code = _netease_code(code)
    end = dt.date.today()
    start = end - dt.timedelta(days=min(count * 2, 365 * 2))
    params = {
        "code": netease_code,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "fields": "TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;VOTURNOVER;VATURNOVER",
    }
    resp = session.get(KLINE_URL, params=params, timeout=20)
    if resp.status_code != 200:
        return []
    try:
        text = resp.content.decode("gbk", errors="replace").strip()
    except Exception:
        text = resp.text.strip()
    if not text or "html" in text.lower()[:200]:
        return []
    rows: List[KlineRow] = []
    try:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            # 网易 CSV 表头可能是中文或英文
            date_str = row.get("日期") or row.get("date") or ""
            if not date_str or len(date_str) < 8:
                continue
            if " " in date_str:
                date_str = date_str.split(" ")[0]
            date_str = date_str.replace("/", "-")
            if len(date_str) == 8:
                date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            open_ = _safe_float(row.get("开盘价") or row.get("TOPEN") or row.get("open") or 0)
            close = _safe_float(row.get("收盘价") or row.get("TCLOSE") or row.get("close") or 0)
            high = _safe_float(row.get("最高价") or row.get("HIGH") or row.get("high") or 0)
            low = _safe_float(row.get("最低价") or row.get("LOW") or row.get("low") or 0)
            volume = _safe_float(row.get("成交量") or row.get("VOTURNOVER") or row.get("volume") or 0)
            amount = _safe_float(row.get("成交金额") or row.get("VATURNOVER") or row.get("amount") or 0)
            chg = _safe_float(row.get("涨跌额") or row.get("CHG") or row.get("chg") or 0)
            pct_chg = _safe_float(row.get("涨跌幅") or row.get("PCHG") or row.get("pct_chg") or 0)
            amplitude = (high - low) / close * 100 if close else 0.0
            rows.append(
                KlineRow(
                    date=date_str,
                    open=open_,
                    close=close,
                    high=high,
                    low=low,
                    volume=volume,
                    amount=amount,
                    amplitude=amplitude,
                    pct_chg=pct_chg,
                    chg=chg,
                    turnover=0.0,
                )
            )
    except Exception:
        return []
    # 保持日期升序（与新浪/东财一致），取最近 count 条
    return rows[-count:] if count and len(rows) > count else rows
