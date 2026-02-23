import argparse
import csv
import os
import socket
from collections import defaultdict
from datetime import datetime
from statistics import mean, median
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.eastmoney import read_cached_kline, read_cached_kline_by_code, stock_items_from_list_csv
from app.paths import GPT_DATA_DIR

try:
    import akshare as ak
except Exception:
    ak = None


DISABLE_PROXIES = True
FORCE_IPV4 = False


def _disable_proxies() -> None:
    if not DISABLE_PROXIES:
        return
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)


def _force_ipv4() -> None:
    if not FORCE_IPV4:
        return
    try:
        import urllib3.util.connection as urllib3_cn

        def allowed_gai_family():
            return socket.AF_INET

        urllib3_cn.allowed_gai_family = allowed_gai_family
    except Exception:
        pass


def _parse_date(value: str) -> Optional[datetime.date]:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _in_range(date_str: str, start: Optional[str], end: Optional[str]) -> bool:
    if not start and not end:
        return True
    d = _parse_date(date_str)
    if d is None:
        return False
    if start:
        s = _parse_date(start)
        if s and d < s:
            return False
    if end:
        e = _parse_date(end)
        if e and d > e:
            return False
    return True


def _detect_cache_format(cache_dir: str) -> str:
    try:
        for name in os.listdir(cache_dir):
            if not name.endswith(".csv"):
                continue
            stem = name[:-4]
            if stem.startswith(("0_", "1_")):
                return "secid"
    except Exception:
        return "code"
    return "code"


def _load_rows(cache_dir: str, cache_format: str, market: int, code: str):
    if cache_format == "secid":
        path = os.path.join(cache_dir, f"{market}_{code}.csv")
        return read_cached_kline(path)
    return read_cached_kline_by_code(cache_dir, code)


def _moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    res = np.full_like(values, np.nan, dtype=float)
    if len(values) < window:
        return res
    weights = np.ones(window, dtype=float) / window
    res[window - 1 :] = np.convolve(values, weights, mode="valid")
    return res


def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full_like(values, np.nan, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    for i in range(window - 1, len(values)):
        out[i] = np.max(values[i - window + 1 : i + 1])
    return out


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    res = np.full_like(values, np.nan, dtype=float)
    if len(values) < window:
        return res
    weights = np.ones(window, dtype=float)
    res[window - 1 :] = np.convolve(values, weights, mode="valid")
    return res


def _load_index_close(path: str) -> Dict[str, float]:
    if not path or not os.path.exists(path):
        return {}
    closes: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            date = str(row.get("date", "")).strip()
            try:
                close = float(row.get("close", 0) or 0)
            except Exception:
                close = 0.0
            if date and close:
                closes[date] = close
    return closes


def _index_ret20_map(index_close: Dict[str, float]) -> Dict[str, float]:
    dates = sorted(index_close.keys())
    ret_map: Dict[str, float] = {}
    for i in range(20, len(dates)):
        d = dates[i]
        base = index_close[dates[i - 20]]
        if base > 0:
            ret_map[d] = (index_close[d] - base) / base * 100
    return ret_map


def _cache_path(base: str, *parts: str) -> str:
    path = os.path.join(base, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _normalize_name(name: str) -> str:
    if not name:
        return ""
    value = str(name)
    for token in ["行业", "板块", "概念", "指数", "类"]:
        value = value.replace(token, "")
    for token in ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ", "Ⅸ", "Ⅹ"]:
        value = value.replace(token, "")
    for token in [" ", "_", "-", "/", "\\", "（", "）", "(", ")", "·", "—", "－"]:
        value = value.replace(token, "")
    return value.strip().lower()


def _list_flow_files(cache_dir: str) -> List[str]:
    flow_dir = os.path.join(cache_dir, "fund_flow")
    if not os.path.exists(flow_dir):
        return []
    names = []
    for name in os.listdir(flow_dir):
        if name.endswith(".csv"):
            names.append(name[:-4])
    return names


def _resolve_flow_name(industry: str, flow_names: List[str]) -> Optional[str]:
    if not industry or not flow_names:
        return None
    target = _normalize_name(industry)
    if not target:
        return None
    normalized = {n: _normalize_name(n) for n in flow_names}
    for name, norm in normalized.items():
        if norm == target:
            return name
    for name, norm in normalized.items():
        if target in norm or norm in target:
            return name
    return None


def _load_or_fetch_industry(code: str, cache_dir: str) -> Optional[str]:
    path = _cache_path(cache_dir, "industry", f"{code}.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = handle.read().strip()
                return value or None
        except Exception:
            pass
    if ak is None:
        return None
    try:
        _disable_proxies()
        df = ak.stock_individual_info_em(symbol=code)
        if df is None or df.empty:
            return None
        row = df[df["item"] == "所属行业"]
        if row.empty:
            return None
        value = str(row["value"].iloc[0]).strip()
        if value:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(value)
            return value
    except Exception:
        return None
    return None


def _load_or_fetch_industry_hist(industry: str, cache_dir: str) -> Optional[List[Dict[str, str]]]:
    if not industry:
        return None
    safe_name = industry.replace("/", "_").replace(" ", "_")
    path = _cache_path(cache_dir, "industry_hist", f"{safe_name}.csv")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
        except Exception:
            pass
    if ak is None:
        return None
    try:
        _disable_proxies()
        df = ak.stock_board_industry_hist_em(symbol=industry)
        if df is None or df.empty:
            return None
        df.to_csv(path, index=False)
        with open(path, "r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return None


def _industry_ret20_map(rows: List[Dict[str, str]]) -> Dict[str, float]:
    if not rows:
        return {}
    dates: List[str] = []
    pct_vals: List[float] = []
    close_vals: List[float] = []
    for row in rows:
        date = row.get("日期") or row.get("date") or row.get("Date")
        pct = row.get("涨跌幅") or row.get("pct_chg") or row.get("涨跌幅(%)")
        close = row.get("收盘") or row.get("收盘价") or row.get("close")
        if not date:
            continue
        dates.append(str(date)[:10])
        if pct is not None:
            try:
                pct_val = float(str(pct).replace("%", ""))
                pct_vals.append(pct_val / 100.0)
                close_vals.append(float("nan"))
                continue
            except Exception:
                pass
        if close is not None:
            try:
                close_vals.append(float(str(close).replace(",", "")))
            except Exception:
                close_vals.append(float("nan"))
        else:
            close_vals.append(float("nan"))

    if len(pct_vals) >= 21:
        ret_map: Dict[str, float] = {}
        for i in range(20, len(pct_vals)):
            window = pct_vals[i - 19 : i + 1]
            cumulative = 1.0
            for p in window:
                cumulative *= 1.0 + p
            ret_map[dates[i]] = (cumulative - 1.0) * 100
        return ret_map

    valid_close = [v for v in close_vals if not np.isnan(v)]
    if len(valid_close) < 21:
        return {}
    ret_map: Dict[str, float] = {}
    for i in range(20, len(close_vals)):
        base = close_vals[i - 20]
        if np.isnan(base) or base <= 0:
            continue
        cur = close_vals[i]
        if np.isnan(cur):
            continue
        ret_map[dates[i]] = (cur - base) / base * 100
    return ret_map


def _load_or_fetch_industry_flow(industry: str, cache_dir: str, flow_names: List[str]) -> Dict[str, float]:
    if not industry:
        return {}
    safe_name = industry.replace("/", "_").replace(" ", "_")
    path = _cache_path(cache_dir, "fund_flow", f"{safe_name}.csv")
    if not os.path.exists(path):
        resolved = _resolve_flow_name(industry, flow_names)
        if resolved:
            path = _cache_path(cache_dir, "fund_flow", f"{resolved}.csv")
    if os.path.exists(path):
        try:
            flow: Dict[str, float] = {}
            with open(path, "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    date = str(row.get("date", "")).strip()
                    try:
                        net = float(row.get("net", 0) or 0)
                    except Exception:
                        net = 0.0
                    if date:
                        flow[date] = net
            return flow
        except Exception:
            pass
    return {}


def _calc_multiple(
    rows, buy_idx: int, hold_days: int, multiple: float
) -> Tuple[int, float, Optional[str]]:
    exit_idx = buy_idx + hold_days
    if buy_idx < 0 or exit_idx >= len(rows):
        return 0, 0.0, None
    buy_price = rows[buy_idx].open
    if buy_price <= 0:
        return 0, 0.0, None
    max_high = max(r.high for r in rows[buy_idx : exit_idx + 1])
    if max_high <= 0:
        return 0, 0.0, None
    hit_date = None
    if max_high / buy_price >= multiple:
        for i in range(buy_idx, exit_idx + 1):
            if rows[i].high / buy_price >= multiple:
                hit_date = rows[i].date
                break
        return 1, max_high / buy_price, hit_date
    return 0, max_high / buy_price, hit_date


def _signals_mode1(
    rows,
    dates: List[str],
    close: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    pct_chg: np.ndarray,
    amount5: np.ndarray,
    amount20: np.ndarray,
    ma10: np.ndarray,
    ma20: np.ndarray,
    vol20: np.ndarray,
    ret20: np.ndarray,
    turnover: np.ndarray,
    ak_loader,
    index_ret20: Dict[str, float],
    start: Optional[str],
    end: Optional[str],
    require_ak: bool,
    debug: Optional[Dict[str, int]] = None,
) -> List[int]:
    signals = []
    ak_loaded = False
    industry_ret20 = None

    def _bump(key: str) -> None:
        if debug is None:
            return
        debug[key] = debug.get(key, 0) + 1
    for i in range(60, len(rows)):
        _bump("total")
        if not _in_range(dates[i], start, end):
            _bump("skip_date")
            continue
        if np.isnan(ma20[i]) or np.isnan(ma10[i]) or np.isnan(vol20[i]):
            _bump("skip_ma")
            continue
        # consolidation in last 30 days
        box_high = np.max(high[i - 30 : i])
        box_low = np.min(low[i - 30 : i])
        if box_low <= 0:
            _bump("skip_box_low")
            continue
        range_pct = (box_high - box_low) / box_low * 100
        if range_pct > 25:
            _bump("skip_range_pct")
            continue
        # avoid extended
        if ret20[i] is not None and ret20[i] > 20:
            _bump("skip_ret20")
            continue
        if close[i] > ma20[i] * 1.12:
            _bump("skip_far_ma20")
            continue
        if close[i] < ma10[i] * 0.95:
            _bump("skip_below_ma10")
            continue

        # tight consolidation before startup (shrink range + shrink volume)
        recent_high = np.max(high[i - 8 : i])
        recent_low = np.min(low[i - 8 : i])
        if recent_low <= 0:
            _bump("skip_recent_low")
            continue
        tight_pct = (recent_high - recent_low) / recent_low * 100
        if tight_pct > 15:
            _bump("skip_tight_pct")
            continue
        recent_vol = np.mean(volume[i - 8 : i])
        if recent_vol > vol20[i] * 1.2:
            _bump("skip_vol_shrink")
            continue

        # avoid late-stage after multiple limit-ups
        if np.sum(pct_chg[i - 6 : i] >= 9.5) > 2:
            _bump("skip_limitups")
            continue
        if ak_loader and not ak_loaded:
            industry_ret20, _, _ = ak_loader()
            ak_loaded = True

        idx_ret = index_ret20.get(dates[i])
        ind_ret = industry_ret20.get(dates[i]) if industry_ret20 else None
        if ind_ret is not None and idx_ret is not None:
            if (ind_ret - idx_ret) < 1:
                _bump("skip_industry_weak")
                continue
        elif require_ak:
            _bump("skip_industry_missing")
            continue

        amt5 = amount5[i] if not np.isnan(amount5[i]) else None
        amt20 = amount20[i] if not np.isnan(amount20[i]) else None
        if amt5 is None or amt20 is None or amt20 <= 0:
            _bump("skip_amount_missing")
            continue
        if (amt5 / amt20) < 1.1:
            _bump("skip_amount_ratio")
            continue
        # turnover expansion
        turn5 = np.mean(turnover[i - 4 : i + 1])
        turn20 = np.mean(turnover[i - 19 : i + 1])
        if turn20 > 0 and (turn5 / turn20) < 1.05:
            _bump("skip_turnover")
            continue
        # pre-breakout: near box high + moderate volume expansion
        if close[i] < box_high * 0.99:
            _bump("skip_below_box")
            continue
        if close[i] > box_high * 1.12:
            _bump("skip_above_box")
            continue
        if volume[i] < vol20[i] * 1.0:
            _bump("skip_volume_low")
            continue
        if volume[i] > vol20[i] * 6.0:
            _bump("skip_volume_high")
            continue
        # close near high
        if high[i] > low[i]:
            if (high[i] - close[i]) / (high[i] - low[i] + 1e-6) > 0.3:
                _bump("skip_close_high")
                continue
        signals.append(i)
        _bump("pass")
    return signals


def _signals_mode2(
    rows,
    dates: List[str],
    close: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    ma10: np.ndarray,
    ma20: np.ndarray,
    vol20: np.ndarray,
    pct_chg: np.ndarray,
    ret20: np.ndarray,
    start: Optional[str],
    end: Optional[str],
) -> List[int]:
    signals = []
    for i in range(60, len(rows) - 5):
        if not _in_range(dates[i], start, end):
            continue
        if pct_chg[i] < 9.5:
            continue
        # pullback in next 1-5 days
        for k in range(i + 1, min(i + 6, len(rows))):
            if np.isnan(ma20[k]) or np.isnan(vol20[k]):
                continue
            if ret20[k] is not None and ret20[k] > 30:
                continue
            near_ma = close[k] <= ma20[k] * 1.02 or close[k] <= ma10[k] * 1.01
            vol_shrink = volume[k] <= vol20[k] * 0.8
            stop_rebound = close[k] >= open_[k] and low[k] >= low[k - 1] * 0.98
            if near_ma and vol_shrink and stop_rebound:
                signals.append(k)
                break
    return signals


def _signals_mode3(
    rows,
    dates: List[str],
    close: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    ret20: np.ndarray,
    start: Optional[str],
    end: Optional[str],
) -> List[int]:
    signals = []
    for i in range(60, len(rows)):
        if not _in_range(dates[i], start, end):
            continue
        if np.isnan(ma10[i]) or np.isnan(ma20[i]) or np.isnan(ma60[i]) or np.isnan(vol20[i]):
            continue
        if ret20[i] is not None and ret20[i] > 25:
            continue
        ma10_slope = ma10[i] - ma10[i - 3]
        ma20_slope = ma20[i] - ma20[i - 3]
        ma60_slope = ma60[i] - ma60[i - 3]
        if not (
            ma10[i] > ma20[i] > ma60[i]
            and ma10_slope > 0
            and ma20_slope > 0
            and ma60_slope > 0
        ):
            continue
        if close[i] < ma20[i]:
            continue
        if volume[i] < vol20[i] * 1.2:
            continue
        signals.append(i)
    return signals


def _score_mode3(
    close: np.ndarray,
    volume: np.ndarray,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    idx: int,
) -> int:
    score = 60.0
    ma20_now = ma20[idx]
    ma60_now = ma60[idx]
    ma10_now = ma10[idx]
    vol20_now = vol20[idx]
    if ma20_now > 0:
        gap = (ma10_now - ma20_now) / ma20_now
        if gap >= 0.02:
            score += 10
        elif gap >= 0.01:
            score += 6
        elif gap >= 0.005:
            score += 3
    if ma60_now > 0:
        gap = (ma20_now - ma60_now) / ma60_now
        if gap >= 0.02:
            score += 10
        elif gap >= 0.01:
            score += 6
        elif gap >= 0.005:
            score += 3
    if vol20_now > 0:
        vol_ratio = volume[idx] / vol20_now
        if vol_ratio >= 1.6:
            score += 15
        elif vol_ratio >= 1.4:
            score += 10
        elif vol_ratio >= 1.2:
            score += 6
    if ma20_now > 0:
        close_gap = (close[idx] - ma20_now) / ma20_now
        if close_gap >= 0.03:
            score += 5
        elif close_gap >= 0.01:
            score += 3
    # 近3日涨幅超过20%则降分
    if idx >= 3 and close[idx - 3] > 0:
        ret3 = (close[idx] - close[idx - 3]) / close[idx - 3] * 100
        if ret3 > 20:
            score -= 10
        elif ret3 > 15:
            score -= 5
    # 5日线拐头向下：今日MA5低于昨日MA5则降分
    if idx >= 5:
        ma5 = _moving_mean(close, 5)
        if not (np.isnan(ma5[idx]) or np.isnan(ma5[idx - 1])) and ma5[idx] < ma5[idx - 1]:
            score -= 5
    return int(max(0, round(score)))


def _score_mode3_ma20_near(
    close: np.ndarray,
    volume: np.ndarray,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    idx: int,
) -> int:
    """mode3-ma20越近越好：距MA20越近分数越高"""
    score = 60.0
    ma20_now = ma20[idx]
    ma60_now = ma60[idx]
    ma10_now = ma10[idx]
    vol20_now = vol20[idx]
    if ma20_now > 0:
        gap = (ma10_now - ma20_now) / ma20_now
        if gap >= 0.02:
            score += 10
        elif gap >= 0.01:
            score += 6
        elif gap >= 0.005:
            score += 3
    if ma60_now > 0:
        gap = (ma20_now - ma60_now) / ma60_now
        if gap >= 0.02:
            score += 10
        elif gap >= 0.01:
            score += 6
        elif gap >= 0.005:
            score += 3
    if vol20_now > 0:
        vol_ratio = volume[idx] / vol20_now
        if vol_ratio >= 1.6:
            score += 15
        elif vol_ratio >= 1.4:
            score += 10
        elif vol_ratio >= 1.2:
            score += 6
    if ma20_now > 0:
        close_gap = (close[idx] - ma20_now) / ma20_now
        if close_gap <= 0.01:
            score += 5
        elif close_gap <= 0.02:
            score += 3
        elif close_gap <= 0.03:
            score += 1
    # 近3日涨幅超过20%则降分
    if idx >= 3 and close[idx - 3] > 0:
        ret3 = (close[idx] - close[idx - 3]) / close[idx - 3] * 100
        if ret3 > 20:
            score -= 10
        elif ret3 > 15:
            score -= 5
    # 5日线拐头向下：今日MA5低于昨日MA5则降分
    if idx >= 5:
        ma5 = _moving_mean(close, 5)
        if not (np.isnan(ma5[idx]) or np.isnan(ma5[idx - 1])) and ma5[idx] < ma5[idx - 1]:
            score -= 5
    return int(max(0, round(score)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare startup modes for bull signals.")
    parser.add_argument("--start-date", default="2025-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="2025-12-31", help="YYYY-MM-DD")
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--multiple", type=float, default=2.0)
    parser.add_argument("--buy-offset", type=int, default=1)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["all", "mode1", "mode2", "mode3", "mode3_ma20_near"],
        default="all",
        help="Only export this mode (default all).",
    )
    parser.add_argument(
        "--rank-by",
        choices=["multiple", "score"],
        default="multiple",
        help="Daily ranking key.",
    )
    parser.add_argument(
        "--index-path",
        default=os.path.join(GPT_DATA_DIR, "index_sh000001.csv"),
        help="Index kline CSV path",
    )
    parser.add_argument(
        "--ak-cache-dir",
        default="data/akshare_cache",
        help="Akshare cache directory",
    )
    parser.add_argument(
        "--use-akshare",
        action="store_true",
        help="Use akshare industry & fund flow filters",
    )
    parser.add_argument(
        "--keep-proxy",
        action="store_true",
        help="Keep proxy env for akshare",
    )
    parser.add_argument(
        "--force-ipv4",
        action="store_true",
        help="Force IPv4 for akshare",
    )
    parser.add_argument(
        "--require-ak",
        action="store_true",
        help="Require akshare data (skip if missing)",
    )
    parser.add_argument(
        "--debug-mode1",
        action="store_true",
        help="Print mode1 filter diagnostics",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"),
        help="Kline cache dir",
    )
    parser.add_argument(
        "--cache-format",
        choices=["auto", "code", "secid"],
        default="auto",
    )
    parser.add_argument(
        "--stock-list",
        default=os.path.join(GPT_DATA_DIR, "stock_list.csv"),
        help="Stock list CSV",
    )
    parser.add_argument(
        "--output",
        default="data/results/startup_mode_compare.csv",
        help="Output CSV",
    )
    parser.add_argument(
        "--output-xlsx",
        default="",
        help="Output Excel path (optional)",
    )
    args = parser.parse_args()
    global DISABLE_PROXIES
    DISABLE_PROXIES = not args.keep_proxy
    global FORCE_IPV4
    FORCE_IPV4 = args.force_ipv4
    _force_ipv4()

    cache_format = args.cache_format
    if cache_format == "auto":
        cache_format = _detect_cache_format(args.cache_dir)

    stock_list = stock_items_from_list_csv(args.stock_list)
    if not stock_list:
        raise RuntimeError("股票列表为空")

    stats = {
        "mode1": {"signals": 0, "hits": 0, "multiples": [], "hit_days": []},
        "mode2": {"signals": 0, "hits": 0, "multiples": [], "hit_days": []},
        "mode3": {"signals": 0, "hits": 0, "multiples": [], "hit_days": []},
        "mode3_ma20_near": {"signals": 0, "hits": 0, "multiples": [], "hit_days": []},
    }
    daily_picks = {k: defaultdict(list) for k in stats.keys()}

    index_close = _load_index_close(args.index_path)
    index_ret20 = _index_ret20_map(index_close) if index_close else {}
    flow_names = _list_flow_files(args.ak_cache_dir) if args.use_akshare else []

    debug_mode1 = {} if args.debug_mode1 else None

    for item in stock_list:
        rows = _load_rows(args.cache_dir, cache_format, item.market, item.code)
        if not rows or len(rows) < 80:
            continue
        dates = [r.date for r in rows]
        close = np.array([r.close for r in rows], dtype=float)
        open_ = np.array([r.open for r in rows], dtype=float)
        high = np.array([r.high for r in rows], dtype=float)
        low = np.array([r.low for r in rows], dtype=float)
        volume = np.array([r.volume for r in rows], dtype=float)
        pct_chg = np.array([r.pct_chg for r in rows], dtype=float)
        amount = np.array([r.amount for r in rows], dtype=float)
        turnover = np.array([r.turnover for r in rows], dtype=float)
        amount_fixed = np.where(amount > 0, amount, close * volume)
        amount5 = _moving_mean(amount_fixed, 5)
        amount20 = _moving_mean(amount_fixed, 20)

        ak_loader = None
        if args.use_akshare:
            def _loader():
                industry = _load_or_fetch_industry(item.code, args.ak_cache_dir)
                hist_rows = _load_or_fetch_industry_hist(industry, args.ak_cache_dir) if industry else None
                ind_ret = _industry_ret20_map(hist_rows) if hist_rows else None
                return ind_ret, None, None
            ak_loader = _loader

        ma10 = _moving_mean(close, 10)
        ma20 = _moving_mean(close, 20)
        ma60 = _moving_mean(close, 60)
        vol20 = _moving_mean(volume, 20)

        ret20 = [None] * len(rows)
        for i in range(20, len(rows)):
            base = close[i - 20]
            ret20[i] = (close[i] - base) / base * 100 if base else None

        signals1 = _signals_mode1(
            rows,
            dates,
            close,
            open_,
            high,
            low,
            volume,
            pct_chg,
            amount5,
            amount20,
            ma10,
            ma20,
            vol20,
            ret20,
            turnover,
            ak_loader,
            index_ret20,
            args.start_date,
            args.end_date,
            args.require_ak,
            debug_mode1,
        )
        signals2 = _signals_mode2(
            rows,
            dates,
            close,
            open_,
            high,
            low,
            volume,
            ma10,
            ma20,
            vol20,
            pct_chg,
            ret20,
            args.start_date,
            args.end_date,
        )
        signals3 = _signals_mode3(
            rows,
            dates,
            close,
            open_,
            high,
            low,
            volume,
            ma10,
            ma20,
            ma60,
            vol20,
            ret20,
            args.start_date,
            args.end_date,
        )

        year_lookback = 240
        year_high_low_ratio_limit = 4.0  # 近一年最高/最低超4倍则排除

        for mode_key, signals in [
            ("mode1", signals1),
            ("mode2", signals2),
            ("mode3", signals3),
            ("mode3_ma20_near", signals3),
        ]:
            for idx in signals:
                if idx >= year_lookback - 1:
                    start = idx - year_lookback + 1
                    max_high = float(np.max(high[start : idx + 1]))
                    min_low = float(np.min(low[start : idx + 1]))
                    if min_low > 0 and max_high / min_low >= year_high_low_ratio_limit:
                        continue
                buy_idx = idx + max(args.buy_offset, 0)
                has_buy_row = buy_idx < len(rows)
                if has_buy_row:
                    label, mult, hit_date = _calc_multiple(
                        rows, buy_idx, args.hold_days, args.multiple
                    )
                    buy_date_str = rows[buy_idx].date
                else:
                    label, mult, hit_date = 0, 0.0, None
                    buy_date_str = "T+1"
                score = None
                if mode_key == "mode3":
                    score = _score_mode3(close, volume, ma10, ma20, ma60, vol20, idx)
                elif mode_key == "mode3_ma20_near":
                    score = _score_mode3_ma20_near(close, volume, ma10, ma20, ma60, vol20, idx)
                stats[mode_key]["signals"] += 1
                stats[mode_key]["multiples"].append(mult)
                if has_buy_row and label == 1 and hit_date:
                    stats[mode_key]["hits"] += 1
                    hit_days = (
                        _parse_date(hit_date) - _parse_date(rows[buy_idx].date)
                    ).days
                    stats[mode_key]["hit_days"].append(hit_days)
                pick = {
                    "date": dates[idx],
                    "code": item.code,
                    "name": item.name,
                    "mode": mode_key,
                    "buy_date": buy_date_str,
                    "multiple": round(mult, 4),
                    "label": label,
                    "score": score,
                }
                if mode_key in ("mode3", "mode3_ma20_near"):
                    ma20_now = ma20[idx]
                    ma60_now = ma60[idx]
                    ma10_now = ma10[idx]
                    vol_ratio = float(volume[idx] / vol20[idx]) if vol20[idx] > 0 else 0.0
                    close_gap = float(abs(close[idx] - ma20_now) / ma20_now) if ma20_now > 0 else 0.0
                    ma20_gap = float((ma10_now - ma20_now) / ma20_now) if ma20_now > 0 else 0.0
                    ma60_gap = float((ma20_now - ma60_now) / ma60_now) if ma60_now > 0 else 0.0
                    ret20_val = float(ret20[idx]) if ret20[idx] is not None else 0.0
                    pick["close_gap"] = close_gap
                    pick["vol_ratio"] = vol_ratio
                    pick["ma20_gap"] = ma20_gap
                    pick["ma60_gap"] = ma60_gap
                    pick["ret20"] = ret20_val
                daily_picks[mode_key][dates[idx]].append(pick)

    # output per-day topk
    out_rows: List[Dict[str, object]] = []
    for mode_key, by_day in daily_picks.items():
        if args.mode != "all" and mode_key != args.mode:
            continue
        for day, items in by_day.items():
            if args.rank_by == "score" and mode_key in ("mode3", "mode3_ma20_near"):
                items.sort(
                    key=lambda x: (
                        -(x["score"] if x["score"] is not None else -1),
                        x.get("close_gap", 0.0),
                        -x.get("vol_ratio", 0.0),
                        -(x.get("ma20_gap", 0.0) + x.get("ma60_gap", 0.0)),
                        x.get("ret20", 0.0),
                        x["code"],
                    )
                )
            else:
                items.sort(key=lambda x: (-x["multiple"], x["code"]))
            for item in items[: args.topk]:
                out_rows.append(item)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "code",
                "name",
                "mode",
                "buy_date",
                "multiple",
                "label",
                "score",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(out_rows)

    if args.output_xlsx:
        try:
            import pandas as pd

            df = pd.DataFrame(out_rows)
            df.to_excel(args.output_xlsx, index=False)
        except Exception as exc:
            print(f"写入Excel失败: {exc}")

    print("回测区间:", args.start_date, "~", args.end_date)
    for mode_key, data in stats.items():
        signals = data["signals"]
        hits = data["hits"]
        precision = hits / signals if signals else 0.0
        avg_mult = mean(data["multiples"]) if data["multiples"] else 0.0
        med_mult = median(data["multiples"]) if data["multiples"] else 0.0
        avg_hit_days = mean(data["hit_days"]) if data["hit_days"] else 0.0
        print(f"{mode_key}: 信号数 {signals} 命中 {hits} 精确率 {precision:.2%} 平均倍数 {avg_mult:.2f} 中位倍数 {med_mult:.2f} 平均命中天数 {avg_hit_days:.1f}")

    print(f"输出: {args.output}")
    if args.debug_mode1:
        print("mode1 过滤诊断:")
        for key, value in sorted(debug_mode1.items()):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
