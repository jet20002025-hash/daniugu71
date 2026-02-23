import csv
import json
import os
import threading
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from flask import Flask, g, jsonify, redirect, render_template, request, session, url_for

from app import sina, tencent
from app.auth import admin_required, get_current_user, login_required, subscription_required
from app.database import init_db, get_user_by_username, create_user, verify_password, set_activated, list_users
from app.eastmoney import (
    KlineRow,
    fetch_index_kline,
    fetch_stock_list,
    get_kline_cached,
    list_cached_stocks,
    list_cached_stocks_flat,
    list_cached_stocks_secid,
    load_stock_list_csv,
    read_cached_kline_by_code,
    read_cached_kline_by_market_code,
    stock_items_from_list_csv,
)
from app.ml_model import MLConfig, load_model_bundle, scan_with_model
from app.paths import GPT_DATA_DIR, LOCAL_STOCK_LIST, MARKET_CAP_PATH
from app.scanner import (
    ScanConfig,
    ScanResult,
    percentile_ranks,
    scan_with_mode3,
    score_stock,
    serialize_results,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
GPT_CACHE_DIR = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
GPT_STOCK_LIST = os.path.join(GPT_DATA_DIR, "stock_list.csv")
GPT_INDEX_PATH = os.path.join(GPT_DATA_DIR, "index_sh000001.csv")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production-" + os.path.basename(BASE_DIR))

# 首次启动初始化用户表
init_db()
# 管理员账号（多个用逗号分隔），如 ADMIN_USERNAMES=admin
_admin_usernames = [u.strip() for u in os.environ.get("ADMIN_USERNAMES", "").split(",") if u.strip()]
if _admin_usernames:
    from app.database import get_connection
    conn = get_connection()
    try:
        for uname in _admin_usernames:
            conn.execute("UPDATE users SET is_admin = 1 WHERE username = ?", (uname,))
        conn.commit()
    finally:
        conn.close()


scan_state = {
    "running": False,
    "progress": 0,
    "total": 0,
    "message": "",
    "last_run": None,
    "error": None,
    "source": "",
}


@app.before_request
def _before_request():
    g.current_user = get_current_user()


def _parse_date(value: Optional[str]) -> Optional[datetime.date]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _filter_rows_by_date(rows, cutoff_date: Optional[str]):
    if rows is None or not cutoff_date:
        return rows
    cutoff = _parse_date(cutoff_date)
    if not cutoff:
        return rows
    filtered = []
    for row in rows:
        try:
            row_date = datetime.strptime(row.date, "%Y-%m-%d").date()
        except Exception:
            continue
        if row_date <= cutoff:
            filtered.append(row)
    return filtered


def _normalize_code(code: str) -> str:
    value = str(code or "").strip()
    if value.isdigit() and len(value) < 6:
        return value.zfill(6)
    return value


def _load_market_caps(path: str) -> Dict[str, float]:
    if not path or not os.path.exists(path):
        return {}
    mapping: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = _normalize_code(row.get("code", ""))
            if not code:
                continue
            cap_value = row.get("total_cap")
            if cap_value is None or cap_value == "":
                cap_value = row.get("market_cap")
            try:
                cap = float(cap_value)
            except Exception:
                continue
            if cap > 0:
                mapping[code] = cap
    return mapping


def _load_local_index_kline(path: str) -> List[KlineRow]:
    if not os.path.exists(path):
        return []
    rows: List[KlineRow] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append(
                    KlineRow(
                        date=row.get("date", ""),
                        open=float(row.get("open", 0) or 0),
                        close=float(row.get("close", 0) or 0),
                        high=float(row.get("high", 0) or 0),
                        low=float(row.get("low", 0) or 0),
                        volume=float(row.get("volume", 0) or 0),
                        amount=float(row.get("amount", 0) or 0),
                        amplitude=float(row.get("amplitude", 0) or 0),
                        pct_chg=float(row.get("pct_chg", 0) or 0),
                        chg=float(row.get("chg", 0) or 0),
                        turnover=float(row.get("turnover", 0) or 0),
                    )
                )
            except Exception:
                continue
    return rows


class ScanJob:
    def __init__(self, config: ScanConfig):
        self.config = config
        self.session = requests.Session()

    def run(self) -> List[ScanResult]:
        return self.run_with_cutoff(
            cutoff_date=None,
            start_date=None,
            data_source="cache",
            remote_provider="eastmoney",
            prefer_local=True,
            market_caps=None,
        )

    def run_with_cutoff(
        self,
        cutoff_date: Optional[str],
        start_date: Optional[str],
        data_source: str,
        remote_provider: str,
        prefer_local: bool,
        market_caps: Optional[Dict[str, float]] = None,
    ) -> List[ScanResult]:
        start_dt = _parse_date(start_date)
        fetch_count = max(120, self.config.year_lookback + 5)
        if data_source == "cache":
            stock_list = list_cached_stocks(CACHE_DIR)
            if not stock_list:
                raise RuntimeError("本地缓存为空，无法进行筛选。")
            kline_loader = lambda item, session: get_kline_cached(
                item.secid,
                cache_dir=CACHE_DIR,
                count=fetch_count,
                session=session,
                max_age_days=self.config.cache_days,
                pause=0.0,
                local_only=True,
                prefer_local=prefer_local,
            )
            index_kline = fetch_index_kline(self.session)
        elif data_source == "gpt":
            name_map = load_stock_list_csv(GPT_STOCK_LIST)
            cache_dir = GPT_CACHE_DIR
            stock_list = list_cached_stocks_flat(cache_dir, name_map=name_map)
            kline_loader = lambda item, session: read_cached_kline_by_code(cache_dir, item.code)
            if not stock_list:
                raise RuntimeError("gpt股票缓存为空，无法进行筛选。")
            index_kline = _load_local_index_kline(GPT_INDEX_PATH)
        else:
            try:
                stock_list = fetch_stock_list(self.session)
            except Exception:
                stock_list = []
            if not stock_list:
                fallback = stock_items_from_list_csv(GPT_STOCK_LIST)
                if not fallback and os.path.exists(LOCAL_STOCK_LIST):
                    fallback = stock_items_from_list_csv(LOCAL_STOCK_LIST)
                if fallback:
                    stock_list = fallback
            if remote_provider == "tencent":
                kline_loader = lambda item, session: tencent.get_kline_cached(
                    item.code,
                    cache_dir=CACHE_DIR,
                    count=fetch_count,
                    session=session,
                    max_age_days=self.config.cache_days,
                    pause=0.0,
                    prefer_local=prefer_local,
                )
                try:
                    index_kline = tencent.fetch_index_kline(self.session)
                except Exception:
                    index_kline = []
            elif remote_provider == "sina":
                kline_loader = lambda item, session: sina.get_kline_cached(
                    item.code,
                    cache_dir=CACHE_DIR,
                    count=fetch_count,
                    session=session,
                    max_age_days=self.config.cache_days,
                    pause=0.0,
                    prefer_local=prefer_local,
                )
                try:
                    index_kline = sina.fetch_index_kline(self.session)
                except Exception:
                    index_kline = []
            elif remote_provider == "netease":
                raise RuntimeError("网易数据源尚未接入，请先选择腾讯或新浪。")
            else:
                kline_loader = lambda item, session: get_kline_cached(
                    item.secid,
                    cache_dir=CACHE_DIR,
                    count=fetch_count,
                    session=session,
                    max_age_days=self.config.cache_days,
                    pause=0.0,
                    local_only=False,
                    prefer_local=prefer_local,
                )
                try:
                    index_kline = fetch_index_kline(self.session)
                except Exception:
                    index_kline = []
        if self.config.max_market_cap and market_caps:
            filtered = []
            for item in stock_list:
                cap_value = market_caps.get(_normalize_code(item.code))
                if cap_value is None:
                    continue
                if cap_value <= self.config.max_market_cap:
                    filtered.append(item)
            stock_list = filtered
            if not stock_list:
                raise RuntimeError("市值过滤后无股票，请检查 market_cap.csv")

        scan_state["total"] = len(stock_list)
        scan_state["progress"] = 0
        cap_note = ""
        if self.config.max_market_cap:
            if market_caps is None:
                cap_note = "，市值过滤未启用(缺缓存)"
            else:
                cap_note = f"，市值≤{self.config.max_market_cap / 1e8:.0f}亿"
        scan_state["message"] = f"加载股票列表成功{cap_note}，开始处理K线"

        index_kline = _filter_rows_by_date(index_kline, cutoff_date)
        index_return_10d = None
        if len(index_kline) > 12:
            index_return_10d = (
                (index_kline[-1].close - index_kline[-11].close) / index_kline[-11].close * 100
            )

        results: List[Optional[ScanResult]] = [None] * len(stock_list)
        returns_10d: List[Optional[float]] = [None] * len(stock_list)
        rows_cache: List[Optional[List]] = [None] * len(stock_list)

        def worker(idx: int, item) -> Tuple[int, Optional[List]]:
            session = requests.Session()
            try:
                rows = kline_loader(item, session)
            except Exception:
                rows = None
            rows = _filter_rows_by_date(rows, cutoff_date)
            return idx, rows

        # First pass: fetch data concurrently
        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            futures = [executor.submit(worker, idx, item) for idx, item in enumerate(stock_list)]
            for future in as_completed(futures):
                idx, rows = future.result()
                rows_cache[idx] = rows
                scan_state["progress"] += 1
                if scan_state["progress"] % 200 == 0 or scan_state["progress"] == scan_state["total"]:
                    scan_state["message"] = f"下载K线中 {scan_state['progress']}/{scan_state['total']}"

        for idx, item in enumerate(stock_list):
            rows = rows_cache[idx]
            if not rows:
                continue
            if start_dt:
                try:
                    last_dt = datetime.strptime(rows[-1].date, "%Y-%m-%d").date()
                except Exception:
                    continue
                if last_dt < start_dt:
                    continue
            if len(rows) > 11:
                base = rows[-11].close
                if base > 0:
                    returns_10d[idx] = (rows[-1].close - base) / base * 100
            results[idx] = score_stock(
                item=item,
                rows=rows,
                index_return_10d=index_return_10d,
                return_percentile_10d=None,
                config=self.config,
            )

        scan_state["message"] = "计算强度排名"
        returns_clean = [r for r in returns_10d if r is not None]
        percentile_map: Dict[int, float] = {}
        if returns_clean:
            full_values = [r if r is not None else float("-inf") for r in returns_10d]
            percentile_map = percentile_ranks(full_values)

        # Second pass: add percentile scoring
        final_results: List[ScanResult] = []
        for idx, item in enumerate(stock_list):
            result = results[idx]
            if result is None:
                continue
            percentile = percentile_map.get(idx)
            rows = rows_cache[idx]
            if not rows:
                continue
            rescored = score_stock(
                item=item,
                rows=rows,
                index_return_10d=index_return_10d,
                return_percentile_10d=percentile,
                config=self.config,
            )
            if rescored is None:
                continue
            if rescored.score >= self.config.min_score:
                final_results.append(rescored)

        final_results.sort(key=lambda r: (-r.score, -r.change_pct))
        return final_results[: self.config.max_results]


def _score_buckets(results: List[ScanResult], model: str) -> Dict[str, int]:
    if model not in (
        "mode3",
        "mode3ok",
        "mode3_avoid",
        "mode3_upper",
        "mode3_upper_strict",
        "mode3_upper_near",
        "mode4",
    ):
        return {}
    buckets = {"ge_120": 0, "ge_140": 0, "ge_160": 0, "ge_180": 0}
    for r in results:
        if r.score >= 120:
            buckets["ge_120"] += 1
        if r.score >= 140:
            buckets["ge_140"] += 1
        if r.score >= 160:
            buckets["ge_160"] += 1
        if r.score >= 180:
            buckets["ge_180"] += 1
    return buckets


def save_results(results: List[ScanResult], model: str = "rule") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    json_path = os.path.join(RESULTS_DIR, "latest.json")
    csv_path = os.path.join(RESULTS_DIR, "latest.csv")
    meta_path = os.path.join(RESULTS_DIR, "latest_meta.json")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(serialize_results(results), handle, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code", "name", "score", "close", "pct_chg", "reasons"])
        for r in results:
            writer.writerow([r.code, r.name, r.score, r.latest_close, r.change_pct, " ".join(r.reasons)])

    meta_payload = {"updated_at": timestamp, "count": len(results), "model": model}
    score_buckets = _score_buckets(results, model)
    if score_buckets:
        meta_payload["score_buckets"] = score_buckets

    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta_payload, handle, ensure_ascii=False, indent=2)


def load_results() -> List[Dict[str, object]]:
    json_path = os.path.join(RESULTS_DIR, "latest.json")
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_meta() -> Dict[str, object]:
    meta_path = os.path.join(RESULTS_DIR, "latest_meta.json")
    if not os.path.exists(meta_path):
        return {"updated_at": None, "count": 0, "model": "rule"}
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if "model" not in meta:
        meta["model"] = "rule"
    return meta


def run_scan(
    config: ScanConfig,
    cutoff_date: Optional[str] = None,
    start_date: Optional[str] = None,
    data_source: str = "cache",
    remote_provider: str = "eastmoney",
    prefer_local: bool = False,
) -> None:
    scan_state["running"] = True
    scan_state["error"] = None
    scan_state["source"] = (
        f"remote/{remote_provider}" if data_source == "remote" else data_source
    )
    try:
        market_caps = _load_market_caps(MARKET_CAP_PATH)
        market_caps = market_caps or None
        job = ScanJob(config)
        results = job.run_with_cutoff(
            cutoff_date=cutoff_date,
            start_date=start_date,
            data_source=data_source,
            remote_provider=remote_provider,
            prefer_local=prefer_local,
            market_caps=market_caps,
        )
        save_results(results, model="rule")
        scan_state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        msg = str(exc).strip()
        if not msg:
            msg = type(exc).__name__
        scan_state["error"] = f"{msg}\n{traceback.format_exc()}"
    finally:
        scan_state["running"] = False
        scan_state["message"] = "完成" if scan_state["error"] is None else "失败"


def run_ml_scan(
    config: MLConfig,
    cutoff_date: Optional[str] = None,
    start_date: Optional[str] = None,
    data_source: str = "cache",
    remote_provider: str = "eastmoney",
    prefer_local: bool = False,
    model_variant: str = "ml",
) -> None:
    scan_state["running"] = True
    scan_state["error"] = None
    scan_state["source"] = (
        f"remote/{remote_provider}" if data_source == "remote" else data_source
    )
    try:
        market_caps = _load_market_caps(MARKET_CAP_PATH)
        market_caps = market_caps or None
        if model_variant == "ml_bull":
            model_path = os.path.join(BASE_DIR, "data", "models", "ml_bull.pkl")
            meta_path = os.path.join(BASE_DIR, "data", "models", "ml_bull_meta.json")
        else:
            model_path = os.path.join(BASE_DIR, "data", "models", "ml_model.pkl")
            meta_path = os.path.join(BASE_DIR, "data", "models", "ml_model_meta.json")
        model, meta = load_model_bundle(model_path, meta_path)
        if model is None or meta is None:
            raise RuntimeError("未找到ML模型，请先运行训练脚本生成模型。")

        meta_cfg = meta.get("config") if isinstance(meta, dict) else None
        if isinstance(meta_cfg, dict):
            config.signal_type = meta_cfg.get("signal_type", config.signal_type)
            config.year_lookback = meta_cfg.get("year_lookback", config.year_lookback)
            config.year_return_limit = meta_cfg.get(
                "year_return_limit", config.year_return_limit
            )
            config.min_history = meta_cfg.get("min_history", config.min_history)
            config.count = max(config.count, int(meta_cfg.get("count", config.count)))

        if model_variant == "ml_bull":
            meta_min_score = meta.get("min_score") if isinstance(meta, dict) else None
            if isinstance(meta_min_score, (int, float)):
                config.min_score = max(config.min_score, int(meta_min_score))
            meta_strict = meta.get("strict_filter") if isinstance(meta, dict) else None
            if isinstance(meta_strict, bool):
                config.bull_strict = meta_strict
            else:
                config.bull_strict = True
            config.max_per_day = max(config.max_per_day, 5)

        if data_source == "cache":
            stock_list = list_cached_stocks(CACHE_DIR)
            if not stock_list:
                raise RuntimeError("本地缓存为空，无法进行筛选。")
            kline_loader = None
            cache_dir = CACHE_DIR
        elif data_source == "gpt":
            name_map = load_stock_list_csv(GPT_STOCK_LIST)
            cache_dir = GPT_CACHE_DIR
            stock_list = list_cached_stocks_flat(cache_dir, name_map=name_map)
            kline_loader = lambda item: read_cached_kline_by_code(cache_dir, item.code)
            if not stock_list:
                raise RuntimeError("gpt股票缓存为空，无法进行筛选。")
        else:
            try:
                stock_list = fetch_stock_list()
            except Exception:
                stock_list = []
            if not stock_list:
                fallback = stock_items_from_list_csv(GPT_STOCK_LIST)
                if not fallback and os.path.exists(LOCAL_STOCK_LIST):
                    fallback = stock_items_from_list_csv(LOCAL_STOCK_LIST)
                if fallback:
                    stock_list = fallback
            if remote_provider == "tencent":
                cache_dir = CACHE_DIR
                kline_loader = lambda item: tencent.get_kline_cached(
                    item.code,
                    cache_dir=cache_dir,
                    count=config.count,
                    max_age_days=config.cache_days,
                    pause=0.0,
                    prefer_local=False,
                )
            elif remote_provider == "sina":
                cache_dir = CACHE_DIR
                kline_loader = lambda item: sina.get_kline_cached(
                    item.code,
                    cache_dir=cache_dir,
                    count=config.count,
                    max_age_days=config.cache_days,
                    pause=0.0,
                    prefer_local=False,
                )
            elif remote_provider == "netease":
                raise RuntimeError("网易数据源尚未接入，请先选择腾讯或新浪。")
            else:
                cache_dir = CACHE_DIR
                kline_loader = None

        if config.max_market_cap and market_caps:
            filtered = []
            for item in stock_list:
                cap_value = market_caps.get(_normalize_code(item.code))
                if cap_value is None:
                    continue
                if cap_value <= config.max_market_cap:
                    filtered.append(item)
            stock_list = filtered
            if not stock_list:
                raise RuntimeError("市值过滤后无股票，请检查 market_cap.csv")

        scan_state["total"] = len(stock_list)
        scan_state["progress"] = 0
        provider_label = remote_provider if data_source == "remote" else data_source
        cap_note = ""
        if config.max_market_cap:
            if market_caps is None:
                cap_note = "，市值过滤未启用(缺缓存)"
            else:
                cap_note = f"，市值≤{config.max_market_cap / 1e8:.0f}亿"
        scan_state["message"] = f"加载ML模型，开始筛选（{provider_label}）{cap_note}"

        if cutoff_date:
            config.end_date = cutoff_date

        def _ml_progress_cb() -> None:
            scan_state["progress"] += 1
            if scan_state["progress"] % 200 == 0 or scan_state["progress"] == scan_state["total"]:
                scan_state["message"] = f"筛选中 {scan_state['progress']}/{scan_state['total']}"

        results = scan_with_model(
            stock_list=stock_list,
            model=model,
            config=config,
            cache_dir=cache_dir,
            progress_cb=_ml_progress_cb,
            local_only=(data_source != "remote"),
            kline_loader=kline_loader,
            prefer_local=prefer_local,
            cutoff_date=cutoff_date,
            start_date=start_date,
        )
        results = [r for r in results if r.score >= config.min_score]
        save_results(results, model=model_variant)
        scan_state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        msg = str(exc).strip()
        if not msg:
            msg = type(exc).__name__
        scan_state["error"] = f"{msg}\n{traceback.format_exc()}"
    finally:
        scan_state["running"] = False
        scan_state["message"] = "完成" if scan_state["error"] is None else "失败"


def run_mode3_scan(
    config: ScanConfig,
    cutoff_date: Optional[str] = None,
    start_date: Optional[str] = None,
    data_source: str = "cache",
    remote_provider: str = "eastmoney",
    prefer_local: bool = False,
    avoid_big_candle: bool = False,
    prefer_upper_shadow: bool = False,
    require_upper_shadow: bool = False,
    require_vol_ratio: bool = False,
    require_close_gap: bool = False,
    mode4_filters: bool = False,
    model_tag_override: Optional[str] = None,
    use_startup_modes_data: bool = False,
    use_71x_standard: bool = False,
) -> None:
    scan_state["running"] = True
    scan_state["error"] = None
    scan_state["source"] = (
        f"remote/{remote_provider}" if data_source == "remote"
        else ("本地" if data_source == "gpt" else data_source)
    )
    try:
        market_caps = _load_market_caps(MARKET_CAP_PATH)
        market_caps = market_caps or None
        if data_source == "cache":
            stock_list = list_cached_stocks(CACHE_DIR)
            if not stock_list:
                raise RuntimeError("本地缓存为空，无法进行筛选。")
            cache_dir = CACHE_DIR
            kline_loader = None
            local_only = True
        elif data_source == "gpt":
            name_map = load_stock_list_csv(GPT_STOCK_LIST)
            cache_dir = GPT_CACHE_DIR
            stock_list = stock_items_from_list_csv(GPT_STOCK_LIST) if use_startup_modes_data else list_cached_stocks_flat(cache_dir, name_map=name_map)
            kline_loader = lambda item: read_cached_kline_by_code(cache_dir, item.code)
            if not stock_list:
                raise RuntimeError("gpt股票缓存为空，无法进行筛选。")
            local_only = True
        else:
            try:
                stock_list = fetch_stock_list()
            except Exception:
                stock_list = []
            if not stock_list:
                fallback = stock_items_from_list_csv(GPT_STOCK_LIST)
                if not fallback and os.path.exists(LOCAL_STOCK_LIST):
                    fallback = stock_items_from_list_csv(LOCAL_STOCK_LIST)
                if fallback:
                    stock_list = fallback
            if remote_provider == "tencent":
                cache_dir = CACHE_DIR
                kline_loader = lambda item: tencent.get_kline_cached(
                    item.code,
                    cache_dir=cache_dir,
                    count=max(260, config.year_lookback + 5),
                    max_age_days=config.cache_days,
                    pause=0.0,
                    prefer_local=prefer_local,
                )
            elif remote_provider == "sina":
                cache_dir = CACHE_DIR
                kline_loader = lambda item: sina.get_kline_cached(
                    item.code,
                    cache_dir=cache_dir,
                    count=max(260, config.year_lookback + 5),
                    max_age_days=config.cache_days,
                    pause=0.0,
                    prefer_local=prefer_local,
                )
            elif remote_provider == "netease":
                raise RuntimeError("网易数据源尚未接入，请先选择腾讯或新浪。")
            else:
                cache_dir = CACHE_DIR
                kline_loader = None
            local_only = False

        scan_state["total"] = len(stock_list)
        scan_state["progress"] = 0
        provider_label = remote_provider if data_source == "remote" else ("本地股票库" if data_source == "gpt" else data_source)
        cap_note = ""
        if config.max_market_cap:
            if market_caps is None:
                cap_note = "，市值过滤未启用(缺缓存)"
            else:
                cap_note = f"，市值≤{config.max_market_cap / 1e8:.0f}亿"
        mode_label = "mode4" if mode4_filters else ("mode3ok" if model_tag_override == "mode3ok" else "mode3")
        scan_state["message"] = f"加载{mode_label}，开始筛选（{provider_label}）{cap_note}"

        def _progress_cb() -> None:
            scan_state["progress"] += 1
            if scan_state["progress"] % 200 == 0 or scan_state["progress"] == scan_state["total"]:
                scan_state["message"] = f"筛选中 {scan_state['progress']}/{scan_state['total']}"

        results = scan_with_mode3(
            stock_list=stock_list,
            config=config,
            cache_dir=cache_dir,
            progress_cb=_progress_cb,
            local_only=local_only,
            kline_loader=kline_loader,
            prefer_local=prefer_local,
            cutoff_date=cutoff_date,
            start_date=start_date,
            market_caps=market_caps,
            avoid_big_candle=avoid_big_candle,
            prefer_upper_shadow=prefer_upper_shadow,
            require_upper_shadow=require_upper_shadow,
            require_vol_ratio=require_vol_ratio,
            require_close_gap=require_close_gap,
            mode4_filters=mode4_filters,
            use_71x_standard=use_71x_standard,
        )
        if model_tag_override:
            model_tag = model_tag_override
        elif mode4_filters:
            model_tag = "mode4"
        elif require_upper_shadow:
            model_tag = "mode3_upper_strict"
        elif require_vol_ratio or require_close_gap:
            model_tag = "mode3_upper_near"
        elif prefer_upper_shadow:
            model_tag = "mode3_upper"
        else:
            model_tag = "mode3_avoid" if avoid_big_candle else "mode3"
        save_results(results, model=model_tag)
        scan_state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        msg = str(exc).strip()
        if not msg:
            msg = type(exc).__name__
        scan_state["error"] = f"{msg}\n{traceback.format_exc()}"
    finally:
        scan_state["running"] = False
        scan_state["message"] = "完成" if scan_state["error"] is None else "失败"


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.current_user and g.current_user.can_use:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        if not username or not password:
            error = "请填写用户名和密码"
        else:
            user = get_user_by_username(username)
            if user and verify_password(user, password):
                session["user_id"] = user.id
                session.permanent = True
                next_url = request.args.get("next") or url_for("index")
                return redirect(next_url)
            error = "用户名或密码错误"
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.current_user and g.current_user.can_use:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        password2 = request.form.get("password2") or ""
        if not username or not password:
            error = "请填写用户名和密码"
        elif len(username) < 2:
            error = "用户名至少 2 个字符"
        elif password != password2:
            error = "两次密码不一致"
        elif len(password) < 6:
            error = "密码至少 6 位"
        else:
            try:
                user = create_user(username, password)
                if user:
                    session["user_id"] = user.id
                    session.permanent = True
                    return redirect(url_for("index"))
                error = "用户名已被注册"
            except Exception as e:
                import traceback
                traceback.print_exc()
                error = "注册失败，请稍后重试或联系管理员"
    return render_template("register.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))


@app.route("/subscription")
def subscription_blocked():
    """试用到期或未开通时提示页"""
    user = get_current_user()
    if user is None:
        return redirect(url_for("login"))
    if user.can_use:
        return redirect(url_for("index"))
    return render_template("subscription_blocked.html", user=user)


@app.route("/admin")
@admin_required
def admin():
    users = list_users()
    return render_template("admin.html", users=users)


@app.route("/admin/activate/<int:user_id>", methods=["POST"])
@admin_required
def admin_activate(user_id):
    set_activated(user_id, True)
    return redirect(url_for("admin"))


@app.route("/admin/deactivate/<int:user_id>", methods=["POST"])
@admin_required
def admin_deactivate(user_id):
    set_activated(user_id, False)
    return redirect(url_for("admin"))


@app.route("/")
@subscription_required
def index():
    meta = load_meta()
    results = load_results()
    return render_template("index.html", meta=meta, results=results, state=scan_state, user=g.current_user)


@app.route("/scan", methods=["POST"])
@subscription_required
def scan():
    if scan_state["running"]:
        return redirect(url_for("index"))

    mode = request.form.get("mode", "mode3")
    if mode not in ("mode3", "mode3ok", "mode3_avoid", "mode3_upper", "mode3_upper_strict", "mode3_upper_near", "mode4"):
        mode = "mode3"
    cutoff_date = request.form.get("cutoff_date") or None
    start_date = request.form.get("start_date") or None
    data_source = request.form.get("data_source", "gpt")
    remote_provider = request.form.get("remote_provider", "eastmoney")
    prefer_local = request.form.get("prefer_local") == "on"
    signal_type = request.form.get("signal_type", "aggressive")
    min_score = int(request.form.get("min_score", 70))
    raw_cap = str(request.form.get("max_market_cap", "")).strip()
    if raw_cap == "":
        cap_billion = 150.0
    else:
        try:
            cap_billion = float(raw_cap)
        except Exception:
            cap_billion = 150.0
    cap_limit = None if cap_billion <= 0 else cap_billion * 1e8
    max_results = int(request.form.get("max_results", 200))
    workers = int(request.form.get("workers", 6))

    config = ScanConfig(
        min_score=min_score,
        max_results=max_results,
        workers=workers,
        max_market_cap=cap_limit,
    )
    # 仅保留 71 倍模型（mode3）
    use_startup_data = True
    use_71x_standard = mode == "mode3"
    thread = threading.Thread(
        target=run_mode3_scan,
        args=(
            config,
            cutoff_date,
            start_date,
            data_source,
            remote_provider,
            prefer_local,
            mode == "mode3_avoid",
            mode in ("mode3_upper", "mode3_upper_near"),
            mode == "mode3_upper_strict",
            mode == "mode3_upper_near",
            mode == "mode3_upper_near",
            mode == "mode4",
            "mode3ok" if mode == "mode3ok" else None,
            use_startup_data,
            use_71x_standard,
        ),
        daemon=True,
    )
    thread.start()
    time.sleep(0.2)
    return redirect(url_for("index"))


@app.route("/status")
@subscription_required
def status():
    data = dict(scan_state)
    if data.get("source") == "gpt":
        data["source"] = "本地"
    return jsonify(data)


@app.route("/score")
@subscription_required
def score_page():
    return render_template("score.html")


@app.route("/score/query", methods=["GET", "POST"])
@subscription_required
def score_query():
    code = (request.args.get("code") or request.form.get("code") or "").strip()
    if not code:
        return jsonify({"error": "请提供股票代码"}), 400
    try:
        from app.stock_score import score_stock
        cutoff = request.args.get("date") or request.form.get("date") or None
        cache_dir = os.path.join(GPT_DATA_DIR, "kline_cache_tencent")
        result = score_stock(code, cutoff, cache_dir, use_secid=False)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="127.0.0.1", port=port, debug=True)
