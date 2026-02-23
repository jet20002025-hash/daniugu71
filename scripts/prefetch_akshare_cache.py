import argparse
import csv
import os
import socket
import time
from typing import Dict, List, Optional, Set, Tuple

from app.eastmoney import stock_items_from_list_csv
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


def _cache_path(base: str, *parts: str) -> str:
    path = os.path.join(base, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _load_cached_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = handle.read().strip()
            return value or None
    except Exception:
        return None


def _save_text(path: str, value: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(value)


def _fetch_industry(code: str) -> Optional[str]:
    if ak is None:
        return None
    _disable_proxies()
    df = ak.stock_individual_info_em(symbol=code)
    if df is None or df.empty:
        return None
    row = df[df["item"] == "所属行业"]
    if row.empty:
        return None
    value = str(row["value"].iloc[0]).strip()
    return value or None


def _safe_industry_name(industry: str) -> str:
    return industry.replace("/", "_").replace(" ", "_")


def _fetch_industry_hist(industry: str, start_date: str, end_date: str):
    if ak is None:
        return None
    _disable_proxies()
    df = ak.stock_board_industry_hist_em(symbol=industry, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return None
    return df


def _build_industry_map(
    industries: Set[str],
    stock_codes: Set[str],
    sleep: float,
    verbose: bool,
) -> Dict[str, str]:
    if ak is None:
        return {}
    mapping: Dict[str, str] = {}
    total = len(industries)
    for idx, industry in enumerate(sorted(industries), start=1):
        try:
            _disable_proxies()
            df = ak.stock_board_industry_cons_em(symbol=industry)
            if df is None or df.empty:
                continue
            code_col = None
            for col in df.columns:
                if col in ("代码", "股票代码", "证券代码", "code"):
                    code_col = col
                    break
            if code_col is None:
                for col in df.columns:
                    if "代码" in col or col.lower() == "code":
                        code_col = col
                        break
            if code_col is None:
                continue
            for raw in df[code_col].astype(str):
                code = raw.strip().zfill(6)
                if code in stock_codes and code not in mapping:
                    mapping[code] = industry
        except Exception as exc:
            if verbose:
                print(f"行业成分失败 {industry}: {exc}")
        if sleep:
            time.sleep(sleep)
        if verbose and (idx % 50 == 0 or idx == total):
            print(f"行业成分进度 {idx}/{total} 已映射 {len(mapping)}")
    return mapping


def _normalize_fund_flow(df):
    if df is None or df.empty:
        return None
    date_col = None
    for col in df.columns:
        if "日期" in col or col.lower() == "date":
            date_col = col
            break
    if date_col is None:
        return None
    net_col = None
    for key in ["主力净流入", "主力净额", "净流入", "净额", "资金净流入", "主力净流入-净额"]:
        for col in df.columns:
            if key in col:
                net_col = col
                break
        if net_col:
            break
    if net_col is None:
        for col in df.columns:
            if "净" in col and ("流入" in col or "净额" in col):
                net_col = col
                break
    if net_col is None:
        return None
    out = df[[date_col, net_col]].copy()
    out.columns = ["date", "net"]
    out["date"] = out["date"].astype(str).str.slice(0, 10)
    out["net"] = out["net"].astype(str).str.replace(",", "").str.replace("%", "")
    out["net"] = out["net"].astype(float)
    return out


def _fetch_fund_flow(industry_name: str):
    if ak is None:
        return None
    _disable_proxies()
    try:
        df = ak.stock_sector_fund_flow_hist(symbol=industry_name)
        if df is not None and not df.empty:
            return _normalize_fund_flow(df)
    except Exception as exc:
        print(f"抓取行业 {industry_name} 资金历史失败: {exc}")
    return None


def _load_fund_flow_industries() -> Optional[List[str]]:
    if ak is None:
        return None
    try:
        _disable_proxies()
        df = ak.stock_sector_fund_flow_rank(indicator="今日")
        if df is None or df.empty:
            return None
        name_col = None
        for col in df.columns:
            if col in ("板块名称", "行业名称", "板块", "行业"):
                name_col = col
                break
        if name_col is None:
            for col in df.columns:
                if "板块" in col or "行业" in col:
                    name_col = col
                    break
        if name_col is None:
            return None
        return [str(v).strip() for v in df[name_col].tolist() if str(v).strip()]
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch akshare caches.")
    parser.add_argument(
        "--stock-list",
        default=os.path.join(GPT_DATA_DIR, "stock_list.csv"),
        help="Stock list CSV",
    )
    parser.add_argument(
        "--ak-cache-dir",
        default="data/akshare_cache",
        help="Akshare cache directory",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit stock count")
    parser.add_argument("--sleep", type=float, default=0.4, help="Sleep between calls")
    parser.add_argument("--refresh", action="store_true", help="Refresh existing cache")
    parser.add_argument("--keep-proxy", action="store_true", help="Keep proxy env")
    parser.add_argument("--force-ipv4", action="store_true", help="Force IPv4")
    parser.add_argument("--verbose", action="store_true", help="Print errors")
    parser.add_argument("--hist-start", default="20100101", help="Industry hist start YYYYMMDD")
    parser.add_argument("--hist-end", default="", help="Industry hist end YYYYMMDD")
    parser.add_argument(
        "--only",
        choices=["all", "industry", "industry_hist", "fund_flow"],
        default="all",
    )
    args = parser.parse_args()
    global DISABLE_PROXIES
    DISABLE_PROXIES = not args.keep_proxy
    global FORCE_IPV4
    FORCE_IPV4 = args.force_ipv4
    _force_ipv4()

    if not args.hist_end:
        args.hist_end = time.strftime("%Y%m%d")

    if ak is None:
        raise RuntimeError("akshare 未安装或不可用")

    stock_list = stock_items_from_list_csv(args.stock_list)
    if not stock_list:
        raise RuntimeError("股票列表为空")
    if args.limit and args.limit > 0:
        stock_list = stock_list[: args.limit]

    # --- 新增逻辑：直接获取行业列表 ---
    industries: Set[str] = set()
    if args.only in ("industry", "industry_hist", "fund_flow", "all"):
        try:
            print("正在直接获取全行业列表...")
            if ak is None:
                raise RuntimeError("akshare 未安装或不可用")
            df_all_ind = ak.stock_board_industry_name_em()
            if df_all_ind is not None and not df_all_ind.empty:
                industries.update(df_all_ind["板块名称"].tolist())
                print(f"成功获取 {len(industries)} 个行业名称")
        except Exception as exc:
            print(f"直接获取行业列表失败: {exc}，将尝试从个股推导...")
    # -----------------------------

    fund_flow_industries: List[str] = []
    if args.only in ("fund_flow", "all"):
        flow_list = _load_fund_flow_industries()
        if flow_list:
            fund_flow_industries = flow_list
            print(f"成功获取 {len(fund_flow_industries)} 个资金流行业名称")

    stock_codes = {item.code for item in stock_list}
    industry_map: Dict[str, str] = {}
    if args.only in ("industry", "all") and industries:
        print("正在根据行业成分构建行业映射...")
        industry_map = _build_industry_map(
            industries, stock_codes, args.sleep, args.verbose
        )
        print(f"行业映射完成: {len(industry_map)}")
    total = len(stock_list)
    ok_industry = 0
    ok_flow = 0
    fail_industry = 0
    fail_flow = 0

    if args.only in ("all", "industry"):
        for idx, item in enumerate(stock_list, start=1):
            code = item.code
            ind_path = _cache_path(args.ak_cache_dir, "industry", f"{code}.txt")
            industry = None
            if not args.refresh:
                industry = _load_cached_text(ind_path)
            if not industry and code in industry_map:
                industry = industry_map[code]
                try:
                    _save_text(ind_path, industry)
                except Exception:
                    pass
                ok_industry += 1
                industries.add(industry)
            if not industry:
                try:
                    industry = _fetch_industry(code)
                    if industry:
                        _save_text(ind_path, industry)
                        ok_industry += 1
                    else:
                        fail_industry += 1
                except Exception as exc:
                    fail_industry += 1
                    if args.verbose:
                        print(f"行业失败 {code}: {exc}")
            if industry:
                industries.add(industry)

            if args.sleep:
                time.sleep(args.sleep)

            if idx % 200 == 0 or idx == total:
                print(f"进度 {idx}/{total} 行业OK {ok_industry} 资金OK {ok_flow} 行业失败 {fail_industry} 资金失败 {fail_flow}")

    target_industries = fund_flow_industries or sorted(industries)
    if args.only in ("all", "fund_flow") and target_industries:
        print(f"正在抓取 {len(target_industries)} 个行业的历史资金流...")
        ok_flow = 0
        fail_flow = 0
        for ind in target_industries:
            safe_name = _safe_industry_name(ind)
            flow_path = _cache_path(args.ak_cache_dir, "fund_flow", f"{safe_name}.csv")

            if not args.refresh and os.path.exists(flow_path):
                ok_flow += 1
                continue

            df = _fetch_fund_flow(ind)
            if df is not None:
                df.to_csv(flow_path, index=False)
                ok_flow += 1
            else:
                fail_flow += 1

            if args.sleep:
                time.sleep(args.sleep)
        print(f"资金流同步完成: OK {ok_flow}, 失败 {fail_flow}")

    ok_hist = 0
    fail_hist = 0
    if args.only in ("all", "industry_hist"):
        for ind in sorted(industries):
            safe_name = _safe_industry_name(ind)
            hist_path = _cache_path(args.ak_cache_dir, "industry_hist", f"{safe_name}.csv")
            if not args.refresh and os.path.exists(hist_path):
                ok_hist += 1
                continue
            try:
                df = _fetch_industry_hist(ind, args.hist_start, args.hist_end)
                if df is not None and not df.empty:
                    df.to_csv(hist_path, index=False)
                    ok_hist += 1
                else:
                    fail_hist += 1
            except Exception as exc:
                fail_hist += 1
                if args.verbose:
                    print(f"行业指数失败 {ind}: {exc}")
            if args.sleep:
                time.sleep(args.sleep)

    print("完成:")
    if args.only in ("all", "industry"):
        print(f"  行业: OK {ok_industry}, 失败 {fail_industry}, 唯一行业 {len(industries)}")
    if args.only in ("all", "industry_hist"):
        print(f"  行业指数: OK {ok_hist}, 失败 {fail_hist}")
    if args.only in ("all", "fund_flow"):
        print(f"  资金流: OK {ok_flow}, 失败 {fail_flow}")


if __name__ == "__main__":
    main()
