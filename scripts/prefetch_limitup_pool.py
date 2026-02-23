import argparse
import os
import time
from datetime import datetime

import pandas as pd


def _to_datestr(value) -> str:
    if isinstance(value, str):
        raw = value
    else:
        try:
            raw = value.strftime("%Y%m%d")
        except Exception:
            raw = str(value)
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Eastmoney limit-up pool for a date range.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--out-dir", default="data/limitup_pool")
    parser.add_argument("--merged-out", default="data/results/limitup_pool_2026.csv")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument(
        "--index-path",
        default=os.path.join("data", "gpt", "index_sh000001.csv"),
        help="Fallback index CSV for trade dates.",
    )
    args = parser.parse_args()

    try:
        import akshare as ak
    except Exception as exc:
        raise SystemExit(f"akshare not available: {exc}")

    trade_dates = None
    try:
        trade_dates = ak.tool_trade_date_hist_sina()
    except Exception as exc:
        print(f"获取交易日历失败: {exc}，尝试使用本地指数日期")
    if trade_dates is None or trade_dates.empty:
        if os.path.exists(args.index_path):
            df_idx = pd.read_csv(args.index_path)
            if "date" in df_idx.columns:
                trade_dates = df_idx["date"].tolist()
    if trade_dates is None or len(trade_dates) == 0:
        raise SystemExit("无法获取交易日历(线上&本地均失败)")

    # filter by year and optional start/end
    dates = []
    for d in trade_dates:
        ds = _to_datestr(d)
        if len(ds) < 8:
            continue
        if args.start_date and ds < args.start_date.replace("-", ""):
            continue
        if args.end_date and ds > args.end_date.replace("-", ""):
            continue
        if not args.start_date and not args.end_date:
            if not ds.startswith(str(args.year)):
                continue
        dates.append(ds)

    os.makedirs(args.out_dir, exist_ok=True)
    merged_rows = []
    for idx, ds in enumerate(dates, start=1):
        out_path = os.path.join(args.out_dir, f"{ds}.csv")
        if os.path.exists(out_path) and not args.refresh:
            df = pd.read_csv(out_path)
        else:
            try:
                df = ak.stock_zt_pool_em(date=ds)
            except Exception as exc:
                print(f"{ds} failed: {exc}")
                continue
            if df is None or df.empty:
                continue
            df.to_csv(out_path, index=False)
            if args.sleep:
                time.sleep(args.sleep)
        df = df.copy()
        df["date"] = ds
        merged_rows.append(df)
        if idx % 20 == 0 or idx == len(dates):
            print(f"progress {idx}/{len(dates)}")

    if merged_rows:
        merged = pd.concat(merged_rows, ignore_index=True)
        os.makedirs(os.path.dirname(args.merged_out), exist_ok=True)
        merged.to_csv(args.merged_out, index=False)
        print(f"merged saved: {args.merged_out}")
    else:
        print("no data merged")


if __name__ == "__main__":
    main()
