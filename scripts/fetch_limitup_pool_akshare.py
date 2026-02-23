import argparse
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch limit-up pool from Eastmoney via akshare.")
    parser.add_argument("--date", required=True, help="YYYYMMDD, e.g. 20260213")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    try:
        import akshare as ak
    except Exception as exc:
        raise SystemExit(f"akshare not available: {exc}")

    df = ak.stock_zt_pool_em(date=args.date)
    if df is None or df.empty:
        print("no data")
        return

    # Normalize column names for downstream use
    rename_map = {}
    for col in df.columns:
        if "代码" in col:
            rename_map[col] = "code"
        if "名称" in col:
            rename_map[col] = "name"
        if "成交额" in col:
            rename_map[col] = "amount"
        if "最新价" in col:
            rename_map[col] = "close"
    df = df.rename(columns=rename_map)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
