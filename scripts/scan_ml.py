import argparse
import os

from app.eastmoney import fetch_stock_list
from app.ml_model import MLConfig, load_model_bundle, scan_with_model
from app.scanner import serialize_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan stocks with the trained ML model.")
    parser.add_argument("--max-results", type=int, default=200, help="Max results")
    parser.add_argument("--signal-lookback", type=int, default=5, help="Signal lookback days")
    parser.add_argument("--count", type=int, default=200, help="Kline rows to fetch per stock")
    parser.add_argument("--end-date", default=None, help="Cutoff date YYYY-MM-DD")
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument(
        "--signal-type",
        choices=["aggressive", "relaxed"],
        default="aggressive",
        help="Signal detection mode",
    )
    args = parser.parse_args()

    model_path = os.path.join("data", "models", "ml_model.pkl")
    meta_path = os.path.join("data", "models", "ml_model_meta.json")
    model, meta = load_model_bundle(model_path, meta_path)
    if model is None or meta is None:
        raise RuntimeError("未找到ML模型，请先运行 scripts/train_ml.py")

    config = MLConfig(
        signal_lookback=args.signal_lookback,
        max_results=args.max_results,
        count=args.count,
        end_date=args.end_date,
        start_date=args.start_date,
        signal_type=args.signal_type,
    )
    config.count = max(config.count, config.year_lookback + 5)

    stock_list = fetch_stock_list()
    results = scan_with_model(
        stock_list=stock_list,
        model=model,
        config=config,
        cache_dir="data/kline_cache",
        cutoff_date=args.end_date,
        start_date=args.start_date,
    )

    json_path = os.path.join("data", "results", "latest.json")
    meta_path_out = os.path.join("data", "results", "latest_meta.json")
    csv_path = os.path.join("data", "results", "latest.csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        import json

        json.dump(serialize_results(results), handle, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.writer(handle)
        writer.writerow(["code", "name", "score", "close", "pct_chg", "reasons"])
        for r in results:
            writer.writerow([r.code, r.name, r.score, r.latest_close, r.change_pct, " ".join(r.reasons)])

    with open(meta_path_out, "w", encoding="utf-8") as handle:
        import json
        from datetime import datetime

        json.dump(
            {"updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "count": len(results), "model": "ml"},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"输出完成: {json_path} / {csv_path}")


if __name__ == "__main__":
    main()
