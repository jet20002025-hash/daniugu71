import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from app.eastmoney import read_cached_kline_by_code
from app.ml_model import MLConfig, _build_features, _detect_signals, load_model_bundle
from app.paths import GPT_DATA_DIR


def _parse_date(value: str) -> Optional[datetime.date]:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _load_samples(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = str(row.get("code", "")).strip()
            date = str(row.get("date", "")).strip()
            if code and date:
                rows.append({"code": code, "date": date})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score sample stocks on their sample dates.")
    parser.add_argument(
        "--samples",
        default=os.path.join(GPT_DATA_DIR, "bull_samples.csv"),
        help="Sample CSV path",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"),
        help="Kline cache directory",
    )
    parser.add_argument("--model", default="data/models/ml_model.pkl", help="ML model path")
    parser.add_argument("--meta", default="data/models/ml_model_meta.json", help="ML model meta path")
    parser.add_argument("--min-score", type=int, default=70, help="Score threshold for comparison")
    parser.add_argument(
        "--signal-type",
        choices=["aggressive", "relaxed"],
        default="aggressive",
        help="Signal detection mode",
    )
    parser.add_argument(
        "--allow-no-signal",
        action="store_true",
        help="Force score even if no signal on the given date",
    )
    args = parser.parse_args()

    model, meta = load_model_bundle(args.model, args.meta)
    if model is None or meta is None:
        raise RuntimeError("未找到ML模型，请先训练模型。")

    config = MLConfig(min_score=args.min_score, signal_type=args.signal_type)
    meta_features = meta.get("feature_names", []) if isinstance(meta, dict) else []
    expected_len = len(meta_features) if meta_features else getattr(model, "n_features_in_", None)
    samples = _load_samples(args.samples)

    print("code,date,has_signal,score,proba,reason")
    for item in samples:
        code = item["code"]
        sample_date = item["date"]
        rows = read_cached_kline_by_code(args.cache_dir, code)
        if not rows:
            print(f"{code},{sample_date},0,,,NO_DATA")
            continue

        idx_by_date = {r.date: i for i, r in enumerate(rows)}
        signal_idx = idx_by_date.get(sample_date)
        if signal_idx is None:
            print(f"{code},{sample_date},0,,,DATE_NOT_FOUND")
            continue

        signals = _detect_signals(rows, config.signal_type)
        signal_lookup = {s[0]: s for s in signals}
        forced = False
        sig = signal_lookup.get(signal_idx)
        if sig is None:
            if not args.allow_no_signal:
                print(f"{code},{sample_date},0,,,NO_SIGNAL")
                continue
            forced = True
            sig = (signal_idx, signal_idx, signal_idx)

        if signal_idx - config.year_lookback < 0:
            print(f"{code},{sample_date},0,,,INSUFFICIENT_1Y")
            continue

        base = rows[signal_idx - config.year_lookback].close
        if base > 0:
            year_return = (rows[signal_idx].close - base) / base * 100
            if year_return >= config.year_return_limit:
                print(f"{code},{sample_date},0,,,{year_return:.2f}>=LIMIT")
                continue

        feature_row = _build_features(
            rows=rows,
            signal_idx=sig[0],
            shake_idx=sig[1],
            stop_idx=sig[2],
            market=1 if code.startswith("6") else 0,
        )
        if feature_row is None or not np.all(np.isfinite(feature_row)):
            print(f"{code},{sample_date},0,,,FEATURE_FAIL")
            continue
        if expected_len is not None:
            feature_row = feature_row[:expected_len]

        proba = float(model.predict_proba([feature_row])[0][1])
        score = int(round(proba * 100))
        if forced:
            reason = "FORCED_OK" if score >= args.min_score else "FORCED_SCORE_BELOW"
        else:
            reason = "OK" if score >= args.min_score else "SCORE_BELOW"
        print(f"{code},{sample_date},1,{score},{proba:.4f},{reason}")


if __name__ == "__main__":
    main()
