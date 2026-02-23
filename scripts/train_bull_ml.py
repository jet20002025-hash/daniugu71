import argparse
import csv
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from app.eastmoney import (
    read_cached_kline,
    read_cached_kline_by_code,
    stock_items_from_list_csv,
)
from app.ml_model import (
    MLConfig,
    _build_features,
    _detect_signals,
    _passes_bull_strict,
    train_model,
    save_model_bundle,
    FEATURE_NAMES,
)
from app.paths import GPT_DATA_DIR


def _default_start_date() -> str:
    today = datetime.now().date()
    start = today - timedelta(days=365 * 3)
    return start.strftime("%Y-%m-%d")


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


def _label_bull(rows, signal_idx: int, hold_days: int, multiple: float) -> Optional[int]:
    buy_idx = signal_idx + 1
    exit_idx = buy_idx + hold_days
    if exit_idx >= len(rows):
        return None
    buy_price = rows[buy_idx].open
    if buy_price <= 0:
        return None
    max_high = max(r.high for r in rows[buy_idx : exit_idx + 1])
    if max_high / buy_price >= multiple:
        return 1
    return 0


def _load_manual_list(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = str(row.get("code", "")).strip()
            date = str(row.get("date", "")).strip()
            name = str(row.get("name", "")).strip()
            if not code:
                continue
            rows.append({"code": code, "date": date, "name": name})
    return rows


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


def _load_rows(
    cache_dir: str,
    cache_format: str,
    market: int,
    code: str,
) -> Optional[List]:
    if cache_format == "secid":
        path = os.path.join(cache_dir, f"{market}_{code}.csv")
        return read_cached_kline(path)
    return read_cached_kline_by_code(cache_dir, code)


def _best_bull_date(
    rows,
    start_date: Optional[str],
    end_date: Optional[str],
    hold_days: int,
    buy_offset: int,
) -> Tuple[Optional[int], float]:
    best_idx = None
    best_multiple = 0.0
    for i, row in enumerate(rows):
        if not _in_range(row.date, start_date, end_date):
            continue
        buy_idx = i + buy_offset
        exit_idx = buy_idx + hold_days
        if buy_idx < 0 or exit_idx >= len(rows):
            continue
        buy_price = rows[buy_idx].open
        if buy_price <= 0:
            continue
        max_high = max(r.high for r in rows[buy_idx : exit_idx + 1])
        if max_high <= 0:
            continue
        multiple = max_high / buy_price
        if multiple > best_multiple:
            best_multiple = multiple
            best_idx = i
    return best_idx, best_multiple


def _find_threshold_for_precision(
    y_true: np.ndarray,
    proba: np.ndarray,
    target_precision: float,
) -> Tuple[float, float, float]:
    if len(y_true) == 0:
        return 1.0, 0.0, 0.0
    order = np.argsort(proba)[::-1]
    total_pos = int(np.sum(y_true))
    tp = 0
    fp = 0
    best_threshold = None
    best_precision = 0.0
    best_recall = 0.0

    for idx in order:
        if y_true[idx] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(total_pos, 1)
        if precision >= target_precision:
            best_threshold = proba[idx]
            best_precision = precision
            best_recall = recall

    if best_threshold is None:
        idx = order[0]
        tp = 1 if y_true[idx] == 1 else 0
        fp = 1 - tp
        best_threshold = proba[idx]
        best_precision = tp / max(tp + fp, 1)
        best_recall = tp / max(total_pos, 1)

    return float(best_threshold), float(best_precision), float(best_recall)


def _build_topk_by_day(
    y_true: np.ndarray,
    proba: np.ndarray,
    dates: List[str],
    k: int,
) -> Dict[str, List[Tuple[float, int]]]:
    grouped: Dict[str, List[int]] = {}
    for idx, date in enumerate(dates):
        grouped.setdefault(date, []).append(idx)

    topk: Dict[str, List[Tuple[float, int]]] = {}
    for date, idxs in grouped.items():
        idxs.sort(key=lambda i: proba[i], reverse=True)
        picks = idxs[:k]
        topk[date] = [(float(proba[i]), int(y_true[i])) for i in picks]
    return topk


def _precision_topk_by_day(
    topk_by_day: Dict[str, List[Tuple[float, int]]],
    threshold: Optional[float] = None,
) -> Tuple[float, int]:
    total_picks = 0
    total_pos = 0
    for picks in topk_by_day.values():
        if threshold is None:
            chosen = picks
        else:
            chosen = [p for p in picks if p[0] >= threshold]
        total_picks += len(chosen)
        total_pos += sum(p[1] for p in chosen)
    precision = total_pos / total_picks if total_picks > 0 else 0.0
    return precision, total_picks


def _find_threshold_for_topk_precision(
    topk_by_day: Dict[str, List[Tuple[float, int]]],
    target_precision: float,
    min_picks: int,
) -> Tuple[float, float, int]:
    scores: List[float] = []
    for picks in topk_by_day.values():
        scores.extend(p[0] for p in picks)
    if not scores:
        return 1.0, 0.0, 0
    unique = np.unique(np.array(scores, dtype=float))
    unique.sort()
    unique = unique[::-1]

    best_threshold = 1.0
    best_precision = 0.0
    best_picks = 0

    for threshold in unique:
        precision, picks = _precision_topk_by_day(
            topk_by_day=topk_by_day, threshold=float(threshold)
        )
        if picks < min_picks:
            continue
        if precision >= target_precision:
            return float(threshold), float(precision), int(picks)
        if precision > best_precision:
            best_precision = precision
            best_threshold = float(threshold)
            best_picks = int(picks)

    return float(best_threshold), float(best_precision), int(best_picks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML model for bull stocks (2x in one month).")
    parser.add_argument("--stock-list", default=os.path.join(GPT_DATA_DIR, "stock_list.csv"))
    parser.add_argument("--cache-dir", default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"))
    parser.add_argument(
        "--cache-format",
        choices=["auto", "code", "secid"],
        default="auto",
        help="Cache filename format (auto|code|secid)",
    )
    parser.add_argument("--start-date", default=_default_start_date())
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--multiple", type=float, default=2.0)
    parser.add_argument(
        "--signal-type",
        choices=["aggressive", "relaxed"],
        default="relaxed",
    )
    parser.add_argument("--count", type=int, default=900)
    parser.add_argument("--min-history", type=int, default=80)
    parser.add_argument("--min-score", type=int, default=0)
    parser.add_argument("--precision-target", type=float, default=0.8)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--min-picks", type=int, default=50)
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="none",
        help="Training weight strategy",
    )
    parser.add_argument(
        "--strict-filter",
        choices=["on", "off"],
        default="on",
        help="Apply strict bull filters before training",
    )
    parser.add_argument(
        "--infer-strict-filter",
        choices=["on", "off"],
        default="on",
        help="Apply strict bull filters during inference",
    )
    parser.add_argument(
        "--manual-list",
        default="data/bull_manual_list.csv",
        help="Optional manual bull list CSV (code,name,date)",
    )
    parser.add_argument(
        "--manual-mode",
        choices=["best", "date"],
        default="best",
        help="best=pick best bull date in range if date missing; date=only use provided dates",
    )
    parser.add_argument("--manual-weight", type=int, default=1, help="Duplicate manual positives")
    parser.add_argument("--manual-buy-offset", type=int, default=1, help="Buy offset for manual samples")
    parser.add_argument("--model-out", default="data/models/ml_bull.pkl")
    parser.add_argument("--meta-out", default="data/models/ml_bull_meta.json")
    args = parser.parse_args()

    config = MLConfig(
        signal_type=args.signal_type,
        start_date=args.start_date,
        end_date=args.end_date,
        min_history=args.min_history,
        count=args.count,
        min_score=args.min_score,
    )

    cache_format = args.cache_format
    if cache_format == "auto":
        cache_format = _detect_cache_format(args.cache_dir)

    stock_list = stock_items_from_list_csv(args.stock_list)
    if not stock_list:
        raise RuntimeError("股票列表为空，无法训练。")

    features: List[List[float]] = []
    labels: List[int] = []
    signal_dates: List[str] = []

    strict_filter = args.strict_filter == "on"
    infer_strict_filter = args.infer_strict_filter == "on"

    for item in stock_list:
        rows = _load_rows(args.cache_dir, cache_format, item.market, item.code)
        if not rows or len(rows) < config.min_history:
            continue

        signals = _detect_signals(rows, config.signal_type)
        if not signals:
            continue

        for signal_idx, shake_idx, stop_idx in signals:
            if not _in_range(rows[signal_idx].date, args.start_date, args.end_date):
                continue
            if signal_idx - config.year_lookback < 0:
                continue
            base = rows[signal_idx - config.year_lookback].close
            if base > 0:
                year_return = (rows[signal_idx].close - base) / base * 100
                if year_return >= config.year_return_limit:
                    continue

            label = _label_bull(rows, signal_idx, args.hold_days, args.multiple)
            if label is None:
                continue

            feature_row = _build_features(
                rows=rows,
                signal_idx=signal_idx,
                shake_idx=shake_idx,
                stop_idx=stop_idx,
                market=1 if item.code.startswith("6") else 0,
            )
            if feature_row is None or not np.all(np.isfinite(feature_row)):
                continue
            if strict_filter and not _passes_bull_strict(feature_row):
                continue

            features.append(feature_row)
            labels.append(label)
            signal_dates.append(rows[signal_idx].date)

    manual_list = _load_manual_list(args.manual_list)
    manual_added = 0
    manual_skipped = 0
    if manual_list:
        code_to_item = {s.code: s for s in stock_list}
        for entry in manual_list:
            code = entry["code"]
            item = code_to_item.get(code)
            if item is None:
                manual_skipped += 1
                continue
            rows = _load_rows(args.cache_dir, cache_format, item.market, item.code)
            if not rows or len(rows) < config.min_history:
                manual_skipped += 1
                continue
            target_date = entry.get("date") or ""
            idx = None
            if target_date:
                idx_by_date = {r.date: i for i, r in enumerate(rows)}
                idx = idx_by_date.get(target_date)
            elif args.manual_mode == "best":
                idx, _multiple = _best_bull_date(
                    rows,
                    args.start_date,
                    args.end_date,
                    args.hold_days,
                    args.manual_buy_offset,
                )
            if idx is None:
                manual_skipped += 1
                continue
            feature_row = _build_features(
                rows=rows,
                signal_idx=idx,
                shake_idx=idx,
                stop_idx=idx,
                market=1 if item.code.startswith("6") else 0,
            )
            if feature_row is None or not np.all(np.isfinite(feature_row)):
                manual_skipped += 1
                continue
            for _ in range(max(args.manual_weight, 1)):
                features.append(feature_row)
                labels.append(1)
                signal_dates.append(rows[idx].date)
                manual_added += 1

    if not features:
        raise RuntimeError("样本为空，无法训练。")

    X = np.array(features, dtype=float)
    y = np.array(labels, dtype=int)
    dates = np.array(signal_dates)

    print(f"样本量: {len(y)} 正样本: {int(np.sum(y))} 负样本: {int(len(y) - np.sum(y))}")
    print(f"特征数: {len(FEATURE_NAMES)}")
    if manual_list:
        print(f"手工样本: {manual_added} 已加入, 跳过: {manual_skipped}")

    X_train, X_val, y_train, y_val, dates_train, dates_val = train_test_split(
        X,
        y,
        dates,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    model = train_model(X_train, y_train, weight_strategy=args.class_weight)

    if len(np.unique(y_val)) > 1:
        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        print(f"验证集 AUC: {auc:.4f}")
    else:
        proba = model.predict_proba(X_val)[:, 1]
        print("验证集仅单一类别，跳过AUC计算。")

    preds = (proba >= 0.5).astype(int)
    print("验证集分类报告:")
    print(classification_report(y_val, preds, digits=4))

    threshold, prec_at_t, recall_at_t = _find_threshold_for_precision(
        y_val, proba, args.precision_target
    )
    min_score = int(round(threshold * 100))
    print(f"目标精确率: {args.precision_target:.2%}")
    print(f"建议阈值: {threshold:.4f} (score>={min_score})")
    print(f"阈值精确率: {prec_at_t:.2%} 召回率: {recall_at_t:.2%}")

    topk_by_day = _build_topk_by_day(
        y_true=y_val, proba=proba, dates=list(dates_val), k=args.topk
    )
    topk_precision, topk_picks = _precision_topk_by_day(topk_by_day)
    print(f"Top{args.topk} 按日精确率: {topk_precision:.2%} (样本数 {topk_picks})")

    topk_threshold, topk_prec, topk_picks = _find_threshold_for_topk_precision(
        topk_by_day=topk_by_day,
        target_precision=args.precision_target,
        min_picks=args.min_picks,
    )
    topk_score = int(round(topk_threshold * 100))
    print(
        f"Top{args.topk} 阈值建议: {topk_threshold:.4f} "
        f"(score>={topk_score}, 精确率 {topk_prec:.2%}, 样本数 {topk_picks})"
    )

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    save_model_bundle(model, config=config, model_path=args.model_out, meta_path=args.meta_out)
    try:
        with open(args.meta_out, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except Exception:
        meta = {}
    meta.update(
        {
            "precision_target": args.precision_target,
            "precision_threshold": threshold,
            "precision_at_threshold": prec_at_t,
            "recall_at_threshold": recall_at_t,
            "min_score": min_score,
            "class_weight": args.class_weight,
            "strict_filter": infer_strict_filter,
            "topk": args.topk,
            "topk_precision": topk_precision,
            "topk_threshold": topk_threshold,
            "topk_min_score": topk_score,
            "topk_picks": topk_picks,
        }
    )
    with open(args.meta_out, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print("模型已保存:")
    print(f"  {args.model_out}")
    print(f"  {args.meta_out}")


if __name__ == "__main__":
    main()
