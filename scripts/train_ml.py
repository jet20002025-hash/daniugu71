"""
ML 模型训练（激进/宽松买点）。特征仅用信号日当日及历史，禁止使用未来数据。
标签为持有期收益/相对指数超额（事后计算），仅作监督目标；不得用未来数据做特征或样本筛选。
"""
import argparse
import os
from datetime import datetime, timedelta

import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from app.eastmoney import (
    fetch_stock_list,
    get_kline_cached,
    read_cached_kline_by_code,
    stock_items_from_list_csv,
)
from app.ml_model import (
    MLConfig,
    FEATURE_NAMES,
    _load_index_kline_from_csv,
    build_dataset,
    save_model_bundle,
    train_model,
)
from app.paths import GPT_DATA_DIR


def _default_start_date() -> str:
    today = datetime.now().date()
    start = today - timedelta(days=365 * 3)
    return start.strftime("%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML model for aggressive buy signals.")
    parser.add_argument("--start-date", default=_default_start_date(), help="Training start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Training end date YYYY-MM-DD")
    parser.add_argument("--hold-days", type=int, default=40, help="Holding period in trading days")
    parser.add_argument("--return-threshold", type=float, default=8.0, help="Return threshold for label")
    parser.add_argument("--index-excess", type=float, default=3.0, help="Index outperformance threshold")
    parser.add_argument("--count", type=int, default=900, help="Kline rows to fetch per stock")
    parser.add_argument("--cache-days", type=int, default=3650, help="Cache max age days")
    parser.add_argument(
        "--signal-type",
        choices=["aggressive", "relaxed"],
        default="aggressive",
        help="Signal detection mode",
    )
    parser.add_argument("--stock-list", default=None, help="Local stock list CSV path")
    parser.add_argument("--cache-dir", default="data/kline_cache", help="Kline cache dir")
    parser.add_argument("--index-path", default=None, help="Local index CSV path")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use only local cache (no network)",
    )
    args = parser.parse_args()

    config = MLConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        hold_days=args.hold_days,
        return_threshold=args.return_threshold,
        index_excess=args.index_excess,
        count=args.count,
        cache_days=args.cache_days,
        signal_type=args.signal_type,
    )

    print("训练参数:")
    print(f"  start_date: {config.start_date}")
    print(f"  end_date: {config.end_date or '最新'}")
    print(f"  hold_days: {config.hold_days}")
    print(f"  return_threshold: {config.return_threshold}")
    print(f"  index_excess: {config.index_excess}")
    print(f"  count: {config.count}")
    print(f"  signal_type: {config.signal_type}")

    if args.stock_list:
        stock_list = stock_items_from_list_csv(args.stock_list)
    else:
        stock_list = fetch_stock_list()
    print(f"股票数量: {len(stock_list)}")

    index_path = args.index_path
    if args.local_only and not index_path:
        candidate = os.path.join(GPT_DATA_DIR, "index_sh000001.csv")
        if os.path.exists(candidate):
            index_path = candidate
        else:
            raise RuntimeError("本地训练需要提供 --index-path（指数CSV）。")
    index_rows = _load_index_kline_from_csv(index_path) if index_path else None

    if args.local_only:
        base = os.path.basename(args.cache_dir)
        if "tencent" in base or "sina" in base:
            kline_loader = lambda item: read_cached_kline_by_code(args.cache_dir, item.code)
        else:
            kline_loader = lambda item: get_kline_cached(
                item.secid,
                cache_dir=args.cache_dir,
                count=config.count,
                max_age_days=config.cache_days,
                local_only=True,
            )
    else:
        kline_loader = None

    X, y = build_dataset(
        stock_list=stock_list,
        config=config,
        cache_dir=args.cache_dir,
        kline_loader=kline_loader,
        index_rows=index_rows,
    )
    if X.size == 0:
        raise RuntimeError("样本为空，检查数据范围或买点规则。")
    if len(np.unique(y)) < 2:
        raise RuntimeError("样本仅单一类别，无法训练。请调整阈值或规则。")

    print(f"样本量: {len(y)}  正样本: {int(np.sum(y))}  负样本: {int(len(y) - np.sum(y))}")
    print("特征数:", len(FEATURE_NAMES))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    model = train_model(X_train, y_train)

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

    model_path = os.path.join("data", "models", "ml_model.pkl")
    meta_path = os.path.join("data", "models", "ml_model_meta.json")
    save_model_bundle(model, config=config, model_path=model_path, meta_path=meta_path)
    print("模型已保存:")
    print(f"  {model_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
