#!/bin/bash
# 每日更新 K 线缓存（仅刷新最近约 100 根，包含当日）
# 部署后请设置 GPT_DATA_DIR，例如: export GPT_DATA_DIR=/data/gpt
# cron 示例（每天 18:05 执行）: 5 18 * * * /path/to/scripts/update_kline_daily.sh >> /var/log/stock_kline.log 2>&1

set -e
cd "$(dirname "$0")/.."
export GPT_DATA_DIR="${GPT_DATA_DIR:-$(pwd)/data/gpt}"

echo "[$(date)] 开始每日 K 线更新"
python -m scripts.prefetch_kline_tencent --count 100 --max-age-days 0 --workers 8
echo "[$(date)] 每日 K 线更新完成"
