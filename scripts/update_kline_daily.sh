#!/bin/bash
# 每日更新 K 线缓存（仅刷新最近约 100 根，包含当日）
# 部署后请设置 GPT_DATA_DIR，例如: export GPT_DATA_DIR=/data/gpt
# cron 示例（每个交易日 15:10 执行）:
#   10 15 * * 1-5 /var/www/stock-app/scripts/update_kline_daily.sh >> /var/www/stock-app/kline_update.log 2>&1

set -e
cd "$(dirname "$0")/.."
export GPT_DATA_DIR="${GPT_DATA_DIR:-/data/gpt}"

# 优先使用项目 venv 的 Python
if [ -x "venv/bin/python" ]; then
  PYTHON="venv/bin/python"
else
  PYTHON="python3"
fi

echo "[$(date)] 开始每日 K 线更新"
$PYTHON -m scripts.prefetch_kline_tencent --count 100 --max-age-days 0 --workers 8
echo "[$(date)] 每日 K 线更新完成"
