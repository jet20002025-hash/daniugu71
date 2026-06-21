#!/usr/bin/env bash
# 将仓库内高科技股票池文件安装到 GPT_DATA_DIR（生产默认 /data/gpt）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${GPT_DATA_DIR:-/data/gpt}"
SRC="$ROOT/data/gpt"

if [[ ! -d "$SRC" ]]; then
  echo "错误: 未找到 $SRC" >&2
  exit 1
fi

mkdir -p "$DEST/manual_blocks"
for f in high_tech_blocks.json high_tech_universe.json high_tech_stock_list.csv; do
  if [[ -f "$SRC/$f" ]]; then
    cp "$SRC/$f" "$DEST/$f"
    echo "已复制 $f -> $DEST/$f"
  else
    echo "跳过（不存在）: $SRC/$f" >&2
  fi
done
if [[ -d "$SRC/manual_blocks" ]]; then
  cp -R "$SRC/manual_blocks/." "$DEST/manual_blocks/"
  echo "已复制 manual_blocks -> $DEST/manual_blocks"
fi

n=$(wc -l < "$DEST/high_tech_stock_list.csv" 2>/dev/null || echo 0)
echo "完成。GPT_DATA_DIR=$DEST，CSV 约 $((n - 1)) 行（含表头）。"
