#!/bin/bash
# 把本地 K 线缓存打包并上传到服务器，覆盖服务器上的缓存，使网络版与本地结果一致。
# 用法：
#   1. 本机执行：./scripts/sync_kline_to_server.sh [服务器用户@IP]
#   2. 按提示在服务器上执行解压与权限命令
#
# 示例：./scripts/sync_kline_to_server.sh admin@47.82.88.220

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# 本地缓存目录（与 app/paths.py 默认一致；若设了 GPT_DATA_DIR 请改下面一行）
GPT="${GPT_DATA_DIR:-$ROOT/data/gpt}"
CACHE="$GPT/kline_cache_tencent"
ARCHIVE="$ROOT/kline_cache_tencent.tar.gz"

if [ ! -d "$CACHE" ]; then
  echo "错误：本地缓存目录不存在: $CACHE"
  echo "若 K 线在其它目录，请设置: export GPT_DATA_DIR=/你的gpt目录"
  exit 1
fi

echo "打包: $CACHE -> $ARCHIVE"
# 在 Mac 上禁用扩展属性，避免在 Linux 解压时出现 LIBARCHIVE.xattr.com.apple.provenance 警告
export COPYFILE_DISABLE=1 2>/dev/null || true
tar -czf "$ARCHIVE" -C "$GPT" kline_cache_tencent
echo "已生成: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"

if [ -n "$1" ]; then
  DEST="$1"
  echo "上传到 $DEST ..."
  scp "$ARCHIVE" "$DEST:/tmp/kline_cache_tencent.tar.gz"
  echo ""
  echo "请在服务器上执行以下命令（覆盖服务器缓存并设权限）："
  echo "  sudo mkdir -p /data/gpt"
  echo "  sudo chown -R \$(whoami) /data/gpt"
  echo "  cd /data/gpt && rm -rf kline_cache_tencent && tar -xzf /tmp/kline_cache_tencent.tar.gz && chown -R \$(whoami) kline_cache_tencent"
  echo "  rm /tmp/kline_cache_tencent.tar.gz"
else
  echo "未传服务器参数，仅打包完成。"
  echo "可选：用 GitHub Release 做中转 → 见 DEPLOY.md 2.6「用 GitHub Release 做中转」"
  echo "上传示例: scp $ARCHIVE 管理员用户@服务器IP:/tmp/"
  echo "服务器上解压: cd /data/gpt && rm -rf kline_cache_tencent && tar -xzf /tmp/kline_cache_tencent.tar.gz && chown -R 运行用户 kline_cache_tencent"
fi
