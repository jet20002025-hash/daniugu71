#!/bin/bash
# 把本地 K 线缓存同步到服务器（网络版与本地一致）。
#
# 推荐（默认）：rsync 增量同步目录，只传变更文件，可断点续传
#   ./scripts/sync_kline_to_server.sh root@47.82.88.220
#   ./scripts/sync_kline_to_server.sh root@47.82.88.220 /data/gpt/kline_cache_tencent
#
# 整包模式：先 tar.gz 再 rsync 上传（可 --partial 续传，适合首次全量）
#   ./scripts/sync_kline_to_server.sh --tar root@47.82.88.220
#
# 环境变量（可选）：
#   GPT_DATA_DIR   本地 gpt 目录，默认项目 data/gpt
#   RSYNC_BANDWIDTH  限速，如 2m（2MB/s）

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GPT="${GPT_DATA_DIR:-$ROOT/data/gpt}"
CACHE="$GPT/kline_cache_tencent"
ARCHIVE="$ROOT/kline_cache_tencent.tar.gz"
MODE="rsync"

usage() {
  cat <<'EOF'
用法:
  sync_kline_to_server.sh [选项] 用户@服务器 [远端目录]

选项:
  --rsync     增量同步目录（默认，推荐）
  --tar       打包 tar.gz 后 rsync 上传（可断点续传）
  -h, --help  显示帮助

示例:
  ./scripts/sync_kline_to_server.sh root@47.82.88.220
  ./scripts/sync_kline_to_server.sh root@47.82.88.220 /data/gpt/kline_cache_tencent
  ./scripts/sync_kline_to_server.sh --tar root@47.82.88.220
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rsync) MODE="rsync"; shift ;;
    --tar) MODE="tar"; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "未知选项: $1" >&2; usage; exit 1 ;;
    *) break ;;
  esac
done

if [ ! -d "$CACHE" ] && [ -d "$ROOT/data/gpt/kline_cache_tencent" ]; then
  echo "提示: GPT_DATA_DIR=$GPT 下无 kline_cache_tencent，已改用: $ROOT/data/gpt" >&2
  GPT="$ROOT/data/gpt"
  CACHE="$GPT/kline_cache_tencent"
fi

if [ ! -d "$CACHE" ]; then
  echo "错误：本地缓存目录不存在: $CACHE" >&2
  echo "若 K 线在其它目录: export GPT_DATA_DIR=/你的gpt目录" >&2
  exit 1
fi

# 跨境 scp 易 stalled：rsync + SSH 保活 + 可选压缩
RSYNC_SSH="ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o TCPKeepAlive=yes -C"
RSYNC_BASE=(rsync -avz --partial --human-readable --progress -e "$RSYNC_SSH")
if [ -n "${RSYNC_BANDWIDTH:-}" ]; then
  RSYNC_BASE+=(--bwlimit="$RSYNC_BANDWIDTH")
fi

do_rsync_dir() {
  local dest="$1"
  local remote_dir="$2"
  local remote="${dest}:${remote_dir}/"
  echo "增量同步: $CACHE/ -> $remote"
  echo "（仅传输有变化的 csv，支持断点续传）"
  ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=8 "$dest" "mkdir -p $(printf '%q' "$remote_dir")"
  "${RSYNC_BASE[@]}" "$CACHE/" "$remote"
  echo ""
  echo "同步完成: $remote"
  if [ -z "${REMOTE_PATH_HINT:-}" ]; then
    echo ""
    echo "若 Web 使用 /data/gpt/kline_cache_tencent，请在服务器执行："
    echo "  sudo mkdir -p /data/gpt"
    echo "  sudo rsync -a ~/kline_cache_tencent/ /data/gpt/kline_cache_tencent/"
    echo "  sudo chown -R \$(whoami) /data/gpt/kline_cache_tencent"
  fi
}

do_rsync_tar() {
  local dest="$1"
  local remote_tgz="${2:-~/kline_cache_tencent.tar.gz}"
  echo "打包: $CACHE -> $ARCHIVE"
  export COPYFILE_DISABLE=1 2>/dev/null || true
  tar -czf "$ARCHIVE" -C "$GPT" kline_cache_tencent
  echo "已生成: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
  echo "上传（rsync 断点续传）: $dest:$remote_tgz"
  "${RSYNC_BASE[@]}" "$ARCHIVE" "${dest}:${remote_tgz}"
  echo ""
  echo "请在服务器上解压到 /data/gpt："
  echo "  sudo mkdir -p /data/gpt"
  echo "  sudo chown -R \$(whoami) /data/gpt"
  echo "  cd /data/gpt && rm -rf kline_cache_tencent && tar -xzf \$HOME/kline_cache_tencent.tar.gz && chown -R \$(whoami) kline_cache_tencent"
  echo "  rm -f \$HOME/kline_cache_tencent.tar.gz"
}

if [ -z "${1:-}" ]; then
  echo "未传服务器参数。仅本地检查通过，缓存: $CACHE ($(find "$CACHE" -name '*.csv' 2>/dev/null | wc -l | tr -d ' ') 个 csv)"
  usage
  exit 0
fi

DEST="$1"
REMOTE_PATH="${2:-}"

if [ "$MODE" = "tar" ]; then
  do_rsync_tar "$DEST" "${REMOTE_PATH:-~/kline_cache_tencent.tar.gz}"
else
  REMOTE_DIR="${REMOTE_PATH:-~/kline_cache_tencent}"
  if [ -z "$REMOTE_PATH" ]; then
    REMOTE_PATH_HINT=1
  fi
  do_rsync_dir "$DEST" "$REMOTE_DIR"
fi
