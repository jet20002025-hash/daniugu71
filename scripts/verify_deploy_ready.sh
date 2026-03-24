#!/usr/bin/env bash
# 在**项目根目录**执行：检查能否用 Gunicorn 加载 wsgi:app（部署前自检）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -d venv ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi
python3 -c "from wsgi import app; print('wsgi:app OK', app.name)"
if command -v gunicorn >/dev/null 2>&1; then
  echo "gunicorn 已安装，生产启动示例:"
  echo "  cd $ROOT && gunicorn -w 2 -b 127.0.0.1:8080 --timeout 120 wsgi:app"
else
  echo "未检测到 gunicorn，请先: pip install -r requirements.txt"
fi
