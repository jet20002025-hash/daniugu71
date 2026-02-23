# Gunicorn 入口：从根目录的 app.py 加载 Flask app（避免与 app 包同名冲突）
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "main_app",
    os.path.join(os.path.dirname(__file__), "app.py"),
)
main_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_app)
app = main_app.app
