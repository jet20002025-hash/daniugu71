#!/usr/bin/env python3
"""
重置任意用户登录密码（直接写 SQLite，用于忘记密码）。

用法（在项目根目录，且已激活 venv 或已安装依赖）：
  python3 scripts/reset_user_password.py superzwj
  python3 scripts/reset_user_password.py superzwj '你的新密码'

数据库路径：环境变量 STOCK_APP_DB，否则默认项目下 data/users.db。
"""
from __future__ import annotations

import getpass
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.chdir(ROOT)

from app.database import get_user_by_username, set_user_password


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python3 scripts/reset_user_password.py <用户名> [新密码]")
        print("示例: python3 scripts/reset_user_password.py superzwj")
        return 1
    username = sys.argv[1].strip()
    if len(sys.argv) >= 3:
        new_pw = sys.argv[2]
    else:
        a = getpass.getpass("新密码: ")
        b = getpass.getpass("再输入一次: ")
        if a != b:
            print("两次输入不一致。")
            return 1
        new_pw = a
    if len(new_pw) < 6:
        print("密码至少 6 位（与网站注册规则一致）。")
        return 1
    if not get_user_by_username(username):
        print(f"用户不存在: {username}（请先注册或通过 create_local_user 创建）")
        return 1
    if set_user_password(username, new_pw):
        print(f"已重置用户「{username}」的密码。")
        return 0
    print("更新失败。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
