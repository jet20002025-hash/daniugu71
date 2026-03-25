#!/usr/bin/env python3
"""
将指定用户设为「已开通、无截止日期」并延长试用期备份，避免订阅页拦截。

用法（项目根目录）：
  python3 scripts/set_user_unlimited_subscription.py superzwj

数据库：环境变量 STOCK_APP_DB，否则默认 data/users.db。

说明：activated_until 置空时，程序认为收费会员未过期（见 app.database.User.subscription_expired）。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.chdir(ROOT)

from app.database import get_connection, get_user_by_username, set_activated  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python3 scripts/set_user_unlimited_subscription.py <用户名>")
        return 1
    username = sys.argv[1].strip()
    u = get_user_by_username(username)
    if not u:
        print(f"用户不存在: {username}")
        return 1
    set_activated(u.id, True, activated_until=None)
    far = (datetime.now() + timedelta(days=365 * 30)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    try:
        if username.lower() == "superzwj":
            conn.execute(
                "UPDATE users SET trial_ends_at = ?, is_admin = 1, is_super_admin = 1 WHERE id = ?",
                (far, u.id),
            )
        else:
            conn.execute("UPDATE users SET trial_ends_at = ? WHERE id = ?", (far, u.id))
        conn.commit()
    finally:
        conn.close()
    u2 = get_user_by_username(username)
    extra = "（含管理员+超级管理员）" if username.lower() == "superzwj" else ""
    print(f"已设置「{username}」：已开通、无截止日期{extra}。")
    print(f"  can_use={u2.can_use}  activated_until={u2.activated_until!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
