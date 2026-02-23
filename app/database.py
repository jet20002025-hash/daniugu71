# 用户与订阅数据库（SQLite）
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

from werkzeug.security import generate_password_hash, check_password_hash

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
DB_PATH = os.environ.get("STOCK_APP_DB", os.path.join(PROJECT_DIR, "data", "users.db"))


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    return sqlite3.connect(DB_PATH)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """确保新列存在（迁移）"""
    cur = conn.execute("PRAGMA table_info(users)")
    names = [r[1] for r in cur.fetchall()]
    if "activated_until" not in names:
        conn.execute("ALTER TABLE users ADD COLUMN activated_until TEXT")
    if "is_super_admin" not in names:
        conn.execute("ALTER TABLE users ADD COLUMN is_super_admin INTEGER NOT NULL DEFAULT 0")
    if "register_ip" not in names:
        conn.execute("ALTER TABLE users ADD COLUMN register_ip TEXT")
    # 将 superzwj 设为超级管理员并确保有管理员权限
    conn.execute(
        "UPDATE users SET is_super_admin = 1, is_admin = 1 WHERE username = ?",
        ("superzwj",),
    )
    conn.commit()


def init_db() -> None:
    """创建用户表并执行迁移"""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                trial_ends_at TEXT NOT NULL,
                is_paid INTEGER NOT NULL DEFAULT 0,
                is_activated INTEGER NOT NULL DEFAULT 0,
                is_admin INTEGER NOT NULL DEFAULT 0,
                activated_until TEXT,
                is_super_admin INTEGER NOT NULL DEFAULT 0,
                register_ip TEXT
            )
        """)
        conn.commit()
        _ensure_columns(conn)
    finally:
        conn.close()


class User:
    def __init__(
        self,
        id: int,
        username: str,
        password_hash: str,
        created_at: str,
        trial_ends_at: str,
        is_paid: int,
        is_activated: int,
        is_admin: int,
        activated_until: Optional[str] = None,
        is_super_admin: int = 0,
        register_ip: Optional[str] = None,
    ):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.created_at = created_at
        self.trial_ends_at = trial_ends_at
        self.is_paid = bool(is_paid)
        self.is_activated = bool(is_activated)
        self.is_admin = bool(is_admin)
        self.activated_until = (activated_until or "").strip() or None
        self.is_super_admin = bool(is_super_admin)
        self.register_ip = (register_ip or "").strip() or None

    @property
    def trial_ended(self) -> bool:
        try:
            end = datetime.strptime(self.trial_ends_at, "%Y-%m-%d %H:%M:%S")
            return datetime.now() > end
        except Exception:
            return True

    @property
    def subscription_expired(self) -> bool:
        """收费会员是否已过截止日期"""
        if not self.activated_until:
            return False
        try:
            end = datetime.strptime(self.activated_until[:10], "%Y-%m-%d").date()
            return datetime.now().date() > end
        except Exception:
            return False

    @property
    def can_use(self) -> bool:
        """试用期内或已开通且在有效期内"""
        if self.is_activated:
            if self.subscription_expired:
                return False
            return True
        return not self.trial_ended


def _row_to_user(row: Tuple) -> User:
    # 兼容 8 列（旧库）、10 列、11 列（含 register_ip）
    activated_until = row[8] if len(row) > 8 else None
    is_super_admin = int(row[9]) if len(row) > 9 else (1 if (row[1] == "superzwj") else 0)
    register_ip = row[10] if len(row) > 10 else None
    return User(
        id=row[0],
        username=row[1],
        password_hash=row[2],
        created_at=row[3],
        trial_ends_at=row[4],
        is_paid=row[5],
        is_activated=row[6],
        is_admin=row[7],
        activated_until=activated_until,
        is_super_admin=is_super_admin,
        register_ip=register_ip,
    )


def _user_columns() -> str:
    return "id, username, password_hash, created_at, trial_ends_at, is_paid, is_activated, is_admin, activated_until, is_super_admin, register_ip"


def get_user_by_id(user_id: int) -> Optional[User]:
    conn = get_connection()
    try:
        cur = conn.execute(
            f"SELECT {_user_columns()} FROM users WHERE id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        return _row_to_user(row) if row else None
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[User]:
    conn = get_connection()
    try:
        cur = conn.execute(
            f"SELECT {_user_columns()} FROM users WHERE username = ?",
            (username.strip(),),
        )
        row = cur.fetchone()
        return _row_to_user(row) if row else None
    finally:
        conn.close()


def count_registrations_by_ip(ip: Optional[str]) -> int:
    """同一 IP 已注册账号数量（用于限制重复注册）"""
    if not (ip or "").strip():
        return 0
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM users WHERE register_ip = ?",
            (ip.strip(),),
        )
        return cur.fetchone()[0]
    finally:
        conn.close()


def create_user(username: str, password: str, register_ip: Optional[str] = None) -> Optional[User]:
    """注册新用户，试用期 1 个月。register_ip 用于限制同一 IP 重复注册。"""
    username = username.strip()
    if not username or not password:
        return None
    now = datetime.now()
    trial_ends = now + timedelta(days=30)
    created = now.strftime("%Y-%m-%d %H:%M:%S")
    trial_ends_at = trial_ends.strftime("%Y-%m-%d %H:%M:%S")
    password_hash = generate_password_hash(password)
    ip_val = (register_ip or "").strip() or None
    conn = get_connection()
    try:
        is_super = 1 if username == "superzwj" else 0
        is_adm = 1 if (username == "superzwj" or is_super) else 0
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at, trial_ends_at, is_paid, is_activated, is_admin, is_super_admin, register_ip) VALUES (?, ?, ?, ?, 0, 0, ?, ?, ?)",
            (username, password_hash, created, trial_ends_at, is_adm, is_super, ip_val),
        )
        conn.commit()
        cur = conn.execute("SELECT last_insert_rowid()")
        uid = cur.fetchone()[0]
        return get_user_by_id(uid)
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def verify_password(user: User, password: str) -> bool:
    return check_password_hash(user.password_hash, password)


def set_activated(user_id: int, activated: bool, activated_until: Optional[str] = None) -> bool:
    conn = get_connection()
    try:
        if activated:
            until = (activated_until or "").strip() or None
            conn.execute(
                "UPDATE users SET is_activated = 1, is_paid = 1, activated_until = ? WHERE id = ?",
                (until, user_id),
            )
        else:
            conn.execute(
                "UPDATE users SET is_activated = 0, is_paid = 0, activated_until = NULL WHERE id = ?",
                (user_id,),
            )
        conn.commit()
        return True
    finally:
        conn.close()


def set_activated_until(user_id: int, date_str: Optional[str]) -> bool:
    """设置收费会员截止日期（仅超级管理员可调用，由路由层校验）"""
    conn = get_connection()
    try:
        until = (date_str or "").strip() or None
        conn.execute("UPDATE users SET activated_until = ? WHERE id = ?", (until, user_id))
        conn.commit()
        return True
    finally:
        conn.close()


def downgrade_expired_subscriptions() -> int:
    """将已过截止日期的收费会员自动降级，返回降级人数"""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT id FROM users WHERE is_activated = 1 AND activated_until IS NOT NULL AND activated_until <> '' AND substr(activated_until, 1, 10) < ?",
            (today,),
        )
        ids = [r[0] for r in cur.fetchall()]
        for uid in ids:
            conn.execute(
                "UPDATE users SET is_activated = 0, is_paid = 0, activated_until = NULL WHERE id = ?",
                (uid,),
            )
        conn.commit()
        return len(ids)
    finally:
        conn.close()


def list_users() -> List[User]:
    conn = get_connection()
    try:
        cur = conn.execute(f"SELECT {_user_columns()} FROM users ORDER BY id")
        return [_row_to_user(row) for row in cur.fetchall()]
    finally:
        conn.close()
