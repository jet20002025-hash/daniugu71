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


def init_db() -> None:
    """创建用户表"""
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
                is_admin INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()
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
    ):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.created_at = created_at
        self.trial_ends_at = trial_ends_at
        self.is_paid = bool(is_paid)
        self.is_activated = bool(is_activated)
        self.is_admin = bool(is_admin)

    @property
    def trial_ended(self) -> bool:
        try:
            end = datetime.strptime(self.trial_ends_at, "%Y-%m-%d %H:%M:%S")
            return datetime.now() > end
        except Exception:
            return True

    @property
    def can_use(self) -> bool:
        """试用期内或已开通（充值并激活）"""
        if self.is_activated:
            return True
        return not self.trial_ended


def _row_to_user(row: Tuple) -> User:
    return User(
        id=row[0],
        username=row[1],
        password_hash=row[2],
        created_at=row[3],
        trial_ends_at=row[4],
        is_paid=row[5],
        is_activated=row[6],
        is_admin=row[7],
    )


def get_user_by_id(user_id: int) -> Optional[User]:
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT id, username, password_hash, created_at, trial_ends_at, is_paid, is_activated, is_admin FROM users WHERE id = ?",
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
            "SELECT id, username, password_hash, created_at, trial_ends_at, is_paid, is_activated, is_admin FROM users WHERE username = ?",
            (username.strip(),),
        )
        row = cur.fetchone()
        return _row_to_user(row) if row else None
    finally:
        conn.close()


def create_user(username: str, password: str) -> Optional[User]:
    """注册新用户，试用期 1 个月"""
    username = username.strip()
    if not username or not password:
        return None
    now = datetime.now()
    trial_ends = now + timedelta(days=30)
    created = now.strftime("%Y-%m-%d %H:%M:%S")
    trial_ends_at = trial_ends.strftime("%Y-%m-%d %H:%M:%S")
    password_hash = generate_password_hash(password)
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at, trial_ends_at, is_paid, is_activated, is_admin) VALUES (?, ?, ?, ?, 0, 0, 0)",
            (username, password_hash, created, trial_ends_at),
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


def set_activated(user_id: int, activated: bool) -> bool:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE users SET is_activated = ?, is_paid = ? WHERE id = ?",
            (1 if activated else 0, 1 if activated else 0, user_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def list_users() -> List[User]:
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT id, username, password_hash, created_at, trial_ends_at, is_paid, is_activated, is_admin FROM users ORDER BY id"
        )
        return [_row_to_user(row) for row in cur.fetchall()]
    finally:
        conn.close()
