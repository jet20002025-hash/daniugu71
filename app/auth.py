# 登录、注册、订阅校验
from functools import wraps
from flask import redirect, request, session, url_for

from app.database import get_user_by_id, get_user_by_username, create_user, verify_password


def get_current_user():
    """从 session 取出当前用户"""
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(int(user_id))


def login_required(f):
    """未登录则跳转登录页"""
    @wraps(f)
    def wrapped(*args, **kwargs):
        if get_current_user() is None:
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)
    return wrapped


def subscription_required(f):
    """已登录且（试用期内或已开通）才能访问"""
    @wraps(f)
    def wrapped(*args, **kwargs):
        user = get_current_user()
        if user is None:
            return redirect(url_for("login", next=request.url))
        if not user.can_use:
            return redirect(url_for("subscription_blocked"))
        return f(*args, **kwargs)
    return wrapped


def admin_required(f):
    """仅管理员"""
    @wraps(f)
    def wrapped(*args, **kwargs):
        user = get_current_user()
        if user is None:
            return redirect(url_for("login", next=request.url))
        if not user.is_admin:
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return wrapped
