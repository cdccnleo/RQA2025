"""
Base Security模块别名

提供安全相关的基础枚举和类
"""

from enum import Enum


class SecurityLevel(Enum):
    """安全级别枚举"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


__all__ = ['SecurityLevel']

