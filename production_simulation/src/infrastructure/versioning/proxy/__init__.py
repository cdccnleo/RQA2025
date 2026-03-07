"""
版本代理模块

提供版本控制代理、缓存层和历史管理功能。
"""

from .proxy import VersionProxy, get_default_version_proxy

__all__ = [
    "VersionProxy",
    "get_default_version_proxy",
]
