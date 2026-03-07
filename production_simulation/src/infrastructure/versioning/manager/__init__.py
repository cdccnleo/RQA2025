"""
版本管理器模块

提供版本管理器和策略管理功能。
"""

from .manager import VersionManager
from .policy import VersionPolicy

__all__ = [
    "VersionManager",
    "VersionPolicy",
]
