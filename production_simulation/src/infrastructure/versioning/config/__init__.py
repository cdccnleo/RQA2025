"""
配置版本管理模块

提供配置文件版本控制和历史管理功能。
"""

from .config_version_manager import ConfigVersionManager, ConfigVersionInfo

__all__ = [
    "ConfigVersionManager",
    "ConfigVersionInfo",
]
