"""
Logger工具模块
提供基础设施层工具使用的日志功能

本模块从components.logger重新导出日志函数，以保持向后兼容
"""

from .components.logger import (
    get_logger,
    setup_logging,
    get_unified_logger,
)

__all__ = [
    'get_logger',
    'setup_logging',
    'get_unified_logger',
]

