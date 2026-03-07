"""
统一异常处理模块别名

提供向后兼容的导入路径
"""

from .foundation.exceptions.unified_exceptions import *

__all__ = [
    'handle_infrastructure_exceptions',
    'handle_core_exceptions'
]

