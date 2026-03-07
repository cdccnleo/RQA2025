"""
错误处理模块
"""

from .error_handler import ErrorHandler

# 定义常用异常类
class DataLoaderError(Exception):
    """数据加载错误"""
    pass

class DataValidationError(Exception):
    """数据验证错误"""
    pass

class DataProcessingError(Exception):
    """数据处理错误"""
    pass

__all__ = ['ErrorHandler', 'DataLoaderError', 'DataValidationError', 'DataProcessingError']
