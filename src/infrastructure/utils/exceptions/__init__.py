from .cache_error import CacheError
from .data_loader_error import DataLoaderError

# 添加DataProcessingError类定义
class DataProcessingError(Exception):
    """数据处理错误"""
    def __init__(self, message: str, context: dict = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

__all__ = ['CacheError', 'DataLoaderError', 'DataProcessingError']
