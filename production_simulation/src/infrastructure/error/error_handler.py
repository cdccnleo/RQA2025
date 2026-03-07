"""
错误处理器模块
"""

try:
    from ..utils.exception_utils import *
except ImportError:
    pass

# 基础错误处理器
class ErrorHandler:
    """基础错误处理器"""
    
    def __init__(self):
        self.errors = []
    
    def handle_error(self, error: Exception) -> bool:
        """处理错误"""
        self.errors.append(error)
        return True
    
    def get_errors(self):
        """获取错误列表"""
        return self.errors

__all__ = ['ErrorHandler']

