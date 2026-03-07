"""
Utils Helpers Module
工具辅助模块

This module contains helper utility functions
"""

class UtilsHelpers:
    """工具辅助类"""

    def __init__(self):
        self.initialized = True

    def format_datetime(self, dt):
        """格式化日期时间"""
        return str(dt) if dt else None

    def safe_divide(self, numerator, denominator, default=0):
        """安全除法"""
        try:
            return numerator / denominator if denominator != 0 else default
        except (ZeroDivisionError, TypeError):
            return default

__all__ = ['UtilsHelpers']
