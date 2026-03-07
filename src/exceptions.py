"""
顶层异常模块（别名）
"""

# 从core.exceptions导入
try:
    from src.core.exceptions import *
except ImportError:
    # 提供基础异常
    class ValidationError(Exception):
        pass
    
    class ConfigurationError(Exception):
        pass
    
    class DataError(Exception):
        pass
