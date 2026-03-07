"""
工具模块（顶层别名）
提供向后兼容的导入路径
"""

# 从infrastructure.utils导入工具
try:
    from src.infrastructure.utils.tools import *
except ImportError:
    # 从infrastructure.utils.date_utils导入
    try:
        from src.infrastructure.utils.tools.date_utils import *
    except ImportError:
        pass

__all__ = []

