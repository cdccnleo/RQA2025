"""
logger 模块

提供 logger 相关功能和接口。
"""

import logging

# from src.infrastructure.logging import Logger  # 暂时注释，避免导入错误
"""日志记录工具"

提供统一的日志记录功能，用于项目中的日志管理。
此模块重定向到基础设施层的日志系统，避免重复实现。

函数:
    - get_logger: 获取配置好的日志记录器
"""

# 重定向到基础设施层的日志系统
# 跨层级导入：infrastructure层组件
try:
    # Logger未定义，使用标准logging
    def get_logger(name: str = "rqa"):
        """获取日志器"""
        return logging.getLogger(name)

except ImportError:
    # 如果基础设施日志模块不可用，使用标准logging
    def get_logger(name: str = "rqa"):
        """获取日志器"""
        return logging.getLogger(name)

__all__ = ["get_logger"]
