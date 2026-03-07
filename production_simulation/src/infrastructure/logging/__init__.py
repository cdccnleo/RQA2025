"""
__init__ 模块

提供 __init__ 相关功能和接口。
"""

# from .core import Logger  # 暂时注释以避免语法错误
"""
基础设施层 - 日志管理系统

提供企业级的结构化日志记录、处理和监控功能。
采用分层架构设计：core（核心）/ handlers（处理器）/ utils（工具）

主要特性：
- 统一日志接口和实现
- 多格式日志输出支持
- 结构化日志记录
- 性能监控和统计
- 企业级异常处理
"""

# 导入核心组件

# 基础设施层专用日志器获取函数


def get_infrastructure_logger(name: str = None, **kwargs):
    """
    获取基础设施层日志器

    Args:
        name: 日志器名称
        **kwargs: 其他参数

    Returns:
        UnifiedLogger: 统一日志器实例
    """
    # 暂时返回标准logging logger，避免循环依赖
    import logging
    return logging.getLogger(name or "infrastructure")


__version__ = "2.0.0"
__all__ = [
    # 核心接口和类
    'ILogger', 'BaseLogger', 'UnifiedLogger',
    # 枚举类型
    'LogLevel', 'LogFormat', 'LogCategory',
    # 专用日志器
    'BusinessLogger', 'AuditLogger', 'PerformanceLogger',
    # 工厂函数
    'get_unified_logger', 'get_infrastructure_logger',
]
