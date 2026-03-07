"""
基础设施专用工具模块 - v2.0

功能分类：
1. 环境管理：
   - environment: 环境变量检查
   - logger: 日志系统配置

2. 缓存管理：
   - cache_utils: 预测结果缓存

使用规范：
1. 仅限基础设施层内部使用
2. 不直接导入其他层模块
3. 保持与通用工具模块的隔离

使用示例：
    from src.infrastructure.utils.environment import is_production
    from src.infrastructure.utils.logger import get_logger
    from src.infrastructure.utils.cache_utils import model_cache

    # 检查环境
    if is_production():
        print("Running in production")

    # 获取日志记录器
    logger = get_logger(__name__)
    logger.info("Application started")

    # 使用缓存装饰器
    @model_cache()
    def predict_model(features):
        # 模型预测逻辑
        pass

注意事项：
1. 避免与src/utils中的通用工具重复
2. 新增工具需添加集成测试
3. 保持工具函数无状态

版本历史：
- v1.0 (2024-02-10): 初始版本
- v2.0 (2024-03-20): 重构为专用工具模块
"""
# 修复导入路径 - 2025-10-31
from .components.environment import is_production, is_development, is_testing
# cache_utils已移动，暂时注释
# from .cache_utils import PredictionCache, model_cache
from .exception_utils import *
from .tools.date_utils import *
# 以下函数可能已移除或移动到其他位置，暂时注释
# from .tools import validate_dates, fill_missing_values, convert_to_ordered_dict
from .datetime_parser import DateTimeParser
# audit模块已移除或移动到其他位置，暂时注释
# from .audit import AuditLogger, audit_log
# 移除循环导入，直接使用基础设施层的logger
from .logger import (
    get_logger,
    setup_logging,
    get_unified_logger,
)
import importlib

try:
    _utils_logging = importlib.import_module("src.infrastructure.utils.logging")
except ModuleNotFoundError:  # pragma: no cover - 回退到标准库
    import logging as _utils_logging

# 确保自定义日志工具可用
try:
    from .logging.logger import UnifiedLogger as _UnifiedLogger, get_unified_logger as _get_unified_logger
except Exception:  # pragma: no cover - 自定义日志模块不可用时跳过
    _UnifiedLogger = None
    _get_unified_logger = None

if _UnifiedLogger is not None:
    setattr(_utils_logging, "UnifiedLogger", _UnifiedLogger)
    setattr(_utils_logging, "get_unified_logger", _get_unified_logger)
    if hasattr(_utils_logging, "__all__"):
        all_list = list(_utils_logging.__all__)
        if "UnifiedLogger" not in all_list:
            all_list.append("UnifiedLogger")
        if "get_unified_logger" not in all_list:
            all_list.append("get_unified_logger")
        _utils_logging.__all__ = all_list

logging = _utils_logging

__all__ = [
    'is_production',
    'is_development',
    'is_testing',
    # cache_utils已移动，暂时注释
    # 'PredictionCache',
    # 'model_cache',
    'DataLoaderError',
    # 'validate_dates',  # 已移动到DateTimeParser类
    # 'fill_missing_values',  # 可能已移除
    'DateTimeParser',
    # 'convert_to_ordered_dict',  # 可能已移除
    'date_utils',
    'exception_utils',
    # 日志相关
    'get_logger',
    'setup_logging',
    'get_unified_logger',
]
if 'logging' not in __all__:
    __all__.append('logging')
