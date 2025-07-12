"""数据层主模块

提供统一的数据访问接口，包含：
- 数据加载与管理
- 缓存策略
- 数据质量监控
- 数据适配器基类
"""

from .data_manager import DataManager, DataModel
from .base import BaseDataAdapter
from .validator import DataValidator
from .registry import DataRegistry
from .quality import DataQualityMonitor, QualityMetric

__all__ = [
    'DataManager',
    'DataModel',
    'BaseDataAdapter',
    'DataValidator',
    'DataRegistry',
    'DataQualityMonitor',
    'QualityMetric'
]
