"""
配置事件处理模块
版本更新记录：
2024-04-02 v3.6.0 - 事件模块更新
    - 重构事件总线
    - 新增死信队列
    - 增强事件过滤器
2024-04-03 v3.6.1 - 过滤器补充
    - 添加EnvironmentFilter导出
    - 完善过滤器文档
"""

from .config_event import ConfigEvent, ConfigEventBus
from .filters import (
    EventFilter,
    EnvironmentFilter,
    SensitiveDataFilter,
    EventTypeFilter,
    CompositeFilter
)

__all__ = [
    'ConfigEvent',
    'EventFilter',
    'EnvironmentFilter',
    'SensitiveDataFilter',
    'EventTypeFilter',
    'CompositeFilter',
    'ConfigEventBus'
]
