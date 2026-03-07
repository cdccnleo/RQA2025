"""
事件总线模块
提供完整的模块化事件驱动架构
"""

# 类型定义
from .types import EventType, EventPriority

# 数据模型
from .models import Event, EventHandler

# 从core导入EventBus
from .core import EventBus
# 从context导入HandlerExecutionContext（避免循环导入）
from .context import HandlerExecutionContext

# 工具类
from .utils import EventRetryManager, EventPerformanceMonitor

# 持久化
from .persistence.event_persistence import EventPersistence, PersistenceMode

__all__ = [
    # 类型和枚举
    'EventType',
    'EventPriority',

    # 数据模型
    'Event',
    'EventHandler',
    'HandlerExecutionContext',

    # 工具类
    'EventRetryManager',
    'EventPerformanceMonitor',

    # 核心组件
    'EventBus',

    # 持久化
    'EventPersistence',
    'PersistenceMode'
]
