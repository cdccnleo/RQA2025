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

# 全局事件总线实例管理
_global_event_bus: EventBus = None
_global_event_bus_lock = False

def get_event_bus() -> EventBus:
    """
    获取全局事件总线实例（单例模式，线程安全）

    Returns:
        EventBus: 全局事件总线实例
    """
    global _global_event_bus, _global_event_bus_lock

    # 双重检查锁定模式
    if _global_event_bus is None:
        if not _global_event_bus_lock:
            _global_event_bus_lock = True
            try:
                if _global_event_bus is None:
                    _global_event_bus = EventBus()
                    # 初始化（BaseComponent.initialize 会检查 _initialized 标志）
                    _global_event_bus.initialize()
            finally:
                _global_event_bus_lock = False

    return _global_event_bus

# 工具类
from .utils import EventRetryManager, EventPerformanceMonitor

# 持久化
from .persistence.event_persistence import EventPersistence, PersistenceMode

# 异常类
try:
    from ..foundation.exceptions.core_exceptions import EventBusException
except ImportError:
    # 如果foundation不可用，提供本地定义
    class EventBusException(Exception):
        pass

# 组件健康状态
try:
    from ..foundation.health.health_status import ComponentHealth
except ImportError:
    # 如果foundation不可用，提供枚举定义
    from enum import Enum
    class ComponentHealth(Enum):
        HEALTHY = "healthy"
        DEGRADED = "degraded"
        UNHEALTHY = "unhealthy"

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

    # 全局实例管理
    'get_event_bus',

    # 持久化
    'EventPersistence',
    'PersistenceMode',

    # 异常和健康状态
    'EventBusException',
    'ComponentHealth'
]
