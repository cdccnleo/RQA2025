"""
事件总线模块
提供完整的模块化事件驱动架构
"""

import threading
from typing import Optional

# 类型定义
from .types import EventType, EventPriority

# 数据模型
from .models import Event, EventHandler

# 从core导入EventBus
from .core import EventBus
# 从context导入HandlerExecutionContext（避免循环导入）
from .context import HandlerExecutionContext

# 全局事件总线实例管理
_global_event_bus: Optional[EventBus] = None
_global_event_bus_lock = threading.Lock()

def get_event_bus() -> EventBus:
    """
    获取全局事件总线实例（线程安全单例模式）

    Returns:
        EventBus: 全局事件总线实例

    Thread Safety:
        此函数是线程安全的，使用双重检查锁定模式确保只有一个实例被创建。
    """
    global _global_event_bus

    # 第一次检查（无锁，性能优化）
    if _global_event_bus is not None:
        return _global_event_bus

    # 第二次检查（加锁，确保线程安全）
    with _global_event_bus_lock:
        if _global_event_bus is None:
            # 创建实例
            event_bus = EventBus()

            # 初始化（BaseComponent.initialize 会检查 _initialized 标志）
            try:
                success = event_bus.initialize()
                if not success:
                    raise RuntimeError("EventBus 初始化返回失败")
            except Exception as e:
                # 初始化失败，不保存实例
                raise RuntimeError(f"事件总线初始化失败: {e}") from e

            # 保存实例（只有在初始化成功后）
            _global_event_bus = event_bus

    return _global_event_bus


def reset_event_bus() -> None:
    """
    重置全局事件总线实例（主要用于测试）

    Warning:
        此方法会关闭现有的事件总线并创建新实例。
        在生产环境中谨慎使用。
    """
    global _global_event_bus

    with _global_event_bus_lock:
        if _global_event_bus is not None:
            try:
                _global_event_bus.shutdown()
            except Exception:
                pass  # 忽略关闭错误
            _global_event_bus = None

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
    'reset_event_bus',

    # 持久化
    'EventPersistence',
    'PersistenceMode',

    # 异常和健康状态
    'EventBusException',
    'ComponentHealth'
]
