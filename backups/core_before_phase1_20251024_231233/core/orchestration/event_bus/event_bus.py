"""
事件总线核心实现

提供完整的异步事件驱动架构，支持：
- 事件发布 / 订阅
- 异步事件处理
- 事件优先级管理
- 事件持久化
- 性能监控
"""

from .publisher_components import EventPublisher
from .dispatcher_components import EventDispatcher
from .subscriber_components import EventSubscriber
import asyncio
import time
from typing import Any, Dict, List, Callable, Optional, Union
from enum import Enum
from dataclasses import dataclass


class EventType(Enum):

    """事件类型枚举"""
    # 核心服务层事件
    EVENT_BUS_INITIALIZED = "event_bus_initialized"
    DEPENDENCY_INJECTED = "dependency_injected"
    ORCHESTRATOR_STARTED = "orchestrator_started"
    BUSINESS_PROCESS_INITIATED = "business_process_initiated"

    # 基础设施层事件
    CONFIG_LOADED = "config_loaded"
    CACHE_INITIALIZED = "cache_initialized"
    LOGGING_SYSTEM_READY = "logging_system_ready"

    # 数据层事件
    DATA_COLLECTION_STARTED = "data_collection_started"
    DATA_RECEIVED = "data_received"
    DATA_VALIDATED = "data_validated"
    DATA_STORED = "data_stored"

    # 其他通用事件
    PROCESS_STARTED = "process_started"
    PROCESS_COMPLETED = "process_completed"
    ERROR_OCCURRED = "error_occurred"


class EventPriority(Enum):

    """事件优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:

    """事件数据类"""
    event_type: Union[EventType, str]
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    source: str = "system"
    priority: EventPriority = EventPriority.NORMAL
    event_id: Optional[str] = None

    def __post_init__(self):

        if self.timestamp is None:
            import time
            self.timestamp = time.time()
        if self.event_id is None:
            import uuid
            self.event_id = str(uuid.uuid4())


# 重新导出核心组件
__all__ = [
    "Event",
    "EventType",
    "EventPriority",
    "EventPublisher",
    "EventSubscriber",
    "EventDispatcher",
    "EventPersistence",
    "EventRetryManager",
    "EventPerformanceMonitor"
]


class EventPersistence:

    """事件持久化管理"""

    def __init__(self, storage_path: str = "events.db"):

        self.storage_path = storage_path
        self._events = []
        self._max_events = 10000  # 限制最大事件数量，防止内存泄漏

    def save_event(self, event: Event) -> bool:
        """保存事件"""
        try:
            self._events.append(event)

            # 检查是否需要清理旧事件（防止内存泄漏）
            if len(self._events) > self._max_events:
                # 清理最旧的20%事件
                remove_count = int(self._max_events * 0.2)
                self._events = self._events[remove_count:]
                print(f"🧹 Cleaned up {remove_count} old events to prevent memory leak")

            return True
        except Exception:
            return False

    def load_events(self, event_type: Optional[EventType] = None) -> List[Event]:
        """加载事件"""
        if event_type:
            return [e for e in self._events if e.event_type == event_type]
        return self._events.copy()

    def clear_events(self, days: int = 7) -> int:
        """清理过期事件"""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        old_count = len(self._events)
        self._events = [e for e in self._events if e.timestamp > cutoff_time]
        return old_count - len(self._events)


class EventRetryManager:

    """事件重试管理"""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):

        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def retry_event(self, event: Event, handler: Callable) -> bool:
        """重试事件处理"""
        for attempt in range(self.max_retries):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
                return True
            except Exception:
                if attempt == self.max_retries - 1:
                    return False
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
        return False


class EventPerformanceMonitor:

    """事件性能监控"""

    def __init__(self):

        self.metrics = {
            'published_events': 0,
            'delivered_events': 0,
            'failed_events': 0,
            'processing_times': [],
            'queue_sizes': []
        }

    def record_event_published(self):
        """记录事件发布"""
        self.metrics['published_events'] += 1

    def record_event_delivered(self, processing_time: float):
        """记录事件交付"""
        self.metrics['delivered_events'] += 1
        self.metrics['processing_times'].append(processing_time)

    def record_event_failed(self):
        """记录事件失败"""
        self.metrics['failed_events'] += 1

    def record_queue_size(self, size: int):
        """记录队列大小"""
        self.metrics['queue_sizes'].append(size)

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            'published_events': self.metrics['published_events'],
            'delivered_events': self.metrics['delivered_events'],
            'failed_events': self.metrics['failed_events'],
            'success_rate': self.metrics['delivered_events'] / max(self.metrics['published_events'], 1),
            'avg_processing_time': sum(self.metrics['processing_times']) / max(len(self.metrics['processing_times']), 1),
            'max_queue_size': max(self.metrics['queue_sizes']) if self.metrics['queue_sizes'] else 0
        }
