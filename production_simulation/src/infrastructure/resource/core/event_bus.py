"""
event_bus 模块

提供 event_bus 相关功能和接口。
"""

import logging

import uuid
import asyncio
import concurrent.futures
import threading
import time

from .event_handler import EventHandler
from .event_storage import EventStorage
from .shared_interfaces import ILogger, StandardLogger
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty
from typing import Any, Dict, List, Callable, Optional, Set, TypeVar
"""
事件总线

Phase 2: 结构优化 - 事件驱动架构

实现事件总线，支持发布-订阅模式，实现组件间的松耦合通信。
"""

logger = logging.getLogger(__name__)

T = TypeVar('T')
SyncEventHandler = Callable[[Any], None]
AsyncEventHandler = Callable[[Any], asyncio.Future[Any]]


@dataclass
class Event:
    """事件基类"""
    event_type: str
    source: str
    data: Dict[str, Any]
    timestamp: Optional[datetime] = None
    event_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())


class ResourceEvent(Event):
    """资源相关事件"""

    def __init__(self, event_type: str, source: str, data: Dict[str, Any],
                 resource_type: str, resource_id: Optional[str] = None,
                 action: str = "", timestamp: Optional[datetime] = None,
                 event_id: Optional[str] = None, correlation_id: Optional[str] = None):
        super().__init__(event_type, source, data, timestamp, event_id, correlation_id)
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.action = action


class SystemEvent(Event):
    """系统相关事件"""

    def __init__(self, event_type: str, source: str, data: Dict[str, Any],
                 severity: str = "info", component: str = "",
                 timestamp: Optional[datetime] = None,
                 event_id: Optional[str] = None, correlation_id: Optional[str] = None):
        super().__init__(event_type, source, data, timestamp, event_id, correlation_id)
        self.severity = severity
        self.component = component


class PerformanceEvent(Event):
    """性能相关事件"""

    def __init__(self, event_type: str, source: str, data: Dict[str, Any],
                 metric_name: str, metric_value: Any = None, threshold: Optional[Any] = None,
                 breached: bool = False, timestamp: Optional[datetime] = None,
                 event_id: Optional[str] = None, correlation_id: Optional[str] = None):
        super().__init__(event_type, source, data, timestamp, event_id, correlation_id)
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.threshold = threshold
        self.breached = breached


class EventFilter:
    """事件过滤器"""

    def __init__(self, event_types: Optional[Set[str]] = None,
                 sources: Optional[Set[str]] = None,
                 custom_filter: Optional[Callable[[Event], bool]] = None):
        self.event_types = event_types or set()
        self.sources = sources or set()
        self.custom_filter = custom_filter

    def matches(self, event: Event) -> bool:
        """检查事件是否匹配过滤器"""
        if self.event_types and event.event_type not in self.event_types:
            return False

        if self.sources and event.source not in self.sources:
            return False

        if self.custom_filter and not self.custom_filter(event):
            return False

        return True


class EventSubscription:
    """事件订阅"""

    def __init__(self, handler: EventHandler, filter: Optional[EventFilter] = None,
                 priority: int = 0, async_handler: bool = False):
        self.handler = handler
        self.filter = filter
        self.priority = priority
        self.async_handler = async_handler
        self.active = True
        self.subscription_id: Optional[str] = None

    def matches(self, event: Event) -> bool:
        """检查事件是否匹配订阅"""
        return self.active and (self.filter is None or self.filter.matches(event))


class EventBus:
    """事件总线

    支持同步和异步事件处理，提供发布-订阅模式。
    """

    def __init__(self, logger: Optional[ILogger] = None, max_workers: int = 4):
        self.logger = logger or StandardLogger(self.__class__.__name__)

        # 使用专用组件
        self._event_handler = EventHandler(logger)
        self._event_storage = EventStorage(logger=logger)

        # 订阅管理
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._subscription_lock = threading.RLock()
        self._next_subscription_id = 1

        # 事件队列和处理
        self._event_queue: Queue = Queue()
        self._processing_thread: Optional[threading.Thread] = None
        self._running = False

        # 异步处理
        self._async_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # 事件历史
        self._event_history: List[Event] = []
        self._max_history_size = 1000
        self._history_lock = threading.RLock()

        # 性能统计
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "handlers_executed": 0,
            "avg_processing_time": 0.0
        }
        self._stats_lock = threading.RLock()

    def start(self):
        """启动事件总线"""
        if self._running:
            return

        self._running = True
        self._processing_thread = threading.Thread(target=self._process_events, daemon=True)
        self._processing_thread.start()
        self.logger.log_info("事件总线已启动")

    def stop(self):
        """停止事件总线"""
        if not self._running:
            return

        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)

        self._async_executor.shutdown(wait=True)
        self.logger.log_info("事件总线已停止")

    def subscribe(self, handler: EventHandler, filter: Optional[EventFilter] = None,
                  priority: int = 0, async_handler: bool = False) -> str:
        """订阅事件"""
        with self._subscription_lock:
            subscription_id = f"sub_{self._next_subscription_id}"
            self._next_subscription_id += 1

            subscription = EventSubscription(
                handler=handler,
                filter=filter,
                priority=priority,
                async_handler=async_handler
            )
            subscription.subscription_id = subscription_id

            self._subscriptions[subscription_id] = subscription
            self.logger.log_info(f"事件订阅已创建: {subscription_id}")

            return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅"""
        with self._subscription_lock:
            if subscription_id in self._subscriptions:
                del self._subscriptions[subscription_id]
                self.logger.log_info(f"事件订阅已取消: {subscription_id}")
                return True
        return False

    def publish(self, event: Event, async_publish: bool = False):
        """发布事件"""
        if async_publish:
            self._async_executor.submit(self._publish_sync, event)
        else:
            self._publish_sync(event)

    def _publish_sync(self, event: Event):
        """同步发布事件"""
        try:
            # 添加到历史
            self._add_to_history(event)

            # 更新统计
            with self._stats_lock:
                self._stats["events_published"] += 1

            # 放入处理队列
            if self._running:
                self._event_queue.put(event)
            else:
                self.logger.log_warning("事件总线未运行，事件被丢弃")

        except Exception as e:
            self.logger.log_error("发布事件失败", error=e)
            with self._stats_lock:
                self._stats["events_failed"] += 1

    async def publish_async(self, event: Event):
        """异步发布事件"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._async_executor, self._publish_sync, event)

    def _process_events(self):
        """事件处理循环"""
        while self._running:
            try:
                # 获取事件（带超时）
                event = self._event_queue.get(timeout=1.0)

                # 处理事件
                start_time = time.time()
                self._handle_event(event)
                processing_time = time.time() - start_time

                # 更新统计
                with self._stats_lock:
                    self._stats["events_processed"] += 1
                    # 更新平均处理时间
                    total_processed = self._stats["events_processed"]
                    self._stats["avg_processing_time"] = (
                        (self._stats["avg_processing_time"] *
                         (total_processed - 1)) + processing_time
                    ) / total_processed

                self._event_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                self.logger.log_error("处理事件失败", error=e)
                with self._stats_lock:
                    self._stats["events_failed"] += 1

    def _handle_event(self, event: Event):
        """处理单个事件"""
        try:
            # 获取匹配的订阅
            matching_subscriptions = self._get_matching_subscriptions(event)

            if not matching_subscriptions:
                self.logger.log_info(f"没有匹配的事件处理器: {event.event_type}")
                return

            # 按优先级排序（高优先级先执行）
            matching_subscriptions.sort(key=lambda s: s.priority, reverse=True)

            # 执行处理器
            for subscription in matching_subscriptions:
                try:
                    if subscription.async_handler:
                        # 异步处理
                        self._async_executor.submit(subscription.handler, event)
                    else:
                        # 同步处理
                        subscription.handler(event)

                    with self._stats_lock:
                        self._stats["handlers_executed"] += 1

                except Exception as e:
                    self.logger.log_error(f"事件处理器执行失败: {subscription.subscription_id}", error=e)

        except Exception as e:
            self.logger.log_error("事件处理失败", error=e)

    def _get_matching_subscriptions(self, event: Event) -> List[EventSubscription]:
        """获取匹配的订阅"""
        matching = []
        with self._subscription_lock:
            for subscription in self._subscriptions.values():
                if subscription.matches(event):
                    matching.append(subscription)
        return matching

    def _add_to_history(self, event: Event):
        """添加到事件历史"""
        with self._history_lock:
            self._event_history.append(event)

            # 限制历史大小
            if len(self._event_history) > self._max_history_size:
                self._event_history.pop(0)

    def get_event_history(self, event_type: Optional[str] = None,
                          source: Optional[str] = None, limit: int = 100) -> List[Event]:
        """获取事件历史"""
        with self._history_lock:
            history = self._event_history

            # 过滤
            if event_type:
                history = [e for e in history if e.event_type == event_type]
            if source:
                history = [e for e in history if e.source == source]

            # 返回最新的limit个
            return history[-limit:] if limit > 0 else history

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._stats_lock:
            stats = self._stats.copy()

        with self._subscription_lock:
            stats["active_subscriptions"] = len(self._subscriptions)

        with self._history_lock:
            stats["events_in_history"] = len(self._event_history)

        return stats

    def clear_history(self):
        """清空事件历史"""
        with self._history_lock:
            self._event_history.clear()
            self.logger.log_info("事件历史已清空")

    def clear_event_history(self):
        """清空事件历史 - 别名方法"""
        self.clear_history()

    def get_subscriptions(self) -> List[Dict[str, Any]]:
        """获取所有订阅信息"""
        with self._subscription_lock:
            return [
                {
                    "subscription_id": sub.subscription_id,
                    "priority": sub.priority,
                    "async_handler": sub.async_handler,
                    "active": sub.active,
                    "filter": {
                        "event_types": list(sub.filter.event_types) if sub.filter else [],
                        "sources": list(sub.filter.sources) if sub.filter else []
                    } if sub.filter else None
                }
                for sub in self._subscriptions.values()
            ]

    def reset_stats(self):
        """重置统计信息"""
        with self._stats_lock:
            self._stats = {
                "events_published": 0,
                "events_processed": 0,
                "events_failed": 0,
                "handlers_executed": 0,
                "avg_processing_time": 0.0
            }
        self.logger.log_info("统计信息已重置")

# =============================================================================
# 预定义事件类型
# =============================================================================


class EventTypes:
    """预定义事件类型"""

    # 资源事件
    RESOURCE_ALLOCATED = "resource.allocated"
    RESOURCE_RELEASED = "resource.released"
    RESOURCE_FAILED = "resource.failed"
    RESOURCE_OPTIMIZED = "resource.optimized"

    # 系统事件
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    SYSTEM_ERROR = "system.error"
    SYSTEM_HEALTH_CHANGED = "system.health_changed"

    # 性能事件
    PERFORMANCE_METRIC = "performance.metric"
    PERFORMANCE_ANOMALY = "performance.anomaly"
    PERFORMANCE_THRESHOLD_BREACHED = "performance.threshold_breached"

    # 配置事件
    CONFIG_UPDATED = "config.updated"
    CONFIG_VALIDATION_FAILED = "config.validation_failed"

    # 监控事件
    MONITOR_STARTED = "monitor.started"
    MONITOR_STOPPED = "monitor.stopped"
    ALERT_TRIGGERED = "alert.triggered"
    ALERT_RESOLVED = "alert.resolved"

# =============================================================================
# 便捷函数
# =============================================================================


def create_resource_event(resource_type: str, resource_id: str, action: str,
                          source: str, **data) -> ResourceEvent:
    """创建资源事件"""
    return ResourceEvent(
        event_type=f"resource.{action}",
        source=source,
        timestamp=datetime.now(),
        data=data,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action
    )


def create_system_event(severity: str, component: str, message: str,
                        source: str, **data) -> SystemEvent:
    """创建系统事件"""
    return SystemEvent(
        event_type=f"system.{severity}",
        source=source,
        timestamp=datetime.now(),
        data={"message": message, **data},
        severity=severity,
        component=component
    )


def create_performance_event(metric_name: str, metric_value: Any,
                             threshold: Optional[Any], breached: bool,
                             source: str, **data) -> PerformanceEvent:
    """创建性能事件"""
    return PerformanceEvent(
        event_type=EventTypes.PERFORMANCE_METRIC,
        source=source,
        timestamp=datetime.now(),
        data=data,
        metric_name=metric_name,
        metric_value=metric_value,
        threshold=threshold,
        breached=breached
    )
