"""
事件总线核心实现 - 重构版 5.1.0
EventBus主类和核心功能 - 进一步优化类型安全和错误处理

作者: 系统架构师
创建时间: 2025-01-28
版本: 5.1.0
"""

from typing import Dict, List, Callable, Any, Union, Optional, Tuple
import time
import logging
import asyncio
import threading
from queue import PriorityQueue, Queue, Empty
from collections import defaultdict, deque
from datetime import datetime
from dataclasses import dataclass

from src.core.constants import (
    DEFAULT_BATCH_SIZE, MAX_QUEUE_SIZE, DEFAULT_TIMEOUT
)

from ..foundation.base import BaseComponent, ComponentStatus, ComponentHealth
from ..foundation.exceptions.core_exceptions import EventBusException
from .types import EventType, EventPriority
from .models import Event, EventHandler
from .context import HandlerExecutionContext  # 从独立模块导入，避免循环导入
from .utils import EventRetryManager, EventPerformanceMonitor
from .persistence.event_persistence import EventPersistence, EventStatus
from .components import EventPublisher, EventSubscriber, EventProcessor, EventMonitor

# 定义事件总线特定常量
MAX_RECORDS = 1000  # 最大记录数

logger = logging.getLogger(__name__)


# 参数封装数据类 - 解决长参数列表问题
@dataclass
class EventBusConfig:
    """事件总线配置参数 - 优化默认值以减少CPU占用"""
    # 使用合理的默认值，避免过高的CPU占用
    max_workers: int = 4  # 默认4个工作线程，避免创建过多线程
    enable_async: bool = True
    enable_persistence: bool = True
    enable_retry: bool = True
    enable_monitoring: bool = True
    batch_size: int = 100  # 默认批处理大小100
    max_queue_size: int = 1000  # 默认队列大小1000


@dataclass
class EventProcessingResult:
    """事件处理结果"""
    event_id: Optional[str]
    event_type: str
    success: bool
    processing_time: float
    sync_handlers_executed: int = 0
    async_handlers_executed: int = 0
    errors: Optional[List[str]] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


# HandlerExecutionContext已移至context.py，避免循环导入
# 已在第28行导入


class EventFilterManager:
    """事件过滤器管理器"""

    def __init__(self):
        self._filters: List[Callable] = []
        self._transformers: List[Callable] = []

    def add_event_filter(self, filter_func: Callable) -> None:
        """添加事件过滤器"""
        if filter_func not in self._filters:
            self._filters.append(filter_func)

    def remove_event_filter(self, filter_func: Callable) -> None:
        """移除事件过滤器"""
        if filter_func in self._filters:
            self._filters.remove(filter_func)

    def add_event_transformer(self, transformer_func: Callable) -> None:
        """添加事件转换器"""
        if transformer_func not in self._transformers:
            self._transformers.append(transformer_func)

    def remove_event_transformer(self, transformer_func: Callable) -> None:
        """移除事件转换器"""
        if transformer_func in self._transformers:
            self._transformers.remove(transformer_func)

    def apply_filters(self, event: Event) -> bool:
        """应用过滤器"""
        for filter_func in self._filters:
            if not filter_func(event):
                return False
        return True

    def apply_transformers(self, event: Event) -> Event:
        """应用转换器"""
        for transformer in self._transformers:
            event = transformer(event)
        return event


class EventRoutingManager:
    """事件路由管理器"""

    def __init__(self):
        self._routes: Dict[str, List[str]] = defaultdict(list)
        self._dead_letter_queue: List[Dict[str, Any]] = []

    def add_event_route(self, from_event: str, to_handlers: List[str]) -> None:
        """添加事件路由"""
        self._routes[from_event].extend(to_handlers)

    def remove_event_route(self, from_event: str) -> None:
        """移除事件路由"""
        if from_event in self._routes:
            del self._routes[from_event]

    def route_event(self, event: Event) -> List[str]:
        """路由事件到处理器"""
        event_type = event.event_type if hasattr(event, 'event_type') else str(event.event_type)
        return self._routes.get(event_type, [])

    def get_dead_letter_events(self) -> List[Dict]:
        """获取死信队列事件"""
        return self._dead_letter_queue.copy()

    def clear_dead_letter_queue(self) -> None:
        """清空死信队列"""
        self._dead_letter_queue.clear()

    def add_to_dead_letter_queue(self, event: Event, error: Exception) -> None:
        """添加到死信队列"""
        self._dead_letter_queue.append({
            'event': event,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })


class EventPersistenceManager:
    """事件持久化管理器"""

    def __init__(self, enable_persistence: bool = True):
        self.enable_persistence = enable_persistence
        self.persistence: Optional[EventPersistence] = None
        self._event_history: Dict[str, List[Event]] = defaultdict(list)

    def initialize_persistence(self, enable_persistence: bool) -> None:
        """初始化持久化"""
        if enable_persistence:
            self.persistence = EventPersistence()
        else:
            self.persistence = None

    def persist_event(self, event: Event) -> None:
        """持久化事件"""
        if self.persistence:
            self.persistence.save_event(event)

    def add_to_history(self, event: Event) -> None:
        """添加到历史记录"""
        event_type = event.event_type if hasattr(event, 'event_type') else str(event.event_type)
        self._event_history[event_type].append(event)

    def get_history(self, event_type: Optional[Union[EventType, str]] = None) -> Dict[str, List[Event]]:
        """获取历史记录"""
        if event_type:
            event_type_str = event_type if isinstance(event_type, str) else str(event_type)
            return {event_type_str: self._event_history.get(event_type_str, [])}
        return dict(self._event_history)


class EventStatisticsManager:
    """事件统计管理器"""

    def __init__(self):
        self._statistics = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'handlers_by_type': defaultdict(int),
            'errors_by_type': defaultdict(int),
            'performance_stats': defaultdict(list)
        }

    def update_statistics(self, event_type: str, handler_count: int = 0, error_count: int = 0) -> None:
        """更新统计信息"""
        self._statistics['total_events'] += 1
        self._statistics['events_by_type'][event_type] += 1
        self._statistics['handlers_by_type'][event_type] += handler_count
        self._statistics['errors_by_type'][event_type] += error_count

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return dict(self._statistics)

    def get_subscriber_count(self, event_type: Union[EventType, str]) -> int:
        """获取订阅者数量"""
        event_type_str = event_type if isinstance(event_type, str) else str(event_type)
        return self._statistics['handlers_by_type'].get(event_type_str, 0)

    def get_event_statistics(self) -> Dict[str, Any]:
        """获取事件统计"""
        return {
            'total_events': self._statistics['total_events'],
            'events_by_type': dict(self._statistics['events_by_type']),
            'errors_by_type': dict(self._statistics['errors_by_type'])
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return dict(self._statistics['performance_stats'])

    def get_recent_events(self, minutes: int = 5) -> Dict[str, int]:
        """获取最近事件"""
        recent = defaultdict(int)
        for event_type, events in self._statistics['events_by_type'].items():
            # 简化处理：返回所有事件，不进行时间过滤
            recent[event_type] = events
        return dict(recent)


class EventBus(BaseComponent):

    """事件总线 - 重构版 5.0.0 - 使用管理器模式"""

    def __init__(self, max_workers: int = None, enable_async: bool = True,
                 enable_persistence: bool = True, enable_retry: bool = True,
                 enable_monitoring: bool = True, batch_size: int = None,
                 max_queue_size: int = None, config: Optional[EventBusConfig] = None):
        """初始化事件总线 - 重构版：参数封装兼容

        Args:
            max_workers: 最大工作线程数
            enable_async: 是否启用异步处理
            enable_persistence: 是否启用持久化
            enable_retry: 是否启用重试
            enable_monitoring: 是否启用监控
            batch_size: 批次大小
            max_queue_size: 最大队列大小
            config: 配置对象（优先使用）
        """
        super().__init__("EventBus", "5.0.0", "事件总线核心组件")

        # 设置合理的默认值，避免过高的CPU占用
        import os
        cpu_count = os.cpu_count() or 4
        default_max_workers = min(cpu_count * 2, 8)  # 默认工作线程数为CPU核心数的2倍，但不超过8
        default_batch_size = 100  # 默认批处理大小
        default_max_queue_size = 1000  # 默认队列大小

        # 使用配置对象或从参数构造
        if config is None:
            config = EventBusConfig(
                max_workers=max_workers if max_workers is not None else default_max_workers,
                enable_async=enable_async,
                enable_persistence=enable_persistence,
                enable_retry=enable_retry,
                enable_monitoring=enable_monitoring,
                batch_size=batch_size if batch_size is not None else default_batch_size,
                max_queue_size=max_queue_size if max_queue_size is not None else default_max_queue_size
            )

        # 保存配置
        self._config = config

        # 保存配置到实例属性（向后兼容）
        self._save_config(
            max_workers=config.max_workers,
            enable_async=config.enable_async,
            enable_persistence=config.enable_persistence,
            enable_retry=config.enable_retry,
            enable_monitoring=config.enable_monitoring,
            batch_size=config.batch_size,
            max_queue_size=config.max_queue_size
        )

        # 初始化所有管理器
        self._initialize_managers(config.enable_persistence)

        # 初始化所有组件
        self._initialize_components(config.max_queue_size, config.batch_size)

        # 初始化统计信息
        self._initialize_statistics()

        # 初始化组件化组件（新架构）
        self._initialize_componentized_components()

    def _save_config(self, max_workers: int, enable_async: bool, enable_persistence: bool,
                     enable_retry: bool, enable_monitoring: bool, batch_size: int, max_queue_size: int):
        """保存配置参数（向后兼容）"""
        self.max_workers = max_workers
        self.enable_async = enable_async
        self.enable_persistence = enable_persistence
        self.enable_retry = enable_retry
        self.enable_monitoring = enable_monitoring
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size

    def _initialize_managers(self, enable_persistence: bool):
        """初始化所有管理器"""
        self.filter_manager = EventFilterManager()
        self.routing_manager = EventRoutingManager()
        self.persistence_manager = EventPersistenceManager(enable_persistence)
        self.statistics_manager = EventStatisticsManager()

        # 初始化持久化管理器
        self.persistence_manager.initialize_persistence(enable_persistence)

    def _initialize_components(self, max_queue_size: int, batch_size: int):
        """初始化所有组件"""
        self._initialize_core_components(max_queue_size)
        self._initialize_threading()
        self._initialize_batch_processing(batch_size)

    def _initialize_statistics(self):
        """初始化统计信息"""
        self._event_counter = 0
        self._processed_counter = 0
        self._start_time = time.time()
        # 初始化清理定时器相关属性
        self._cleanup_timer = None
        self._cleanup_interval = 3600  # 1小时清理一次
        self._max_history_age = 86400 * 7  # 保留7天的历史记录

    def _initialize_core_components(self, max_queue_size: int) -> None:
        """初始化核心组件"""
        self._event_queue = PriorityQueue(maxsize=max_queue_size)
        self._handlers = defaultdict(list)
        self._async_handlers = defaultdict(list)
        self._persistence = None
        self._retry_manager = None
        self._performance_monitor = None
        # 移除_event_history，因为现在由持久化管理器处理

    def _initialize_threading(self) -> None:
        """初始化线程和锁"""
        self._worker_threads = []
        self._running = False
        self._lock = threading.RLock()
        self._init_lock = threading.Lock()  # 初始化锁（保护初始化过程）
        self._event_counter = 0

    def _initialize_batch_processing(self, batch_size: int) -> None:
        """初始化批处理相关"""
        self._batch_queue = Queue(maxsize=batch_size * 2)
        self._batch_timer = None
        self._batch_interval = 1.0  # 1秒批处理间隔，减少CPU占用

    def _initialize_componentized_components(self):
        """初始化组件化组件（新架构）"""
        print("DEBUG: _initialize_componentized_components called")
        # 需要先初始化死信队列（在routing_manager中）
        if not hasattr(self.routing_manager, '_dead_letter_queue'):
            self.routing_manager._dead_letter_queue = deque(maxlen=MAX_RECORDS)

        # 初始化事件发布组件
        self._publisher = EventPublisher(
            filter_manager=self.filter_manager,
            routing_manager=self.routing_manager,
            persistence_manager=self.persistence_manager,
            statistics_manager=self.statistics_manager,
            event_queue=self._event_queue,
            lock=self._lock,
            dead_letter_queue=self.routing_manager._dead_letter_queue
        )
        print("DEBUG: _publisher initialized")

        # 初始化事件订阅组件
        self._subscriber = EventSubscriber(
            handlers=self._handlers,
            async_handlers=self._async_handlers,
            lock=self._lock
        )

        # 初始化事件处理组件（需要延迟初始化，因为retry_manager和performance_monitor稍后才初始化）
        # 将在_initialize_impl中完成

        # 初始化事件监控组件
        self._monitor = EventMonitor(
            statistics_manager=self.statistics_manager,
            event_queue=self._event_queue,
            max_queue_size=self._config.max_queue_size,
            worker_threads=self._worker_threads,
            start_time=self._start_time
        )

    def add_event_filter(self, filter_func: Callable) -> None:
        """添加事件过滤器"""
        self.filter_manager.add_event_filter(filter_func)

    def remove_event_filter(self, filter_func: Callable) -> None:
        """移除事件过滤器"""
        self.filter_manager.remove_event_filter(filter_func)

    def add_event_transformer(self, transformer_func: Callable) -> None:
        """添加事件转换器"""
        self.filter_manager.add_event_transformer(transformer_func)

    def remove_event_transformer(self, transformer_func: Callable) -> None:
        """移除事件转换器"""
        self.filter_manager.remove_event_transformer(transformer_func)

    def add_event_route(self, from_event: str, to_handlers: List[str]) -> None:
        """添加事件路由规则"""
        self.routing_manager.add_event_route(from_event, to_handlers)

    def remove_event_route(self, from_event: str) -> None:
        """移除事件路由规则"""
        self.routing_manager.remove_event_route(from_event)

    def get_dead_letter_events(self) -> List[dict]:
        """获取死信队列中的事件"""
        return self.routing_manager.get_dead_letter_events()

    def clear_dead_letter_queue(self) -> None:
        """清空死信队列"""
        self.routing_manager.clear_dead_letter_queue()

    def _add_to_dead_letter_queue(self, event: Event, error: Exception) -> None:
        """将失败的事件添加到死信队列"""
        self.routing_manager.add_to_dead_letter_queue(event, error)
        logger.warning(f"事件 {event.event_id} 已添加到死信队列: {error}")

    def _persist_event(self, event: Event) -> None:
        """持久化事件"""
        # 使用持久化管理器
        self.persistence_manager.persist_event(event)

    def _add_to_event_history(self, event: Event) -> None:
        """添加事件到历史记录"""
        # 使用持久化管理器添加历史记录
        self.persistence_manager.add_to_history(event)

    def _route_and_publish_events(self, event: Event) -> None:
        """路由并发布事件"""
        routed_handlers = self._route_event(event)
        if routed_handlers:
            # 为每个路由的目标创建新事件并发布
            for target_event_type in routed_handlers:
                try:
                    # 创建路由事件，保持原始事件的元数据
                    routed_event = Event(
                        event_type=target_event_type,
                        data=event.data.copy(),
                        source=event.source,
                        priority=event.priority,
                        correlation_id=event.correlation_id
                    )
                    # 递归发布路由事件
                    self.publish_event(routed_event)
                    logger.debug(f"路由事件 {event.event_id} 到 {target_event_type}")
                except Exception as e:
                    logger.warning(f"事件路由失败: {e}")

    def _enqueue_event(self, event: Event) -> None:
        """将事件加入处理队列"""
        with self._lock:
            self._event_counter += 1
            self._event_queue.put((event.priority.value, self._event_counter, event))

    def _handle_publish_error(self, event: Event, error: Exception) -> None:
        """处理发布事件错误"""
        with self._lock:
            if (self._dead_letter_queue.maxlen is not None and 
                    len(self._dead_letter_queue) >= self._dead_letter_queue.maxlen):
                self._dead_letter_queue.popleft()
            self._dead_letter_queue.append({
                'event': event.__dict__,
                'error': str(error),
                'timestamp': time.time()
            })

    def _apply_event_filters(self, event: Event) -> bool:
        """应用事件过滤器"""
        return self.filter_manager.apply_filters(event)

    def _apply_event_transformers(self, event: Event) -> Event:
        """应用事件转换器"""
        return self.filter_manager.apply_transformers(event)

    def _route_event(self, event: Event) -> List[str]:
        """根据路由规则确定目标处理器"""
        return self.routing_manager.route_event(event)

    def _initialize_impl(self) -> bool:
        """
        实现BaseComponent的初始化（线程安全，带重复初始化检查）

        Returns:
            bool: 初始化是否成功

        Thread Safety:
            此方法是线程安全的，使用锁保护初始化过程。
        """
        with self._init_lock:
            # 检查是否已经在运行（防止重复初始化导致线程泄漏）
            if self._running:
                logger.debug("EventBus 已经在运行中，跳过重复初始化")
                return True

            try:
                logger.info("开始初始化EventBus...")
                print("DEBUG: About to check enable_persistence")
                # 初始化持久化
                logger.info(f"初始化持久化: {self._config.enable_persistence}")
                print(f"DEBUG: enable_persistence = {self._config.enable_persistence}")
                if self._config.enable_persistence:
                    print("DEBUG: Initializing persistence")
                    self._persistence = EventPersistence()

                # 初始化重试管理器
                if self._config.enable_retry:
                    self._retry_manager = EventRetryManager(
                        max_retries=3,
                        retry_delay=1.0,
                        dead_letter_callback=self._add_to_dead_letter_queue
                    )
                    self._retry_manager.start()

                # 初始化性能监控
                if self._config.enable_monitoring:
                    self._performance_monitor = EventPerformanceMonitor()

                # 初始化事件处理组件（现在retry_manager和performance_monitor已初始化）
                self._processor = EventProcessor(
                    subscriber=self._subscriber,
                    statistics_manager=self.statistics_manager,
                    lock=self._lock,
                    retry_manager=self._retry_manager,
                    performance_monitor=self._performance_monitor
                )

                # 启动工作线程
                self._running = True
                for i in range(self._config.max_workers):
                    thread = threading.Thread(
                        target=self._process_events,
                        name=f"EventBus-Worker-{i}",
                        daemon=True
                    )
                    thread.start()
                    self._worker_threads.append(thread)

                # 启动批处理线程
                if self._config.batch_size > 1:
                    batch_thread = threading.Thread(
                        target=self._batch_processor,
                        name="EventBus-BatchProcessor",
                        daemon=True
                    )
                    batch_thread.start()
                    self._worker_threads.append(batch_thread)

                logger.info(f"EventBus 初始化完成，工作线程数: {self._config.max_workers}")
                return True

            except Exception as e:
                logger.error(f"EventBus 初始化失败: {e}")
                self._cleanup_on_init_failure()
                raise EventBusException(f"EventBus 初始化失败: {e}") from e

    def _cleanup_on_init_failure(self):
        """
        初始化失败时的清理

        确保在初始化失败时清理已创建的资源，防止资源泄漏。
        """
        logger.warning("执行初始化失败清理...")
        self._running = False

        # 停止重试管理器
        if self._retry_manager is not None:
            try:
                self._retry_manager.stop()
                logger.debug("重试管理器已停止")
            except Exception as e:
                logger.warning(f"停止重试管理器时出错: {e}")

        # 等待工作线程结束
        for i, thread in enumerate(self._worker_threads):
            try:
                thread.join(timeout=1.0)
                logger.debug(f"工作线程 {i} 已停止")
            except Exception:
                pass
        self._worker_threads.clear()

        logger.warning("初始化失败清理完成")

    def shutdown(self) -> bool:
        """关闭事件总线"""
        try:
            self.set_status(ComponentStatus.STOPPING)
            self._running = False

            # 停止重试管理器
            if self._retry_manager:
                self._retry_manager.stop()

            # 关闭事件持久化
            if self._persistence:
                self._persistence.shutdown()

            # 等待工作线程结束
            for thread in self._worker_threads:
                thread.join(timeout=5)

            self._worker_threads.clear()
            self.set_status(ComponentStatus.STOPPED)
            self.set_health(ComponentHealth.HEALTHY)

            logger.info("事件总线已关闭")
            return True

        except Exception as e:
            logger.error(f"关闭事件总线失败: {e}")
            return False

    def _start_impl(self) -> bool:
        """启动实现"""
        if not self._initialized:
            raise EventBusException("事件总线未初始化")

        # 初始化事件持久化
        if self._persistence and not self._persistence.initialize():
            logger.error("事件持久化初始化失败")
            return False

        self._running = True
        self.set_status(ComponentStatus.RUNNING)

        # 启动定期清理定时器
        self._start_cleanup_timer()

        return True

    def _stop_impl(self) -> bool:
        """停止实现"""
        self._running = False

        # 停止清理定时器
        self._stop_cleanup_timer()

        self.set_status(ComponentStatus.STOPPED)
        return True

    def subscribe(self, event_type: Union[EventType, str], handler: Callable,
                  priority: EventPriority = EventPriority.NORMAL,
                  async_handler: bool = False,
                  retry_on_failure: bool = True,
                  max_retries: int = 3):
        """订阅事件 - 优化版（委托给EventSubscriber组件）"""
        if not self._initialized:
            event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
            raise EventBusException("事件总线未初始化", event_type=event_type_str)

        # 使用组件化实现
        if hasattr(self, '_subscriber'):
            return self._subscriber.subscribe(
                event_type, handler, priority, async_handler,
                retry_on_failure, max_retries
            )
        else:
            # 向后兼容的原始实现
            event_type_str = str(event_type)
            handler_info = EventHandler(
                handler=handler,
                priority=priority,
                async_handler=async_handler,
                retry_on_failure=retry_on_failure,
                max_retries=max_retries
            )

            with self._lock:
                if async_handler:
                    self._async_handlers[event_type_str].append(handler_info)
                else:
                    self._handlers[event_type_str].append(handler_info)

            logger.debug(f"订阅事件: {event_type_str}")
            return True

    def subscribe_async(self, event_type: Union[EventType, str], handler: Callable):
        """订阅异步事件（委托给EventSubscriber组件）"""
        if hasattr(self, '_subscriber'):
            return self._subscriber.subscribe_async(event_type, handler)
        else:
            return self.subscribe(event_type, handler, async_handler=True)

    def unsubscribe(self, event_type: Union[EventType, str], handler: Callable):
        """取消订阅事件（委托给EventSubscriber组件）"""
        if not self._initialized:
            raise EventBusException("事件总线未初始化")

        if hasattr(self, '_subscriber'):
            return self._subscriber.unsubscribe(event_type, handler)
        else:
            # 向后兼容的原始实现
            event_type_str = str(event_type)

            with self._lock:
                self._handlers[event_type_str] = [
                    h for h in self._handlers[event_type_str]
                    if h.handler != handler
                ]

                self._async_handlers[event_type_str] = [
                    h for h in self._async_handlers[event_type_str]
                    if h.handler != handler
                ]

            logger.debug(f"取消订阅事件: {event_type_str}")
            return True

    def check_health(self) -> ComponentHealth:
        """检查EventBus健康状态（委托给EventMonitor组件）"""
        if hasattr(self, '_monitor'):
            return self._monitor.check_health(self._status)
        else:
            # 向后兼容的原始实现
            try:
                if self._status in [ComponentStatus.ERROR, ComponentStatus.UNHEALTHY]:
                    return ComponentHealth.UNHEALTHY

                queue_size = self._event_queue.qsize()
                if queue_size > self.max_queue_size * 0.9:
                    return ComponentHealth.UNHEALTHY

                active_threads = sum(1 for t in self._worker_threads if t.is_alive())
                if active_threads < self.max_workers:
                    return ComponentHealth.UNHEALTHY

                return ComponentHealth.HEALTHY

            except Exception as e:
                logger.error(f"EventBus健康检查失败: {str(e)}")
                return ComponentHealth.UNHEALTHY

    def get_statistics(self) -> Dict[str, Any]:
        """获取EventBus统计信息（委托给EventMonitor组件）"""
        if hasattr(self, '_monitor'):
            return self._monitor.get_statistics(
                event_counter=getattr(self, '_event_counter', 0),
                processed_counter=getattr(self, '_processed_counter', 0),
                handlers=self._handlers,
                async_handlers=self._async_handlers,
                filter_manager=self.filter_manager,
                routing_manager=self.routing_manager,
                performance_monitor=getattr(self, '_performance_monitor', None)
            )
        else:
            # 向后兼容的原始实现
            try:
                stats = self.statistics_manager.get_statistics()
                stats.update({
                    "total_events_published": getattr(self, '_event_counter', 0),
                    "total_events_processed": getattr(self, '_processed_counter', 0),
                    "active_handlers": (sum(len(handlers) for handlers in self._handlers.values()) + 
                                          sum(len(handlers) for handlers in self._async_handlers.values())),
                    "queue_size": self._event_queue.qsize() if hasattr(self, '_event_queue') else 0,
                    "worker_threads": len(getattr(self, '_worker_threads', [])),
                    "event_filters": len(self.filter_manager._filters),
                    "event_transformers": len(getattr(self.filter_manager, '_transformers', [])),
                    "event_routes": len(getattr(self.routing_manager, '_routes', {})),
                    "dead_letter_queue_size": len(getattr(self.routing_manager, '_dead_letter_queue', [])),
                    "uptime": time.time() - getattr(self, '_start_time', time.time()),
                })

                if hasattr(self, '_performance_monitor') and self._performance_monitor:
                    try:
                        perf_stats = {
                            "avg_processing_time": getattr(self._performance_monitor, 'avg_processing_time', 0),
                            "max_processing_time": getattr(self._performance_monitor, 'max_processing_time', 0),
                            "total_processed_events": getattr(self._performance_monitor, 'total_processed', 0),
                        }
                        stats.update(perf_stats)
                    except Exception:
                        pass

                return stats

            except Exception as e:
                return {"error": f"获取统计信息失败: {str(e)}"}

    def publish(self, event_type: Union[EventType, str], data: Optional[Dict[str, Any]] = None,
                source: str = "system", priority: EventPriority = EventPriority.NORMAL,
                event_id: Optional[str] = None, correlation_id: Optional[str] = None) -> str:
        """发布事件 - 优化版（委托给EventPublisher组件）"""
        if not self._initialized:
            raise EventBusException("事件总线未初始化")

        if hasattr(self, '_publisher'):
            return self._publisher.publish(
                event_type, data, source, priority, event_id, correlation_id
            )
        else:
            # 向后兼容的原始实现
            event = Event(
                event_type=event_type,
                data=data or {},
                source=source,
                priority=priority,
                event_id=event_id,
                correlation_id=correlation_id
            )

            event_id = self.publish_event(event)
            if event_id is None:
                raise EventBusException("事件发布失败，未生成事件ID")
            return event_id

    def publish_event(self, event: Event) -> str:
        """发布事件对象 - 增强版（委托给EventPublisher组件）"""
        if hasattr(self, '_publisher'):
            return self._publisher.publish_event(event)
        else:
            # 向后兼容的原始实现
            try:
                if not self._apply_event_filters(event):
                    logger.debug(f"事件被过滤: {event.event_id} ({event.event_type})")
                    if event.event_id is None:
                        raise EventBusException("事件ID不能为空")
                    return event.event_id

                transformed_event = self._apply_event_transformers(event)
                self._persist_event(transformed_event)
                self._add_to_event_history(transformed_event)
                self._route_and_publish_events(transformed_event)
                self._enqueue_event(transformed_event)
                self.statistics_manager.update_statistics(
                    str(transformed_event.event_type), handler_count=0, error_count=0)
                logger.debug(f"发布事件: {transformed_event.event_id} ({transformed_event.event_type})")

                if transformed_event.event_id is None:
                    raise EventBusException("事件ID不能为空")
                return transformed_event.event_id

            except Exception as e:
                logger.error(f"发布事件失败: {e}")
                self._handle_publish_error(event, e)
                raise EventBusException(f"发布事件失败: {e}")

    def _process_events(self):
        """处理事件的工作线程 - 优化版"""
        import traceback
        
        while self._running:
            try:
                # 从队列获取事件
                priority, _, event = self._event_queue.get(timeout=1)
                self._handle_event(event)
                self._event_queue.task_done()

            except Empty:
                # 队列为空是正常情况（timeout），不需要记录错误
                continue
            except Exception as e:
                if self._running:  # 只有在运行时才记录错误
                    event_info = (f"事件ID: {getattr(event, 'event_id', 'unknown')}, "
                                  f"类型: {getattr(event, 'event_type', 'unknown')}") if 'event' in locals() else "未知事件"
                    logger.error(
                        f"处理事件异常: {e}\n"
                        f"事件信息: {event_info}\n"
                        f"异常类型: {type(e).__name__}\n"
                        f"异常详情: {traceback.format_exc()}",
                        exc_info=True
                    )

    def _batch_processor(self):
        """批处理线程 - 优化版，减少CPU占用"""
        batch = []
        last_process_time = time.time()
        empty_loop_count = 0  # 空循环计数器

        while self._running:
            try:
                # 收集批处理事件，使用更长的超时时间减少CPU占用
                try:
                    event = self._batch_queue.get(timeout=0.5)  # 从0.1秒增加到0.5秒
                    batch.append(event)
                    empty_loop_count = 0  # 重置计数器
                except Exception:
                    empty_loop_count += 1
                    # 如果连续空循环超过10次，增加睡眠时间
                    if empty_loop_count > 10:
                        time.sleep(0.1)  # 增加100ms睡眠，减少CPU占用
                        empty_loop_count = 0
                    continue

                current_time = time.time()

                # 检查是否需要处理批次
                # 安全获取 batch_size：优先使用实例属性，其次使用配置对象，最后使用默认值
                if hasattr(self, 'batch_size'):
                    batch_size = self.batch_size
                elif hasattr(self, '_config') and hasattr(self._config, 'batch_size'):
                    batch_size = self._config.batch_size
                else:
                    batch_size = DEFAULT_BATCH_SIZE
                    
                if (len(batch) >= batch_size
                        or (batch and current_time - last_process_time >= self._batch_interval)):

                    if batch:
                        self._handle_batch(batch)
                        batch.clear()
                        last_process_time = current_time

            except Exception as e:
                if self._running:
                    logger.error(f"批处理异常: {e}")

    def _handle_batch(self, events: List[Event]):
        """处理事件批次"""
        if not events:
            return

        # 按事件类型分组
        event_groups = defaultdict(list)
        for event in events:
            event_type_str = str(event.event_type)
            event_groups[event_type_str].append(event)

        # 批量处理每种类型的事件
        for event_type, event_batch in event_groups.items():
            try:
                self._handle_event_batch(event_type, event_batch)
            except Exception as e:
                logger.error(f"批量处理事件失败: {event_type}, 错误: {e}")

    def _handle_event_batch(self, event_type: str, events: List[Event]):
        """处理同类型事件批次"""
        with self._lock:
            handlers = self._handlers[event_type].copy()
            async_handlers = self._async_handlers[event_type].copy()

        # 批量执行同步处理器
        for handler_info in handlers:
            try:
                for event in events:
                    self._sync_handler_wrapper(handler_info, event)
            except Exception as e:
                logger.error(f"批量同步处理器异常: {e}")

        # 批量执行异步处理器
        for handler_info in async_handlers:
            try:
                for event in events:
                    asyncio.create_task(self._async_handler_wrapper(handler_info.handler, event))
            except Exception as e:
                logger.error(f"批量异步处理器异常: {e}")

    def _handle_event(self, event: Event) -> EventProcessingResult:
        """处理单个事件 - 重构版（委托给EventProcessor组件）"""
        if hasattr(self, '_processor'):
            return self._processor.handle_event(event)
        else:
            # 向后兼容的原始实现
            start_time = time.time()
            event_type_str = str(event.event_type)

            result = EventProcessingResult(
                event_id=getattr(event, 'event_id', ''),
                event_type=event_type_str,
                success=False,
                processing_time=0.0
            )

            try:
                handlers, async_handlers = self._get_event_handlers(event_type_str)
                logger.debug(f"事件 {event.event_id} 找到 {len(handlers)} 个同步处理器和 {len(async_handlers)} 个异步处理器")
                
                sync_success, sync_count = self._execute_sync_handlers_with_count(event, handlers)
                async_success, async_count = self._execute_async_handlers_with_count(event, async_handlers)

                result.success = sync_success and async_success
                result.sync_handlers_executed = sync_count
                result.async_handlers_executed = async_count

                processing_time = time.time() - start_time
                result.processing_time = processing_time
                self._update_event_statistics(event, result.success, processing_time, sync_count, async_count)
                
                logger.info(f"事件处理完成: {event.event_id} ({event_type_str}), "
                            f"成功: {result.success}, "
                            f"同步处理器: {sync_count}, 异步处理器: {async_count}, "
                            f"处理时间: {processing_time:.3f}s")

            except Exception as e:
                logger.error(f"处理事件失败: {e}", exc_info=True)
                if hasattr(result, 'errors') and result.errors is not None:
                    result.errors.append(str(e))
                else:
                    result.errors = [str(e)]
                self._handle_event_error(event, event_type_str)

            return result

    def _get_event_handlers(self, event_type_str: str) -> tuple:
        """获取事件处理器列表"""
        with self._lock:
            handlers = self._handlers[event_type_str].copy()
            async_handlers = self._async_handlers[event_type_str].copy()
        return handlers, async_handlers

    def _execute_sync_handlers_with_count(self, event: Event, handlers: List) -> Tuple[bool, int]:
        """执行同步处理器，返回执行结果和数量"""
        success = True
        executed_count = 0

        for handler_info in handlers:
            try:
                context = HandlerExecutionContext(
                    event=event,
                    handler_info=handler_info,
                    start_time=time.time()
                )
                self._sync_handler_wrapper_with_context(context)
                executed_count += 1
            except Exception as e:
                logger.error(f"同步处理器异常: {e}")
                success = False
                self._handle_sync_handler_error(event, handler_info, e)

        return success, executed_count

    def _sync_handler_wrapper_with_context(self, context: HandlerExecutionContext) -> None:
        """使用上下文包装的同步处理器执行"""
        try:
            # 检查是否过期
            if context.is_expired:
                logger.warning(f"处理器执行超时: {context.handler_info}")
                return

            # 执行处理器
            self._sync_handler_wrapper(context.handler_info, context.event)

        except Exception as e:
            logger.error(f"同步处理器执行失败: {e}")
            raise

    def _execute_async_handlers_with_count(self, event: Event, async_handlers: List) -> Tuple[bool, int]:
        """执行异步处理器 - 简化为同步调用避免线程问题"""
        success = True
        executed_count = 0

        for handler_info in async_handlers:
            try:
                # 直接在当前线程中创建新的事件循环来执行异步处理器
                # 这样避免了复杂的线程间异步调用问题
                import asyncio
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    # 同步执行异步处理器
                    new_loop.run_until_complete(
                        self._async_handler_wrapper(handler_info, event)
                    )
                    executed_count += 1
                    logger.debug(f"异步处理器执行成功: {handler_info.handler.__name__}")
                finally:
                    new_loop.close()
            except Exception as e:
                logger.error(f"异步处理器执行异常: {e}", exc_info=True)
                success = False

        return success, executed_count

    def _async_handler_wrapper_with_context(self, context: HandlerExecutionContext) -> None:
        """使用上下文包装的异步处理器执行"""
        try:
            # 检查是否过期
            if context.is_expired:
                logger.warning(f"异步处理器执行超时: {context.handler_info}")
                return

            # 执行异步处理器
            # 这里已经在事件循环中，可以直接使用 await
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，创建任务
                    asyncio.create_task(self._async_handler_wrapper(context.handler_info.handler, context.event))
                else:
                    # 如果事件循环不运行，直接运行
                    loop.run_until_complete(
                        self._async_handler_wrapper(context.handler_info.handler, context.event))
            except RuntimeError:
                # 没有事件循环，创建新的事件循环
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(
                        self._async_handler_wrapper(context.handler_info.handler, context.event))
                finally:
                    new_loop.close()

        except Exception as e:
            logger.error(f"异步处理器执行失败: {e}", exc_info=True)
            raise

    def _handle_sync_handler_error(self, event: Event, handler_info, error: Exception):
        """处理同步处理器错误"""
        if (handler_info.retry_on_failure and self._retry_manager and
                getattr(event, 'retry_count', 0) < handler_info.max_retries):
            self._retry_manager.add_retry_event(event, error)
        else:
            self._add_to_dead_letter_queue(event, error)

    def _update_event_statistics(self, event: Event, success: bool, processing_time: float,
                                 sync_count: int, async_count: int):
        """更新事件统计信息"""
        # 记录性能监控
        if self._performance_monitor:
            self._performance_monitor.record_event_end(
                str(event.event_type), success, processing_time
            )

        total_handlers = sync_count + async_count

        if success:
            self.statistics_manager.update_statistics(str(event.event_type),
                                                      handler_count=total_handlers, error_count=0)
            if self._persistence and event.event_id is not None:
                self._persistence.update_event_status(event.event_id, EventStatus.COMPLETED)
        else:
            self.statistics_manager.update_statistics(str(event.event_type),
                                                      handler_count=0, error_count=1)
            if self._persistence and event.event_id is not None:
                self._persistence.update_event_status(event.event_id, EventStatus.FAILED)

    def _handle_event_error(self, event: Event, event_type_str: str):
        """处理事件处理错误"""
        self.statistics_manager.update_statistics(event_type_str, handler_count=0, error_count=1)
        if self._persistence and event.event_id is not None:
            self._persistence.update_event_status(event.event_id, EventStatus.FAILED)

    def _sync_handler_wrapper(self, handler_info: EventHandler, event: Event):
        """同步处理器包装器 - 优化版"""
        try:
            start_time = time.time()
            handler_info.handler(event)
            processing_time = time.time() - start_time

            # 记录处理时间
            if processing_time > handler_info.timeout:
                logger.warning(f"处理器超时: {processing_time:.2f}s > {handler_info.timeout}s")

        except Exception as e:
            logger.error(f"同步处理器执行失败: {e}")
            raise

    async def _async_handler_wrapper(self, handler: Callable, event: Event):
        """异步处理器包装器 - 优化版"""
        try:
            start_time = time.time()
            if asyncio.iscoroutinefunction(handler):
                await asyncio.wait_for(handler(event), timeout=DEFAULT_TIMEOUT)
            else:
                # 在线程池中执行同步函数
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, handler, event),
                    timeout=DEFAULT_TIMEOUT
                )

            processing_time = time.time() - start_time
            if processing_time > DEFAULT_TIMEOUT:
                logger.warning(f"异步处理器超时: {processing_time:.2f}s > {DEFAULT_TIMEOUT}s")

        except Exception as e:
            logger.error(f"异步处理器执行失败: {e}")
            raise

    def get_event_history(self, event_type: Optional[Union[EventType, str]] = None,
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None,
                          limit: Optional[int] = None) -> Dict[str, List[Event]]:
        """获取事件历史"""
        try:
            # 使用持久化管理器获取历史记录
            return self.persistence_manager.get_history(event_type)
        except Exception as e:
            logger.error(f"获取事件历史失败: {e}")
            return {}

    def clear_history(self, before_time: Optional[float] = None):
        """清除事件历史"""
        try:
            # 注意：持久化管理器当前不支持按时间清除
            # 这是一个简化实现，实际应该扩展持久化管理器
            logger.info(f"清除事件历史 (before_time: {before_time})")

        except Exception as e:
            logger.error(f"清除事件历史失败: {e}")

    def get_subscriber_count(self, event_type: Union[EventType, str]) -> int:
        """获取订阅者数量（委托给EventMonitor组件）"""
        if hasattr(self, '_monitor'):
            return self._monitor.get_subscriber_count(
                event_type, self._handlers, self._async_handlers
            )
        else:
            return self.statistics_manager.get_subscriber_count(event_type)

    def get_event_statistics(self) -> Dict[str, Any]:
        """获取事件统计"""
        try:
            # 获取统计管理器的统计
            stats = self.statistics_manager.get_event_statistics()

            # 添加性能统计
            if self._performance_monitor:
                performance_stats = self._performance_monitor.get_performance_stats()
                if isinstance(performance_stats, dict):
                    stats['performance'] = performance_stats

            # 添加持久化统计
            if self._persistence:
                persistence_stats = self._persistence.get_stats()
                if isinstance(persistence_stats, dict):
                    stats['persistence'] = persistence_stats

            return stats

        except Exception as e:
            logger.error(f"获取事件统计失败: {e}")
            return {
                'total_events': 0,
                'processed_events': 0,
                'failed_events': 0,
                'pending_events': 0,
                'error': str(e)
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息（委托给EventMonitor组件）"""
        if hasattr(self, '_monitor'):
            return self._monitor.get_performance_stats()
        else:
            # 向后兼容的原始实现
            stats = self.statistics_manager.get_performance_stats()
            if self._performance_monitor:
                perf_stats = self._performance_monitor.get_performance_stats()
                stats.update(perf_stats)
            return stats

    def get_recent_events(self, minutes: int = 5) -> Dict[str, int]:
        """获取最近的事件（委托给EventMonitor组件）"""
        if hasattr(self, '_monitor'):
            return self._monitor.get_recent_events(minutes)
        else:
            return self.statistics_manager.get_recent_events(minutes)

    def health_check(self) -> bool:
        """健康检查"""
        try:
            # 检查基本状态
            if not self._initialized:
                return False

            # 检查工作线程状态
            active_threads = sum(1 for thread in self._worker_threads if thread.is_alive())
            if active_threads == 0 and self._running:
                return False

            # 检查队列状态
            if self._event_queue.qsize() > MAX_QUEUE_SIZE:  # 队列过大
                return False

            self.set_health(ComponentHealth.HEALTHY)
            return True

        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            self.set_health(ComponentHealth.UNHEALTHY)
            return False

    def _start_cleanup_timer(self):
        """启动定期清理定时器"""
        try:
            # 确保清理定时器相关属性已初始化
            if not hasattr(self, '_cleanup_timer'):
                self._cleanup_timer = None
            if not hasattr(self, '_cleanup_interval'):
                self._cleanup_interval = 3600  # 默认1小时
            if not hasattr(self, '_max_history_age'):
                self._max_history_age = 86400 * 7  # 默认7天
            
            if self._cleanup_timer is None:
                self._cleanup_timer = threading.Timer(
                    self._cleanup_interval, self._cleanup_expired_events)
                self._cleanup_timer.daemon = True
                self._cleanup_timer.start()
                logger.info(f"事件历史清理定时器已启动，间隔: {self._cleanup_interval}秒")
        except Exception as e:
            logger.error(f"启动清理定时器失败: {e}", exc_info=True)

    def _stop_cleanup_timer(self):
        """停止定期清理定时器"""
        try:
            if self._cleanup_timer:
                self._cleanup_timer.cancel()
                self._cleanup_timer = None
                logger.info("事件历史清理定时器已停止")
        except Exception as e:
            logger.error(f"停止清理定时器失败: {e}")

    def _cleanup_expired_events(self):
        """清理过期事件 - 防止内存泄漏"""
        try:
            current_time = time.time()
            cutoff_time = current_time - self._max_history_age

            with self._lock:
                # 清理过期事件
                original_size = len(self._event_history)
                self._event_history = deque(
                    [e for e in self._event_history if e.timestamp >= cutoff_time],
                    maxlen=MAX_QUEUE_SIZE
                )
                cleaned_count = original_size - len(self._event_history)

                if cleaned_count > 0:
                    logger.info(f"已清理 {cleaned_count} 个过期事件，当前历史大小: {len(self._event_history)}")

                # 重新启动定时器
                self._cleanup_timer = threading.Timer(
                    self._cleanup_interval, self._cleanup_expired_events)
                self._cleanup_timer.daemon = True
                self._cleanup_timer.start()

        except Exception as e:
            logger.error(f"清理过期事件失败: {e}")
            # 即使清理失败也要重新启动定时器
            try:
                self._cleanup_timer = threading.Timer(
                    self._cleanup_interval, self._cleanup_expired_events)
                self._cleanup_timer.daemon = True
                self._cleanup_timer.start()
            except Exception as e2:
                logger.error(f"重新启动清理定时器失败: {e2}")
