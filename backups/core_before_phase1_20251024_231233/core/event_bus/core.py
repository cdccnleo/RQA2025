"""
事件总线核心实现
EventBus主类和核心功能 - 重构版 4.0.0
"""

from typing import Dict, List, Callable, Any, Union, Optional
import time
import logging
import asyncio
import threading
from queue import PriorityQueue, Queue
from collections import defaultdict, deque

from ..foundation.base import BaseComponent, ComponentStatus, ComponentHealth
from ..foundation.exceptions.core_exceptions import EventBusException
from .types import EventType, EventPriority
from .models import Event, EventHandler
from .utils import EventRetryManager, EventPerformanceMonitor
from .persistence.event_persistence import EventPersistence, PersistenceMode

logger = logging.getLogger(__name__)


class EventBus(BaseComponent):

    """事件总线 - 重构版 4.0.0"""

    def __init__(self, max_workers: int = 10, enable_async: bool = True,

                 enable_persistence: bool = True, enable_retry: bool = True,
                 enable_monitoring: bool = True, batch_size: int = 10,
                 max_queue_size: int = 10000):
        super().__init__("EventBus", "4.0.0", "事件总线核心组件")

        # 保存配置参数
        self.max_workers = max_workers
        self.enable_async = enable_async
        self.enable_persistence = enable_persistence
        self.enable_retry = enable_retry
        self.enable_monitoring = enable_monitoring
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size

        # 初始化各个组件
        self._initialize_core_components(max_queue_size)
        self._initialize_persistence(enable_persistence)
        self._initialize_threading()
        self._initialize_event_management()
        self._initialize_memory_management()
        self._initialize_statistics()
        self._initialize_batch_processing(batch_size)

        # 初始化统计计数器
        self._event_counter = 0
        self._processed_counter = 0
        self._start_time = time.time()

    def _initialize_core_components(self, max_queue_size: int) -> None:
        """初始化核心组件"""
        self._event_queue = PriorityQueue(maxsize=max_queue_size)
        self._handlers = defaultdict(list)
        self._async_handlers = defaultdict(list)
        self._event_history = deque(maxlen=10000)  # 限制历史记录大小，防止内存泄漏
        self._persistence = None
        self._retry_manager = None
        self._performance_monitor = None

    def _initialize_persistence(self, enable_persistence: bool) -> None:
        """初始化事件持久化"""
        if enable_persistence:
            try:
                # 延迟导入EventPersistence
                persistence_config = {
                    'storage_path': './event_storage',
                    'cleanup_interval': 3600,
                    'max_age_days': 30
                }
                self._persistence = EventPersistence(
                    mode=PersistenceMode.FILE,
                    config=persistence_config
                )
            except ImportError as e:
                logger.warning(f"EventPersistence导入失败，使用内存模式: {e}")
                self._persistence = None

    def _initialize_threading(self) -> None:
        """初始化线程和锁"""
        self._worker_threads = []
        self._running = False
        self._lock = threading.RLock()
        self._event_counter = 0

    def _initialize_event_management(self) -> None:
        """初始化事件管理增强功能"""
        self._event_filters: List[Callable] = []  # 事件过滤器
        self._event_transformers: List[Callable] = []  # 事件转换器
        self._event_routes: Dict[str, List[str]] = defaultdict(list)  # 事件路由规则
        self._dead_letter_queue = deque(maxlen=1000)  # 死信队列

    def _initialize_memory_management(self) -> None:
        """初始化内存管理"""
        self._cleanup_timer = None
        self._cleanup_interval = 3600  # 每小时清理一次
        self._max_history_age = 86400  # 24小时过期时间

    def _initialize_statistics(self) -> None:
        """初始化统计信息"""
        self._stats = {
            'total_events': 0,
            'processed_events': 0,
            'failed_events': 0,
            'pending_events': 0
        }

    def _initialize_batch_processing(self, batch_size: int) -> None:
        """初始化批处理相关"""
        self._batch_queue = Queue(maxsize=batch_size * 2)
        self._batch_timer = None
        self._batch_interval = 0.1  # 100ms批处理间隔

    def add_event_filter(self, filter_func: Callable) -> None:
        """添加事件过滤器"""
        with self._lock:
            self._event_filters.append(filter_func)

    def remove_event_filter(self, filter_func: Callable) -> None:
        """移除事件过滤器"""
        with self._lock:
            if filter_func in self._event_filters:
                self._event_filters.remove(filter_func)

    def add_event_transformer(self, transformer_func: Callable) -> None:
        """添加事件转换器"""
        with self._lock:
            self._event_transformers.append(transformer_func)

    def remove_event_transformer(self, transformer_func: Callable) -> None:
        """移除事件转换器"""
        with self._lock:
            if transformer_func in self._event_transformers:
                self._event_transformers.remove(transformer_func)

    def add_event_route(self, from_event: str, to_handlers: List[str]) -> None:
        """添加事件路由规则"""
        with self._lock:
            self._event_routes[from_event] = to_handlers

    def remove_event_route(self, from_event: str) -> None:
        """移除事件路由规则"""
        with self._lock:
            self._event_routes.pop(from_event, None)

    def get_dead_letter_events(self) -> List[dict]:
        """获取死信队列中的事件"""
        with self._lock:
            return list(self._dead_letter_queue)

    def clear_dead_letter_queue(self) -> None:
        """清空死信队列"""
        with self._lock:
            self._dead_letter_queue.clear()

    def _add_to_dead_letter_queue(self, event: Event, error: Exception) -> None:
        """将失败的事件添加到死信队列"""
        with self._lock:
            if len(self._dead_letter_queue) >= self._dead_letter_queue.maxlen:
                self._dead_letter_queue.popleft()
            self._dead_letter_queue.append({
                'event': event.__dict__,
                'error': str(error),
                'timestamp': time.time(),
                'retry_count': getattr(event, 'retry_count', 0),
                'max_retries': getattr(event, 'max_retries', 3)
            })
        logger.warning(f"事件 {event.event_id} 已添加到死信队列: {error}")

    def _apply_event_filters(self, event: Event) -> bool:
        """应用事件过滤器"""
        for filter_func in self._event_filters:
            try:
                if not filter_func(event):
                    return False
            except Exception as e:
                logger.warning(f"事件过滤器执行失败: {e}")
        return True

    def _apply_event_transformers(self, event: Event) -> Event:
        """应用事件转换器"""
        transformed_event = event
        for transformer_func in self._event_transformers:
            try:
                transformed_event = transformer_func(transformed_event)
            except Exception as e:
                logger.warning(f"事件转换器执行失败: {e}")
        return transformed_event

    def _route_event(self, event: Event) -> List[str]:
        """根据路由规则确定目标处理器"""
        event_type_str = str(event.event_type)
        if event_type_str in self._event_routes:
            return self._event_routes[event_type_str]
        return []

    def _initialize_impl(self) -> bool:
        """实现BaseComponent的初始化"""
        try:
            # 初始化持久化
            if self.enable_persistence:
                self._persistence = EventPersistence()

            # 初始化重试管理器
            if self.enable_retry:
                self._retry_manager = EventRetryManager(
                    max_retries=3,
                    retry_delay=1.0,
                    dead_letter_callback=self._add_to_dead_letter_queue
                )
                self._retry_manager.start()

            # 初始化性能监控
            if self.enable_monitoring:
                self._performance_monitor = EventPerformanceMonitor()

            # 启动工作线程
            self._running = True
            for i in range(self.max_workers):
                thread = threading.Thread(target=self._process_events, daemon=True)
                thread.start()
                self._worker_threads.append(thread)

            # 启动批处理线程
            if self.batch_size > 1:
                batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
                batch_thread.start()
                self._worker_threads.append(batch_thread)

            logger.info(f"事件总线初始化完成，工作线程数: {self.max_workers}")
            return True

        except Exception as e:
            logger.error(f"事件总线初始化失败: {e}")
            raise EventBusException(f"事件总线初始化失败: {e}")

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
        """订阅事件 - 优化版"""
        if not self._initialized:
            raise EventBusException("事件总线未初始化")

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
        """订阅异步事件"""
        return self.subscribe(event_type, handler, async_handler=True)

    def unsubscribe(self, event_type: Union[EventType, str], handler: Callable):
        """取消订阅事件"""
        if not self._initialized:
            raise EventBusException("事件总线未初始化")

        event_type_str = str(event_type)

        with self._lock:
            # 移除同步处理器
            self._handlers[event_type_str] = [
                h for h in self._handlers[event_type_str]
                if h.handler != handler
            ]

            # 移除异步处理器
            self._async_handlers[event_type_str] = [
                h for h in self._async_handlers[event_type_str]
                if h.handler != handler
            ]

        logger.debug(f"取消订阅事件: {event_type_str}")
        return True

    def check_health(self) -> ComponentHealth:
        """检查EventBus健康状态"""
        try:
            # 检查基本状态
            if self._status in [ComponentStatus.ERROR, ComponentStatus.UNHEALTHY]:
                return ComponentHealth.UNHEALTHY

            # 检查队列状态
            queue_size = self._event_queue.qsize()
            if queue_size > self.max_queue_size * 0.9:  # 队列使用率超过90%
                return ComponentHealth.UNHEALTHY

            # 检查工作线程状态
            active_threads = sum(1 for t in self._worker_threads if t.is_alive())
            if active_threads < self.max_workers:
                return ComponentHealth.UNHEALTHY

            return ComponentHealth.HEALTHY

        except Exception as e:
            logger.error(f"EventBus健康检查失败: {str(e)}")
            return ComponentHealth.UNHEALTHY

    def get_statistics(self) -> Dict[str, Any]:
        """获取EventBus统计信息"""
        try:
            stats = {
                "total_events_published": getattr(self, '_event_counter', 0),
                "total_events_processed": getattr(self, '_processed_counter', 0),
                "active_handlers": sum(len(handlers) for handlers in self._handlers.values()) +
                sum(len(handlers) for handlers in self._async_handlers.values()),
                "queue_size": self._event_queue.qsize() if hasattr(self, '_event_queue') else 0,
                "worker_threads": len(getattr(self, '_worker_threads', [])),
                "event_filters": len(getattr(self, '_event_filters', [])),
                "event_transformers": len(getattr(self, '_event_transformers', [])),
                "event_routes": len(getattr(self, '_event_routes', {})),
                "dead_letter_queue_size": len(getattr(self, '_dead_letter_queue', [])),
                "uptime": time.time() - getattr(self, '_start_time', time.time()),
            }

            # 添加性能监控信息（如果启用）
            if hasattr(self, '_performance_monitor') and self._performance_monitor:
                try:
                    perf_stats = {
                        "avg_processing_time": getattr(self._performance_monitor, 'avg_processing_time', 0),
                        "max_processing_time": getattr(self._performance_monitor, 'max_processing_time', 0),
                        "total_processed_events": getattr(self._performance_monitor, 'total_processed', 0),
                    }
                    stats.update(perf_stats)
                except:
                    pass

            return stats

        except Exception as e:
            return {"error": f"获取统计信息失败: {str(e)}"}

    def publish(self, event_type: Union[EventType, str], data: Dict[str, Any] = None,


                source: str = "system", priority: EventPriority = EventPriority.NORMAL,
                event_id: Optional[str] = None, correlation_id: Optional[str] = None) -> str:
        """发布事件 - 优化版"""
        if not self._initialized:
            raise EventBusException("事件总线未初始化")

        event = Event(
            event_type=event_type,
            data=data or {},
            source=source,
            priority=priority,
            event_id=event_id,
            correlation_id=correlation_id
        )

        return self.publish_event(event)

    def publish_event(self, event: Event) -> str:
        """发布事件对象 - 增强版"""
        try:
            # 应用事件过滤器
            if not self._apply_event_filters(event):
                logger.debug(f"事件被过滤: {event.event_id} ({event.event_type})")
                return event.event_id

            # 应用事件转换器
            transformed_event = self._apply_event_transformers(event)

            # 保存到持久化存储
            if self._persistence:
                try:
                    success = self._persistence.store_event(
                        event_id=transformed_event.event_id,
                        event_type=str(transformed_event.event_type),
                        event_data=transformed_event.data,
                        timestamp=transformed_event.timestamp,
                        metadata={
                            'source': transformed_event.source,
                            'priority': transformed_event.priority.value,
                            'correlation_id': transformed_event.correlation_id
                        }
                    )
                except Exception as e:
                    logger.warning(f"事件持久化失败: {transformed_event.event_id}, 错误: {e}")
                    success = False
                if not success:
                    logger.warning(f"事件持久化失败: {transformed_event.event_id}")

            # 添加到事件历史（限制大小）
            if len(self._event_history) >= self._event_history.maxlen:
                self._event_history.popleft()
            self._event_history.append(transformed_event)

            # 检查路由规则并触发路由的事件
            routed_handlers = self._route_event(transformed_event)
            if routed_handlers:
                # 为每个路由的目标创建新事件并发布
                for target_event_type in routed_handlers:
                    try:
                        # 创建路由事件，保持原始事件的元数据
                        routed_event = Event(
                            event_type=target_event_type,
                            data=transformed_event.data.copy(),
                            source=transformed_event.source,
                            priority=transformed_event.priority,
                            correlation_id=transformed_event.correlation_id
                        )
                        # 递归发布路由事件
                        self.publish_event(routed_event)
                        logger.debug(f"路由事件 {transformed_event.event_id} 到 {target_event_type}")
                    except Exception as e:
                        logger.warning(f"事件路由失败: {e}")

            # 添加到处理队列（使用计数器确保唯一性）
            with self._lock:
                self._event_counter += 1
                self._event_queue.put((transformed_event.priority.value,
                                      self._event_counter, transformed_event))

            self._stats['total_events'] += 1
            logger.debug(f"发布事件: {transformed_event.event_id} ({transformed_event.event_type})")

            return transformed_event.event_id

        except Exception as e:
            logger.error(f"发布事件失败: {e}")
            # 将失败的事件添加到死信队列
            with self._lock:
                if len(self._dead_letter_queue) >= self._dead_letter_queue.maxlen:
                    self._dead_letter_queue.popleft()
                self._dead_letter_queue.append({
                    'event': event.__dict__,
                    'error': str(e),
                    'timestamp': time.time()
                })
            raise EventBusException(f"发布事件失败: {e}")

    def _process_events(self):
        """处理事件的工作线程 - 优化版"""
        while self._running:
            try:
                # 从队列获取事件
                priority, _, event = self._event_queue.get(timeout=1)
                self._handle_event(event)
                self._event_queue.task_done()

            except Exception as e:
                if self._running:  # 只有在运行时才记录错误
                    logger.error(f"处理事件异常: {e}")

    def _batch_processor(self):
        """批处理线程"""
        batch = []
        last_process_time = time.time()

        while self._running:
            try:
                # 收集批处理事件
                try:
                    event = self._batch_queue.get(timeout=0.1)
                    batch.append(event)
                except Exception:
                    pass

                current_time = time.time()

                # 检查是否需要处理批次
                if (len(batch) >= self.batch_size
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

    def _handle_event(self, event: Event):
        """处理单个事件 - 优化版"""
        start_time = time.time()
        success = True

        try:
            event_type_str = str(event.event_type)

            # 处理同步处理器
            with self._lock:
                handlers = self._handlers[event_type_str].copy()
                async_handlers = self._async_handlers[event_type_str].copy()

            # 执行同步处理器
            for handler_info in handlers:
                try:
                    self._sync_handler_wrapper(handler_info, event)
                except Exception as e:
                    logger.error(f"同步处理器异常: {e}")
                    success = False
                    # 如果启用重试且未达到最大重试次数，则加入重试队列
                    if (handler_info.retry_on_failure and self._retry_manager and
                            getattr(event, 'retry_count', 0) < handler_info.max_retries):
                        self._retry_manager.add_retry_event(event, e)
                    else:
                        # 达到最大重试次数或不启用重试，直接加入死信队列
                        self._add_to_dead_letter_queue(event, e)

            # 执行异步处理器
            for handler_info in async_handlers:
                try:
                    asyncio.create_task(self._async_handler_wrapper(handler_info.handler, event))
                except Exception as e:
                    logger.error(f"异步处理器异常: {e}")
                    success = False
                    # 对于异步处理器，简化处理，直接加入死信队列
                    self._add_to_dead_letter_queue(event, e)

            # 更新统计
            processing_time = time.time() - start_time
            if self._performance_monitor:
                self._performance_monitor.record_event_end(
                    str(event.event_type), success, processing_time
                )

            if success:
                self._stats['processed_events'] += 1
                self._processed_counter += 1
                if self._persistence:
                    self._persistence.update_event_status(event.event_id, 'completed')
            else:
                self._stats['failed_events'] += 1
                if self._persistence:
                    self._persistence.update_event_status(event.event_id, 'failed')

        except Exception as e:
            logger.error(f"处理事件失败: {e}")
            self._stats['failed_events'] += 1
            if self._persistence:
                self._persistence.update_event_status(event.event_id, 'failed')

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
                await asyncio.wait_for(handler(event), timeout=30.0)
            else:
                # 在线程池中执行同步函数
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, handler, event),
                    timeout=30.0
                )

            processing_time = time.time() - start_time
            if processing_time > 30.0:
                logger.warning(f"异步处理器超时: {processing_time:.2f}s > 30.0s")

        except Exception as e:
            logger.error(f"异步处理器执行失败: {e}")
            raise

    def get_event_history(self, event_type: Optional[Union[EventType, str]] = None,


                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None,
                          limit: Optional[int] = None) -> List[Event]:
        """获取事件历史"""
        try:
            events = list(self._event_history)

            # 按事件类型过滤
            if event_type:
                event_type_str = str(event_type)
                events = [e for e in events if str(e.event_type) == event_type_str]

            # 按时间范围过滤
            if start_time:
                events = [e for e in events if e.timestamp >= start_time]
            if end_time:
                events = [e for e in events if e.timestamp <= end_time]

            # 按时间倒序排序
            events.sort(key=lambda x: x.timestamp, reverse=True)

            # 限制数量
            if limit:
                events = events[:limit]

            return events

        except Exception as e:
            logger.error(f"获取事件历史失败: {e}")
            return []

    def clear_history(self, before_time: Optional[float] = None):
        """清除事件历史"""
        try:
            if before_time:
                self._event_history = deque(
                    [e for e in self._event_history if e.timestamp >= before_time],
                    maxlen=10000
                )
            else:
                self._event_history.clear()

            logger.info("事件历史已清除")

        except Exception as e:
            logger.error(f"清除事件历史失败: {e}")

    def get_subscriber_count(self, event_type: Union[EventType, str]) -> int:
        """获取订阅者数量"""
        try:
            event_type_str = str(event_type)
            with self._lock:
                return len(self._handlers[event_type_str]) + len(self._async_handlers[event_type_str])
        except Exception as e:
            logger.error(f"获取订阅者数量失败: {e}")
            return 0

    def get_event_statistics(self) -> Dict[str, Any]:
        """获取事件统计"""
        try:
            stats = self._stats.copy()

            # 添加性能统计
            if self._performance_monitor:
                stats['performance'] = self._performance_monitor.get_performance_stats()

            # 添加持久化统计
            if self._persistence:
                stats['persistence'] = self._persistence.get_event_stats()

            return stats

        except Exception as e:
            logger.error(f"获取事件统计失败: {e}")
            return self._stats.copy()

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if self._performance_monitor:
            return self._performance_monitor.get_performance_stats()
        return {}

    def get_recent_events(self, minutes: int = 5) -> Dict[str, int]:
        """获取最近事件统计"""
        if self._performance_monitor:
            return self._performance_monitor.get_recent_events(minutes)
        return {}

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
            if self._event_queue.qsize() > 10000:  # 队列过大
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
            if self._cleanup_timer is None:
                self._cleanup_timer = threading.Timer(
                    self._cleanup_interval, self._cleanup_expired_events)
                self._cleanup_timer.daemon = True
                self._cleanup_timer.start()
                logger.info(f"事件历史清理定时器已启动，间隔: {self._cleanup_interval}秒")
        except Exception as e:
            logger.error(f"启动清理定时器失败: {e}")

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
                    maxlen=10000
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
