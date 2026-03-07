"""
事件发布组件

负责事件的创建、过滤、转换和发布。
"""

from typing import Dict, Optional, Union
import logging
import time

from ..models import Event
from ..types import EventType, EventPriority
from ..utils import EventRetryManager
from ...foundation.exceptions.core_exceptions import EventBusException

logger = logging.getLogger(__name__)


class EventPublisher:
    """事件发布组件"""

    def __init__(self, filter_manager, routing_manager, persistence_manager, statistics_manager,
                 event_queue, lock, dead_letter_queue):
        """初始化事件发布组件

        Args:
            filter_manager: 过滤器管理器
            routing_manager: 路由管理器
            persistence_manager: 持久化管理器
            statistics_manager: 统计管理器
            event_queue: 事件队列
            lock: 线程锁
            dead_letter_queue: 死信队列
        """
        self.filter_manager = filter_manager
        self.routing_manager = routing_manager
        self.persistence_manager = persistence_manager
        self.statistics_manager = statistics_manager
        self._event_queue = event_queue
        self._lock = lock
        self._dead_letter_queue = dead_letter_queue
        self._event_counter = 0

    def publish(self, event_type: Union[EventType, str], data: Optional[Dict] = None,
                source: str = "system", priority: EventPriority = EventPriority.NORMAL,
                event_id: Optional[str] = None, correlation_id: Optional[str] = None) -> str:
        """发布事件

        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件来源
            priority: 事件优先级
            event_id: 事件ID（可选）
            correlation_id: 关联ID（可选）

        Returns:
            事件ID
        """
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
        """发布事件对象

        Args:
            event: 事件对象

        Returns:
            事件ID

        Raises:
            EventBusException: 发布失败时抛出
        """
        try:
            # 应用事件过滤器
            if not self._apply_event_filters(event):
                logger.debug(f"事件被过滤: {event.event_id} ({event.event_type})")
                if event.event_id is None:
                    raise EventBusException("事件ID不能为空")
                return event.event_id

            # 应用事件转换器
            transformed_event = self._apply_event_transformers(event)

            # 保存到持久化存储
            self.persistence_manager.persist_event(transformed_event)

            # 添加到事件历史
            self.persistence_manager.add_to_history(transformed_event)

            # 检查路由规则并触发路由的事件
            self._route_and_publish_events(transformed_event)

            # 添加到处理队列
            self._enqueue_event(transformed_event)

            # 更新统计管理器
            self.statistics_manager.update_statistics(
                str(transformed_event.event_type),
                handler_count=0,
                error_count=0
            )

            # 记录事件发布信息（包括订阅者数量）
            # 注意：这里需要从外部获取订阅者数量，因为EventPublisher没有直接访问
            logger.info(f"发布事件: {transformed_event.event_id} ({transformed_event.event_type}), 来源: {transformed_event.source}")
            logger.debug(f"事件数据: {transformed_event.data}")

            if transformed_event.event_id is None:
                raise EventBusException("事件ID不能为空")
            return transformed_event.event_id

        except Exception as e:
            logger.error(f"发布事件失败: {e}")
            self._handle_publish_error(event, e)
            raise EventBusException(f"发布事件失败: {e}")

    def _apply_event_filters(self, event: Event) -> bool:
        """应用事件过滤器"""
        return self.filter_manager.apply_filters(event)

    def _apply_event_transformers(self, event: Event) -> Event:
        """应用事件转换器"""
        return self.filter_manager.apply_transformers(event)

    def _route_and_publish_events(self, event: Event) -> None:
        """路由并发布事件"""
        routed_handlers = self.routing_manager.route_event(event)
        if routed_handlers:
            # 为每个路由的目标创建新事件并发布
            for target_event_type in routed_handlers:
                try:
                    routed_event = Event(
                        event_type=target_event_type,
                        data=event.data.copy(),
                        source=event.source,
                        priority=event.priority,
                        correlation_id=event.correlation_id
                    )
                    # 递归发布路由事件（注意：这里需要访问EventPublisher实例）
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
            if self._dead_letter_queue.maxlen is not None and len(self._dead_letter_queue) >= self._dead_letter_queue.maxlen:
                self._dead_letter_queue.popleft()
            self._dead_letter_queue.append({
                'event': event.__dict__,
                'error': str(error),
                'timestamp': time.time()
            })

