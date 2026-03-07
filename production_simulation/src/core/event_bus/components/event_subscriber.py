"""
事件订阅组件

负责事件的订阅、取消订阅和处理器管理。
"""

from typing import Dict, List, Callable, Union
import logging
# defaultdict 已移除（未使用）

from ..models import EventHandler
from ..types import EventType, EventPriority

logger = logging.getLogger(__name__)


class EventSubscriber:
    """事件订阅组件"""

    def __init__(self, handlers: Dict, async_handlers: Dict, lock):
        """初始化事件订阅组件

        Args:
            handlers: 同步处理器字典
            async_handlers: 异步处理器字典
            lock: 线程锁
        """
        self._handlers = handlers
        self._async_handlers = async_handlers
        self._lock = lock

    def subscribe(self, event_type: Union[EventType, str], handler: Callable,
                  priority: EventPriority = EventPriority.NORMAL,
                  async_handler: bool = False,
                  retry_on_failure: bool = True,
                  max_retries: int = 3) -> bool:
        """订阅事件

        Args:
            event_type: 事件类型
            handler: 事件处理器
            priority: 处理器优先级
            async_handler: 是否为异步处理器
            retry_on_failure: 是否在失败时重试
            max_retries: 最大重试次数

        Returns:
            是否订阅成功
        """
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

    def subscribe_async(self, event_type: Union[EventType, str], handler: Callable) -> bool:
        """订阅异步事件

        Args:
            event_type: 事件类型
            handler: 事件处理器

        Returns:
            是否订阅成功
        """
        return self.subscribe(event_type, handler, async_handler=True)

    def unsubscribe(self, event_type: Union[EventType, str], handler: Callable) -> bool:
        """取消订阅事件

        Args:
            event_type: 事件类型
            handler: 事件处理器

        Returns:
            是否取消成功
        """
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

    def get_handlers(self, event_type: Union[EventType, str]) -> tuple:
        """获取事件处理器列表

        Args:
            event_type: 事件类型

        Returns:
            (同步处理器列表, 异步处理器列表)
        """
        event_type_str = str(event_type)
        with self._lock:
            handlers = self._handlers[event_type_str].copy()
            async_handlers = self._async_handlers[event_type_str].copy()
        return handlers, async_handlers

    def get_subscriber_count(self, event_type: Union[EventType, str]) -> int:
        """获取订阅者数量

        Args:
            event_type: 事件类型

        Returns:
            订阅者数量
        """
        event_type_str = str(event_type)
        with self._lock:
            sync_count = len(self._handlers.get(event_type_str, []))
            async_count = len(self._async_handlers.get(event_type_str, []))
            return sync_count + async_count

