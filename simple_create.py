content = '''
"""事件订阅组件

负责事件的订阅、取消订阅和处理器管理。
"""

from typing import Dict, List, Callable, Union
import logging

from core.event_bus.models import EventHandler
from core.event_bus.types import EventType, EventPriority

logger = logging.getLogger(__name__)


class EventSubscriber:
    """事件订阅组件"""

    def __init__(self, handlers: Dict, async_handlers: Dict, lock):
        """初始化事件订阅组件"""
        self._handlers = handlers
        self._async_handlers = async_handlers
        self._lock = lock

    def subscribe(self, event_type: Union[EventType, str], handler: Callable,
                  priority: EventPriority = EventPriority.NORMAL,
                  async_handler: bool = False,
                  retry_on_failure: bool = True,
                  max_retries: int = 3) -> bool:
        """订阅事件"""
        event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
        handler_info = EventHandler(
            handler=handler,
            priority=priority,
            async_handler=async_handler,
            retry_on_failure=retry_on_failure,
            max_retries=max_retries
        )

        self._lock.acquire()
        try:
            if async_handler:
                if event_type_str not in self._async_handlers:
                    self._async_handlers[event_type_str] = []
                self._async_handlers[event_type_str].append(handler_info)
            else:
                if event_type_str not in self._handlers:
                    self._handlers[event_type_str] = []
                self._handlers[event_type_str].append(handler_info)
        finally:
            self._lock.release()

        logger.debug(f"订阅事件: {event_type_str}")
        return True

    def subscribe_async(self, event_type: Union[EventType, str], handler: Callable) -> bool:
        """订阅异步事件"""
        return self.subscribe(event_type, handler, async_handler=True)

    def unsubscribe(self, event_type: Union[EventType, str], handler: Callable) -> bool:
        """取消订阅事件"""
        event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)

        self._lock.acquire()
        try:
            if event_type_str in self._handlers:
                self._handlers[event_type_str] = [
                    h for h in self._handlers[event_type_str]
                    if h.handler != handler
                ]
            if event_type_str in self._async_handlers:
                self._async_handlers[event_type_str] = [
                    h for h in self._async_handlers[event_type_str]
                    if h.handler != handler
                ]
        finally:
            self._lock.release()

        logger.debug(f"取消订阅事件: {event_type_str}")
        return True

    def get_handlers(self, event_type: Union[EventType, str]) -> tuple:
        """获取事件处理器列表"""
        event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
        self._lock.acquire()
        try:
            handlers = self._handlers.get(event_type_str, []).copy()
            async_handlers = self._async_handlers.get(event_type_str, []).copy()
        finally:
            self._lock.release()
        return handlers, async_handlers

    def get_subscriber_count(self, event_type: Union[EventType, str]) -> int:
        """获取订阅者数量"""
        event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
        self._lock.acquire()
        try:
            sync_count = len(self._handlers.get(event_type_str, []))
            async_count = len(self._async_handlers.get(event_type_str, []))
            return sync_count + async_count
        finally:
            self._lock.release()
'''

with open('src/core/event_bus/components/event_subscriber.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Recreated EventSubscriber file')

