
import asyncio
import threading

from .shared_interfaces import ILogger, StandardLogger
from typing import Any, Dict, Callable, Optional, Set
"""
事件处理器

职责：处理事件的分发和执行逻辑
"""


class EventHandler:
    """
    事件处理器

    职责：管理事件处理器的注册、分发和执行
    """

    def __init__(self, logger: Optional[ILogger] = None):
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")
        self._handlers: Dict[str, Set[Callable]] = {}
        self._async_handlers: Dict[str, Set[Callable]] = {}
        self._lock = threading.RLock()

    def register_handler(self, event_type: str, handler: Callable[[Any], None]) -> None:
        """注册同步事件处理器"""
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = set()
            self._handlers[event_type].add(handler)
            self.logger.log_debug(f"已注册事件处理器: {event_type}")

    def register_async_handler(self, event_type: str, handler: Callable[[Any], asyncio.Future[Any]]) -> None:
        """注册异步事件处理器"""
        with self._lock:
            if event_type not in self._async_handlers:
                self._async_handlers[event_type] = set()
            self._async_handlers[event_type].add(handler)
            self.logger.log_debug(f"已注册异步事件处理器: {event_type}")

    def unregister_handler(self, event_type: str, handler: Callable) -> None:
        """注销事件处理器"""
        with self._lock:
            if event_type in self._handlers:
                self._handlers[event_type].discard(handler)
            if event_type in self._async_handlers:
                self._async_handlers[event_type].discard(handler)
            self.logger.log_debug(f"已注销事件处理器: {event_type}")

    def dispatch_event(self, event_type: str, event_data: Any) -> None:
        """分发事件到所有注册的处理器"""
        # 同步处理器
        if event_type in self._handlers:
            for handler in self._handlers[event_type].copy():
                try:
                    handler(event_data)
                except Exception as e:
                    self.logger.log_error(f"事件处理器执行失败: {event_type}, {e}")

        # 异步处理器
        if event_type in self._async_handlers:
            for handler in self._async_handlers[event_type].copy():
                try:
                    # 在新线程中执行异步处理器
                    threading.Thread(
                        target=self._execute_async_handler,
                        args=(handler, event_data),
                        daemon=True
                    ).start()
                except Exception as e:
                    self.logger.log_error(f"异步事件处理器执行失败: {event_type}, {e}")

    def _execute_async_handler(self, handler: Callable[[Any], asyncio.Future[Any]], event_data: Any):
        """执行异步事件处理器"""
        try:
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(handler(event_data))
            loop.close()
        except Exception as e:
            self.logger.log_error(f"异步处理器执行异常: {e}")

    def get_handler_count(self, event_type: str) -> int:
        """获取指定事件类型的处理器数量"""
        with self._lock:
            sync_count = len(self._handlers.get(event_type, set()))
            async_count = len(self._async_handlers.get(event_type, set()))
            return sync_count + async_count

    def clear_handlers(self, event_type: Optional[str] = None) -> None:
        """清除事件处理器"""
        with self._lock:
            if event_type:
                self._handlers.pop(event_type, None)
                self._async_handlers.pop(event_type, None)
                self.logger.log_info(f"已清除事件处理器: {event_type}")
            else:
                self._handlers.clear()
                self._async_handlers.clear()
                self.logger.log_info("已清除所有事件处理器")
