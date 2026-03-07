"""
事件总线组件

职责:
- 事件的订阅和发布
- 事件处理器管理
- 事件历史记录
"""

import logging
import asyncio
from typing import Dict, List, Callable, Any
from collections import defaultdict, deque

from ..models.event_models import EventType, Event, create_event

logger = logging.getLogger(__name__)


class EventBus:
    """
    事件总线组件

    提供发布-订阅模式的事件管理
    支持同步和异步事件发布
    """

    def __init__(self, config: 'EventBusConfig'):
        """
        初始化事件总线

        Args:
            config: 事件总线配置
        """
        self.config = config
        self._handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_history: deque = deque(
            maxlen=config.max_history_size if config.enable_history else 0
        )
        self._event_count = 0

        logger.info(f"事件总线初始化完成 (历史记录: {config.max_history_size})")

    def subscribe(self, event_type: EventType, handler: Callable):
        """
        订阅事件

        Args:
            event_type: 事件类型
            handler: 处理函数
        """
        self._handlers[event_type].append(handler)
        logger.debug(f"事件订阅: {event_type.value}, 处理器数: {len(self._handlers[event_type])}")

    def unsubscribe(self, event_type: EventType, handler: Callable):
        """
        取消订阅

        Args:
            event_type: 事件类型
            handler: 处理函数
        """
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            logger.debug(f"取消订阅: {event_type.value}")

    def publish(self, event_type: EventType, data: Dict[str, Any], source: str = ""):
        """
        发布事件（同步）

        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件源
        """
        # 创建事件对象
        event = create_event(event_type, data, source)

        # 记录历史
        if self.config.enable_history:
            self._event_history.append(event)

        # 记录日志
        if self.config.enable_logging:
            logger.debug(f"发布事件: {event_type.value}, 处理器数: {len(self._handlers[event_type])}")

        # 调用所有处理器
        for handler in self._handlers[event_type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"事件处理器异常 [{event_type.value}]: {e}")

        self._event_count += 1

    async def publish_async(self, event_type: EventType, data: Dict[str, Any], source: str = ""):
        """
        发布事件（异步）

        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件源
        """
        if not self.config.enable_async:
            # 如果未启用异步，回退到同步
            self.publish(event_type, data, source)
            return

        # 创建事件对象
        event = create_event(event_type, data, source)

        # 记录历史
        if self.config.enable_history:
            self._event_history.append(event)

        # 异步调用处理器
        tasks = []
        for handler in self._handlers[event_type]:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(event))
            else:
                # 同步函数包装为异步
                tasks.append(asyncio.to_thread(handler, event))

        # 并发执行
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 检查异常
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"异步事件处理器异常 [{event_type.value}]: {result}")

        self._event_count += 1

    def get_event_history(self) -> List[Event]:
        """
        获取事件历史

        Returns:
            List[Event]: 事件历史列表
        """
        return list(self._event_history)

    def clear_history(self):
        """清空事件历史"""
        self._event_history.clear()
        logger.info("事件历史已清空")

    def get_handler_count(self, event_type: EventType) -> int:
        """获取事件处理器数量"""
        return len(self._handlers[event_type])

    def get_total_events(self) -> int:
        """获取总事件数"""
        return self._event_count

    def get_subscribed_event_types(self) -> List[EventType]:
        """获取已订阅的事件类型"""
        return [event_type for event_type in self._handlers.keys()]

    def get_status(self) -> Dict[str, Any]:
        """获取事件总线状态"""
        return {
            'total_events': self._event_count,
            'history_size': len(self._event_history),
            'subscribed_types': len(self._handlers),
            'config': {
                'enable_history': self.config.enable_history,
                'max_history_size': self.config.max_history_size,
                'enable_async': self.config.enable_async
            }
        }


# 从configs导入（避免循环导入时用Any）
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..configs.orchestrator_configs import EventBusConfig
