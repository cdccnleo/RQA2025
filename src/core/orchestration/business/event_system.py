"""
RQA2025系统事件系统
Event System for RQA2025 System

实现观察者模式，支持事件的发布和订阅
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging
from src.unified_exceptions import handle_business_exceptions

logger = logging.getLogger(__name__)


class Event:
    """
    事件类

    封装事件数据和元信息
    """

    def __init__(self, event_type: str, data: Any = None,
                 source: str = None, timestamp: Optional[float] = None,
                 correlation_id: Optional[str] = None, **kwargs):
        """
        初始化事件

        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件源
            timestamp: 时间戳
            correlation_id: 关联ID
            **kwargs: 额外属性
        """
        import time
        self.event_type = event_type
        self.data = data or {}
        self.source = source or "unknown"
        self.timestamp = timestamp or time.time()
        self.correlation_id = correlation_id
        self.metadata = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'event_type': self.event_type,
            'data': self.data,
            'source': self.source,
            'timestamp': self.timestamp,
            'correlation_id': self.correlation_id,
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        return f"Event(type={self.event_type}, source={self.source}, timestamp={self.timestamp})"


class EventHandler:
    """
    事件处理器接口

    定义事件处理器的标准接口
    """

    def can_handle(self, event: Event) -> bool:
        """
        检查是否可以处理该事件

        Args:
            event: 事件对象

        Returns:
            是否可以处理
        """
        return True

    def handle(self, event: Event) -> None:
        """
        处理事件

        Args:
            event: 事件对象
        """
        raise NotImplementedError("子类必须实现handle方法")

    def get_supported_event_types(self) -> List[str]:
        """
        获取支持的事件类型

        Returns:
            支持的事件类型列表
        """
        return []

    def get_priority(self) -> int:
        """
        获取处理器优先级（数字越小优先级越高）

        Returns:
            优先级数字
        """
        return 100


class EventPublisher:
    """
    事件发布器

    负责发布事件到所有订阅者
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._subscribers: Dict[str, List[Tuple[EventHandler, str]]] = defaultdict(list)
        self._event_history: List[Event] = []
        self._max_history_size = 1000
        self._async_mode = False
        self._thread_pool = None

    def enable_async_mode(self, max_workers: int = 4) -> None:
        """
        启用异步模式

        Args:
            max_workers: 最大工作线程数
        """
        import concurrent.futures
        self._async_mode = True
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"事件发布器 '{self.name}' 已启用异步模式")

    def disable_async_mode(self) -> None:
        """禁用异步模式"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        self._async_mode = False
        logger.info(f"事件发布器 '{self.name}' 已禁用异步模式")

    @handle_business_exceptions
    def subscribe(self, event_type: str, handler: EventHandler) -> str:
        """
        订阅事件

        Args:
            event_type: 事件类型
            handler: 事件处理器

        Returns:
            订阅ID
        """
        import uuid
        subscription_id = str(uuid.uuid4())

        # 按优先级排序插入
        priority = handler.get_priority()
        subscribers = self._subscribers[event_type]

        # 找到插入位置（优先级从高到低）
        insert_pos = 0
        for i, (existing_handler, _) in enumerate(subscribers):
            if priority < existing_handler.get_priority():
                insert_pos = i
                break
            insert_pos = i + 1

        subscribers.insert(insert_pos, (handler, subscription_id))

        logger.info(
            f"处理器 {handler.__class__.__name__} 已订阅事件 '{event_type}' (ID: {subscription_id})")
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        取消订阅

        Args:
            subscription_id: 订阅ID

        Returns:
            是否成功取消
        """
        for event_type, subscribers in self._subscribers.items():
            for i, (handler, sub_id) in enumerate(subscribers):
                if sub_id == subscription_id:
                    subscribers.pop(i)
                    logger.info(f"订阅 {subscription_id} 已取消")
                    return True
        return False

    def unsubscribe_handler(self, event_type: str, handler: EventHandler) -> bool:
        """
        取消处理器订阅

        Args:
            event_type: 事件类型
            handler: 事件处理器

        Returns:
            是否成功取消
        """
        if event_type in self._subscribers:
            original_length = len(self._subscribers[event_type])
            self._subscribers[event_type] = [
                (h, sub_id) for h, sub_id in self._subscribers[event_type]
                if h is not handler
            ]
            removed = original_length - len(self._subscribers[event_type])
            if removed > 0:
                logger.info(f"处理器 {handler.__class__.__name__} 已从事件 '{event_type}' 取消订阅")
                return True
        return False

    @handle_business_exceptions
    def publish(self, event: Event) -> int:
        """
        发布事件

        Args:
            event: 事件对象

        Returns:
            通知的处理器数量
        """
        event_type = event.event_type
        notified_count = 0

        if event_type not in self._subscribers:
            logger.debug(f"事件 '{event_type}' 没有订阅者")
            return 0

        # 记录事件历史
        self._record_event(event)

        # 获取订阅者列表的副本，避免并发修改问题
        subscribers = self._subscribers[event_type].copy()

        logger.debug(f"发布事件 '{event_type}' 给 {len(subscribers)} 个订阅者")

        # 通知所有订阅者
        for handler, subscription_id in subscribers:
            try:
                if handler.can_handle(event):
                    if self._async_mode and self._thread_pool:
                        # 异步处理
                        self._thread_pool.submit(self._handle_event_async,
                                                 handler, event, subscription_id)
                    else:
                        # 同步处理
                        self._handle_event_sync(handler, event, subscription_id)
                    notified_count += 1
                else:
                    logger.debug(f"处理器 {handler.__class__.__name__} 拒绝处理事件 '{event_type}'")
            except Exception as e:
                logger.error(f"事件处理器 {handler.__class__.__name__} 执行失败: {e}")

        return notified_count

    def _handle_event_sync(self, handler: EventHandler, event: Event, subscription_id: str) -> None:
        """
        同步处理事件

        Args:
            handler: 事件处理器
            event: 事件对象
            subscription_id: 订阅ID
        """
        try:
            logger.debug(f"同步处理事件 '{event.event_type}' 由 {handler.__class__.__name__}")
            handler.handle(event)
        except Exception as e:
            logger.error(f"事件处理器 {handler.__class__.__name__} 同步处理失败: {e}")

    def _handle_event_async(self, handler: EventHandler, event: Event, subscription_id: str) -> None:
        """
        异步处理事件

        Args:
            handler: 事件处理器
            event: 事件对象
            subscription_id: 订阅ID
        """
        try:
            logger.debug(f"异步处理事件 '{event.event_type}' 由 {handler.__class__.__name__}")
            handler.handle(event)
        except Exception as e:
            logger.error(f"事件处理器 {handler.__class__.__name__} 异步处理失败: {e}")

    def _record_event(self, event: Event) -> None:
        """
        记录事件历史

        Args:
            event: 事件对象
        """
        self._event_history.append(event)

        # 限制历史记录数量
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)

    def get_subscriptions(self, event_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        获取订阅信息

        Args:
            event_type: 事件类型（可选，不指定则返回所有）

        Returns:
            订阅信息字典
        """
        if event_type:
            subscribers = self._subscribers.get(event_type, [])
            return {event_type: [handler.__class__.__name__ for handler, _ in subscribers]}

        result = {}
        for et, subscribers in self._subscribers.items():
            result[et] = [handler.__class__.__name__ for handler, _ in subscribers]
        return result

    def get_event_history(self, event_type: Optional[str] = None,
                          limit: int = 100) -> List[Event]:
        """
        获取事件历史

        Args:
            event_type: 事件类型过滤（可选）
            limit: 返回记录数量限制

        Returns:
            事件历史列表
        """
        history = self._event_history

        if event_type:
            history = [e for e in history if e.event_type == event_type]

        return history[-limit:].copy()

    def clear_event_history(self) -> None:
        """清空事件历史"""
        self._event_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        event_type_counts = defaultdict(int)
        for event in self._event_history:
            event_type_counts[event.event_type] += 1

        return {
            'total_events': len(self._event_history),
            'event_types': dict(event_type_counts),
            'subscriptions': {et: len(subs) for et, subs in self._subscribers.items()},
            'total_subscriptions': sum(len(subs) for subs in self._subscribers.values()),
            'async_mode': self._async_mode
        }


# 全局事件发布器实例
global_event_publisher = EventPublisher("global")


# 便捷函数
def publish_event(event_type: str, data: Any = None, source: str = None, **kwargs) -> int:
    """
    发布事件

    Args:
        event_type: 事件类型
        data: 事件数据
        source: 事件源
        **kwargs: 额外参数

    Returns:
        通知的处理器数量
    """
    event = Event(event_type, data, source, **kwargs)
    return global_event_publisher.publish(event)


def subscribe_event(event_type: str, handler: EventHandler) -> str:
    """
    订阅事件

    Args:
        event_type: 事件类型
        handler: 事件处理器

    Returns:
        订阅ID
    """
    return global_event_publisher.subscribe(event_type, handler)


def unsubscribe_event(subscription_id: str) -> bool:
    """
    取消订阅事件

    Args:
        subscription_id: 订阅ID

    Returns:
        是否成功取消
    """
    return global_event_publisher.unsubscribe(subscription_id)


def enable_async_events(max_workers: int = 4) -> None:
    """
    启用异步事件处理

    Args:
        max_workers: 最大工作线程数
    """
    global_event_publisher.enable_async_mode(max_workers)


def disable_async_events() -> None:
    """禁用异步事件处理"""
    global_event_publisher.disable_async_mode()


def get_event_stats() -> Dict[str, Any]:
    """
    获取事件统计信息

    Returns:
        统计信息
    """
    return global_event_publisher.get_stats()


# 示例事件处理器
class LoggingEventHandler(EventHandler):
    """日志记录事件处理器"""

    def can_handle(self, event: Event) -> bool:
        return True  # 处理所有事件

    def handle(self, event: Event) -> None:
        logger.info(f"事件记录: {event}")

    def get_supported_event_types(self) -> List[str]:
        return ["*"]  # 支持所有事件类型

    def get_priority(self) -> int:
        return 1000  # 最低优先级


class MetricsEventHandler(EventHandler):
    """指标收集事件处理器"""

    def __init__(self):
        self.event_counts = defaultdict(int)

    def can_handle(self, event: Event) -> bool:
        return True

    def handle(self, event: Event) -> None:
        self.event_counts[event.event_type] += 1
        logger.debug(f"指标更新: {event.event_type} 计数={self.event_counts[event.event_type]}")

    def get_metrics(self) -> Dict[str, int]:
        return dict(self.event_counts)

    def get_priority(self) -> int:
        return 500  # 中等优先级


# 初始化默认处理器
def initialize_default_handlers():
    """初始化默认事件处理器"""
    logging_handler = LoggingEventHandler()
    metrics_handler = MetricsEventHandler()

    # 订阅所有事件
    global_event_publisher.subscribe("*", logging_handler)
    global_event_publisher.subscribe("*", metrics_handler)

    logger.info("默认事件处理器已初始化")


# 自动初始化
initialize_default_handlers()
