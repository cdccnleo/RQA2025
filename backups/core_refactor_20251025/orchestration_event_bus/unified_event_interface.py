#!/usr/bin/env python3
"""
统一事件总线接口

定义核心服务层事件总线的标准接口，确保所有事件总线实现统一的API。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class EventPriority(Enum):
    """事件优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


class EventDeliveryMode(Enum):
    """事件传递模式"""
    SYNCHRONOUS = "sync"  # 同步传递
    ASYNCHRONOUS = "async"  # 异步传递
    AT_LEAST_ONCE = "at_least_once"  # 至少一次传递
    EXACTLY_ONCE = "exactly_once"  # 精确一次传递


class EventPersistence(Enum):
    """事件持久化策略"""
    NONE = "none"  # 不持久化
    MEMORY = "memory"  # 内存持久化
    DATABASE = "database"  # 数据库持久化
    FILE = "file"  # 文件持久化
    DISTRIBUTED = "distributed"  # 分布式持久化


@dataclass
class Event:
    """
    事件数据类

    表示系统中传递的事件对象。
    """
    type: str
    data: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    timestamp: Optional[datetime] = None
    source: Optional[str] = None
    target: Optional[str] = None
    correlation_id: Optional[str] = None
    delivery_mode: EventDeliveryMode = EventDeliveryMode.ASYNCHRONOUS
    persistence: EventPersistence = EventPersistence.NONE
    ttl: Optional[int] = None  # 生存时间(秒)
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class IEventHandler(ABC):
    """
    事件处理器接口
    """

    @abstractmethod
    def handle_event(self, event: Event) -> None:
        """
        处理事件

        Args:
            event: 事件对象
        """

    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """
        判断是否可以处理指定事件

        Args:
            event: 事件对象

        Returns:
            是否可以处理
        """

    @abstractmethod
    def get_supported_event_types(self) -> List[str]:
        """
        获取支持的事件类型列表

        Returns:
            事件类型列表
        """


class IEventPublisher(ABC):
    """
    事件发布器接口
    """

    @abstractmethod
    def publish(self, event: Event) -> bool:
        """
        发布事件

        Args:
            event: 事件对象

        Returns:
            是否发布成功
        """

    @abstractmethod
    def publish_sync(self, event: Union[Event, Dict[str, Any]]) -> Any:
        """
        同步发布事件

        Args:
            event: 事件对象或事件数据字典

        Returns:
            发布结果
        """

    @abstractmethod
    def publish_async(self, event: Union[Event, Dict[str, Any]]) -> str:
        """
        异步发布事件

        Args:
            event: 事件对象或事件数据字典

        Returns:
            事件ID
        """

    @abstractmethod
    def publish_batch(self, events: List[Union[Event, Dict[str, Any]]]) -> Dict[str, bool]:
        """
        批量发布事件

        Args:
            events: 事件对象或事件数据字典列表

        Returns:
            发布结果字典 {event_id: success}
        """


class IEventSubscriber(ABC):
    """
    事件订阅器接口
    """

    @abstractmethod
    def subscribe(self, event_type: str, handler: Union[IEventHandler, Callable]) -> bool:
        """
        订阅事件

        Args:
            event_type: 事件类型
            handler: 事件处理器或回调函数

        Returns:
            是否订阅成功
        """

    @abstractmethod
    def unsubscribe(self, event_type: str, handler: Union[IEventHandler, Callable]) -> bool:
        """
        取消订阅事件

        Args:
            event_type: 事件类型
            handler: 事件处理器或回调函数

        Returns:
            是否取消订阅成功
        """

    @abstractmethod
    def get_subscriptions(self) -> Dict[str, List[Union[IEventHandler, Callable]]]:
        """
        获取所有订阅

        Returns:
            订阅字典 {event_type: [handlers]}
        """


class IEventBus(ABC):
    """
    事件总线统一接口

    所有事件总线实现必须遵循此接口，确保API的一致性。
    """

    @abstractmethod
    def publish(self, event: Union[Event, Dict[str, Any]]) -> bool:
        """
        发布事件

        Args:
            event: 事件对象或事件数据字典

        Returns:
            是否发布成功
        """

    @abstractmethod
    def publish_sync(self, event: Union[Event, Dict[str, Any]]) -> Any:
        """
        同步发布事件

        Args:
            event: 事件对象或事件数据字典

        Returns:
            发布结果
        """

    @abstractmethod
    def publish_async(self, event: Union[Event, Dict[str, Any]]) -> str:
        """
        异步发布事件

        Args:
            event: 事件对象或事件数据字典

        Returns:
            事件ID
        """

    @abstractmethod
    def subscribe(self, event_type: str, handler: Union[IEventHandler, Callable]) -> bool:
        """
        订阅事件

        Args:
            event_type: 事件类型
            handler: 事件处理器或回调函数

        Returns:
            是否订阅成功
        """

    @abstractmethod
    def unsubscribe(self, event_type: str, handler: Union[IEventHandler, Callable]) -> bool:
        """
        取消订阅事件

        Args:
            event_type: 事件类型
            handler: 事件处理器或回调函数

        Returns:
            是否取消订阅成功
        """

    @abstractmethod
    def register_handler(self, handler: IEventHandler) -> bool:
        """
        注册事件处理器

        Args:
            handler: 事件处理器实例

        Returns:
            是否注册成功
        """

    @abstractmethod
    def unregister_handler(self, handler: IEventHandler) -> bool:
        """
        注销事件处理器

        Args:
            handler: 事件处理器实例

        Returns:
            是否注销成功
        """

    @abstractmethod
    def get_handlers(self, event_type: str) -> List[Union[IEventHandler, Callable]]:
        """
        获取指定事件类型的处理器列表

        Args:
            event_type: 事件类型

        Returns:
            处理器列表
        """

    @abstractmethod
    def get_all_handlers(self) -> Dict[str, List[Union[IEventHandler, Callable]]]:
        """
        获取所有事件处理器

        Returns:
            处理器字典 {event_type: [handlers]}
        """

    @abstractmethod
    def set_delivery_mode(self, event_type: str, mode: EventDeliveryMode) -> None:
        """
        设置事件传递模式

        Args:
            event_type: 事件类型
            mode: 传递模式
        """

    @abstractmethod
    def get_delivery_mode(self, event_type: str) -> EventDeliveryMode:
        """
        获取事件传递模式

        Args:
            event_type: 事件类型

        Returns:
            传递模式
        """

    @abstractmethod
    def enable_persistence(self, event_type: str, persistence: EventPersistence) -> None:
        """
        启用事件持久化

        Args:
            event_type: 事件类型
            persistence: 持久化策略
        """

    @abstractmethod
    def disable_persistence(self, event_type: str) -> None:
        """
        禁用事件持久化

        Args:
            event_type: 事件类型
        """

    @abstractmethod
    def get_event_history(self, event_type: str = None, limit: int = 100) -> List[Event]:
        """
        获取事件历史

        Args:
            event_type: 事件类型（可选，None表示所有类型）
            limit: 限制返回数量

        Returns:
            事件历史列表
        """

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        获取事件总线统计信息

        Returns:
            统计信息字典
        """

    @abstractmethod
    def clear_event_history(self, event_type: str = None) -> int:
        """
        清空事件历史

        Args:
            event_type: 事件类型（可选，None表示所有类型）

        Returns:
            清除的事件数量
        """

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        执行健康检查

        Returns:
            健康检查结果字典
        """

    @abstractmethod
    def start(self) -> bool:
        """
        启动事件总线

        Returns:
            是否启动成功
        """

    @abstractmethod
    def stop(self) -> bool:
        """
        停止事件总线

        Returns:
            是否停止成功
        """

    @abstractmethod
    def is_running(self) -> bool:
        """
        检查事件总线是否正在运行

        Returns:
            是否正在运行
        """


class IEventFilter(ABC):
    """
    事件过滤器接口
    """

    @abstractmethod
    def filter(self, event: Event) -> bool:
        """
        过滤事件

        Args:
            event: 事件对象

        Returns:
            是否通过过滤
        """

    @abstractmethod
    def get_filter_criteria(self) -> Dict[str, Any]:
        """
        获取过滤条件

        Returns:
            过滤条件字典
        """


class IEventTransformer(ABC):
    """
    事件转换器接口
    """

    @abstractmethod
    def transform(self, event: Event) -> Event:
        """
        转换事件

        Args:
            event: 原始事件对象

        Returns:
            转换后的事件对象
        """

    @abstractmethod
    def can_transform(self, event: Event) -> bool:
        """
        判断是否可以转换指定事件

        Args:
            event: 事件对象

        Returns:
            是否可以转换
        """


class IEventRouter(ABC):
    """
    事件路由器接口
    """

    @abstractmethod
    def route(self, event: Event) -> List[str]:
        """
        路由事件到目标处理器

        Args:
            event: 事件对象

        Returns:
            目标处理器ID列表
        """

    @abstractmethod
    def add_route(self, event_type: str, target_ids: List[str]) -> bool:
        """
        添加路由规则

        Args:
            event_type: 事件类型
            target_ids: 目标处理器ID列表

        Returns:
            是否添加成功
        """

    @abstractmethod
    def remove_route(self, event_type: str) -> bool:
        """
        移除路由规则

        Args:
            event_type: 事件类型

        Returns:
            是否移除成功
        """


class IEventMonitor(ABC):
    """
    事件监控器接口
    """

    @abstractmethod
    def on_event_published(self, event: Event) -> None:
        """
        事件发布回调

        Args:
            event: 发布的事件
        """

    @abstractmethod
    def on_event_processed(self, event: Event, handler_id: str, processing_time: float) -> None:
        """
        事件处理完成回调

        Args:
            event: 处理的事件
            handler_id: 处理器ID
            processing_time: 处理时间
        """

    @abstractmethod
    def on_event_failed(self, event: Event, handler_id: str, error: Exception) -> None:
        """
        事件处理失败回调

        Args:
            event: 失败的事件
            handler_id: 处理器ID
            error: 错误信息
        """

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标

        Returns:
            性能指标字典
        """

    @abstractmethod
    def get_event_flow_stats(self) -> Dict[str, Any]:
        """
        获取事件流统计

        Returns:
            事件流统计字典
        """
