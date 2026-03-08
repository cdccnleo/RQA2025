"""
业务事件系统模块

提供事件驱动架构支持：
- Event: 事件基类
- EventHandler: 事件处理器接口
- EventPublisher: 事件发布器
- publish_event: 发布事件便捷函数
- subscribe_event: 订阅事件便捷函数
"""

try:
    from .event_system import (
        Event,
        EventHandler,
        EventPublisher,
        publish_event,
        subscribe_event,
        unsubscribe_event,
        enable_async_events,
        disable_async_events,
        get_event_stats
    )
    EVENT_SYSTEM_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"event_system模块导入失败: {e}，跳过相关功能")
    EVENT_SYSTEM_AVAILABLE = False
    Event = None
    EventHandler = None
    EventPublisher = None
    publish_event = None
    subscribe_event = None
    unsubscribe_event = None
    enable_async_events = None
    disable_async_events = None
    get_event_stats = None

__all__ = [
    'Event',
    'EventHandler',
    'EventPublisher',
    'publish_event',
    'subscribe_event',
    'unsubscribe_event',
    'enable_async_events',
    'disable_async_events',
    'get_event_stats'
]
