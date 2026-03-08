"""
事件总线代理模块

提供事件总线的降级机制和可靠性保障，确保在事件总线不可用时
事件不会丢失，并在恢复后自动重发。

Author: RQA2025 Development Team
Date: 2026-02-13
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import json

logger = logging.getLogger(__name__)


class EventBusStatus(Enum):
    """事件总线状态"""
    HEALTHY = "healthy"           # 健康
    DEGRADED = "degraded"         # 降级运行
    UNAVAILABLE = "unavailable"   # 不可用
    RECOVERING = "recovering"     # 恢复中


@dataclass
class EventBusEvent:
    """事件总线事件"""
    event_type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    priority: int = 0  # 0=normal, 1=high, 2=critical
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp,
            'source': self.source,
            'priority': self.priority,
            'metadata': self.metadata,
            'retry_count': self.retry_count
        }


@dataclass
class EventBusStats:
    """事件总线统计"""
    total_events: int = 0
    successful_events: int = 0
    failed_events: int = 0
    queued_events: int = 0
    dropped_events: int = 0
    avg_latency_ms: float = 0.0
    last_event_time: Optional[float] = None
    status: EventBusStatus = EventBusStatus.UNAVAILABLE


class EventBusProxy:
    """
    事件总线代理
    
    提供以下功能：
    1. 事件总线连接的自动管理
    2. 降级模式：事件总线不可用时缓存事件
    3. 自动恢复：事件总线恢复后重发缓存事件
    4. 事件优先级管理
    5. 事件持久化（可选）
    
    Attributes:
        max_queue_size: 最大队列大小
        recovery_interval: 恢复检查间隔（秒）
        max_retry_count: 最大重试次数
    """
    
    def __init__(
        self,
        max_queue_size: int = 10000,
        recovery_interval: float = 5.0,
        max_retry_count: int = 3,
        enable_persistence: bool = False,
        persistence_path: Optional[str] = None
    ):
        self.max_queue_size = max_queue_size
        self.recovery_interval = recovery_interval
        self.max_retry_count = max_retry_count
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path
        
        # 事件总线实例
        self._event_bus: Optional[Any] = None
        self._event_bus_status = EventBusStatus.UNAVAILABLE
        
        # 事件队列（按优先级排序）
        self._high_priority_queue: List[EventBusEvent] = []
        self._normal_priority_queue: List[EventBusEvent] = []
        self._low_priority_queue: List[EventBusEvent] = []
        
        # 锁
        self._lock = threading.RLock()
        
        # 运行状态
        self._running = False
        self._recovery_task: Optional[asyncio.Task] = None
        
        # 统计
        self._stats = EventBusStats()
        
        # 回调函数
        self._status_callbacks: List[Callable[[EventBusStatus], None]] = []
        
        logger.info("EventBusProxy initialized")
    
    async def initialize(self) -> bool:
        """
        初始化事件总线代理
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 尝试连接事件总线
            await self._connect_event_bus()
            
            # 启动恢复任务
            self._running = True
            self._recovery_task = asyncio.create_task(self._recovery_loop())
            
            # 如果有持久化，加载缓存的事件
            if self.enable_persistence:
                await self._load_persisted_events()
            
            logger.info("EventBusProxy initialized successfully")
            return True
            
        except Exception as e:
            logger.warning(f"EventBusProxy initialization with fallback mode: {e}")
            self._event_bus_status = EventBusStatus.DEGRADED
            self._running = True
            self._recovery_task = asyncio.create_task(self._recovery_loop())
            return True
    
    async def shutdown(self) -> bool:
        """
        关闭事件总线代理
        
        Returns:
            bool: 关闭是否成功
        """
        try:
            self._running = False
            
            # 停止恢复任务
            if self._recovery_task:
                self._recovery_task.cancel()
                try:
                    await self._recovery_task
                except asyncio.CancelledError:
                    pass
            
            # 持久化缓存的事件
            if self.enable_persistence:
                await self._persist_events()
            
            # 断开事件总线
            self._event_bus = None
            self._event_bus_status = EventBusStatus.UNAVAILABLE
            
            logger.info("EventBusProxy shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"EventBusProxy shutdown failed: {e}")
            return False
    
    async def _connect_event_bus(self) -> bool:
        """
        连接事件总线
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 尝试导入事件总线
            from . import get_event_bus
            self._event_bus = get_event_bus()
            
            if self._event_bus:
                self._event_bus_status = EventBusStatus.HEALTHY
                logger.info("Event bus connected successfully")
                
                # 处理积压事件
                await self._process_fallback_queues()
                return True
            else:
                self._event_bus_status = EventBusStatus.UNAVAILABLE
                logger.warning("Event bus is None")
                return False
                
        except ImportError as e:
            self._event_bus_status = EventBusStatus.UNAVAILABLE
            logger.warning(f"Event bus import failed: {e}")
            return False
        except Exception as e:
            self._event_bus_status = EventBusStatus.UNAVAILABLE
            logger.error(f"Event bus connection failed: {e}")
            return False
    
    def publish_sync(self, event_data: Dict[str, Any]) -> bool:
        """
        同步发布事件
        
        Args:
            event_data: 事件数据，应包含event_type字段
            
        Returns:
            bool: 发布是否成功
        """
        try:
            # 创建事件对象
            event = EventBusEvent(
                event_type=event_data.get('event_type', 'unknown'),
                data=event_data.get('data', {}),
                source=event_data.get('source', ''),
                priority=event_data.get('priority', 0),
                metadata=event_data.get('metadata', {}),
                max_retries=self.max_retry_count
            )
            
            return self._publish_event(event)
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    async def publish_async(self, event_data: Dict[str, Any]) -> bool:
        """
        异步发布事件
        
        Args:
            event_data: 事件数据
            
        Returns:
            bool: 发布是否成功
        """
        return self.publish_sync(event_data)
    
    def _publish_event(self, event: EventBusEvent) -> bool:
        """
        发布事件（内部方法）
        
        Args:
            event: 事件对象
            
        Returns:
            bool: 发布是否成功
        """
        with self._lock:
            self._stats.total_events += 1
            self._stats.last_event_time = time.time()
            
            # 如果事件总线健康，直接发送
            if self._event_bus_status == EventBusStatus.HEALTHY and self._event_bus:
                try:
                    self._event_bus.publish_sync(event.to_dict())
                    self._stats.successful_events += 1
                    return True
                except Exception as e:
                    logger.warning(f"Event bus publish failed, queueing event: {e}")
                    self._event_bus_status = EventBusStatus.DEGRADED
            
            # 降级模式：缓存事件
            return self._queue_event(event)
    
    def _queue_event(self, event: EventBusEvent) -> bool:
        """
        将事件加入队列
        
        Args:
            event: 事件对象
            
        Returns:
            bool: 是否成功加入队列
        """
        with self._lock:
            # 根据优先级选择队列
            if event.priority >= 2:  # Critical
                queue = self._high_priority_queue
            elif event.priority == 1:  # High
                queue = self._normal_priority_queue
            else:  # Normal
                queue = self._low_priority_queue
            
            # 检查队列大小
            total_size = (
                len(self._high_priority_queue) +
                len(self._normal_priority_queue) +
                len(self._low_priority_queue)
            )
            
            if total_size >= self.max_queue_size:
                # 队列已满，丢弃低优先级事件
                if event.priority == 0 and self._low_priority_queue:
                    self._low_priority_queue.pop(0)
                    self._stats.dropped_events += 1
                    logger.warning("Dropped low priority event due to queue full")
                else:
                    logger.error("Event queue full, cannot queue event")
                    self._stats.failed_events += 1
                    return False
            
            # 加入队列
            queue.append(event)
            self._stats.queued_events = total_size + 1
            
            logger.debug(f"Event queued: {event.event_type}, priority: {event.priority}")
            return True
    
    async def _recovery_loop(self):
        """恢复循环"""
        while self._running:
            try:
                # 如果事件总线不可用，尝试重连
                if self._event_bus_status in [EventBusStatus.UNAVAILABLE, EventBusStatus.DEGRADED]:
                    logger.debug("Attempting to recover event bus connection")
                    success = await self._connect_event_bus()
                    
                    if success:
                        logger.info("Event bus recovered successfully")
                        self._notify_status_change(EventBusStatus.HEALTHY)
                    else:
                        # 指数退避
                        await asyncio.sleep(self.recovery_interval)
                
                # 定期处理队列中的事件
                elif self._event_bus_status == EventBusStatus.HEALTHY:
                    await self._process_fallback_queues()
                
                await asyncio.sleep(1)  # 每秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Recovery loop error: {e}")
                await asyncio.sleep(self.recovery_interval)
    
    async def _process_fallback_queues(self) -> int:
        """
        处理降级队列中的事件
        
        Returns:
            int: 成功处理的事件数量
        """
        processed_count = 0
        
        with self._lock:
            if not self._event_bus:
                return 0
            
            # 按优先级处理：高 -> 中 -> 低
            queues = [
                (self._high_priority_queue, "high"),
                (self._normal_priority_queue, "normal"),
                (self._low_priority_queue, "low")
            ]
            
            for queue, priority_name in queues:
                events_to_remove = []
                
                for event in queue:
                    try:
                        self._event_bus.publish_sync(event.to_dict())
                        events_to_remove.append(event)
                        processed_count += 1
                        self._stats.successful_events += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to publish queued event: {e}")
                        event.retry_count += 1
                        
                        # 超过最大重试次数，丢弃事件
                        if event.retry_count >= event.max_retries:
                            events_to_remove.append(event)
                            self._stats.dropped_events += 1
                            logger.error(f"Event dropped after {event.max_retries} retries")
                
                # 移除已处理的事件
                for event in events_to_remove:
                    queue.remove(event)
        
        if processed_count > 0:
            logger.info(f"Processed {processed_count} queued events")
        
        return processed_count
    
    async def _load_persisted_events(self) -> bool:
        """
        加载持久化的事件
        
        Returns:
            bool: 加载是否成功
        """
        if not self.persistence_path:
            return False
        
        try:
            import os
            if not os.path.exists(self.persistence_path):
                return True
            
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)
            
            # 恢复事件到队列
            for event_data in data.get('events', []):
                event = EventBusEvent(**event_data)
                self._queue_event(event)
            
            logger.info(f"Loaded {len(data.get('events', []))} persisted events")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load persisted events: {e}")
            return False
    
    async def _persist_events(self) -> bool:
        """
        持久化事件
        
        Returns:
            bool: 持久化是否成功
        """
        if not self.persistence_path:
            return False
        
        try:
            import os
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            
            with self._lock:
                all_events = (
                    self._high_priority_queue +
                    self._normal_priority_queue +
                    self._low_priority_queue
                )
                
                data = {
                    'timestamp': time.time(),
                    'events': [event.to_dict() for event in all_events]
                }
            
            with open(self.persistence_path, 'w') as f:
                json.dump(data, f)
            
            logger.info(f"Persisted {len(all_events)} events")
            return True
            
        except Exception as e:
            logger.error(f"Failed to persist events: {e}")
            return False
    
    def get_stats(self) -> EventBusStats:
        """
        获取统计信息
        
        Returns:
            EventBusStats: 统计信息
        """
        with self._lock:
            self._stats.queued_events = (
                len(self._high_priority_queue) +
                len(self._normal_priority_queue) +
                len(self._low_priority_queue)
            )
            return self._stats
    
    def get_status(self) -> EventBusStatus:
        """
        获取事件总线状态
        
        Returns:
            EventBusStatus: 当前状态
        """
        return self._event_bus_status
    
    def register_status_callback(self, callback: Callable[[EventBusStatus], None]):
        """
        注册状态变更回调
        
        Args:
            callback: 回调函数
        """
        self._status_callbacks.append(callback)
    
    def _notify_status_change(self, new_status: EventBusStatus):
        """
        通知状态变更
        
        Args:
            new_status: 新状态
        """
        self._event_bus_status = new_status
        for callback in self._status_callbacks:
            try:
                callback(new_status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")


# 全局事件总线代理实例
_event_bus_proxy: Optional[EventBusProxy] = None


async def get_event_bus_proxy() -> EventBusProxy:
    """
    获取全局事件总线代理实例
    
    Returns:
        EventBusProxy: 事件总线代理实例
    """
    global _event_bus_proxy
    
    if _event_bus_proxy is None:
        _event_bus_proxy = EventBusProxy()
        await _event_bus_proxy.initialize()
    
    return _event_bus_proxy


def reset_event_bus_proxy():
    """重置事件总线代理实例"""
    global _event_bus_proxy
    _event_bus_proxy = None
    logger.info("Event bus proxy reset")


# 便捷函数
async def publish_event(event_type: str, data: Dict[str, Any], **kwargs) -> bool:
    """
    发布事件的便捷函数
    
    Args:
        event_type: 事件类型
        data: 事件数据
        **kwargs: 其他参数
        
    Returns:
        bool: 发布是否成功
    """
    proxy = await get_event_bus_proxy()
    return proxy.publish_sync({
        'event_type': event_type,
        'data': data,
        **kwargs
    })


__all__ = [
    'EventBusProxy',
    'EventBusEvent',
    'EventBusStats',
    'EventBusStatus',
    'get_event_bus_proxy',
    'reset_event_bus_proxy',
    'publish_event'
]