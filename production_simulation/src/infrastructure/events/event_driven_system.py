#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件驱动架构系统

提供完整的事件驱动架构，包括事件发布、订阅、处理、存储和监控。
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Awaitable, Union, Set
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """事件优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class EventStatus(Enum):
    """事件状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"


class EventType(Enum):
    """事件类型"""
    BUSINESS = "business"
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class Event:
    """事件对象"""
    event_type: EventType
    event_name: str
    payload: Any = None
    source: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    headers: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    status: EventStatus = EventStatus.PENDING
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl: Optional[float] = None

    @property
    def is_expired(self) -> bool:
        """检查事件是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    @property
    def age_seconds(self) -> float:
        """获取事件年龄（秒）"""
        return time.time() - self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'event_type': self.event_type.value,
            'event_name': self.event_name,
            'payload': self.payload,
            'source': self.source,
            'timestamp': self.timestamp,
            'priority': self.priority.value,
            'headers': self.headers,
            'correlation_id': self.correlation_id,
            'causation_id': self.cusation_id,
            'status': self.status.value,
            'tags': list(self.tags),
            'metadata': self.metadata,
            'ttl': self.ttl
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """从字典创建事件"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            event_type=EventType(data.get('event_type', 'custom')),
            event_name=data.get('event_name', ''),
            payload=data.get('payload'),
            source=data.get('source', ''),
            timestamp=data.get('timestamp', time.time()),
            priority=EventPriority(data.get('priority', 1)),
            headers=data.get('headers', {}),
            correlation_id=data.get('correlation_id'),
            causation_id=data.get('causation_id'),
            status=EventStatus(data.get('status', 'pending')),
            tags=set(data.get('tags', [])),
            metadata=data.get('metadata', {}),
            ttl=data.get('ttl')
        )


class EventHandler:
    """事件处理器"""

    def __init__(self,
                 handler_func: Callable[[Event], Awaitable[None]],
                 filter_func: Optional[Callable[[Event], bool]] = None,
                 priority: int = 0):
        self.handler_func = handler_func
        self.filter_func = filter_func
        self.priority = priority  # 处理器优先级
        self.is_active = True
        self.processed_count = 0
        self.failed_count = 0
        self.last_processed_time: Optional[float] = None

    async def handle(self, event: Event) -> bool:
        """处理事件"""
        if not self.is_active:
            return False

        if self.filter_func and not self.filter_func(event):
            return False

        try:
            await self.handler_func(event)
            self.processed_count += 1
            self.last_processed_time = time.time()
            return True
        except Exception as e:
            logger.error(f"Event handler error: {e}")
            self.failed_count += 1
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        return {
            'is_active': self.is_active,
            'priority': self.priority,
            'processed_count': self.processed_count,
            'failed_count': self.failed_count,
            'success_rate': self.processed_count / (self.processed_count + self.failed_count) if (self.processed_count + self.failed_count) > 0 else 0.0,
            'last_processed_time': self.last_processed_time
        }


class EventProcessor:
    """事件处理器"""

    def __init__(self, name: str, handler: EventHandler, concurrency_limit: int = 10):
        self.name = name
        self.handler = handler
        self.concurrency_limit = concurrency_limit
        self.active_tasks: Set[asyncio.Task] = set()
        self.task_lock = asyncio.Lock()

    async def process_event(self, event: Event) -> bool:
        """处理单个事件"""
        async with self.task_lock:
            if len(self.active_tasks) >= self.concurrency_limit:
                logger.warning(f"Processor {self.name} at concurrency limit")
                return False

            # 创建处理任务
            task = asyncio.create_task(self._process_with_cleanup(event))
            self.active_tasks.add(task)

            # 清理已完成的任务
            self.active_tasks = {t for t in self.active_tasks if not t.done()}

            return True

    async def _process_with_cleanup(self, event: Event):
        """处理事件并清理任务"""
        try:
            await self.handler.handle(event)
        finally:
            # 从活跃任务中移除
            self.active_tasks.discard(asyncio.current_task())

    async def wait_for_completion(self, timeout: Optional[float] = None):
        """等待所有处理任务完成"""
        if self.active_tasks:
            await asyncio.wait(self.active_tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED)

    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        return {
            'name': self.name,
            'active_tasks': len(self.active_tasks),
            'concurrency_limit': self.concurrency_limit,
            'handler_stats': self.handler.get_stats()
        }


class EventDrivenSystem:
    """
    事件驱动系统

    提供完整的事件驱动架构：
    - 事件发布和订阅
    - 事件路由和过滤
    - 事件存储和重放
    - 事件监控和统计
    - 事件流处理
    """

    def __init__(self,
                 max_event_queue_size: int = 10000,
                 enable_persistence: bool = False,
                 enable_monitoring: bool = True,
                 default_ttl: float = 3600.0):  # 1小时默认TTL
        self.max_event_queue_size = max_event_queue_size
        self.enable_persistence = enable_persistence
        self.enable_monitoring = enable_monitoring
        self.default_ttl = default_ttl

        # 事件队列和处理
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_event_queue_size)
        self._processing_tasks: Set[asyncio.Task] = set()

        # 订阅管理
        self._event_handlers: Dict[str, List[EventHandler]] = {}
        self._event_processors: Dict[str, EventProcessor] = {}
        self._global_handlers: List[EventHandler] = []

        # 事件存储
        self._event_store: List[Event] = []
        self._max_store_size = 100000

        # 监控统计
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'active_subscriptions': 0,
            'uptime_seconds': 0
        }
        self._start_time = time.time()

        # 控制信号
        self._shutdown_event = asyncio.Event()
        self._worker_tasks: List[asyncio.Task] = []

        # 线程池用于同步操作
        self._thread_pool = ThreadPoolExecutor(max_workers=4)

    async def start(self, worker_count: int = 4):
        """启动事件驱动系统"""
        logger.info(f"Starting EventDrivenSystem with {worker_count} workers")

        # 启动工作线程
        for i in range(worker_count):
            task = asyncio.create_task(self._event_worker(i))
            self._worker_tasks.append(task)

        logger.info("EventDrivenSystem started")

    async def stop(self):
        """停止事件驱动系统"""
        logger.info("Stopping EventDrivenSystem")

        # 设置关闭信号
        self._shutdown_event.set()

        # 等待所有工作线程完成
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        # 等待所有处理器完成
        for processor in self._event_processors.values():
            await processor.wait_for_completion(timeout=5.0)

        # 关闭线程池
        self._thread_pool.shutdown(wait=True)

        self._stats['uptime_seconds'] = time.time() - self._start_time
        logger.info("EventDrivenSystem stopped")

    async def publish_event(self, event: Event) -> bool:
        """
        发布事件

        Args:
            event: 要发布的事件

        Returns:
            是否成功发布
        """
        if self._shutdown_event.is_set():
            return False

        if self._event_queue.qsize() >= self.max_event_queue_size:
            logger.warning("Event queue is full, dropping event")
            return False

        try:
            await self._event_queue.put(event)
            self._stats['events_published'] += 1

            # 存储事件（如果启用持久化）
            if self.enable_persistence:
                await self._store_event(event)

            logger.debug(f"Event published: {event.id} ({event.event_name})")
            return True
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False

    async def publish(self, event_name: str, payload: Any = None,
                     event_type: EventType = EventType.CUSTOM,
                     source: str = "", priority: EventPriority = EventPriority.NORMAL,
                     headers: Optional[Dict[str, Any]] = None,
                     tags: Optional[Set[str]] = None,
                     correlation_id: Optional[str] = None,
                     causation_id: Optional[str] = None,
                     ttl: Optional[float] = None) -> bool:
        """
        便捷的事件发布方法

        Args:
            event_name: 事件名称
            payload: 事件负载
            event_type: 事件类型
            source: 事件源
            priority: 事件优先级
            headers: 事件头
            tags: 事件标签
            correlation_id: 关联ID
            causation_id: 因果ID
            ttl: 生存时间

        Returns:
            是否成功发布
        """
        event = Event(
            event_type=event_type,
            event_name=event_name,
            payload=payload,
            source=source,
            priority=priority,
            headers=headers or {},
            tags=tags or set(),
            correlation_id=correlation_id,
            causation_id=causation_id,
            ttl=ttl or self.default_ttl
        )

        return await self.publish_event(event)

    async def subscribe(self, event_pattern: str,
                       handler: Callable[[Event], Awaitable[None]],
                       filter_func: Optional[Callable[[Event], bool]] = None,
                       priority: int = 0,
                       processor_name: Optional[str] = None) -> str:
        """
        订阅事件

        Args:
            event_pattern: 事件模式（支持通配符）
            handler: 事件处理函数
            filter_func: 事件过滤函数
            priority: 处理器优先级
            processor_name: 处理器名称

        Returns:
            订阅ID
        """
        event_handler = EventHandler(handler, filter_func, priority)

        if processor_name:
            # 创建命名处理器
            if processor_name not in self._event_processors:
                processor = EventProcessor(processor_name, event_handler)
                self._event_processors[processor_name] = processor
            else:
                # 添加到现有处理器
                self._event_processors[processor_name].handler = event_handler
        else:
            # 添加到模式处理器
            if event_pattern not in self._event_handlers:
                self._event_handlers[event_pattern] = []
            self._event_handlers[event_pattern].append(event_handler)

        self._stats['active_subscriptions'] += 1
        subscription_id = f"{event_pattern}_{id(event_handler)}"

        logger.info(f"Subscribed to event pattern: {event_pattern}")
        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        取消订阅

        Args:
            subscription_id: 订阅ID

        Returns:
            是否成功取消订阅
        """
        # 解析订阅ID
        parts = subscription_id.split('_', 1)
        if len(parts) != 2:
            return False

        pattern_or_processor = parts[0]
        handler_id = parts[1]

        # 检查是否是处理器
        if pattern_or_processor in self._event_processors:
            # 移除处理器
            if pattern_or_processor in self._event_processors:
                del self._event_processors[pattern_or_processor]
                self._stats['active_subscriptions'] -= 1
                return True
        else:
            # 检查模式处理器
            if pattern_or_processor in self._event_handlers:
                handlers = self._event_handlers[pattern_or_processor]
                original_count = len(handlers)

                # 移除匹配的处理器
                self._event_handlers[pattern_or_processor] = [
                    h for h in handlers if f"{pattern_or_processor}_{id(h)}" != subscription_id
                ]

                removed_count = original_count - len(self._event_handlers[pattern_or_processor])
                if removed_count > 0:
                    self._stats['active_subscriptions'] -= removed_count
                    return True

        return False

    async def add_global_handler(self, handler: Callable[[Event], Awaitable[None]],
                               filter_func: Optional[Callable[[Event], bool]] = None,
                               priority: int = 0) -> str:
        """
        添加全局事件处理器

        Args:
            handler: 事件处理函数
            filter_func: 事件过滤函数
            priority: 处理器优先级

        Returns:
            处理器ID
        """
        event_handler = EventHandler(handler, filter_func, priority)
        self._global_handlers.append(event_handler)
        self._stats['active_subscriptions'] += 1

        handler_id = f"global_{id(event_handler)}"
        logger.info("Added global event handler")
        return handler_id

    async def _event_worker(self, worker_id: int):
        """事件工作线程"""
        logger.debug(f"Event worker {worker_id} started")

        while not self._shutdown_event.is_set():
            try:
                # 获取事件
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # 处理事件
                await self._process_event(event)

            except Exception as e:
                logger.error(f"Event worker {worker_id} error: {e}")
                await asyncio.sleep(1.0)

        logger.debug(f"Event worker {worker_id} stopped")

    async def _process_event(self, event: Event):
        """处理事件"""
        # 检查事件是否过期
        if event.is_expired:
            logger.warning(f"Event expired: {event.id}")
            event.status = EventStatus.FAILED
            self._stats['events_failed'] += 1
            return

        # 设置处理状态
        event.status = EventStatus.PROCESSING

        try:
            # 查找处理器
            handlers = await self._find_handlers(event)

            if not handlers:
                logger.debug(f"No handlers found for event: {event.event_name}")
                event.status = EventStatus.PROCESSED
                self._stats['events_processed'] += 1
                return

            # 按优先级排序处理器
            handlers.sort(key=lambda h: h.priority, reverse=True)

            # 处理事件
            success_count = 0
            for handler in handlers:
                try:
                    if await handler.handle(event):
                        success_count += 1
                except Exception as e:
                    logger.error(f"Handler error for event {event.id}: {e}")

            # 更新事件状态
            if success_count > 0:
                event.status = EventStatus.PROCESSED
                self._stats['events_processed'] += 1
                logger.debug(f"Event processed successfully: {event.id}")
            else:
                event.status = EventStatus.FAILED
                self._stats['events_failed'] += 1
                logger.warning(f"Event processing failed: {event.id}")

        except Exception as e:
            logger.error(f"Event processing error: {e}")
            event.status = EventStatus.FAILED
            self._stats['events_failed'] += 1

    async def _find_handlers(self, event: Event) -> List[EventHandler]:
        """查找事件处理器"""
        handlers = []

        # 全局处理器
        handlers.extend(self._global_handlers)

        # 模式匹配处理器
        for pattern, pattern_handlers in self._event_handlers.items():
            if self._matches_pattern(event.event_name, pattern):
                handlers.extend(pattern_handlers)

        # 命名处理器
        for processor_name, processor in self._event_processors.items():
            if self._matches_pattern(event.event_name, processor_name):
                handlers.append(processor.handler)

        return handlers

    def _matches_pattern(self, event_name: str, pattern: str) -> bool:
        """检查事件名称是否匹配模式"""
        # 简单实现：支持*通配符
        if pattern == "*":
            return True

        if "*" in pattern:
            # 转换为正则表达式
            import re
            regex_pattern = pattern.replace("*", ".*")
            return bool(re.match(regex_pattern, event_name))

        return event_name == pattern

    async def _store_event(self, event: Event):
        """存储事件"""
        if len(self._event_store) >= self._max_store_size:
            # 移除旧事件
            remove_count = self._max_store_size // 10  # 移除10%
            self._event_store = self._event_store[remove_count:]

        self._event_store.append(event)

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        current_time = time.time()
        uptime = current_time - self._start_time

        stats = self._stats.copy()
        stats.update({
            'uptime_seconds': uptime,
            'queue_size': self._event_queue.qsize(),
            'stored_events': len(self._event_store),
            'active_processors': len(self._event_processors),
            'event_handlers': sum(len(handlers) for handlers in self._event_handlers.values()),
            'global_handlers': len(self._global_handlers),
            'total_subscriptions': stats['active_subscriptions']
        })

        return stats

    def get_processor_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        return {
            processor_name: processor.get_stats()
            for processor_name, processor in self._event_processors.items()
        }

    def get_recent_events(self, limit: int = 100) -> List[Event]:
        """获取最近的事件"""
        return self._event_store[-limit:] if self._event_store else []

    def get_events_by_type(self, event_type: EventType, limit: int = 100) -> List[Event]:
        """按类型获取事件"""
        matching_events = [
            event for event in self._event_store
            if event.event_type == event_type
        ]
        return matching_events[-limit:] if matching_events else []

    async def replay_event(self, event_id: str) -> bool:
        """重放事件"""
        for event in self._event_store:
            if event.id == event_id:
                # 重置事件状态
                event.status = EventStatus.PENDING
                event.timestamp = time.time()

                # 重新发布
                return await self.publish_event(event)

        return False

    def clear_event_store(self):
        """清空事件存储"""
        count = len(self._event_store)
        self._event_store.clear()
        logger.info(f"Cleared {count} events from store")

    def enable_monitoring(self):
        """启用监控"""
        self.enable_monitoring = True

    def disable_monitoring(self):
        """禁用监控"""
        self.enable_monitoring = False
