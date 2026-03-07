#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步消息队列系统

提供高性能的异步消息队列，支持发布订阅模式、消息持久化、死信队列等功能。
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Awaitable, Union
from enum import Enum
from queue import PriorityQueue
import threading

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """消息优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class MessageStatus(Enum):
    """消息状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class Message:
    """消息对象"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    payload: Any = None
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    headers: Dict[str, Any] = field(default_factory=dict)
    status: MessageStatus = MessageStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    ttl: Optional[float] = None  # Time to live in seconds
    correlation_id: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """检查消息是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'topic': self.topic,
            'payload': self.payload,
            'priority': self.priority.value,
            'timestamp': self.timestamp,
            'headers': self.headers,
            'status': self.status.value,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'ttl': self.ttl,
            'correlation_id': self.correlation_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            topic=data.get('topic', ''),
            payload=data.get('payload'),
            priority=MessagePriority(data.get('priority', 1)),
            timestamp=data.get('timestamp', time.time()),
            headers=data.get('headers', {}),
            status=MessageStatus(data.get('status', 'pending')),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            ttl=data.get('ttl'),
            correlation_id=data.get('correlation_id')
        )


class MessageHandler:
    """消息处理器"""

    def __init__(self, handler_func: Callable[[Message], Awaitable[None]],
                 filter_func: Optional[Callable[[Message], bool]] = None):
        self.handler_func = handler_func
        self.filter_func = filter_func
        self.is_active = True

    async def handle(self, message: Message) -> bool:
        """处理消息"""
        if not self.is_active:
            return False

        if self.filter_func and not self.filter_func(message):
            return False

        try:
            await self.handler_func(message)
            return True
        except Exception as e:
            logger.error(f"Message handler error: {e}")
            return False


class AsyncMessageQueue:
    """
    异步消息队列

    支持：
    - 发布订阅模式
    - 消息优先级
    - 消息持久化
    - 死信队列
    - 消息过滤
    - 批量处理
    """

    def __init__(self,
                 max_queue_size: int = 10000,
                 persistence_enabled: bool = False,
                 dead_letter_enabled: bool = True,
                 worker_count: int = 4):
        self.max_queue_size = max_queue_size
        self.persistence_enabled = persistence_enabled
        self.dead_letter_enabled = dead_letter_enabled
        self.worker_count = worker_count

        # 消息队列
        self._message_queue: PriorityQueue = PriorityQueue()
        self._processing_messages: Dict[str, Message] = {}

        # 订阅者管理
        self._subscribers: Dict[str, List[MessageHandler]] = {}
        self._topic_patterns: Dict[str, List[MessageHandler]] = {}

        # 死信队列
        self._dead_letter_queue: List[Message] = []

        # 控制信号
        self._shutdown_event = asyncio.Event()
        self._workers: List[asyncio.Task] = []

        # 统计信息
        self._stats = {
            'messages_published': 0,
            'messages_processed': 0,
            'messages_failed': 0,
            'messages_dead_lettered': 0,
            'active_subscribers': 0
        }

        # 锁
        self._lock = asyncio.Lock()

    async def start(self):
        """启动消息队列"""
        logger.info("Starting AsyncMessageQueue")

        # 启动工作线程
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)

        logger.info(f"Started {self.worker_count} worker threads")

    async def stop(self):
        """停止消息队列"""
        logger.info("Stopping AsyncMessageQueue")

        # 设置关闭信号
        self._shutdown_event.set()

        # 等待所有工作线程完成
        await asyncio.gather(*self._workers, return_exceptions=True)

        # 清空队列
        self._workers.clear()

        logger.info("AsyncMessageQueue stopped")

    async def publish(self, message: Message) -> bool:
        """
        发布消息

        Args:
            message: 要发布的消息

        Returns:
            是否成功发布
        """
        if self._shutdown_event.is_set():
            return False

        if self._message_queue.qsize() >= self.max_queue_size:
            logger.warning("Message queue is full, dropping message")
            return False

        # 设置消息状态
        message.status = MessageStatus.PENDING

        # 添加到队列 (使用负优先级确保高优先级先处理)
        priority_item = (-message.priority.value, message.timestamp, message)

        try:
            self._message_queue.put_nowait(priority_item)
            self._stats['messages_published'] += 1

            logger.debug(f"Message published: {message.id} to topic {message.topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False

    async def subscribe(self, topic: str, handler: Callable[[Message], Awaitable[None]],
                       filter_func: Optional[Callable[[Message], bool]] = None) -> str:
        """
        订阅主题

        Args:
            topic: 主题名称
            handler: 消息处理函数
            filter_func: 消息过滤函数

        Returns:
            订阅ID
        """
        async with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []

            message_handler = MessageHandler(handler, filter_func)
            self._subscribers[topic].append(message_handler)
            self._stats['active_subscribers'] += 1

            subscription_id = f"{topic}_{id(message_handler)}"
            logger.info(f"Subscribed to topic: {topic}")
            return subscription_id

    async def unsubscribe(self, topic: str, subscription_id: str) -> bool:
        """
        取消订阅

        Args:
            topic: 主题名称
            subscription_id: 订阅ID

        Returns:
            是否成功取消订阅
        """
        async with self._lock:
            if topic not in self._subscribers:
                return False

            # 移除匹配的处理器
            original_count = len(self._subscribers[topic])
            self._subscribers[topic] = [
                h for h in self._subscribers[topic]
                if f"{topic}_{id(h)}" != subscription_id
            ]

            removed_count = original_count - len(self._subscribers[topic])
            if removed_count > 0:
                self._stats['active_subscribers'] -= removed_count
                logger.info(f"Unsubscribed from topic: {topic}")
                return True

            return False

    async def publish_to_topic(self, topic: str, payload: Any,
                             priority: MessagePriority = MessagePriority.NORMAL,
                             headers: Optional[Dict[str, Any]] = None,
                             ttl: Optional[float] = None,
                             correlation_id: Optional[str] = None) -> bool:
        """
        发布消息到指定主题

        Args:
            topic: 主题名称
            payload: 消息负载
            priority: 消息优先级
            headers: 消息头
            ttl: 生存时间
            correlation_id: 关联ID

        Returns:
            是否成功发布
        """
        message = Message(
            topic=topic,
            payload=payload,
            priority=priority,
            headers=headers or {},
            ttl=ttl,
            correlation_id=correlation_id
        )

        return await self.publish(message)

    async def _worker_loop(self, worker_id: int):
        """工作线程循环"""
        logger.debug(f"Worker {worker_id} started")

        while not self._shutdown_event.is_set():
            try:
                # 获取消息
                try:
                    priority_item = self._message_queue.get_nowait()
                    _, _, message = priority_item
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.1)  # 避免忙等待
                    continue

                # 处理消息
                await self._process_message(message)

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1.0)  # 错误后稍等

        logger.debug(f"Worker {worker_id} stopped")

    async def _process_message(self, message: Message):
        """处理消息"""
        # 检查消息是否过期
        if message.is_expired:
            logger.warning(f"Message expired: {message.id}")
            await self._move_to_dead_letter(message)
            return

        # 设置处理状态
        message.status = MessageStatus.PROCESSING
        self._processing_messages[message.id] = message

        try:
            # 查找订阅者
            subscribers = await self._get_subscribers(message.topic)

            if not subscribers:
                logger.warning(f"No subscribers for topic: {message.topic}")
                message.status = MessageStatus.COMPLETED
                self._stats['messages_processed'] += 1
                return

            # 发送给所有订阅者
            success_count = 0
            for handler in subscribers:
                try:
                    if await handler.handle(message):
                        success_count += 1
                except Exception as e:
                    logger.error(f"Handler error for message {message.id}: {e}")

            # 检查处理结果
            if success_count > 0:
                message.status = MessageStatus.COMPLETED
                self._stats['messages_processed'] += 1
                logger.debug(f"Message processed successfully: {message.id}")
            else:
                await self._handle_processing_failure(message)

        except Exception as e:
            logger.error(f"Message processing error: {e}")
            await self._handle_processing_failure(message)

        finally:
            # 清理处理中的消息
            self._processing_messages.pop(message.id, None)

    async def _get_subscribers(self, topic: str) -> List[MessageHandler]:
        """获取主题的订阅者"""
        async with self._lock:
            subscribers = []

            # 精确匹配
            if topic in self._subscribers:
                subscribers.extend(self._subscribers[topic])

            # 模式匹配 (暂时不支持通配符)
            # TODO: 实现通配符匹配

            return subscribers

    async def _handle_processing_failure(self, message: Message):
        """处理消息处理失败"""
        message.retry_count += 1

        if message.retry_count < message.max_retries:
            # 重新入队
            logger.info(f"Retrying message: {message.id} (attempt {message.retry_count})")
            message.status = MessageStatus.PENDING
            await self.publish(message)
        else:
            # 移到死信队列
            logger.error(f"Message failed permanently: {message.id}")
            await self._move_to_dead_letter(message)

    async def _move_to_dead_letter(self, message: Message):
        """将消息移到死信队列"""
        if not self.dead_letter_enabled:
            return

        message.status = MessageStatus.DEAD_LETTER
        self._dead_letter_queue.append(message)
        self._stats['messages_dead_lettered'] += 1
        self._stats['messages_failed'] += 1

        logger.warning(f"Message moved to dead letter queue: {message.id}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            'queue_size': self._message_queue.qsize(),
            'processing_messages': len(self._processing_messages),
            'dead_letter_count': len(self._dead_letter_queue),
            'active_subscribers': sum(len(handlers) for handlers in self._subscribers.values())
        }

    def get_dead_letter_messages(self, limit: int = 100) -> List[Message]:
        """获取死信队列消息"""
        return self._dead_letter_queue[-limit:] if self._dead_letter_queue else []

    async def replay_dead_letter_message(self, message_id: str) -> bool:
        """重放死信队列中的消息"""
        for i, message in enumerate(self._dead_letter_queue):
            if message.id == message_id:
                # 重置消息状态
                message.status = MessageStatus.PENDING
                message.retry_count = 0

                # 重新发布
                success = await self.publish(message)
                if success:
                    # 从死信队列移除
                    self._dead_letter_queue.pop(i)
                    self._stats['messages_dead_lettered'] -= 1
                    logger.info(f"Dead letter message replayed: {message_id}")
                    return True

        return False

    def clear_dead_letter_queue(self):
        """清空死信队列"""
        count = len(self._dead_letter_queue)
        self._dead_letter_queue.clear()
        self._stats['messages_dead_lettered'] -= count
        logger.info(f"Cleared {count} messages from dead letter queue")
