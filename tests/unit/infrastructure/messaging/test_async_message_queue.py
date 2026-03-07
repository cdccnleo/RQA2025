#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层messaging/async_message_queue.py模块测试

测试目标：提升async_message_queue.py的真实覆盖率
实际导入和使用src.infrastructure.messaging.async_message_queue模块
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.infrastructure.messaging.async_message_queue import (
    MessagePriority,
    MessageStatus,
    Message,
    MessageHandler,
    AsyncMessageQueue
)


class TestMessagePriority:
    """测试MessagePriority枚举"""
    
    def test_message_priority_values(self):
        """测试消息优先级值"""
        assert MessagePriority.LOW.value == 0
        assert MessagePriority.NORMAL.value == 1
        assert MessagePriority.HIGH.value == 2
        assert MessagePriority.CRITICAL.value == 3


class TestMessageStatus:
    """测试MessageStatus枚举"""
    
    def test_message_status_values(self):
        """测试消息状态值"""
        assert MessageStatus.PENDING.value == "pending"
        assert MessageStatus.PROCESSING.value == "processing"
        assert MessageStatus.COMPLETED.value == "completed"
        assert MessageStatus.FAILED.value == "failed"
        assert MessageStatus.DEAD_LETTER.value == "dead_letter"


class TestMessage:
    """测试Message类"""
    
    def test_message_creation(self):
        """测试消息创建"""
        message = Message(
            topic="test_topic",
            payload={"key": "value"},
            priority=MessagePriority.HIGH
        )
        
        assert message.topic == "test_topic"
        assert message.payload == {"key": "value"}
        assert message.priority == MessagePriority.HIGH
        assert message.status == MessageStatus.PENDING
        assert message.id is not None
        assert message.timestamp > 0
    
    def test_message_is_expired(self):
        """测试消息过期检查"""
        # 未设置TTL的消息不过期
        message = Message(topic="test", ttl=None)
        assert not message.is_expired
        
        # 设置TTL的消息
        import time
        message = Message(topic="test", ttl=1.0, timestamp=time.time() - 2.0)
        assert message.is_expired
    
    def test_message_to_dict(self):
        """测试消息转换为字典"""
        message = Message(
            topic="test_topic",
            payload={"key": "value"},
            priority=MessagePriority.NORMAL
        )
        
        data = message.to_dict()
        assert data['topic'] == "test_topic"
        assert data['payload'] == {"key": "value"}
        assert data['priority'] == MessagePriority.NORMAL.value
        assert data['status'] == MessageStatus.PENDING.value
    
    def test_message_from_dict(self):
        """测试从字典创建消息"""
        data = {
            'id': 'test_id',
            'topic': 'test_topic',
            'payload': {'key': 'value'},
            'priority': 1,
            'status': 'pending',
            'timestamp': 1234567890.0,
            'headers': {},
            'retry_count': 0,
            'max_retries': 3
        }
        
        message = Message.from_dict(data)
        assert message.id == 'test_id'
        assert message.topic == 'test_topic'
        assert message.payload == {'key': 'value'}
        assert message.priority == MessagePriority.NORMAL


class TestMessageHandler:
    """测试MessageHandler类"""
    
    @pytest.mark.asyncio
    async def test_message_handler_creation(self):
        """测试消息处理器创建"""
        async def handler_func(msg):
            pass
        
        handler = MessageHandler(handler_func)
        assert handler.handler_func == handler_func
        assert handler.is_active is True
    
    @pytest.mark.asyncio
    async def test_message_handler_handle(self):
        """测试消息处理器处理消息"""
        processed_messages = []
        
        async def handler_func(msg):
            processed_messages.append(msg)
        
        handler = MessageHandler(handler_func)
        message = Message(topic="test", payload="data")
        
        result = await handler.handle(message)
        
        assert result is True
        assert len(processed_messages) == 1
        assert processed_messages[0] == message
    
    @pytest.mark.asyncio
    async def test_message_handler_handle_exception(self):
        """测试消息处理器处理异常"""
        async def handler_func(msg):
            raise Exception("Test error")
        
        handler = MessageHandler(handler_func)
        message = Message(topic="test", payload="data")
        
        result = await handler.handle(message)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_message_handler_with_filter(self):
        """测试带过滤器的消息处理器"""
        processed_messages = []
        
        async def handler_func(msg):
            processed_messages.append(msg)
        
        def filter_func(msg):
            return msg.topic == "allowed"
        
        handler = MessageHandler(handler_func, filter_func)
        
        # 允许的消息
        message1 = Message(topic="allowed", payload="data")
        result1 = await handler.handle(message1)
        assert result1 is True
        
        # 不允许的消息
        message2 = Message(topic="blocked", payload="data")
        result2 = await handler.handle(message2)
        assert result2 is False
        assert len(processed_messages) == 1


class TestAsyncMessageQueue:
    """测试AsyncMessageQueue类"""
    
    @pytest.mark.asyncio
    async def test_queue_creation(self):
        """测试队列创建"""
        queue = AsyncMessageQueue(max_queue_size=100)
        
        assert queue.max_queue_size == 100
        assert queue.worker_count == 4
        assert queue.persistence_enabled is False
        assert queue.dead_letter_enabled is True
    
    @pytest.mark.asyncio
    async def test_queue_start_stop(self):
        """测试队列启动和停止"""
        queue = AsyncMessageQueue(worker_count=1)
        
        await queue.start()
        assert len(queue._workers) == 1
        
        await queue.stop()
        assert len(queue._workers) == 0
    
    @pytest.mark.asyncio
    async def test_publish_message(self):
        """测试发布消息"""
        queue = AsyncMessageQueue(worker_count=1)
        await queue.start()
        
        try:
            message = Message(topic="test", payload="data")
            result = await queue.publish(message)
            
            assert result is True
            assert queue._stats['messages_published'] > 0
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_publish_to_topic(self):
        """测试发布消息到主题"""
        queue = AsyncMessageQueue(worker_count=1)
        await queue.start()
        
        try:
            result = await queue.publish_to_topic(
                topic="test_topic",
                payload="data",
                priority=MessagePriority.HIGH
            )
            
            assert result is True
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_subscribe(self):
        """测试订阅主题"""
        queue = AsyncMessageQueue(worker_count=1)
        await queue.start()
        
        try:
            async def handler(msg):
                pass
            
            subscription_id = await queue.subscribe("test_topic", handler)
            
            assert subscription_id is not None
            assert "test_topic" in subscription_id
            assert queue._stats['active_subscribers'] > 0
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """测试取消订阅"""
        queue = AsyncMessageQueue(worker_count=1)
        await queue.start()
        
        try:
            async def handler(msg):
                pass
            
            subscription_id = await queue.subscribe("test_topic", handler)
            result = await queue.unsubscribe("test_topic", subscription_id)
            
            assert result is True
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """测试获取队列统计信息"""
        queue = AsyncMessageQueue()
        await queue.start()
        
        try:
            stats = queue.get_stats()
            
            assert isinstance(stats, dict)
            assert 'messages_published' in stats
            assert 'messages_processed' in stats
            assert 'messages_failed' in stats
            assert 'messages_dead_lettered' in stats
            assert 'active_subscribers' in stats
        finally:
            await queue.stop()

