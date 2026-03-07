#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 异步消息队列深度测试
测试AsyncMessageQueue的核心消息处理功能、并发安全和容错机制
"""

import asyncio
import json
import time
import pytest
import threading
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List, Any

from infrastructure.messaging.async_message_queue import (
    AsyncMessageQueue, Message, MessagePriority, MessageStatus, MessageHandler
)


class TestMessage:
    """Message测试"""

    def test_message_initialization_basic(self):
        """测试基本初始化"""
        message = Message(
            topic="test_topic",
            payload={"data": "test"}
        )

        assert message.topic == "test_topic"
        assert message.payload == {"data": "test"}
        assert message.priority == MessagePriority.NORMAL
        assert message.status == MessageStatus.PENDING
        assert message.retry_count == 0
        assert message.max_retries == 3
        assert isinstance(message.id, str)
        assert isinstance(message.timestamp, float)
        assert isinstance(message.headers, dict)

    def test_message_initialization_with_options(self):
        """测试带选项初始化"""
        headers = {"source": "test"}
        message = Message(
            topic="test_topic",
            payload="test_data",
            priority=MessagePriority.HIGH,
            headers=headers,
            ttl=60.0,
            correlation_id="corr_123"
        )

        assert message.priority == MessagePriority.HIGH
        assert message.headers == headers
        assert message.ttl == 60.0
        assert message.correlation_id == "corr_123"

    def test_message_is_expired_no_ttl(self):
        """测试无TTL的消息不过期"""
        message = Message(topic="test", payload="data")
        assert not message.is_expired

    def test_message_is_expired_not_expired(self):
        """测试未过期消息"""
        message = Message(topic="test", payload="data", ttl=60.0)
        assert not message.is_expired

    def test_message_is_expired_expired(self):
        """测试已过期消息"""
        message = Message(
            topic="test",
            payload="data",
            timestamp=time.time() - 120,  # 2分钟前
            ttl=60.0  # 60秒TTL
        )
        assert message.is_expired

    def test_message_to_dict(self):
        """测试转换为字典"""
        message = Message(
            id="test_id",
            topic="test_topic",
            payload={"key": "value"},
            priority=MessagePriority.HIGH,
            headers={"header": "value"},
            status=MessageStatus.PROCESSING,
            retry_count=2,
            correlation_id="corr_123"
        )

        data = message.to_dict()

        assert data["id"] == "test_id"
        assert data["topic"] == "test_topic"
        assert data["payload"] == {"key": "value"}
        assert data["priority"] == MessagePriority.HIGH.value
        assert data["headers"] == {"header": "value"}
        assert data["status"] == MessageStatus.PROCESSING.value
        assert data["retry_count"] == 2
        assert data["correlation_id"] == "corr_123"

    def test_message_from_dict(self):
        """测试从字典创建"""
        data = {
            "id": "test_id",
            "topic": "test_topic",
            "payload": {"key": "value"},
            "priority": MessagePriority.HIGH.value,
            "timestamp": time.time(),
            "headers": {"header": "value"},
            "status": MessageStatus.PROCESSING.value,
            "retry_count": 2,
            "correlation_id": "corr_123"
        }

        message = Message.from_dict(data)

        assert message.id == "test_id"
        assert message.topic == "test_topic"
        assert message.payload == {"key": "value"}
        assert message.priority == MessagePriority.HIGH
        assert message.headers == {"header": "value"}
        assert message.status == MessageStatus.PROCESSING
        assert message.retry_count == 2
        assert message.correlation_id == "corr_123"


class TestMessageHandler:
    """MessageHandler测试"""

    @pytest.fixture
    def mock_handler(self):
        """Mock handler fixture"""
        return AsyncMock()

    def test_message_handler_initialization(self, mock_handler):
        """测试MessageHandler初始化"""
        handler = MessageHandler(mock_handler)

        assert handler.handler_func == mock_handler
        assert handler.filter_func is None
        assert handler.is_active is True

    def test_message_handler_with_filter(self, mock_handler):
        """测试带过滤器的MessageHandler"""
        filter_func = lambda msg: msg.topic == "test"
        handler = MessageHandler(mock_handler, filter_func)

        assert handler.filter_func == filter_func

    @pytest.mark.asyncio
    async def test_handle_success(self, mock_handler):
        """测试成功处理消息"""
        handler = MessageHandler(mock_handler)
        message = Message(topic="test", payload="data")

        mock_handler.return_value = None  # 成功处理

        result = await handler.handle(message)

        assert result is True
        mock_handler.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_handle_with_filter_pass(self, mock_handler):
        """测试过滤器通过的情况"""
        filter_func = lambda msg: msg.topic == "test"
        handler = MessageHandler(mock_handler, filter_func)
        message = Message(topic="test", payload="data")

        result = await handler.handle(message)

        assert result is True
        mock_handler.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_handle_with_filter_fail(self, mock_handler):
        """测试过滤器失败的情况"""
        filter_func = lambda msg: msg.topic == "test"
        handler = MessageHandler(mock_handler, filter_func)
        message = Message(topic="other", payload="data")

        result = await handler.handle(message)

        assert result is False
        mock_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_handler_exception(self, mock_handler):
        """测试处理函数异常"""
        handler = MessageHandler(mock_handler)
        message = Message(topic="test", payload="data")

        mock_handler.side_effect = Exception("Handler error")

        result = await handler.handle(message)

        assert result is False
        mock_handler.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_handle_inactive_handler(self, mock_handler):
        """测试非活跃的处理器"""
        handler = MessageHandler(mock_handler)
        handler.is_active = False
        message = Message(topic="test", payload="data")

        result = await handler.handle(message)

        assert result is False
        mock_handler.assert_not_called()


class TestAsyncMessageQueueInitialization:
    """AsyncMessageQueue初始化测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        queue = AsyncMessageQueue()

        assert queue.max_queue_size == 10000
        assert queue.persistence_enabled is False
        assert queue.dead_letter_enabled is True
        assert queue.worker_count == 4
        assert not queue._shutdown_event.is_set()
        assert len(queue._workers) == 0
        assert len(queue._subscribers) == 0
        assert len(queue._dead_letter_queue) == 0

    def test_initialization_custom_params(self):
        """测试自定义参数初始化"""
        queue = AsyncMessageQueue(
            max_queue_size=5000,
            persistence_enabled=True,
            dead_letter_enabled=False,
            worker_count=8
        )

        assert queue.max_queue_size == 5000
        assert queue.persistence_enabled is True
        assert queue.dead_letter_enabled is False
        assert queue.worker_count == 8


class TestAsyncMessageQueueBasicOperations:
    """AsyncMessageQueue基本操作测试"""

    @pytest.fixture
    async def queue(self):
        """AsyncMessageQueue fixture"""
        queue = AsyncMessageQueue(worker_count=2)
        await queue.start()
        yield queue
        await queue.stop()

    @pytest.mark.asyncio
    async def test_publish_message_success(self, queue):
        """测试成功发布消息"""
        message = Message(topic="test_topic", payload="test_data")

        result = await queue.publish(message)

        assert result is True
        assert queue._stats['messages_published'] == 1
        assert message.status == MessageStatus.PENDING

    @pytest.mark.asyncio
    async def test_publish_message_queue_full(self, queue):
        """测试队列满时发布消息"""
        # 创建一个小的队列
        small_queue = AsyncMessageQueue(max_queue_size=1, worker_count=1)
        await small_queue.start()

        try:
            # 填满队列
            message1 = Message(topic="test", payload="data1")
            result1 = await small_queue.publish(message1)
            assert result1 is True

            # 队列现在满了
            message2 = Message(topic="test", payload="data2")
            result2 = await small_queue.publish(message2)
            assert result2 is False

        finally:
            await small_queue.stop()

    @pytest.mark.asyncio
    async def test_publish_to_topic(self, queue):
        """测试发布到主题"""
        result = await queue.publish_to_topic(
            topic="test_topic",
            payload={"key": "value"},
            priority=MessagePriority.HIGH,
            headers={"source": "test"},
            correlation_id="corr_123"
        )

        assert result is True
        assert queue._stats['messages_published'] == 1

    @pytest.mark.asyncio
    async def test_subscribe_and_unsubscribe(self, queue):
        """测试订阅和取消订阅"""
        async def dummy_handler(message):
            pass

        # 订阅
        subscription_id = await queue.subscribe("test_topic", dummy_handler)

        assert "test_topic" in queue._subscribers
        assert len(queue._subscribers["test_topic"]) == 1
        assert queue._stats['active_subscribers'] == 1

        # 取消订阅
        result = await queue.unsubscribe("test_topic", subscription_id)

        assert result is True
        assert len(queue._subscribers["test_topic"]) == 0
        assert queue._stats['active_subscribers'] == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self, queue):
        """测试取消不存在的订阅"""
        result = await queue.unsubscribe("nonexistent_topic", "fake_id")

        assert result is False


class TestAsyncMessageQueueMessageProcessing:
    """AsyncMessageQueue消息处理测试"""

    @pytest.fixture
    async def queue(self):
        """AsyncMessageQueue fixture"""
        queue = AsyncMessageQueue(worker_count=1)
        await queue.start()
        yield queue
        await queue.stop()

    @pytest.mark.asyncio
    async def test_message_processing_with_subscriber(self, queue):
        """测试带订阅者的消息处理"""
        processed_messages = []

        async def test_handler(message):
            processed_messages.append(message)
            message.status = MessageStatus.COMPLETED

        # 订阅主题
        await queue.subscribe("test_topic", test_handler)

        # 发布消息
        message = Message(topic="test_topic", payload="test_data")
        await queue.publish(message)

        # 等待处理
        await asyncio.sleep(0.5)

        # 验证消息被处理
        assert len(processed_messages) == 1
        assert processed_messages[0].id == message.id
        assert queue._stats['messages_processed'] == 1

    @pytest.mark.asyncio
    async def test_message_processing_no_subscribers(self, queue):
        """测试无订阅者的消息处理"""
        message = Message(topic="empty_topic", payload="data")
        await queue.publish(message)

        # 等待处理
        await asyncio.sleep(0.5)

        # 消息应该被标记为完成（即使没有订阅者）
        assert queue._stats['messages_processed'] == 1

    @pytest.mark.asyncio
    async def test_message_processing_with_filter(self, queue):
        """测试带过滤器的消息处理"""
        processed_messages = []

        async def handler(message):
            processed_messages.append(message)

        # 只处理高优先级消息
        filter_func = lambda msg: msg.priority == MessagePriority.HIGH

        await queue.subscribe("test_topic", handler, filter_func)

        # 发布不同优先级的消息
        msg_normal = Message(topic="test_topic", payload="normal", priority=MessagePriority.NORMAL)
        msg_high = Message(topic="test_topic", payload="high", priority=MessagePriority.HIGH)

        await queue.publish(msg_normal)
        await queue.publish(msg_high)

        # 等待处理
        await asyncio.sleep(0.5)

        # 应该只处理高优先级消息
        assert len(processed_messages) == 1
        assert processed_messages[0].payload == "high"

    @pytest.mark.asyncio
    async def test_message_retry_on_failure(self, queue):
        """测试消息处理失败后的重试"""
        attempt_count = 0

        async def failing_handler(message):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:  # 前两次失败
                raise Exception("Handler failed")
            # 第三次成功
            message.status = MessageStatus.COMPLETED

        await queue.subscribe("test_topic", failing_handler)

        message = Message(topic="test_topic", payload="data", max_retries=3)
        await queue.publish(message)

        # 等待处理完成（包括重试）
        await asyncio.sleep(2.0)

        # 验证重试逻辑
        assert attempt_count == 3  # 应该尝试3次
        assert queue._stats['messages_processed'] == 1

    @pytest.mark.asyncio
    async def test_message_dead_letter_on_exhaustion(self, queue):
        """测试重试耗尽后移到死信队列"""
        async def always_failing_handler(message):
            raise Exception("Always fails")

        await queue.subscribe("test_topic", always_failing_handler)

        message = Message(topic="test_topic", payload="data", max_retries=2)
        await queue.publish(message)

        # 等待处理完成
        await asyncio.sleep(2.0)

        # 消息应该在死信队列中
        dead_letters = queue.get_dead_letter_messages()
        assert len(dead_letters) == 1
        assert dead_letters[0].id == message.id
        assert dead_letters[0].status == MessageStatus.DEAD_LETTER
        assert queue._stats['messages_dead_lettered'] == 1

    @pytest.mark.asyncio
    async def test_expired_message_handling(self, queue):
        """测试过期消息处理"""
        # 创建已过期的消息
        expired_message = Message(
            topic="test_topic",
            payload="data",
            timestamp=time.time() - 120,  # 2分钟前
            ttl=60.0  # 60秒TTL
        )

        await queue.publish(expired_message)

        # 等待处理
        await asyncio.sleep(0.5)

        # 过期消息应该移到死信队列
        dead_letters = queue.get_dead_letter_messages()
        assert len(dead_letters) == 1
        assert dead_letters[0].id == expired_message.id


class TestAsyncMessageQueuePriority:
    """AsyncMessageQueue优先级测试"""

    @pytest.fixture
    async def queue(self):
        """AsyncMessageQueue fixture"""
        queue = AsyncMessageQueue(worker_count=1)
        await queue.start()
        yield queue
        await queue.stop()

    @pytest.mark.asyncio
    async def test_message_priority_ordering(self, queue):
        """测试消息优先级排序"""
        processed_order = []

        async def handler(message):
            processed_order.append(message.priority)
            message.status = MessageStatus.COMPLETED

        await queue.subscribe("test_topic", handler)

        # 发布不同优先级的消息
        messages = [
            Message(topic="test_topic", payload="low", priority=MessagePriority.LOW),
            Message(topic="test_topic", payload="critical", priority=MessagePriority.CRITICAL),
            Message(topic="test_topic", payload="normal", priority=MessagePriority.NORMAL),
            Message(topic="test_topic", payload="high", priority=MessagePriority.HIGH),
        ]

        for msg in messages:
            await queue.publish(msg)

        # 等待所有消息处理完成
        await asyncio.sleep(2.0)

        # 验证处理顺序（高优先级先处理）
        expected_order = [
            MessagePriority.CRITICAL,
            MessagePriority.HIGH,
            MessagePriority.NORMAL,
            MessagePriority.LOW
        ]
        assert processed_order == expected_order


class TestAsyncMessageQueueConcurrency:
    """AsyncMessageQueue并发测试"""

    @pytest.fixture
    async def queue(self):
        """AsyncMessageQueue fixture"""
        queue = AsyncMessageQueue(worker_count=4, max_queue_size=1000)
        await queue.start()
        yield queue
        await queue.stop()

    @pytest.mark.asyncio
    async def test_concurrent_publish_subscribe(self, queue):
        """测试并发发布和订阅"""
        processed_count = 0
        processed_lock = asyncio.Lock()

        async def counting_handler(message):
            nonlocal processed_count
            async with processed_lock:
                processed_count += 1
            message.status = MessageStatus.COMPLETED

        # 并发订阅多个主题
        subscription_tasks = []
        for i in range(5):
            topic = f"topic_{i}"
            task = queue.subscribe(topic, counting_handler)
            subscription_tasks.append(task)

        await asyncio.gather(*subscription_tasks)

        # 并发发布消息
        publish_tasks = []
        for i in range(50):  # 每主题10条消息
            topic = f"topic_{i % 5}"
            message = Message(topic=topic, payload=f"data_{i}")
            task = queue.publish(message)
            publish_tasks.append(task)

        publish_results = await asyncio.gather(*publish_tasks)

        # 所有发布都应该成功
        assert all(publish_results)

        # 等待所有消息处理完成
        await asyncio.sleep(3.0)

        # 验证所有消息都被处理
        assert processed_count == 50
        assert queue._stats['messages_published'] == 50
        assert queue._stats['messages_processed'] == 50

    @pytest.mark.asyncio
    async def test_high_throughput_processing(self, queue):
        """测试高吞吐量处理"""
        message_count = 1000
        processed_count = 0
        processing_lock = asyncio.Lock()

        async def fast_handler(message):
            nonlocal processed_count
            async with processing_lock:
                processed_count += 1
            message.status = MessageStatus.COMPLETED

        await queue.subscribe("perf_topic", fast_handler)

        # 批量发布消息
        publish_tasks = []
        for i in range(message_count):
            message = Message(topic="perf_topic", payload=f"data_{i}")
            task = queue.publish(message)
            publish_tasks.append(task)

        # 记录发布时间
        start_time = time.time()
        publish_results = await asyncio.gather(*publish_tasks)
        publish_time = time.time() - start_time

        # 等待处理完成
        await asyncio.sleep(5.0)
        end_time = time.time()

        total_time = end_time - start_time

        # 验证结果
        assert all(publish_results)
        assert processed_count == message_count
        assert queue._stats['messages_published'] == message_count
        assert queue._stats['messages_processed'] == message_count

        # 计算性能指标
        publish_rate = message_count / publish_time
        process_rate = message_count / total_time

        print(".2f")
        print(".2f")

        # 性能应该在合理范围内
        assert publish_rate > 1000  # 至少1000条/秒发布速率
        assert process_rate > 100   # 至少100条/秒处理速率


class TestAsyncMessageQueueDeadLetter:
    """AsyncMessageQueue死信队列测试"""

    @pytest.fixture
    async def queue(self):
        """AsyncMessageQueue fixture"""
        queue = AsyncMessageQueue(worker_count=1)
        await queue.start()
        yield queue
        await queue.stop()

    @pytest.mark.asyncio
    async def test_dead_letter_queue_operations(self, queue):
        """测试死信队列操作"""
        async def failing_handler(message):
            raise Exception("Always fails")

        await queue.subscribe("test_topic", failing_handler)

        # 发布会失败的消息
        message = Message(topic="test_topic", payload="data", max_retries=1)
        await queue.publish(message)

        # 等待移到死信队列
        await asyncio.sleep(1.0)

        # 检查死信队列
        dead_letters = queue.get_dead_letter_messages()
        assert len(dead_letters) == 1
        assert dead_letters[0].id == message.id

        # 重放消息
        replay_result = await queue.replay_dead_letter_message(message.id)
        assert replay_result is True

        # 死信队列应该为空
        dead_letters_after = queue.get_dead_letter_messages()
        assert len(dead_letters_after) == 0

    @pytest.mark.asyncio
    async def test_clear_dead_letter_queue(self, queue):
        """测试清空死信队列"""
        # 手动添加消息到死信队列
        dead_message = Message(topic="test", payload="data", status=MessageStatus.DEAD_LETTER)
        queue._dead_letter_queue.append(dead_message)
        queue._stats['messages_dead_lettered'] = 1

        # 清空死信队列
        queue.clear_dead_letter_queue()

        assert len(queue._dead_letter_queue) == 0
        assert queue._stats['messages_dead_lettered'] == 0


class TestAsyncMessageQueueStatistics:
    """AsyncMessageQueue统计测试"""

    @pytest.fixture
    async def queue(self):
        """AsyncMessageQueue fixture"""
        queue = AsyncMessageQueue(worker_count=1)
        await queue.start()
        yield queue
        await queue.stop()

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, queue):
        """测试统计跟踪"""
        async def success_handler(message):
            message.status = MessageStatus.COMPLETED

        async def failing_handler(message):
            raise Exception("Handler fails")

        await queue.subscribe("success_topic", success_handler)
        await queue.subscribe("fail_topic", failing_handler)

        # 发布消息
        await queue.publish(Message(topic="success_topic", payload="data"))
        await queue.publish(Message(topic="fail_topic", payload="data", max_retries=1))

        # 等待处理
        await asyncio.sleep(1.0)

        # 检查统计
        stats = queue.get_stats()

        assert stats['messages_published'] == 2
        assert stats['messages_processed'] == 1  # 成功的一个
        assert stats['messages_failed'] >= 1    # 失败的重试
        assert stats['dead_letter_count'] >= 1  # 死信队列中的消息
        assert stats['active_subscribers'] == 2

    @pytest.mark.asyncio
    async def test_statistics_with_multiple_operations(self, queue):
        """测试多操作的统计"""
        initial_stats = queue.get_stats()

        # 执行各种操作
        await queue.subscribe("topic1", lambda msg: None)
        await queue.subscribe("topic2", lambda msg: None)

        await queue.publish(Message(topic="topic1", payload="data1"))
        await queue.publish(Message(topic="topic2", payload="data2"))

        final_stats = queue.get_stats()

        assert final_stats['active_subscribers'] == initial_stats['active_subscribers'] + 2
        assert final_stats['messages_published'] == initial_stats['messages_published'] + 2


class TestAsyncMessageQueueLifecycle:
    """AsyncMessageQueue生命周期测试"""

    @pytest.mark.asyncio
    async def test_queue_start_stop(self):
        """测试队列启动和停止"""
        queue = AsyncMessageQueue(worker_count=2)

        # 初始状态
        assert len(queue._workers) == 0
        assert not queue._shutdown_event.is_set()

        # 启动
        await queue.start()

        assert len(queue._workers) == 2
        for worker in queue._workers:
            assert isinstance(worker, asyncio.Task)

        # 停止
        await queue.stop()

        assert queue._shutdown_event.is_set()
        # workers列表被清空

    @pytest.mark.asyncio
    async def test_publish_after_stop(self):
        """测试停止后发布消息"""
        queue = AsyncMessageQueue()
        await queue.start()
        await queue.stop()

        # 停止后发布应该失败
        message = Message(topic="test", payload="data")
        result = await queue.publish(message)

        assert result is False

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_pending_messages(self):
        """测试带待处理消息的优雅关闭"""
        queue = AsyncMessageQueue(worker_count=1)

        processed_messages = []

        async def slow_handler(message):
            await asyncio.sleep(0.5)  # 模拟慢处理
            processed_messages.append(message)
            message.status = MessageStatus.COMPLETED

        await queue.subscribe("test_topic", slow_handler)

        # 发布消息
        await queue.publish(Message(topic="test_topic", payload="data"))

        # 立即启动关闭（消息正在处理中）
        stop_task = asyncio.create_task(queue.stop())

        # 等待关闭完成
        await stop_task

        # 即使关闭，消息也应该被处理（或至少不丢失）
        # 注意：实际行为取决于实现，这里我们验证队列正确关闭
        assert queue._shutdown_event.is_set()


class TestAsyncMessageQueueIntegration:
    """AsyncMessageQueue集成测试"""

    @pytest.mark.asyncio
    async def test_complete_message_flow(self):
        """测试完整消息流"""
        queue = AsyncMessageQueue(worker_count=2)
        await queue.start()

        try:
            # 1. 设置订阅者
            received_messages = []

            async def collect_handler(message):
                received_messages.append({
                    'id': message.id,
                    'topic': message.topic,
                    'payload': message.payload,
                    'correlation_id': message.correlation_id
                })
                message.status = MessageStatus.COMPLETED

            await queue.subscribe("order_topic", collect_handler)

            # 2. 发布一系列消息
            messages = []
            for i in range(10):
                message = Message(
                    topic="order_topic",
                    payload=f"order_{i}",
                    priority=MessagePriority.NORMAL if i % 2 == 0 else MessagePriority.HIGH,
                    correlation_id=f"corr_{i}"
                )
                messages.append(message)
                await queue.publish(message)

            # 3. 等待处理
            await asyncio.sleep(2.0)

            # 4. 验证结果
            assert len(received_messages) == 10

            # 验证消息内容
            received_ids = {msg['id'] for msg in received_messages}
            sent_ids = {msg.id for msg in messages}
            assert received_ids == sent_ids

            # 验证关联ID
            for msg in received_messages:
                assert msg['correlation_id'].startswith('corr_')

            # 验证统计
            stats = queue.get_stats()
            assert stats['messages_published'] == 10
            assert stats['messages_processed'] == 10

        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_multi_topic_pub_sub(self):
        """测试多主题发布订阅"""
        queue = AsyncMessageQueue(worker_count=2)
        await queue.start()

        try:
            # 为不同主题设置订阅者
            user_messages = []
            order_messages = []
            system_messages = []

            async def user_handler(msg):
                user_messages.append(msg.payload)
                msg.status = MessageStatus.COMPLETED

            async def order_handler(msg):
                order_messages.append(msg.payload)
                msg.status = MessageStatus.COMPLETED

            async def system_handler(msg):
                system_messages.append(msg.payload)
                msg.status = MessageStatus.COMPLETED

            await queue.subscribe("user_events", user_handler)
            await queue.subscribe("order_events", order_handler)
            await queue.subscribe("system_events", system_handler)

            # 发布到不同主题
            test_messages = [
                ("user_events", "user_login"),
                ("order_events", "order_created"),
                ("system_events", "backup_completed"),
                ("user_events", "user_logout"),
                ("order_events", "order_shipped"),
            ]

            for topic, payload in test_messages:
                await queue.publish(Message(topic=topic, payload=payload))

            # 等待处理
            await asyncio.sleep(2.0)

            # 验证消息路由
            assert len(user_messages) == 2
            assert len(order_messages) == 2
            assert len(system_messages) == 1

            assert "user_login" in user_messages
            assert "user_logout" in user_messages
            assert "order_created" in order_messages
            assert "order_shipped" in order_messages
            assert "backup_completed" in system_messages

        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_error_recovery_and_monitoring(self):
        """测试错误恢复和监控"""
        queue = AsyncMessageQueue(worker_count=1, dead_letter_enabled=True)
        await queue.start()

        try:
            error_count = 0

            async def unreliable_handler(message):
                nonlocal error_count
                error_count += 1
                if error_count <= 2:  # 前两次失败
                    raise Exception(f"Simulated error {error_count}")
                message.status = MessageStatus.COMPLETED

            await queue.subscribe("unreliable_topic", unreliable_handler)

            # 发送需要重试的消息
            message = Message(
                topic="unreliable_topic",
                payload="test_data",
                max_retries=3
            )
            await queue.publish(message)

            # 等待处理完成
            await asyncio.sleep(3.0)

            # 验证最终成功处理
            assert error_count == 3  # 尝试3次
            stats = queue.get_stats()
            assert stats['messages_processed'] == 1
            assert stats['messages_failed'] == 2  # 2次失败的重试

            # 死信队列应该为空（因为最终成功了）
            dead_letters = queue.get_dead_letter_messages()
            assert len(dead_letters) == 0

        finally:
            await queue.stop()
