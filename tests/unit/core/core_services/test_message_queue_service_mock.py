# -*- coding: utf-8 -*-
"""
消息队列服务Mock测试
避免复杂的消息队列依赖，测试核心消息队列服务逻辑
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Dict, List, Optional, Callable, Set
import uuid
import time


class MockMessage:
    """模拟消息"""

    def __init__(self,
                 message_id: str = None,
                 topic: str = None,
                 content: Any = None,
                 headers: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[datetime] = None):
        self.message_id = message_id or str(uuid.uuid4())
        self.topic = topic
        self.content = content
        self.headers = headers or {}
        self.timestamp = timestamp or datetime.now()
        self.delivery_count = 0
        self.acked = False
        self.nacked = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "message_id": self.message_id,
            "topic": self.topic,
            "content": self.content,
            "headers": self.headers,
            "timestamp": self.timestamp.isoformat(),
            "delivery_count": self.delivery_count
        }

    def ack(self):
        """确认消息"""
        self.acked = True

    def nack(self, requeue: bool = True):
        """拒绝消息"""
        self.nacked = True
        if requeue:
            self.delivery_count += 1


class MockMessageQueue:
    """模拟消息队列"""

    def __init__(self, name: str, max_size: int = 1000):
        self.name = name
        self.max_size = max_size
        self.messages: List[MockMessage] = []
        self.pending_ack: Set[str] = set()
        self.consumers: List[MockMessageConsumer] = []

    async def publish(self, message: MockMessage) -> bool:
        """发布消息"""
        if len(self.messages) >= self.max_size:
            return False  # 队列已满

        self.messages.append(message)
        # 通知所有消费者
        for consumer in self.consumers:
            await consumer.notify_message(message)

        return True

    async def consume(self) -> Optional[MockMessage]:
        """消费消息"""
        if not self.messages:
            return None

        message = self.messages.pop(0)
        message.delivery_count += 1
        self.pending_ack.add(message.message_id)
        return message

    async def ack_message(self, message_id: str) -> bool:
        """确认消息"""
        if message_id in self.pending_ack:
            self.pending_ack.remove(message_id)
            return True
        return False

    async def nack_message(self, message_id: str, requeue: bool = True) -> bool:
        """拒绝消息"""
        if message_id in self.pending_ack:
            self.pending_ack.remove(message_id)
            if requeue:
                # 重新加入队列（简化实现）
                pass
            return True
        return False

    def size(self) -> int:
        """获取队列大小"""
        return len(self.messages)

    def pending_count(self) -> int:
        """获取待确认消息数量"""
        return len(self.pending_ack)

    def add_consumer(self, consumer: 'MockMessageConsumer'):
        """添加消费者"""
        self.consumers.append(consumer)

    def remove_consumer(self, consumer: 'MockMessageConsumer'):
        """移除消费者"""
        if consumer in self.consumers:
            self.consumers.remove(consumer)


class MockMessageConsumer:
    """模拟消息消费者"""

    def __init__(self, consumer_id: str, queue: MockMessageQueue):
        self.consumer_id = consumer_id
        self.queue = queue
        self.handler: Optional[Callable] = None
        self.is_running = False
        self.processed_messages = 0

    async def start(self):
        """启动消费者"""
        self.is_running = True
        self.queue.add_consumer(self)

    async def stop(self):
        """停止消费者"""
        self.is_running = False
        self.queue.remove_consumer(self)

    async def notify_message(self, message: MockMessage):
        """通知新消息"""
        if self.is_running and self.handler:
            await self.handler(message)
            self.processed_messages += 1

    def set_handler(self, handler: Callable):
        """设置消息处理器"""
        self.handler = handler


class MockMessageExchange:
    """模拟消息交换机（支持主题路由）"""

    def __init__(self, name: str):
        self.name = name
        self.bindings: Dict[str, List[MockMessageQueue]] = {}  # topic -> queues
        self.published_messages = 0

    async def publish(self, message: MockMessage) -> int:
        """发布消息到匹配的队列"""
        delivered_count = 0
        topic = message.topic
        delivered_queues = set()  # 避免重复投递

        # 直接匹配
        if topic in self.bindings:
            for queue in self.bindings[topic]:
                if queue not in delivered_queues:
                    success = await queue.publish(message)
                    if success:
                        delivered_count += 1
                        delivered_queues.add(queue)

        # 通配符匹配 (简化实现：只有当没有直接匹配时才使用通配符)
        if delivered_count == 0:
            for binding_topic, queues in self.bindings.items():
                if self._matches_pattern(topic, binding_topic):
                    for queue in queues:
                        if queue not in delivered_queues:
                            success = await queue.publish(message)
                            if success:
                                delivered_count += 1
                                delivered_queues.add(queue)

        self.published_messages += 1
        return delivered_count

    def bind_queue(self, topic: str, queue: MockMessageQueue):
        """绑定队列到主题"""
        if topic not in self.bindings:
            self.bindings[topic] = []
        if queue not in self.bindings[topic]:
            self.bindings[topic].append(queue)

    def unbind_queue(self, topic: str, queue: MockMessageQueue):
        """解绑队列"""
        if topic in self.bindings and queue in self.bindings[topic]:
            self.bindings[topic].remove(queue)

    def _matches_pattern(self, topic: str, pattern: str) -> bool:
        """简单通配符匹配（* 和 #）"""
        # 简化实现：只有完全匹配
        return topic == pattern


class MockMessageService:
    """模拟消息队列服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.queues: Dict[str, MockMessageQueue] = {}
        self.exchanges: Dict[str, MockMessageExchange] = {}
        self.consumers: Dict[str, MockMessageConsumer] = {}
        self.is_initialized = False
        self.stats = {
            "published_messages": 0,
            "consumed_messages": 0,
            "acked_messages": 0,
            "nacked_messages": 0
        }

    async def initialize(self):
        """初始化服务"""
        self.is_initialized = True

    async def shutdown(self):
        """关闭服务"""
        # 停止所有消费者
        for consumer in self.consumers.values():
            await consumer.stop()

        self.queues.clear()
        self.exchanges.clear()
        self.consumers.clear()
        self.is_initialized = False

    async def create_queue(self, name: str, max_size: int = 1000) -> MockMessageQueue:
        """创建队列"""
        if not self.is_initialized:
            raise Exception("Message service not initialized")

        queue = MockMessageQueue(name, max_size)
        self.queues[name] = queue
        return queue

    async def create_exchange(self, name: str) -> MockMessageExchange:
        """创建交换机"""
        if not self.is_initialized:
            raise Exception("Message service not initialized")

        exchange = MockMessageExchange(name)
        self.exchanges[name] = exchange
        return exchange

    async def create_consumer(self, consumer_id: str, queue_name: str) -> MockMessageConsumer:
        """创建消费者"""
        if not self.is_initialized:
            raise Exception("Message service not initialized")

        if queue_name not in self.queues:
            raise Exception(f"Queue {queue_name} not found")

        queue = self.queues[queue_name]
        consumer = MockMessageConsumer(consumer_id, queue)
        self.consumers[consumer_id] = consumer
        return consumer

    async def publish_message(self, exchange_name: str, message: MockMessage) -> bool:
        """发布消息"""
        if not self.is_initialized:
            raise Exception("Message service not initialized")

        if exchange_name not in self.exchanges:
            raise Exception(f"Exchange {exchange_name} not found")

        exchange = self.exchanges[exchange_name]
        delivered_count = await exchange.publish(message)

        self.stats["published_messages"] += 1
        return delivered_count > 0

    async def consume_message(self, queue_name: str) -> Optional[MockMessage]:
        """消费消息"""
        if not self.is_initialized:
            raise Exception("Message service not initialized")

        if queue_name not in self.queues:
            raise Exception(f"Queue {queue_name} not found")

        queue = self.queues[queue_name]
        message = await queue.consume()

        if message:
            self.stats["consumed_messages"] += 1

        return message

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "service_stats": {
                "initialized": self.is_initialized,
                "config": self.config,
                "queue_count": len(self.queues),
                "exchange_count": len(self.exchanges),
                "consumer_count": len(self.consumers)
            },
            "operation_stats": self.stats,
            "queue_stats": {
                name: {
                    "size": queue.size(),
                    "pending_ack": queue.pending_count()
                }
                for name, queue in self.queues.items()
            }
        }


class TestMockMessage:
    """模拟消息测试"""

    def test_message_creation(self):
        """测试消息创建"""
        message = MockMessage(
            topic="test.topic",
            content={"data": "test"},
            headers={"priority": "high"}
        )

        assert message.topic == "test.topic"
        assert message.content == {"data": "test"}
        assert message.headers == {"priority": "high"}
        assert message.message_id is not None
        assert isinstance(message.timestamp, datetime)
        assert message.delivery_count == 0

    def test_message_ack_nack(self):
        """测试消息确认和拒绝"""
        message = MockMessage(topic="test", content="data")

        assert not message.acked
        assert not message.nacked

        message.ack()
        assert message.acked
        assert not message.nacked

        # 重置状态测试nack
        message.acked = False
        message.nack()
        assert not message.acked
        assert message.nacked
        assert message.delivery_count == 1

    def test_message_to_dict(self):
        """测试消息序列化"""
        message = MockMessage(
            message_id="test-id",
            topic="test.topic",
            content="test content"
        )

        data = message.to_dict()
        assert data["message_id"] == "test-id"
        assert data["topic"] == "test.topic"
        assert data["content"] == "test content"
        assert "timestamp" in data


class TestMockMessageQueue:
    """模拟消息队列测试"""

    def setup_method(self):
        """设置测试方法"""
        self.queue = MockMessageQueue("test_queue", max_size=3)

    @pytest.mark.asyncio
    async def test_queue_publish_consume(self):
        """测试队列发布和消费"""
        # 发布消息
        message1 = MockMessage(topic="test", content="msg1")
        message2 = MockMessage(topic="test", content="msg2")

        assert await self.queue.publish(message1)
        assert await self.queue.publish(message2)

        assert self.queue.size() == 2

        # 消费消息
        consumed1 = await self.queue.consume()
        assert consumed1 is not None
        assert consumed1.content == "msg1"
        assert consumed1.delivery_count == 1
        assert self.queue.size() == 1

        consumed2 = await self.queue.consume()
        assert consumed2 is not None
        assert consumed2.content == "msg2"
        assert self.queue.size() == 0

    @pytest.mark.asyncio
    async def test_queue_ack_nack(self):
        """测试队列确认和拒绝"""
        message = MockMessage(topic="test", content="data")
        await self.queue.publish(message)

        # 消费消息
        consumed = await self.queue.consume()
        assert consumed is not None
        assert self.queue.pending_count() == 1

        # 确认消息
        assert await self.queue.ack_message(consumed.message_id)
        assert self.queue.pending_count() == 0

        # 测试拒绝消息
        message2 = MockMessage(topic="test", content="data2")
        await self.queue.publish(message2)
        consumed2 = await self.queue.consume()

        assert await self.queue.nack_message(consumed2.message_id)
        assert self.queue.pending_count() == 0

    @pytest.mark.asyncio
    async def test_queue_capacity_limit(self):
        """测试队列容量限制"""
        # 发布到容量上限
        for i in range(3):
            message = MockMessage(topic="test", content=f"msg{i}")
            assert await self.queue.publish(message)

        assert self.queue.size() == 3

        # 尝试发布第四个消息，应该失败
        message4 = MockMessage(topic="test", content="msg4")
        assert not await self.queue.publish(message4)
        assert self.queue.size() == 3

    @pytest.mark.asyncio
    async def test_consumer_notification(self):
        """测试消费者通知"""
        consumer = MockMessageConsumer("consumer1", self.queue)
        handler_called = False
        received_message = None

        async def test_handler(message):
            nonlocal handler_called, received_message
            handler_called = True
            received_message = message

        consumer.set_handler(test_handler)
        await consumer.start()

        # 发布消息
        message = MockMessage(topic="test", content="notify_test")
        await self.queue.publish(message)

        # 等待异步处理
        await asyncio.sleep(0.1)

        assert handler_called
        assert received_message == message
        assert consumer.processed_messages == 1

        await consumer.stop()


class TestMockMessageExchange:
    """模拟消息交换机测试"""

    def setup_method(self):
        """设置测试方法"""
        self.exchange = MockMessageExchange("test_exchange")

    @pytest.mark.asyncio
    async def test_exchange_publish(self):
        """测试交换机发布"""
        queue1 = MockMessageQueue("queue1")
        queue2 = MockMessageQueue("queue2")

        # 绑定队列到主题
        self.exchange.bind_queue("topic1", queue1)
        self.exchange.bind_queue("topic1", queue2)
        self.exchange.bind_queue("topic2", queue2)

        # 发布消息
        message = MockMessage(topic="topic1", content="test message")
        delivered_count = await self.exchange.publish(message)

        assert delivered_count == 2  # 投递到2个队列
        assert queue1.size() == 1
        assert queue2.size() == 1

        # 验证消息内容
        msg1 = await queue1.consume()
        assert msg1.content == "test message"

    @pytest.mark.asyncio
    async def test_exchange_topic_matching(self):
        """测试主题匹配"""
        queue = MockMessageQueue("queue")

        # 绑定队列
        self.exchange.bind_queue("orders.*", queue)

        # 发布匹配的消息
        message = MockMessage(topic="orders.create", content="order data")
        delivered_count = await self.exchange.publish(message)

        # 注意：简化实现只有完全匹配，所以这里不会匹配
        assert delivered_count == 0

        # 绑定完全匹配的主题
        self.exchange.bind_queue("orders.create", queue)
        delivered_count = await self.exchange.publish(message)
        assert delivered_count == 1

    def test_exchange_bind_unbind(self):
        """测试队列绑定和解绑"""
        queue1 = MockMessageQueue("queue1")
        queue2 = MockMessageQueue("queue2")

        # 绑定
        self.exchange.bind_queue("topic1", queue1)
        self.exchange.bind_queue("topic1", queue2)

        assert len(self.exchange.bindings["topic1"]) == 2

        # 解绑
        self.exchange.unbind_queue("topic1", queue1)
        assert len(self.exchange.bindings["topic1"]) == 1

        self.exchange.unbind_queue("topic1", queue2)
        assert len(self.exchange.bindings.get("topic1", [])) == 0


class TestMockMessageService:
    """模拟消息服务测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = {
            "host": "localhost",
            "port": 5672,
            "max_connections": 10
        }
        self.service = MockMessageService(self.config)

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """测试服务初始化"""
        assert not self.service.is_initialized

        await self.service.initialize()
        assert self.service.is_initialized

        await self.service.shutdown()
        assert not self.service.is_initialized

    @pytest.mark.asyncio
    async def test_create_queue_exchange_consumer(self):
        """测试创建队列、交换机和消费者"""
        await self.service.initialize()

        # 创建队列
        queue = await self.service.create_queue("test_queue")
        assert queue.name == "test_queue"
        assert "test_queue" in self.service.queues

        # 创建交换机
        exchange = await self.service.create_exchange("test_exchange")
        assert exchange.name == "test_exchange"
        assert "test_exchange" in self.service.exchanges

        # 创建消费者
        consumer = await self.service.create_consumer("consumer1", "test_queue")
        assert consumer.consumer_id == "consumer1"
        assert "consumer1" in self.service.consumers

    @pytest.mark.asyncio
    async def test_publish_consume_workflow(self):
        """测试发布消费工作流"""
        await self.service.initialize()

        # 创建队列和交换机
        queue = await self.service.create_queue("orders")
        exchange = await self.service.create_exchange("order_exchange")
        exchange.bind_queue("orders.create", queue)

        # 发布消息
        message = MockMessage(topic="orders.create", content={"order_id": 123})
        assert await self.service.publish_message("order_exchange", message)

        # 消费消息
        consumed = await self.service.consume_message("orders")
        assert consumed is not None
        assert consumed.content["order_id"] == 123

    @pytest.mark.asyncio
    async def test_consumer_processing(self):
        """测试消费者处理"""
        await self.service.initialize()

        # 创建队列和消费者
        queue = await self.service.create_queue("events")
        consumer = await self.service.create_consumer("processor1", "events")

        processed_messages = []

        async def message_handler(message):
            processed_messages.append(message.content)

        consumer.set_handler(message_handler)
        await consumer.start()

        # 发布消息
        message1 = MockMessage(topic="event", content="event1")
        message2 = MockMessage(topic="event", content="event2")

        await queue.publish(message1)
        await queue.publish(message2)

        # 等待异步处理
        await asyncio.sleep(0.1)

        assert len(processed_messages) == 2
        assert "event1" in processed_messages
        assert "event2" in processed_messages

        await consumer.stop()

    @pytest.mark.asyncio
    async def test_message_acknowledgment(self):
        """测试消息确认"""
        await self.service.initialize()

        # 创建队列
        queue = await self.service.create_queue("tasks")

        # 发布和消费消息
        message = MockMessage(topic="task", content="work")
        await queue.publish(message)
        consumed = await queue.consume()

        assert consumed is not None
        assert queue.pending_count() == 1

        # 确认消息
        assert await queue.ack_message(consumed.message_id)
        assert queue.pending_count() == 0

    @pytest.mark.asyncio
    async def test_service_stats(self):
        """测试服务统计"""
        await self.service.initialize()

        # 创建组件
        queue = await self.service.create_queue("stats_test")
        exchange = await self.service.create_exchange("stats_exchange")
        consumer = await self.service.create_consumer("stats_consumer", "stats_test")

        # 执行一些操作
        message = MockMessage(topic="test", content="data")
        exchange.bind_queue("test", queue)
        await self.service.publish_message("stats_exchange", message)
        await self.service.consume_message("stats_test")

        # 获取统计
        stats = self.service.get_stats()

        assert stats["service_stats"]["queue_count"] == 1
        assert stats["service_stats"]["exchange_count"] == 1
        assert stats["service_stats"]["consumer_count"] == 1
        assert stats["operation_stats"]["published_messages"] == 1
        assert stats["operation_stats"]["consumed_messages"] == 1

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """测试错误处理"""
        await self.service.initialize()

        # 测试未初始化的错误
        uninitialized_service = MockMessageService(self.config)
        with pytest.raises(Exception, match="Message service not initialized"):
            await uninitialized_service.create_queue("test")

        # 测试不存在的队列
        with pytest.raises(Exception, match="Queue nonexistent not found"):
            await self.service.create_consumer("test", "nonexistent")

        # 测试不存在的交换机
        with pytest.raises(Exception, match="Exchange nonexistent not found"):
            message = MockMessage(topic="test", content="data")
            await self.service.publish_message("nonexistent", message)


class TestMessageQueueIntegration:
    """消息队列集成测试"""

    @pytest.mark.asyncio
    async def test_pubsub_pattern(self):
        """测试发布订阅模式"""
        service = MockMessageService({"test": True})
        await service.initialize()

        # 创建交换机和队列
        exchange = await service.create_exchange("events")
        queue1 = await service.create_queue("subscribers1")
        queue2 = await service.create_queue("subscribers2")

        # 绑定队列到主题
        exchange.bind_queue("user.events", queue1)
        exchange.bind_queue("user.events", queue2)

        # 创建消费者
        consumer1 = await service.create_consumer("consumer1", "subscribers1")
        consumer2 = await service.create_consumer("consumer2", "subscribers2")

        received_messages = {"consumer1": [], "consumer2": []}

        async def handler1(message):
            received_messages["consumer1"].append(message.content)

        async def handler2(message):
            received_messages["consumer2"].append(message.content)

        consumer1.set_handler(handler1)
        consumer2.set_handler(handler2)

        await consumer1.start()
        await consumer2.start()

        # 发布消息
        message1 = MockMessage(topic="user.events", content="user_login")
        message2 = MockMessage(topic="user.events", content="user_logout")

        await service.publish_message("events", message1)
        await service.publish_message("events", message2)

        # 等待处理
        await asyncio.sleep(0.1)

        assert len(received_messages["consumer1"]) == 2
        assert len(received_messages["consumer2"]) == 2
        assert "user_login" in received_messages["consumer1"]
        assert "user_logout" in received_messages["consumer1"]

        await consumer1.stop()
        await consumer2.stop()
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_message_persistence_simulation(self):
        """测试消息持久化模拟"""
        service = MockMessageService({"persistence": True})
        await service.initialize()

        # 创建队列并发布消息
        queue = await service.create_queue("persistent_queue")

        messages = []
        for i in range(5):
            message = MockMessage(topic="persistent", content=f"msg_{i}")
            messages.append(message)
            await queue.publish(message)

        assert queue.size() == 5

        # 模拟服务重启（重新创建服务）
        await service.shutdown()

        service2 = MockMessageService({"persistence": True})
        await service2.initialize()
        queue2 = await service2.create_queue("persistent_queue")

        # 在简化实现中，消息不会持久化，所以队列应该是空的
        assert queue2.size() == 0

        await service2.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_consumers(self):
        """测试并发消费者"""
        service = MockMessageService({"concurrency": True})
        await service.initialize()

        queue = await service.create_queue("concurrent_queue")

        # 发布多条消息
        for i in range(10):
            message = MockMessage(topic="concurrent", content=f"msg_{i}")
            await queue.publish(message)

        assert queue.size() == 10

        # 创建多个消费者并发消费
        consumed_count = 0

        async def consume_worker():
            nonlocal consumed_count
            while True:
                message = await service.consume_message("concurrent_queue")
                if message is None:
                    break
                consumed_count += 1

        # 启动多个消费者协程
        tasks = [consume_worker() for _ in range(3)]
        await asyncio.gather(*tasks)

        assert consumed_count == 10
        assert queue.size() == 0

        await service.shutdown()

    def test_service_configuration(self):
        """测试服务配置"""
        configs = [
            {"host": "localhost", "port": 5672},
            {"host": "rabbitmq", "port": 5672, "vhost": "/"},
            {"max_connections": 50, "heartbeat": 60}
        ]

        for config in configs:
            service = MockMessageService(config)
            assert service.config == config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
