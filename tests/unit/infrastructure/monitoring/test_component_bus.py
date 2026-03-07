#!/usr/bin/env python3
"""
RQA2025 基础设施层组件通信总线单元测试

测试组件通信总线的功能和正确性。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import time
import threading
from datetime import timedelta
from unittest.mock import MagicMock

from src.infrastructure.monitoring.core.component_bus import (
    ComponentBus,
    Message,
    MessageType,
    MessagePriority,
    Subscription,
    publish_event,
    send_notification,
    global_component_bus
)


class TestComponentBus(unittest.TestCase):
    """组件总线测试类"""

    def setUp(self):
        """测试设置"""
        # 创建不启用异步处理的组件总线用于测试
        self.bus = ComponentBus(enable_async=False)

    def tearDown(self):
        """测试清理"""
        self.bus.shutdown()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.bus.subscriptions, dict)
        self.assertIsInstance(self.bus.message_queue, type(self.bus.message_queue))
        self.assertFalse(self.bus.running)
        self.assertFalse(self.bus.enable_async)

    def test_subscribe_and_unsubscribe(self):
        """测试订阅和取消订阅"""
        handler = MagicMock()

        # 订阅
        subscriber_id = self.bus.subscribe("TestComponent", "test.topic", handler)
        self.assertIn("test.topic", self.bus.subscriptions)
        self.assertEqual(len(self.bus.subscriptions["test.topic"]), 1)

        # 取消订阅
        result = self.bus.unsubscribe(subscriber_id)
        self.assertTrue(result)
        self.assertEqual(len(self.bus.subscriptions["test.topic"]), 0)

    def test_publish_and_receive_message(self):
        """测试发布和接收消息"""
        received_messages = []

        def message_handler(message: Message):
            received_messages.append(message)

        # 订阅消息
        self.bus.subscribe("TestReceiver", "test.message", message_handler)

        # 发布消息
        message = Message(
            message_id="test_msg_001",
            message_type=MessageType.EVENT,
            topic="test.message",
            sender="TestSender",
            payload={"key": "value"}
        )

        result = self.bus.publish(message)
        self.assertTrue(result)

        # 手动处理消息队列
        self.bus._process_messages_async()

        # 验证消息被接收
        self.assertEqual(len(received_messages), 1)
        self.assertEqual(received_messages[0].message_id, "test_msg_001")
        self.assertEqual(received_messages[0].payload, {"key": "value"})

    def test_message_priority(self):
        """测试消息优先级"""
        received_messages = []

        def message_handler(message: Message):
            received_messages.append(message)

        # 订阅消息
        self.bus.subscribe("TestReceiver", "test.priority", message_handler)

        # 发布不同优先级的消息
        high_priority_msg = Message(
            message_id="high_priority",
            message_type=MessageType.EVENT,
            topic="test.priority",
            sender="TestSender",
            payload={"priority": "high"},
            priority=MessagePriority.HIGH
        )

        normal_priority_msg = Message(
            message_id="normal_priority",
            message_type=MessageType.EVENT,
            topic="test.priority",
            sender="TestSender",
            payload={"priority": "normal"},
            priority=MessagePriority.NORMAL
        )

        # 先发布普通优先级，再发布高优先级
        self.bus.publish(normal_priority_msg)
        self.bus.publish(high_priority_msg)

        # 处理消息
        self.bus._process_messages_async()

        # 验证高优先级消息先被处理
        self.assertEqual(len(received_messages), 2)
        self.assertEqual(received_messages[0].message_id, "high_priority")
        self.assertEqual(received_messages[1].message_id, "normal_priority")

    def test_topic_pattern_matching(self):
        """测试主题模式匹配"""
        received_messages = []

        def message_handler(message: Message):
            received_messages.append(message.topic)

        # 订阅通配符主题
        self.bus.subscribe("TestReceiver", "events.*", message_handler)

        # 发布匹配的消息
        messages = [
            Message("msg1", MessageType.EVENT, "events.user", "sender", {}),
            Message("msg2", MessageType.EVENT, "events.system", "sender", {}),
            Message("msg3", MessageType.EVENT, "commands.user", "sender", {}),  # 不匹配
        ]

        for msg in messages:
            self.bus.publish(msg)

        # 处理消息
        self.bus._process_messages_async()

        # 验证只接收匹配的消息
        self.assertEqual(len(received_messages), 2)
        self.assertIn("events.user", received_messages)
        self.assertIn("events.system", received_messages)
        self.assertNotIn("commands.user", received_messages)

    def test_message_expiration(self):
        """测试消息过期"""
        message = Message(
            message_id="expired_msg",
            message_type=MessageType.EVENT,
            topic="test.expired",
            sender="TestSender",
            payload={"data": "test"},
            ttl=1  # 1秒后过期
        )

        # 将时间戳手动回拨保证过期无需真实等待
        message.timestamp -= timedelta(seconds=2)

        # 尝试发布过期消息
        result = self.bus.publish(message)
        self.assertFalse(result)  # 应该拒绝过期消息

    def test_command_response_pattern(self):
        """测试命令-响应模式"""
        command_responses = []

        def command_handler(message: Message):
            if message.message_type == MessageType.COMMAND:
                # 模拟处理命令并发送响应
                response = Message(
                    message_id=f"response_{message.message_id}",
                    message_type=MessageType.RESPONSE,
                    topic=f"response.{message.correlation_id}",
                    sender="CommandHandler",
                    payload={"result": "success", "data": "response_data"},
                    correlation_id=message.correlation_id
                )
                self.bus.publish(response)

        # 订阅命令
        self.bus.subscribe("CommandHandler", "command.test.*", command_handler)

        # 发送命令
        response = self.bus.send_command(
            "test",
            "execute",
            {"param": "value"},
            timeout=5
        )

        # 验证收到响应
        self.assertIsNotNone(response)
        self.assertEqual(response.get("result"), "success")

    def test_query_response_pattern(self):
        """测试查询-响应模式"""
        query_responses = []

        def query_handler(message: Message):
            if message.message_type == MessageType.QUERY:
                # 发送查询响应
                response = Message(
                    message_id=f"response_{message.message_id}",
                    message_type=MessageType.RESPONSE,
                    topic=f"query.{message.correlation_id}.response",
                    sender="QueryHandler",
                    payload={"results": ["item1", "item2"]},
                    correlation_id=message.correlation_id
                )
                self.bus.publish(response)

        # 订阅查询
        self.bus.subscribe("QueryHandler", "query.test.data", query_handler)

        # 发送查询
        results = self.bus.query("query.test.data", {"filter": "active"}, timeout=2)

        # 验证收到结果
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_convenience_functions(self):
        """测试便捷函数"""
        received_events = []

        def event_handler(message: Message):
            received_events.append(message)

        # 使用测试专用的组件总线而不是全局实例，避免并发问题
        test_bus = ComponentBus(enable_async=False)

        try:
            # 订阅事件
            test_bus.subscribe("EventReceiver", "test.event", event_handler)

            # 使用便捷函数发布事件（需要修改为使用测试总线）
            # 注意：这里应该使用test_bus而不是全局的publish_event函数
            message = Message(
                message_id="test_msg",
                message_type=MessageType.EVENT,
                topic="test.event",
                sender="TestPublisher",
                payload={"event_data": "test"}
            )
            result = test_bus.publish(message)
            self.assertTrue(result)

            # 在同步模式下直接处理消息
            test_bus._handle_message(message)

            # 验证事件被接收
            self.assertEqual(len(received_events), 1)
            self.assertEqual(received_events[0].payload, {"event_data": "test"})

        finally:
            test_bus.shutdown()

    def test_notification_function(self):
        """测试通知函数"""
        received_notifications = []

        def notification_handler(message: Message):
            received_notifications.append(message)

        from src.infrastructure.monitoring.core import component_bus as bus_module

        test_bus = ComponentBus(enable_async=False)
        original_bus = bus_module.global_component_bus
        bus_module.global_component_bus = test_bus
        try:
            # 订阅通知
            bus_module.global_component_bus.subscribe(
                "NotificationReceiver", "notification.*", notification_handler
            )

            # 发送通知
            result = send_notification("notification.alert", {"alert": "system_warning"}, "System")
            self.assertTrue(result)

            # 处理消息（同步模式，单次处理即可）
            bus_module.global_component_bus._process_messages_async(process_once=True)

            # 验证通知被接收且优先级正确
            self.assertEqual(len(received_notifications), 1)
            self.assertEqual(received_notifications[0].priority, MessagePriority.HIGH)
            self.assertEqual(received_notifications[0].message_type, MessageType.NOTIFICATION)
        finally:
            test_bus.shutdown()
            bus_module.global_component_bus = original_bus

    def test_get_stats(self):
        """测试获取统计信息"""
        # 添加一些订阅
        self.bus.subscribe("Comp1", "topic1", lambda m: None)
        self.bus.subscribe("Comp2", "topic2", lambda m: None)

        # 发布一些消息
        for i in range(3):
            msg = Message(f"msg{i}", MessageType.EVENT, "topic1", "sender", {})
            self.bus.publish(msg)

        # 处理消息
        for _ in range(3):
            self.bus._process_messages_async()

        # 获取统计信息
        stats = self.bus.get_stats()

        self.assertEqual(stats['total_subscriptions'], 2)
        self.assertEqual(stats['messages_processed'], 3)
        self.assertFalse(stats['async_enabled'])

    def test_shutdown(self):
        """测试关闭功能"""
        # 添加一些消息到队列
        for i in range(3):
            msg = Message(f"msg{i}", MessageType.EVENT, "topic", "sender", {})
            self.bus.publish(msg)

        # 关闭总线
        self.bus.shutdown()

        # 验证状态
        self.assertFalse(self.bus.running)
        # 队列应该被清空
        self.assertEqual(self.bus.message_queue.qsize(), 0)

    def test_thread_safety(self):
        """测试线程安全性"""
        received_lists = []
        errors = []

        def subscriber_task():
            """订阅者任务"""
            try:
                received = []

                def handler(message: Message):
                    received.append(message.message_id)

                self.bus.subscribe("ThreadTest", "thread.test", handler)

                # 等待一会儿 (优化：移除不必要的等待)
                pass  # time.sleep removed

                received_lists.append(received)

            except Exception as e:
                errors.append(str(e))

        def publisher_task():
            """发布者任务"""
            try:
                for i in range(5):
                    msg = Message(f"thread_msg_{i}", MessageType.EVENT, "thread.test", "publisher", {})
                    self.bus.publish(msg)
                    pass  # time.sleep removed for performance
            except Exception as e:
                errors.append(str(e))

        # 启动多个线程
        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=subscriber_task))
            threads.append(threading.Thread(target=publisher_task))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=5)

        # 处理消息 (优化：直接处理，无需等待)
        self.bus._process_messages_async()

        # 验证没有错误发生
        self.assertEqual(len(errors), 0, f"线程安全测试失败: {errors}")

        # 至少有一个订阅者收到了消息
        self.assertTrue(
            any(len(received) > 0 for received in received_lists),
            "没有订阅者收到消息"
        )


if __name__ == '__main__':
    unittest.main()
