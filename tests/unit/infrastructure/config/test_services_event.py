#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 事件服务深度测试
验证ConfigEventBus和EventSubscriber的完整功能覆盖，目标覆盖率85%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Callable, Optional, Any


class TestEventService(unittest.TestCase):
    """测试事件服务"""

    def setUp(self):
        """测试前准备"""
        self.version_manager = Mock()
        self.version_manager.get_latest_version.return_value = "v1.0.0"

        # 创建事件总线
        from src.infrastructure.config.services.event_service import ConfigEventBus
        self.event_bus = ConfigEventBus(self.version_manager)

    def tearDown(self):
        """测试后清理"""
        # 清理事件总线状态
        self.event_bus._subscribers.clear()
        self.event_bus._subscription_ids.clear()
        self.event_bus._dead_letters.clear()

    # ==================== 基础功能测试 ====================

    def test_config_event_bus_initialization(self):
        """测试配置事件总线初始化"""
        from src.infrastructure.config.services.event_service import ConfigEventBus

        event_bus = ConfigEventBus(self.version_manager)

        # 验证初始化
        self.assertIsInstance(event_bus._subscribers, dict)
        self.assertIsInstance(event_bus._subscription_ids, dict)
        self.assertIsInstance(event_bus._dead_letters, list)
        self.assertEqual(event_bus._version_manager, self.version_manager)

    def test_event_subscriber_initialization(self):
        """测试事件订阅者初始化"""
        from src.infrastructure.config.services.event_service import EventSubscriber

        subscriber = EventSubscriber(self.event_bus)

        # 验证初始化
        self.assertEqual(subscriber._event_bus, self.event_bus)

    # ==================== 发布和订阅测试 ====================

    def test_publish_event(self):
        """测试发布事件"""
        # 设置版本管理器
        self.version_manager.get_latest_version.return_value = "v2.0.0"

        # 发布事件
        payload = {"key": "test_key", "value": "test_value"}
        self.event_bus.publish("test_event", payload)

        # 验证版本管理器被调用
        self.version_manager.get_latest_version.assert_called_once()

    def test_publish_event_without_subscribers(self):
        """测试发布事件（无订阅者）"""
        payload = {"message": "test"}
        # 不应该抛出异常
        self.event_bus.publish("no_subscribers_event", payload)

    def test_subscribe_and_publish(self):
        """测试订阅和发布"""
        received_events = []

        def test_handler(event_data: Dict):
            received_events.append(event_data)

        # 订阅事件
        subscription_id = self.event_bus.subscribe("test_event", test_handler)

        # 验证订阅ID
        self.assertIsInstance(subscription_id, str)
        self.assertTrue(subscription_id.startswith("sub-"))

        # 验证订阅者已注册
        self.assertIn("test_event", self.event_bus._subscribers)
        self.assertEqual(len(self.event_bus._subscribers["test_event"]), 1)
        self.assertIn(subscription_id, self.event_bus._subscription_ids["test_event"])

        # 发布事件
        payload = {"action": "update", "target": "config"}
        self.event_bus.publish("test_event", payload)

        # 验证事件被接收
        self.assertEqual(len(received_events), 1)
        received_event = received_events[0]

        # 验证事件数据包含原始payload和版本信息
        self.assertEqual(received_event["action"], "update")
        self.assertEqual(received_event["target"], "config")
        self.assertEqual(received_event["version"], "v1.0.0")

    def test_subscribe_with_filter(self):
        """测试带过滤器的订阅"""
        received_events = []

        def test_handler(event_data: Dict):
            received_events.append(event_data)

        def filter_func(event_data: Dict) -> bool:
            return event_data.get("priority", 0) > 5

        # 订阅带过滤器的事件
        subscription_id = self.event_bus.subscribe("filtered_event", test_handler, filter_func)

        # 发布低优先级事件（应该被过滤）
        self.event_bus.publish("filtered_event", {"message": "low_priority", "priority": 3})

        # 发布高优先级事件（应该通过过滤）
        self.event_bus.publish("filtered_event", {"message": "high_priority", "priority": 8})

        # 验证只有高优先级事件被接收
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0]["priority"], 8)

    def test_unsubscribe(self):
        """测试取消订阅"""
        def test_handler(event_data: Dict):
            pass

        # 订阅事件
        subscription_id = self.event_bus.subscribe("test_event", test_handler)

        # 验证已订阅
        self.assertIn("test_event", self.event_bus._subscribers)
        self.assertIn(subscription_id, self.event_bus._subscription_ids["test_event"])

        # 取消订阅
        result = self.event_bus.unsubscribe("test_event", subscription_id)

        # 验证取消订阅成功
        self.assertTrue(result)
        self.assertNotIn(subscription_id, self.event_bus._subscription_ids["test_event"])
        self.assertEqual(len(self.event_bus._subscribers["test_event"]), 0)

    def test_unsubscribe_nonexistent(self):
        """测试取消不存在的订阅"""
        # 尝试取消不存在的订阅
        result = self.event_bus.unsubscribe("nonexistent_event", "nonexistent_id")

        # 验证返回False
        self.assertFalse(result)

    # ==================== 事件处理测试 ====================

    def test_event_handler_exception(self):
        """测试事件处理器异常"""
        from src.infrastructure.config.config_exceptions import ConfigLoadError

        def failing_handler(event_data: Dict):
            raise ValueError("Handler failed")

        # 订阅失败的处理器
        self.event_bus.subscribe("failing_event", failing_handler)

        # 发布事件应该抛出异常
        payload = {"test": "data"}
        with self.assertRaises(ConfigLoadError) as cm:
            self.event_bus.publish("failing_event", payload)

        self.assertIn("事件处理失败", str(cm.exception))

        # 验证死信队列包含失败的事件
        dead_letters = self.event_bus.get_dead_letters()
        self.assertEqual(len(dead_letters), 1)

        dead_letter = dead_letters[0]
        self.assertEqual(dead_letter["event"], "failing_event")
        self.assertEqual(dead_letter["payload"], payload)
        self.assertIn("Handler failed", dead_letter["error"])

    def test_multiple_subscribers_same_event(self):
        """测试同一事件的多个订阅者"""
        received_by_handler1 = []
        received_by_handler2 = []

        def handler1(event_data: Dict):
            received_by_handler1.append(event_data)

        def handler2(event_data: Dict):
            received_by_handler2.append(event_data)

        # 订阅同一个事件的两个处理器
        self.event_bus.subscribe("multi_event", handler1)
        self.event_bus.subscribe("multi_event", handler2)

        # 发布事件
        payload = {"shared": "data"}
        self.event_bus.publish("multi_event", payload)

        # 验证两个处理器都接收到了事件
        self.assertEqual(len(received_by_handler1), 1)
        self.assertEqual(len(received_by_handler2), 1)
        self.assertEqual(received_by_handler1[0]["shared"], "data")
        self.assertEqual(received_by_handler2[0]["shared"], "data")

    def test_get_subscribers(self):
        """测试获取订阅者"""
        def handler1(event_data: Dict):
            pass

        def handler2(event_data: Dict):
            pass

        # 订阅事件
        sub_id1 = self.event_bus.subscribe("test_event", handler1)
        sub_id2 = self.event_bus.subscribe("test_event", handler2)

        # 获取订阅者
        subscribers = self.event_bus.get_subscribers("test_event")

        # 验证订阅者信息
        self.assertIsInstance(subscribers, dict)
        self.assertIn(sub_id1, subscribers)
        self.assertIn(sub_id2, subscribers)
        self.assertEqual(subscribers[sub_id1], handler1)
        self.assertEqual(subscribers[sub_id2], handler2)

        # 测试获取不存在事件的订阅者
        empty_subscribers = self.event_bus.get_subscribers("nonexistent_event")
        self.assertEqual(empty_subscribers, {})

    # ==================== 通知方法测试 ====================

    def test_notify_config_updated(self):
        """测试配置更新通知"""
        received_events = []

        def test_handler(event_data: Dict):
            received_events.append(event_data)

        # 订阅配置更新事件
        self.event_bus.subscribe("config_updated", test_handler)

        # 发送配置更新通知
        self.event_bus.notify_config_updated("database.host", "old_host", "new_host")

        # 验证事件被接收
        self.assertEqual(len(received_events), 1)
        event = received_events[0]
        self.assertEqual(event["key"], "database.host")
        self.assertEqual(event["old_value"], "old_host")
        self.assertEqual(event["new_value"], "new_host")
        self.assertEqual(event["version"], "v1.0.0")

    def test_notify_config_error(self):
        """测试配置错误通知"""
        received_events = []

        def test_handler(event_data: Dict):
            received_events.append(event_data)

        # 订阅配置错误事件
        self.event_bus.subscribe("config_error", test_handler)

        # 发送配置错误通知
        error_details = {"source": "config.json", "line": 10}
        self.event_bus.notify_config_error("Invalid JSON format", error_details)

        # 验证事件被接收
        self.assertEqual(len(received_events), 1)
        event = received_events[0]
        self.assertEqual(event["error"], "Invalid JSON format")
        self.assertEqual(event["details"], error_details)
        self.assertEqual(event["version"], "v1.0.0")

    def test_notify_config_loaded(self):
        """测试配置加载通知"""
        received_events = []

        def test_handler(event_data: Dict):
            received_events.append(event_data)

        # 订阅配置加载事件
        self.event_bus.subscribe("config_loaded", test_handler)

        # 发送配置加载通知
        config_data = {"database": {"host": "localhost"}}
        self.event_bus.notify_config_loaded("config.json", config_data)

        # 验证事件被接收
        self.assertEqual(len(received_events), 1)
        event = received_events[0]
        self.assertEqual(event["source"], "config.json")
        self.assertEqual(event["config"], config_data)
        self.assertEqual(event["version"], "v1.0.0")

    # ==================== 事件发射测试 ====================

    def test_emit_config_changed(self):
        """测试发射配置变更事件"""
        received_events = []

        def test_handler(event_data: Dict):
            received_events.append(event_data)

        # 订阅配置变更事件
        self.event_bus.subscribe("config_changed", test_handler)

        # 发射配置变更事件
        import time
        before_time = time.time()
        self.event_bus.emit_config_changed("app.debug", False, True)
        after_time = time.time()

        # 验证事件被接收
        self.assertEqual(len(received_events), 1)
        event = received_events[0]
        self.assertEqual(event["key"], "app.debug")
        self.assertEqual(event["old_value"], False)
        self.assertEqual(event["new_value"], True)
        self.assertGreaterEqual(event["timestamp"], before_time)
        self.assertLessEqual(event["timestamp"], after_time)

    def test_emit_config_loaded(self):
        """测试发射配置加载事件"""
        received_events = []

        def test_handler(event_data: Dict):
            received_events.append(event_data)

        # 订阅配置加载事件
        self.event_bus.subscribe("config_loaded", test_handler)

        # 发射配置加载事件
        import time
        before_time = time.time()
        self.event_bus.emit_config_loaded("config.yaml")
        after_time = time.time()

        # 验证事件被接收
        self.assertEqual(len(received_events), 1)
        event = received_events[0]
        self.assertEqual(event["source"], "config.yaml")
        self.assertGreaterEqual(event["timestamp"], before_time)
        self.assertLessEqual(event["timestamp"], after_time)

    # ==================== 死信队列测试 ====================

    def test_dead_letters_management(self):
        """测试死信队列管理"""
        # 验证初始状态为空
        self.assertEqual(len(self.event_bus.get_dead_letters()), 0)

        # 添加死信
        dead_letter = {
            "event": "failed_event",
            "payload": {"test": "data"},
            "error": "Handler crashed"
        }
        self.event_bus._dead_letters.append(dead_letter)

        # 验证死信存在
        dead_letters = self.event_bus.get_dead_letters()
        self.assertEqual(len(dead_letters), 1)
        self.assertEqual(dead_letters[0], dead_letter)

        # 清空死信队列
        self.event_bus.clear_dead_letters()

        # 验证已清空
        self.assertEqual(len(self.event_bus.get_dead_letters()), 0)

    # ==================== 版本管理集成测试 ====================

    def test_version_manager_integration(self):
        """测试版本管理器集成"""
        # 设置不同的版本返回
        self.version_manager.get_latest_version.return_value = "v3.0.0"

        received_events = []

        def test_handler(event_data: Dict):
            received_events.append(event_data)

        # 订阅事件
        self.event_bus.subscribe("version_test", test_handler)

        # 发布事件
        self.event_bus.publish("version_test", {"action": "test"})

        # 验证版本信息被正确添加
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0]["version"], "v3.0.0")

    def test_version_manager_fallback(self):
        """测试版本管理器回退机制"""
        # 设置版本管理器没有最新版本方法
        delattr(self.version_manager, 'get_latest_version')

        # 设置_versions属性（注意列表顺序，最后一个是最新版本）
        self.version_manager._versions = {
            "env1": [{"id": "v1.0.0"}, {"id": "v1.1.0"}],
            "env2": [{"id": "v2.0.0"}]
        }

        received_events = []

        def test_handler(event_data: Dict):
            received_events.append(event_data)

        # 订阅事件
        self.event_bus.subscribe("fallback_test", test_handler)

        # 发布事件
        self.event_bus.publish("fallback_test", {"action": "test"})

        # 验证使用最新版本
        self.assertEqual(len(received_events), 1)
        # 根据实际实现，第一个环境的最新版本是v1.1.0
        self.assertEqual(received_events[0]["version"], "v1.1.0")  # 最新的版本

    def test_version_manager_unknown_version(self):
        """测试版本管理器未知版本"""
        # 设置版本管理器返回None
        self.version_manager.get_latest_version.return_value = None

        # 设置空的_versions
        self.version_manager._versions = {}

        received_events = []

        def test_handler(event_data: Dict):
            received_events.append(event_data)

        # 订阅事件
        self.event_bus.subscribe("unknown_version_test", test_handler)

        # 发布事件
        self.event_bus.publish("unknown_version_test", {"action": "test"})

        # 验证使用默认版本
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0]["version"], "unknown")

    # ==================== 并发测试 ====================

    def test_concurrent_event_publishing(self):
        """测试并发事件发布"""
        import threading

        received_events = []
        event_lock = threading.Lock()

        def test_handler(event_data: Dict):
            with event_lock:
                received_events.append(event_data)

        # 订阅事件
        self.event_bus.subscribe("concurrent_event", test_handler)

        def publish_worker(worker_id: int):
            """发布工作线程"""
            for i in range(10):
                self.event_bus.publish("concurrent_event", {
                    "worker": worker_id,
                    "sequence": i
                })

        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=publish_worker, args=(i,))
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证所有事件都被接收（5个线程 * 10个事件 = 50个事件）
        self.assertEqual(len(received_events), 50)

        # 验证事件数据完整性
        worker_counts = {}
        for event in received_events:
            worker_id = event["worker"]
            worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1

        # 每个worker应该有10个事件
        for worker_id in range(5):
            self.assertEqual(worker_counts[worker_id], 10)

    # ==================== 错误处理测试 ====================

    def test_publish_error_handling(self):
        """测试发布错误处理"""
        from src.infrastructure.config.config_exceptions import ConfigLoadError

        # Mock版本管理器抛出异常
        self.version_manager.get_latest_version.side_effect = Exception("Version manager failed")

        # 发布事件应该抛出ConfigLoadError
        with self.assertRaises(ConfigLoadError) as cm:
            self.event_bus.publish("error_test", {"test": "data"})

        self.assertIn("事件发布失败", str(cm.exception))

    def test_event_delivery_error_handling(self):
        """测试事件传递错误处理"""
        from src.infrastructure.config.config_exceptions import ConfigLoadError

        def failing_handler(event_data: Dict):
            raise ValueError("Handler failed")

        # 订阅失败的处理器
        self.event_bus.subscribe("delivery_error_test", failing_handler)

        # 发布事件应该抛出异常
        with self.assertRaises(ConfigLoadError) as cm:
            self.event_bus.publish("delivery_error_test", {"test": "data"})

        self.assertIn("事件处理失败", str(cm.exception))

        # 验证死信队列
        dead_letters = self.event_bus.get_dead_letters()
        self.assertEqual(len(dead_letters), 1)

    # ==================== 边界情况测试 ====================

    def test_empty_event_type(self):
        """测试空事件类型"""
        def test_handler(event_data: Dict):
            pass

        # 订阅空事件类型
        subscription_id = self.event_bus.subscribe("", test_handler)

        # 验证订阅成功
        self.assertIsNotNone(subscription_id)
        self.assertIn("", self.event_bus._subscribers)

        # 发布空事件类型
        self.event_bus.publish("", {"test": "data"})

    def test_none_payload_handling(self):
        """测试None payload处理"""
        received_events = []

        def test_handler(event_data: Dict):
            received_events.append(event_data)

        # 订阅事件
        self.event_bus.subscribe("none_payload_test", test_handler)

        # 发布None payload
        self.event_bus.publish("none_payload_test", None)

        # 验证事件被接收（payload为None）
        self.assertEqual(len(received_events), 1)
        # 注意：根据实现，None payload可能会被处理

    def test_large_payload_handling(self):
        """测试大payload处理"""
        # 创建大的payload
        large_payload = {"data": "x" * 10000}  # 10KB数据

        received_events = []

        def test_handler(event_data: Dict):
            received_events.append(event_data)

        # 订阅事件
        self.event_bus.subscribe("large_payload_test", test_handler)

        # 发布大payload
        self.event_bus.publish("large_payload_test", large_payload)

        # 验证事件被正确处理
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0]["data"], "x" * 10000)

    # ==================== EventSubscriber测试 ====================

    def test_event_subscriber_abstract_method(self):
        """测试EventSubscriber抽象方法"""
        from src.infrastructure.config.services.event_service import EventSubscriber

        subscriber = EventSubscriber(self.event_bus)

        # 测试抽象方法（应该抛出NotImplementedError）
        with self.assertRaises(NotImplementedError):
            subscriber.handle_event({"test": "data"})

    def test_event_subscriber_concrete_implementation(self):
        """测试EventSubscriber具体实现"""
        from src.infrastructure.config.services.event_service import EventSubscriber

        class ConcreteSubscriber(EventSubscriber):
            def __init__(self, event_bus):
                super().__init__(event_bus)
                self.received_events = []

            def handle_event(self, event: Dict) -> bool:
                self.received_events.append(event)
                return True

        subscriber = ConcreteSubscriber(self.event_bus)

        # 订阅事件
        self.event_bus.subscribe("subscriber_test", subscriber.handle_event)

        # 发布事件
        self.event_bus.publish("subscriber_test", {"message": "test"})

        # 验证事件被接收
        self.assertEqual(len(subscriber.received_events), 1)
        self.assertEqual(subscriber.received_events[0]["message"], "test")

    # ==================== 性能测试 ====================

    def test_event_publishing_performance(self):
        """测试事件发布性能"""
        import time

        received_count = 0

        def counting_handler(event_data: Dict):
            nonlocal received_count
            received_count += 1

        # 订阅事件
        self.event_bus.subscribe("perf_test", counting_handler)

        # 测试大量事件发布性能
        start_time = time.time()

        for i in range(1000):
            self.event_bus.publish("perf_test", {"index": i})

        end_time = time.time()
        total_time = end_time - start_time

        # 验证性能（1000个事件应该在合理时间内完成）
        self.assertLess(total_time, 2.0)  # 2秒内完成
        self.assertEqual(received_count, 1000)  # 所有事件都被处理

    # ==================== 内存泄漏测试 ====================

    def test_memory_leak_prevention(self):
        """测试内存泄漏预防"""
        # 创建大量临时订阅者
        subscription_ids = []

        for i in range(100):
            def handler(event_data: Dict):
                pass

            subscription_id = self.event_bus.subscribe(f"temp_event_{i}", handler)
            subscription_ids.append((f"temp_event_{i}", subscription_id))

        # 验证订阅者数量
        total_subscribers = sum(len(subscribers) for subscribers in self.event_bus._subscribers.values())
        self.assertEqual(total_subscribers, 100)

        # 取消所有订阅
        for event_type, subscription_id in subscription_ids:
            self.event_bus.unsubscribe(event_type, subscription_id)

        # 验证所有订阅都被清理
        total_subscribers_after = sum(len(subscribers) for subscribers in self.event_bus._subscribers.values())
        self.assertEqual(total_subscribers_after, 0)

        # 验证订阅ID映射也被清理
        total_subscription_ids = sum(len(ids) for ids in self.event_bus._subscription_ids.values())
        self.assertEqual(total_subscription_ids, 0)

    # ==================== 集成测试 ====================

    def test_complete_event_workflow(self):
        """测试完整事件工作流程"""
        workflow_events = []

        def config_updated_handler(event_data: Dict):
            workflow_events.append(("config_updated", event_data))

        def config_error_handler(event_data: Dict):
            workflow_events.append(("config_error", event_data))

        def config_loaded_handler(event_data: Dict):
            workflow_events.append(("config_loaded", event_data))

        # 订阅各种事件
        self.event_bus.subscribe("config_updated", config_updated_handler)
        self.event_bus.subscribe("config_error", config_error_handler)
        self.event_bus.subscribe("config_loaded", config_loaded_handler)

        # 执行各种通知操作
        self.event_bus.notify_config_updated("db.host", "old_host", "new_host")
        self.event_bus.notify_config_error("Invalid config", {"line": 10})
        self.event_bus.notify_config_loaded("config.json", {"loaded": True})

        # 验证所有事件都被正确处理
        self.assertEqual(len(workflow_events), 3)

        # 验证事件类型
        event_types = [event[0] for event in workflow_events]
        self.assertIn("config_updated", event_types)
        self.assertIn("config_error", event_types)
        self.assertIn("config_loaded", event_types)

        # 验证事件数据
        for event_type, event_data in workflow_events:
            self.assertIn("version", event_data)
            if event_type == "config_updated":
                self.assertEqual(event_data["key"], "db.host")
            elif event_type == "config_error":
                self.assertEqual(event_data["error"], "Invalid config")
            elif event_type == "config_loaded":
                self.assertEqual(event_data["source"], "config.json")


if __name__ == '__main__':
    unittest.main()
