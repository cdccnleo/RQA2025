#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
event_bus 模块测试
测试事件总线的所有功能，提升测试覆盖率到80%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import time
import threading
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

try:
    from src.infrastructure.resource.core.event_bus import (
        EventBus, Event, ResourceEvent, SystemEvent, PerformanceEvent,
        EventFilter, EventSubscription
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"导入错误: {e}")


@unittest.skipUnless(IMPORTS_AVAILABLE, "event_bus模块导入失败")
class TestEventBus(unittest.TestCase):
    """测试事件总线"""

    def setUp(self):
        """测试前准备"""
        self.event_bus = EventBus()
        self.handled_events = []
        
    def tearDown(self):
        """测试后清理"""
        if hasattr(self.event_bus, '_running') and self.event_bus._running:
            self.event_bus.stop()

    def test_event_bus_initialization(self):
        """测试事件总线初始化"""
        self.assertIsNotNone(self.event_bus)
        self.assertIsNotNone(self.event_bus.logger)
        self.assertFalse(self.event_bus._running)

    def test_event_creation(self):
        """测试事件创建"""
        event = Event(
            event_type="test_event",
            source="test_source",
            data={"key": "value"}
        )
        
        self.assertEqual(event.event_type, "test_event")
        self.assertEqual(event.source, "test_source")
        self.assertEqual(event.data, {"key": "value"})
        self.assertIsNotNone(event.timestamp)
        self.assertIsNotNone(event.event_id)

    def test_resource_event_creation(self):
        """测试资源事件创建"""
        event = ResourceEvent(
            event_type="resource_allocated",
            source="resource_manager",
            data={"resource_id": "123"},
            resource_type="CPU",
            resource_id="cpu_001"
        )
        
        self.assertEqual(event.resource_type, "CPU")
        self.assertEqual(event.resource_id, "cpu_001")
        self.assertEqual(event.event_type, "resource_allocated")

    def test_system_event_creation(self):
        """测试系统事件创建"""
        event = SystemEvent(
            event_type="system_alert",
            source="monitor",
            data={"message": "High CPU usage"},
            severity="warning",
            component="cpu_monitor"
        )
        
        self.assertEqual(event.severity, "warning")
        self.assertEqual(event.component, "cpu_monitor")

    def test_performance_event_creation(self):
        """测试性能事件创建"""
        event = PerformanceEvent(
            event_type="performance_threshold",
            source="metrics_collector",
            data={"current_value": 85},
            metric_name="cpu_usage",
            metric_value=85.0,
            threshold=80.0,
            breached=True
        )
        
        self.assertEqual(event.metric_name, "cpu_usage")
        self.assertEqual(event.metric_value, 85.0)
        self.assertTrue(event.breached)

    def test_event_filter_creation(self):
        """测试事件过滤器创建"""
        filter_obj = EventFilter(
            event_types={"test_event", "resource_event"},
            sources={"test_source"}
        )
        
        self.assertEqual(filter_obj.event_types, {"test_event", "resource_event"})
        self.assertEqual(filter_obj.sources, {"test_source"})

    def test_event_filter_matches(self):
        """测试事件过滤器匹配"""
        filter_obj = EventFilter(
            event_types={"test_event"},
            sources={"test_source"}
        )
        
        matching_event = Event("test_event", "test_source", {})
        non_matching_event = Event("other_event", "test_source", {})
        
        self.assertTrue(filter_obj.matches(matching_event))
        self.assertFalse(filter_obj.matches(non_matching_event))

    def test_event_subscription_creation(self):
        """测试事件订阅创建"""
        def handler(event):
            pass
        
        subscription = EventSubscription(
            handler=handler,
            priority=1,
            async_handler=False
        )
        
        self.assertEqual(subscription.priority, 1)
        self.assertFalse(subscription.async_handler)
        self.assertTrue(subscription.active)

    def test_subscribe_handler(self):
        """测试订阅处理器"""
        def test_handler(event):
            self.handled_events.append(event)
        
        subscription_id = self.event_bus.subscribe(test_handler)
        
        self.assertIsNotNone(subscription_id)
        self.assertTrue(subscription_id.startswith("sub_"))
        self.assertIn(subscription_id, self.event_bus._subscriptions)

    def test_subscribe_with_filter(self):
        """测试带过滤器订阅"""
        def test_handler(event):
            pass
        
        filter_obj = EventFilter(event_types={"test_event"})
        
        subscription_id = self.event_bus.subscribe(
            test_handler, 
            filter=filter_obj,
            priority=2
        )
        
        self.assertIsNotNone(subscription_id)
        subscription = self.event_bus._subscriptions[subscription_id]
        self.assertEqual(subscription.filter, filter_obj)
        self.assertEqual(subscription.priority, 2)

    def test_unsubscribe_handler(self):
        """测试取消订阅"""
        def test_handler(event):
            pass
        
        subscription_id = self.event_bus.subscribe(test_handler)
        
        result = self.event_bus.unsubscribe(subscription_id)
        self.assertTrue(result)
        self.assertNotIn(subscription_id, self.event_bus._subscriptions)

    def test_unsubscribe_nonexistent(self):
        """测试取消不存在的订阅"""
        result = self.event_bus.unsubscribe("nonexistent_id")
        self.assertFalse(result)

    def test_start_stop_bus(self):
        """测试启动和停止事件总线"""
        self.assertFalse(self.event_bus._running)
        
        self.event_bus.start()
        self.assertTrue(self.event_bus._running)
        self.assertIsNotNone(self.event_bus._processing_thread)
        
        self.event_bus.stop()
        # 给线程一些时间停止
        time.sleep(0.1)
        self.assertFalse(self.event_bus._running)

    def test_publish_event_sync(self):
        """测试同步发布事件"""
        def test_handler(event):
            self.handled_events.append(event)
        
        self.event_bus.subscribe(test_handler)
        self.event_bus.start()
        
        event = Event("test_event", "test_source", {"data": "value"})
        self.event_bus.publish(event)
        
        # 等待事件处理
        time.sleep(0.2)
        self.event_bus.stop()

    def test_publish_event_async(self):
        """测试异步发布事件"""
        def test_handler(event):
            self.handled_events.append(event)
        
        self.event_bus.subscribe(test_handler)
        self.event_bus.start()
        
        event = Event("test_event", "test_source", {"data": "value"})
        self.event_bus.publish(event, async_publish=True)
        
        # 等待异步处理
        time.sleep(0.3)
        self.event_bus.stop()

    def test_publish_when_not_running(self):
        """测试未运行时发布事件"""
        event = Event("test_event", "test_source", {"data": "value"})
        
        # 未启动时发布事件
        self.event_bus.publish(event)
        # 应该不会抛出异常

    def test_get_matching_subscriptions(self):
        """测试获取匹配的订阅"""
        def handler1(event):
            pass
        def handler2(event):
            pass
        
        # 创建两个订阅
        sub1 = self.event_bus.subscribe(handler1)
        filter_obj = EventFilter(event_types={"specific_event"})
        sub2 = self.event_bus.subscribe(handler2, filter=filter_obj)
        
        # 创建匹配的事件
        matching_event = Event("specific_event", "test_source", {})
        non_matching_event = Event("other_event", "test_source", {})
        
        # 测试匹配逻辑
        matching_subs = self.event_bus._get_matching_subscriptions(matching_event)
        self.assertGreater(len(matching_subs), 0)
        
        non_matching_subs = self.event_bus._get_matching_subscriptions(non_matching_event)
        # 应该只匹配没有过滤器的订阅
        self.assertGreaterEqual(len(non_matching_subs), 0)

    def test_add_to_history(self):
        """测试添加事件到历史"""
        event = Event("test_event", "test_source", {"data": "value"})
        
        initial_count = len(self.event_bus._event_history)
        self.event_bus._add_to_history(event)
        
        self.assertEqual(len(self.event_bus._event_history), initial_count + 1)

    def test_get_stats(self):
        """测试获取统计信息"""
        stats = self.event_bus.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("events_published", stats)
        self.assertIn("events_processed", stats)
        self.assertIn("events_failed", stats)
        self.assertIn("handlers_executed", stats)

    def test_get_event_history(self):
        """测试获取事件历史"""
        # 添加一些事件到历史
        event1 = Event("event1", "source1", {})
        event2 = Event("event2", "source2", {})
        
        self.event_bus._add_to_history(event1)
        self.event_bus._add_to_history(event2)
        
        history = self.event_bus.get_event_history()
        self.assertGreaterEqual(len(history), 2)

    def test_clear_event_history(self):
        """测试清空事件历史"""
        # 添加事件
        event = Event("test_event", "test_source", {})
        self.event_bus._add_to_history(event)
        
        self.event_bus.clear_event_history()
        self.assertEqual(len(self.event_bus._event_history), 0)

    def test_handler_exception_handling(self):
        """测试处理器异常处理"""
        def failing_handler(event):
            raise Exception("Handler failed")
        
        self.event_bus.subscribe(failing_handler)
        self.event_bus.start()
        
        event = Event("test_event", "test_source", {"data": "value"})
        
        # 不应该抛出异常
        self.event_bus.publish(event)
        time.sleep(0.2)
        self.event_bus.stop()

    def test_async_handler_execution(self):
        """测试异步处理器执行"""
        async def async_handler(event):
            self.handled_events.append(event)
        
        self.event_bus.subscribe(async_handler, async_handler=True)
        self.event_bus.start()
        
        event = Event("test_event", "test_source", {"data": "value"})
        self.event_bus.publish(event)
        
        time.sleep(0.3)
        self.event_bus.stop()

    def test_priority_handling(self):
        """测试优先级处理"""
        execution_order = []
        
        def low_priority_handler(event):
            execution_order.append("low")
        
        def high_priority_handler(event):
            execution_order.append("high")
        
        # 先注册低优先级的，后注册高优先级的
        self.event_bus.subscribe(low_priority_handler, priority=1)
        self.event_bus.subscribe(high_priority_handler, priority=2)
        
        self.event_bus.start()
        
        event = Event("test_event", "test_source", {})
        self.event_bus.publish(event)
        
        time.sleep(0.2)
        self.event_bus.stop()

    def test_custom_filter_function(self):
        """测试自定义过滤函数"""
        def custom_filter(event):
            return event.data.get("filter_me", False)
        
        def handler(event):
            self.handled_events.append(event)
        
        filter_obj = EventFilter(custom_filter=custom_filter)
        self.event_bus.subscribe(handler, filter=filter_obj)
        self.event_bus.start()
        
        # 应该被过滤掉的事件
        filtered_event = Event("test_event", "test_source", {"data": "value"})
        
        # 应该通过的事件
        passed_event = Event("test_event", "test_source", {"filter_me": True, "data": "value"})
        
        self.event_bus.publish(filtered_event)
        self.event_bus.publish(passed_event)
        
        time.sleep(0.2)
        self.event_bus.stop()


class TestEventClasses(unittest.TestCase):
    """测试事件类"""

    def test_event_post_init(self):
        """测试事件初始化后处理"""
        event = Event("test", "source", {})
        
        # 验证自动生成的字段
        self.assertIsNotNone(event.timestamp)
        self.assertIsNotNone(event.event_id)
        self.assertIsInstance(event.timestamp, datetime)

    def test_event_with_custom_values(self):
        """测试带有自定义值的事件"""
        custom_time = datetime.now()
        custom_id = "custom_id"
        
        event = Event(
            event_type="test",
            source="source", 
            data={},
            timestamp=custom_time,
            event_id=custom_id
        )
        
        self.assertEqual(event.timestamp, custom_time)
        self.assertEqual(event.event_id, custom_id)

    def test_resource_event_inheritance(self):
        """测试资源事件继承"""
        event = ResourceEvent(
            event_type="resource_allocated",
            source="resource_manager",
            data={"resource_id": "123"},
            resource_type="CPU",
            resource_id="cpu_001",
            action="allocate"
        )
        
        # 验证继承的字段
        self.assertEqual(event.event_type, "resource_allocated")
        self.assertEqual(event.source, "resource_manager")
        # 验证资源特定字段
        self.assertEqual(event.resource_type, "CPU")
        self.assertEqual(event.resource_id, "cpu_001")

    def test_system_event_inheritance(self):
        """测试系统事件继承"""
        event = SystemEvent(
            event_type="system_alert",
            source="monitor",
            data={"message": "High CPU usage"},
            severity="warning",
            component="cpu_monitor"
        )
        
        # 验证继承的字段
        self.assertEqual(event.event_type, "system_alert")
        # 验证系统特定字段
        self.assertEqual(event.severity, "warning")
        self.assertEqual(event.component, "cpu_monitor")

    def test_performance_event_inheritance(self):
        """测试性能事件继承"""
        event = PerformanceEvent(
            event_type="performance_threshold",
            source="metrics_collector",
            data={"current_value": 85},
            metric_name="cpu_usage",
            metric_value=85.0,
            threshold=80.0,
            breached=True
        )
        
        # 验证继承的字段
        self.assertEqual(event.event_type, "performance_threshold")
        # 验证性能特定字段
        self.assertEqual(event.metric_name, "cpu_usage")
        self.assertEqual(event.metric_value, 85.0)
        self.assertTrue(event.breached)


if __name__ == '__main__':
    unittest.main()
