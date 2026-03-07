# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
测试配置监控模块

测试ConfigMonitor类的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import sys
import os
import unittest
from unittest.mock import Mock, patch, call
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.infrastructure.config.config_monitor import ConfigMonitor


class TestConfigMonitor(unittest.TestCase):
    """测试配置监控器"""

    def setUp(self):
        """测试前准备"""
        self.monitor = ConfigMonitor()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.monitor.listeners, list)
        self.assertEqual(len(self.monitor.listeners), 0)
        self.assertIsInstance(self.monitor.change_history, list)
        self.assertEqual(len(self.monitor.change_history), 0)
        self.assertEqual(self.monitor.max_history_size, 1000)

    def test_add_listener(self):
        """测试添加监听器"""
        listener = Mock()
        self.monitor.add_listener(listener)

        self.assertIn(listener, self.monitor.listeners)
        self.assertEqual(len(self.monitor.listeners), 1)

    def test_add_duplicate_listener(self):
        """测试添加重复监听器"""
        listener = Mock()
        self.monitor.add_listener(listener)
        self.monitor.add_listener(listener)  # 重复添加

        self.assertEqual(len(self.monitor.listeners), 1)  # 应该只有一个

    def test_remove_listener(self):
        """测试移除监听器"""
        listener = Mock()
        self.monitor.add_listener(listener)
        self.assertEqual(len(self.monitor.listeners), 1)

        self.monitor.remove_listener(listener)
        self.assertEqual(len(self.monitor.listeners), 0)

    def test_remove_nonexistent_listener(self):
        """测试移除不存在的监听器"""
        listener1 = Mock()
        listener2 = Mock()

        self.monitor.add_listener(listener1)
        self.monitor.remove_listener(listener2)  # 移除不存在的监听器

        self.assertEqual(len(self.monitor.listeners), 1)

    def test_record_config_change(self):
        """测试记录配置变更"""
        self.monitor.record_config_change("test.key", "old_value", "new_value")

        self.assertEqual(len(self.monitor.change_history), 1)
        event = self.monitor.change_history[0]
        self.assertEqual(event.data["key"], "test.key")
        self.assertEqual(event.data["old_value"], "old_value")
        self.assertEqual(event.data["new_value"], "new_value")

    def test_record_config_change_with_listeners(self):
        """测试记录配置变更并通知监听器"""
        listener1 = Mock()
        listener2 = Mock()
        self.monitor.add_listener(listener1)
        self.monitor.add_listener(listener2)

        self.monitor.record_config_change("test.key", "old", "new")

        # 验证监听器被调用
        listener1.assert_called_once()
        listener2.assert_called_once()

        # 验证传递的事件
        event = listener1.call_args[0][0]
        self.assertEqual(event.data["key"], "test.key")
        self.assertEqual(event.data["old_value"], "old")
        self.assertEqual(event.data["new_value"], "new")

    def test_record_config_change_listener_exception(self):
        """测试监听器异常处理"""
        listener = Mock(side_effect=Exception("Test exception"))
        self.monitor.add_listener(listener)

        # 应该不会抛出异常
        self.monitor.record_config_change("test.key", "old", "new")

        # 事件仍然应该被记录
        self.assertEqual(len(self.monitor.change_history), 1)

    def test_change_history_limit(self):
        """测试变更历史限制"""
        self.monitor.max_history_size = 2

        # 添加3个变更
        self.monitor.record_config_change("key1", "old1", "new1")
        self.monitor.record_config_change("key2", "old2", "new2")
        self.monitor.record_config_change("key3", "old3", "new3")

        # 应该只保留最新的2个
        self.assertEqual(len(self.monitor.change_history), 2)
        self.assertEqual(self.monitor.change_history[0].data["key"], "key2")
        self.assertEqual(self.monitor.change_history[1].data["key"], "key3")

    def test_get_recent_changes(self):
        """测试获取最近变更"""
        # 添加一些变更
        self.monitor.record_config_change("key1", "old1", "new1")
        self.monitor.record_config_change("key2", "old2", "new2")
        self.monitor.record_config_change("key3", "old3", "new3")

        recent_changes = self.monitor.get_recent_changes(2)

        self.assertEqual(len(recent_changes), 2)
        self.assertEqual(recent_changes[0]["key"], "key2")  # 倒数第2个
        self.assertEqual(recent_changes[1]["key"], "key3")  # 最后一个

    def test_get_recent_changes_all(self):
        """测试获取所有最近变更"""
        # 添加一些变更
        self.monitor.record_config_change("key1", "old1", "new1")
        self.monitor.record_config_change("key2", "old2", "new2")

        recent_changes = self.monitor.get_recent_changes()

        self.assertEqual(len(recent_changes), 2)

    def test_get_recent_changes_limit_zero(self):
        """测试获取最近变更限制为0（返回所有记录）"""
        self.monitor.record_config_change("key1", "old1", "new1")

        recent_changes = self.monitor.get_recent_changes(0)

        self.assertEqual(len(recent_changes), 1)  # limit=0返回所有记录

    def test_get_recent_changes_more_than_available(self):
        """测试获取的变更数量超过可用数量"""
        self.monitor.record_config_change("key1", "old1", "new1")

        recent_changes = self.monitor.get_recent_changes(5)

        self.assertEqual(len(recent_changes), 1)

    def test_get_change_statistics_with_data(self):
        """测试获取变更统计信息（有数据）"""
        self.monitor.record_config_change("key1", "old1", "new1")
        self.monitor.record_config_change("key1", "new1", "new2")  # 同一个key
        self.monitor.record_config_change("key2", "old2", "new2")

        stats = self.monitor.get_change_statistics()

        self.assertIn("total_changes", stats)
        self.assertIn("unique_keys", stats)
        self.assertIn("change_types", stats)
        self.assertIn("time_range", stats)

        self.assertEqual(stats["total_changes"], 3)
        self.assertEqual(stats["unique_keys"], 2)

    def test_get_change_statistics_single_change(self):
        """测试获取变更统计信息（单个变更）"""
        self.monitor.record_config_change("key1", "old1", "new1")

        stats = self.monitor.get_change_statistics()

        self.assertEqual(stats["total_changes"], 1)
        self.assertEqual(stats["unique_keys"], 1)

    def test_get_change_statistics_no_data(self):
        """测试获取变更统计信息（无数据）"""
        stats = self.monitor.get_change_statistics()

        self.assertEqual(stats["total_changes"], 0)
        self.assertEqual(stats["unique_keys"], 0)
        self.assertEqual(len(stats["change_types"]), 0)

    def test_clear_history(self):
        """测试清除历史记录"""
        self.monitor.record_config_change("key1", "old1", "new1")
        self.monitor.record_config_change("key2", "old2", "new2")

        self.assertEqual(len(self.monitor.change_history), 2)

        self.monitor.clear_history()

        self.assertEqual(len(self.monitor.change_history), 0)

    def test_monitor_performance_large_history(self):
        """测试大历史记录的性能"""
        # 添加大量变更
        for i in range(1500):
            self.monitor.record_config_change(f"key{i}", f"old{i}", f"new{i}")

        # 应该只保留max_history_size个
        self.assertEqual(len(self.monitor.change_history), self.monitor.max_history_size)

        # 应该能够获取统计信息
        stats = self.monitor.get_change_statistics()
        self.assertEqual(stats["total_changes"], self.monitor.max_history_size)


if __name__ == '__main__':
    unittest.main()
