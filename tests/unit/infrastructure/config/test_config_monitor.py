#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置监控器测试
测试配置监控、变更监听、统计报告功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.infrastructure.config.config_monitor import (
    ConfigChangeEvent,
    ConfigMonitor
)


class TestConfigChangeEvent:
    """测试配置变更事件"""

    def test_config_change_event_creation(self):
        """测试配置变更事件创建"""
        event = ConfigChangeEvent("database.host", "old_host", "new_host")

        assert event.key == "database.host"
        assert event.old_value == "old_host"
        assert event.new_value == "new_host"
        assert isinstance(event.timestamp, float)

    def test_config_change_event_with_custom_timestamp(self):
        """测试配置变更事件自定义时间戳"""
        custom_time = 1234567890.0
        event = ConfigChangeEvent("cache.ttl", 300, 600, custom_time)

        assert event.timestamp == custom_time

    def test_config_change_event_none_values(self):
        """测试配置变更事件空值处理"""
        # 添加新配置
        event = ConfigChangeEvent("new.key", None, "value")
        assert event.key == "new.key"
        assert event.old_value is None
        assert event.new_value == "value"

        # 删除配置
        event = ConfigChangeEvent("old.key", "value", None)
        assert event.key == "old.key"
        assert event.old_value == "value"
        assert event.new_value is None

    def test_config_change_event_complex_values(self):
        """测试配置变更事件复杂值"""
        old_config = {"host": "localhost", "port": 5432}
        new_config = {"host": "prod-db", "port": 5432, "ssl": True}

        event = ConfigChangeEvent("database.config", old_config, new_config)

        assert event.key == "database.config"
        assert event.old_value == old_config
        assert event.new_value == new_config


class TestConfigMonitor:
    """测试配置监控器"""

    def setup_method(self):
        """设置测试方法"""
        self.monitor = ConfigMonitor()

    def test_config_monitor_initialization(self):
        """测试配置监控器初始化"""
        assert self.monitor.listeners == []
        assert self.monitor.change_history == []
        assert self.monitor.max_history_size == 1000

    def test_add_listener(self):
        """测试添加监听器"""
        listener1 = Mock()
        listener2 = Mock()

        self.monitor.add_listener(listener1)
        assert listener1 in self.monitor.listeners
        assert len(self.monitor.listeners) == 1

        self.monitor.add_listener(listener2)
        assert listener2 in self.monitor.listeners
        assert len(self.monitor.listeners) == 2

        # 测试重复添加
        self.monitor.add_listener(listener1)
        assert len(self.monitor.listeners) == 2  # 不应该重复添加

    def test_remove_listener(self):
        """测试移除监听器"""
        listener1 = Mock()
        listener2 = Mock()

        self.monitor.add_listener(listener1)
        self.monitor.add_listener(listener2)
        assert len(self.monitor.listeners) == 2

        self.monitor.remove_listener(listener1)
        assert listener1 not in self.monitor.listeners
        assert listener2 in self.monitor.listeners
        assert len(self.monitor.listeners) == 1

        # 测试移除不存在的监听器
        self.monitor.remove_listener(listener1)
        assert len(self.monitor.listeners) == 1

    def test_record_config_change(self):
        """测试记录配置变更"""
        listener = Mock()

        self.monitor.add_listener(listener)
        self.monitor.record_config_change("test.key", "old", "new")

        # 验证历史记录
        assert len(self.monitor.change_history) == 1
        event = self.monitor.change_history[0]
        assert event.key == "test.key"
        assert event.old_value == "old"
        assert event.new_value == "new"

        # 验证监听器被调用
        listener.assert_called_once()
        called_event = listener.call_args[0][0]
        assert called_event.key == "test.key"
        assert called_event.old_value == "old"
        assert called_event.new_value == "new"

    def test_record_config_change_listener_exception(self):
        """测试记录配置变更时监听器异常"""
        listener = Mock(side_effect=Exception("Listener error"))

        self.monitor.add_listener(listener)

        # 不应该抛出异常
        self.monitor.record_config_change("test.key", "old", "new")

        # 监听器仍然被调用
        listener.assert_called_once()

        # 历史记录仍然添加
        assert len(self.monitor.change_history) == 1

    def test_record_config_change_multiple_listeners(self):
        """测试记录配置变更多个监听器"""
        listener1 = Mock()
        listener2 = Mock()
        listener3 = Mock()

        self.monitor.add_listener(listener1)
        self.monitor.add_listener(listener2)
        self.monitor.add_listener(listener3)

        self.monitor.record_config_change("test.key", "old", "new")

        # 所有监听器都被调用
        listener1.assert_called_once()
        listener2.assert_called_once()
        listener3.assert_called_once()

        # 所有监听器接收到相同的事件
        for listener in [listener1, listener2, listener3]:
            event = listener.call_args[0][0]
            assert event.key == "test.key"
            assert event.old_value == "old"
            assert event.new_value == "new"

    def test_change_history_limit(self):
        """测试变更历史限制"""
        self.monitor.max_history_size = 3

        # 添加4个变更
        for i in range(4):
            self.monitor.record_config_change(f"key{i}", f"old{i}", f"new{i}")

        # 应该只保留最近的3个
        assert len(self.monitor.change_history) == 3
        assert self.monitor.change_history[0].key == "key1"
        assert self.monitor.change_history[1].key == "key2"
        assert self.monitor.change_history[2].key == "key3"

    def test_get_recent_changes(self):
        """测试获取最近变更"""
        # 添加一些变更
        changes = []
        for i in range(5):
            self.monitor.record_config_change(f"key{i}", f"old{i}", f"new{i}")
            changes.append({
                'key': f"key{i}",
                'old_value': f"old{i}",
                'new_value': f"new{i}"
            })

        # 获取最近的3个变更
        recent_changes = self.monitor.get_recent_changes(3)
        assert len(recent_changes) == 3

        # 应该是最新的3个
        assert recent_changes[0]['key'] == "key2"
        assert recent_changes[1]['key'] == "key3"
        assert recent_changes[2]['key'] == "key4"

        # 验证数据结构
        for change in recent_changes:
            assert 'key' in change
            assert 'old_value' in change
            assert 'new_value' in change
            assert 'timestamp' in change
            assert isinstance(change['timestamp'], float)

    def test_get_recent_changes_all(self):
        """测试获取所有最近变更"""
        for i in range(3):
            self.monitor.record_config_change(f"key{i}", f"old{i}", f"new{i}")

        recent_changes = self.monitor.get_recent_changes()
        assert len(recent_changes) == 3

    def test_get_recent_changes_limit_zero(self):
        """测试获取最近变更限制为0"""
        for i in range(3):
            self.monitor.record_config_change(f"key{i}", f"old{i}", f"new{i}")

        recent_changes = self.monitor.get_recent_changes(0)
        assert len(recent_changes) == 3  # 返回所有记录

    def test_get_recent_changes_empty(self):
        """测试获取空历史记录的最近变更"""
        recent_changes = self.monitor.get_recent_changes()
        assert recent_changes == []

    def test_get_status(self):
        """测试获取监控器状态"""
        listener1 = Mock()
        listener2 = Mock()

        self.monitor.add_listener(listener1)
        self.monitor.add_listener(listener2)

        # 添加一些变更
        for i in range(5):
            self.monitor.record_config_change(f"key{i}", f"old{i}", f"new{i}")

        status = self.monitor.get_status()

        assert status['listener_count'] == 2
        assert status['change_history_size'] == 5
        assert status['max_history_size'] == 1000
        assert status['is_active'] is True

    def test_get_change_statistics_empty(self):
        """测试获取空变更统计"""
        stats = self.monitor.get_change_statistics()

        assert stats['total_changes'] == 0
        assert stats['unique_keys'] == 0
        assert stats['time_range'] is None

    def test_get_change_statistics_with_data(self):
        """测试获取变更统计"""
        # 添加一些变更
        self.monitor.record_config_change("key1", "old1", "new1")
        time.sleep(0.01)  # 确保时间戳不同
        self.monitor.record_config_change("key2", "old2", "new2")
        time.sleep(0.01)
        self.monitor.record_config_change("key1", "new1", "newer1")  # 同一个key的第二次变更

        stats = self.monitor.get_change_statistics()

        assert stats['total_changes'] == 3
        assert stats['unique_keys'] == 2  # key1和key2
        assert 'time_range' in stats
        assert 'oldest' in stats['time_range']
        assert 'newest' in stats['time_range']
        assert 'duration' in stats['time_range']

        # 时间范围应该合理
        assert stats['time_range']['oldest'] < stats['time_range']['newest']
        assert stats['time_range']['duration'] > 0

    def test_get_change_statistics_single_change(self):
        """测试获取单个变更的统计"""
        self.monitor.record_config_change("single.key", "old", "new")

        stats = self.monitor.get_change_statistics()

        assert stats['total_changes'] == 1
        assert stats['unique_keys'] == 1
        assert 'time_range' in stats

        # 单个变更的时间范围应该为0
        assert stats['time_range']['oldest'] == stats['time_range']['newest']
        assert stats['time_range']['duration'] == 0

    def test_monitor_thread_safety(self):
        """测试监控器的线程安全性"""
        import threading
        import concurrent.futures

        results = []
        errors = []

        def add_changes():
            try:
                for i in range(50):
                    self.monitor.record_config_change(f"thread_key_{i}", f"old_{i}", f"new_{i}")
                    results.append(f"change_{i}")
            except Exception as e:
                errors.append(str(e))

        def add_listeners():
            try:
                for i in range(10):
                    listener = Mock()
                    self.monitor.add_listener(listener)
                    results.append(f"listener_{i}")
            except Exception as e:
                errors.append(str(e))

        # 使用线程池执行并发操作
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _ in range(2):
                futures.append(executor.submit(add_changes))
            for _ in range(2):
                futures.append(executor.submit(add_listeners))

            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                future.result()

        # 验证没有错误发生
        assert len(errors) == 0
        assert len(results) > 0

        # 验证最终状态
        assert len(self.monitor.change_history) > 0

    def test_monitor_performance_large_history(self):
        """测试监控器处理大量历史记录的性能"""
        # 设置较小的历史大小来测试清理机制
        self.monitor.max_history_size = 100

        # 添加大量变更
        for i in range(150):
            self.monitor.record_config_change(f"perf_key_{i}", f"old_{i}", f"new_{i}")

        # 应该只保留最新的100个
        assert len(self.monitor.change_history) == 100

        # 验证保留的是最新的记录
        assert self.monitor.change_history[0].key == "perf_key_50"
        assert self.monitor.change_history[-1].key == "perf_key_149"

    def test_monitor_memory_efficiency(self):
        """测试监控器的内存效率"""
        import gc

        # 添加一些变更
        for i in range(10):
            self.monitor.record_config_change(f"mem_key_{i}", f"old_{i}", f"new_{i}")

        # 获取初始内存状态
        initial_history_size = len(self.monitor.change_history)
        initial_listeners_count = len(self.monitor.listeners)

        # 强制垃圾回收
        gc.collect()

        # 验证数据完整性
        assert len(self.monitor.change_history) == initial_history_size
        assert len(self.monitor.listeners) == initial_listeners_count

        # 验证可以正常添加新的变更
        self.monitor.record_config_change("new_key", "old", "new")
        assert len(self.monitor.change_history) == initial_history_size + 1

    def test_monitor_configuration_changes(self):
        """测试监控器配置变更"""
        # 测试修改最大历史大小
        original_max = self.monitor.max_history_size
        self.monitor.max_history_size = 500

        assert self.monitor.max_history_size == 500

        # 添加超过新限制的变更
        for i in range(600):
            self.monitor.record_config_change(f"config_key_{i}", f"old_{i}", f"new_{i}")

        # 应该只保留500个
        assert len(self.monitor.change_history) == 500

        # 恢复原始配置
        self.monitor.max_history_size = original_max
        assert self.monitor.max_history_size == original_max

    def test_monitor_listener_management(self):
        """测试监听器管理功能"""
        listeners = [Mock() for _ in range(5)]

        # 添加所有监听器
        for listener in listeners:
            self.monitor.add_listener(listener)

        assert len(self.monitor.listeners) == 5

        # 移除几个监听器
        self.monitor.remove_listener(listeners[1])
        self.monitor.remove_listener(listeners[3])

        assert len(self.monitor.listeners) == 3
        assert listeners[1] not in self.monitor.listeners
        assert listeners[3] not in self.monitor.listeners

        # 测试记录变更时只通知剩余的监听器
        self.monitor.record_config_change("test", "old", "new")

        # 移除的监听器不应该被调用
        listeners[1].assert_not_called()
        listeners[3].assert_not_called()

        # 剩余的监听器应该被调用
        for listener in [listeners[0], listeners[2], listeners[4]]:
            listener.assert_called_once()

