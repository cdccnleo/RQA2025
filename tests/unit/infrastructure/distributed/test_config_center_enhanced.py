#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式配置中心增强测试
测试config_center模块的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

from src.infrastructure.distributed.config_center import (
    ConfigCenterManager,
    ConfigEntry,
    ConfigEvent,
    ConfigEventType
)


class TestConfigCenterManager:
    """测试配置中心管理器"""

    def setup_method(self):
        """测试前准备"""
        self.config_center = ConfigCenterManager()

    def test_config_center_initialization(self):
        """测试配置中心初始化"""
        assert isinstance(self.config_center._configs, dict)
        assert isinstance(self.config_center._listeners, dict)
        assert self.config_center.cache_enabled is True
        assert self.config_center.cache_ttl == 300

    def test_config_center_set_config_success(self):
        """测试成功设置配置"""
        key = "test.config.key"
        value = "test_value"
        
        result = self.config_center.set_config(key, value)
        
        assert result is True
        assert key in self.config_center._configs
        assert self.config_center._configs[key].value == value

    def test_config_center_get_config_success(self):
        """测试成功获取配置"""
        key = "test.config.key"
        value = "test_value"
        
        # 先设置配置
        self.config_center.set_config(key, value)
        
        # 获取配置
        result = self.config_center.get_config(key)
        
        assert result == value

    def test_config_center_get_config_default(self):
        """测试获取不存在的配置返回默认值"""
        key = "nonexistent.key"
        default_value = "default_value"
        
        result = self.config_center.get_config(key, default_value)
        
        assert result == default_value

    def test_config_center_delete_config_success(self):
        """测试成功删除配置"""
        key = "test.config.key"
        value = "test_value"
        
        # 先设置配置
        self.config_center.set_config(key, value)
        assert key in self.config_center._configs
        
        # 删除配置
        result = self.config_center.delete_config(key)
        
        assert result is True
        assert key not in self.config_center._configs

    def test_config_center_delete_config_nonexistent(self):
        """测试删除不存在的配置"""
        key = "nonexistent.key"
        
        result = self.config_center.delete_config(key)
        
        assert result is False

    def test_config_center_list_configs(self):
        """测试列出配置"""
        # 设置多个配置
        configs = {
            "config1": "value1",
            "config2": "value2",
            "prefix.config3": "value3"
        }
        
        for key, value in configs.items():
            self.config_center.set_config(key, value)
        
        # 列出所有配置
        all_configs = self.config_center.list_configs()
        assert len(all_configs) == 3
        assert all_configs["config1"] == "value1"
        assert all_configs["config2"] == "value2"
        assert all_configs["prefix.config3"] == "value3"
        
        # 按前缀列出配置
        prefixed_configs = self.config_center.list_configs("prefix")
        assert len(prefixed_configs) == 1
        assert prefixed_configs["prefix.config3"] == "value3"

    def test_config_center_watch_config(self):
        """测试监听配置变化"""
        key = "test.config.key"
        callback = Mock()
        
        # 添加监听器
        self.config_center.watch_config(key, callback)
        
        assert key in self.config_center._listeners
        assert callback in self.config_center._listeners[key]

    def test_config_center_unwatch_config(self):
        """测试取消监听配置变化"""
        key = "test.config.key"
        callback = Mock()
        
        # 先添加监听器
        self.config_center.watch_config(key, callback)
        assert callback in self.config_center._listeners[key]
        
        # 移除监听器
        self.config_center.unwatch_config(key, callback)
        assert callback not in self.config_center._listeners[key]

    def test_config_center_sync_configs(self):
        """测试同步配置"""
        remote_configs = {
            "remote.config1": "remote_value1",
            "remote.config2": "remote_value2"
        }
        
        synced_count = self.config_center.sync_configs(remote_configs)
        
        assert synced_count == 2
        assert self.config_center.get_config("remote.config1") == "remote_value1"
        assert self.config_center.get_config("remote.config2") == "remote_value2"

    def test_config_center_export_configs(self):
        """测试导出配置"""
        # 设置多个配置
        configs = {
            "export.config1": "value1",
            "export.config2": "value2"
        }
        
        for key, value in configs.items():
            self.config_center.set_config(key, value)
        
        # 导出配置
        exported = self.config_center.export_configs()
        
        assert isinstance(exported, dict)
        assert len(exported) == 2
        assert exported["export.config1"] == "value1"
        assert exported["export.config2"] == "value2"

    def test_config_center_get_config_info(self):
        """测试获取配置详细信息"""
        key = "test.config.key"
        value = "test_value"
        metadata = {"source": "test", "priority": 1}
        
        # 设置配置
        self.config_center.set_config(key, value, metadata)
        
        # 获取配置信息
        info = self.config_center.get_config_info(key)
        
        assert info is not None
        assert info["key"] == key
        assert info["value"] == value
        assert info["metadata"] == metadata
        assert "version" in info
        assert "timestamp" in info
        assert "checksum" in info

    def test_config_center_get_config_info_nonexistent(self):
        """测试获取不存在配置的详细信息"""
        key = "nonexistent.key"
        
        info = self.config_center.get_config_info(key)
        
        assert info is None

    def test_config_center_notify_listeners(self):
        """测试通知监听器"""
        key = "test.config.key"
        callback = Mock()
        event = ConfigEvent(ConfigEventType.UPDATED, key, "old_value", "new_value")
        
        # 添加监听器
        self.config_center.watch_config(key, callback)
        
        # 通知监听器
        self.config_center._notify_listeners(key, event)
        
        # 验证回调被调用
        callback.assert_called_once_with(event)

    def test_config_center_clear_expired_configs(self):
        """测试清除过期配置"""
        # 设置缓存TTL为很小的值
        self.config_center.cache_ttl = 0.001
        
        # 设置配置
        self.config_center.set_config("expired.config", "value")
        
        # 等待过期
        import time
        time.sleep(0.002)
        
        # 清除过期配置
        cleared_count = self.config_center.clear_expired_configs()
        
        assert cleared_count == 1

    def test_config_center_config_update_event(self):
        """测试配置更新事件"""
        key = "test.config.key"
        old_value = "old_value"
        new_value = "new_value"
        callback = Mock()
        
        # 添加监听器
        self.config_center.watch_config(key, callback)
        
        # 设置初始配置
        self.config_center.set_config(key, old_value)
        
        # 更新配置
        self.config_center.set_config(key, new_value)
        
        # 验证监听器被调用
        assert callback.call_count == 2  # 一次创建，一次更新

    def test_config_center_config_delete_event(self):
        """测试配置删除事件"""
        key = "test.config.key"
        value = "test_value"
        callback = Mock()
        
        # 添加监听器
        self.config_center.watch_config(key, callback)
        
        # 设置配置
        self.config_center.set_config(key, value)
        
        # 删除配置
        self.config_center.delete_config(key)
        
        # 验证监听器被调用
        assert callback.call_count == 2  # 一次创建，一次删除

    def test_config_center_sync_event(self):
        """测试配置同步事件"""
        key = "sync.config.key"
        value = "sync_value"
        callback = Mock()
        
        # 添加监听器
        self.config_center.watch_config(key, callback)
        
        # 同步配置
        self.config_center.sync_configs({key: value})
        
        # 验证监听器被调用
        assert callback.call_count == 1

    def test_config_center_error_handling(self):
        """测试错误处理"""
        key = "test.config.key"
        callback = Mock(side_effect=Exception("Callback error"))
        
        # 添加会抛出异常的监听器
        self.config_center.watch_config(key, callback)
        
        # 设置配置，应该不会因为监听器异常而失败
        result = self.config_center.set_config(key, "test_value")
        
        assert result is True  # 设置应该成功
        callback.assert_called_once()  # 回调应该被调用

    def test_config_center_config_entry(self):
        """测试配置条目"""
        key = "test.config.key"
        value = "test_value"
        version = 1
        timestamp = 1234567890.0
        checksum = "test_checksum"
        metadata = {"test": "metadata"}
        
        entry = ConfigEntry(key, value, version, timestamp, checksum, metadata)
        
        assert entry.key == key
        assert entry.value == value
        assert entry.version == version
        assert entry.timestamp == timestamp
        assert entry.checksum == checksum
        assert entry.metadata == metadata

    def test_config_center_config_event(self):
        """测试配置事件"""
        event_type = ConfigEventType.CREATED
        key = "test.config.key"
        old_value = "old_value"
        new_value = "new_value"
        
        event = ConfigEvent(event_type, key, old_value, new_value)
        
        assert event.event_type == event_type
        assert event.key == key
        assert event.old_value == old_value
        assert event.new_value == new_value
        assert isinstance(event.timestamp, float)

    def test_config_center_config_event_auto_timestamp(self):
        """测试配置事件自动时间戳"""
        event = ConfigEvent(ConfigEventType.CREATED, "test.key")
        
        assert isinstance(event.timestamp, float)
        assert event.timestamp > 0

    def test_config_center_calculate_checksum(self):
        """测试计算校验和"""
        value = {"key": "value", "number": 123}
        checksum1 = self.config_center._calculate_checksum(value)
        checksum2 = self.config_center._calculate_checksum(value)
        
        # 相同值应该产生相同校验和
        assert checksum1 == checksum2
        
        # 不同值应该产生不同校验和
        different_value = {"key": "different", "number": 456}
        checksum3 = self.config_center._calculate_checksum(different_value)
        assert checksum1 != checksum3

    def test_config_center_is_expired(self):
        """测试配置过期检查"""
        # 设置缓存TTL为很小的值
        self.config_center.cache_ttl = 0.001
        
        # 创建一个条目
        entry = ConfigEntry("test.key", "test_value", 1, 1234567890.0, "checksum", {})
        
        # 立即检查应该不过期
        assert self.config_center._is_expired(entry) is False
        
        # 修改时间戳使其过期
        entry.timestamp = 1234567890.0 - 10  # 10秒前
        assert self.config_center._is_expired(entry) is True

    def test_config_center_config_with_metadata(self):
        """测试带元数据的配置"""
        key = "test.config.key"
        value = "test_value"
        metadata = {
            "source": "test",
            "priority": 1,
            "tags": ["tag1", "tag2"]
        }
        
        # 设置带元数据的配置
        result = self.config_center.set_config(key, value, metadata)
        
        assert result is True
        
        # 获取配置信息
        info = self.config_center.get_config_info(key)
        assert info is not None
        assert info["metadata"] == metadata

    def test_config_center_config_types(self):
        """测试不同类型的配置值"""
        test_configs = {
            "string_config": "string_value",
            "int_config": 42,
            "float_config": 3.14,
            "bool_config": True,
            "list_config": [1, 2, 3],
            "dict_config": {"nested": "value"},
            "none_config": None
        }
        
        # 设置所有类型的配置
        for key, value in test_configs.items():
            result = self.config_center.set_config(key, value)
            assert result is True
        
        # 获取所有类型的配置
        for key, expected_value in test_configs.items():
            actual_value = self.config_center.get_config(key)
            assert actual_value == expected_value

    def test_config_center_concurrent_access(self):
        """测试并发访问"""
        import threading
        import time
        
        key = "concurrent.config.key"
        results = []
        
        def set_config(value):
            result = self.config_center.set_config(key, value)
            results.append(result)
        
        # 创建多个线程同时设置配置
        threads = []
        for i in range(10):
            thread = threading.Thread(target=set_config, args=(f"value_{i}",))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有操作都成功
        assert all(result is True for result in results)
        assert key in self.config_center._configs

    def test_config_center_large_config(self):
        """测试大配置值"""
        key = "large.config.key"
        # 创建一个较大的配置值
        large_value = "x" * 10000  # 10KB的字符串
        
        result = self.config_center.set_config(key, large_value)
        
        assert result is True
        assert self.config_center.get_config(key) == large_value

    def test_config_center_nested_config_updates(self):
        """测试嵌套配置更新"""
        key = "nested.config.key"
        initial_value = {"level1": {"level2": "initial"}}
        updated_value = {"level1": {"level2": "updated", "new_field": "new_value"}}
        
        # 设置初始值
        self.config_center.set_config(key, initial_value)
        assert self.config_center.get_config(key) == initial_value
        
        # 更新值
        self.config_center.set_config(key, updated_value)
        assert self.config_center.get_config(key) == updated_value

    def test_config_center_config_versioning(self):
        """测试配置版本控制"""
        key = "versioned.config.key"
        
        # 第一次设置
        self.config_center.set_config(key, "value1")
        info1 = self.config_center.get_config_info(key)
        assert info1 is not None
        version1 = info1["version"]
        
        # 更新配置
        self.config_center.set_config(key, "value2")
        info2 = self.config_center.get_config_info(key)
        assert info2 is not None
        version2 = info2["version"]
        
        # 版本应该递增
        assert version2 > version1
        assert version2 == version1 + 1

if __name__ == '__main__':
    pytest.main([__file__, "-v"])