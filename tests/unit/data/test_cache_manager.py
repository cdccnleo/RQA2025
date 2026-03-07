#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据缓存管理器测试
测试数据层缓存管理器组件
"""

import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Optional, Dict

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent  # 并发测试
]

from src.data.cache.cache_manager import (
    CacheManager, CacheConfig, CacheEntry, CacheStats,
    CacheEvictionStrategy, ICacheStrategy
)


class TestCacheConfig:
    """缓存配置测试"""

    def test_cache_config_initialization(self):
        """测试缓存配置初始化"""
        config = CacheConfig(
            max_size=1000,
            ttl=300,
            enable_disk_cache=True,
            disk_cache_dir="/tmp/cache",
            cleanup_interval=60
        )

        assert config.max_size == 1000
        assert config.ttl == 300
        assert config.enable_disk_cache is True
        assert config.disk_cache_dir == "/tmp/cache"
        assert config.cleanup_interval == 60

    def test_cache_config_defaults(self):
        """测试缓存配置默认值"""
        config = CacheConfig()

        assert config.max_size == 1000
        assert config.ttl == 3600  # 1小时
        assert config.enable_disk_cache is True
        assert config.disk_cache_dir is not None
        assert config.cleanup_interval == 300  # 5分钟


class TestCacheEntry:
    """缓存条目测试"""

    def test_cache_entry_initialization(self):
        """测试缓存条目初始化"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=60
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl == 60
        assert entry.created_at is not None
        assert entry.last_accessed == entry.created_at

    def test_cache_entry_access(self):
        """测试缓存条目访问"""
        entry = CacheEntry("test_key", "test_value")

        # 记录初始访问时间
        initial_access = entry.last_accessed

        # 等待一小段时间
        time.sleep(0.01)

        # 访问条目
        entry.access()

        # 访问时间应该更新
        assert entry.last_accessed > initial_access

    def test_cache_entry_expiration(self):
        """测试缓存条目过期"""
        # 创建1秒过期的条目
        entry = CacheEntry("test_key", "test_value", ttl=1)

        # 应该未过期
        assert not entry.is_expired()

        # 等待过期
        time.sleep(1.1)

        # 应该已过期
        assert entry.is_expired()

    def test_cache_entry_no_ttl(self):
        """测试无TTL的缓存条目"""
        entry = CacheEntry("test_key", "test_value")

        # 永不过期
        time.sleep(0.1)
        assert not entry.is_expired()

    def test_cache_entry_serialization(self):
        """测试缓存条目序列化"""
        entry = CacheEntry("test_key", "test_value", ttl=60)

        # 序列化为字典
        data = entry.to_dict()

        assert data["key"] == "test_key"
        assert data["value"] == "test_value"
        assert data["ttl"] == 60
        assert "created_at" in data
        assert "last_accessed" in data

        # 从字典反序列化
        entry2 = CacheEntry.from_dict(data)

        assert entry2.key == entry.key
        assert entry2.value == entry.value
        assert entry2.ttl == entry.ttl


class TestCacheStats:
    """缓存统计测试"""

    def test_cache_stats_initialization(self):
        """测试缓存统计初始化"""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.deletes == 0
        assert stats.evictions == 0
        assert stats.errors == 0

    def test_cache_stats_operations(self):
        """测试缓存统计操作"""
        stats = CacheStats()

        # 测试命中
        stats.hit()
        assert stats.hits == 1

        # 测试未命中
        stats.miss()
        assert stats.misses == 1

        # 测试设置
        stats.set()
        assert stats.sets == 1

        # 测试删除
        stats.delete()
        assert stats.deletes == 1

        # 测试驱逐
        stats.evict()
        assert stats.evictions == 1

        # 测试错误
        stats.error()
        assert stats.errors == 1

    def test_cache_stats_getters(self):
        """测试缓存统计获取器"""
        stats = CacheStats()

        # 测试命中率
        assert stats.hit_rate == 0.0

        stats.hit()
        stats.hit()
        stats.miss()
        assert stats.hit_rate == 2.0 / 3.0

        # 测试总请求数
        assert stats.total_requests == 3

        # 测试统计信息
        info = stats.get_stats()
        assert isinstance(info, dict)
        assert "hits" in info
        assert "misses" in info
        assert "hit_rate" in info


class TestCacheManager:
    """缓存管理器测试"""

    def setup_method(self):
        """测试前准备"""
        self.config = CacheConfig(max_size=100, ttl=60)
        self.cache = CacheManager(self.config)

    def teardown_method(self):
        """测试后清理"""
        if hasattr(self, 'cache') and self.cache is not None:
            try:
                if hasattr(self.cache, 'stop'):
                    self.cache.stop()
                # 确保所有后台线程都被清理
                import gc
                gc.collect()
            except Exception as e:
                print(f"清理缓存管理器时出错: {e}")
            finally:
                self.cache = None

    def test_cache_manager_initialization(self):
        """测试缓存管理器初始化"""
        assert self.cache.config == self.config
        assert isinstance(self.cache._cache, dict)
        assert isinstance(self.cache._stats, CacheStats)
        assert self.cache._lock is not None
        assert hasattr(self.cache, 'logger')

    def test_cache_manager_initialization_with_disk_cache(self):
        """测试带磁盘缓存的缓存管理器初始化"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                max_size=100,
                enable_disk_cache=True,
                disk_cache_dir=temp_dir
            )
            cache = CacheManager(config)

            assert cache.disk_cache is not None
            cache.stop()

    def test_cache_manager_basic_operations(self):
        """测试缓存管理器基本操作"""
        # 测试设置和获取
        assert self.cache.set("key1", "value1")
        assert self.cache.get("key1") == "value1"

        # 测试不存在的键
        assert self.cache.get("nonexistent") is None

        # 测试删除
        assert self.cache.delete("key1")
        assert self.cache.get("key1") is None

    def test_cache_manager_exists(self):
        """测试缓存管理器存在性检查"""
        # 设置缓存
        self.cache.set("key1", "value1")

        # 应该存在
        assert self.cache.exists("key1") is True

        # 删除后不应该存在
        self.cache.delete("key1")
        assert self.cache.exists("key1") is False

        # 不存在的键
        assert self.cache.exists("nonexistent") is False

    def test_cache_manager_ttl(self):
        """测试缓存管理器TTL功能"""
        # 设置带TTL的缓存
        self.cache.set("ttl_key", "ttl_value", ttl=1)

        # 应该存在
        assert self.cache.get("ttl_key") == "ttl_value"

        # 等待过期
        time.sleep(1.1)

        # 应该过期
        assert self.cache.get("ttl_key") is None
        assert self.cache.exists("ttl_key") is False

    def test_cache_manager_eviction(self):
        """测试缓存管理器驱逐功能"""
        # 创建小容量缓存
        config = CacheConfig(max_size=2, ttl=3600)
        cache = CacheManager(config)

        try:
            # 添加超出容量的条目
            cache.set("key1", "value1")
            cache.set("key2", "value2")
            cache.set("key3", "value3")  # 这应该触发驱逐

            # 应该只保留2个条目
            assert len(cache._cache) == 2

            # 统计应该记录驱逐
            assert cache._stats.evictions > 0
        finally:
            cache.stop()

    def test_cache_manager_clear(self):
        """测试缓存管理器清空功能"""
        # 添加一些缓存
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")

        # 验证存在
        assert self.cache.get("key1") == "value1"
        assert len(self.cache._cache) == 2

        # 清空缓存
        self.cache.clear()

        # 应该为空
        assert len(self.cache._cache) == 0
        assert self.cache.get("key1") is None

    def test_cache_manager_stats(self):
        """测试缓存管理器统计功能"""
        # 执行一些操作
        self.cache.set("key1", "value1")
        self.cache.get("key1")  # 命中
        self.cache.get("nonexistent")  # 未命中
        self.cache.delete("key1")

        # 检查统计
        stats = self.cache.get_stats()
        assert isinstance(stats, dict)
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["deletes"] == 1

    @pytest.mark.timeout(10)
    def test_cache_manager_thread_safety(self):
        """测试缓存管理器线程安全"""
        results = []

        def worker(worker_id):
            """工作线程"""
            for i in range(10):
                key = f"key_{worker_id}_{i}"
                value = f"value_{worker_id}_{i}"

                self.cache.set(key, value)
                retrieved = self.cache.get(key)
                results.append(retrieved == value)

        # 创建多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 所有操作都应该成功
        assert all(results)

    def test_cache_manager_stop(self):
        """测试缓存管理器停止功能"""
        # 创建带清理线程的缓存
        config = CacheConfig(enable_stats=True)
        cache = CacheManager(config)

        # 验证清理线程存在
        assert cache._cleanup_thread is not None
        assert cache._cleanup_thread.is_alive()

        # 停止缓存
        cache.stop()

        # 验证清理线程停止
        assert cache._stop_cleanup is True

        # 等待线程停止
        if cache._cleanup_thread.is_alive():
            cache._cleanup_thread.join(timeout=2)

    def test_cache_manager_with_strategy(self):
        """测试带策略的缓存管理器"""
        # 创建模拟策略
        strategy = Mock(spec=ICacheStrategy)
        # 配置Mock对象的方法
        strategy.on_get = Mock()
        strategy.on_set = Mock()
        strategy.on_access = Mock()
        strategy.on_evict = Mock()
        strategy.should_evict = Mock(return_value=False)

        config = CacheConfig()
        cache = CacheManager(config, strategy)

        try:
            # 执行操作，验证策略被调用
            cache.set("test_key", "test_value")
            cache.get("test_key")  # 触发on_get

            # 策略的on_get应该被调用
            strategy.on_get.assert_called_once()
        finally:
            cache.stop()


class TestCacheManagerIntegration:
    """缓存管理器集成测试"""

    def test_cache_manager_memory_and_disk_integration(self):
        """测试内存和磁盘缓存集成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                max_size=10,
                enable_disk_cache=True,
                disk_cache_dir=temp_dir,
                ttl=3600
            )
            cache = CacheManager(config)

            try:
                # 设置缓存
                cache.set("test_key", "test_value")

                # 从内存获取
                assert cache.get("test_key") == "test_value"

                # 清除内存缓存
                cache.memory_cache.clear()

                # 应该从磁盘加载
                assert cache.get("test_key") == "test_value"
            finally:
                cache.stop()

    @pytest.mark.timeout(5)
    def test_cache_manager_expiration_cleanup(self):
        """测试过期条目清理"""
        config = CacheConfig(
            max_size=100,
            ttl=1,
            enable_stats=True
        )
        cache = CacheManager(config)

        try:
            # 添加会过期的条目
            cache.set("expire_key", "expire_value", ttl=1)

            # 等待过期
            time.sleep(1.2)

            # 手动触发清理
            cache._cleanup_expired()

            # 应该已被清理
            assert cache.get("expire_key") is None
            assert len(cache._cache) == 0
        finally:
            cache.stop()

    @pytest.mark.timeout(15)
    def test_cache_manager_concurrent_access(self):
        """测试并发访问"""
        config = CacheConfig(max_size=100)
        cache = CacheManager(config)

        results = []
        errors = []

        def concurrent_worker(worker_id):
            """并发工作线程"""
            try:
                for i in range(50):
                    key = f"concurrent_key_{worker_id}_{i}"
                    value = f"concurrent_value_{worker_id}_{i}"

                    cache.set(key, value)
                    retrieved = cache.get(key)

                    if retrieved != value:
                        errors.append(f"Worker {worker_id}: expected {value}, got {retrieved}")
                    else:
                        results.append(True)
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程
        for thread in threads:
            thread.join()

        try:
            # 验证没有错误
            assert len(errors) == 0, f"Concurrent access errors: {errors}"

            # 验证结果
            assert len(results) == 250  # 5 workers * 50 operations

            # 验证缓存状态
            assert len(cache._cache) > 0
        finally:
            cache.stop()
