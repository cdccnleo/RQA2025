#!/usr/bin/env python3
"""
基础设施层 - 内存缓存管理器测试

测试memory_cache_manager.py中的内存缓存管理功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import time
import threading
import pytest
from unittest.mock import patch
from src.infrastructure.cache.manager.memory_cache_manager import MemoryCacheManager


class TestMemoryCacheManager:
    """测试内存缓存管理器"""

    def setup_method(self):
        """测试前准备"""
        self.cache = MemoryCacheManager(max_size=10, ttl=60)

    def test_initialization(self):
        """测试初始化"""
        assert self.cache.max_size == 10
        assert self.cache.default_ttl == 60
        assert len(self.cache._cache) == 0
        assert self.cache._stats['hits'] == 0
        assert self.cache._stats['misses'] == 0

    def test_get_nonexistent_key(self):
        """测试获取不存在的键"""
        result = self.cache.get("nonexistent")
        assert result is None
        assert self.cache._stats['misses'] == 1

    def test_set_and_get(self):
        """测试设置和获取"""
        self.cache.set("key1", "value1")
        result = self.cache.get("key1")

        assert result == "value1"
        assert self.cache._stats['sets'] == 1
        assert self.cache._stats['hits'] == 1

    def test_set_with_custom_ttl(self):
        """测试设置自定义TTL"""
        self.cache.set("key1", "value1", ttl=30)
        result = self.cache.get("key1")

        assert result == "value1"
        # 检查过期时间是否正确设置
        entry = self.cache._cache["key1"]
        expected_expires = entry['created_at'] + 30
        assert abs(entry['expires_at'] - expected_expires) < 0.1

    def test_get_expired_key(self):
        """测试获取过期的键"""
        # 设置一个已经过期的条目
        past_time = time.time() - 10
        self.cache._cache["expired"] = {
            'value': 'expired_value',
            'expires_at': past_time,
            'created_at': past_time - 60
        }

        result = self.cache.get("expired")
        assert result is None
        assert "expired" not in self.cache._cache  # 应该被自动清理
        assert self.cache._stats['misses'] == 1

    def test_delete_existing_key(self):
        """测试删除存在的键"""
        self.cache.set("key1", "value1")
        result = self.cache.delete("key1")

        assert result is True
        assert self.cache.get("key1") is None
        assert self.cache._stats['deletes'] == 1

    def test_delete_nonexistent_key(self):
        """测试删除不存在的键"""
        result = self.cache.delete("nonexistent")
        assert result is False

    def test_clear_cache(self):
        """测试清空缓存"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        assert len(self.cache._cache) == 2

        self.cache.clear()
        assert len(self.cache._cache) == 0
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None

    def test_lru_eviction(self):
        """测试LRU淘汰策略"""
        # 填满缓存
        for i in range(10):
            self.cache.set(f"key{i}", f"value{i}")

        assert len(self.cache._cache) == 10

        # 添加第11个条目，应该淘汰最旧的
        self.cache.set("key10", "value10")
        assert len(self.cache._cache) == 10
        assert "key0" not in self.cache._cache  # 最旧的被淘汰
        assert self.cache._stats['evictions'] == 1

    def test_lru_access_order(self):
        """测试LRU访问顺序"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")

        # 访问key1，使其变为最近使用的
        self.cache.get("key1")

        # 添加新条目，应该淘汰key2（而不是key1）
        for i in range(7):  # 填满剩余空间
            self.cache.set(f"key{4+i}", f"value{4+i}")

        # 现在添加第11个，应该淘汰key2
        self.cache.set("key11", "value11")

        assert "key1" in self.cache._cache  # 最近访问的保留
        assert "key2" not in self.cache._cache  # 最旧的未访问的被淘汰
        assert "key3" in self.cache._cache  # key3仍在缓存中

    def test_cleanup_expired(self):
        """测试清理过期条目"""
        # 设置一些条目
        self.cache.set("valid", "value", ttl=60)
        self.cache.set("expired1", "value1", ttl=-1)  # 已经过期
        self.cache.set("expired2", "value2", ttl=-2)  # 已经过期

        # 手动设置过期时间为过去
        self.cache._cache["expired1"]["expires_at"] = time.time() - 10
        self.cache._cache["expired2"]["expires_at"] = time.time() - 20

        cleaned_count = self.cache.cleanup_expired()

        assert cleaned_count == 2
        assert "valid" in self.cache._cache
        assert "expired1" not in self.cache._cache
        assert "expired2" not in self.cache._cache

    def test_get_stats(self):
        """测试获取统计信息"""
        # 执行一些操作
        self.cache.set("key1", "value1")
        self.cache.get("key1")  # hit
        self.cache.get("key2")  # miss
        self.cache.delete("key1")

        stats = self.cache.get_stats()

        assert stats['size'] == 0
        assert stats['max_size'] == 10
        assert stats['hit_rate'] == 0.5  # 1 hit, 1 miss
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['sets'] == 1
        assert stats['deletes'] == 1
        assert stats['evictions'] == 0
        assert stats['default_ttl'] == 60

    def test_get_stats_empty_cache(self):
        """测试获取空缓存的统计信息"""
        stats = self.cache.get_stats()

        assert stats['size'] == 0
        assert stats['max_size'] == 10
        assert stats['hit_rate'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['sets'] == 0
        assert stats['deletes'] == 0
        assert stats['evictions'] == 0

    def test_get_all_keys(self):
        """测试获取所有键"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")

        keys = self.cache.get_all_keys()
        assert set(keys) == {"key1", "key2", "key3"}

    def test_get_all_keys_empty_cache(self):
        """测试获取空缓存的所有键"""
        keys = self.cache.get_all_keys()
        assert keys == []

    def test_has_key_existing_valid(self):
        """测试检查存在的有效键"""
        self.cache.set("key1", "value1")
        assert self.cache.has_key("key1") is True

    def test_has_key_existing_expired(self):
        """测试检查存在的过期键"""
        self.cache.set("key1", "value1", ttl=-1)  # 立即过期
        assert self.cache.has_key("key1") is False

    def test_has_key_nonexistent(self):
        """测试检查不存在的键"""
        assert self.cache.has_key("nonexistent") is False

    def test_touch_existing_key(self):
        """测试更新现有键的过期时间"""
        self.cache.set("key1", "value1", ttl=30)
        original_expires = self.cache._cache["key1"]["expires_at"]

        # 等待一小段时间
        time.sleep(0.01)

        result = self.cache.touch("key1", ttl=60)
        assert result is True

        new_expires = self.cache._cache["key1"]["expires_at"]
        assert new_expires > original_expires

    def test_touch_nonexistent_key(self):
        """测试更新不存在键的过期时间"""
        result = self.cache.touch("nonexistent")
        assert result is False

    def test_touch_with_default_ttl(self):
        """测试使用默认TTL更新过期时间"""
        self.cache.set("key1", "value1", ttl=30)
        original_expires = self.cache._cache["key1"]["expires_at"]

        time.sleep(0.01)

        result = self.cache.touch("key1")  # 不指定TTL，使用默认值
        assert result is True

        new_expires = self.cache._cache["key1"]["expires_at"]
        expected_expires = self.cache._cache["key1"]["created_at"] + self.cache.default_ttl
        assert abs(new_expires - expected_expires) < 0.1

    def test_thread_safety(self):
        """测试线程安全性"""
        results = []
        errors = []

        def worker(worker_id):
            try:
                # 每个线程执行一系列操作
                for i in range(100):
                    key = f"key_{worker_id}_{i}"
                    self.cache.set(key, f"value_{worker_id}_{i}")
                    value = self.cache.get(key)
                    if value != f"value_{worker_id}_{i}":
                        errors.append(f"Thread {worker_id}: value mismatch")
                    self.cache.delete(key)
                results.append(f"Thread {worker_id} completed")
            except Exception as e:
                errors.append(f"Thread {worker_id}: {e}")

        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 5
        assert len(errors) == 0

        # 最终缓存应该是空的（所有条目都被删除了）
        assert len(self.cache._cache) == 0

    def test_concurrent_access(self):
        """测试并发访问"""
        # 使用更大的缓存来减少淘汰干扰
        cache = MemoryCacheManager(max_size=1000, ttl=300)

        def concurrent_worker():
            for i in range(50):
                cache.set(f"concurrent_key_{threading.current_thread().ident}_{i}",
                         f"concurrent_value_{i}")
                cache.get(f"concurrent_key_{threading.current_thread().ident}_{i}")

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=concurrent_worker)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # 验证统计信息合理
        stats = cache.get_stats()
        assert stats['sets'] > 0
        assert stats['hits'] >= 0
        assert stats['size'] > 0

    def test_max_size_boundary(self):
        """测试最大大小边界"""
        small_cache = MemoryCacheManager(max_size=3, ttl=300)

        # 添加超过最大大小的条目
        for i in range(5):
            small_cache.set(f"boundary_key_{i}", f"boundary_value_{i}")

        # 应该只保留最后3个
        assert len(small_cache._cache) == 3
        assert small_cache._stats['evictions'] == 2

        # 验证保留的是最近的条目
        keys = small_cache.get_all_keys()
        expected_keys = [f"boundary_key_{i}" for i in range(2, 5)]
        assert set(keys) == set(expected_keys)
