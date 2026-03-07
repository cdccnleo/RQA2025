# -*- coding: utf-8 -*-
"""
数据缓存组件Mock测试
测试缓存组件的核心功能和缓存策略
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import json


class MockCacheEntry:
    """模拟缓存条目"""

    def __init__(self, key: str, value: Any, ttl: Optional[int] = None, created_at: Optional[datetime] = None):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = created_at or datetime.now()
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        age = datetime.now() - self.created_at
        return age.total_seconds() > self.ttl

    def access(self):
        """访问条目"""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "value": self.value,
            "ttl": self.ttl,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat()
        }


class MockLRUCacheStrategy:
    """模拟LRU缓存策略"""

    def __init__(self):
        self.access_order: List[str] = []

    def should_evict(self, key: str, value: Any, cache_size: int, max_size: int) -> bool:
        """判断是否应该驱逐"""
        return cache_size >= max_size

    def on_access(self, key: str, value: Any) -> None:
        """访问时的回调"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def on_evict(self, key: str, value: Any) -> None:
        """驱逐时的回调"""
        if key in self.access_order:
            self.access_order.remove(key)

    def get_evict_candidate(self) -> Optional[str]:
        """获取驱逐候选"""
        return self.access_order[0] if self.access_order else None


class MockLFUCacheStrategy:
    """模拟LFU缓存策略"""

    def __init__(self):
        self.access_counts: Dict[str, int] = {}

    def should_evict(self, key: str, value: Any, cache_size: int, max_size: int) -> bool:
        """判断是否应该驱逐"""
        return cache_size >= max_size

    def on_access(self, key: str, value: Any) -> None:
        """访问时的回调"""
        self.access_counts[key] = self.access_counts.get(key, 0) + 1

    def on_evict(self, key: str, value: Any) -> None:
        """驱逐时的回调"""
        if key in self.access_counts:
            del self.access_counts[key]

    def get_evict_candidate(self) -> Optional[str]:
        """获取驱逐候选（最少使用）"""
        if not self.access_counts:
            return None
        return min(self.access_counts.items(), key=lambda x: x[1])[0]


class MockCacheStorage:
    """模拟缓存存储"""

    def __init__(self, max_size: int = 100, strategy: Optional[Any] = None):
        self.max_size = max_size
        self.entries: Dict[str, MockCacheEntry] = {}
        self.strategy = strategy or MockLRUCacheStrategy()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key in self.entries:
            entry = self.entries[key]
            if not entry.is_expired():
                entry.access()
                self.strategy.on_access(key, entry.value)
                self.hits += 1
                return entry.value
            else:
                # 过期条目，删除
                del self.entries[key]
                self.strategy.on_evict(key, entry.value)
                self.evictions += 1

        self.misses += 1
        return None

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        # 检查容量限制
        if key not in self.entries and len(self.entries) >= self.max_size:
            self._evict()

        entry = MockCacheEntry(key, value, ttl)
        self.entries[key] = entry
        self.strategy.on_access(key, value)
        return True

    def delete(self, key: str) -> bool:
        """删除缓存值"""
        if key in self.entries:
            entry = self.entries[key]
            del self.entries[key]
            self.strategy.on_evict(key, entry.value)
            return True
        return False

    def clear(self) -> bool:
        """清空缓存"""
        for key, entry in self.entries.items():
            self.strategy.on_evict(key, entry.value)
        self.entries.clear()
        return True

    def size(self) -> int:
        """获取缓存大小"""
        # 清理过期条目
        expired_keys = [k for k, v in self.entries.items() if v.is_expired()]
        for key in expired_keys:
            entry = self.entries[key]
            del self.entries[key]
            self.strategy.on_evict(key, entry.value)
            self.evictions += 1
        return len(self.entries)

    def _evict(self):
        """执行驱逐"""
        evict_key = self.strategy.get_evict_candidate()
        if evict_key:
            if evict_key in self.entries:
                entry = self.entries[evict_key]
                del self.entries[evict_key]
                self.strategy.on_evict(evict_key, entry.value)
                self.evictions += 1
        else:
            # 如果策略没有返回候选，采用默认的FIFO策略（删除第一个）
            if self.entries:
                first_key = next(iter(self.entries.keys()))
                entry = self.entries[first_key]
                del self.entries[first_key]
                self.strategy.on_evict(first_key, entry.value)
                self.evictions += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "size": len(self.entries),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.0
        }


class MockCacheComponent:
    """模拟缓存组件"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        strategy_config = config.get("strategy")
        if isinstance(strategy_config, str):
            # 如果是字符串配置，创建对应的策略对象
            if strategy_config.lower() == "lru":
                strategy = MockLRUCacheStrategy()
            elif strategy_config.lower() == "lfu":
                strategy = MockLFUCacheStrategy()
            else:
                strategy = MockLRUCacheStrategy()
        else:
            # 如果是策略对象，直接使用
            strategy = strategy_config or MockLRUCacheStrategy()

        self.storage = MockCacheStorage(
            max_size=config.get("max_size", 100),
            strategy=strategy
        )
        self.is_initialized = False
        self.operations = []

    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        try:
            self.config.update(config)
            self.is_initialized = True
            self.operations.append("initialize")
            return True
        except Exception:
            return False

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self.is_initialized:
            raise Exception("Cache component not initialized")
        self.operations.append(f"get:{key}")
        return self.storage.get(key)

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        if not self.is_initialized:
            raise Exception("Cache component not initialized")
        self.operations.append(f"put:{key}")
        return self.storage.put(key, value, ttl)

    def delete(self, key: str) -> bool:
        """删除缓存值"""
        if not self.is_initialized:
            raise Exception("Cache component not initialized")
        self.operations.append(f"delete:{key}")
        return self.storage.delete(key)

    def clear(self) -> bool:
        """清空缓存"""
        if not self.is_initialized:
            raise Exception("Cache component not initialized")
        self.operations.append("clear")
        return self.storage.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "component_stats": {
                "initialized": self.is_initialized,
                "config": self.config,
                "operations_count": len(self.operations)
            },
            "storage_stats": self.storage.get_stats()
        }


class TestMockCacheEntry:
    """模拟缓存条目测试"""

    def test_cache_entry_creation(self):
        """测试缓存条目创建"""
        entry = MockCacheEntry("test_key", "test_value", ttl=300)

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl == 300
        assert entry.access_count == 0
        assert not entry.is_expired()

    def test_cache_entry_expiration(self):
        """测试缓存条目过期"""
        entry = MockCacheEntry("test_key", "test_value", ttl=0)
        time.sleep(0.01)  # 短暂等待

        assert entry.is_expired()

    def test_cache_entry_no_expiration(self):
        """测试缓存条目无过期时间"""
        entry = MockCacheEntry("test_key", "test_value")  # 无ttl

        # 即使时间过去也很久，也不会过期
        time.sleep(0.01)
        assert not entry.is_expired()

    def test_cache_entry_access(self):
        """测试缓存条目访问"""
        entry = MockCacheEntry("test_key", "test_value")

        assert entry.access_count == 0

        entry.access()
        assert entry.access_count == 1

        entry.access()
        assert entry.access_count == 2

    def test_cache_entry_to_dict(self):
        """测试条目序列化"""
        created_at = datetime.now()
        entry = MockCacheEntry("test_key", {"data": "value"}, ttl=60, created_at=created_at)

        data = entry.to_dict()
        assert data["key"] == "test_key"
        assert data["value"] == {"data": "value"}
        assert data["ttl"] == 60
        assert data["access_count"] == 0
        assert "created_at" in data
        assert "last_accessed" in data


class TestMockLRUCacheStrategy:
    """模拟LRU缓存策略测试"""

    def setup_method(self):
        """设置测试方法"""
        self.strategy = MockLRUCacheStrategy()

    def test_lru_initialization(self):
        """测试LRU初始化"""
        assert self.strategy.access_order == []

    def test_lru_should_evict(self):
        """测试驱逐判断"""
        # 未达到容量限制
        assert not self.strategy.should_evict("key1", "value1", 50, 100)

        # 达到容量限制
        assert self.strategy.should_evict("key1", "value1", 100, 100)

    def test_lru_on_access(self):
        """测试访问回调"""
        self.strategy.on_access("key1", "value1")
        assert self.strategy.access_order == ["key1"]

        self.strategy.on_access("key2", "value2")
        assert self.strategy.access_order == ["key1", "key2"]

        # 再次访问key1，应该移到末尾
        self.strategy.on_access("key1", "value1")
        assert self.strategy.access_order == ["key2", "key1"]

    def test_lru_on_evict(self):
        """测试驱逐回调"""
        self.strategy.on_access("key1", "value1")
        self.strategy.on_access("key2", "value2")

        self.strategy.on_evict("key1", "value1")
        assert self.strategy.access_order == ["key2"]

    def test_lru_get_evict_candidate(self):
        """测试获取驱逐候选"""
        # 空策略
        assert self.strategy.get_evict_candidate() is None

        # 添加元素
        self.strategy.on_access("key1", "value1")
        self.strategy.on_access("key2", "value2")
        self.strategy.on_access("key3", "value3")

        # 最久未使用的应该是key1
        assert self.strategy.get_evict_candidate() == "key1"


class TestMockLFUCacheStrategy:
    """模拟LFU缓存策略测试"""

    def setup_method(self):
        """设置测试方法"""
        self.strategy = MockLFUCacheStrategy()

    def test_lfu_initialization(self):
        """测试LFU初始化"""
        assert self.strategy.access_counts == {}

    def test_lfu_on_access(self):
        """测试LFU访问回调"""
        self.strategy.on_access("key1", "value1")
        assert self.strategy.access_counts["key1"] == 1

        self.strategy.on_access("key1", "value1")
        assert self.strategy.access_counts["key1"] == 2

        self.strategy.on_access("key2", "value2")
        assert self.strategy.access_counts["key2"] == 1

    def test_lfu_on_evict(self):
        """测试LFU驱逐回调"""
        self.strategy.on_access("key1", "value1")
        self.strategy.on_access("key2", "value2")

        self.strategy.on_evict("key1", "value1")
        assert "key1" not in self.strategy.access_counts
        assert self.strategy.access_counts["key2"] == 1

    def test_lfu_get_evict_candidate(self):
        """测试LFU获取驱逐候选"""
        # 空策略
        assert self.strategy.get_evict_candidate() is None

        # 添加不同访问频率的元素
        self.strategy.on_access("key1", "value1")  # 1次
        self.strategy.on_access("key2", "value2")  # 1次
        self.strategy.on_access("key1", "value1")  # 2次
        self.strategy.on_access("key1", "value1")  # 3次
        self.strategy.on_access("key3", "value3")  # 1次

        # 最少使用的应该是key2或key3（都是1次）
        candidate = self.strategy.get_evict_candidate()
        assert candidate in ["key2", "key3"]


class TestMockCacheStorage:
    """模拟缓存存储测试"""

    def setup_method(self):
        """设置测试方法"""
        self.storage = MockCacheStorage(max_size=3)

    def test_storage_initialization(self):
        """测试存储初始化"""
        assert self.storage.max_size == 3
        assert len(self.storage.entries) == 0
        assert isinstance(self.storage.strategy, MockLRUCacheStrategy)

    def test_storage_get_put(self):
        """测试存储的获取和设置"""
        # 设置值
        assert self.storage.put("key1", "value1")
        assert self.storage.put("key2", {"data": "value2"})

        assert self.storage.size() == 2

        # 获取值
        value1 = self.storage.get("key1")
        assert value1 == "value1"

        value2 = self.storage.get("key2")
        assert value2 == {"data": "value2"}

    def test_storage_delete(self):
        """测试存储删除"""
        self.storage.put("key1", "value1")
        assert self.storage.size() == 1

        assert self.storage.delete("key1")
        assert self.storage.size() == 0

        # 删除不存在的键
        assert not self.storage.delete("nonexistent")

    def test_storage_clear(self):
        """测试存储清空"""
        self.storage.put("key1", "value1")
        self.storage.put("key2", "value2")
        assert self.storage.size() == 2

        assert self.storage.clear()
        assert self.storage.size() == 0

    def test_storage_capacity_limit_lru(self):
        """测试LRU容量限制"""
        # 添加到容量上限
        self.storage.put("key1", "value1")
        self.storage.put("key2", "value2")
        self.storage.put("key3", "value3")
        assert self.storage.size() == 3

        # 添加第四个，触发LRU驱逐（最久未使用的key1被驱逐）
        self.storage.put("key4", "value4")
        assert self.storage.size() == 3
        assert self.storage.get("key1") is None  # key1被驱逐
        assert self.storage.get("key4") is not None  # key4存在

    def test_storage_capacity_limit_lfu(self):
        """测试LFU容量限制"""
        lfu_storage = MockCacheStorage(max_size=3, strategy=MockLFUCacheStrategy())

        # 添加元素并设置访问频率
        lfu_storage.put("key1", "value1")  # 访问1次
        lfu_storage.get("key1")  # 再访问1次，共2次
        lfu_storage.get("key1")  # 再访问1次，共3次

        lfu_storage.put("key2", "value2")  # 访问1次
        lfu_storage.get("key2")  # 再访问1次，共2次

        lfu_storage.put("key3", "value3")  # 访问1次

        assert lfu_storage.size() == 3

        # 添加第四个，触发LFU驱逐（最少使用的key3被驱逐）
        lfu_storage.put("key4", "value4")
        assert lfu_storage.size() == 3
        assert lfu_storage.get("key3") is None  # key3被驱逐（最少使用）
        assert lfu_storage.get("key1") is not None  # key1保留（最多使用）

    def test_storage_expiration(self):
        """测试存储过期"""
        # 设置短期TTL
        self.storage.put("short_ttl", "value", ttl=0)
        time.sleep(0.01)

        # 获取时应该返回None（已过期并删除）
        assert self.storage.get("short_ttl") is None
        assert self.storage.size() == 0

    def test_storage_miss(self):
        """测试缓存未命中"""
        assert self.storage.get("nonexistent") is None
        assert self.storage.misses == 1

    def test_storage_stats(self):
        """测试存储统计"""
        # 初始状态
        stats = self.storage.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

        # 执行操作
        self.storage.put("key1", "value1")
        self.storage.get("key1")  # 命中
        self.storage.get("key2")  # 未命中

        stats = self.storage.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestMockCacheComponent:
    """模拟缓存组件测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = {"max_size": 10, "strategy": "lru"}
        self.component = MockCacheComponent(self.config)

    def test_component_initialization(self):
        """测试组件初始化"""
        assert not self.component.is_initialized

        config = {"max_size": 100, "ttl": 3600}
        assert self.component.initialize(config)

        assert self.component.is_initialized
        assert self.component.config["max_size"] == 100
        assert self.component.config["ttl"] == 3600
        assert self.component.operations == ["initialize"]

    def test_component_operations(self):
        """测试组件操作"""
        self.component.initialize({})

        # 设置值
        assert self.component.put("key1", "value1")
        assert self.component.operations[-1] == "put:key1"

        # 获取值
        value = self.component.get("key1")
        assert value == "value1"
        assert self.component.operations[-1] == "get:key1"

        # 删除值
        assert self.component.delete("key1")
        assert self.component.operations[-1] == "delete:key1"

        # 清空
        assert self.component.clear()
        assert self.component.operations[-1] == "clear"

    def test_component_uninitialized_error(self):
        """测试未初始化错误"""
        with pytest.raises(Exception, match="Cache component not initialized"):
            self.component.get("key")

        with pytest.raises(Exception, match="Cache component not initialized"):
            self.component.put("key", "value")

    def test_component_stats(self):
        """测试组件统计"""
        self.component.initialize({})

        # 执行一些操作
        self.component.put("key1", "value1")
        self.component.get("key1")
        self.component.get("nonexistent")

        stats = self.component.get_stats()

        assert stats["component_stats"]["initialized"] is True
        assert stats["component_stats"]["operations_count"] == 4  # init + put + get + get
        assert "storage_stats" in stats
        assert stats["storage_stats"]["hits"] == 1
        assert stats["storage_stats"]["misses"] == 1


class TestCacheStrategiesComparison:
    """缓存策略对比测试"""

    def test_lru_vs_lfu_behavior(self):
        """测试LRU和LFU的不同行为"""
        # LRU存储
        lru_storage = MockCacheStorage(max_size=3, strategy=MockLRUCacheStrategy())
        lru_storage.put("key1", "value1")
        lru_storage.put("key2", "value2")
        lru_storage.put("key3", "value3")

        # 访问key1，使其变为最近使用
        lru_storage.get("key1")

        # 添加第四个，LRU会驱逐最久未使用的key2
        lru_storage.put("key4", "value4")
        assert lru_storage.get("key2") is None  # key2被驱逐
        assert lru_storage.get("key1") is not None  # key1保留

        # LFU存储
        lfu_storage = MockCacheStorage(max_size=3, strategy=MockLFUCacheStrategy())
        lfu_storage.put("key1", "value1")
        lfu_storage.put("key2", "value2")
        lfu_storage.put("key3", "value3")

        # 频繁访问key1
        lfu_storage.get("key1")
        lfu_storage.get("key1")
        lfu_storage.get("key1")  # key1访问3次

        # 较少访问key2
        lfu_storage.get("key2")  # key2访问1次

        # 添加第四个，LFU会驱逐最少使用的键（key2或key3）
        lfu_storage.put("key4", "value4")

        # 验证最常使用的key1保留
        assert lfu_storage.get("key1") is not None  # key1保留（访问最频繁）

        # 验证至少有一个最少使用的键被驱逐
        evicted_count = 0
        if lfu_storage.get("key2") is None:
            evicted_count += 1
        if lfu_storage.get("key3") is None:
            evicted_count += 1

        assert evicted_count > 0  # 至少有一个最少使用的键被驱逐


class TestCacheComponentsIntegration:
    """缓存组件集成测试"""

    def test_complete_cache_workflow(self):
        """测试完整的缓存工作流"""
        component = MockCacheComponent({"max_size": 5})

        # 1. 初始化
        assert component.initialize({"ttl": 3600})
        assert component.is_initialized

        # 2. 设置缓存数据
        test_data = [
            ("user:1", {"name": "Alice", "age": 30}),
            ("user:2", {"name": "Bob", "age": 25}),
            ("product:1", {"name": "Widget", "price": 99.99}),
            ("config:app", {"version": "1.0.0", "env": "prod"})
        ]

        for key, value in test_data:
            assert component.put(key, value)

        # 3. 验证缓存命中
        for key, expected_value in test_data:
            actual_value = component.get(key)
            assert actual_value == expected_value

        # 4. 测试缓存统计
        stats = component.get_stats()
        assert stats["storage_stats"]["hits"] == len(test_data)
        assert stats["storage_stats"]["size"] == len(test_data)

        # 5. 测试删除和清空
        assert component.delete("user:1")
        assert component.get("user:1") is None

        assert component.clear()
        assert component.get("user:2") is None

        # 6. 验证最终统计
        final_stats = component.get_stats()
        assert final_stats["storage_stats"]["size"] == 0
        assert final_stats["component_stats"]["operations_count"] > len(test_data) * 2

    def test_cache_performance_simulation(self):
        """测试缓存性能模拟"""
        component = MockCacheComponent({"max_size": 100})
        component.initialize({})

        # 模拟高频访问模式
        hot_keys = [f"hot:{i}" for i in range(10)]
        cold_keys = [f"cold:{i}" for i in range(20)]

        # 设置数据
        for key in hot_keys + cold_keys:
            component.put(key, f"value:{key}")

        # 模拟热数据高频访问
        for _ in range(5):
            for key in hot_keys:
                component.get(key)

        # 模拟冷数据低频访问
        for key in cold_keys[:5]:  # 只访问一半的冷数据
            component.get(key)

        # 访问一些不存在的键来制造misses
        for i in range(3):
            component.get(f"nonexistent:{i}")

        # 验证统计
        stats = component.get_stats()
        storage_stats = stats["storage_stats"]

        # 应该有很多命中（热数据）
        assert storage_stats["hits"] >= len(hot_keys) * 5 + 5  # 热数据 + 冷数据访问
        # 应该有一些未命中（未访问的冷数据 + 不存在的键）
        assert storage_stats["misses"] >= 3  # 不存在的键
        # 命中率应该较高
        assert storage_stats["hit_rate"] > 0.8

    def test_cache_eviction_strategies(self):
        """测试缓存驱逐策略"""
        # 测试LRU策略
        lru_component = MockCacheComponent({"max_size": 3})
        lru_component.initialize({})

        # 填充缓存
        for i in range(3):
            lru_component.put(f"key{i}", f"value{i}")

        # 访问第一个键，使其变为最近使用
        lru_component.get("key0")

        # 添加第四个，触发驱逐
        lru_component.put("key3", "value3")

        # 验证LRU行为：key1（最久未使用）被驱逐，key0保留
        assert lru_component.get("key1") is None
        assert lru_component.get("key0") is not None
        assert lru_component.get("key3") is not None

        # 测试LFU策略
        lfu_component = MockCacheComponent({"max_size": 3})
        lfu_config = {"strategy": MockLFUCacheStrategy()}
        lfu_component.storage.strategy = lfu_config["strategy"]
        lfu_component.initialize({})

        # 填充缓存并设置访问模式
        lfu_component.put("key0", "value0")
        lfu_component.put("key1", "value1")
        lfu_component.put("key2", "value2")

        # 频繁访问key0
        for _ in range(3):
            lfu_component.get("key0")

        # 较少访问key1
        lfu_component.get("key1")

        # 添加第四个，触发LFU驱逐
        lfu_component.put("key3", "value3")

        # 验证LFU行为：key2（最少使用）被驱逐，key0保留
        assert lfu_component.get("key2") is None
        assert lfu_component.get("key0") is not None
        assert lfu_component.get("key3") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
