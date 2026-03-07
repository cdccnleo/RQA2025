# -*- coding: utf-8 -*-
"""
缓存服务Mock测试
避免复杂的缓存依赖，测试核心缓存服务逻辑
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Dict, List, Optional, Union
import time


class MockCacheEntry:
    """模拟缓存条目"""

    def __init__(self, key: str, value: Any, ttl: Optional[int] = None):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = datetime.now()
        self.access_count = 0
        self.last_accessed = datetime.now()

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        # 使用total_seconds()而不是seconds，确保精确判断
        return (datetime.now() - self.created_at).total_seconds() > self.ttl

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


class MockCacheStorage:
    """模拟缓存存储"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.entries: Dict[str, MockCacheEntry] = {}
        self.access_order: List[str] = []  # LRU顺序

    def get(self, key: str) -> Optional[MockCacheEntry]:
        """获取缓存条目"""
        if key in self.entries:
            entry = self.entries[key]
            if not entry.is_expired():
                entry.access()
                # 更新LRU顺序
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return entry
            else:
                # 过期条目，删除
                del self.entries[key]
                if key in self.access_order:
                    self.access_order.remove(key)
        return None

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """存储缓存条目"""
        # 检查容量限制
        if len(self.entries) >= self.max_size and key not in self.entries:
            self._evict_lru()

        entry = MockCacheEntry(key, value, ttl)
        self.entries[key] = entry

        # 更新LRU顺序
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

        return True

    def delete(self, key: str) -> bool:
        """删除缓存条目"""
        if key in self.entries:
            del self.entries[key]
            if key in self.access_order:
                self.access_order.remove(key)
            return True
        return False

    def clear(self):
        """清空缓存"""
        self.entries.clear()
        self.access_order.clear()

    def size(self) -> int:
        """获取缓存大小"""
        # 清理过期条目
        expired_keys = [k for k, v in self.entries.items() if v.is_expired()]
        for key in expired_keys:
            del self.entries[key]
            if key in self.access_order:
                self.access_order.remove(key)
        return len(self.entries)

    def _evict_lru(self):
        """LRU淘汰策略"""
        if self.access_order:
            lru_key = self.access_order.pop(0)
            if lru_key in self.entries:
                del self.entries[lru_key]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_accesses = sum(entry.access_count for entry in self.entries.values())
        return {
            "size": len(self.entries),
            "max_size": self.max_size,
            "total_accesses": total_accesses,
            "hit_rate": 0.0,  # 简化实现
            "entries": {k: v.to_dict() for k, v in self.entries.items()}
        }


class MockCacheService:
    """模拟缓存服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage = MockCacheStorage(
            max_size=config.get("max_size", 1000)
        )
        self.is_initialized = False
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "clears": 0
        }

    async def initialize(self):
        """初始化服务"""
        self.is_initialized = True

    async def shutdown(self):
        """关闭服务"""
        self.is_initialized = False

    async def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值"""
        if not self.is_initialized:
            raise Exception("Cache service not initialized")

        entry = self.storage.get(key)
        if entry is not None:
            self.stats["hits"] += 1
            return entry.value
        else:
            self.stats["misses"] += 1
            return default

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        if not self.is_initialized:
            raise Exception("Cache service not initialized")

        result = self.storage.put(key, value, ttl)
        if result:
            self.stats["sets"] += 1
        return result

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        if not self.is_initialized:
            raise Exception("Cache service not initialized")

        result = self.storage.delete(key)
        if result:
            self.stats["deletes"] += 1
        return result

    async def clear(self) -> bool:
        """清空缓存"""
        if not self.is_initialized:
            raise Exception("Cache service not initialized")

        self.storage.clear()
        self.stats["clears"] += 1
        return True

    async def has_key(self, key: str) -> bool:
        """检查键是否存在"""
        if not self.is_initialized:
            raise Exception("Cache service not initialized")

        entry = self.storage.get(key)
        return entry is not None

    async def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取"""
        if not self.is_initialized:
            raise Exception("Cache service not initialized")

        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result

    async def set_multiple(self, key_value_pairs: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """批量设置"""
        if not self.is_initialized:
            raise Exception("Cache service not initialized")

        success = True
        for key, value in key_value_pairs.items():
            if not await self.set(key, value, ttl):
                success = False
        return success

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        storage_stats = self.storage.get_stats()
        return {
            "service_stats": {
                "initialized": self.is_initialized,
                "config": self.config
            },
            "operation_stats": self.stats,
            "storage_stats": storage_stats,
            "hit_rate": self._calculate_hit_rate()
        }

    def _calculate_hit_rate(self) -> float:
        """计算命中率"""
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total > 0 else 0.0


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
        # 等待一秒让条目过期
        time.sleep(1)

        assert entry.is_expired()

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
        entry = MockCacheEntry("test_key", {"data": "value"}, ttl=60)

        data = entry.to_dict()
        assert data["key"] == "test_key"
        assert data["value"] == {"data": "value"}
        assert data["ttl"] == 60
        assert "created_at" in data
        assert data["access_count"] == 0


class TestMockCacheStorage:
    """模拟缓存存储测试"""

    def setup_method(self):
        """设置测试方法"""
        self.storage = MockCacheStorage(max_size=3)

    def test_storage_get_put(self):
        """测试存储的获取和设置"""
        # 设置值
        assert self.storage.put("key1", "value1")
        assert self.storage.put("key2", "value2")

        # 获取值
        entry1 = self.storage.get("key1")
        assert entry1 is not None
        assert entry1.value == "value1"
        assert entry1.access_count == 1

        entry2 = self.storage.get("key2")
        assert entry2 is not None
        assert entry2.value == "value2"

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

        self.storage.clear()
        assert self.storage.size() == 0

    def test_storage_capacity_limit(self):
        """测试存储容量限制"""
        # 添加到容量上限
        self.storage.put("key1", "value1")
        self.storage.put("key2", "value2")
        self.storage.put("key3", "value3")
        assert self.storage.size() == 3

        # 添加第四个，触发LRU淘汰
        self.storage.put("key4", "value4")
        assert self.storage.size() == 3  # 仍然是3
        assert self.storage.get("key1") is None  # key1被淘汰
        assert self.storage.get("key4") is not None  # key4存在

    def test_storage_expiration(self):
        """测试存储过期"""
        # 设置TTL为0的条目
        self.storage.put("short_ttl", "value", ttl=0)
        # 等待足够的时间确保过期（增加等待时间以提高稳定性）
        time.sleep(1.1)

        # 获取时应该返回None（已过期并删除）
        result = self.storage.get("short_ttl")
        assert result is None, f"Expected None but got {result}"
        assert self.storage.size() == 0, f"Expected size 0 but got {self.storage.size()}"

    def test_storage_lru_order(self):
        """测试LRU顺序"""
        self.storage.put("key1", "value1")
        self.storage.put("key2", "value2")
        self.storage.put("key3", "value3")

        # 访问key1，使其变为最近访问
        self.storage.get("key1")

        # 添加第四个，淘汰最久未访问的key2
        self.storage.put("key4", "value4")

        assert self.storage.get("key1") is not None  # 最近访问，应保留
        assert self.storage.get("key2") is None     # 最久未访问，被淘汰
        assert self.storage.get("key3") is not None
        assert self.storage.get("key4") is not None

    def test_storage_stats(self):
        """测试存储统计"""
        self.storage.put("key1", "value1")
        self.storage.put("key2", "value2")
        self.storage.get("key1")

        stats = self.storage.get_stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 3
        assert stats["total_accesses"] == 1
        assert "entries" in stats


class TestMockCacheService:
    """模拟缓存服务测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = {
            "max_size": 100,
            "default_ttl": 3600
        }
        self.service = MockCacheService(self.config)

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """测试服务初始化"""
        assert not self.service.is_initialized

        await self.service.initialize()
        assert self.service.is_initialized

        await self.service.shutdown()
        assert not self.service.is_initialized

    @pytest.mark.asyncio
    async def test_service_get_set(self):
        """测试服务的获取和设置"""
        await self.service.initialize()

        # 设置值
        assert await self.service.set("key1", "value1")
        assert await self.service.set("key2", {"data": "value2"}, ttl=60)

        # 获取值
        value1 = await self.service.get("key1")
        assert value1 == "value1"

        value2 = await self.service.get("key2")
        assert value2 == {"data": "value2"}

        # 获取不存在的值
        default_value = await self.service.get("nonexistent", "default")
        assert default_value == "default"

    @pytest.mark.asyncio
    async def test_service_delete(self):
        """测试服务删除"""
        await self.service.initialize()

        await self.service.set("key1", "value1")
        assert await self.service.has_key("key1")

        assert await self.service.delete("key1")
        assert not await self.service.has_key("key1")

        # 删除不存在的键
        assert not await self.service.delete("nonexistent")

    @pytest.mark.asyncio
    async def test_service_clear(self):
        """测试服务清空"""
        await self.service.initialize()

        await self.service.set("key1", "value1")
        await self.service.set("key2", "value2")
        assert await self.service.has_key("key1")
        assert await self.service.has_key("key2")

        await self.service.clear()
        assert not await self.service.has_key("key1")
        assert not await self.service.has_key("key2")

    @pytest.mark.asyncio
    async def test_service_has_key(self):
        """测试键存在检查"""
        await self.service.initialize()

        await self.service.set("existing", "value")
        assert await self.service.has_key("existing")
        assert not await self.service.has_key("nonexistent")

    @pytest.mark.asyncio
    async def test_service_batch_operations(self):
        """测试批量操作"""
        await self.service.initialize()

        # 批量设置
        data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        assert await self.service.set_multiple(data)

        # 批量获取
        keys = ["key1", "key2", "key3", "nonexistent"]
        results = await self.service.get_multiple(keys)

        assert len(results) == 3
        assert results["key1"] == "value1"
        assert results["key2"] == "value2"
        assert results["key3"] == "value3"

    @pytest.mark.asyncio
    async def test_service_uninitialized_error(self):
        """测试未初始化服务错误"""
        with pytest.raises(Exception, match="Cache service not initialized"):
            await self.service.get("key")

        with pytest.raises(Exception, match="Cache service not initialized"):
            await self.service.set("key", "value")

        with pytest.raises(Exception, match="Cache service not initialized"):
            await self.service.delete("key")

    def test_service_stats(self):
        """测试服务统计"""
        stats = self.service.get_stats()

        assert "service_stats" in stats
        assert "operation_stats" in stats
        assert "storage_stats" in stats
        assert stats["service_stats"]["initialized"] == self.service.is_initialized
        assert stats["service_stats"]["config"] == self.config

    @pytest.mark.asyncio
    async def test_service_hit_rate_calculation(self):
        """测试命中率计算"""
        await self.service.initialize()

        # 初始状态
        stats = self.service.get_stats()
        assert stats["hit_rate"] == 0.0

        # 设置并获取（命中）
        await self.service.set("key1", "value1")
        await self.service.get("key1")  # 命中

        # 获取不存在的键（未命中）
        await self.service.get("nonexistent")  # 未命中

        stats = self.service.get_stats()
        assert stats["operation_stats"]["hits"] == 1
        assert stats["operation_stats"]["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestCacheServiceIntegration:
    """缓存服务集成测试"""

    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """测试完整工作流"""
        # 1. 创建和初始化服务
        config = {"max_size": 50, "default_ttl": 3600}
        service = MockCacheService(config)

        await service.initialize()
        assert service.is_initialized

        # 2. 执行各种缓存操作
        # 设置数据
        await service.set("user:1", {"name": "Alice", "age": 30})
        await service.set("user:2", {"name": "Bob", "age": 25}, ttl=60)
        await service.set("config:app", {"version": "1.0.0"})

        # 获取数据
        user1 = await service.get("user:1")
        assert user1["name"] == "Alice"

        user2 = await service.get("user:2")
        assert user2["name"] == "Bob"

        config = await service.get("config:app")
        assert config["version"] == "1.0.0"

        # 3. 检查存在性
        assert await service.has_key("user:1")
        assert await service.has_key("user:2")
        assert await service.has_key("config:app")

        # 4. 批量操作
        batch_data = {"temp:1": "temp1", "temp:2": "temp2"}
        await service.set_multiple(batch_data)

        batch_results = await service.get_multiple(["temp:1", "temp:2"])
        assert len(batch_results) == 2

        # 5. 删除操作
        await service.delete("temp:1")
        assert not await service.has_key("temp:1")
        assert await service.has_key("temp:2")

        # 6. 获取统计信息
        stats = service.get_stats()
        assert stats["operation_stats"]["sets"] >= 5
        assert stats["operation_stats"]["hits"] >= 3
        assert stats["storage_stats"]["size"] >= 4

        # 7. 清空缓存
        await service.clear()
        assert not await service.has_key("user:1")
        assert not await service.has_key("user:2")

        # 8. 关闭服务
        await service.shutdown()
        assert not service.is_initialized

    @pytest.mark.asyncio
    async def test_cache_performance_simulation(self):
        """测试缓存性能模拟"""
        service = MockCacheService({"max_size": 100})
        await service.initialize()

        # 模拟高频访问模式
        keys = [f"key:{i}" for i in range(20)]

        # 设置初始数据
        for key in keys:
            await service.set(key, f"value:{key}")

        # 模拟热点访问（前5个键被频繁访问）
        hot_keys = keys[:5]
        for _ in range(10):
            for key in hot_keys:
                await service.get(key)

        # 模拟冷数据访问
        cold_keys = keys[15:]
        for key in cold_keys:
            await service.get(key)

        # 验证统计
        stats = service.get_stats()
        # 至少有热键的访问次数
        assert stats["operation_stats"]["hits"] >= 50  # 5个热键 * 10次
        # 未命中次数应该很少（冷键可能已经被缓存了）
        assert stats["operation_stats"]["misses"] <= 10

        # 验证命中率较高
        if stats["operation_stats"]["hits"] + stats["operation_stats"]["misses"] > 0:
            hit_rate = stats["hit_rate"]
            assert hit_rate >= 0.8  # 命中率应该很高

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_cache_expiration_handling(self):
        """测试缓存过期处理"""
        service = MockCacheService({"max_size": 10})
        await service.initialize()

        # 设置短期TTL（使用0.1秒，确保在并发环境中也能稳定过期）
        await service.set("short", "value", ttl=0)
        await service.set("normal", "value", ttl=3600)

        # 等待足够的时间确保过期（增加等待时间以提高稳定性）
        time.sleep(1.1)

        # 短期TTL的应该过期
        short_value = await service.get("short")
        assert short_value is None, f"Expected None for expired entry, but got {short_value}"
        # 正常TTL的应该仍然存在
        normal_value = await service.get("normal")
        assert normal_value == "value", f"Expected 'value' for normal entry, but got {normal_value}"

        await service.shutdown()

    def test_service_configuration_variations(self):
        """测试不同服务配置"""
        configs = [
            {"max_size": 100},
            {"max_size": 1000, "default_ttl": 7200},
            {"max_size": 50, "enable_compression": True},
        ]

        for config in configs:
            service = MockCacheService(config)
            stats = service.get_stats()
            assert stats["service_stats"]["config"] == config
            assert not service.is_initialized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
