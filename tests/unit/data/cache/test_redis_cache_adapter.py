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


import fnmatch

import pytest

from src.data.cache.redis_cache_adapter import RedisCacheAdapter, RedisCacheConfig


class StubRedisClient:
    def __init__(self):
        self.store = {}
        self.closed = False

    def setex(self, key, ttl, value):
        self.store[key] = value
        return True

    def set(self, key, value):
        self.store[key] = value
        return True

    def get(self, key):
        return self.store.get(key)

    def delete(self, *keys):
        count = 0
        for key in keys:
            if key in self.store:
                del self.store[key]
                count += 1
        return count

    def exists(self, key):
        return 1 if key in self.store else 0

    def mget(self, keys):
        return [self.store.get(key) for key in keys]

    def keys(self, pattern):
        return [key for key in self.store if fnmatch.fnmatch(key, pattern)]

    def info(self):
        return {"redis_version": "7.0", "used_memory_human": "1M", "connected_clients": 1, "total_commands_processed": 10}

    def ping(self):
        return True

    def pipeline(self):
        return StubPipeline(self)

    def close(self):
        self.closed = True


class StubPipeline:
    def __init__(self, client):
        self.client = client
        self.operations = []

    def setex(self, key, ttl, value):
        self.operations.append(("setex", key, ttl, value))
        return self

    def set(self, key, value):
        self.operations.append(("set", key, value))
        return self

    def execute(self):
        results = []
        for op in self.operations:
            if op[0] == "setex":
                _, key, ttl, value = op
                results.append(self.client.setex(key, ttl, value))
            elif op[0] == "set":
                _, key, value = op
                results.append(self.client.set(key, value))
        self.operations.clear()
        return results


@pytest.fixture
def adapter(monkeypatch):
    # 强制走 mock client 分支
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
    instance = RedisCacheAdapter(RedisCacheConfig(enable_compression=True, compression_threshold=1))
    instance.client = StubRedisClient()
    return instance


def test_serialization_formats_and_compression(adapter):
    adapter.config.serialization_format = "pickle"
    compressed = adapter._serialize_data({"data": "x" * 2000})
    assert compressed.startswith(b"COMPRESSED:")
    restored = adapter._deserialize_data(compressed)
    assert restored["data"].startswith("x")

    adapter.config.serialization_format = "json"
    adapter.config.enable_compression = False
    serialized = adapter._serialize_data({"value": 1})
    assert serialized.startswith(b"UNCOMPRESSED:")
    assert adapter._deserialize_data(serialized)["value"] == 1


def test_make_key_and_clear_pattern(adapter):
    adapter.set("pattern:test1", 1, ttl=5)
    adapter.set("pattern:test2", 2, ttl=5)
    deleted = adapter.clear_pattern("pattern:*")
    assert deleted >= 2


def test_mset_and_mget(adapter):
    data = {"k1": {"v": 1}, "k2": {"v": 2}}
    assert adapter.mset(data, ttl=5) is True
    result = adapter.mget(list(data.keys()))
    assert result["k1"]["v"] == 1
    assert result["k2"]["v"] == 2


def test_get_stats_and_health_check(adapter):
    stats = adapter.get_stats()
    assert "cache_stats" in stats
    assert adapter.health_check() is True


def test_close_handles_missing_method(adapter, monkeypatch):
    class DummyClient:
        def __init__(self):
            self.closed = False

    dummy = DummyClient()
    adapter.client = dummy
    adapter.close()
    assert adapter.client is dummy

