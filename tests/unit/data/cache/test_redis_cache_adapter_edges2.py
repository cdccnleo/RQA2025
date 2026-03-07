"""
边界测试：redis_cache_adapter.py
测试边界情况和异常场景
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
import fnmatch
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from src.data.cache.redis_cache_adapter import RedisCacheAdapter, RedisCacheConfig


# 使用与 test_redis_cache_adapter.py 相同的 StubRedisClient
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


def test_redis_cache_adapter_set_no_ttl(monkeypatch):
    """测试 RedisCacheAdapter（set 方法，TTL 为 None，覆盖 325 行）"""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
    config = RedisCacheConfig()
    adapter = RedisCacheAdapter(config)
    adapter.client = StubRedisClient()
    
    # 设置 TTL 为 None，应该使用 client.set 而不是 client.setex
    # 由于 default_ttl 可能不为 None，我们需要设置 default_ttl 为 0
    adapter.config.default_ttl = 0
    result = adapter.set("test_key", "test_value", ttl=None)
    assert result is True
    # 验证数据已存储
    assert adapter.client.get(adapter._make_key("test_key")) is not None


def test_redis_cache_adapter_set_exception(monkeypatch):
    """测试 RedisCacheAdapter（set 方法，异常处理，覆盖 333-336 行）"""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
    config = RedisCacheConfig()
    adapter = RedisCacheAdapter(config)
    mock_client = MagicMock()
    mock_client.setex.side_effect = Exception("Redis connection error")
    mock_client.set.side_effect = Exception("Redis connection error")
    adapter.client = mock_client
    
    result = adapter.set("test_key", "test_value", ttl=60)
    assert result is False
    assert adapter.stats['errors'] > 0


def test_redis_cache_adapter_delete_exception(monkeypatch):
    """测试 RedisCacheAdapter（delete 方法，异常处理，覆盖 358-361 行）"""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
    config = RedisCacheConfig()
    adapter = RedisCacheAdapter(config)
    mock_client = MagicMock()
    mock_client.delete.side_effect = Exception("Redis connection error")
    adapter.client = mock_client
    
    result = adapter.delete("test_key")
    assert result is False
    assert adapter.stats['errors'] > 0


def test_redis_cache_adapter_exists_exception(monkeypatch):
    """测试 RedisCacheAdapter（exists 方法，异常处理，覆盖 379-381 行）"""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
    config = RedisCacheConfig()
    adapter = RedisCacheAdapter(config)
    mock_client = MagicMock()
    mock_client.exists.side_effect = Exception("Redis connection error")
    adapter.client = mock_client
    
    result = adapter.exists("test_key")
    assert result is False


def test_redis_cache_adapter_mget_exception(monkeypatch):
    """测试 RedisCacheAdapter（mget 方法，异常处理，覆盖 415-418 行）"""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
    config = RedisCacheConfig()
    adapter = RedisCacheAdapter(config)
    mock_client = MagicMock()
    mock_client.mget.side_effect = Exception("Redis connection error")
    adapter.client = mock_client
    
    result = adapter.mget(["key1", "key2"])
    assert result == {}
    assert adapter.stats['errors'] > 0


def test_redis_cache_adapter_mset_no_ttl(monkeypatch):
    """测试 RedisCacheAdapter（mset 方法，TTL 为 None，覆盖 447 行）"""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
    config = RedisCacheConfig()
    adapter = RedisCacheAdapter(config)
    adapter.client = StubRedisClient()
    
    # 设置 TTL 为 None，应该使用 pipeline.set 而不是 pipeline.setex
    # 由于 default_ttl 可能不为 None，我们需要设置 default_ttl 为 0
    adapter.config.default_ttl = 0
    data = {"key1": "value1", "key2": "value2"}
    result = adapter.mset(data, ttl=None)
    assert result is True
    # 验证数据已存储
    assert adapter.client.get(adapter._make_key("key1")) is not None
    assert adapter.client.get(adapter._make_key("key2")) is not None


def test_redis_cache_adapter_mset_exception(monkeypatch):
    """测试 RedisCacheAdapter（mset 方法，异常处理，覆盖 456-459 行）"""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
    config = RedisCacheConfig()
    adapter = RedisCacheAdapter(config)
    mock_client = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.execute.side_effect = Exception("Redis pipeline error")
    mock_client.pipeline.return_value = mock_pipeline
    adapter.client = mock_client
    
    data = {"key1": "value1", "key2": "value2"}
    result = adapter.mset(data, ttl=60)
    assert result is False
    assert adapter.stats['errors'] > 0


def test_redis_cache_adapter_clear_pattern_exception(monkeypatch):
    """测试 RedisCacheAdapter（clear_pattern 方法，异常处理，覆盖 482-485 行）"""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
    config = RedisCacheConfig()
    adapter = RedisCacheAdapter(config)
    mock_client = MagicMock()
    mock_client.keys.side_effect = Exception("Redis connection error")
    adapter.client = mock_client
    
    result = adapter.clear_pattern("test:*")
    assert result == 0
    assert adapter.stats['errors'] > 0


def test_redis_cache_adapter_setup_real_client_exception(monkeypatch):
    """测试 RedisCacheAdapter（_setup_real_client 方法，异常处理，覆盖 117-121 行）"""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    config = RedisCacheConfig(host="invalid_host")
    
    # 模拟 redis 导入失败或连接失败
    # 需要在 _setup_real_client 方法内部 patch redis.ConnectionPool
    import sys
    import types
    
    # 创建一个 mock redis 模块
    mock_redis_module = types.ModuleType('redis')
    mock_redis_module.ConnectionPool = MagicMock(side_effect=Exception("Connection failed"))
    
    # 临时替换 redis 模块
    original_redis = sys.modules.get('redis')
    sys.modules['redis'] = mock_redis_module
    
    try:
        with pytest.raises(Exception):
            adapter = RedisCacheAdapter(config)
    finally:
        # 恢复原始 redis 模块
        if original_redis is not None:
            sys.modules['redis'] = original_redis
        elif 'redis' in sys.modules and sys.modules['redis'] is mock_redis_module:
            del sys.modules['redis']


def test_create_redis_cache(monkeypatch):
    """测试 create_redis_cache 函数（覆盖 553 行）"""
    from src.data.cache.redis_cache_adapter import create_redis_cache
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
    config = RedisCacheConfig()
    adapter = create_redis_cache(config)
    assert isinstance(adapter, RedisCacheAdapter)


def test_redis_cache_adapter_set_ttl_zero(monkeypatch):
    """测试 RedisCacheAdapter（set 方法，TTL 为 0，应该使用 set 而不是 setex）"""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
    config = RedisCacheConfig()
    adapter = RedisCacheAdapter(config)
    adapter.client = StubRedisClient()
    
    # 设置 TTL 为 0，应该使用 client.set 而不是 client.setex
    result = adapter.set("test_key", "test_value", ttl=0)
    assert result is True
    # 验证数据已存储
    assert adapter.client.get(adapter._make_key("test_key")) is not None


def test_redis_cache_adapter_mset_ttl_zero(monkeypatch):
    """测试 RedisCacheAdapter（mset 方法，TTL 为 0，应该使用 set 而不是 setex）"""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
    config = RedisCacheConfig()
    adapter = RedisCacheAdapter(config)
    adapter.client = StubRedisClient()
    
    # 设置 TTL 为 0，应该使用 pipeline.set 而不是 pipeline.setex
    data = {"key1": "value1", "key2": "value2"}
    result = adapter.mset(data, ttl=0)
    assert result is True
    # 验证数据已存储
    assert adapter.client.get(adapter._make_key("key1")) is not None
    assert adapter.client.get(adapter._make_key("key2")) is not None

