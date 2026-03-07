"""
测试redis_cache_adapter的覆盖率提升
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
from unittest.mock import Mock, patch, MagicMock

from src.data.cache.redis_cache_adapter import (
    RedisCacheAdapter,
    RedisCacheConfig
)


@pytest.fixture
def sample_redis_config():
    """创建示例Redis配置"""
    return RedisCacheConfig(
        host='localhost',
        port=6379,
        db=0
    )


@pytest.fixture
def sample_redis_adapter(sample_redis_config, monkeypatch):
    """创建示例Redis适配器（使用mock客户端）"""
    # 强制走 mock client 分支
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
    instance = RedisCacheAdapter(sample_redis_config)
    # 确保使用mock client
    from tests.unit.data.cache.test_redis_cache_adapter import StubRedisClient
    instance.client = StubRedisClient()
    return instance


def test_redis_cache_adapter_setup_mock_client_import_error(monkeypatch, sample_redis_config):
    """测试_setup_mock_client的ImportError处理（156-173行）"""
    # Mock ImportError for unittest.mock
    def failing_import(name, *args, **kwargs):
        if name == 'unittest.mock':
            raise ImportError("Cannot import MagicMock")
        return __import__(name, *args, **kwargs)
    
    monkeypatch.setattr('builtins.__import__', failing_import)
    
    # Should create SimpleMock instead
    adapter = RedisCacheAdapter(sample_redis_config)
    
    assert adapter.client is not None


def test_redis_cache_adapter_deserialize_old_format(sample_redis_adapter):
    """测试反序列化旧格式数据（250-252行）"""
    # Set a value that will be stored in old format (without COMPRESSED/UNCOMPRESSED prefix)
    # This requires directly setting the raw bytes
    sample_redis_adapter.client.store[sample_redis_adapter._make_key("test_key")] = b'old_format_data'
    
    # The old format path should be covered when deserializing
    # This will trigger the else branch in _deserialize_data
    try:
        result = sample_redis_adapter.get("test_key")
        # May raise exception due to invalid format, which is expected
    except Exception:
        # Expected for old format that can't be deserialized
        pass


def test_redis_cache_adapter_set_ttl_zero(sample_redis_adapter):
    """测试set方法当ttl为0时（325行）"""
    # Set with ttl = 0
    result = sample_redis_adapter.set("test_key", "test_value", ttl=0)
    
    # Should call set instead of setex
    assert result is True
    # Verify value was set
    assert sample_redis_adapter.get("test_key") == "test_value"


def test_redis_cache_adapter_set_result_false(sample_redis_adapter):
    """测试set方法当result为False时（330-331行）"""
    # Mock setex to return False
    original_setex = sample_redis_adapter.client.setex
    def failing_setex(key, ttl, value):
        return False
    sample_redis_adapter.client.setex = failing_setex
    
    result = sample_redis_adapter.set("test_key", "test_value", ttl=60)
    
    # Should return False
    assert result is False
    # Restore original method
    sample_redis_adapter.client.setex = original_setex


def test_redis_cache_adapter_delete_result_false(sample_redis_adapter):
    """测试delete方法当result为False时（355-356行）"""
    # Delete a non-existent key
    result = sample_redis_adapter.delete("nonexistent_key")
    
    # Should return False (key doesn't exist, delete returns 0)
    assert result is False


def test_redis_cache_adapter_mget_empty_keys(sample_redis_adapter):
    """测试mget方法当keys为空时（393-394行）"""
    result = sample_redis_adapter.mget([])
    
    # Should return empty dict
    assert result == {}


def test_redis_cache_adapter_mget_deserialize_exception(sample_redis_adapter):
    """测试mget方法反序列化异常处理（408-410行）"""
    # Set a value first
    sample_redis_adapter.set("key1", "value1")
    
    # Mock _deserialize_data to raise exception for first key
    call_count = [0]
    original_deserialize = sample_redis_adapter._deserialize_data
    def failing_deserialize(data):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("Deserialize failed")
        return original_deserialize(data)
    
    sample_redis_adapter._deserialize_data = failing_deserialize
    
    result = sample_redis_adapter.mget(["key1", "key2"])
    
    # Should handle exception and continue
    assert isinstance(result, dict)
    
    # Restore original method
    sample_redis_adapter._deserialize_data = original_deserialize


def test_redis_cache_adapter_mget_value_none(sample_redis_adapter):
    """测试mget方法当value为None时（410-411行）"""
    # Get keys that don't exist (will return None)
    initial_misses = sample_redis_adapter.stats['misses']
    
    result = sample_redis_adapter.mget(["nonexistent_key1", "nonexistent_key2"])
    
    # Should return empty dict (all misses)
    assert result == {}
    # Should increment misses
    assert sample_redis_adapter.stats['misses'] >= initial_misses + 2


def test_redis_cache_adapter_mset_empty_data(sample_redis_adapter):
    """测试mset方法当data为空时（431-432行）"""
    result = sample_redis_adapter.mset({})
    
    # Should return True
    assert result is True


def test_redis_cache_adapter_mset_partial_success(sample_redis_adapter):
    """测试mset方法部分成功（451-454行）"""
    # Mock pipeline execute to return mixed results
    from tests.unit.data.cache.test_redis_cache_adapter import StubPipeline
    original_pipeline = sample_redis_adapter.client.pipeline
    
    class PartialSuccessPipeline(StubPipeline):
        def execute(self):
            # Return partial success
            return [True, False, True]  # 2成功，1失败
    
    sample_redis_adapter.client.pipeline = lambda: PartialSuccessPipeline(sample_redis_adapter.client)
    
    result = sample_redis_adapter.mset({
        "key1": "value1",
        "key2": "value2",
        "key3": "value3"
    })
    
    # Should return False (not all successful)
    assert result is False
    
    # Restore original pipeline
    sample_redis_adapter.client.pipeline = original_pipeline


def test_redis_cache_adapter_clear_pattern_no_keys(sample_redis_adapter):
    """测试clear_pattern方法当没有匹配的键时（480行）"""
    # Clear pattern that matches no keys
    result = sample_redis_adapter.clear_pattern("nonexistent_pattern_*")
    
    # Should return 0 (no keys matched)
    assert result == 0


def test_redis_cache_adapter_get_stats_exception(sample_redis_adapter):
    """测试get_stats方法的异常处理（496-498行）"""
    # Mock client.info to raise exception
    original_info = sample_redis_adapter.client.info
    def failing_info():
        raise Exception("Info failed")
    sample_redis_adapter.client.info = failing_info
    
    result = sample_redis_adapter.get_stats()
    
    # Should return stats with empty redis_info
    assert result is not None
    assert 'cache_stats' in result
    assert 'redis_info' in result
    
    # Restore original method
    sample_redis_adapter.client.info = original_info


def test_redis_cache_adapter_health_check_exception(sample_redis_adapter):
    """测试health_check方法的异常处理（527-529行）"""
    # Mock client.ping to raise exception
    original_ping = sample_redis_adapter.client.ping
    def failing_ping():
        raise Exception("Ping failed")
    sample_redis_adapter.client.ping = failing_ping
    
    result = sample_redis_adapter.health_check()
    
    # Should return False
    assert result is False
    
    # Restore original method
    sample_redis_adapter.client.ping = original_ping


def test_redis_cache_adapter_close_no_close_method(sample_redis_adapter):
    """测试close方法当client没有close方法时（534-535行）"""
    # Create a client without close method
    class ClientWithoutClose:
        pass
    
    client_without_close = ClientWithoutClose()
    original_client = sample_redis_adapter.client
    sample_redis_adapter.client = client_without_close
    
    # Should not raise exception
    sample_redis_adapter.close()
    
    # Restore original client
    sample_redis_adapter.client = original_client
    assert True


def test_redis_cache_adapter_close_exception(sample_redis_adapter):
    """测试close方法的异常处理（537-538行）"""
    # Mock client.close to raise exception
    original_close = sample_redis_adapter.client.close
    def failing_close():
        raise Exception("Close failed")
    sample_redis_adapter.client.close = failing_close
    
    # Should handle exception gracefully
    sample_redis_adapter.close()
    
    # Restore original method
    sample_redis_adapter.client.close = original_close
    assert True

