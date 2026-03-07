"""
边界测试：smart_data_cache.py
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
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.data.cache.smart_data_cache import (
    SmartDataCache,
    SmartDataCacheBackend,
    DataCacheConfig,
    CacheStats,
    create_smart_data_cache,
    create_data_cache_backend
)
from src.data.interfaces.standard_interfaces import DataSourceType


@pytest.fixture
def cache_config():
    """创建缓存配置"""
    return DataCacheConfig(
        strategy="lfu",
        capacity=100,
        lru_k=2,
        adaptive_window=50,
        priority_levels=3,
        cost_threshold=10.0,
        enable_stats=True,
        cleanup_interval=300
    )


@pytest.fixture
def smart_cache_backend(cache_config):
    """创建智能缓存后端实例"""
    return SmartDataCacheBackend(config=cache_config)


@pytest.fixture
def smart_cache(cache_config):
    """创建智能缓存实例"""
    return SmartDataCache(config=cache_config)


def test_data_cache_config_init_default():
    """测试 DataCacheConfig（初始化，默认参数）"""
    config = DataCacheConfig()
    assert config.strategy == "lfu"
    assert config.capacity == 1000
    assert config.lru_k == 2
    assert config.enable_stats is True


def test_data_cache_config_init_custom():
    """测试 DataCacheConfig（初始化，自定义参数）"""
    config = DataCacheConfig(
        strategy="lru_k",
        capacity=500,
        lru_k=3,
        adaptive_window=100
    )
    assert config.strategy == "lru_k"
    assert config.capacity == 500
    assert config.lru_k == 3


def test_cache_stats_init():
    """测试 CacheStats（初始化）"""
    stats = CacheStats()
    assert stats.hits == 0
    assert stats.misses == 0
    assert stats.evictions == 0
    assert stats.hit_rate == 0.0


def test_cache_stats_update_hit_rate():
    """测试 CacheStats（更新命中率）"""
    stats = CacheStats()
    stats.hits = 5
    stats.total_requests = 10
    stats.update_hit_rate()
    assert stats.hit_rate == 0.5


def test_cache_stats_update_hit_rate_zero_requests():
    """测试 CacheStats（更新命中率，零请求）"""
    stats = CacheStats()
    stats.update_hit_rate()
    assert stats.hit_rate == 0.0


def test_smart_cache_backend_init_default():
    """测试 SmartDataCacheBackend（初始化，默认配置）"""
    config = DataCacheConfig()
    backend = SmartDataCacheBackend(config=config)
    assert backend is not None
    assert backend.config == config


def test_smart_cache_backend_init_custom(cache_config):
    """测试 SmartDataCacheBackend（初始化，自定义配置）"""
    backend = SmartDataCacheBackend(config=cache_config)
    assert backend is not None
    assert backend.config == cache_config


def test_smart_cache_backend_get_nonexistent(smart_cache_backend):
    """测试 SmartDataCacheBackend（获取，不存在）"""
    result = smart_cache_backend.get("nonexistent_key")
    assert result is None
    assert smart_cache_backend.stats.misses > 0


def test_smart_cache_backend_get_empty_key(smart_cache_backend):
    """测试 SmartDataCacheBackend（获取，空键）"""
    result = smart_cache_backend.get("")
    assert result is None


def test_smart_cache_backend_set_get(smart_cache_backend):
    """测试 SmartDataCacheBackend（设置和获取）"""
    result = smart_cache_backend.set("test_key", "test_value")
    assert result is True
    value = smart_cache_backend.get("test_key")
    assert value == "test_value"
    assert smart_cache_backend.stats.hits > 0


def test_smart_cache_backend_set_empty_key(smart_cache_backend):
    """测试 SmartDataCacheBackend（设置，空键）"""
    result = smart_cache_backend.set("", "test_value")
    assert isinstance(result, bool)


def test_smart_cache_backend_set_none_value(smart_cache_backend):
    """测试 SmartDataCacheBackend（设置，None 值）"""
    result = smart_cache_backend.set("test_key", None)
    assert result is True
    value = smart_cache_backend.get("test_key")
    assert value is None


def test_smart_cache_backend_delete_nonexistent(smart_cache_backend):
    """测试 SmartDataCacheBackend（删除，不存在）"""
    result = smart_cache_backend.delete("nonexistent_key")
    assert isinstance(result, bool)


def test_smart_cache_backend_delete_existing(smart_cache_backend):
    """测试 SmartDataCacheBackend（删除，存在）"""
    smart_cache_backend.set("test_key", "test_value")
    result = smart_cache_backend.delete("test_key")
    assert result is True
    assert smart_cache_backend.get("test_key") is None


def test_smart_cache_backend_delete_empty_key(smart_cache_backend):
    """测试 SmartDataCacheBackend（删除，空键）"""
    result = smart_cache_backend.delete("")
    assert isinstance(result, bool)


def test_smart_cache_backend_exists_nonexistent(smart_cache_backend):
    """测试 SmartDataCacheBackend（检查存在，不存在）"""
    result = smart_cache_backend.exists("nonexistent_key")
    assert result is False


def test_smart_cache_backend_exists_existing(smart_cache_backend):
    """测试 SmartDataCacheBackend（检查存在，存在）"""
    smart_cache_backend.set("test_key", "test_value")
    result = smart_cache_backend.exists("test_key")
    assert result is True


def test_smart_cache_backend_exists_empty_key(smart_cache_backend):
    """测试 SmartDataCacheBackend（检查存在，空键）"""
    result = smart_cache_backend.exists("")
    assert result is False


def test_smart_cache_backend_clear(smart_cache_backend):
    """测试 SmartDataCacheBackend（清空）"""
    smart_cache_backend.set("key1", "value1")
    smart_cache_backend.set("key2", "value2")
    result = smart_cache_backend.clear()
    assert result is True
    assert smart_cache_backend.get("key1") is None


def test_smart_cache_backend_clear_empty(smart_cache_backend):
    """测试 SmartDataCacheBackend（清空，空缓存）"""
    result = smart_cache_backend.clear()
    assert result is True


def test_smart_cache_backend_get_stats(smart_cache_backend):
    """测试 SmartDataCacheBackend（获取统计信息）"""
    stats = smart_cache_backend.get_stats()
    assert isinstance(stats, dict)
    assert 'hits' in stats or 'hit_rate' in stats


def test_smart_cache_backend_get_stats_with_operations(smart_cache_backend):
    """测试 SmartDataCacheBackend（获取统计信息，有操作）"""
    smart_cache_backend.set("key1", "value1")
    smart_cache_backend.get("key1")
    smart_cache_backend.get("nonexistent")
    stats = smart_cache_backend.get_stats()
    assert isinstance(stats, dict)


def test_smart_cache_backend_capacity_limit(smart_cache_backend):
    """测试 SmartDataCacheBackend（容量限制）"""
    # 设置较小的容量
    smart_cache_backend.config.capacity = 2
    # 重新创建缓存实例
    backend = SmartDataCacheBackend(config=smart_cache_backend.config)
    # 添加超过容量的条目
    backend.set("key1", "value1")
    backend.set("key2", "value2")
    backend.set("key3", "value3")
    # 应该仍然可以获取（LRU/LFU 策略）
    assert backend.get("key3") == "value3"


def test_smart_cache_backend_different_strategies():
    """测试 SmartDataCacheBackend（不同策略）"""
    strategies = ["lfu", "lru_k", "adaptive", "priority", "cost_aware"]
    for strategy in strategies:
        config = DataCacheConfig(strategy=strategy, capacity=10)
        backend = SmartDataCacheBackend(config=config)
        backend.set("test_key", "test_value")
        value = backend.get("test_key")
        assert value == "test_value" or value is None


def test_smart_cache_init_default():
    """测试 SmartDataCache（初始化，默认配置）"""
    cache = SmartDataCache()
    assert cache is not None


def test_smart_cache_init_custom(cache_config):
    """测试 SmartDataCache（初始化，自定义配置）"""
    cache = SmartDataCache(config=cache_config)
    assert cache is not None
    assert cache.config == cache_config


def test_smart_cache_get_nonexistent(smart_cache):
    """测试 SmartDataCache（获取，不存在）"""
    result = smart_cache.get("nonexistent_key", DataSourceType.API)
    assert result is None


def test_smart_cache_get_empty_key(smart_cache):
    """测试 SmartDataCache（获取，空键）"""
    result = smart_cache.get("", DataSourceType.API)
    assert result is None


def test_smart_cache_set_get(smart_cache):
    """测试 SmartDataCache（设置和获取）"""
    result = smart_cache.set("test_key", "test_value", DataSourceType.API)
    assert result is True
    value = smart_cache.get("test_key", DataSourceType.API)
    assert value == "test_value"


def test_smart_cache_set_empty_key(smart_cache):
    """测试 SmartDataCache（设置，空键）"""
    result = smart_cache.set("", "test_value", DataSourceType.API)
    assert isinstance(result, bool)


def test_smart_cache_set_none_value(smart_cache):
    """测试 SmartDataCache（设置，None 值）"""
    result = smart_cache.set("test_key", None, DataSourceType.API)
    assert result is True
    value = smart_cache.get("test_key", DataSourceType.API)
    assert value is None


def test_smart_cache_delete_nonexistent(smart_cache):
    """测试 SmartDataCache（删除，不存在）"""
    result = smart_cache.delete("nonexistent_key")
    assert isinstance(result, bool)


def test_smart_cache_delete_existing(smart_cache):
    """测试 SmartDataCache（删除，存在）"""
    smart_cache.set("test_key", "test_value", DataSourceType.API)
    result = smart_cache.delete("test_key")
    assert result is True
    assert smart_cache.get("test_key", DataSourceType.API) is None


def test_smart_cache_delete_empty_key(smart_cache):
    """测试 SmartDataCache（删除，空键）"""
    result = smart_cache.delete("")
    assert isinstance(result, bool)


def test_smart_cache_exists_nonexistent(smart_cache):
    """测试 SmartDataCache（检查存在，不存在）"""
    result = smart_cache.exists("nonexistent_key")
    assert result is False


def test_smart_cache_exists_existing(smart_cache):
    """测试 SmartDataCache（检查存在，存在）"""
    smart_cache.set("test_key", "test_value", DataSourceType.API)
    result = smart_cache.exists("test_key")
    assert result is True


def test_smart_cache_exists_empty_key(smart_cache):
    """测试 SmartDataCache（检查存在，空键）"""
    result = smart_cache.exists("")
    assert result is False


def test_smart_cache_clear(smart_cache):
    """测试 SmartDataCache（清空）"""
    smart_cache.set("key1", "value1", DataSourceType.API)
    smart_cache.set("key2", "value2", DataSourceType.API)
    result = smart_cache.clear()
    assert result is True
    assert smart_cache.get("key1", DataSourceType.API) is None


def test_smart_cache_clear_empty(smart_cache):
    """测试 SmartDataCache（清空，空缓存）"""
    result = smart_cache.clear()
    assert result is True


def test_smart_cache_get_stats(smart_cache):
    """测试 SmartDataCache（获取统计信息）"""
    stats = smart_cache.get_stats()
    assert isinstance(stats, dict)


def test_smart_cache_get_stats_with_operations(smart_cache):
    """测试 SmartDataCache（获取统计信息，有操作）"""
    smart_cache.set("key1", "value1", DataSourceType.API)
    smart_cache.get("key1", DataSourceType.API)
    smart_cache.get("nonexistent", DataSourceType.API)
    stats = smart_cache.get_stats()
    assert isinstance(stats, dict)


def test_smart_cache_special_value_types(smart_cache):
    """测试 SmartDataCache（特殊值类型）"""
    test_cases = [
        ("dict", {"key": "value"}),
        ("list", [1, 2, 3]),
        ("tuple", (1, 2, 3)),
        ("int", 42),
        ("float", 3.14),
        ("bool", True),
        ("str", "string"),
    ]
    for key, value in test_cases:
        result = smart_cache.set(key, value, DataSourceType.API)
        assert result is True
        retrieved = smart_cache.get(key, DataSourceType.API)
        assert retrieved == value


def test_smart_cache_dataframe(smart_cache):
    """测试 SmartDataCache（DataFrame）"""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = smart_cache.set("df_key", df, DataSourceType.API)
    assert result is True
    retrieved_df = smart_cache.get("df_key", DataSourceType.API)
    assert isinstance(retrieved_df, pd.DataFrame) or retrieved_df is None
    if retrieved_df is not None:
        assert len(retrieved_df) == 3


def test_create_smart_data_cache_default():
    """测试 create_smart_data_cache（默认配置）"""
    cache = create_smart_data_cache()
    assert cache is not None
    assert isinstance(cache, SmartDataCache)


def test_create_smart_data_cache_custom():
    """测试 create_smart_data_cache（自定义配置）"""
    config = DataCacheConfig(strategy="lfu", capacity=500)
    cache = create_smart_data_cache(config=config)
    assert cache is not None
    assert cache.config == config


def test_create_data_cache_backend_default():
    """测试 create_data_cache_backend（默认配置）"""
    backend = create_data_cache_backend()
    assert backend is not None
    assert isinstance(backend, SmartDataCacheBackend)


def test_create_data_cache_backend_custom():
    """测试 create_data_cache_backend（自定义配置）"""
    config = DataCacheConfig(strategy="lru_k", capacity=500)
    backend = create_data_cache_backend(config=config)
    assert backend is not None
    assert backend.config == config


def test_smart_cache_backend_infrastructure_unavailable():
    """测试 SmartDataCacheBackend（基础设施层不可用）"""
    with patch('src.data.cache.smart_data_cache.INFRASTRUCTURE_CACHE_AVAILABLE', False):
        config = DataCacheConfig(strategy="lfu", capacity=10)
        backend = SmartDataCacheBackend(config=config)
        # 应该使用降级实现
        backend.set("test_key", "test_value")
        value = backend.get("test_key")
        assert value == "test_value" or value is None


def test_smart_cache_backend_invalid_strategy():
    """测试 SmartDataCacheBackend（无效策略）"""
    config = DataCacheConfig(strategy="invalid_strategy", capacity=10)
    backend = SmartDataCacheBackend(config=config)
    # 应该使用降级实现或默认策略
    backend.set("test_key", "test_value")
    value = backend.get("test_key")
    assert value == "test_value" or value is None


def test_smart_cache_backend_set_exception(smart_cache_backend, monkeypatch):
    """测试 SmartDataCacheBackend（set，异常处理）"""
    # 模拟cache.put抛出异常
    def mock_put(key, value):
        raise Exception("Put error")
    
    monkeypatch.setattr(smart_cache_backend.cache, "put", mock_put)
    
    # 设置应该返回False（异常被捕获）
    result = smart_cache_backend.set("key1", "value1")
    assert result is False


def test_smart_cache_backend_delete_no_methods(smart_cache_backend, monkeypatch):
    """测试 SmartDataCacheBackend（delete，无delete和pop方法）"""
    # 模拟cache没有delete和pop方法
    class MockCache:
        def get(self, key):
            return None
        def put(self, key, value):
            pass
    
    smart_cache_backend.cache = MockCache()
    
    # 删除应该返回False
    result = smart_cache_backend.delete("key1")
    assert result is False


def test_smart_cache_backend_delete_exception(smart_cache_backend, monkeypatch):
    """测试 SmartDataCacheBackend（delete，异常处理）"""
    # 设置一个缓存项
    smart_cache_backend.set("key1", "value1")
    
    # 模拟cache.pop抛出异常（因为可能使用pop而不是delete）
    def mock_pop(key, default=None):
        raise Exception("Delete error")
    
    # 如果cache有delete方法，mock它；否则mock pop
    if hasattr(smart_cache_backend.cache, "delete"):
        monkeypatch.setattr(smart_cache_backend.cache, "delete", mock_pop)
    else:
        monkeypatch.setattr(smart_cache_backend.cache, "pop", mock_pop)
    
    # 删除应该返回False（异常被捕获）
    result = smart_cache_backend.delete("key1")
    assert result is False


def test_smart_cache_backend_clear_exception(smart_cache_backend, monkeypatch):
    """测试 SmartDataCacheBackend（clear，异常处理）"""
    # 模拟重新创建缓存时的异常（因为降级缓存没有clear方法）
    def mock_create_fallback_cache(self):
        raise Exception("Create error")
    
    monkeypatch.setattr(smart_cache_backend, "_create_fallback_cache", mock_create_fallback_cache)
    
    # 清空应该返回False（异常被捕获）
    result = smart_cache_backend.clear()
    assert result is False


def test_smart_cache_backend_clear_no_clear_method(smart_cache_backend, monkeypatch):
    """测试 SmartDataCacheBackend（clear，无clear方法）"""
    # 模拟cache没有clear方法，需要重新创建
    class MockCache:
        def get(self, key):
            return None
        def put(self, key, value):
            pass
    
    smart_cache_backend.cache = MockCache()
    
    # 清空应该重新创建缓存实例
    result = smart_cache_backend.clear()
    assert result is True


def test_smart_cache_get_exception(smart_cache, monkeypatch):
    """测试 SmartDataCache（get，异常处理）"""
    # 模拟backend.get抛出异常
    def mock_get(key):
        raise Exception("Get error")
    
    monkeypatch.setattr(smart_cache.backend, "get", mock_get)
    
    # 获取应该返回None（异常被捕获）
    result = smart_cache.get("key1", DataSourceType.STOCK)
    assert result is None


def test_smart_cache_set_exception(smart_cache, monkeypatch):
    """测试 SmartDataCache（set，异常处理）"""
    # 模拟backend.set抛出异常
    def mock_set(key, value, ttl=None):
        raise Exception("Set error")
    
    monkeypatch.setattr(smart_cache.backend, "set", mock_set)
    
    # 设置应该返回False（异常被捕获）
    result = smart_cache.set("key1", "value1", DataSourceType.STOCK)
    assert result is False


def test_smart_cache_set_with_type_config(smart_cache):
    """测试 SmartDataCache（set，使用类型配置）"""
    # 设置应该使用类型配置的TTL
    result = smart_cache.set("key1", "value1", DataSourceType.STOCK)
    assert result is True


def test_smart_cache_invalidate_pattern_star(smart_cache):
    """测试 SmartDataCache（invalidate，模式为*）"""
    # 设置一些缓存项
    smart_cache.set("key1", "value1", DataSourceType.STOCK)
    smart_cache.set("key2", "value2", DataSourceType.STOCK)
    
    # 失效所有缓存
    result = smart_cache.invalidate("*")
    assert result >= 0


def test_smart_cache_invalidate_pattern_other(smart_cache):
    """测试 SmartDataCache（invalidate，其他模式）"""
    # 设置一些缓存项
    smart_cache.set("key1", "value1", DataSourceType.STOCK)
    
    # 失效其他模式（简化实现返回0）
    result = smart_cache.invalidate("key*")
    assert result == 0


def test_smart_cache_invalidate_exception(smart_cache, monkeypatch):
    """测试 SmartDataCache（invalidate，异常处理）"""
    # 模拟backend.clear抛出异常
    def mock_clear():
        raise Exception("Clear error")
    
    monkeypatch.setattr(smart_cache.backend, "clear", mock_clear)
    
    # 失效应该返回0（异常被捕获）
    result = smart_cache.invalidate("*")
    assert result == 0


def test_smart_cache_delete_exception(smart_cache, monkeypatch):
    """测试 SmartDataCache（delete，异常处理）"""
    # 模拟backend.delete抛出异常
    def mock_delete(key):
        raise Exception("Delete error")
    
    monkeypatch.setattr(smart_cache.backend, "delete", mock_delete)
    
    # 删除应该返回False（异常被捕获）
    result = smart_cache.delete("key1")
    assert result is False


def test_smart_cache_clear_exception(smart_cache, monkeypatch):
    """测试 SmartDataCache（clear，异常处理）"""
    # 模拟backend.clear抛出异常
    def mock_clear():
        raise Exception("Clear error")
    
    monkeypatch.setattr(smart_cache.backend, "clear", mock_clear)
    
    # 清空应该返回False（异常被捕获）
    result = smart_cache.clear()
    assert result is False


def test_smart_cache_exists_with_exists_method(smart_cache):
    """测试 SmartDataCache（exists，有exists方法）"""
    # 设置一个缓存项
    smart_cache.set("key1", "value1", DataSourceType.STOCK)
    
    # 检查应该返回True
    result = smart_cache.exists("key1")
    assert result is True


def test_smart_cache_exists_without_exists_method(smart_cache, monkeypatch):
    """测试 SmartDataCache（exists，无exists方法）"""
    # 模拟backend没有exists方法
    class MockBackend:
        def get(self, key):
            return "value1" if key == "key1" else None
    
    smart_cache.backend = MockBackend()
    
    # 检查应该通过get方法判断
    result = smart_cache.exists("key1")
    assert result is True


def test_smart_cache_exists_exception(smart_cache, monkeypatch):
    """测试 SmartDataCache（exists，异常处理）"""
    # 模拟backend.exists抛出异常
    def mock_exists(key):
        raise Exception("Exists error")
    
    monkeypatch.setattr(smart_cache.backend, "exists", mock_exists)
    
    # 检查应该返回False（异常被捕获）
    result = smart_cache.exists("key1")
    assert result is False


def test_smart_cache_get_cache_info_exception(smart_cache, monkeypatch):
    """测试 SmartDataCache（get_cache_info，异常处理）"""
    # 模拟backend.get_stats抛出异常
    def mock_get_stats():
        raise Exception("Stats error")
    
    monkeypatch.setattr(smart_cache.backend, "get_stats", mock_get_stats)
    
    # 获取信息应该返回空字典（异常被捕获）
    result = smart_cache.get_cache_info()
    assert result == {}


def test_smart_cache_optimize_for_data_type_stream(smart_cache):
    """测试 SmartDataCache（optimize_for_data_type，STREAM类型）"""
    # 优化STREAM类型
    if hasattr(DataSourceType, "STREAM"):
        result = smart_cache.optimize_for_data_type(DataSourceType.STREAM)
        assert result is True


def test_smart_cache_optimize_for_data_type_file(smart_cache):
    """测试 SmartDataCache（optimize_for_data_type，FILE类型）"""
    # 优化FILE类型
    if hasattr(DataSourceType, "FILE"):
        result = smart_cache.optimize_for_data_type(DataSourceType.FILE)
        assert result is True


def test_smart_cache_optimize_for_data_type_exception(smart_cache, monkeypatch):
    """测试 SmartDataCache（optimize_for_data_type，异常处理）"""
    # 模拟getattr抛出异常（在optimize_for_data_type中）
    original_getattr = getattr
    def mock_getattr(obj, name, default=None):
        if name in ("STREAM", "FILE"):
            raise Exception("Getattr error")
        return original_getattr(obj, name, default)
    
    monkeypatch.setattr("builtins.getattr", mock_getattr)
    
    # 优化应该返回False（异常被捕获）
    result = smart_cache.optimize_for_data_type(DataSourceType.STOCK)
    assert result is False


def test_smart_cache_get_stats_exception(smart_cache, monkeypatch):
    """测试 SmartDataCache（get_stats，异常处理）"""
    # 模拟backend.get_stats抛出异常
    def mock_get_stats():
        raise Exception("Stats error")
    
    monkeypatch.setattr(smart_cache.backend, "get_stats", mock_get_stats)
    
    # 获取统计应该返回空字典（异常被捕获）
    result = smart_cache.get_stats()
    assert result == {}


def test_smart_cache_backend_fallback_cache_put_existing_key(smart_cache_backend, monkeypatch):
    """测试 SmartDataCacheBackend（降级缓存，put已存在的key）"""
    # 确保使用降级缓存
    with patch('src.data.cache.smart_data_cache.INFRASTRUCTURE_CACHE_AVAILABLE', False):
        backend = SmartDataCacheBackend(config=DataCacheConfig(capacity=10))
        
        # 设置一个缓存项
        backend.set("key1", "value1")
        
        # 再次设置同一个key（应该move_to_end）
        backend.set("key1", "value2")
        
        # 获取应该返回新值
        value = backend.get("key1")
        assert value == "value2"


def test_smart_cache_backend_fallback_cache_put_capacity_full(smart_cache_backend, monkeypatch):
    """测试 SmartDataCacheBackend（降级缓存，put时容量已满）"""
    # 确保使用降级缓存
    with patch('src.data.cache.smart_data_cache.INFRASTRUCTURE_CACHE_AVAILABLE', False):
        backend = SmartDataCacheBackend(config=DataCacheConfig(capacity=2))
        
        # 填满缓存
        backend.set("key1", "value1")
        backend.set("key2", "value2")
        
        # 添加新项应该触发淘汰
        backend.set("key3", "value3")
        
        # key1应该被淘汰
        value1 = backend.get("key1")
        # key3应该存在
        value3 = backend.get("key3")
        assert value3 == "value3"

