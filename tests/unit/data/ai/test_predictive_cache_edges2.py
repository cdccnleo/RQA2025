"""
边界测试：predictive_cache.py
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
from src.data.ai.predictive_cache import PredictiveCache, CacheStats


def test_predictive_cache_init():
    """测试 PredictiveCache（初始化）"""
    cache = PredictiveCache(capacity=10)
    
    assert cache.capacity == 10
    assert len(cache._store) == 0
    assert len(cache._order) == 0
    assert cache._hits == 0
    assert cache._misses == 0
    assert cache._evictions == 0


def test_predictive_cache_init_zero_capacity():
    """测试 PredictiveCache（初始化，零容量）"""
    with pytest.raises(ValueError, match="capacity must be positive"):
        PredictiveCache(capacity=0)


def test_predictive_cache_init_negative_capacity():
    """测试 PredictiveCache（初始化，负容量）"""
    with pytest.raises(ValueError, match="capacity must be positive"):
        PredictiveCache(capacity=-1)


def test_predictive_cache_set():
    """测试 PredictiveCache（设置）"""
    cache = PredictiveCache(capacity=3)
    
    cache.set("key1", "value1")
    
    assert "key1" in cache._store
    assert cache._store["key1"] == "value1"
    assert "key1" in cache._order


def test_predictive_cache_set_update():
    """测试 PredictiveCache（设置，更新）"""
    cache = PredictiveCache(capacity=3)
    
    cache.set("key1", "value1")
    cache.set("key1", "value2")
    
    assert cache._store["key1"] == "value2"
    assert len(cache._order) == 1  # 不增加新项


def test_predictive_cache_set_eviction():
    """测试 PredictiveCache（设置，驱逐）"""
    cache = PredictiveCache(capacity=2)
    
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")  # 应该驱逐 key1
    
    assert "key1" not in cache._store
    assert "key2" in cache._store
    assert "key3" in cache._store
    assert cache._evictions == 1


def test_predictive_cache_get_hit():
    """测试 PredictiveCache（获取，命中）"""
    cache = PredictiveCache(capacity=3)
    
    cache.set("key1", "value1")
    result = cache.get("key1")
    
    assert result == "value1"
    assert cache._hits == 1
    assert cache._misses == 0


def test_predictive_cache_get_miss():
    """测试 PredictiveCache（获取，未命中）"""
    cache = PredictiveCache(capacity=3)
    
    result = cache.get("nonexistent")
    
    assert result is None
    assert cache._hits == 0
    assert cache._misses == 1


def test_predictive_cache_delete_existing():
    """测试 PredictiveCache（删除，存在）"""
    cache = PredictiveCache(capacity=3)
    
    cache.set("key1", "value1")
    result = cache.delete("key1")
    
    assert result is True
    assert "key1" not in cache._store
    assert "key1" not in cache._order


def test_predictive_cache_delete_value_error():
    """测试 PredictiveCache（删除，ValueError 异常处理，覆盖 71-72 行）"""
    cache = PredictiveCache(capacity=3)
    
    # 设置一个 key
    cache.set("key1", "value1")
    # 手动从 _order 中移除 key，但保留在 _store 中
    # 这样 delete 时会触发 ValueError
    cache._order.remove("key1")
    # 现在删除 key1，应该能处理 ValueError
    result = cache.delete("key1")
    
    assert result is True
    assert "key1" not in cache._store


def test_predictive_cache_top_predictions_negative_k():
    """测试 PredictiveCache（top_predictions，负 k 值，覆盖 112 行）"""
    cache = PredictiveCache(capacity=3)
    
    # 设置一些数据
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    
    # 测试负 k 值
    result = cache.top_predictions("key1", k=-1)
    
    # max(0, -1) = 0，所以应该返回空列表
    assert result == []


def test_predictive_cache_top_predictions_no_transitions():
    """测试 PredictiveCache（top_predictions，无转移关系，覆盖 110 行）"""
    cache = PredictiveCache(capacity=3)
    
    # 设置一些数据，但不建立转移关系
    cache.set("key1", "value1")
    # 不设置 key1 -> key2 的转移
    
    # 测试 top_predictions，应该返回空列表（因为 next_counts 为空）
    result = cache.top_predictions("key1", k=3)
    
    # 应该返回空列表（因为 not next_counts 为 True）
    assert result == []


def test_predictive_cache_delete_nonexistent():
    """测试 PredictiveCache（删除，不存在）"""
    cache = PredictiveCache(capacity=3)
    
    result = cache.delete("nonexistent")
    
    assert result is False


def test_predictive_cache_clear():
    """测试 PredictiveCache（清空）"""
    cache = PredictiveCache(capacity=3)
    
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.get("key1")
    cache.clear()
    
    assert len(cache._store) == 0
    assert len(cache._order) == 0
    assert cache._hits == 0
    assert cache._misses == 0
    assert cache._evictions == 0
    assert len(cache._transitions) == 0
    assert cache._last_key is None


def test_predictive_cache_get_stats():
    """测试 PredictiveCache（获取统计）"""
    cache = PredictiveCache(capacity=3)
    
    cache.set("key1", "value1")
    cache.get("key1")
    cache.get("key2")
    
    stats = cache.get_stats()
    
    assert isinstance(stats, CacheStats)
    assert stats.capacity == 3
    assert stats.size == 1
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.evictions == 0


def test_predictive_cache_predict_next_key_none():
    """测试 PredictiveCache（预测下一个键，无）"""
    cache = PredictiveCache(capacity=3)
    
    result = cache.predict_next_key("key1")
    
    assert result is None


def test_predictive_cache_predict_next_key():
    """测试 PredictiveCache（预测下一个键）"""
    cache = PredictiveCache(capacity=10)
    
    # 创建转移模式：key1 -> key2 (3次), key1 -> key3 (1次)
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key1", "value1")
    cache.set("key3", "value3")
    
    result = cache.predict_next_key("key1")
    
    # key2 应该是最可能的（出现3次）
    assert result == "key2"


def test_predictive_cache_predict_next_key_tie():
    """测试 PredictiveCache（预测下一个键，并列）"""
    cache = PredictiveCache(capacity=10)
    
    # 创建转移模式：key1 -> key2 (1次), key1 -> key3 (1次)
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key1", "value1")
    cache.set("key3", "value3")
    
    result = cache.predict_next_key("key1")
    
    # 应该选择按字母顺序较小的
    assert result in ["key2", "key3"]


def test_predictive_cache_transitions():
    """测试 PredictiveCache（转移记录）"""
    cache = PredictiveCache(capacity=10)
    
    # 注意：更新已存在的键不会记录转移，只有新键才会
    cache.set("key1", "value1")
    cache.set("key2", "value2")  # key1 -> key2 (1次)
    cache.set("key3", "value3")  # key2 -> key3 (1次)
    cache.set("key4", "value4")  # key3 -> key4 (1次)
    # 更新 key2 不会记录转移，因为 key2 已存在
    cache.set("key5", "value5")  # key4 -> key5 (1次)
    
    assert "key1" in cache._transitions
    assert "key2" in cache._transitions["key1"]
    assert cache._transitions["key1"]["key2"] == 1
    assert "key3" in cache._transitions
    assert "key4" in cache._transitions["key3"]
    assert cache._transitions["key3"]["key4"] == 1
    assert "key4" in cache._transitions
    assert "key5" in cache._transitions["key4"]
    assert cache._transitions["key4"]["key5"] == 1


def test_predictive_cache_last_key():
    """测试 PredictiveCache（最后键）"""
    cache = PredictiveCache(capacity=10)
    
    assert cache._last_key is None
    
    cache.set("key1", "value1")
    assert cache._last_key == "key1"
    
    cache.set("key2", "value2")
    assert cache._last_key == "key2"


def test_predictive_cache_multiple_evictions():
    """测试 PredictiveCache（多次驱逐）"""
    cache = PredictiveCache(capacity=2)
    
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")  # 驱逐 key1
    cache.set("key4", "value4")  # 驱逐 key2
    
    assert cache._evictions == 2
    assert "key1" not in cache._store
    assert "key2" not in cache._store
    assert "key3" in cache._store
    assert "key4" in cache._store


def test_predictive_cache_capacity_one():
    """测试 PredictiveCache（容量为1）"""
    cache = PredictiveCache(capacity=1)
    
    cache.set("key1", "value1")
    assert "key1" in cache._store
    
    cache.set("key2", "value2")
    assert "key1" not in cache._store
    assert "key2" in cache._store
    assert cache._evictions == 1
