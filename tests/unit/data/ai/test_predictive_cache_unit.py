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


from src.data.ai.predictive_cache import PredictiveCache, CacheStats


def test_set_get_stats_and_fifo_eviction():
    cache = PredictiveCache(capacity=2)
    cache.set("k1", 1)
    cache.set("k2", 2)
    assert cache.get("k1") == 1
    assert cache.get("kX") is None
    stats = cache.get_stats()
    assert isinstance(stats, CacheStats)
    assert stats.hits == 1 and stats.misses == 1 and stats.evictions == 0
    # 触发 FIFO 淘汰
    cache.set("k3", 3)  # 淘汰 k1
    assert cache.get("k1") is None
    assert cache.get("k2") == 2
    assert cache.get("k3") == 3
    stats2 = cache.get_stats()
    assert stats2.evictions == 1


def test_predict_next_key_and_top_predictions():
    cache = PredictiveCache(capacity=10)
    # 通过 set 的顺序建立转移：a->b 出现 2 次，a->c 出现 1 次
    cache.set("a", 0)
    cache.set("b", 1)
    cache.set("a", 2)  # 覆盖不改变转移
    cache.set("b", 3)
    cache.set("a", 4)
    cache.set("c", 5)
    # 预测
    assert cache.predict_next_key("a") in {"b", "c"}
    top = cache.top_predictions("a", k=2)
    assert top[0] == cache.predict_next_key("a")
    assert len(top) >= 1


def test_delete_and_clear():
    cache = PredictiveCache(capacity=2)
    cache.set("x", 1)
    assert cache.delete("x") is True
    assert cache.get("x") is None
    cache.set("y", 2)
    cache.clear()
    st = cache.get_stats()
    assert st.size == 0 and st.hits == 0 and st.misses == 0 and st.evictions == 0


