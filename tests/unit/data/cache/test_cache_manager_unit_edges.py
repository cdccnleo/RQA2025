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


import time
from src.data.cache.cache_manager import CacheManager, CacheConfig


def test_set_get_exists_and_ttl_expiry(monkeypatch):
    cfg = CacheConfig(max_size=10, enable_disk_cache=False, ttl=1, enable_stats=True)
    cm = CacheManager(cfg)
    assert cm.set("k", "v", ttl=1) is True
    assert cm.get("k") == "v"
    assert cm.exists("k") is True
    # 强制过期：回拨 created_at
    cm.memory_cache["k"].created_at = time.time() - 10
    assert cm.get("k") is None
    assert cm.exists("k") is False


def test_eviction_on_max_size_and_list_keys():
    cfg = CacheConfig(max_size=2, enable_disk_cache=False, ttl=60, enable_stats=True)
    cm = CacheManager(cfg)
    cm.set("k1", 1)
    time.sleep(0.01)
    cm.set("k2", 2)
    time.sleep(0.01)
    cm.set("k3", 3)  # 触发 LRU 驱逐，k1 应被淘汰
    keys = cm.list_keys()
    assert "k1" not in keys and "k2" in keys and "k3" in keys


def test_delete_clear_stats_and_health_check(tmp_path):
    cfg = CacheConfig(max_size=5, enable_disk_cache=False, enable_stats=True, disk_cache_dir=str(tmp_path))
    cm = CacheManager(cfg)
    cm.set("a", 1)
    cm.set("b", 2)
    assert cm.delete("a") is True
    stats = cm.get_stats()
    assert stats["cache_size"] >= 1
    cleared = cm.clear()
    assert isinstance(cleared, int) and cleared >= 1
    hc = cm.health_check()
    assert hc["status"] in {"healthy", "error"}
    cm.close()


