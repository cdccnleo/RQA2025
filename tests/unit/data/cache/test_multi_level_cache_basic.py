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
from src.data.cache.multi_level_cache import MultiLevelCache, CacheConfig


def test_memory_set_get_and_ttl_expire_then_miss_and_clean():
    cfg = CacheConfig(memory_max_size=10, memory_ttl=1, disk_enabled=False)
    cache = MultiLevelCache(cfg)
    assert cache.set("k", "v", ttl=1) is True
    assert cache.get("k") == "v"
    time.sleep(1.2)
    # 过期后应 miss
    assert cache.get("k") is None
    # 清理应返回0或更多（不强绑定具体数），但不抛错
    cleaned = cache.clean_expired()
    assert isinstance(cleaned, int)


def test_disk_fallback_and_promotion_to_memory():
    cfg = CacheConfig(memory_max_size=10, memory_ttl=1, disk_enabled=True, disk_ttl=60)
    cache = MultiLevelCache(cfg)
    assert cache.set("d", {"x": 1}, ttl=5) is True
    # 首次命中内存
    assert cache.get("d") == {"x": 1}
    # 等待内存过期，但磁盘有效
    time.sleep(1.2)
    v = cache.get("d")
    assert v == {"x": 1}
    st = cache.get_stats()
    # 应至少有一次 disk 命中或总请求 > 0
    assert st["performance"]["total_requests"] >= 2


