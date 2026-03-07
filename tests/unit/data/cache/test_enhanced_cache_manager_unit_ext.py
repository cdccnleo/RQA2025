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


from typing import Any, Optional

import pytest

from src.data.cache.enhanced_cache_manager import EnhancedCacheManager


def test_set_get_and_stats_with_prefix(tmp_path):
    cache_dir = tmp_path / "cache"
    manager = EnhancedCacheManager(cache_dir=str(cache_dir), max_memory_size=1024 * 1024, max_disk_size=1024 * 1024)

    assert manager.set("k1", {"v": 1}, expire=5, prefix="p")
    assert manager.get("k1", prefix="p") == {"v": 1}
    # miss
    assert manager.get("miss", prefix="p") is None

    stats = manager.get_stats()
    assert stats["total_operations"] >= 2
    assert stats["memory_cache_count"] >= 0
    assert "hit_rate" in stats

    # 清理指定前缀
    manager.clear(prefix="p")
    assert manager.get("k1", prefix="p") is None

    manager.shutdown()


def test_reject_invalid_inputs(tmp_path):
    cache_dir = tmp_path / "cache"
    manager = EnhancedCacheManager(cache_dir=str(cache_dir))

    with pytest.raises(ValueError):
        manager.set("", 1)
    with pytest.raises(ValueError):
        manager.set("a", None)
    with pytest.raises(ValueError):
        manager.set("a", 1, expire=-1)

    manager.shutdown()


