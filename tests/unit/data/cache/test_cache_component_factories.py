import asyncio
import pandas as pd
from unittest.mock import Mock
from typing import Any, Optional

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
from unittest.mock import patch
from typing import Optional

from src.data.cache.buffer_components import BufferComponentFactory
from src.data.cache.repository_components import RepositoryComponentFactory
from src.data.cache.store_components import StoreComponentFactory
from src.data.cache.smart_cache_optimizer import SmartCacheOptimizer
from src.data.interfaces.standard_interfaces import DataSourceType
from src.data.interfaces.data_interfaces import IDataCache


class DummyCache(IDataCache):
    """简易缓存实现，用于隔离 SmartCacheOptimizer 的外部依赖。"""

    def __init__(self):
        self._store = {}

    def get_cached_data(self, key: str):
        return self._store.get(key)

    def get(self, key: str):
        """兼容性方法"""
        return self.get_cached_data(key)

    def set_cached_data(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        self._store[key] = data
        return True

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """兼容性方法"""
        return self.set_cached_data(key, value, ttl)

    def invalidate_cache(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def delete(self, key: str) -> bool:
        """兼容性方法"""
        return self.invalidate_cache(key)

    def clear_cache(self) -> bool:
        self._store.clear()
        return True

    def get_cache_stats(self):
        return {"item_count": len(self._store)}


# ---------------------------------------------------------------------------
# Buffer / Repository / Store 工厂轻量行为校验
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("factory", "create_method", "valid_ids", "info_key"),
    [
        (BufferComponentFactory, BufferComponentFactory.create_component, BufferComponentFactory.get_available_buffers(), "buffer_id"),
        (RepositoryComponentFactory, RepositoryComponentFactory.create_component, RepositoryComponentFactory.get_available_repositorys(), "repository_id"),
        (StoreComponentFactory, StoreComponentFactory.create_component, StoreComponentFactory.get_available_stores(), "store_id"),
    ],
)
def test_component_factory_creates_valid_components(factory, create_method, valid_ids, info_key):
    """验证工厂可以创建支持的组件，并返回基本信息。"""
    component_id = valid_ids[0]
    component = create_method(component_id)

    info = component.get_info()
    status = component.get_status()
    result = component.process({"payload": "data"})

    assert info[info_key] == component_id
    assert status[info_key] == component_id
    assert result["status"] == "success"


@pytest.mark.parametrize(
    ("create_method", "invalid_id"),
    [
        (BufferComponentFactory.create_component, 999),
        (RepositoryComponentFactory.create_component, 999),
        (StoreComponentFactory.create_component, 999),
    ],
)
def test_component_factory_invalid_id_raises(create_method, invalid_id):
    with pytest.raises(ValueError):
        create_method(invalid_id)


# ---------------------------------------------------------------------------
# SmartCacheOptimizer 核心行为
# ---------------------------------------------------------------------------

def _build_optimizer(cache: IDataCache) -> SmartCacheOptimizer:
    """辅助函数：构造不启动后台线程的 SmartCacheOptimizer。"""
    with patch.object(SmartCacheOptimizer, "_start_preload_scheduler", lambda self: None), \
         patch.object(SmartCacheOptimizer, "_initialize_preload_rules", lambda self: None):
        optimizer = SmartCacheOptimizer(cache)
    return optimizer


def test_smart_cache_optimizer_set_get_and_stats():
    cache = DummyCache()
    optimizer = _build_optimizer(cache)

    try:
        # 写入并读取缓存
        assert optimizer.smart_set("key", "value", DataSourceType.STOCK, ttl_seconds=120) is True
        assert optimizer.smart_get("key", DataSourceType.STOCK) == "value"

        entry = optimizer._cache_entries["key"]
        assert entry.access_count == 1
        assert entry.priority == optimizer._get_configured_priority(DataSourceType.STOCK)
    finally:
        optimizer.shutdown()


def test_smart_cache_optimizer_invalidate_removes_entry():
    cache = DummyCache()
    optimizer = _build_optimizer(cache)

    try:
        optimizer.smart_set("key", "value", DataSourceType.CRYPTO, ttl_seconds=10)
        assert "key" in optimizer._cache_entries

        optimizer.invalidate_cache_entry("key", DataSourceType.CRYPTO)
        assert "key" not in optimizer._cache_entries
        assert cache.get("key") is None
    finally:
        optimizer.shutdown()

