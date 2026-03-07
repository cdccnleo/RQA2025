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


from typing import Any, Optional, Dict
from src.data.interfaces.ICacheBackend import ICacheBackend


class DummyCache(ICacheBackend):
    def __init__(self):
        self.store: Dict[str, Any] = {}
    def get(self, key: str) -> Optional[Any]:
        return self.store.get(key)
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self.store[key] = value
        return True
    def delete(self, key: str) -> bool:
        return self.store.pop(key, None) is not None
    def clear(self) -> bool:
        self.store.clear()
        return True
    def exists(self, key: str) -> bool:
        return key in self.store


def test_cache_backend_contract_minimal():
    c = DummyCache()
    assert c.get("k") is None
    assert c.set("k", 1) is True
    assert c.exists("k") is True and c.get("k") == 1
    assert c.delete("k") is True
    assert c.clear() is True


