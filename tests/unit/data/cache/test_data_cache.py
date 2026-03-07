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


import pandas as pd
import pytest

import src.data.cache.data_cache as data_cache_module
from src.data.cache.data_cache import DataCache


class FakeCacheManager:

    def __init__(self, config=None):

        self.storage = {}
        self.cleared = False
        self.exists_calls = []
        self.deleted_keys = []

    def get(self, key):

        return self.storage.get(key)

    def set(self, key, value):

        self.storage[key] = value
        return True

    def exists(self, key):

        self.exists_calls.append(key)
        return key in self.storage

    def delete(self, key):

        self.deleted_keys.append(key)
        return self.storage.pop(key, None) is not None

    def clear(self):

        self.storage.clear()
        self.cleared = True
        return True


@pytest.fixture
def fake_cache_manager_cls(monkeypatch):

    monkeypatch.setattr(data_cache_module, "CacheManager", FakeCacheManager)
    return FakeCacheManager


def test_get_or_compute_uses_cache(fake_cache_manager_cls, tmp_path):

    cache = DataCache(cache_dir=str(tmp_path / "first"))
    cache.cache_manager = fake_cache_manager_cls()
    compute_calls = []

    def compute(value):

        compute_calls.append(value)
        return {"value": value}

    first = cache.get_or_compute("key", compute, 1)
    assert first == {"value": 1}
    assert compute_calls == [1]

    second = cache.get_or_compute("key", compute, 2)
    assert second == {"value": 1}
    assert compute_calls == [1]


def test_dataframe_methods_handle_success_and_errors(fake_cache_manager_cls, tmp_path):

    cache = DataCache(cache_dir=str(tmp_path / "df"))
    cache.cache_manager = fake_cache_manager_cls()
    other_cache = DataCache(cache_dir=str(tmp_path / "df2"))
    other_cache.cache_manager = fake_cache_manager_cls()
    df = pd.DataFrame({"a": [1, 2]})

    assert cache.set_dataframe("frame", df) is True
    returned = cache.get_dataframe("frame")
    pd.testing.assert_frame_equal(returned, df)

    class BrokenManager(FakeCacheManager):

        def get(self, key):

            raise RuntimeError("boom")

        def set(self, key, value):

            raise RuntimeError("boom")

    other_cache.cache_manager = BrokenManager()
    assert other_cache.get_dataframe("frame") is None
    assert other_cache.set_dataframe("frame", df) is False


def test_dict_and_basic_operations(fake_cache_manager_cls, tmp_path):

    cache = DataCache(cache_dir=str(tmp_path / "dict"))
    cache.cache_manager = fake_cache_manager_cls()
    assert cache.set_dict("dict", {"k": "v"}) is True
    assert cache.get_dict("dict") == {"k": "v"}
    assert cache.exists("dict") is True
    assert cache.delete("dict") is True
    assert cache.get_dict("dict") is None

    assert cache.clear() is True
    assert cache.cache_manager.cleared is True


def test_get_and_set_handle_exceptions(fake_cache_manager_cls, tmp_path):

    cache = DataCache(cache_dir=str(tmp_path / "raise"))
    class RaisingManager(FakeCacheManager):

        def get(self, key):

            raise RuntimeError("boom")

        def set(self, key, value):

            raise RuntimeError("boom")

    cache.cache_manager = RaisingManager()
    assert cache.get("missing") is None
    assert cache.set("missing", 1) is False

