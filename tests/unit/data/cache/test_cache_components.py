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

from src.data.cache import cache_components as module
from src.data.cache.cache_components import (
    CacheComponent,
    DataCacheComponentFactory,
    ComponentFactory,
)


def test_cache_component_process_success_and_error():

    component = CacheComponent(cache_id=1, component_type="Hot")
    payload = {"key": "value"}
    success = component.process(payload)
    assert success["status"] == "success"
    assert success["cache_id"] == 1
    assert success["input_data"] == payload
    assert component.get_cache_id() == 1
    assert "processed_at" in success

    class BrokenName:

        def __str__(self):

            raise RuntimeError("boom")

    component.component_name = BrokenName()
    error = component.process({"bad": 1})
    assert error["status"] == "error"
    assert error["error"] == "boom"


def test_cache_component_info_and_status():

    component = CacheComponent(cache_id=5, component_type="Cold")
    info = component.get_info()
    status = component.get_status()
    assert info["cache_id"] == 5
    assert status["health"] == "good"


def test_data_cache_factory_supported_ids():

    assert DataCacheComponentFactory.get_available_caches() == [1, 5, 9, 13, 17, 21]
    created = DataCacheComponentFactory.create_all_caches()
    assert set(created.keys()) == set([1, 5, 9, 13, 17, 21])


def test_data_cache_factory_invalid_id():

    with pytest.raises(ValueError):
        DataCacheComponentFactory.create_component(0)


def test_legacy_cache_factory_functions():

    for func_name in [
        "create_datacache_cache_component_1",
        "create_datacache_cache_component_5",
        "create_datacache_cache_component_9",
        "create_datacache_cache_component_13",
        "create_datacache_cache_component_17",
        "create_datacache_cache_component_21",
    ]:
        component = getattr(module, func_name)()
        assert isinstance(component, CacheComponent)


def test_component_factory_paths(monkeypatch):

    factory = ComponentFactory()

    class DummyComponent:

        def __init__(self):
            self.initialized = False

        def initialize(self, config):

            self.initialized = True
            return True

    monkeypatch.setattr(factory, "_create_component_instance", lambda *a, **k: DummyComponent())
    assert factory.create_component("demo", {"x": 1}).initialized is True

    class FalsyComponent(DummyComponent):

        def initialize(self, config):

            return False

    monkeypatch.setattr(factory, "_create_component_instance", lambda *a, **k: FalsyComponent())
    assert factory.create_component("demo", {}) is None

    def broken(*a, **k):

        raise RuntimeError("boom")

    monkeypatch.setattr(factory, "_create_component_instance", broken)
    assert factory.create_component("demo", {}) is None


def test_factory_info_contains_metadata():

    info = DataCacheComponentFactory.get_factory_info()
    assert info["factory_name"] == "DataCacheComponentFactory"
    assert "created_at" in info


def test_default_component_factory_instance():

    factory = ComponentFactory()
    assert factory._create_component_instance("demo", {}) is None

