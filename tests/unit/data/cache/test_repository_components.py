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

from src.data.cache import repository_components as module
from src.data.cache.repository_components import (
    RepositoryComponent,
    RepositoryComponentFactory,
    ComponentFactory,
)


def test_repository_component_process_success_and_error():

    component = RepositoryComponent(repository_id=4, component_type="Repo")
    payload = {"table": "orders"}
    success = component.process(payload)
    assert success["status"] == "success"
    assert success["repository_id"] == 4
    assert component.get_repository_id() == 4
    assert success["input_data"] == payload

    class BrokenName:

        def __str__(self):

            raise RuntimeError("boom")

    component.component_name = BrokenName()
    error = component.process({"bad": 1})
    assert error["status"] == "error"
    assert error["error"] == "boom"


def test_repository_component_info_and_status():

    component = RepositoryComponent(repository_id=8, component_type="Repo")
    info = component.get_info()
    status = component.get_status()
    assert info["repository_id"] == 8
    assert status["status"] == "active"


def test_repository_factory_supported_ids():

    assert RepositoryComponentFactory.get_available_repositorys() == [4, 8, 12, 16, 20, 24]
    created = RepositoryComponentFactory.create_all_repositorys()
    assert set(created.keys()) == {4, 8, 12, 16, 20, 24}


def test_repository_factory_invalid_id():

    with pytest.raises(ValueError):
        RepositoryComponentFactory.create_component(0)


def test_legacy_repository_factory_functions():

    for func_name in [
        "create_repository_repository_component_4",
        "create_repository_repository_component_8",
        "create_repository_repository_component_12",
        "create_repository_repository_component_16",
        "create_repository_repository_component_20",
        "create_repository_repository_component_24",
    ]:
        component = getattr(module, func_name)()
        assert isinstance(component, RepositoryComponent)


def test_repository_component_factory_paths(monkeypatch):

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


def test_repository_factory_info_contains_metadata():

    info = RepositoryComponentFactory.get_factory_info()
    assert info["factory_name"] == "RepositoryComponentFactory"
    assert "created_at" in info


def test_repository_component_factory_default_instance():

    factory = ComponentFactory()
    assert factory._create_component_instance("demo", {}) is None

