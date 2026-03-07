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

from src.data.cache.buffer_components import (
    ComponentFactory,
    BufferComponent,
    BufferComponentFactory,
)


def test_buffer_component_process_success_and_error():

    component = BufferComponent(buffer_id=2, component_type="Trade")
    payload = {"symbol": "RQA", "value": 123}
    result = component.process(payload)

    assert result["status"] == "success"
    assert result["buffer_id"] == 2
    assert result["input_data"] == payload
    assert "processed_at" in result

    class BrokenName:

        def __str__(self):

            raise RuntimeError("fail")

    component.component_name = BrokenName()
    error_result = component.process({"bad": 1})
    assert error_result["status"] == "error"
    assert error_result["error"] == "fail"


def test_buffer_component_status_and_info():

    component = BufferComponent(buffer_id=6, component_type="Signal")
    info = component.get_info()
    status = component.get_status()
    assert component.get_buffer_id() == 6

    assert info["buffer_id"] == 6
    assert info["component_type"] == "Signal"
    assert status["status"] == "active"
    assert status["buffer_id"] == 6


def test_buffer_component_factory_supported_ids():

    available = BufferComponentFactory.get_available_buffers()
    assert available == [2, 6, 10, 14, 18, 22]

    created = BufferComponentFactory.create_all_buffers()
    assert set(created.keys()) == set(available)
    assert isinstance(created[2], BufferComponent)


def test_buffer_component_factory_invalid_id():

    with pytest.raises(ValueError):
        BufferComponentFactory.create_component(1)


def test_legacy_factory_functions_create_components():

    from src.data.cache import buffer_components as module

    for func_name in [
        "create_buffer_buffer_component_2",
        "create_buffer_buffer_component_6",
        "create_buffer_buffer_component_10",
        "create_buffer_buffer_component_14",
        "create_buffer_buffer_component_18",
        "create_buffer_buffer_component_22",
    ]:
        factory_func = getattr(module, func_name)
        component = factory_func()
        assert isinstance(component, BufferComponent)


def test_component_factory_creation_paths(monkeypatch):

    factory = ComponentFactory()

    class DummyComponent:

        def __init__(self):
            self.initialized = False

        def initialize(self, config):

            self.initialized = True
            return True

    monkeypatch.setattr(factory, "_create_component_instance", lambda c, cfg: DummyComponent())
    component = factory.create_component("demo", {"a": 1})
    assert component.initialized is True

    class FalsyComponent(DummyComponent):

        def initialize(self, config):

            return False

    monkeypatch.setattr(factory, "_create_component_instance", lambda c, cfg: FalsyComponent())
    assert factory.create_component("demo", {}) is None

    def broken_create(*args, **kwargs):

        raise RuntimeError("boom")

    monkeypatch.setattr(factory, "_create_component_instance", broken_create)
    assert factory.create_component("demo", {}) is None


def test_factory_info_includes_metadata():

    info = BufferComponentFactory.get_factory_info()
    assert info["factory_name"] == "BufferComponentFactory"
    assert "created_at" in info


def test_component_factory_default_instance():

    factory = ComponentFactory()
    assert factory._create_component_instance("demo", {}) is None

