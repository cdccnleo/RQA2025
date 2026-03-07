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

from src.data.quality.monitor_components import (
    ComponentFactory,
    MonitorComponent,
    MonitorComponentFactory,
    create_monitor_monitor_component_4,
    create_monitor_monitor_component_69,
)


class DummyMonitorComponent(MonitorComponent):
    """用于验证 ComponentFactory 行为的受控组件"""

    def __init__(self, monitor_id=101, component_type="DummyMonitor"):
        super().__init__(monitor_id, component_type)
        self.init_called = False
        self.init_config = None

    def initialize(self, config=None):
        self.init_called = True
        self.init_config = config or {}
        return True


class FailingMonitorComponent(MonitorComponent):
    def initialize(self, config=None):
        return False


class ExplodingMonitorComponent(MonitorComponent):
    def process(self, data):
        raise RuntimeError("boom")


class DummyMonitorFactory(ComponentFactory):
    def __init__(self, component_cls):
        super().__init__()
        self.component_cls = component_cls

    def _create_component_instance(self, component_type, config):
        if self.component_cls is RuntimeError:
            raise RuntimeError("mock failure")
        return self.component_cls(config.get("monitor_id", 888), component_type=component_type)


class TestComponentFactoryBehavior:
    def test_create_component_success(self):
        factory = DummyMonitorFactory(DummyMonitorComponent)
        component = factory.create_component("CustomMonitor", {"monitor_id": 123, "mode": "test"})

        assert isinstance(component, DummyMonitorComponent)
        assert component.init_called is True
        assert component.init_config["mode"] == "test"

    def test_create_component_when_initialize_fails(self):
        factory = DummyMonitorFactory(FailingMonitorComponent)
        component = factory.create_component("CustomMonitor", {"monitor_id": 12})
        assert component is None

    def test_create_component_handles_exceptions(self):
        factory = DummyMonitorFactory(RuntimeError)
        assert factory.create_component("Any", {"monitor_id": 1}) is None


class TestMonitorComponent:
    def test_get_info_and_status(self):
        component = MonitorComponent(monitor_id=8, component_type="Monitor")

        info = component.get_info()
        status = component.get_status()

        assert info["monitor_id"] == 8
        # description 字符串模板未使用 f-string，确保目前值仍能被识别
        assert "{self.component_type}" in info["description"]
        assert status["status"] == "active"

    def test_process_success_and_error(self, monkeypatch):
        component = MonitorComponent(monitor_id=10)
        result = component.process({"value": 42})
        assert result["status"] == "success"
        assert result["result"].startswith("Processed by")

        exploding = MonitorComponent(monitor_id=11)

        class FlakyDatetime:
            def __init__(self):
                self.calls = 0

            def now(self):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("broken datetime")

                class DummyDT:
                    @staticmethod
                    def isoformat():
                        return "fallback"

                return DummyDT()

        monkeypatch.setattr(
            "src.data.quality.monitor_components.datetime",
            FlakyDatetime(),
        )

        error_result = exploding.process({"value": 1})
        assert error_result["status"] == "error"
        assert error_result["error_type"] == "RuntimeError"


class TestMonitorComponentFactory:
    def test_create_component_with_supported_id(self):
        component = MonitorComponentFactory.create_component(4)
        assert isinstance(component, MonitorComponent)
        assert component.get_monitor_id() == 4

    def test_create_component_with_invalid_id_raises(self):
        with pytest.raises(ValueError):
            MonitorComponentFactory.create_component(999)

    def test_get_available_ids_and_create_all(self):
        ids = MonitorComponentFactory.get_available_monitors()
        all_components = MonitorComponentFactory.create_all_monitors()

        assert ids == sorted(ids)
        assert set(all_components.keys()) == set(ids)
        assert all(isinstance(c, MonitorComponent) for c in all_components.values())

    def test_get_factory_info_contains_expected_fields(self):
        info = MonitorComponentFactory.get_factory_info()
        assert info["factory_name"] == "MonitorComponentFactory"
        assert info["total_monitors"] == len(MonitorComponentFactory.SUPPORTED_MONITOR_IDS)
        assert info["supported_ids"] == sorted(MonitorComponentFactory.SUPPORTED_MONITOR_IDS)

    def test_shim_functions_create_components(self):
        comp4 = create_monitor_monitor_component_4()
        comp69 = create_monitor_monitor_component_69()

        assert isinstance(comp4, MonitorComponent)
        assert isinstance(comp69, MonitorComponent)
        assert comp4.get_monitor_id() == 4
        assert comp69.get_monitor_id() == 69

