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

from src.data.quality.quality_components import (
    ComponentFactory,
    QualityComponent,
    QualityComponentFactory,
    create_quality_quality_component_1,
    create_quality_quality_component_66,
)


class DummyQualityComponent(QualityComponent):
    """用于验证 ComponentFactory 行为的受控组件"""

    def __init__(self, quality_id=99, component_type="Dummy"):
        super().__init__(quality_id, component_type)
        self.init_called = False
        self.init_config = None

    def initialize(self, config=None):
        self.init_called = True
        self.init_config = config or {}
        return True


class FailingInitComponent(QualityComponent):
    def initialize(self, config=None):
        return False


class ExplodingComponent(QualityComponent):
    def process(self, data):
        raise ValueError("boom")


class DummyComponentFactory(ComponentFactory):
    def __init__(self, component_cls):
        super().__init__()
        self.component_cls = component_cls

    def _create_component_instance(self, component_type, config):
        if self.component_cls is RuntimeError:
            raise RuntimeError("mock failure")
        return self.component_cls(config.get("quality_id", 88), component_type=component_type)


class TestComponentFactory:
    """覆盖 ComponentFactory 核心分支"""

    def test_create_component_success(self):
        factory = DummyComponentFactory(DummyQualityComponent)
        component = factory.create_component("Custom", {"quality_id": 123, "mode": "test"})

        assert isinstance(component, DummyQualityComponent)
        assert component.init_called is True
        assert component.init_config["mode"] == "test"

    def test_create_component_when_initialize_fails(self):
        factory = DummyComponentFactory(FailingInitComponent)
        component = factory.create_component("Custom", {"quality_id": 5})
        assert component is None

    def test_create_component_handles_exceptions(self):
        factory = DummyComponentFactory(RuntimeError)
        # 异常路径会返回 None
        assert factory.create_component("Custom", {"quality_id": 5}) is None


class TestQualityComponent:
    """覆盖 QualityComponent 行为"""

    def test_get_info_status_and_validate(self):
        component = QualityComponent(quality_id=7, component_type="Monitor")
        component.initialize({"threshold": 0.9})

        info = component.get_info()
        status = component.get_status()

        assert info["quality_id"] == 7
        assert info["component_type"] == "Monitor"
        assert status["status"] == "active"
        assert component.validate({"any": "payload"}) is True

    def test_process_success_and_error_wrapping(self):
        component = QualityComponent(quality_id=10, component_type="Quality")
        result = component.process({"value": 42})
        assert result["status"] == "success"
        assert result["result"].startswith("Processed by")

        exploding = ExplodingComponent(quality_id=11, component_type="Quality")
        error_result = exploding.process({"value": 1})
        assert error_result["status"] == "error"
        assert error_result["error_type"] == "ValueError"


class TestQualityComponentFactory:
    """覆盖 QualityComponentFactory 静态方法及向后兼容函数"""

    def test_create_component_with_supported_id(self):
        component = QualityComponentFactory.create_component(1)
        assert isinstance(component, QualityComponent)
        assert component.get_quality_id() == 1

    def test_create_component_with_invalid_id_raises(self):
        with pytest.raises(ValueError):
            QualityComponentFactory.create_component(9999)

    def test_get_available_ids_and_create_all(self):
        ids = QualityComponentFactory.get_available_qualitys()
        all_components = QualityComponentFactory.create_all_qualitys()

        assert ids == sorted(ids)
        assert set(all_components.keys()) == set(ids)
        assert all(isinstance(c, QualityComponent) for c in all_components.values())

    def test_get_factory_info_contains_expected_fields(self):
        info = QualityComponentFactory.get_factory_info()
        assert info["factory_name"] == "QualityComponentFactory"
        assert info["total_qualities"] == len(QualityComponentFactory.SUPPORTED_QUALITY_IDS)
        assert info["supported_ids"] == sorted(QualityComponentFactory.SUPPORTED_QUALITY_IDS)

    def test_shim_functions_create_components(self):
        comp1 = create_quality_quality_component_1()
        comp66 = create_quality_quality_component_66()

        assert isinstance(comp1, QualityComponent)
        assert isinstance(comp66, QualityComponent)
        assert comp1.get_quality_id() == 1
        assert comp66.get_quality_id() == 66

