"""
测试超参数组件
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.ml.tuning.hyperparameter_components import (
    ComponentFactory,
    IHyperparameterComponent,
    HyperparameterComponent,
    HyperparameterComponentFactory
)


class TestComponentFactory:
    """测试组件工厂"""

    def setup_method(self):
        """测试前准备"""
        self.factory = ComponentFactory()

    def test_component_factory_init(self):
        """测试组件工厂初始化"""
        assert isinstance(self.factory._components, dict)
        assert len(self.factory._components) == 0

    def test_component_factory_create_component(self):
        """测试创建组件"""
        result = self.factory.create_component("test_type", {"param": "value"})
        assert result is None  # 当前实现返回None


class MockHyperparameterComponent(HyperparameterComponent):
    """用于测试的超参数组件模拟实现"""

    def __init__(self, hyperparameter_id=1, component_type="Mock"):
        super().__init__(hyperparameter_id, component_type)

    def get_info(self):
        """获取组件信息"""
        return {
            "hyperparameter_id": self.hyperparameter_id,
            "component_type": self.component_type,
            "type": "mock",
            "created_at": datetime.now().isoformat()
        }

    def process(self, data):
        """处理数据"""
        return {
            "processed": True,
            "input_data": data,
            "hyperparameter_id": self.hyperparameter_id
        }

    def get_status(self):
        """获取组件状态"""
        return {
            "status": "active",
            "hyperparameter_id": self.hyperparameter_id,
            "last_updated": datetime.now().isoformat()
        }


class TestIHyperparameterComponent:
    """测试超参数组件接口"""

    def test_ihyperparameter_component_is_abstract(self):
        """测试IHyperparameterComponent是抽象类"""
        import inspect
        assert inspect.isabstract(IHyperparameterComponent)

        # 检查抽象方法
        assert hasattr(IHyperparameterComponent, 'get_info')
        assert hasattr(IHyperparameterComponent, 'process')
        assert hasattr(IHyperparameterComponent, 'get_status')

        # 验证无法实例化
        with pytest.raises(TypeError):
            IHyperparameterComponent()


class TestHyperparameterComponent:
    """测试超参数组件"""

    def setup_method(self):
        """测试前准备"""
        self.component = MockHyperparameterComponent(
            hyperparameter_id=123,
            component_type="TestComponent"
        )

    def test_hyperparameter_component_init(self):
        """测试超参数组件初始化"""
        assert self.component.hyperparameter_id == 123
        assert self.component.component_type == "TestComponent"
        assert self.component.component_name == "TestComponent_Component_123"

    def test_hyperparameter_component_get_info(self):
        """测试获取组件信息"""
        info = self.component.get_info()
        assert isinstance(info, dict)
        assert info["hyperparameter_id"] == 123
        assert info["component_type"] == "TestComponent"
        assert info["type"] == "mock"
        assert "created_at" in info

    def test_hyperparameter_component_process(self):
        """测试处理数据"""
        test_data = {"input": "test", "value": 123}
        result = self.component.process(test_data)
        assert isinstance(result, dict)
        assert result["processed"] == True
        assert result["input_data"] == test_data
        assert result["hyperparameter_id"] == 123

    def test_hyperparameter_component_get_status(self):
        """测试获取组件状态"""
        status = self.component.get_status()
        assert isinstance(status, dict)
        assert status["status"] == "active"
        assert status["hyperparameter_id"] == 123
        assert "last_updated" in status


class TestHyperparameterComponentFactory:
    """测试超参数组件工厂"""

    def test_hyperparameter_component_factory_supported_ids(self):
        """测试支持的hyperparameter ID"""
        supported_ids = HyperparameterComponentFactory.SUPPORTED_HYPERPARAMETER_IDS
        assert isinstance(supported_ids, list)
        assert len(supported_ids) > 0
        assert all(isinstance(id, int) for id in supported_ids)

    def test_hyperparameter_component_factory_create_component(self):
        """测试创建超参数组件"""
        component = HyperparameterComponentFactory.create_component(3)
        assert isinstance(component, HyperparameterComponent)
        assert component.hyperparameter_id == 3

    def test_hyperparameter_component_factory_get_available_hyperparameters(self):
        """测试获取可用hyperparameter"""
        available_ids = HyperparameterComponentFactory.get_available_hyperparameters()
        assert isinstance(available_ids, list)
        assert len(available_ids) > 0
        assert 3 in available_ids  # 3是支持的ID之一

    def test_hyperparameter_component_factory_create_all_hyperparameters(self):
        """测试创建所有hyperparameter"""
        all_components = HyperparameterComponentFactory.create_all_hyperparameters()
        assert isinstance(all_components, dict)
        assert len(all_components) > 0
        assert all(isinstance(comp, HyperparameterComponent) for comp in all_components.values())

    def test_hyperparameter_component_factory_get_factory_info(self):
        """测试获取工厂信息"""
        info = HyperparameterComponentFactory.get_factory_info()
        assert isinstance(info, dict)
        assert "factory_name" in info
        assert "version" in info
        assert "total_hyperparameters" in info

    def test_hyperparameter_component_factory_invalid_hyperparameter_id(self):
        """测试无效的hyperparameter ID"""
        with pytest.raises(ValueError):
            HyperparameterComponentFactory.create_component(999)  # 999不在支持列表中
