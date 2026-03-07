"""
测试优化器组件
"""

import pytest
from datetime import datetime
from src.ml.tuning.optimizer_components import (
    ComponentFactory,
    IOptimizerComponent,
    OptimizerComponent,
    MLTuningOptimizerComponentFactory
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


class MockOptimizerComponent(OptimizerComponent):
    """用于测试的优化器组件模拟实现"""

    def __init__(self, optimizer_id=1, component_type="MockOptimizer"):
        super().__init__(optimizer_id, component_type)

    def get_info(self):
        """获取组件信息"""
        return {
            "optimizer_id": self.optimizer_id,
            "component_type": self.component_type,
            "type": "mock",
            "created_at": datetime.now().isoformat()
        }

    def process(self, data):
        """处理数据"""
        return {
            "processed": True,
            "input_data": data,
            "optimizer_id": self.optimizer_id
        }

    def get_status(self):
        """获取组件状态"""
        return {
            "status": "active",
            "optimizer_id": self.optimizer_id,
            "last_updated": datetime.now().isoformat()
        }


class TestIOptimizerComponent:
    """测试优化器组件接口"""

    def test_ioptimizer_component_is_abstract(self):
        """测试IOptimizerComponent是抽象类"""
        import inspect
        assert inspect.isabstract(IOptimizerComponent)

        # 检查抽象方法
        assert hasattr(IOptimizerComponent, 'get_info')
        assert hasattr(IOptimizerComponent, 'process')
        assert hasattr(IOptimizerComponent, 'get_status')

        # 验证无法实例化
        with pytest.raises(TypeError):
            IOptimizerComponent()


class TestOptimizerComponent:
    """测试优化器组件"""

    def setup_method(self):
        """测试前准备"""
        self.component = MockOptimizerComponent(
            optimizer_id=789,
            component_type="TestOptimizerComponent"
        )

    def test_optimizer_component_init(self):
        """测试优化器组件初始化"""
        assert self.component.optimizer_id == 789
        assert self.component.component_type == "TestOptimizerComponent"
        assert self.component.component_name == "TestOptimizerComponent_Component_789"

    def test_optimizer_component_get_optimizer_id(self):
        """测试获取优化器ID"""
        assert self.component.get_optimizer_id() == 789

    def test_optimizer_component_get_info(self):
        """测试获取组件信息"""
        info = self.component.get_info()
        assert isinstance(info, dict)
        assert info["optimizer_id"] == 789
        assert info["component_type"] == "TestOptimizerComponent"
        assert info["type"] == "mock"
        assert "created_at" in info

    def test_optimizer_component_process(self):
        """测试处理数据"""
        test_data = {"input": "test", "value": 123}
        result = self.component.process(test_data)
        assert isinstance(result, dict)
        assert result["processed"] == True
        assert result["input_data"] == test_data
        assert result["optimizer_id"] == 789

    def test_optimizer_component_get_status(self):
        """测试获取组件状态"""
        status = self.component.get_status()
        assert isinstance(status, dict)
        assert status["status"] == "active"
        assert status["optimizer_id"] == 789
        assert "last_updated" in status


class TestMLTuningOptimizerComponentFactory:
    """测试ML调优优化器组件工厂"""

    def test_mltuning_optimizer_component_factory_supported_ids(self):
        """测试支持的优化器ID"""
        supported_ids = MLTuningOptimizerComponentFactory.SUPPORTED_OPTIMIZER_IDS
        assert isinstance(supported_ids, list)
        assert len(supported_ids) > 0
        assert all(isinstance(id, int) for id in supported_ids)

    def test_mltuning_optimizer_component_factory_create_component(self):
        """测试创建ML调优优化器组件"""
        component = MLTuningOptimizerComponentFactory.create_component(2)
        assert isinstance(component, OptimizerComponent)
        assert component.optimizer_id == 2

    def test_mltuning_optimizer_component_factory_get_available_optimizers(self):
        """测试获取可用优化器"""
        available_ids = MLTuningOptimizerComponentFactory.get_available_optimizers()
        assert isinstance(available_ids, list)
        assert len(available_ids) > 0
        assert 2 in available_ids  # 2是支持的ID之一

    def test_mltuning_optimizer_component_factory_create_all_optimizers(self):
        """测试创建所有优化器"""
        all_components = MLTuningOptimizerComponentFactory.create_all_optimizers()
        assert isinstance(all_components, dict)
        assert len(all_components) > 0
        assert all(isinstance(comp, OptimizerComponent) for comp in all_components.values())

    def test_mltuning_optimizer_component_factory_get_factory_info(self):
        """测试获取工厂信息"""
        info = MLTuningOptimizerComponentFactory.get_factory_info()
        assert isinstance(info, dict)
        assert "factory_name" in info
        assert "version" in info
        assert "total_optimizers" in info

    def test_mltuning_optimizer_component_factory_invalid_optimizer_id(self):
        """测试无效的优化器ID"""
        with pytest.raises(ValueError):
            MLTuningOptimizerComponentFactory.create_component(999)  # 999不在支持列表中
