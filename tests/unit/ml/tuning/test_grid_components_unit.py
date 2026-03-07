"""
测试网格组件
"""

import pytest
from datetime import datetime
from src.ml.tuning.grid_components import (
    ComponentFactory,
    IGridComponent,
    GridComponent,
    GridComponentFactory
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


class MockGridComponent(GridComponent):
    """用于测试的网格组件模拟实现"""

    def __init__(self, grid_id=1, component_type="MockGrid"):
        super().__init__(grid_id, component_type)

    def get_info(self):
        """获取组件信息"""
        return {
            "grid_id": self.grid_id,
            "component_type": self.component_type,
            "type": "mock",
            "created_at": datetime.now().isoformat()
        }

    def process(self, data):
        """处理数据"""
        return {
            "processed": True,
            "input_data": data,
            "grid_id": self.grid_id
        }

    def get_status(self):
        """获取组件状态"""
        return {
            "status": "active",
            "grid_id": self.grid_id,
            "last_updated": datetime.now().isoformat()
        }


class TestIGridComponent:
    """测试网格组件接口"""

    def test_igrid_component_is_abstract(self):
        """测试IGridComponent是抽象类"""
        import inspect
        assert inspect.isabstract(IGridComponent)

        # 检查抽象方法
        assert hasattr(IGridComponent, 'get_info')
        assert hasattr(IGridComponent, 'process')
        assert hasattr(IGridComponent, 'get_status')
        assert hasattr(IGridComponent, 'get_grid_id')

        # 验证无法实例化
        with pytest.raises(TypeError):
            IGridComponent()


class TestGridComponent:
    """测试网格组件"""

    def setup_method(self):
        """测试前准备"""
        self.component = MockGridComponent(
            grid_id=456,
            component_type="TestGridComponent"
        )

    def test_grid_component_init(self):
        """测试网格组件初始化"""
        assert self.component.grid_id == 456
        assert self.component.component_type == "TestGridComponent"
        assert self.component.component_name == "TestGridComponent_Component_456"

    def test_grid_component_get_grid_id(self):
        """测试获取grid ID"""
        assert self.component.get_grid_id() == 456

    def test_grid_component_get_info(self):
        """测试获取组件信息"""
        info = self.component.get_info()
        assert isinstance(info, dict)
        assert info["grid_id"] == 456
        assert info["component_type"] == "TestGridComponent"
        assert info["type"] == "mock"
        assert "created_at" in info

    def test_grid_component_process(self):
        """测试处理数据"""
        test_data = {"input": "test", "value": 123}
        result = self.component.process(test_data)
        assert isinstance(result, dict)
        assert result["processed"] == True
        assert result["input_data"] == test_data
        assert result["grid_id"] == 456

    def test_grid_component_get_status(self):
        """测试获取组件状态"""
        status = self.component.get_status()
        assert isinstance(status, dict)
        assert status["status"] == "active"
        assert status["grid_id"] == 456
        assert "last_updated" in status


class TestGridComponentFactory:
    """测试网格组件工厂"""

    def test_grid_component_factory_supported_ids(self):
        """测试支持的grid ID"""
        supported_ids = GridComponentFactory.SUPPORTED_GRID_IDS
        assert isinstance(supported_ids, list)
        assert len(supported_ids) > 0
        assert all(isinstance(id, int) for id in supported_ids)

    def test_grid_component_factory_create_component(self):
        """测试创建网格组件"""
        component = GridComponentFactory.create_component(5)
        assert isinstance(component, GridComponent)
        assert component.grid_id == 5

    def test_grid_component_factory_get_available_grids(self):
        """测试获取可用grid"""
        available_ids = GridComponentFactory.get_available_grids()
        assert isinstance(available_ids, list)
        assert len(available_ids) > 0
        assert 5 in available_ids  # 5是支持的ID之一

    def test_grid_component_factory_create_all_grids(self):
        """测试创建所有grid"""
        all_components = GridComponentFactory.create_all_grids()
        assert isinstance(all_components, dict)
        assert len(all_components) > 0
        assert all(isinstance(comp, GridComponent) for comp in all_components.values())

    def test_grid_component_factory_get_factory_info(self):
        """测试获取工厂信息"""
        info = GridComponentFactory.get_factory_info()
        assert isinstance(info, dict)
        assert "factory_name" in info
        assert "version" in info
        assert "total_grids" in info

    def test_grid_component_factory_invalid_grid_id(self):
        """测试无效的grid ID"""
        with pytest.raises(ValueError):
            GridComponentFactory.create_component(999)  # 999不在支持列表中
