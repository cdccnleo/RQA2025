"""
测试搜索组件
"""

import pytest
from datetime import datetime
from src.ml.tuning.search_components import (
    ComponentFactory,
    ISearchComponent,
    SearchComponent,
    SearchComponentFactory
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


class MockSearchComponent(SearchComponent):
    """用于测试的搜索组件模拟实现"""

    def __init__(self, search_id=1, component_type="MockSearch"):
        super().__init__(search_id, component_type)

    def get_info(self):
        """获取组件信息"""
        return {
            "search_id": self.search_id,
            "component_type": self.component_type,
            "type": "mock",
            "created_at": datetime.now().isoformat()
        }

    def process(self, data):
        """处理数据"""
        return {
            "processed": True,
            "input_data": data,
            "search_id": self.search_id
        }

    def get_status(self):
        """获取组件状态"""
        return {
            "status": "active",
            "search_id": self.search_id,
            "last_updated": datetime.now().isoformat()
        }


class TestISearchComponent:
    """测试搜索组件接口"""

    def test_isearch_component_is_abstract(self):
        """测试ISearchComponent是抽象类"""
        import inspect
        assert inspect.isabstract(ISearchComponent)

        # 检查抽象方法
        assert hasattr(ISearchComponent, 'get_info')
        assert hasattr(ISearchComponent, 'process')
        assert hasattr(ISearchComponent, 'get_status')
        assert hasattr(ISearchComponent, 'get_search_id')

        # 验证无法实例化
        with pytest.raises(TypeError):
            ISearchComponent()


class TestSearchComponent:
    """测试搜索组件"""

    def setup_method(self):
        """测试前准备"""
        self.component = MockSearchComponent(
            search_id=654,
            component_type="TestSearchComponent"
        )

    def test_search_component_init(self):
        """测试搜索组件初始化"""
        assert self.component.search_id == 654
        assert self.component.component_type == "TestSearchComponent"
        assert self.component.component_name == "TestSearchComponent_Component_654"

    def test_search_component_get_search_id(self):
        """测试获取search ID"""
        assert self.component.get_search_id() == 654

    def test_search_component_get_info(self):
        """测试获取组件信息"""
        info = self.component.get_info()
        assert isinstance(info, dict)
        assert info["search_id"] == 654
        assert info["component_type"] == "TestSearchComponent"
        assert info["type"] == "mock"
        assert "created_at" in info

    def test_search_component_process(self):
        """测试处理数据"""
        test_data = {"input": "test", "value": 123}
        result = self.component.process(test_data)
        assert isinstance(result, dict)
        assert result["processed"] == True
        assert result["input_data"] == test_data
        assert result["search_id"] == 654

    def test_search_component_get_status(self):
        """测试获取组件状态"""
        status = self.component.get_status()
        assert isinstance(status, dict)
        assert status["status"] == "active"
        assert status["search_id"] == 654
        assert "last_updated" in status


class TestSearchComponentFactory:
    """测试搜索组件工厂"""

    def test_search_component_factory_supported_ids(self):
        """测试支持的search ID"""
        supported_ids = SearchComponentFactory.SUPPORTED_SEARCH_IDS
        assert isinstance(supported_ids, list)
        assert len(supported_ids) > 0
        assert all(isinstance(id, int) for id in supported_ids)

    def test_search_component_factory_create_component(self):
        """测试创建搜索组件"""
        component = SearchComponentFactory.create_component(4)
        assert isinstance(component, SearchComponent)
        assert component.search_id == 4

    def test_search_component_factory_get_available_searches(self):
        """测试获取可用search"""
        available_ids = SearchComponentFactory.get_available_searches()
        assert isinstance(available_ids, list)
        assert len(available_ids) > 0
        assert 4 in available_ids  # 4是支持的ID之一

    def test_search_component_factory_create_all_searches(self):
        """测试创建所有search"""
        all_components = SearchComponentFactory.create_all_searches()
        assert isinstance(all_components, dict)
        assert len(all_components) > 0
        assert all(isinstance(comp, SearchComponent) for comp in all_components.values())

    def test_search_component_factory_get_factory_info(self):
        """测试获取工厂信息"""
        info = SearchComponentFactory.get_factory_info()
        assert isinstance(info, dict)
        assert "factory_name" in info
        assert "version" in info
        assert "total_searches" in info

    def test_search_component_factory_invalid_search_id(self):
        """测试无效的search ID"""
        with pytest.raises(ValueError):
            SearchComponentFactory.create_component(999)  # 999不在支持列表中
