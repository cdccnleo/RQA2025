"""
边界测试：filter_components.py
测试边界情况和异常场景
"""
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
from datetime import datetime
from unittest.mock import MagicMock
from src.data.processing.filter_components import (
    IFilterComponent,
    FilterComponent,
    FilterComponentFactory,
    ComponentFactory
)


def test_ifilter_component_abstract():
    """测试 IFilterComponent（抽象接口）"""
    # 不能直接实例化抽象类
    with pytest.raises(TypeError):
        IFilterComponent()


def test_filter_component_init():
    """测试 FilterComponent（初始化）"""
    component = FilterComponent(5)
    
    assert component.filter_id == 5
    assert component.component_type == "Filter"
    assert component.component_name == "Filter_Component_5"
    assert isinstance(component.creation_time, datetime)


def test_filter_component_init_custom_type():
    """测试 FilterComponent（初始化，自定义类型）"""
    component = FilterComponent(10, "CustomFilter")
    
    assert component.filter_id == 10
    assert component.component_type == "CustomFilter"
    assert component.component_name == "CustomFilter_Component_10"


def test_filter_component_get_filter_id():
    """测试 FilterComponent（获取filter ID）"""
    component = FilterComponent(15)
    
    assert component.get_filter_id() == 15


def test_filter_component_get_info():
    """测试 FilterComponent（获取组件信息）"""
    component = FilterComponent(20)
    info = component.get_info()
    
    assert info["filter_id"] == 20
    assert info["component_name"] == "Filter_Component_20"
    assert info["component_type"] == "Filter"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"


def test_filter_component_process_success():
    """测试 FilterComponent（处理数据，成功）"""
    component = FilterComponent(25)
    data = {"key": "value"}
    
    result = component.process(data)
    
    assert result["filter_id"] == 25
    assert result["status"] == "success"
    assert result["input_data"] == data
    assert "processed_at" in result


def test_filter_component_process_empty():
    """测试 FilterComponent（处理数据，空数据）"""
    component = FilterComponent(30)
    
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_filter_component_process_none():
    """测试 FilterComponent（处理数据，None）"""
    component = FilterComponent(35)
    
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_filter_component_process_exception():
    """测试 FilterComponent（处理数据，异常）"""
    component = FilterComponent(5)
    # 创建一个会导致异常的数据
    class BadData:
        def __str__(self):
            raise ValueError("Test error")
    
    bad_data = BadData()
    result = component.process(bad_data)
    
    # 验证至少返回了有效的结果
    assert "status" in result
    assert result["filter_id"] == 5


def test_filter_component_process_large_data():
    """测试 FilterComponent（处理数据，大数据）"""
    component = FilterComponent(10)
    large_data = {"key" + str(i): "value" * 1000 for i in range(1000)}
    
    result = component.process(large_data)
    
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_filter_component_process_nested_data():
    """测试 FilterComponent（处理数据，嵌套数据）"""
    component = FilterComponent(15)
    nested_data = {
        "level1": {
            "level2": {
                "level3": "value"
            }
        }
    }
    
    result = component.process(nested_data)
    
    assert result["status"] == "success"
    assert result["input_data"] == nested_data


def test_filter_component_get_status():
    """测试 FilterComponent（获取组件状态）"""
    component = FilterComponent(20)
    status = component.get_status()
    
    assert status["filter_id"] == 20
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_filter_component_factory_create_component_valid():
    """测试 FilterComponentFactory（创建组件，有效ID）"""
    component = FilterComponentFactory.create_component(5)
    
    assert isinstance(component, FilterComponent)
    assert component.filter_id == 5


def test_filter_component_factory_create_component_invalid():
    """测试 FilterComponentFactory（创建组件，无效ID）"""
    with pytest.raises(ValueError, match="不支持的filter ID"):
        FilterComponentFactory.create_component(99)


def test_filter_component_factory_create_component_negative():
    """测试 FilterComponentFactory（创建组件，负数ID）"""
    with pytest.raises(ValueError, match="不支持的filter ID"):
        FilterComponentFactory.create_component(-1)


def test_filter_component_factory_create_component_zero():
    """测试 FilterComponentFactory（创建组件，零ID）"""
    with pytest.raises(ValueError, match="不支持的filter ID"):
        FilterComponentFactory.create_component(0)


def test_filter_component_factory_get_available_filters():
    """测试 FilterComponentFactory（获取可用filter列表）"""
    filters = FilterComponentFactory.get_available_filters()
    
    assert isinstance(filters, list)
    assert len(filters) == 7
    assert 5 in filters
    assert 35 in filters
    assert filters == sorted(filters)


def test_filter_component_factory_create_all_filters():
    """测试 FilterComponentFactory（创建所有filter）"""
    all_filters = FilterComponentFactory.create_all_filters()
    
    assert isinstance(all_filters, dict)
    assert len(all_filters) == 7
    for filter_id, component in all_filters.items():
        assert isinstance(component, FilterComponent)
        assert component.filter_id == filter_id


def test_filter_component_factory_get_factory_info():
    """测试 FilterComponentFactory（获取工厂信息）"""
    info = FilterComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "FilterComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_filters"] == 7
    assert len(info["supported_ids"]) == 7
    assert "created_at" in info


def test_filter_component_factory_backward_compatible():
    """测试 FilterComponentFactory（向后兼容函数）"""
    from src.data.processing.filter_components import (
        create_filter_filter_component_5,
        create_filter_filter_component_10,
        create_filter_filter_component_15,
        create_filter_filter_component_20,
        create_filter_filter_component_25,
        create_filter_filter_component_30,
        create_filter_filter_component_35
    )
    
    component5 = create_filter_filter_component_5()
    assert component5.filter_id == 5
    
    component10 = create_filter_filter_component_10()
    assert component10.filter_id == 10


def test_component_factory_init():
    """测试 ComponentFactory（初始化）"""
    factory = ComponentFactory()
    
    assert factory._components == {}


def test_component_factory_create_component_none():
    """测试 ComponentFactory（创建组件，返回None）"""
    factory = ComponentFactory()
    
    result = factory.create_component("test", {})
    
    assert result is None


def test_filter_component_multiple_instances():
    """测试 FilterComponent（多个实例）"""
    component1 = FilterComponent(5)
    component2 = FilterComponent(10)
    
    assert component1.filter_id == 5
    assert component2.filter_id == 10
    assert component1.component_name != component2.component_name


def test_filter_component_same_id_different_types():
    """测试 FilterComponent（相同ID，不同类型）"""
    component1 = FilterComponent(15, "Type1")
    component2 = FilterComponent(15, "Type2")
    
    assert component1.filter_id == component2.filter_id
    assert component1.component_type != component2.component_type
    assert component1.component_name != component2.component_name

