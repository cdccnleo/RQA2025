"""
边界测试：assertion_components.py
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
from src.data.validation.assertion_components import (
    IAssertionComponent,
    AssertionComponent,
    AssertionComponentFactory,
    ComponentFactory
)


def test_iassertion_component_abstract():
    """测试 IAssertionComponent（抽象接口）"""
    # 不能直接实例化抽象类
    with pytest.raises(TypeError):
        IAssertionComponent()


def test_assertion_component_init():
    """测试 AssertionComponent（初始化）"""
    component = AssertionComponent(5)
    
    assert component.assertion_id == 5
    assert component.component_type == "Assertion"
    assert component.component_name == "Assertion_Component_5"
    assert isinstance(component.creation_time, datetime)


def test_assertion_component_init_custom_type():
    """测试 AssertionComponent（初始化，自定义类型）"""
    component = AssertionComponent(10, "CustomAssertion")
    
    assert component.assertion_id == 10
    assert component.component_type == "CustomAssertion"
    assert component.component_name == "CustomAssertion_Component_10"


def test_assertion_component_get_assertion_id():
    """测试 AssertionComponent（获取assertion ID）"""
    component = AssertionComponent(15)
    
    assert component.get_assertion_id() == 15


def test_assertion_component_get_info():
    """测试 AssertionComponent（获取组件信息）"""
    component = AssertionComponent(20)
    info = component.get_info()
    
    assert info["assertion_id"] == 20
    assert info["component_name"] == "Assertion_Component_20"
    assert info["component_type"] == "Assertion"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"


def test_assertion_component_process_success():
    """测试 AssertionComponent（处理数据，成功）"""
    component = AssertionComponent(25)
    data = {"key": "value"}
    
    result = component.process(data)
    
    assert result["assertion_id"] == 25
    assert result["status"] == "success"
    assert result["input_data"] == data
    assert "processed_at" in result


def test_assertion_component_process_empty():
    """测试 AssertionComponent（处理数据，空数据）"""
    component = AssertionComponent(30)
    
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_assertion_component_process_none():
    """测试 AssertionComponent（处理数据，None）"""
    component = AssertionComponent(5)
    
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_assertion_component_process_exception():
    """测试 AssertionComponent（处理数据，异常）"""
    component = AssertionComponent(10)
    # 创建一个会导致异常的数据
    class BadData:
        def __str__(self):
            raise ValueError("Test error")
    
    bad_data = BadData()
    result = component.process(bad_data)
    
    # 验证至少返回了有效的结果
    assert "status" in result
    assert result["assertion_id"] == 10


def test_assertion_component_process_large_data():
    """测试 AssertionComponent（处理数据，大数据）"""
    component = AssertionComponent(15)
    large_data = {"key" + str(i): "value" * 1000 for i in range(1000)}
    
    result = component.process(large_data)
    
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_assertion_component_process_nested_data():
    """测试 AssertionComponent（处理数据，嵌套数据）"""
    component = AssertionComponent(20)
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


def test_assertion_component_get_status():
    """测试 AssertionComponent（获取组件状态）"""
    component = AssertionComponent(25)
    status = component.get_status()
    
    assert status["assertion_id"] == 25
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_assertion_component_factory_create_component_valid():
    """测试 AssertionComponentFactory（创建组件，有效ID）"""
    component = AssertionComponentFactory.create_component(5)
    
    assert isinstance(component, AssertionComponent)
    assert component.assertion_id == 5


def test_assertion_component_factory_create_component_invalid():
    """测试 AssertionComponentFactory（创建组件，无效ID）"""
    with pytest.raises(ValueError, match="不支持的assertion ID"):
        AssertionComponentFactory.create_component(99)


def test_assertion_component_factory_create_component_negative():
    """测试 AssertionComponentFactory（创建组件，负数ID）"""
    with pytest.raises(ValueError, match="不支持的assertion ID"):
        AssertionComponentFactory.create_component(-1)


def test_assertion_component_factory_create_component_zero():
    """测试 AssertionComponentFactory（创建组件，零ID）"""
    with pytest.raises(ValueError, match="不支持的assertion ID"):
        AssertionComponentFactory.create_component(0)


def test_assertion_component_factory_get_available_assertions():
    """测试 AssertionComponentFactory（获取可用assertion列表）"""
    assertions = AssertionComponentFactory.get_available_assertions()
    
    assert isinstance(assertions, list)
    assert len(assertions) == 6
    assert 5 in assertions
    assert 30 in assertions
    assert assertions == sorted(assertions)


def test_assertion_component_factory_create_all_assertions():
    """测试 AssertionComponentFactory（创建所有assertion）"""
    all_assertions = AssertionComponentFactory.create_all_assertions()
    
    assert isinstance(all_assertions, dict)
    assert len(all_assertions) == 6
    for assertion_id, component in all_assertions.items():
        assert isinstance(component, AssertionComponent)
        assert component.assertion_id == assertion_id


def test_assertion_component_factory_get_factory_info():
    """测试 AssertionComponentFactory（获取工厂信息）"""
    info = AssertionComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "AssertionComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_assertions"] == 6
    assert len(info["supported_ids"]) == 6
    assert "created_at" in info


def test_assertion_component_factory_backward_compatible():
    """测试 AssertionComponentFactory（向后兼容函数）"""
    from src.data.validation.assertion_components import (
        create_assertion_assertion_component_5,
        create_assertion_assertion_component_10,
        create_assertion_assertion_component_15,
        create_assertion_assertion_component_20,
        create_assertion_assertion_component_25,
        create_assertion_assertion_component_30
    )
    
    component5 = create_assertion_assertion_component_5()
    assert component5.assertion_id == 5
    
    component10 = create_assertion_assertion_component_10()
    assert component10.assertion_id == 10


def test_component_factory_init():
    """测试 ComponentFactory（初始化）"""
    factory = ComponentFactory()
    
    assert factory._components == {}


def test_component_factory_create_component_none():
    """测试 ComponentFactory（创建组件，返回None）"""
    factory = ComponentFactory()
    
    result = factory.create_component("test", {})
    
    assert result is None


def test_assertion_component_multiple_instances():
    """测试 AssertionComponent（多个实例）"""
    component1 = AssertionComponent(5)
    component2 = AssertionComponent(10)
    
    assert component1.assertion_id == 5
    assert component2.assertion_id == 10
    assert component1.component_name != component2.component_name


def test_assertion_component_same_id_different_types():
    """测试 AssertionComponent（相同ID，不同类型）"""
    component1 = AssertionComponent(15, "Type1")
    component2 = AssertionComponent(15, "Type2")
    
    assert component1.assertion_id == component2.assertion_id
    assert component1.component_type != component2.component_type
    assert component1.component_name != component2.component_name
