"""
边界测试：checker_components.py
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
from src.data.validation.checker_components import (
    ICheckerComponent,
    CheckerComponent,
    CheckerComponentFactory,
    ComponentFactory
)


def test_ichecker_component_abstract():
    """测试 ICheckerComponent（抽象接口）"""
    # 不能直接实例化抽象类
    with pytest.raises(TypeError):
        ICheckerComponent()


def test_checker_component_init():
    """测试 CheckerComponent（初始化）"""
    component = CheckerComponent(2)
    
    assert component.checker_id == 2
    assert component.component_type == "Checker"
    assert component.component_name == "Checker_Component_2"
    assert isinstance(component.creation_time, datetime)


def test_checker_component_init_custom_type():
    """测试 CheckerComponent（初始化，自定义类型）"""
    component = CheckerComponent(7, "CustomChecker")
    
    assert component.checker_id == 7
    assert component.component_type == "CustomChecker"
    assert component.component_name == "CustomChecker_Component_7"


def test_checker_component_get_checker_id():
    """测试 CheckerComponent（获取checker ID）"""
    component = CheckerComponent(12)
    
    assert component.get_checker_id() == 12


def test_checker_component_get_info():
    """测试 CheckerComponent（获取组件信息）"""
    component = CheckerComponent(17)
    info = component.get_info()
    
    assert info["checker_id"] == 17
    assert info["component_name"] == "Checker_Component_17"
    assert info["component_type"] == "Checker"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"


def test_checker_component_process_success():
    """测试 CheckerComponent（处理数据，成功）"""
    component = CheckerComponent(22)
    data = {"key": "value"}
    
    result = component.process(data)
    
    assert result["checker_id"] == 22
    assert result["status"] == "success"
    assert result["input_data"] == data
    assert "processed_at" in result


def test_checker_component_process_empty():
    """测试 CheckerComponent（处理数据，空数据）"""
    component = CheckerComponent(27)
    
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_checker_component_process_none():
    """测试 CheckerComponent（处理数据，None）"""
    component = CheckerComponent(32)
    
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_checker_component_process_exception():
    """测试 CheckerComponent（处理数据，异常）"""
    component = CheckerComponent(2)
    # 创建一个会导致异常的数据
    class BadData:
        def __str__(self):
            raise ValueError("Test error")
    
    bad_data = BadData()
    result = component.process(bad_data)
    
    # 验证至少返回了有效的结果
    assert "status" in result
    assert result["checker_id"] == 2


def test_checker_component_process_large_data():
    """测试 CheckerComponent（处理数据，大数据）"""
    component = CheckerComponent(7)
    large_data = {"key" + str(i): "value" * 1000 for i in range(1000)}
    
    result = component.process(large_data)
    
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_checker_component_process_nested_data():
    """测试 CheckerComponent（处理数据，嵌套数据）"""
    component = CheckerComponent(12)
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


def test_checker_component_get_status():
    """测试 CheckerComponent（获取组件状态）"""
    component = CheckerComponent(17)
    status = component.get_status()
    
    assert status["checker_id"] == 17
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_checker_component_factory_create_component_valid():
    """测试 CheckerComponentFactory（创建组件，有效ID）"""
    component = CheckerComponentFactory.create_component(2)
    
    assert isinstance(component, CheckerComponent)
    assert component.checker_id == 2


def test_checker_component_factory_create_component_invalid():
    """测试 CheckerComponentFactory（创建组件，无效ID）"""
    with pytest.raises(ValueError, match="不支持的checker ID"):
        CheckerComponentFactory.create_component(99)


def test_checker_component_factory_create_component_negative():
    """测试 CheckerComponentFactory（创建组件，负数ID）"""
    with pytest.raises(ValueError, match="不支持的checker ID"):
        CheckerComponentFactory.create_component(-1)


def test_checker_component_factory_create_component_zero():
    """测试 CheckerComponentFactory（创建组件，零ID）"""
    with pytest.raises(ValueError, match="不支持的checker ID"):
        CheckerComponentFactory.create_component(0)


def test_checker_component_factory_get_available_checkers():
    """测试 CheckerComponentFactory（获取可用checker列表）"""
    checkers = CheckerComponentFactory.get_available_checkers()
    
    assert isinstance(checkers, list)
    assert len(checkers) == 7
    assert 2 in checkers
    assert 32 in checkers
    assert checkers == sorted(checkers)


def test_checker_component_factory_create_all_checkers():
    """测试 CheckerComponentFactory（创建所有checker）"""
    all_checkers = CheckerComponentFactory.create_all_checkers()
    
    assert isinstance(all_checkers, dict)
    assert len(all_checkers) == 7
    for checker_id, component in all_checkers.items():
        assert isinstance(component, CheckerComponent)
        assert component.checker_id == checker_id


def test_checker_component_factory_get_factory_info():
    """测试 CheckerComponentFactory（获取工厂信息）"""
    info = CheckerComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "CheckerComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_checkers"] == 7
    assert len(info["supported_ids"]) == 7
    assert "created_at" in info


def test_checker_component_factory_backward_compatible():
    """测试 CheckerComponentFactory（向后兼容函数）"""
    from src.data.validation.checker_components import (
        create_checker_checker_component_2,
        create_checker_checker_component_7,
        create_checker_checker_component_12,
        create_checker_checker_component_17,
        create_checker_checker_component_22,
        create_checker_checker_component_27,
        create_checker_checker_component_32
    )
    
    component2 = create_checker_checker_component_2()
    assert component2.checker_id == 2
    
    component7 = create_checker_checker_component_7()
    assert component7.checker_id == 7


def test_component_factory_init():
    """测试 ComponentFactory（初始化）"""
    factory = ComponentFactory()
    
    assert factory._components == {}


def test_component_factory_create_component_none():
    """测试 ComponentFactory（创建组件，返回None）"""
    factory = ComponentFactory()
    
    result = factory.create_component("test", {})
    
    assert result is None


def test_checker_component_multiple_instances():
    """测试 CheckerComponent（多个实例）"""
    component1 = CheckerComponent(2)
    component2 = CheckerComponent(7)
    
    assert component1.checker_id == 2
    assert component2.checker_id == 7
    assert component1.component_name != component2.component_name


def test_checker_component_same_id_different_types():
    """测试 CheckerComponent（相同ID，不同类型）"""
    component1 = CheckerComponent(12, "Type1")
    component2 = CheckerComponent(12, "Type2")
    
    assert component1.checker_id == component2.checker_id
    assert component1.component_type != component2.component_type
    assert component1.component_name != component2.component_name
