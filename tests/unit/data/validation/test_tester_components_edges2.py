"""
边界测试：tester_components.py
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
from src.data.validation.tester_components import (
    ITesterComponent,
    TesterComponent,
    TesterComponentFactory,
    ComponentFactory
)


def test_itester_component_abstract():
    """测试 ITesterComponent（抽象接口）"""
    # 不能直接实例化抽象类
    with pytest.raises(TypeError):
        ITesterComponent()


def test_tester_component_init():
    """测试 TesterComponent（初始化）"""
    component = TesterComponent(4)
    
    assert component.tester_id == 4
    assert component.component_type == "Tester"
    assert component.component_name == "Tester_Component_4"
    assert isinstance(component.creation_time, datetime)


def test_tester_component_init_custom_type():
    """测试 TesterComponent（初始化，自定义类型）"""
    component = TesterComponent(9, "CustomTester")
    
    assert component.tester_id == 9
    assert component.component_type == "CustomTester"
    assert component.component_name == "CustomTester_Component_9"


def test_tester_component_get_tester_id():
    """测试 TesterComponent（获取tester ID）"""
    component = TesterComponent(14)
    
    assert component.get_tester_id() == 14


def test_tester_component_get_info():
    """测试 TesterComponent（获取组件信息）"""
    component = TesterComponent(19)
    info = component.get_info()
    
    assert info["tester_id"] == 19
    assert info["component_name"] == "Tester_Component_19"
    assert info["component_type"] == "Tester"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"


def test_tester_component_process_success():
    """测试 TesterComponent（处理数据，成功）"""
    component = TesterComponent(24)
    data = {"key": "value"}
    
    result = component.process(data)
    
    assert result["tester_id"] == 24
    assert result["status"] == "success"
    assert result["input_data"] == data
    assert "processed_at" in result


def test_tester_component_process_empty():
    """测试 TesterComponent（处理数据，空数据）"""
    component = TesterComponent(29)
    
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_tester_component_process_none():
    """测试 TesterComponent（处理数据，None）"""
    component = TesterComponent(34)
    
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_tester_component_process_exception():
    """测试 TesterComponent（处理数据，异常）"""
    component = TesterComponent(4)
    # 创建一个会导致异常的数据
    class BadData:
        def __str__(self):
            raise ValueError("Test error")
    
    bad_data = BadData()
    result = component.process(bad_data)
    
    # 验证至少返回了有效的结果
    assert "status" in result
    assert result["tester_id"] == 4


def test_tester_component_process_large_data():
    """测试 TesterComponent（处理数据，大数据）"""
    component = TesterComponent(9)
    large_data = {"key" + str(i): "value" * 1000 for i in range(1000)}
    
    result = component.process(large_data)
    
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_tester_component_process_nested_data():
    """测试 TesterComponent（处理数据，嵌套数据）"""
    component = TesterComponent(14)
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


def test_tester_component_get_status():
    """测试 TesterComponent（获取组件状态）"""
    component = TesterComponent(19)
    status = component.get_status()
    
    assert status["tester_id"] == 19
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_tester_component_factory_create_component_valid():
    """测试 TesterComponentFactory（创建组件，有效ID）"""
    component = TesterComponentFactory.create_component(4)
    
    assert isinstance(component, TesterComponent)
    assert component.tester_id == 4


def test_tester_component_factory_create_component_invalid():
    """测试 TesterComponentFactory（创建组件，无效ID）"""
    with pytest.raises(ValueError, match="不支持的tester ID"):
        TesterComponentFactory.create_component(99)


def test_tester_component_factory_create_component_negative():
    """测试 TesterComponentFactory（创建组件，负数ID）"""
    with pytest.raises(ValueError, match="不支持的tester ID"):
        TesterComponentFactory.create_component(-1)


def test_tester_component_factory_create_component_zero():
    """测试 TesterComponentFactory（创建组件，零ID）"""
    with pytest.raises(ValueError, match="不支持的tester ID"):
        TesterComponentFactory.create_component(0)


def test_tester_component_factory_get_available_testers():
    """测试 TesterComponentFactory（获取可用tester列表）"""
    testers = TesterComponentFactory.get_available_testers()
    
    assert isinstance(testers, list)
    assert len(testers) == 7
    assert 4 in testers
    assert 34 in testers
    assert testers == sorted(testers)


def test_tester_component_factory_create_all_testers():
    """测试 TesterComponentFactory（创建所有tester）"""
    all_testers = TesterComponentFactory.create_all_testers()
    
    assert isinstance(all_testers, dict)
    assert len(all_testers) == 7
    for tester_id, component in all_testers.items():
        assert isinstance(component, TesterComponent)
        assert component.tester_id == tester_id


def test_tester_component_factory_get_factory_info():
    """测试 TesterComponentFactory（获取工厂信息）"""
    info = TesterComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "TesterComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_testers"] == 7
    assert len(info["supported_ids"]) == 7
    assert "created_at" in info


def test_tester_component_factory_backward_compatible():
    """测试 TesterComponentFactory（向后兼容函数）"""
    from src.data.validation.tester_components import (
        create_tester_tester_component_4,
        create_tester_tester_component_9,
        create_tester_tester_component_14,
        create_tester_tester_component_19,
        create_tester_tester_component_24,
        create_tester_tester_component_29,
        create_tester_tester_component_34
    )
    
    component4 = create_tester_tester_component_4()
    assert component4.tester_id == 4
    
    component9 = create_tester_tester_component_9()
    assert component9.tester_id == 9


def test_component_factory_init():
    """测试 ComponentFactory（初始化）"""
    factory = ComponentFactory()
    
    assert factory._components == {}


def test_component_factory_create_component_none():
    """测试 ComponentFactory（创建组件，返回None）"""
    factory = ComponentFactory()
    
    result = factory.create_component("test", {})
    
    assert result is None


def test_tester_component_multiple_instances():
    """测试 TesterComponent（多个实例）"""
    component1 = TesterComponent(4)
    component2 = TesterComponent(9)
    
    assert component1.tester_id == 4
    assert component2.tester_id == 9
    assert component1.component_name != component2.component_name


def test_tester_component_same_id_different_types():
    """测试 TesterComponent（相同ID，不同类型）"""
    component1 = TesterComponent(14, "Type1")
    component2 = TesterComponent(14, "Type2")
    
    assert component1.tester_id == component2.tester_id
    assert component1.component_type != component2.component_type
    assert component1.component_name != component2.component_name
