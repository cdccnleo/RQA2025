"""
边界测试：verifier_components.py
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
from src.data.validation.verifier_components import (
    IVerifierComponent,
    VerifierComponent,
    VerifierComponentFactory,
    ComponentFactory
)


def test_iverifier_component_abstract():
    """测试 IVerifierComponent（抽象接口）"""
    # 不能直接实例化抽象类
    with pytest.raises(TypeError):
        IVerifierComponent()


def test_verifier_component_init():
    """测试 VerifierComponent（初始化）"""
    component = VerifierComponent(3)
    
    assert component.verifier_id == 3
    assert component.component_type == "Verifier"
    assert component.component_name == "Verifier_Component_3"
    assert isinstance(component.creation_time, datetime)


def test_verifier_component_init_custom_type():
    """测试 VerifierComponent（初始化，自定义类型）"""
    component = VerifierComponent(8, "CustomVerifier")
    
    assert component.verifier_id == 8
    assert component.component_type == "CustomVerifier"
    assert component.component_name == "CustomVerifier_Component_8"


def test_verifier_component_get_verifier_id():
    """测试 VerifierComponent（获取verifier ID）"""
    component = VerifierComponent(13)
    
    assert component.get_verifier_id() == 13


def test_verifier_component_get_info():
    """测试 VerifierComponent（获取组件信息）"""
    component = VerifierComponent(18)
    info = component.get_info()
    
    assert info["verifier_id"] == 18
    assert info["component_name"] == "Verifier_Component_18"
    assert info["component_type"] == "Verifier"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"


def test_verifier_component_process_success():
    """测试 VerifierComponent（处理数据，成功）"""
    component = VerifierComponent(23)
    data = {"key": "value"}
    
    result = component.process(data)
    
    assert result["verifier_id"] == 23
    assert result["status"] == "success"
    assert result["input_data"] == data
    assert "processed_at" in result


def test_verifier_component_process_empty():
    """测试 VerifierComponent（处理数据，空数据）"""
    component = VerifierComponent(28)
    
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_verifier_component_process_none():
    """测试 VerifierComponent（处理数据，None）"""
    component = VerifierComponent(33)
    
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_verifier_component_process_exception():
    """测试 VerifierComponent（处理数据，异常）"""
    component = VerifierComponent(3)
    # 创建一个会导致异常的数据
    class BadData:
        def __str__(self):
            raise ValueError("Test error")
    
    bad_data = BadData()
    result = component.process(bad_data)
    
    # 验证至少返回了有效的结果
    assert "status" in result
    assert result["verifier_id"] == 3


def test_verifier_component_process_large_data():
    """测试 VerifierComponent（处理数据，大数据）"""
    component = VerifierComponent(8)
    large_data = {"key" + str(i): "value" * 1000 for i in range(1000)}
    
    result = component.process(large_data)
    
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_verifier_component_process_nested_data():
    """测试 VerifierComponent（处理数据，嵌套数据）"""
    component = VerifierComponent(13)
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


def test_verifier_component_get_status():
    """测试 VerifierComponent（获取组件状态）"""
    component = VerifierComponent(18)
    status = component.get_status()
    
    assert status["verifier_id"] == 18
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_verifier_component_factory_create_component_valid():
    """测试 VerifierComponentFactory（创建组件，有效ID）"""
    component = VerifierComponentFactory.create_component(3)
    
    assert isinstance(component, VerifierComponent)
    assert component.verifier_id == 3


def test_verifier_component_factory_create_component_invalid():
    """测试 VerifierComponentFactory（创建组件，无效ID）"""
    with pytest.raises(ValueError, match="不支持的verifier ID"):
        VerifierComponentFactory.create_component(99)


def test_verifier_component_factory_create_component_negative():
    """测试 VerifierComponentFactory（创建组件，负数ID）"""
    with pytest.raises(ValueError, match="不支持的verifier ID"):
        VerifierComponentFactory.create_component(-1)


def test_verifier_component_factory_create_component_zero():
    """测试 VerifierComponentFactory（创建组件，零ID）"""
    with pytest.raises(ValueError, match="不支持的verifier ID"):
        VerifierComponentFactory.create_component(0)


def test_verifier_component_factory_get_available_verifiers():
    """测试 VerifierComponentFactory（获取可用verifier列表）"""
    verifiers = VerifierComponentFactory.get_available_verifiers()
    
    assert isinstance(verifiers, list)
    assert len(verifiers) == 7
    assert 3 in verifiers
    assert 33 in verifiers
    assert verifiers == sorted(verifiers)


def test_verifier_component_factory_create_all_verifiers():
    """测试 VerifierComponentFactory（创建所有verifier）"""
    all_verifiers = VerifierComponentFactory.create_all_verifiers()
    
    assert isinstance(all_verifiers, dict)
    assert len(all_verifiers) == 7
    for verifier_id, component in all_verifiers.items():
        assert isinstance(component, VerifierComponent)
        assert component.verifier_id == verifier_id


def test_verifier_component_factory_get_factory_info():
    """测试 VerifierComponentFactory（获取工厂信息）"""
    info = VerifierComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "VerifierComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_verifiers"] == 7
    assert len(info["supported_ids"]) == 7
    assert "created_at" in info


def test_verifier_component_factory_backward_compatible():
    """测试 VerifierComponentFactory（向后兼容函数）"""
    from src.data.validation.verifier_components import (
        create_verifier_verifier_component_3,
        create_verifier_verifier_component_8,
        create_verifier_verifier_component_13,
        create_verifier_verifier_component_18,
        create_verifier_verifier_component_23,
        create_verifier_verifier_component_28,
        create_verifier_verifier_component_33
    )
    
    component3 = create_verifier_verifier_component_3()
    assert component3.verifier_id == 3
    
    component8 = create_verifier_verifier_component_8()
    assert component8.verifier_id == 8


def test_component_factory_init():
    """测试 ComponentFactory（初始化）"""
    factory = ComponentFactory()
    
    assert factory._components == {}


def test_component_factory_create_component_none():
    """测试 ComponentFactory（创建组件，返回None）"""
    factory = ComponentFactory()
    
    result = factory.create_component("test", {})
    
    assert result is None


def test_verifier_component_multiple_instances():
    """测试 VerifierComponent（多个实例）"""
    component1 = VerifierComponent(3)
    component2 = VerifierComponent(8)
    
    assert component1.verifier_id == 3
    assert component2.verifier_id == 8
    assert component1.component_name != component2.component_name


def test_verifier_component_same_id_different_types():
    """测试 VerifierComponent（相同ID，不同类型）"""
    component1 = VerifierComponent(13, "Type1")
    component2 = VerifierComponent(13, "Type2")
    
    assert component1.verifier_id == component2.verifier_id
    assert component1.component_type != component2.component_type
    assert component1.component_name != component2.component_name
