"""
边界测试：validator_components.py
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
from src.data.quality.validator_components import (
    IValidatorComponent,
    ValidatorComponent,
    ValidatorComponentFactory,
    ComponentFactory
)


def test_ivalidator_component_abstract():
    """测试 IValidatorComponent（抽象接口）"""
    # 不能直接实例化抽象类
    with pytest.raises(TypeError):
        IValidatorComponent()


def test_validator_component_init():
    """测试 ValidatorComponent（初始化）"""
    component = ValidatorComponent(2)
    
    assert component.validator_id == 2
    assert component.component_type == "Validator"
    assert component.component_name == "Validator_Component_2"
    assert isinstance(component.creation_time, datetime)


def test_validator_component_init_custom_type():
    """测试 ValidatorComponent（初始化，自定义类型）"""
    component = ValidatorComponent(7, "CustomValidator")
    
    assert component.validator_id == 7
    assert component.component_type == "CustomValidator"
    assert component.component_name == "CustomValidator_Component_7"


def test_validator_component_get_validator_id():
    """测试 ValidatorComponent（获取validator ID）"""
    component = ValidatorComponent(12)
    
    assert component.get_validator_id() == 12


def test_validator_component_get_info():
    """测试 ValidatorComponent（获取组件信息）"""
    component = ValidatorComponent(17)
    info = component.get_info()
    
    assert info["validator_id"] == 17
    assert info["component_name"] == "Validator_Component_17"
    assert info["component_type"] == "Validator"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"


def test_validator_component_process_success():
    """测试 ValidatorComponent（处理数据，成功）"""
    component = ValidatorComponent(22)
    data = {"key": "value"}
    
    result = component.process(data)
    
    assert result["validator_id"] == 22
    assert result["status"] == "success"
    assert result["input_data"] == data
    assert "processed_at" in result


def test_validator_component_process_empty():
    """测试 ValidatorComponent（处理数据，空数据）"""
    component = ValidatorComponent(27)
    
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_validator_component_process_none():
    """测试 ValidatorComponent（处理数据，None）"""
    component = ValidatorComponent(32)
    
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_validator_component_process_exception():
    """测试 ValidatorComponent（处理数据，异常）"""
    component = ValidatorComponent(37)
    # 创建一个会导致异常的数据
    class BadData:
        def __str__(self):
            raise ValueError("Test error")
    
    bad_data = BadData()
    result = component.process(bad_data)
    
    # 验证至少返回了有效的结果
    assert "status" in result
    assert result["validator_id"] == 37


def test_validator_component_process_large_data():
    """测试 ValidatorComponent（处理数据，大数据）"""
    component = ValidatorComponent(42)
    large_data = {"key" + str(i): "value" * 1000 for i in range(1000)}
    
    result = component.process(large_data)
    
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_validator_component_process_nested_data():
    """测试 ValidatorComponent（处理数据，嵌套数据）"""
    component = ValidatorComponent(47)
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


def test_validator_component_get_status():
    """测试 ValidatorComponent（获取组件状态）"""
    component = ValidatorComponent(52)
    status = component.get_status()
    
    assert status["validator_id"] == 52
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_validator_component_factory_create_component_valid():
    """测试 ValidatorComponentFactory（创建组件，有效ID）"""
    component = ValidatorComponentFactory.create_component(2)
    
    assert isinstance(component, ValidatorComponent)
    assert component.validator_id == 2


def test_validator_component_factory_create_component_invalid():
    """测试 ValidatorComponentFactory（创建组件，无效ID）"""
    with pytest.raises(ValueError, match="不支持的validator ID"):
        ValidatorComponentFactory.create_component(99)


def test_validator_component_factory_create_component_negative():
    """测试 ValidatorComponentFactory（创建组件，负数ID）"""
    with pytest.raises(ValueError, match="不支持的validator ID"):
        ValidatorComponentFactory.create_component(-1)


def test_validator_component_factory_create_component_zero():
    """测试 ValidatorComponentFactory（创建组件，零ID）"""
    with pytest.raises(ValueError, match="不支持的validator ID"):
        ValidatorComponentFactory.create_component(0)


def test_validator_component_factory_get_available_validators():
    """测试 ValidatorComponentFactory（获取可用validator列表）"""
    validators = ValidatorComponentFactory.get_available_validators()
    
    assert isinstance(validators, list)
    assert len(validators) == 14
    assert 2 in validators
    assert 67 in validators
    assert validators == sorted(validators)


def test_validator_component_factory_create_all_validators():
    """测试 ValidatorComponentFactory（创建所有validator）"""
    all_validators = ValidatorComponentFactory.create_all_validators()
    
    assert isinstance(all_validators, dict)
    assert len(all_validators) == 14
    for validator_id, component in all_validators.items():
        assert isinstance(component, ValidatorComponent)
        assert component.validator_id == validator_id


def test_validator_component_factory_get_factory_info():
    """测试 ValidatorComponentFactory（获取工厂信息）"""
    info = ValidatorComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "ValidatorComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_validators"] == 14
    assert len(info["supported_ids"]) == 14
    assert "created_at" in info


def test_validator_component_factory_backward_compatible():
    """测试 ValidatorComponentFactory（向后兼容函数）"""
    from src.data.quality.validator_components import (
        create_validator_validator_component_2,
        create_validator_validator_component_7,
        create_validator_validator_component_12,
        create_validator_validator_component_17,
        create_validator_validator_component_22,
        create_validator_validator_component_27,
        create_validator_validator_component_32,
        create_validator_validator_component_37,
        create_validator_validator_component_42,
        create_validator_validator_component_47,
        create_validator_validator_component_52,
        create_validator_validator_component_57,
        create_validator_validator_component_62,
        create_validator_validator_component_67
    )
    
    component2 = create_validator_validator_component_2()
    assert component2.validator_id == 2
    
    component7 = create_validator_validator_component_7()
    assert component7.validator_id == 7


def test_component_factory_init():
    """测试 ComponentFactory（初始化）"""
    factory = ComponentFactory()
    
    assert factory._components == {}


def test_component_factory_create_component_none():
    """测试 ComponentFactory（创建组件，返回None）"""
    factory = ComponentFactory()
    
    result = factory.create_component("test", {})
    
    assert result is None


def test_validator_component_multiple_instances():
    """测试 ValidatorComponent（多个实例）"""
    component1 = ValidatorComponent(2)
    component2 = ValidatorComponent(7)
    
    assert component1.validator_id == 2
    assert component2.validator_id == 7
    assert component1.component_name != component2.component_name


def test_validator_component_same_id_different_types():
    """测试 ValidatorComponent（相同ID，不同类型）"""
    component1 = ValidatorComponent(12, "Type1")
    component2 = ValidatorComponent(12, "Type2")
    
    assert component1.validator_id == component2.validator_id
    assert component1.component_type != component2.component_type
    assert component1.component_name != component2.component_name

