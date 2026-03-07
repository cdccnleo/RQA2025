"""
边界测试：quality_components.py
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
from src.data.quality.quality_components import (
    IQualityComponent,
    QualityComponent,
    QualityComponentFactory,
    ComponentFactory
)


def test_iquality_component_abstract():
    """测试 IQualityComponent（抽象接口）"""
    # 不能直接实例化抽象类
    with pytest.raises(TypeError):
        IQualityComponent()


def test_quality_component_init():
    """测试 QualityComponent（初始化）"""
    component = QualityComponent(1)
    
    assert component.quality_id == 1
    assert component.component_type == "Quality"
    assert component.component_name == "Quality_Component_1"
    assert isinstance(component.creation_time, datetime)


def test_quality_component_init_custom_type():
    """测试 QualityComponent（初始化，自定义类型）"""
    component = QualityComponent(6, "CustomQuality")
    
    assert component.quality_id == 6
    assert component.component_type == "CustomQuality"
    assert component.component_name == "CustomQuality_Component_6"


def test_quality_component_get_quality_id():
    """测试 QualityComponent（获取quality ID）"""
    component = QualityComponent(11)
    
    assert component.get_quality_id() == 11


def test_quality_component_get_info():
    """测试 QualityComponent（获取组件信息）"""
    component = QualityComponent(16)
    info = component.get_info()
    
    assert info["quality_id"] == 16
    assert info["component_name"] == "Quality_Component_16"
    assert info["component_type"] == "Quality"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"


def test_quality_component_process_success():
    """测试 QualityComponent（处理数据，成功）"""
    component = QualityComponent(21)
    data = {"key": "value"}
    
    result = component.process(data)
    
    assert result["quality_id"] == 21
    assert result["status"] == "success"
    assert result["input_data"] == data
    assert "processed_at" in result


def test_quality_component_process_empty():
    """测试 QualityComponent（处理数据，空数据）"""
    component = QualityComponent(26)
    
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_quality_component_process_none():
    """测试 QualityComponent（处理数据，None）"""
    component = QualityComponent(31)
    
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_quality_component_process_exception():
    """测试 QualityComponent（处理数据，异常）"""
    component = QualityComponent(36)
    # 创建一个会导致异常的数据
    class BadData:
        def __str__(self):
            raise ValueError("Test error")
    
    bad_data = BadData()
    result = component.process(bad_data)
    
    # 验证至少返回了有效的结果
    assert "status" in result
    assert result["quality_id"] == 36


def test_quality_component_process_large_data():
    """测试 QualityComponent（处理数据，大数据）"""
    component = QualityComponent(41)
    large_data = {"key" + str(i): "value" * 1000 for i in range(1000)}
    
    result = component.process(large_data)
    
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_quality_component_process_nested_data():
    """测试 QualityComponent（处理数据，嵌套数据）"""
    component = QualityComponent(46)
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


def test_quality_component_get_status():
    """测试 QualityComponent（获取组件状态）"""
    component = QualityComponent(51)
    status = component.get_status()
    
    assert status["quality_id"] == 51
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_quality_component_initialize():
    """测试 QualityComponent（初始化配置）"""
    component = QualityComponent(56)
    
    result = component.initialize({"key": "value"})
    
    assert result is True
    assert component.config == {"key": "value"}
    assert hasattr(component, "initialized_at")


def test_quality_component_initialize_none():
    """测试 QualityComponent（初始化配置，None）"""
    component = QualityComponent(61)
    
    result = component.initialize(None)
    
    assert result is True
    assert component.config == {}


def test_quality_component_validate():
    """测试 QualityComponent（验证数据）"""
    component = QualityComponent(66)
    
    result = component.validate({"key": "value"})
    
    assert result is True


def test_quality_component_validate_none():
    """测试 QualityComponent（验证数据，None）"""
    component = QualityComponent(1)
    
    result = component.validate(None)
    
    assert result is True


def test_quality_component_factory_create_component_valid():
    """测试 QualityComponentFactory（创建组件，有效ID）"""
    component = QualityComponentFactory.create_component(1)
    
    assert isinstance(component, QualityComponent)
    assert component.quality_id == 1


def test_quality_component_factory_create_component_invalid():
    """测试 QualityComponentFactory（创建组件，无效ID）"""
    with pytest.raises(ValueError, match="不支持的quality ID"):
        QualityComponentFactory.create_component(99)


def test_quality_component_factory_create_component_negative():
    """测试 QualityComponentFactory（创建组件，负数ID）"""
    with pytest.raises(ValueError, match="不支持的quality ID"):
        QualityComponentFactory.create_component(-1)


def test_quality_component_factory_create_component_zero():
    """测试 QualityComponentFactory（创建组件，零ID）"""
    with pytest.raises(ValueError, match="不支持的quality ID"):
        QualityComponentFactory.create_component(0)


def test_quality_component_factory_get_available_qualitys():
    """测试 QualityComponentFactory（获取可用quality列表）"""
    qualities = QualityComponentFactory.get_available_qualitys()
    
    assert isinstance(qualities, list)
    assert len(qualities) == 14
    assert 1 in qualities
    assert 66 in qualities
    assert qualities == sorted(qualities)


def test_quality_component_factory_create_all_qualitys():
    """测试 QualityComponentFactory（创建所有quality）"""
    all_qualities = QualityComponentFactory.create_all_qualitys()
    
    assert isinstance(all_qualities, dict)
    assert len(all_qualities) == 14
    for quality_id, component in all_qualities.items():
        assert isinstance(component, QualityComponent)
        assert component.quality_id == quality_id


def test_quality_component_factory_get_factory_info():
    """测试 QualityComponentFactory（获取工厂信息）"""
    info = QualityComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "QualityComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_qualities"] == 14
    assert len(info["supported_ids"]) == 14
    assert "created_at" in info


def test_quality_component_factory_backward_compatible():
    """测试 QualityComponentFactory（向后兼容函数）"""
    from src.data.quality.quality_components import (
        create_quality_quality_component_1,
        create_quality_quality_component_6,
        create_quality_quality_component_11,
        create_quality_quality_component_16,
        create_quality_quality_component_21,
        create_quality_quality_component_26,
        create_quality_quality_component_31,
        create_quality_quality_component_36,
        create_quality_quality_component_41,
        create_quality_quality_component_46,
        create_quality_quality_component_51,
        create_quality_quality_component_56,
        create_quality_quality_component_61,
        create_quality_quality_component_66
    )
    
    component1 = create_quality_quality_component_1()
    assert component1.quality_id == 1
    
    component6 = create_quality_quality_component_6()
    assert component6.quality_id == 6


def test_component_factory_init():
    """测试 ComponentFactory（初始化）"""
    factory = ComponentFactory()
    
    assert factory._components == {}


def test_component_factory_create_component_none():
    """测试 ComponentFactory（创建组件，返回None）"""
    factory = ComponentFactory()
    
    result = factory.create_component("test", {})
    
    assert result is None


def test_quality_component_multiple_instances():
    """测试 QualityComponent（多个实例）"""
    component1 = QualityComponent(1)
    component2 = QualityComponent(6)
    
    assert component1.quality_id == 1
    assert component2.quality_id == 6
    assert component1.component_name != component2.component_name


def test_quality_component_same_id_different_types():
    """测试 QualityComponent（相同ID，不同类型）"""
    component1 = QualityComponent(11, "Type1")
    component2 = QualityComponent(11, "Type2")
    
    assert component1.quality_id == component2.quality_id
    assert component1.component_type != component2.component_type
    assert component1.component_name != component2.component_name

