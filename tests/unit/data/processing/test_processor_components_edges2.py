"""
边界测试：processor_components.py
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
from src.data.processing.processor_components import (
    IDataProcessorComponent,
    DataProcessorComponent,
    DataProcessorComponentFactory,
    ComponentFactory
)


def test_idata_processor_component_abstract():
    """测试 IDataProcessorComponent（抽象接口）"""
    # 不能直接实例化抽象类
    with pytest.raises(TypeError):
        IDataProcessorComponent()


def test_data_processor_component_init():
    """测试 DataProcessorComponent（初始化）"""
    component = DataProcessorComponent(1)
    
    assert component.processor_id == 1
    assert component.component_name == "DataProcessor_Component_1"
    assert isinstance(component.creation_time, datetime)


def test_data_processor_component_get_processor_id():
    """测试 DataProcessorComponent（获取处理器ID）"""
    component = DataProcessorComponent(6)
    
    assert component.get_processor_id() == 6


def test_data_processor_component_get_info():
    """测试 DataProcessorComponent（获取组件信息）"""
    component = DataProcessorComponent(11)
    info = component.get_info()
    
    assert info["processor_id"] == 11
    assert info["component_name"] == "DataProcessor_Component_11"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"


def test_data_processor_component_process_success():
    """测试 DataProcessorComponent（处理数据，成功）"""
    component = DataProcessorComponent(16)
    data = {"key": "value"}
    
    result = component.process(data)
    
    assert result["processor_id"] == 16
    assert result["status"] == "success"
    assert result["input_data"] == data
    assert "processed_at" in result


def test_data_processor_component_process_empty():
    """测试 DataProcessorComponent（处理数据，空数据）"""
    component = DataProcessorComponent(21)
    
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_data_processor_component_process_none():
    """测试 DataProcessorComponent（处理数据，None）"""
    component = DataProcessorComponent(26)
    
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_data_processor_component_process_exception():
    """测试 DataProcessorComponent（处理数据，异常）"""
    component = DataProcessorComponent(31)
    # 创建一个会导致异常的数据
    class BadData:
        def __str__(self):
            raise ValueError("Test error")
    
    bad_data = BadData()
    result = component.process(bad_data)
    
    # 验证至少返回了有效的结果
    assert "status" in result
    assert result["processor_id"] == 31


def test_data_processor_component_process_large_data():
    """测试 DataProcessorComponent（处理数据，大数据）"""
    component = DataProcessorComponent(36)
    large_data = {"key" + str(i): "value" * 1000 for i in range(1000)}
    
    result = component.process(large_data)
    
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_data_processor_component_process_nested_data():
    """测试 DataProcessorComponent（处理数据，嵌套数据）"""
    component = DataProcessorComponent(1)
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


def test_data_processor_component_get_status():
    """测试 DataProcessorComponent（获取组件状态）"""
    component = DataProcessorComponent(6)
    status = component.get_status()
    
    assert status["processor_id"] == 6
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_data_processor_component_factory_create_component_valid():
    """测试 DataProcessorComponentFactory（创建组件，有效ID）"""
    component = DataProcessorComponentFactory.create_component(1)
    
    assert isinstance(component, DataProcessorComponent)
    assert component.processor_id == 1


def test_data_processor_component_factory_create_component_invalid():
    """测试 DataProcessorComponentFactory（创建组件，无效ID）"""
    with pytest.raises(ValueError, match="不支持的处理器ID"):
        DataProcessorComponentFactory.create_component(99)


def test_data_processor_component_factory_create_component_negative():
    """测试 DataProcessorComponentFactory（创建组件，负数ID）"""
    with pytest.raises(ValueError, match="不支持的处理器ID"):
        DataProcessorComponentFactory.create_component(-1)


def test_data_processor_component_factory_create_component_zero():
    """测试 DataProcessorComponentFactory（创建组件，零ID）"""
    with pytest.raises(ValueError, match="不支持的处理器ID"):
        DataProcessorComponentFactory.create_component(0)


def test_data_processor_component_factory_get_available_processors():
    """测试 DataProcessorComponentFactory（获取可用处理器列表）"""
    processors = DataProcessorComponentFactory.get_available_processors()
    
    assert isinstance(processors, list)
    assert len(processors) == 8
    assert 1 in processors
    assert 36 in processors
    assert processors == sorted(processors)


def test_data_processor_component_factory_create_all_processors():
    """测试 DataProcessorComponentFactory（创建所有处理器）"""
    all_processors = DataProcessorComponentFactory.create_all_processors()
    
    assert isinstance(all_processors, dict)
    assert len(all_processors) == 8
    for processor_id, component in all_processors.items():
        assert isinstance(component, DataProcessorComponent)
        assert component.processor_id == processor_id


def test_data_processor_component_factory_get_factory_info():
    """测试 DataProcessorComponentFactory（获取工厂信息）"""
    info = DataProcessorComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "DataProcessorComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_processors"] == 8
    assert len(info["supported_ids"]) == 8
    assert "created_at" in info


def test_data_processor_component_factory_backward_compatible():
    """测试 DataProcessorComponentFactory（向后兼容函数）"""
    from src.data.processing.processor_components import (
        create_data_processor_component_1,
        create_data_processor_component_6,
        create_data_processor_component_11,
        create_data_processor_component_16,
        create_data_processor_component_21,
        create_data_processor_component_26,
        create_data_processor_component_31,
        create_data_processor_component_36
    )
    
    component1 = create_data_processor_component_1()
    assert component1.processor_id == 1
    
    component6 = create_data_processor_component_6()
    assert component6.processor_id == 6


def test_component_factory_init():
    """测试 ComponentFactory（初始化）"""
    factory = ComponentFactory()
    
    assert factory._components == {}


def test_component_factory_create_component_none():
    """测试 ComponentFactory（创建组件，返回None）"""
    factory = ComponentFactory()
    
    result = factory.create_component("test", {})
    
    assert result is None


def test_data_processor_component_multiple_instances():
    """测试 DataProcessorComponent（多个实例）"""
    component1 = DataProcessorComponent(1)
    component2 = DataProcessorComponent(6)
    
    assert component1.processor_id == 1
    assert component2.processor_id == 6
    assert component1.component_name != component2.component_name

