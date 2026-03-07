"""
边界测试：buffer_components.py
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
from unittest.mock import patch

from src.data.cache.buffer_components import (
    IBufferComponent,
    BufferComponent,
    BufferComponentFactory,
    create_buffer_buffer_component_2,
    create_buffer_buffer_component_6,
    create_buffer_buffer_component_10,
    create_buffer_buffer_component_14,
    create_buffer_buffer_component_18,
    create_buffer_buffer_component_22,
)


def test_ibuffer_component_abstract():
    """测试 IBufferComponent（抽象接口）"""
    with pytest.raises(TypeError):
        IBufferComponent()


def test_buffer_component_init():
    """测试 BufferComponent（初始化）"""
    component = BufferComponent(2)
    
    assert component.buffer_id == 2
    assert component.component_type == "Buffer"
    assert component.component_name == "Buffer_Component_2"


def test_buffer_component_get_buffer_id():
    """测试 BufferComponent（获取buffer ID）"""
    component = BufferComponent(6)
    
    assert component.get_buffer_id() == 6


def test_buffer_component_get_info():
    """测试 BufferComponent（获取组件信息）"""
    component = BufferComponent(2)
    
    info = component.get_info()
    
    assert info["buffer_id"] == 2
    assert info["component_type"] == "Buffer"
    assert info["version"] == "2.0.0"


def test_buffer_component_process_success():
    """测试 BufferComponent（处理数据，成功）"""
    component = BufferComponent(2)
    data = {"key": "value"}
    
    result = component.process(data)
    
    assert result["buffer_id"] == 2
    assert result["status"] == "success"
    assert result["input_data"] == data


def test_buffer_component_process_empty():
    """测试 BufferComponent（处理数据，空数据）"""
    component = BufferComponent(2)
    
    result = component.process({})
    
    assert result["status"] == "success"


def test_buffer_component_process_none():
    """测试 BufferComponent（处理数据，None）"""
    component = BufferComponent(2)
    
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_buffer_component_process_large_data():
    """测试 BufferComponent（处理数据，大数据）"""
    component = BufferComponent(2)
    large_data = {"key" + str(i): "value" * 1000 for i in range(1000)}
    
    result = component.process(large_data)
    
    assert result["status"] == "success"


def test_buffer_component_process_nested_data():
    """测试 BufferComponent（处理数据，嵌套数据）"""
    component = BufferComponent(2)
    nested_data = {"level1": {"level2": {"level3": "value"}}}
    
    result = component.process(nested_data)
    
    assert result["status"] == "success"


def test_buffer_component_get_status():
    """测试 BufferComponent（获取组件状态）"""
    component = BufferComponent(2)
    
    status = component.get_status()
    
    assert status["buffer_id"] == 2
    assert status["status"] == "active"
    assert status["health"] == "good"


def test_buffer_component_factory_create_component_valid():
    """测试 BufferComponentFactory（创建组件，有效ID）"""
    component = BufferComponentFactory.create_component(2)
    
    assert isinstance(component, BufferComponent)
    assert component.buffer_id == 2


def test_buffer_component_factory_create_component_invalid():
    """测试 BufferComponentFactory（创建组件，无效ID）"""
    with pytest.raises(ValueError, match="不支持的buffer ID"):
        BufferComponentFactory.create_component(99)


def test_buffer_component_factory_create_component_negative():
    """测试 BufferComponentFactory（创建组件，负数ID）"""
    with pytest.raises(ValueError):
        BufferComponentFactory.create_component(-1)


def test_buffer_component_factory_create_component_zero():
    """测试 BufferComponentFactory（创建组件，零ID）"""
    with pytest.raises(ValueError):
        BufferComponentFactory.create_component(0)


def test_buffer_component_factory_get_available_buffers():
    """测试 BufferComponentFactory（获取可用buffer列表）"""
    buffers = BufferComponentFactory.get_available_buffers()
    
    assert isinstance(buffers, list)
    assert len(buffers) == 6
    assert 2 in buffers
    assert 6 in buffers
    assert 10 in buffers
    assert 14 in buffers
    assert 18 in buffers
    assert 22 in buffers


def test_buffer_component_factory_create_all_buffers():
    """测试 BufferComponentFactory（创建所有buffer）"""
    all_buffers = BufferComponentFactory.create_all_buffers()
    
    assert isinstance(all_buffers, dict)
    assert len(all_buffers) == 6


def test_buffer_component_factory_get_factory_info():
    """测试 BufferComponentFactory（获取工厂信息）"""
    info = BufferComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "BufferComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_buffers"] == 6


def test_create_buffer_buffer_component_2():
    """测试 create_buffer_buffer_component_2（向后兼容函数）"""
    component = create_buffer_buffer_component_2()
    
    assert isinstance(component, BufferComponent)
    assert component.buffer_id == 2


def test_create_buffer_buffer_component_6():
    """测试 create_buffer_buffer_component_6（向后兼容函数）"""
    component = create_buffer_buffer_component_6()
    
    assert component.buffer_id == 6


def test_create_buffer_buffer_component_10():
    """测试 create_buffer_buffer_component_10（向后兼容函数）"""
    component = create_buffer_buffer_component_10()
    
    assert component.buffer_id == 10


def test_create_buffer_buffer_component_14():
    """测试 create_buffer_buffer_component_14（向后兼容函数）"""
    component = create_buffer_buffer_component_14()
    
    assert component.buffer_id == 14


def test_create_buffer_buffer_component_18():
    """测试 create_buffer_buffer_component_18（向后兼容函数）"""
    component = create_buffer_buffer_component_18()
    
    assert component.buffer_id == 18


def test_create_buffer_buffer_component_22():
    """测试 create_buffer_buffer_component_22（向后兼容函数）"""
    component = create_buffer_buffer_component_22()
    
    assert component.buffer_id == 22


def test_buffer_component_multiple_instances():
    """测试 BufferComponent（多个实例）"""
    component1 = BufferComponent(2)
    component2 = BufferComponent(6)
    
    assert component1.buffer_id == 2
    assert component2.buffer_id == 6
    assert component1 is not component2

