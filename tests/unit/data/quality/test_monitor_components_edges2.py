"""
边界测试：monitor_components.py
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
from src.data.quality.monitor_components import (
    IMonitorComponent,
    MonitorComponent,
    MonitorComponentFactory,
    ComponentFactory
)


def test_imonitor_component_abstract():
    """测试 IMonitorComponent（抽象接口）"""
    # 不能直接实例化抽象类
    with pytest.raises(TypeError):
        IMonitorComponent()


def test_monitor_component_init():
    """测试 MonitorComponent（初始化）"""
    component = MonitorComponent(4)
    
    assert component.monitor_id == 4
    assert component.component_type == "Monitor"
    assert component.component_name == "Monitor_Component_4"
    assert isinstance(component.creation_time, datetime)


def test_monitor_component_init_custom_type():
    """测试 MonitorComponent（初始化，自定义类型）"""
    component = MonitorComponent(9, "CustomMonitor")
    
    assert component.monitor_id == 9
    assert component.component_type == "CustomMonitor"
    assert component.component_name == "CustomMonitor_Component_9"


def test_monitor_component_get_monitor_id():
    """测试 MonitorComponent（获取monitor ID）"""
    component = MonitorComponent(14)
    
    assert component.get_monitor_id() == 14


def test_monitor_component_get_info():
    """测试 MonitorComponent（获取组件信息）"""
    component = MonitorComponent(19)
    info = component.get_info()
    
    assert info["monitor_id"] == 19
    assert info["component_name"] == "Monitor_Component_19"
    assert info["component_type"] == "Monitor"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"


def test_monitor_component_process_success():
    """测试 MonitorComponent（处理数据，成功）"""
    component = MonitorComponent(24)
    data = {"key": "value"}
    
    result = component.process(data)
    
    assert result["monitor_id"] == 24
    assert result["status"] == "success"
    assert result["input_data"] == data
    assert "processed_at" in result


def test_monitor_component_process_empty():
    """测试 MonitorComponent（处理数据，空数据）"""
    component = MonitorComponent(29)
    
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_monitor_component_process_none():
    """测试 MonitorComponent（处理数据，None）"""
    component = MonitorComponent(34)
    
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_monitor_component_process_exception():
    """测试 MonitorComponent（处理数据，异常）"""
    component = MonitorComponent(39)
    # 创建一个会导致异常的数据
    class BadData:
        def __str__(self):
            raise ValueError("Test error")
    
    bad_data = BadData()
    result = component.process(bad_data)
    
    # 验证至少返回了有效的结果
    assert "status" in result
    assert result["monitor_id"] == 39


def test_monitor_component_process_large_data():
    """测试 MonitorComponent（处理数据，大数据）"""
    component = MonitorComponent(44)
    large_data = {"key" + str(i): "value" * 1000 for i in range(1000)}
    
    result = component.process(large_data)
    
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_monitor_component_process_nested_data():
    """测试 MonitorComponent（处理数据，嵌套数据）"""
    component = MonitorComponent(49)
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


def test_monitor_component_get_status():
    """测试 MonitorComponent（获取组件状态）"""
    component = MonitorComponent(54)
    status = component.get_status()
    
    assert status["monitor_id"] == 54
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_monitor_component_factory_create_component_valid():
    """测试 MonitorComponentFactory（创建组件，有效ID）"""
    component = MonitorComponentFactory.create_component(4)
    
    assert isinstance(component, MonitorComponent)
    assert component.monitor_id == 4


def test_monitor_component_factory_create_component_invalid():
    """测试 MonitorComponentFactory（创建组件，无效ID）"""
    with pytest.raises(ValueError, match="不支持的monitor ID"):
        MonitorComponentFactory.create_component(99)


def test_monitor_component_factory_create_component_negative():
    """测试 MonitorComponentFactory（创建组件，负数ID）"""
    with pytest.raises(ValueError, match="不支持的monitor ID"):
        MonitorComponentFactory.create_component(-1)


def test_monitor_component_factory_create_component_zero():
    """测试 MonitorComponentFactory（创建组件，零ID）"""
    with pytest.raises(ValueError, match="不支持的monitor ID"):
        MonitorComponentFactory.create_component(0)


def test_monitor_component_factory_get_available_monitors():
    """测试 MonitorComponentFactory（获取可用monitor列表）"""
    monitors = MonitorComponentFactory.get_available_monitors()
    
    assert isinstance(monitors, list)
    assert len(monitors) == 14
    assert 4 in monitors
    assert 69 in monitors
    assert monitors == sorted(monitors)


def test_monitor_component_factory_create_all_monitors():
    """测试 MonitorComponentFactory（创建所有monitor）"""
    all_monitors = MonitorComponentFactory.create_all_monitors()
    
    assert isinstance(all_monitors, dict)
    assert len(all_monitors) == 14
    for monitor_id, component in all_monitors.items():
        assert isinstance(component, MonitorComponent)
        assert component.monitor_id == monitor_id


def test_monitor_component_factory_get_factory_info():
    """测试 MonitorComponentFactory（获取工厂信息）"""
    info = MonitorComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "MonitorComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_monitors"] == 14
    assert len(info["supported_ids"]) == 14
    assert "created_at" in info


def test_monitor_component_factory_backward_compatible():
    """测试 MonitorComponentFactory（向后兼容函数）"""
    from src.data.quality.monitor_components import (
        create_monitor_monitor_component_4,
        create_monitor_monitor_component_9,
        create_monitor_monitor_component_14,
        create_monitor_monitor_component_19,
        create_monitor_monitor_component_24,
        create_monitor_monitor_component_29,
        create_monitor_monitor_component_34,
        create_monitor_monitor_component_39,
        create_monitor_monitor_component_44,
        create_monitor_monitor_component_49,
        create_monitor_monitor_component_54,
        create_monitor_monitor_component_59,
        create_monitor_monitor_component_64,
        create_monitor_monitor_component_69
    )
    
    component4 = create_monitor_monitor_component_4()
    assert component4.monitor_id == 4
    
    component9 = create_monitor_monitor_component_9()
    assert component9.monitor_id == 9


def test_component_factory_init():
    """测试 ComponentFactory（初始化）"""
    factory = ComponentFactory()
    
    assert factory._components == {}


def test_component_factory_create_component_none():
    """测试 ComponentFactory（创建组件，返回None）"""
    factory = ComponentFactory()
    
    result = factory.create_component("test", {})
    
    assert result is None


def test_monitor_component_multiple_instances():
    """测试 MonitorComponent（多个实例）"""
    component1 = MonitorComponent(4)
    component2 = MonitorComponent(9)
    
    assert component1.monitor_id == 4
    assert component2.monitor_id == 9
    assert component1.component_name != component2.component_name


def test_monitor_component_same_id_different_types():
    """测试 MonitorComponent（相同ID，不同类型）"""
    component1 = MonitorComponent(14, "Type1")
    component2 = MonitorComponent(14, "Type2")
    
    assert component1.monitor_id == component2.monitor_id
    assert component1.component_type != component2.component_type
    assert component1.component_name != component2.component_name

