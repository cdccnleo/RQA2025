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

from src.data.monitoring.monitor_components import (
    ComponentFactory,
    IMonitorComponent,
    MonitorComponent,
    MonitorComponentFactory,
    create_monitor_monitor_component_1,
    create_monitor_monitor_component_6,
    create_monitor_monitor_component_11,
    create_monitor_monitor_component_16,
    create_monitor_monitor_component_21,
    create_monitor_monitor_component_26,
    create_monitor_monitor_component_31,
    create_monitor_monitor_component_36,
    create_monitor_monitor_component_41,
)


def test_imonitor_component_abstract():
    """测试 IMonitorComponent（抽象接口）"""
    with pytest.raises(TypeError):
        IMonitorComponent()


def test_monitor_component_init_default():
    """测试 MonitorComponent（初始化，默认类型）"""
    component = MonitorComponent(monitor_id=1)
    assert component.monitor_id == 1
    assert component.component_type == "Monitor"
    assert component.component_name == "Monitor_Component_1"
    assert component.creation_time is not None


def test_monitor_component_init_custom():
    """测试 MonitorComponent（初始化，自定义类型）"""
    component = MonitorComponent(monitor_id=6, component_type="CustomMonitor")
    assert component.monitor_id == 6
    assert component.component_type == "CustomMonitor"
    assert component.component_name == "CustomMonitor_Component_6"


def test_monitor_component_get_monitor_id():
    """测试 MonitorComponent（获取 monitor ID）"""
    component = MonitorComponent(monitor_id=11)
    assert component.get_monitor_id() == 11


def test_monitor_component_get_info():
    """测试 MonitorComponent（获取组件信息）"""
    component = MonitorComponent(monitor_id=16, component_type="TestMonitor")
    info = component.get_info()
    
    assert info["monitor_id"] == 16
    assert info["component_name"] == "TestMonitor_Component_16"
    assert info["component_type"] == "TestMonitor"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"
    assert info["type"] == "unified_data_monitoring_component"


def test_monitor_component_get_status():
    """测试 MonitorComponent（获取组件状态）"""
    component = MonitorComponent(monitor_id=21)
    status = component.get_status()
    
    assert status["monitor_id"] == 21
    assert status["component_name"] == "Monitor_Component_21"
    assert status["component_type"] == "Monitor"
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_monitor_component_process_success():
    """测试 MonitorComponent（处理数据，成功）"""
    component = MonitorComponent(monitor_id=26, component_type="DataMonitor")
    data = {"key": "value", "number": 42}
    
    result = component.process(data)
    
    assert result["status"] == "success"
    assert result["monitor_id"] == 26
    assert result["component_name"] == "DataMonitor_Component_26"
    assert result["input_data"] == data
    assert "processed_at" in result
    assert "result" in result
    assert result["processing_type"] == "unified_monitor_processing"


def test_monitor_component_process_empty_data():
    """测试 MonitorComponent（处理数据，空数据）"""
    component = MonitorComponent(monitor_id=1)
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_monitor_component_process_none_data():
    """测试 MonitorComponent（处理数据，None 数据）"""
    component = MonitorComponent(monitor_id=6)
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_monitor_component_process_error():
    """测试 MonitorComponent（处理数据，异常）"""
    component = MonitorComponent(monitor_id=11)
    
    class BrokenName:
        def __str__(self):
            raise RuntimeError("processing failed")
    
    component.component_name = BrokenName()
    result = component.process({"test": "data"})
    
    assert result["status"] == "error"
    assert "error" in result
    assert result["error"] == "processing failed"
    assert result["error_type"] == "RuntimeError"


def test_monitor_component_factory_create_component_valid():
    """测试 MonitorComponentFactory（创建组件，有效 ID）"""
    component = MonitorComponentFactory.create_component(1)
    assert isinstance(component, MonitorComponent)
    assert component.monitor_id == 1
    assert component.component_type == "Monitor"


def test_monitor_component_factory_create_component_invalid():
    """测试 MonitorComponentFactory（创建组件，无效 ID）"""
    with pytest.raises(ValueError, match="不支持的monitor ID"):
        MonitorComponentFactory.create_component(99)


def test_monitor_component_factory_create_component_negative():
    """测试 MonitorComponentFactory（创建组件，负 ID）"""
    with pytest.raises(ValueError, match="不支持的monitor ID"):
        MonitorComponentFactory.create_component(-1)


def test_monitor_component_factory_create_component_zero():
    """测试 MonitorComponentFactory（创建组件，零 ID）"""
    with pytest.raises(ValueError, match="不支持的monitor ID"):
        MonitorComponentFactory.create_component(0)


def test_monitor_component_factory_get_available_monitors():
    """测试 MonitorComponentFactory（获取可用 monitor ID 列表）"""
    monitors = MonitorComponentFactory.get_available_monitors()
    assert monitors == [1, 6, 11, 16, 21, 26, 31, 36, 41]
    assert isinstance(monitors, list)


def test_monitor_component_factory_create_all_monitors():
    """测试 MonitorComponentFactory（创建所有 monitor）"""
    all_monitors = MonitorComponentFactory.create_all_monitors()
    
    assert isinstance(all_monitors, dict)
    assert len(all_monitors) == 9
    assert all(isinstance(monitor, MonitorComponent) for monitor in all_monitors.values())
    assert all(monitor_id in [1, 6, 11, 16, 21, 26, 31, 36, 41] for monitor_id in all_monitors.keys())


def test_monitor_component_factory_get_factory_info():
    """测试 MonitorComponentFactory（获取工厂信息）"""
    info = MonitorComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "MonitorComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_monitors"] == 9
    assert info["supported_ids"] == [1, 6, 11, 16, 21, 26, 31, 36, 41]
    assert "created_at" in info
    assert "description" in info


def test_legacy_factory_functions():
    """测试向后兼容的工厂函数"""
    component1 = create_monitor_monitor_component_1()
    assert component1.monitor_id == 1
    
    component6 = create_monitor_monitor_component_6()
    assert component6.monitor_id == 6
    
    component11 = create_monitor_monitor_component_11()
    assert component11.monitor_id == 11
    
    component16 = create_monitor_monitor_component_16()
    assert component16.monitor_id == 16
    
    component21 = create_monitor_monitor_component_21()
    assert component21.monitor_id == 21
    
    component26 = create_monitor_monitor_component_26()
    assert component26.monitor_id == 26
    
    component31 = create_monitor_monitor_component_31()
    assert component31.monitor_id == 31
    
    component36 = create_monitor_monitor_component_36()
    assert component36.monitor_id == 36
    
    component41 = create_monitor_monitor_component_41()
    assert component41.monitor_id == 41


def test_component_factory_init():
    """测试 ComponentFactory（初始化）"""
    factory = ComponentFactory()
    assert factory._components == {}


def test_component_factory_create_component_none_instance():
    """测试 ComponentFactory（创建组件，实例为 None）"""
    factory = ComponentFactory()
    result = factory.create_component("test_type", {})
    assert result is None


def test_component_factory_create_component_initialize_fails():
    """测试 ComponentFactory（创建组件，初始化失败）"""
    factory = ComponentFactory()
    
    class MockComponent:
        def initialize(self, config):
            return False
    
    factory._create_component_instance = lambda t, c: MockComponent()
    result = factory.create_component("test_type", {})
    assert result is None


def test_component_factory_create_component_exception():
    """测试 ComponentFactory（创建组件，抛出异常）"""
    factory = ComponentFactory()
    
    def _bad_create(component_type, config):
        raise RuntimeError("creation failed")
    
    factory._create_component_instance = _bad_create
    result = factory.create_component("test_type", {})
    assert result is None


def test_monitor_component_process_large_data():
    """测试 MonitorComponent（处理数据，大数据）"""
    component = MonitorComponent(monitor_id=1)
    large_data = {"key" + str(i): "value" + str(i) for i in range(1000)}
    
    result = component.process(large_data)
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_monitor_component_process_nested_data():
    """测试 MonitorComponent（处理数据，嵌套数据）"""
    component = MonitorComponent(monitor_id=6)
    nested_data = {
        "level1": {
            "level2": {
                "level3": "value"
            }
        },
        "list": [1, 2, 3, {"nested": "item"}]
    }
    
    result = component.process(nested_data)
    assert result["status"] == "success"
    assert result["input_data"] == nested_data


def test_monitor_component_multiple_instances():
    """测试 MonitorComponent（多个实例）"""
    component1 = MonitorComponent(monitor_id=1)
    component2 = MonitorComponent(monitor_id=6)
    component3 = MonitorComponent(monitor_id=11)
    
    assert component1.monitor_id == 1
    assert component2.monitor_id == 6
    assert component3.monitor_id == 11
    assert component1.component_name != component2.component_name
    assert component2.component_name != component3.component_name


def test_monitor_component_same_id_different_type():
    """测试 MonitorComponent（相同 ID，不同类型）"""
    component1 = MonitorComponent(monitor_id=1, component_type="Type1")
    component2 = MonitorComponent(monitor_id=1, component_type="Type2")
    
    assert component1.monitor_id == component2.monitor_id == 1
    assert component1.component_type == "Type1"
    assert component2.component_type == "Type2"
    assert component1.component_name != component2.component_name

