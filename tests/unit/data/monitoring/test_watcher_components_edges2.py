"""
边界测试：watcher_components.py
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

from src.data.monitoring.watcher_components import (
    ComponentFactory,
    IWatcherComponent,
    WatcherComponent,
    WatcherComponentFactory,
    create_watcher_watcher_component_2,
    create_watcher_watcher_component_7,
    create_watcher_watcher_component_12,
    create_watcher_watcher_component_17,
    create_watcher_watcher_component_22,
    create_watcher_watcher_component_27,
    create_watcher_watcher_component_32,
    create_watcher_watcher_component_37,
    create_watcher_watcher_component_42,
)


def test_iwatcher_component_abstract():
    """测试 IWatcherComponent（抽象接口）"""
    with pytest.raises(TypeError):
        IWatcherComponent()


def test_watcher_component_init_default():
    """测试 WatcherComponent（初始化，默认类型）"""
    component = WatcherComponent(watcher_id=2)
    assert component.watcher_id == 2
    assert component.component_type == "Watcher"
    assert component.component_name == "Watcher_Component_2"
    assert component.creation_time is not None


def test_watcher_component_init_custom():
    """测试 WatcherComponent（初始化，自定义类型）"""
    component = WatcherComponent(watcher_id=7, component_type="CustomWatcher")
    assert component.watcher_id == 7
    assert component.component_type == "CustomWatcher"
    assert component.component_name == "CustomWatcher_Component_7"


def test_watcher_component_get_watcher_id():
    """测试 WatcherComponent（获取 watcher ID）"""
    component = WatcherComponent(watcher_id=12)
    assert component.get_watcher_id() == 12


def test_watcher_component_get_info():
    """测试 WatcherComponent（获取组件信息）"""
    component = WatcherComponent(watcher_id=17, component_type="TestWatcher")
    info = component.get_info()
    
    assert info["watcher_id"] == 17
    assert info["component_name"] == "TestWatcher_Component_17"
    assert info["component_type"] == "TestWatcher"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"
    assert info["type"] == "unified_data_monitoring_component"


def test_watcher_component_get_status():
    """测试 WatcherComponent（获取组件状态）"""
    component = WatcherComponent(watcher_id=22)
    status = component.get_status()
    
    assert status["watcher_id"] == 22
    assert status["component_name"] == "Watcher_Component_22"
    assert status["component_type"] == "Watcher"
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_watcher_component_process_success():
    """测试 WatcherComponent（处理数据，成功）"""
    component = WatcherComponent(watcher_id=27, component_type="DataWatcher")
    data = {"key": "value", "number": 42}
    
    result = component.process(data)
    
    assert result["status"] == "success"
    assert result["watcher_id"] == 27
    assert result["component_name"] == "DataWatcher_Component_27"
    assert result["input_data"] == data
    assert "processed_at" in result
    assert "result" in result
    assert result["processing_type"] == "unified_watcher_processing"


def test_watcher_component_process_empty_data():
    """测试 WatcherComponent（处理数据，空数据）"""
    component = WatcherComponent(watcher_id=2)
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_watcher_component_process_none_data():
    """测试 WatcherComponent（处理数据，None 数据）"""
    component = WatcherComponent(watcher_id=7)
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_watcher_component_process_error():
    """测试 WatcherComponent（处理数据，异常）"""
    component = WatcherComponent(watcher_id=12)
    
    class BrokenName:
        def __str__(self):
            raise RuntimeError("processing failed")
    
    component.component_name = BrokenName()
    result = component.process({"test": "data"})
    
    assert result["status"] == "error"
    assert "error" in result
    assert result["error"] == "processing failed"
    assert result["error_type"] == "RuntimeError"


def test_watcher_component_factory_create_component_valid():
    """测试 WatcherComponentFactory（创建组件，有效 ID）"""
    component = WatcherComponentFactory.create_component(2)
    assert isinstance(component, WatcherComponent)
    assert component.watcher_id == 2
    assert component.component_type == "Watcher"


def test_watcher_component_factory_create_component_invalid():
    """测试 WatcherComponentFactory（创建组件，无效 ID）"""
    with pytest.raises(ValueError, match="不支持的watcher ID"):
        WatcherComponentFactory.create_component(99)


def test_watcher_component_factory_create_component_negative():
    """测试 WatcherComponentFactory（创建组件，负 ID）"""
    with pytest.raises(ValueError, match="不支持的watcher ID"):
        WatcherComponentFactory.create_component(-1)


def test_watcher_component_factory_create_component_zero():
    """测试 WatcherComponentFactory（创建组件，零 ID）"""
    with pytest.raises(ValueError, match="不支持的watcher ID"):
        WatcherComponentFactory.create_component(0)


def test_watcher_component_factory_get_available_watchers():
    """测试 WatcherComponentFactory（获取可用 watcher ID 列表）"""
    watchers = WatcherComponentFactory.get_available_watchers()
    assert watchers == [2, 7, 12, 17, 22, 27, 32, 37, 42]
    assert isinstance(watchers, list)


def test_watcher_component_factory_create_all_watchers():
    """测试 WatcherComponentFactory（创建所有 watcher）"""
    all_watchers = WatcherComponentFactory.create_all_watchers()
    
    assert isinstance(all_watchers, dict)
    assert len(all_watchers) == 9
    assert all(isinstance(watcher, WatcherComponent) for watcher in all_watchers.values())
    assert all(watcher_id in [2, 7, 12, 17, 22, 27, 32, 37, 42] for watcher_id in all_watchers.keys())


def test_watcher_component_factory_get_factory_info():
    """测试 WatcherComponentFactory（获取工厂信息）"""
    info = WatcherComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "WatcherComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_watchers"] == 9
    assert info["supported_ids"] == [2, 7, 12, 17, 22, 27, 32, 37, 42]
    assert "created_at" in info
    assert "description" in info


def test_legacy_factory_functions():
    """测试向后兼容的工厂函数"""
    component2 = create_watcher_watcher_component_2()
    assert component2.watcher_id == 2
    
    component7 = create_watcher_watcher_component_7()
    assert component7.watcher_id == 7
    
    component12 = create_watcher_watcher_component_12()
    assert component12.watcher_id == 12
    
    component17 = create_watcher_watcher_component_17()
    assert component17.watcher_id == 17
    
    component22 = create_watcher_watcher_component_22()
    assert component22.watcher_id == 22
    
    component27 = create_watcher_watcher_component_27()
    assert component27.watcher_id == 27
    
    component32 = create_watcher_watcher_component_32()
    assert component32.watcher_id == 32
    
    component37 = create_watcher_watcher_component_37()
    assert component37.watcher_id == 37
    
    component42 = create_watcher_watcher_component_42()
    assert component42.watcher_id == 42


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


def test_watcher_component_process_large_data():
    """测试 WatcherComponent（处理数据，大数据）"""
    component = WatcherComponent(watcher_id=2)
    large_data = {"key" + str(i): "value" + str(i) for i in range(1000)}
    
    result = component.process(large_data)
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_watcher_component_process_nested_data():
    """测试 WatcherComponent（处理数据，嵌套数据）"""
    component = WatcherComponent(watcher_id=7)
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


def test_watcher_component_multiple_instances():
    """测试 WatcherComponent（多个实例）"""
    component1 = WatcherComponent(watcher_id=2)
    component2 = WatcherComponent(watcher_id=7)
    component3 = WatcherComponent(watcher_id=12)
    
    assert component1.watcher_id == 2
    assert component2.watcher_id == 7
    assert component3.watcher_id == 12
    assert component1.component_name != component2.component_name
    assert component2.component_name != component3.component_name


def test_watcher_component_same_id_different_type():
    """测试 WatcherComponent（相同 ID，不同类型）"""
    component1 = WatcherComponent(watcher_id=2, component_type="Type1")
    component2 = WatcherComponent(watcher_id=2, component_type="Type2")
    
    assert component1.watcher_id == component2.watcher_id == 2
    assert component1.component_type == "Type1"
    assert component2.component_type == "Type2"
    assert component1.component_name != component2.component_name

