"""
边界测试：observer_components.py
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

from src.data.monitoring.observer_components import (
    ComponentFactory,
    IObserverComponent,
    ObserverComponent,
    ObserverComponentFactory,
    create_observer_observer_component_4,
    create_observer_observer_component_9,
    create_observer_observer_component_14,
    create_observer_observer_component_19,
    create_observer_observer_component_24,
    create_observer_observer_component_29,
    create_observer_observer_component_34,
    create_observer_observer_component_39,
    create_observer_observer_component_44,
)


def test_iobserver_component_abstract():
    """测试 IObserverComponent（抽象接口）"""
    with pytest.raises(TypeError):
        IObserverComponent()


def test_observer_component_init_default():
    """测试 ObserverComponent（初始化，默认类型）"""
    component = ObserverComponent(observer_id=4)
    assert component.observer_id == 4
    assert component.component_type == "Observer"
    assert component.component_name == "Observer_Component_4"
    assert component.creation_time is not None


def test_observer_component_init_custom():
    """测试 ObserverComponent（初始化，自定义类型）"""
    component = ObserverComponent(observer_id=9, component_type="CustomObserver")
    assert component.observer_id == 9
    assert component.component_type == "CustomObserver"
    assert component.component_name == "CustomObserver_Component_9"


def test_observer_component_get_observer_id():
    """测试 ObserverComponent（获取 observer ID）"""
    component = ObserverComponent(observer_id=14)
    assert component.get_observer_id() == 14


def test_observer_component_get_info():
    """测试 ObserverComponent（获取组件信息）"""
    component = ObserverComponent(observer_id=19, component_type="TestObserver")
    info = component.get_info()
    
    assert info["observer_id"] == 19
    assert info["component_name"] == "TestObserver_Component_19"
    assert info["component_type"] == "TestObserver"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"
    assert info["type"] == "unified_data_monitoring_component"


def test_observer_component_get_status():
    """测试 ObserverComponent（获取组件状态）"""
    component = ObserverComponent(observer_id=24)
    status = component.get_status()
    
    assert status["observer_id"] == 24
    assert status["component_name"] == "Observer_Component_24"
    assert status["component_type"] == "Observer"
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_observer_component_process_success():
    """测试 ObserverComponent（处理数据，成功）"""
    component = ObserverComponent(observer_id=29, component_type="DataObserver")
    data = {"key": "value", "number": 42}
    
    result = component.process(data)
    
    assert result["status"] == "success"
    assert result["observer_id"] == 29
    assert result["component_name"] == "DataObserver_Component_29"
    assert result["input_data"] == data
    assert "processed_at" in result
    assert "result" in result
    assert result["processing_type"] == "unified_observer_processing"


def test_observer_component_process_empty_data():
    """测试 ObserverComponent（处理数据，空数据）"""
    component = ObserverComponent(observer_id=4)
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_observer_component_process_none_data():
    """测试 ObserverComponent（处理数据，None 数据）"""
    component = ObserverComponent(observer_id=9)
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_observer_component_process_error():
    """测试 ObserverComponent（处理数据，异常）"""
    component = ObserverComponent(observer_id=14)
    
    class BrokenName:
        def __str__(self):
            raise RuntimeError("processing failed")
    
    component.component_name = BrokenName()
    result = component.process({"test": "data"})
    
    assert result["status"] == "error"
    assert "error" in result
    assert result["error"] == "processing failed"
    assert result["error_type"] == "RuntimeError"


def test_observer_component_factory_create_component_valid():
    """测试 ObserverComponentFactory（创建组件，有效 ID）"""
    component = ObserverComponentFactory.create_component(4)
    assert isinstance(component, ObserverComponent)
    assert component.observer_id == 4
    assert component.component_type == "Observer"


def test_observer_component_factory_create_component_invalid():
    """测试 ObserverComponentFactory（创建组件，无效 ID）"""
    with pytest.raises(ValueError, match="不支持的observer ID"):
        ObserverComponentFactory.create_component(99)


def test_observer_component_factory_create_component_negative():
    """测试 ObserverComponentFactory（创建组件，负 ID）"""
    with pytest.raises(ValueError, match="不支持的observer ID"):
        ObserverComponentFactory.create_component(-1)


def test_observer_component_factory_create_component_zero():
    """测试 ObserverComponentFactory（创建组件，零 ID）"""
    with pytest.raises(ValueError, match="不支持的observer ID"):
        ObserverComponentFactory.create_component(0)


def test_observer_component_factory_get_available_observers():
    """测试 ObserverComponentFactory（获取可用 observer ID 列表）"""
    observers = ObserverComponentFactory.get_available_observers()
    assert observers == [4, 9, 14, 19, 24, 29, 34, 39, 44]
    assert isinstance(observers, list)


def test_observer_component_factory_create_all_observers():
    """测试 ObserverComponentFactory（创建所有 observer）"""
    all_observers = ObserverComponentFactory.create_all_observers()
    
    assert isinstance(all_observers, dict)
    assert len(all_observers) == 9
    assert all(isinstance(observer, ObserverComponent) for observer in all_observers.values())
    assert all(observer_id in [4, 9, 14, 19, 24, 29, 34, 39, 44] for observer_id in all_observers.keys())


def test_observer_component_factory_get_factory_info():
    """测试 ObserverComponentFactory（获取工厂信息）"""
    info = ObserverComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "ObserverComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_observers"] == 9
    assert info["supported_ids"] == [4, 9, 14, 19, 24, 29, 34, 39, 44]
    assert "created_at" in info
    assert "description" in info


def test_legacy_factory_functions():
    """测试向后兼容的工厂函数"""
    component4 = create_observer_observer_component_4()
    assert component4.observer_id == 4
    
    component9 = create_observer_observer_component_9()
    assert component9.observer_id == 9
    
    component14 = create_observer_observer_component_14()
    assert component14.observer_id == 14
    
    component19 = create_observer_observer_component_19()
    assert component19.observer_id == 19
    
    component24 = create_observer_observer_component_24()
    assert component24.observer_id == 24
    
    component29 = create_observer_observer_component_29()
    assert component29.observer_id == 29
    
    component34 = create_observer_observer_component_34()
    assert component34.observer_id == 34
    
    component39 = create_observer_observer_component_39()
    assert component39.observer_id == 39
    
    component44 = create_observer_observer_component_44()
    assert component44.observer_id == 44


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


def test_observer_component_process_large_data():
    """测试 ObserverComponent（处理数据，大数据）"""
    component = ObserverComponent(observer_id=4)
    large_data = {"key" + str(i): "value" + str(i) for i in range(1000)}
    
    result = component.process(large_data)
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_observer_component_process_nested_data():
    """测试 ObserverComponent（处理数据，嵌套数据）"""
    component = ObserverComponent(observer_id=9)
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


def test_observer_component_multiple_instances():
    """测试 ObserverComponent（多个实例）"""
    component1 = ObserverComponent(observer_id=4)
    component2 = ObserverComponent(observer_id=9)
    component3 = ObserverComponent(observer_id=14)
    
    assert component1.observer_id == 4
    assert component2.observer_id == 9
    assert component3.observer_id == 14
    assert component1.component_name != component2.component_name
    assert component2.component_name != component3.component_name


def test_observer_component_same_id_different_type():
    """测试 ObserverComponent（相同 ID，不同类型）"""
    component1 = ObserverComponent(observer_id=4, component_type="Type1")
    component2 = ObserverComponent(observer_id=4, component_type="Type2")
    
    assert component1.observer_id == component2.observer_id == 4
    assert component1.component_type == "Type1"
    assert component2.component_type == "Type2"
    assert component1.component_name != component2.component_name

