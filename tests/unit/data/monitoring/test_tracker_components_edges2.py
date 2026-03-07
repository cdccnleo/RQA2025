"""
边界测试：tracker_components.py
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

from src.data.monitoring.tracker_components import (
    ComponentFactory,
    ITrackerComponent,
    TrackerComponent,
    TrackerComponentFactory,
    create_tracker_tracker_component_3,
    create_tracker_tracker_component_8,
    create_tracker_tracker_component_13,
    create_tracker_tracker_component_18,
    create_tracker_tracker_component_23,
    create_tracker_tracker_component_28,
    create_tracker_tracker_component_33,
    create_tracker_tracker_component_38,
    create_tracker_tracker_component_43,
)


def test_itracker_component_abstract():
    """测试 ITrackerComponent（抽象接口）"""
    with pytest.raises(TypeError):
        ITrackerComponent()


def test_tracker_component_init_default():
    """测试 TrackerComponent（初始化，默认类型）"""
    component = TrackerComponent(tracker_id=3)
    assert component.tracker_id == 3
    assert component.component_type == "Tracker"
    assert component.component_name == "Tracker_Component_3"
    assert component.creation_time is not None


def test_tracker_component_init_custom():
    """测试 TrackerComponent（初始化，自定义类型）"""
    component = TrackerComponent(tracker_id=8, component_type="CustomTracker")
    assert component.tracker_id == 8
    assert component.component_type == "CustomTracker"
    assert component.component_name == "CustomTracker_Component_8"


def test_tracker_component_get_tracker_id():
    """测试 TrackerComponent（获取 tracker ID）"""
    component = TrackerComponent(tracker_id=13)
    assert component.get_tracker_id() == 13


def test_tracker_component_get_info():
    """测试 TrackerComponent（获取组件信息）"""
    component = TrackerComponent(tracker_id=18, component_type="TestTracker")
    info = component.get_info()
    
    assert info["tracker_id"] == 18
    assert info["component_name"] == "TestTracker_Component_18"
    assert info["component_type"] == "TestTracker"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"
    assert info["type"] == "unified_data_monitoring_component"


def test_tracker_component_get_status():
    """测试 TrackerComponent（获取组件状态）"""
    component = TrackerComponent(tracker_id=23)
    status = component.get_status()
    
    assert status["tracker_id"] == 23
    assert status["component_name"] == "Tracker_Component_23"
    assert status["component_type"] == "Tracker"
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_tracker_component_process_success():
    """测试 TrackerComponent（处理数据，成功）"""
    component = TrackerComponent(tracker_id=28, component_type="DataTracker")
    data = {"key": "value", "number": 42}
    
    result = component.process(data)
    
    assert result["status"] == "success"
    assert result["tracker_id"] == 28
    assert result["component_name"] == "DataTracker_Component_28"
    assert result["input_data"] == data
    assert "processed_at" in result
    assert "result" in result
    assert result["processing_type"] == "unified_tracker_processing"


def test_tracker_component_process_empty_data():
    """测试 TrackerComponent（处理数据，空数据）"""
    component = TrackerComponent(tracker_id=3)
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_tracker_component_process_none_data():
    """测试 TrackerComponent（处理数据，None 数据）"""
    component = TrackerComponent(tracker_id=8)
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_tracker_component_process_error():
    """测试 TrackerComponent（处理数据，异常）"""
    component = TrackerComponent(tracker_id=13)
    
    class BrokenName:
        def __str__(self):
            raise RuntimeError("processing failed")
    
    component.component_name = BrokenName()
    result = component.process({"test": "data"})
    
    assert result["status"] == "error"
    assert "error" in result
    assert result["error"] == "processing failed"
    assert result["error_type"] == "RuntimeError"


def test_tracker_component_factory_create_component_valid():
    """测试 TrackerComponentFactory（创建组件，有效 ID）"""
    component = TrackerComponentFactory.create_component(3)
    assert isinstance(component, TrackerComponent)
    assert component.tracker_id == 3
    assert component.component_type == "Tracker"


def test_tracker_component_factory_create_component_invalid():
    """测试 TrackerComponentFactory（创建组件，无效 ID）"""
    with pytest.raises(ValueError, match="不支持的tracker ID"):
        TrackerComponentFactory.create_component(99)


def test_tracker_component_factory_create_component_negative():
    """测试 TrackerComponentFactory（创建组件，负 ID）"""
    with pytest.raises(ValueError, match="不支持的tracker ID"):
        TrackerComponentFactory.create_component(-1)


def test_tracker_component_factory_create_component_zero():
    """测试 TrackerComponentFactory（创建组件，零 ID）"""
    with pytest.raises(ValueError, match="不支持的tracker ID"):
        TrackerComponentFactory.create_component(0)


def test_tracker_component_factory_get_available_trackers():
    """测试 TrackerComponentFactory（获取可用 tracker ID 列表）"""
    trackers = TrackerComponentFactory.get_available_trackers()
    assert trackers == [3, 8, 13, 18, 23, 28, 33, 38, 43]
    assert isinstance(trackers, list)


def test_tracker_component_factory_create_all_trackers():
    """测试 TrackerComponentFactory（创建所有 tracker）"""
    all_trackers = TrackerComponentFactory.create_all_trackers()
    
    assert isinstance(all_trackers, dict)
    assert len(all_trackers) == 9
    assert all(isinstance(tracker, TrackerComponent) for tracker in all_trackers.values())
    assert all(tracker_id in [3, 8, 13, 18, 23, 28, 33, 38, 43] for tracker_id in all_trackers.keys())


def test_tracker_component_factory_get_factory_info():
    """测试 TrackerComponentFactory（获取工厂信息）"""
    info = TrackerComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "TrackerComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_trackers"] == 9
    assert info["supported_ids"] == [3, 8, 13, 18, 23, 28, 33, 38, 43]
    assert "created_at" in info
    assert "description" in info


def test_legacy_factory_functions():
    """测试向后兼容的工厂函数"""
    component3 = create_tracker_tracker_component_3()
    assert component3.tracker_id == 3
    
    component8 = create_tracker_tracker_component_8()
    assert component8.tracker_id == 8
    
    component13 = create_tracker_tracker_component_13()
    assert component13.tracker_id == 13
    
    component18 = create_tracker_tracker_component_18()
    assert component18.tracker_id == 18
    
    component23 = create_tracker_tracker_component_23()
    assert component23.tracker_id == 23
    
    component28 = create_tracker_tracker_component_28()
    assert component28.tracker_id == 28
    
    component33 = create_tracker_tracker_component_33()
    assert component33.tracker_id == 33
    
    component38 = create_tracker_tracker_component_38()
    assert component38.tracker_id == 38
    
    component43 = create_tracker_tracker_component_43()
    assert component43.tracker_id == 43


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


def test_tracker_component_process_large_data():
    """测试 TrackerComponent（处理数据，大数据）"""
    component = TrackerComponent(tracker_id=3)
    large_data = {"key" + str(i): "value" + str(i) for i in range(1000)}
    
    result = component.process(large_data)
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_tracker_component_process_nested_data():
    """测试 TrackerComponent（处理数据，嵌套数据）"""
    component = TrackerComponent(tracker_id=8)
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


def test_tracker_component_multiple_instances():
    """测试 TrackerComponent（多个实例）"""
    component1 = TrackerComponent(tracker_id=3)
    component2 = TrackerComponent(tracker_id=8)
    component3 = TrackerComponent(tracker_id=13)
    
    assert component1.tracker_id == 3
    assert component2.tracker_id == 8
    assert component3.tracker_id == 13
    assert component1.component_name != component2.component_name
    assert component2.component_name != component3.component_name


def test_tracker_component_same_id_different_type():
    """测试 TrackerComponent（相同 ID，不同类型）"""
    component1 = TrackerComponent(tracker_id=3, component_type="Type1")
    component2 = TrackerComponent(tracker_id=3, component_type="Type2")
    
    assert component1.tracker_id == component2.tracker_id == 3
    assert component1.component_type == "Type1"
    assert component2.component_type == "Type2"
    assert component1.component_name != component2.component_name

