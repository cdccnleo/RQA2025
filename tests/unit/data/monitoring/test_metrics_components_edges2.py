"""
边界测试：metrics_components.py
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

from src.data.monitoring.metrics_components import (
    ComponentFactory,
    IMetricsComponent,
    MetricsComponent,
    MetricsComponentFactory,
    create_metrics_metrics_component_5,
    create_metrics_metrics_component_10,
    create_metrics_metrics_component_15,
    create_metrics_metrics_component_20,
    create_metrics_metrics_component_25,
    create_metrics_metrics_component_30,
    create_metrics_metrics_component_35,
    create_metrics_metrics_component_40,
)


def test_imetrics_component_abstract():
    """测试 IMetricsComponent（抽象接口）"""
    # 抽象类不能直接实例化
    with pytest.raises(TypeError):
        IMetricsComponent()


def test_metrics_component_init_default():
    """测试 MetricsComponent（初始化，默认类型）"""
    component = MetricsComponent(metrics_id=5)
    assert component.metrics_id == 5
    assert component.component_type == "Metrics"
    assert component.component_name == "Metrics_Component_5"
    assert component.creation_time is not None


def test_metrics_component_init_custom():
    """测试 MetricsComponent（初始化，自定义类型）"""
    component = MetricsComponent(metrics_id=10, component_type="CustomMetrics")
    assert component.metrics_id == 10
    assert component.component_type == "CustomMetrics"
    assert component.component_name == "CustomMetrics_Component_10"


def test_metrics_component_get_metrics_id():
    """测试 MetricsComponent（获取 metrics ID）"""
    component = MetricsComponent(metrics_id=15)
    assert component.get_metrics_id() == 15


def test_metrics_component_get_info():
    """测试 MetricsComponent（获取组件信息）"""
    component = MetricsComponent(metrics_id=20, component_type="TestMetrics")
    info = component.get_info()
    
    assert info["metrics_id"] == 20
    assert info["component_name"] == "TestMetrics_Component_20"
    assert info["component_type"] == "TestMetrics"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"
    assert info["type"] == "unified_data_monitoring_component"


def test_metrics_component_get_status():
    """测试 MetricsComponent（获取组件状态）"""
    component = MetricsComponent(metrics_id=25)
    status = component.get_status()
    
    assert status["metrics_id"] == 25
    assert status["component_name"] == "Metrics_Component_25"
    assert status["component_type"] == "Metrics"
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_metrics_component_process_success():
    """测试 MetricsComponent（处理数据，成功）"""
    component = MetricsComponent(metrics_id=30, component_type="DataMetrics")
    data = {"key": "value", "number": 42}
    
    result = component.process(data)
    
    assert result["status"] == "success"
    assert result["metrics_id"] == 30
    assert result["component_name"] == "DataMetrics_Component_30"
    assert result["input_data"] == data
    assert "processed_at" in result
    assert "result" in result
    assert result["processing_type"] == "unified_metrics_processing"


def test_metrics_component_process_empty_data():
    """测试 MetricsComponent（处理数据，空数据）"""
    component = MetricsComponent(metrics_id=5)
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_metrics_component_process_none_data():
    """测试 MetricsComponent（处理数据，None 数据）"""
    component = MetricsComponent(metrics_id=10)
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_metrics_component_process_error():
    """测试 MetricsComponent（处理数据，异常）"""
    component = MetricsComponent(metrics_id=15)
    
    # 创建一个会导致异常的对象
    class BrokenName:
        def __str__(self):
            raise RuntimeError("processing failed")
    
    component.component_name = BrokenName()
    result = component.process({"test": "data"})
    
    assert result["status"] == "error"
    assert "error" in result
    assert result["error"] == "processing failed"
    assert result["error_type"] == "RuntimeError"


def test_metrics_component_factory_create_component_valid():
    """测试 MetricsComponentFactory（创建组件，有效 ID）"""
    component = MetricsComponentFactory.create_component(5)
    assert isinstance(component, MetricsComponent)
    assert component.metrics_id == 5
    assert component.component_type == "Metrics"


def test_metrics_component_factory_create_component_invalid():
    """测试 MetricsComponentFactory（创建组件，无效 ID）"""
    with pytest.raises(ValueError, match="不支持的metrics ID"):
        MetricsComponentFactory.create_component(99)


def test_metrics_component_factory_create_component_negative():
    """测试 MetricsComponentFactory（创建组件，负 ID）"""
    with pytest.raises(ValueError, match="不支持的metrics ID"):
        MetricsComponentFactory.create_component(-1)


def test_metrics_component_factory_create_component_zero():
    """测试 MetricsComponentFactory（创建组件，零 ID）"""
    with pytest.raises(ValueError, match="不支持的metrics ID"):
        MetricsComponentFactory.create_component(0)


def test_metrics_component_factory_get_available_metricss():
    """测试 MetricsComponentFactory（获取可用 metrics ID 列表）"""
    metricss = MetricsComponentFactory.get_available_metricss()
    assert metricss == [5, 10, 15, 20, 25, 30, 35, 40]
    assert isinstance(metricss, list)


def test_metrics_component_factory_create_all_metricss():
    """测试 MetricsComponentFactory（创建所有 metrics）"""
    all_metricss = MetricsComponentFactory.create_all_metricss()
    
    assert isinstance(all_metricss, dict)
    assert len(all_metricss) == 8
    assert all(isinstance(metrics, MetricsComponent) for metrics in all_metricss.values())
    assert all(metrics_id in [5, 10, 15, 20, 25, 30, 35, 40] for metrics_id in all_metricss.keys())


def test_metrics_component_factory_get_factory_info():
    """测试 MetricsComponentFactory（获取工厂信息）"""
    info = MetricsComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "MetricsComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_metricss"] == 8
    assert info["supported_ids"] == [5, 10, 15, 20, 25, 30, 35, 40]
    assert "created_at" in info
    assert "description" in info


def test_legacy_factory_functions():
    """测试向后兼容的工厂函数"""
    # 测试所有向后兼容函数
    component5 = create_metrics_metrics_component_5()
    assert component5.metrics_id == 5
    
    component10 = create_metrics_metrics_component_10()
    assert component10.metrics_id == 10
    
    component15 = create_metrics_metrics_component_15()
    assert component15.metrics_id == 15
    
    component20 = create_metrics_metrics_component_20()
    assert component20.metrics_id == 20
    
    component25 = create_metrics_metrics_component_25()
    assert component25.metrics_id == 25
    
    component30 = create_metrics_metrics_component_30()
    assert component30.metrics_id == 30
    
    component35 = create_metrics_metrics_component_35()
    assert component35.metrics_id == 35
    
    component40 = create_metrics_metrics_component_40()
    assert component40.metrics_id == 40


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
    
    # Mock _create_component_instance 返回一个对象，但 initialize 返回 False
    class MockComponent:
        def initialize(self, config):
            return False
    
    factory._create_component_instance = lambda t, c: MockComponent()
    result = factory.create_component("test_type", {})
    assert result is None


def test_component_factory_create_component_exception():
    """测试 ComponentFactory（创建组件，抛出异常）"""
    factory = ComponentFactory()
    
    # Mock _create_component_instance 抛出异常
    def _bad_create(component_type, config):
        raise RuntimeError("creation failed")
    
    factory._create_component_instance = _bad_create
    result = factory.create_component("test_type", {})
    assert result is None


def test_metrics_component_process_large_data():
    """测试 MetricsComponent（处理数据，大数据）"""
    component = MetricsComponent(metrics_id=5)
    large_data = {"key" + str(i): "value" + str(i) for i in range(1000)}
    
    result = component.process(large_data)
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_metrics_component_process_nested_data():
    """测试 MetricsComponent（处理数据，嵌套数据）"""
    component = MetricsComponent(metrics_id=10)
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


def test_metrics_component_multiple_instances():
    """测试 MetricsComponent（多个实例）"""
    component1 = MetricsComponent(metrics_id=5)
    component2 = MetricsComponent(metrics_id=10)
    component3 = MetricsComponent(metrics_id=15)
    
    assert component1.metrics_id == 5
    assert component2.metrics_id == 10
    assert component3.metrics_id == 15
    assert component1.component_name != component2.component_name
    assert component2.component_name != component3.component_name


def test_metrics_component_same_id_different_type():
    """测试 MetricsComponent（相同 ID，不同类型）"""
    component1 = MetricsComponent(metrics_id=5, component_type="Type1")
    component2 = MetricsComponent(metrics_id=5, component_type="Type2")
    
    assert component1.metrics_id == component2.metrics_id == 5
    assert component1.component_type == "Type1"
    assert component2.component_type == "Type2"
    assert component1.component_name != component2.component_name

