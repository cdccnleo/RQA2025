"""
边界测试：store_components.py
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

from src.data.cache.store_components import (
    ComponentFactory,
    IStoreComponent,
    StoreComponent,
    StoreComponentFactory,
    create_store_store_component_3,
    create_store_store_component_7,
    create_store_store_component_11,
    create_store_store_component_15,
    create_store_store_component_19,
    create_store_store_component_23,
)


def test_istore_component_abstract():
    """测试 IStoreComponent（抽象接口）"""
    # 抽象类不能直接实例化
    with pytest.raises(TypeError):
        IStoreComponent()


def test_store_component_init_default():
    """测试 StoreComponent（初始化，默认类型）"""
    component = StoreComponent(store_id=3)
    assert component.store_id == 3
    assert component.component_type == "Store"
    assert component.component_name == "Store_Component_3"
    assert component.creation_time is not None


def test_store_component_init_custom():
    """测试 StoreComponent（初始化，自定义类型）"""
    component = StoreComponent(store_id=7, component_type="CustomStore")
    assert component.store_id == 7
    assert component.component_type == "CustomStore"
    assert component.component_name == "CustomStore_Component_7"


def test_store_component_get_store_id():
    """测试 StoreComponent（获取 store ID）"""
    component = StoreComponent(store_id=11)
    assert component.get_store_id() == 11


def test_store_component_get_info():
    """测试 StoreComponent（获取组件信息）"""
    component = StoreComponent(store_id=15, component_type="TestStore")
    info = component.get_info()
    
    assert info["store_id"] == 15
    assert info["component_name"] == "TestStore_Component_15"
    assert info["component_type"] == "TestStore"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"
    assert info["type"] == "unified_data_cache_component"


def test_store_component_get_status():
    """测试 StoreComponent（获取组件状态）"""
    component = StoreComponent(store_id=19)
    status = component.get_status()
    
    assert status["store_id"] == 19
    assert status["component_name"] == "Store_Component_19"
    assert status["component_type"] == "Store"
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_store_component_process_success():
    """测试 StoreComponent（处理数据，成功）"""
    component = StoreComponent(store_id=23, component_type="DataStore")
    data = {"key": "value", "number": 42}
    
    result = component.process(data)
    
    assert result["status"] == "success"
    assert result["store_id"] == 23
    assert result["component_name"] == "DataStore_Component_23"
    assert result["input_data"] == data
    assert "processed_at" in result
    assert "result" in result
    assert result["processing_type"] == "unified_store_processing"


def test_store_component_process_empty_data():
    """测试 StoreComponent（处理数据，空数据）"""
    component = StoreComponent(store_id=3)
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_store_component_process_none_data():
    """测试 StoreComponent（处理数据，None 数据）"""
    component = StoreComponent(store_id=7)
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_store_component_process_error():
    """测试 StoreComponent（处理数据，异常）"""
    component = StoreComponent(store_id=11)
    
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


def test_store_component_factory_create_component_valid():
    """测试 StoreComponentFactory（创建组件，有效 ID）"""
    component = StoreComponentFactory.create_component(3)
    assert isinstance(component, StoreComponent)
    assert component.store_id == 3
    assert component.component_type == "Store"


def test_store_component_factory_create_component_invalid():
    """测试 StoreComponentFactory（创建组件，无效 ID）"""
    with pytest.raises(ValueError, match="不支持的store ID"):
        StoreComponentFactory.create_component(99)


def test_store_component_factory_create_component_negative():
    """测试 StoreComponentFactory（创建组件，负 ID）"""
    with pytest.raises(ValueError, match="不支持的store ID"):
        StoreComponentFactory.create_component(-1)


def test_store_component_factory_create_component_zero():
    """测试 StoreComponentFactory（创建组件，零 ID）"""
    with pytest.raises(ValueError, match="不支持的store ID"):
        StoreComponentFactory.create_component(0)


def test_store_component_factory_get_available_stores():
    """测试 StoreComponentFactory（获取可用 store ID 列表）"""
    stores = StoreComponentFactory.get_available_stores()
    assert stores == [3, 7, 11, 15, 19, 23]
    assert isinstance(stores, list)


def test_store_component_factory_create_all_stores():
    """测试 StoreComponentFactory（创建所有 store）"""
    all_stores = StoreComponentFactory.create_all_stores()
    
    assert isinstance(all_stores, dict)
    assert len(all_stores) == 6
    assert all(isinstance(store, StoreComponent) for store in all_stores.values())
    assert all(store_id in [3, 7, 11, 15, 19, 23] for store_id in all_stores.keys())


def test_store_component_factory_get_factory_info():
    """测试 StoreComponentFactory（获取工厂信息）"""
    info = StoreComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "StoreComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_stores"] == 6
    assert info["supported_ids"] == [3, 7, 11, 15, 19, 23]
    assert "created_at" in info
    assert "description" in info


def test_legacy_factory_functions():
    """测试向后兼容的工厂函数"""
    # 测试所有向后兼容函数
    component3 = create_store_store_component_3()
    assert component3.store_id == 3
    
    component7 = create_store_store_component_7()
    assert component7.store_id == 7
    
    component11 = create_store_store_component_11()
    assert component11.store_id == 11
    
    component15 = create_store_store_component_15()
    assert component15.store_id == 15
    
    component19 = create_store_store_component_19()
    assert component19.store_id == 19
    
    component23 = create_store_store_component_23()
    assert component23.store_id == 23


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


def test_store_component_process_large_data():
    """测试 StoreComponent（处理数据，大数据）"""
    component = StoreComponent(store_id=3)
    large_data = {"key" + str(i): "value" + str(i) for i in range(1000)}
    
    result = component.process(large_data)
    assert result["status"] == "success"
    assert result["input_data"] == large_data


def test_store_component_process_nested_data():
    """测试 StoreComponent（处理数据，嵌套数据）"""
    component = StoreComponent(store_id=7)
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


def test_store_component_multiple_instances():
    """测试 StoreComponent（多个实例）"""
    component1 = StoreComponent(store_id=3)
    component2 = StoreComponent(store_id=7)
    component3 = StoreComponent(store_id=11)
    
    assert component1.store_id == 3
    assert component2.store_id == 7
    assert component3.store_id == 11
    assert component1.component_name != component2.component_name
    assert component2.component_name != component3.component_name


def test_store_component_same_id_different_type():
    """测试 StoreComponent（相同 ID，不同类型）"""
    component1 = StoreComponent(store_id=3, component_type="Type1")
    component2 = StoreComponent(store_id=3, component_type="Type2")
    
    assert component1.store_id == component2.store_id == 3
    assert component1.component_type == "Type1"
    assert component2.component_type == "Type2"
    assert component1.component_name != component2.component_name

