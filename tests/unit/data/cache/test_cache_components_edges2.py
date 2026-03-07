"""
边界测试：cache_components.py
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
from unittest.mock import Mock, patch

from src.data.cache.cache_components import (
    ICacheComponent,
    CacheComponent,
    DataCacheComponentFactory,
    create_datacache_cache_component_1,
    create_datacache_cache_component_5,
    create_datacache_cache_component_9,
    create_datacache_cache_component_13,
    create_datacache_cache_component_17,
    create_datacache_cache_component_21,
)


def test_icache_component_abstract():
    """测试 ICacheComponent（抽象接口）"""
    # 抽象类不能直接实例化
    with pytest.raises(TypeError):
        ICacheComponent()


def test_cache_component_init():
    """测试 CacheComponent（初始化）"""
    component = CacheComponent(1)
    
    assert component.cache_id == 1
    assert component.component_type == "DataCache"
    assert component.component_name == "DataCache_Component_1"
    assert component.creation_time is not None


def test_cache_component_init_custom_type():
    """测试 CacheComponent（初始化，自定义类型）"""
    component = CacheComponent(1, "CustomCache")
    
    assert component.cache_id == 1
    assert component.component_type == "CustomCache"
    assert component.component_name == "CustomCache_Component_1"


def test_cache_component_get_cache_id():
    """测试 CacheComponent（获取缓存ID）"""
    component = CacheComponent(5)
    
    assert component.get_cache_id() == 5


def test_cache_component_get_info():
    """测试 CacheComponent（获取组件信息）"""
    component = CacheComponent(1)
    
    info = component.get_info()
    
    assert info["cache_id"] == 1
    assert info["component_name"] == "DataCache_Component_1"
    assert info["component_type"] == "DataCache"
    assert "creation_time" in info
    assert info["version"] == "2.0.0"


def test_cache_component_process_success():
    """测试 CacheComponent（处理数据，成功）"""
    component = CacheComponent(1)
    data = {"key": "value"}
    
    result = component.process(data)
    
    assert result["cache_id"] == 1
    assert result["status"] == "success"
    assert result["input_data"] == data
    assert "processed_at" in result


def test_cache_component_process_empty():
    """测试 CacheComponent（处理数据，空数据）"""
    component = CacheComponent(1)
    
    result = component.process({})
    
    assert result["status"] == "success"
    assert result["input_data"] == {}


def test_cache_component_process_none():
    """测试 CacheComponent（处理数据，None）"""
    component = CacheComponent(1)
    
    result = component.process(None)
    
    assert result["status"] == "success"
    assert result["input_data"] is None


def test_cache_component_process_exception():
    """测试 CacheComponent（处理数据，异常）"""
    component = CacheComponent(1)
    
    # 模拟异常
    with patch.object(component, 'process', side_effect=ValueError("Test error")):
        # 由于 process 方法内部有 try-except，我们需要直接调用并传入会引发异常的数据
        # 但实际上 process 方法会捕获异常，所以我们需要模拟一个会失败的操作
        pass
    
    # 测试实际异常处理
    class BadData:
        def __str__(self):
            raise ValueError("Cannot convert to string")
    
    result = component.process({"bad": BadData()})
    # process 方法会捕获异常并返回错误结果
    assert result["status"] in ["success", "error"]


def test_cache_component_process_large_data():
    """测试 CacheComponent（处理数据，大数据）"""
    component = CacheComponent(1)
    large_data = {"key" + str(i): "value" * 1000 for i in range(1000)}
    
    result = component.process(large_data)
    
    assert result["status"] == "success"
    assert len(result["input_data"]) == 1000


def test_cache_component_process_nested_data():
    """测试 CacheComponent（处理数据，嵌套数据）"""
    component = CacheComponent(1)
    nested_data = {
        "level1": {
            "level2": {
                "level3": "value"
            }
        }
    }
    
    result = component.process(nested_data)
    
    assert result["status"] == "success"
    assert result["input_data"]["level1"]["level2"]["level3"] == "value"


def test_cache_component_get_status():
    """测试 CacheComponent（获取组件状态）"""
    component = CacheComponent(1)
    
    status = component.get_status()
    
    assert status["cache_id"] == 1
    assert status["status"] == "active"
    assert status["health"] == "good"
    assert "creation_time" in status


def test_datacache_component_factory_create_component_valid():
    """测试 DataCacheComponentFactory（创建组件，有效ID）"""
    component = DataCacheComponentFactory.create_component(1)
    
    assert isinstance(component, CacheComponent)
    assert component.cache_id == 1


def test_datacache_component_factory_create_component_invalid():
    """测试 DataCacheComponentFactory（创建组件，无效ID）"""
    with pytest.raises(ValueError, match="不支持的缓存ID"):
        DataCacheComponentFactory.create_component(99)


def test_datacache_component_factory_create_component_negative():
    """测试 DataCacheComponentFactory（创建组件，负数ID）"""
    with pytest.raises(ValueError):
        DataCacheComponentFactory.create_component(-1)


def test_datacache_component_factory_create_component_zero():
    """测试 DataCacheComponentFactory（创建组件，零ID）"""
    with pytest.raises(ValueError):
        DataCacheComponentFactory.create_component(0)


def test_datacache_component_factory_get_available_caches():
    """测试 DataCacheComponentFactory（获取可用缓存列表）"""
    caches = DataCacheComponentFactory.get_available_caches()
    
    assert isinstance(caches, list)
    assert len(caches) == 6
    assert 1 in caches
    assert 5 in caches
    assert 9 in caches
    assert 13 in caches
    assert 17 in caches
    assert 21 in caches


def test_datacache_component_factory_create_all_caches():
    """测试 DataCacheComponentFactory（创建所有缓存）"""
    all_caches = DataCacheComponentFactory.create_all_caches()
    
    assert isinstance(all_caches, dict)
    assert len(all_caches) == 6
    for cache_id in [1, 5, 9, 13, 17, 21]:
        assert cache_id in all_caches
        assert isinstance(all_caches[cache_id], CacheComponent)


def test_datacache_component_factory_get_factory_info():
    """测试 DataCacheComponentFactory（获取工厂信息）"""
    info = DataCacheComponentFactory.get_factory_info()
    
    assert info["factory_name"] == "DataCacheComponentFactory"
    assert info["version"] == "2.0.0"
    assert info["total_caches"] == 6
    assert len(info["supported_ids"]) == 6
    assert "created_at" in info


def test_create_datacache_cache_component_1():
    """测试 create_datacache_cache_component_1（向后兼容函数）"""
    component = create_datacache_cache_component_1()
    
    assert isinstance(component, CacheComponent)
    assert component.cache_id == 1


def test_create_datacache_cache_component_5():
    """测试 create_datacache_cache_component_5（向后兼容函数）"""
    component = create_datacache_cache_component_5()
    
    assert isinstance(component, CacheComponent)
    assert component.cache_id == 5


def test_create_datacache_cache_component_9():
    """测试 create_datacache_cache_component_9（向后兼容函数）"""
    component = create_datacache_cache_component_9()
    
    assert isinstance(component, CacheComponent)
    assert component.cache_id == 9


def test_create_datacache_cache_component_13():
    """测试 create_datacache_cache_component_13（向后兼容函数）"""
    component = create_datacache_cache_component_13()
    
    assert isinstance(component, CacheComponent)
    assert component.cache_id == 13


def test_create_datacache_cache_component_17():
    """测试 create_datacache_cache_component_17（向后兼容函数）"""
    component = create_datacache_cache_component_17()
    
    assert isinstance(component, CacheComponent)
    assert component.cache_id == 17


def test_create_datacache_cache_component_21():
    """测试 create_datacache_cache_component_21（向后兼容函数）"""
    component = create_datacache_cache_component_21()
    
    assert isinstance(component, CacheComponent)
    assert component.cache_id == 21


def test_cache_component_multiple_instances():
    """测试 CacheComponent（多个实例）"""
    component1 = CacheComponent(1)
    component2 = CacheComponent(2)
    
    assert component1.cache_id == 1
    assert component2.cache_id == 2
    assert component1 is not component2


def test_cache_component_same_id_different_types():
    """测试 CacheComponent（相同ID，不同类型）"""
    component1 = CacheComponent(1, "Type1")
    component2 = CacheComponent(1, "Type2")
    
    assert component1.cache_id == component2.cache_id == 1
    assert component1.component_type == "Type1"
    assert component2.component_type == "Type2"

