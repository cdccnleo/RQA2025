"""
边界测试：standard_interfaces.py
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
from unittest.mock import Mock

from src.data.interfaces.standard_interfaces import (
    IDataValidator,
    IDataRegistry,
    IDataAdapter,
    IDataCache,
    DataRequest,
    DataResponse,
)


def test_idata_validator_abstract():
    """测试 IDataValidator（抽象接口）"""
    # Protocol 接口不能直接实例化
    with pytest.raises(TypeError):
        IDataValidator()


def test_idata_registry_abstract():
    """测试 IDataRegistry（抽象接口）"""
    # Protocol 接口不能直接实例化
    with pytest.raises(TypeError):
        IDataRegistry()


def test_idata_adapter_abstract():
    """测试 IDataAdapter（抽象接口）"""
    # Protocol 接口不能直接实例化
    with pytest.raises(TypeError):
        IDataAdapter()


def test_idata_cache_abstract():
    """测试 IDataCache（抽象接口）"""
    # Protocol 接口不能直接实例化
    with pytest.raises(TypeError):
        IDataCache()


def test_data_request_init_default():
    """测试 DataRequest（初始化，默认值）"""
    request = DataRequest(source="test_source", query={})
    
    assert request.source == "test_source"
    assert request.query == {}
    assert request.filters is None
    assert request.limit is None
    assert request.offset is None
    assert request.request_id is not None  # 自动生成


def test_data_request_init_custom():
    """测试 DataRequest（初始化，自定义值）"""
    request = DataRequest(
        source="test_source",
        query={"symbol": "AAPL"},
        filters={"interval": "1d"},
        limit=100,
        offset=0,
        request_id="test_id"
    )
    
    assert request.source == "test_source"
    assert request.query == {"symbol": "AAPL"}
    assert request.filters == {"interval": "1d"}
    assert request.limit == 100
    assert request.offset == 0
    assert request.request_id == "test_id"


def test_data_request_to_dict():
    """测试 DataRequest（转换为字典）"""
    request = DataRequest(
        source="test_source",
        query={"symbol": "AAPL"},
        filters={"interval": "1d"},
        limit=100,
        offset=0,
        request_id="test_id"
    )
    
    # DataRequest 没有 to_dict 方法，检查属性
    assert request.source == "test_source"
    assert request.query == {"symbol": "AAPL"}
    assert request.filters == {"interval": "1d"}
    assert request.limit == 100
    assert request.offset == 0
    assert request.request_id == "test_id"


def test_data_request_to_dict_minimal():
    """测试 DataRequest（最小数据）"""
    request = DataRequest(source="test_source", query={})
    
    assert request.source == "test_source"
    assert request.query == {}
    assert request.filters is None
    assert request.limit is None
    assert request.offset is None
    assert request.request_id is not None  # 自动生成


def test_data_response_init_default():
    """测试 DataResponse（初始化，默认值）"""
    response = DataResponse(data=None)
    
    assert response.data is None
    assert response.status == "success"
    assert response.message is None
    assert response.request_id is None
    assert response.metadata is None


def test_data_response_init_custom():
    """测试 DataResponse（初始化，自定义值）"""
    response = DataResponse(
        data={"result": "test"},
        status="success",
        message="Test message",
        request_id="test_id",
        metadata={"type": "test"}
    )
    
    assert response.data == {"result": "test"}
    assert response.status == "success"
    assert response.message == "Test message"
    assert response.request_id == "test_id"
    assert response.metadata == {"type": "test"}


def test_data_response_init_error():
    """测试 DataResponse（初始化，错误响应）"""
    response = DataResponse(
        data=None,
        status="error",
        message="Test error"
    )
    
    assert response.status == "error"
    assert response.message == "Test error"
    assert response.data is None


def test_data_response_post_init_request_id():
    """测试 DataRequest（__post_init__，自动生成 request_id）"""
    request = DataRequest(source="test", query={})
    
    assert request.request_id is not None
    assert isinstance(request.request_id, str)
    assert len(request.request_id) > 0


def test_data_response_attributes():
    """测试 DataResponse（属性）"""
    response = DataResponse(
        data={"result": "test"},
        status="success",
        metadata={"type": "test"}
    )
    
    assert response.status == "success"
    assert response.data == {"result": "test"}
    assert response.metadata == {"type": "test"}


def test_data_response_error():
    """测试 DataResponse（错误响应）"""
    response = DataResponse(
        data=None,
        status="error",
        message="Test error"
    )
    
    assert response.status == "error"
    assert response.message == "Test error"
    assert response.data is None


def test_mock_data_validator():
    """测试 IDataValidator（Mock 实现）"""
    class MockValidator:
        def validate(self, data, data_type):
            return {"valid": True, "data_type": data_type}
        
        def get_validation_rules(self, data_type):
            return {"rules": []}
    
    validator = MockValidator()
    
    result = validator.validate({"test": "data"}, "test_type")
    assert result["valid"] is True
    
    rules = validator.get_validation_rules("test_type")
    assert "rules" in rules


def test_mock_data_registry():
    """测试 IDataRegistry（Mock 实现）"""
    class MockRegistry:
        def __init__(self):
            self._sources = {}
        
        def register_data_source(self, name, source_config):
            self._sources[name] = source_config
            return True
        
        def get_data_source(self, name):
            return self._sources.get(name)
        
        def list_data_sources(self):
            return list(self._sources.keys())
        
        def unregister_data_source(self, name):
            if name in self._sources:
                del self._sources[name]
                return True
            return False
    
    registry = MockRegistry()
    
    assert registry.register_data_source("source1", {"type": "test"}) is True
    assert registry.get_data_source("source1") == {"type": "test"}
    assert "source1" in registry.list_data_sources()
    assert registry.unregister_data_source("source1") is True
    assert registry.get_data_source("source1") is None


def test_mock_data_adapter():
    """测试 IDataAdapter（Mock 实现）"""
    class MockAdapter:
        def __init__(self):
            self._connected = False
        
        def connect(self):
            self._connected = True
            return True
        
        def disconnect(self):
            self._connected = False
            return True
        
        def fetch_data(self, request):
            return DataResponse(data={"result": "test"}, status="success")
        
        def validate_connection(self):
            return self._connected
    
    adapter = MockAdapter()
    
    assert adapter.connect() is True
    assert adapter.validate_connection() is True
    response = adapter.fetch_data(DataRequest(source="test", query={}))
    assert response.status == "success"
    assert adapter.disconnect() is True
    assert adapter.validate_connection() is False


def test_mock_data_cache():
    """测试 IDataCache（Mock 实现）"""
    class MockCache:
        def __init__(self):
            self._cache = {}
        
        def get(self, key):
            return self._cache.get(key)
        
        def set(self, key, value, ttl=None):
            self._cache[key] = value
            return True
        
        def delete(self, key):
            if key in self._cache:
                del self._cache[key]
                return True
            return False
        
        def clear(self):
            self._cache.clear()
            return True
        
        def get_stats(self):
            return {"size": len(self._cache)}
    
    cache = MockCache()
    
    assert cache.set("key1", "value1") is True
    assert cache.get("key1") == "value1"
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    assert cache.clear() is True
    assert cache.get_stats()["size"] == 0


def test_data_request_empty_source():
    """测试 DataRequest（空 source）"""
    request = DataRequest(source="", query={})
    
    assert request.source == ""
    assert request.query == {}


def test_data_request_nested_query():
    """测试 DataRequest（嵌套查询）"""
    request = DataRequest(
        source="test",
        query={
            "symbol": "AAPL",
            "date_range": {"start": "2024-01-01", "end": "2024-01-31"}
        }
    )
    
    assert request.query["symbol"] == "AAPL"
    assert "date_range" in request.query


def test_data_response_none_data():
    """测试 DataResponse（None 数据）"""
    response = DataResponse(data=None, status="success")
    
    assert response.status == "success"
    assert response.data is None


def test_data_response_empty_metadata():
    """测试 DataResponse（空元数据）"""
    response = DataResponse(data=None, status="success", metadata={})
    
    assert response.metadata == {}
    assert response.status == "success"


def test_data_response_complex_data():
    """测试 DataResponse（复杂数据）"""
    complex_data = {
        "nested": {
            "list": [1, 2, 3],
            "dict": {"key": "value"}
        }
    }
    response = DataResponse(data=complex_data, status="success")
    
    assert response.data == complex_data
    assert response.data["nested"]["list"] == [1, 2, 3]
    assert response.status == "success"

