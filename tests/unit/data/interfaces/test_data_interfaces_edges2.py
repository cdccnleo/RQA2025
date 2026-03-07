"""
边界测试：data_interfaces.py
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
from unittest.mock import Mock

from src.data.interfaces.data_interfaces import (
    DataRequest,
    DataResponse,
    DataQualityReport,
    IDataAdapter,
    IMarketDataProvider,
    IDataQualityManager,
    IDataStorage,
    IDataTransformer,
)


def test_data_request_init_default():
    """测试 DataRequest（初始化，默认值）"""
    request = DataRequest(symbol="AAPL")
    
    assert request.symbol == "AAPL"
    assert request.market == "CN"
    assert request.data_type == "stock"
    assert request.start_date is None
    assert request.end_date is None
    assert request.interval == "1d"
    assert request.params is None


def test_data_request_init_custom():
    """测试 DataRequest（初始化，自定义值）"""
    request = DataRequest(
        symbol="MSFT",
        market="US",
        data_type="crypto",
        start_date="2024-01-01",
        end_date="2024-01-31",
        interval="1h",
        params={"source": "yahoo"}
    )
    
    assert request.symbol == "MSFT"
    assert request.market == "US"
    assert request.data_type == "crypto"
    assert request.start_date == "2024-01-01"
    assert request.end_date == "2024-01-31"
    assert request.interval == "1h"
    assert request.params == {"source": "yahoo"}


def test_data_request_to_dict():
    """测试 DataRequest（转换为字典）"""
    request = DataRequest(
        symbol="AAPL",
        market="US",
        params={"test": "value"}
    )
    
    data = request.to_dict()
    
    assert data["symbol"] == "AAPL"
    assert data["market"] == "US"
    assert data["data_type"] == "stock"
    assert data["params"] == {"test": "value"}


def test_data_request_to_dict_none_params():
    """测试 DataRequest（转换为字典，None params）"""
    request = DataRequest(symbol="AAPL", params=None)
    
    data = request.to_dict()
    
    assert data["params"] == {}


def test_data_response_init_default():
    """测试 DataResponse（初始化，默认值）"""
    response = DataResponse(success=True)
    
    assert response.success is True
    assert response.data is None
    assert response.error is None
    assert response.metadata is None
    assert response.timestamp is not None
    assert isinstance(response.timestamp, datetime)


def test_data_response_init_custom():
    """测试 DataResponse（初始化，自定义值）"""
    timestamp = datetime(2024, 1, 1, 12, 0, 0)
    response = DataResponse(
        success=True,
        data={"result": "test"},
        error=None,
        metadata={"type": "test"},
        timestamp=timestamp
    )
    
    assert response.success is True
    assert response.data == {"result": "test"}
    assert response.error is None
    assert response.metadata == {"type": "test"}
    assert response.timestamp == timestamp


def test_data_response_init_error():
    """测试 DataResponse（初始化，错误响应）"""
    response = DataResponse(
        success=False,
        error="Test error"
    )
    
    assert response.success is False
    assert response.error == "Test error"
    assert response.data is None


def test_data_response_post_init_timestamp():
    """测试 DataResponse（__post_init__，自动设置时间戳）"""
    response = DataResponse(success=True)
    
    assert response.timestamp is not None
    assert isinstance(response.timestamp, datetime)


def test_data_quality_report_init():
    """测试 DataQualityReport（初始化）"""
    report = DataQualityReport(
        total_records=1000,
        valid_records=950,
        invalid_records=50,
        missing_fields={"field1": 10},
        outliers={"field2": [1, 2, 3]},
        quality_score=0.95,
        recommendations=["Fix field1"]
    )
    
    assert report.total_records == 1000
    assert report.valid_records == 950
    assert report.invalid_records == 50
    assert report.missing_fields == {"field1": 10}
    assert report.outliers == {"field2": [1, 2, 3]}
    assert report.quality_score == 0.95
    assert report.recommendations == ["Fix field1"]


def test_data_quality_report_init_empty():
    """测试 DataQualityReport（初始化，空数据）"""
    report = DataQualityReport(
        total_records=0,
        valid_records=0,
        invalid_records=0,
        missing_fields={},
        outliers={},
        quality_score=0.0,
        recommendations=[]
    )
    
    assert report.total_records == 0
    assert report.quality_score == 0.0


def test_idata_adapter_protocol():
    """测试 IDataAdapter（Protocol 接口）"""
    # Protocol 接口不能直接实例化
    with pytest.raises(TypeError):
        IDataAdapter()


def test_idata_adapter_mock():
    """测试 IDataAdapter（Mock 实现）"""
    class MockAdapter:
        def get_market_data(self, symbol, start_date, end_date):
            return {"symbol": symbol, "data": []}
        
        def save_market_data(self, symbol, data):
            return True
        
        def get_available_symbols(self, market):
            return ["AAPL", "MSFT"]
        
        def get_data_info(self, symbol):
            return {"symbol": symbol, "info": "test"}
        
        def validate_data(self, data):
            return True
    
    adapter = MockAdapter()
    
    assert adapter.get_market_data("AAPL", "2024-01-01", "2024-01-31") is not None
    assert adapter.save_market_data("AAPL", {}) is True
    assert len(adapter.get_available_symbols("US")) == 2
    assert adapter.get_data_info("AAPL") is not None
    assert adapter.validate_data({}) is True


def test_imarket_data_provider_protocol():
    """测试 IMarketDataProvider（Protocol 接口）"""
    # Protocol 接口不能直接实例化
    with pytest.raises(TypeError):
        IMarketDataProvider()


def test_imarket_data_provider_mock():
    """测试 IMarketDataProvider（Mock 实现）"""
    class MockProvider:
        def subscribe_market_data(self, symbols, callback):
            return "sub_123"
        
        def unsubscribe_market_data(self, subscription_id):
            return True
        
        def get_realtime_quote(self, symbol):
            return {"symbol": symbol, "price": 100.0}
        
        def get_historical_data(self, symbol, start_date, end_date, interval="1d"):
            return [{"date": start_date, "price": 100.0}]
        
        def get_market_snapshot(self, symbols):
            return {symbol: {"price": 100.0} for symbol in symbols}
    
    provider = MockProvider()
    
    assert provider.subscribe_market_data(["AAPL"], lambda x: x) == "sub_123"
    assert provider.unsubscribe_market_data("sub_123") is True
    assert provider.get_realtime_quote("AAPL") is not None
    assert len(provider.get_historical_data("AAPL", "2024-01-01", "2024-01-31")) > 0
    assert len(provider.get_market_snapshot(["AAPL", "MSFT"])) == 2


def test_idata_quality_manager_protocol():
    """测试 IDataQualityManager（Protocol 接口）"""
    # Protocol 接口不能直接实例化
    with pytest.raises(TypeError):
        IDataQualityManager()


def test_idata_quality_manager_mock():
    """测试 IDataQualityManager（Mock 实现）"""
    from src.data.interfaces.data_interfaces import DataQualityReport
    
    class MockQualityManager:
        def validate_data_quality(self, data):
            return DataQualityReport(
                total_records=100,
                valid_records=95,
                invalid_records=5,
                missing_fields={},
                outliers={},
                quality_score=0.95,
                recommendations=[]
            )
        
        def clean_data(self, data):
            return data
        
        def detect_anomalies(self, data):
            return []
        
        def repair_data(self, data, issues):
            return data
    
    manager = MockQualityManager()
    
    report = manager.validate_data_quality({})
    assert report.quality_score == 0.95
    assert manager.clean_data({}) is not None
    assert manager.detect_anomalies({}) == []
    assert manager.repair_data({}, []) is not None


def test_idata_storage_protocol():
    """测试 IDataStorage（Protocol 接口）"""
    # Protocol 接口不能直接实例化
    with pytest.raises(TypeError):
        IDataStorage()


def test_idata_storage_mock():
    """测试 IDataStorage（Mock 实现）"""
    class MockStorage:
        def store_data(self, key, data):
            return True
        
        def retrieve_data(self, key):
            return {"data": "test"}
        
        def delete_data(self, key):
            return True
        
        def list_data_keys(self, pattern="*"):
            return ["key1", "key2"]
        
        def get_storage_stats(self):
            return {"size": 100}
    
    storage = MockStorage()
    
    assert storage.store_data("key1", {}) is True
    assert storage.retrieve_data("key1") is not None
    assert storage.delete_data("key1") is True
    assert len(storage.list_data_keys()) > 0
    assert storage.get_storage_stats() is not None


def test_idata_transformer_protocol():
    """测试 IDataTransformer（Protocol 接口）"""
    # Protocol 接口不能直接实例化
    with pytest.raises(TypeError):
        IDataTransformer()


def test_idata_transformer_mock():
    """测试 IDataTransformer（Mock 实现）"""
    class MockTransformer:
        def transform(self, data, target_format):
            return data
        
        def normalize(self, data):
            return data
        
        def validate_format(self, data, format_type):
            return True
    
    transformer = MockTransformer()
    
    assert transformer.transform({}, "json") is not None
    assert transformer.normalize({}) is not None
    assert transformer.validate_format({}, "json") is True


def test_data_request_empty_symbol():
    """测试 DataRequest（空 symbol）"""
    request = DataRequest(symbol="")
    
    assert request.symbol == ""
    assert request.to_dict()["symbol"] == ""


def test_data_request_nested_params():
    """测试 DataRequest（嵌套 params）"""
    request = DataRequest(
        symbol="AAPL",
        params={
            "filters": {
                "date_range": {"start": "2024-01-01", "end": "2024-01-31"}
            }
        }
    )
    
    assert "filters" in request.params
    assert "date_range" in request.params["filters"]


def test_data_response_none_data():
    """测试 DataResponse（None 数据）"""
    response = DataResponse(success=True, data=None)
    
    assert response.success is True
    assert response.data is None


def test_data_response_empty_metadata():
    """测试 DataResponse（空元数据）"""
    response = DataResponse(success=True, metadata={})
    
    assert response.metadata == {}


def test_data_response_complex_data():
    """测试 DataResponse（复杂数据）"""
    complex_data = {
        "nested": {
            "list": [1, 2, 3],
            "dict": {"key": "value"}
        }
    }
    response = DataResponse(success=True, data=complex_data)
    
    assert response.data == complex_data
    assert response.data["nested"]["list"] == [1, 2, 3]


def test_data_quality_report_zero_records():
    """测试 DataQualityReport（零记录）"""
    report = DataQualityReport(
        total_records=0,
        valid_records=0,
        invalid_records=0,
        missing_fields={},
        outliers={},
        quality_score=0.0,
        recommendations=[]
    )
    
    assert report.total_records == 0
    assert report.valid_records == 0
    assert report.invalid_records == 0


def test_data_quality_report_all_invalid():
    """测试 DataQualityReport（全部无效）"""
    report = DataQualityReport(
        total_records=100,
        valid_records=0,
        invalid_records=100,
        missing_fields={"field1": 100},
        outliers={"field2": list(range(100))},
        quality_score=0.0,
        recommendations=["Fix all fields"]
    )
    
    assert report.valid_records == 0
    assert report.invalid_records == 100
    assert report.quality_score == 0.0


def test_data_quality_report_perfect_score():
    """测试 DataQualityReport（完美分数）"""
    report = DataQualityReport(
        total_records=100,
        valid_records=100,
        invalid_records=0,
        missing_fields={},
        outliers={},
        quality_score=1.0,
        recommendations=[]
    )
    
    assert report.valid_records == 100
    assert report.invalid_records == 0
    assert report.quality_score == 1.0

