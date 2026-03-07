"""
边界测试：data_source_manager.py
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
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.data.sources.data_source_manager import (
    DataSourceManager,
    DataSource,
    YahooFinanceSource
)


class MockDataSource(DataSource):
    """模拟数据源用于测试"""
    
    def __init__(self, name: str, available: bool = True, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self._available = available
        self._mock_data = pd.DataFrame({'close': [100, 101, 102]})
    
    def check_availability(self) -> bool:
        return self._available
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame()
        return self._mock_data.copy()


def test_data_source_init():
    """测试 DataSource（初始化）"""
    source = MockDataSource("test_source")
    assert source.name == "test_source"
    assert source.is_available is False  # 初始为 False
    assert isinstance(source.config, dict)


def test_data_source_get_info():
    """测试 DataSource（获取信息）"""
    source = MockDataSource("test_source")
    info = source.get_info()
    assert info['name'] == "test_source"
    assert 'available' in info
    assert 'config' in info


def test_yahoo_finance_source_init():
    """测试 YahooFinanceSource（初始化）"""
    source = YahooFinanceSource()
    assert source.name == "Yahoo Finance"
    assert isinstance(source.config, dict)


def test_yahoo_finance_source_check_availability():
    """测试 YahooFinanceSource（检查可用性）"""
    source = YahooFinanceSource()
    # 由于网络请求可能失败，我们只验证方法可以调用
    result = source.check_availability()
    assert isinstance(result, bool)


@patch('requests.head')
def test_yahoo_finance_source_check_availability_mock(mock_head):
    """测试 YahooFinanceSource（检查可用性，mock）"""
    source = YahooFinanceSource()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_head.return_value = mock_response
    
    result = source.check_availability()
    assert result is True
    assert source.is_available is True


@patch('requests.head')
def test_yahoo_finance_source_check_availability_failed(mock_head):
    """测试 YahooFinanceSource（检查可用性，失败）"""
    source = YahooFinanceSource()
    mock_head.side_effect = Exception("Network error")
    
    result = source.check_availability()
    assert result is False
    assert source.is_available is False


@patch('requests.get')
def test_yahoo_finance_source_fetch_data_success(mock_get):
    """测试 YahooFinanceSource（获取数据，成功）"""
    source = YahooFinanceSource()
    mock_response = Mock()
    mock_response.json.return_value = {
        'chart': {
            'result': [{
                'timestamp': [1609459200, 1609545600],
                'indicators': {
                    'quote': [{
                        'open': [100, 101],
                        'high': [102, 103],
                        'low': [99, 100],
                        'close': [101, 102],
                        'volume': [1000, 1100]
                    }]
                }
            }]
        }
    }
    mock_get.return_value = mock_response
    
    df = source.fetch_data("AAPL", "2024-01-01", "2024-01-02")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@patch('requests.get')
def test_yahoo_finance_source_fetch_data_failed(mock_get):
    """测试 YahooFinanceSource（获取数据，失败）"""
    source = YahooFinanceSource()
    mock_get.side_effect = Exception("Network error")
    
    df = source.fetch_data("AAPL", "2024-01-01", "2024-01-02")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@patch('requests.get')
def test_yahoo_finance_source_fetch_data_empty_response(mock_get):
    """测试 YahooFinanceSource（获取数据，空响应）"""
    source = YahooFinanceSource()
    mock_response = Mock()
    mock_response.json.return_value = {'chart': {'result': []}}
    mock_get.return_value = mock_response
    
    df = source.fetch_data("AAPL", "2024-01-01", "2024-01-02")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_data_source_manager_init():
    """测试 DataSourceManager（初始化）"""
    manager = DataSourceManager()
    assert isinstance(manager.sources, dict)
    assert 'yahoo' in manager.sources


def test_data_source_manager_get_available_sources_empty():
    """测试 DataSourceManager（获取可用数据源，空）"""
    manager = DataSourceManager()
    # 由于网络请求可能失败，可用源可能为空
    available = manager.get_available_sources()
    assert isinstance(available, list)


def test_data_source_manager_get_available_sources_with_mock():
    """测试 DataSourceManager（获取可用数据源，mock）"""
    manager = DataSourceManager()
    # 替换为 mock 数据源
    mock_source = MockDataSource("mock", available=True)
    manager.sources['mock'] = mock_source
    
    available = manager.get_available_sources()
    assert 'mock' in available


def test_data_source_manager_fetch_data_with_fallback_no_sources():
    """测试 DataSourceManager（获取数据，后备机制，无可用源）"""
    manager = DataSourceManager()
    # 将所有源设为不可用
    for source in manager.sources.values():
        source.is_available = False
    
    df = manager.fetch_data_with_fallback("AAPL", "2024-01-01", "2024-01-02")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_data_source_manager_fetch_data_with_fallback_preferred_source():
    """测试 DataSourceManager（获取数据，后备机制，首选源）"""
    manager = DataSourceManager()
    # 添加 mock 数据源
    mock_source = MockDataSource("mock", available=True)
    manager.sources['mock'] = mock_source
    
    df = manager.fetch_data_with_fallback(
        "AAPL", "2024-01-01", "2024-01-02",
        preferred_source="mock"
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_data_source_manager_fetch_data_with_fallback_preferred_unavailable():
    """测试 DataSourceManager（获取数据，后备机制，首选源不可用）"""
    manager = DataSourceManager()
    # 添加 mock 数据源作为首选，但不可用
    mock_source1 = MockDataSource("mock1", available=False)
    manager.sources['mock1'] = mock_source1
    # 将 yahoo 替换为可用的 mock 源
    mock_source2 = MockDataSource("yahoo", available=True)
    manager.sources['yahoo'] = mock_source2
    
    df = manager.fetch_data_with_fallback(
        "AAPL", "2024-01-01", "2024-01-02",
        preferred_source="mock1"
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0  # 应该从 yahoo (mock2) 获取


def test_data_source_manager_fetch_data_with_fallback_empty_data():
    """测试 DataSourceManager（获取数据，后备机制，空数据）"""
    manager = DataSourceManager()
    # 添加返回空数据的 mock 源
    empty_source = MockDataSource("empty", available=True)
    empty_source._mock_data = pd.DataFrame()
    manager.sources['empty'] = empty_source
    
    df = manager.fetch_data_with_fallback(
        "AAPL", "2024-01-01", "2024-01-02",
        preferred_source="empty"
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_data_source_manager_fetch_data_with_fallback_none_preferred():
    """测试 DataSourceManager（获取数据，后备机制，None 首选源）"""
    manager = DataSourceManager()
    # 添加 mock 数据源
    mock_source = MockDataSource("mock", available=True)
    manager.sources['mock'] = mock_source
    
    df = manager.fetch_data_with_fallback(
        "AAPL", "2024-01-01", "2024-01-02",
        preferred_source=None
    )
    assert isinstance(df, pd.DataFrame)


def test_data_source_manager_fetch_data_with_fallback_invalid_preferred():
    """测试 DataSourceManager（获取数据，后备机制，无效首选源）"""
    manager = DataSourceManager()
    # 将 yahoo 替换为可用的 mock 源
    mock_source = MockDataSource("yahoo", available=True)
    manager.sources['yahoo'] = mock_source
    
    df = manager.fetch_data_with_fallback(
        "AAPL", "2024-01-01", "2024-01-02",
        preferred_source="nonexistent"
    )
    assert isinstance(df, pd.DataFrame)
    # 应该从 yahoo (mock) 获取
    assert len(df) > 0

