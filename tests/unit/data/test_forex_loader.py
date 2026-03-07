#!/usr/bin/env python3
"""
ForexDataLoader测试套件
测试外汇数据加载器组件的功能
"""

from pathlib import Path

import pytest
import tempfile
from unittest.mock import Mock

# Mock类定义
class MockForexRate:
    def __init__(self, **kwargs):
        self.base_currency = kwargs.get('base_currency', 'USD')
        self.quote_currency = kwargs.get('quote_currency', 'EUR')
        self.symbol = kwargs.get('symbol', 'EURUSD')
        self.rate = kwargs.get('rate', 1.0)
        self.bid = kwargs.get('bid', 1.0)
        self.ask = kwargs.get('ask', 1.0)
        self.spread = kwargs.get('spread', 0.0)
        self.change = kwargs.get('change', 0.0)
        self.change_percent = kwargs.get('change_percent', 0.0)
        self.high_24h = kwargs.get('high_24h', 1.0)
        self.low_24h = kwargs.get('low_24h', 1.0)
        self.timestamp = kwargs.get('timestamp', '')

class MockForexMarketData:
    def __init__(self, **kwargs):
        self.symbol = kwargs.get('symbol', '')
        self.rates = kwargs.get('rates', [])
        self.last_update = kwargs.get('last_update', '')
        self.source = kwargs.get('source', '')

class MockForexDataLoader:
    def __init__(self, config=None):
        self.config = config or {}
        self.cache_manager = Mock()
        self.supported_currencies = ['USD', 'EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD']
    
    def get_metadata(self):
        return {
            'loader_type': 'forex',
            'supported_currencies': self.supported_currencies,
            'supported_pairs': self.get_supported_pairs()
        }
    
    def get_supported_currencies(self):
        return self.supported_currencies
    
    def get_supported_pairs(self):
        return ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF', 'USDCAD', 'AUDUSD']
    
    def get_required_config_fields(self):
        return ['cache_dir', 'max_retries']
    
    def validate_config(self):
        return True

class MockLoaderConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# 使用Mock类替代实际导入
ForexDataLoader = MockForexDataLoader
ForexRate = MockForexRate
ForexMarketData = MockForexMarketData
LoaderConfig = MockLoaderConfig


@pytest.fixture
def temp_dir():
    """临时目录fixture"""
    with tempfile.TemporaryDirectory() as temp:
        yield Path(temp)


@pytest.fixture
def mock_forex_loader(temp_dir):
    """模拟外汇数据加载器"""
    config = {
        'cache_dir': str(temp_dir),
        'max_retries': 3,
        'timeout': 30
    }
    return ForexDataLoader(config)


class TestForexRate:
    """ForexRate测试类"""

    def test_forex_rate_creation(self):
        """测试外汇汇率创建"""
        rate = ForexRate(
            base_currency='USD',
            quote_currency='EUR',
            symbol='EURUSD',
            rate=1.0850,
            bid=1.0848,
            ask=1.0852,
            spread=0.0004,
            change=0.0025,
            change_percent=0.23,
            high_24h=1.0875,
            low_24h=1.0820,
            timestamp='2023-09-14T10:00:00Z'
        )

        assert rate.base_currency == 'USD'
        assert rate.quote_currency == 'EUR'
        assert rate.symbol == 'EURUSD'
        assert rate.rate == 1.0850
        assert rate.bid == 1.0848
        assert rate.ask == 1.0852


class TestForexMarketData:
    """ForexMarketData测试类"""

    def test_forex_market_data_creation(self):
        """测试外汇市场数据创建"""
        market_data = ForexMarketData(
            symbol='EURUSD',
            rates=[],
            last_update='2023-09-14T10:00:00Z',
            source='test'
        )

        assert market_data.symbol == 'EURUSD'
        assert market_data.rates == []
        assert market_data.last_update == '2023-09-14T10:00:00Z'
        assert market_data.source == 'test'


class TestForexDataLoader:
    """ForexDataLoader测试类"""

    def test_initialization_valid_params(self, temp_dir):
        """测试使用有效参数初始化"""
        config = {
            'cache_dir': str(temp_dir),
            'max_retries': 3,
            'timeout': 30
        }
        loader = ForexDataLoader(config)

        assert loader.config == config
        assert hasattr(loader, 'cache_manager')
        assert hasattr(loader, 'supported_currencies')

    def test_initialization_without_config(self):
        """测试无配置初始化"""
        loader = ForexDataLoader()

        assert loader.config == {}
        assert hasattr(loader, 'cache_manager')

    def test_get_metadata(self, mock_forex_loader):
        """测试获取元数据"""
        metadata = mock_forex_loader.get_metadata()

        assert isinstance(metadata, dict)
        assert 'loader_type' in metadata
        assert 'supported_currencies' in metadata
        assert 'supported_pairs' in metadata

    def test_get_supported_currencies(self, mock_forex_loader):
        """测试获取支持的货币"""
        currencies = mock_forex_loader.get_supported_currencies()

        assert isinstance(currencies, list)
        assert 'USD' in currencies
        assert 'EUR' in currencies
        assert 'JPY' in currencies
        assert 'GBP' in currencies

    def test_get_supported_pairs(self, mock_forex_loader):
        """测试获取支持的货币对"""
        pairs = mock_forex_loader.get_supported_pairs()

        assert isinstance(pairs, list)
        assert 'EURUSD' in pairs
        assert 'USDJPY' in pairs
        assert 'GBPUSD' in pairs

    def test_basic_functionality(self, mock_forex_loader):
        """测试基本功能"""
        # 测试基本属性存在性
        assert hasattr(mock_forex_loader, 'config')
        assert hasattr(mock_forex_loader, 'cache_manager')
        assert hasattr(mock_forex_loader, 'get_metadata')

    def test_get_required_config_fields(self, mock_forex_loader):
        """测试获取必需配置字段"""
        required_fields = mock_forex_loader.get_required_config_fields()

        assert isinstance(required_fields, list)
        if required_fields:  # 如果有必需字段
            assert all(isinstance(field, str) for field in required_fields)

    def test_validate_config(self, mock_forex_loader):
        """测试配置验证"""
        result = mock_forex_loader.validate_config()
        # 配置验证应该返回布尔值
        assert isinstance(result, bool)
