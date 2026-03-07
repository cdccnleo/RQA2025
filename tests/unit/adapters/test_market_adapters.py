# -*- coding: utf-8 -*-
"""
市场适配器单元测试
测试各种市场适配器的核心功能
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import pandas as pd

from src.adapters.market.market_adapters import (
    MarketType,
    AssetClass,
    MarketAdapter,
    AStockAdapter,
    HStockAdapter,
    USStockAdapter,
    FuturesAdapter,
    CryptoAdapter,
    MarketAdapterManager
)


class TestMarketType:
    """市场类型枚举测试"""

    def test_market_type_values(self):
        """测试市场类型枚举值"""
        assert MarketType.A_STOCK.value == "a_stock"
        assert MarketType.H_STOCK.value == "h_stock"
        assert MarketType.US_STOCK.value == "us_stock"
        assert MarketType.CRYPTO.value == "crypto"


class TestAssetClass:
    """资产类别枚举测试"""

    def test_asset_class_values(self):
        """测试资产类别枚举值"""
        assert AssetClass.EQUITY.value == "equity"
        assert AssetClass.DERIVATIVE.value == "derivative"
        assert AssetClass.CURRENCY.value == "currency"


class TestMarketAdapter:
    """市场适配器抽象基类测试"""

    def test_market_adapter_initialization(self):
        """测试市场适配器初始化"""
        config = {"api_key": "test_key"}
        adapter = MarketAdapter(MarketType.A_STOCK, config)

        assert adapter.market_type == MarketType.A_STOCK
        assert adapter.config == config
        assert hasattr(adapter, 'logger')

    def test_market_adapter_abstract_methods(self):
        """测试抽象方法"""
        adapter = MarketAdapter(MarketType.A_STOCK)

        # 抽象方法应该抛出NotImplementedError
        with pytest.raises(NotImplementedError):
            adapter.connect()

        with pytest.raises(NotImplementedError):
            adapter.disconnect()

        with pytest.raises(NotImplementedError):
            adapter.get_market_data("000001")

        with pytest.raises(NotImplementedError):
            adapter.get_quotes(["000001"])


class TestAStockAdapter:
    """A股适配器测试"""

    @patch('requests.get')
    def test_a_stock_adapter_initialization(self, mock_get):
        """测试A股适配器初始化"""
        config = {"base_url": "http://test.com"}
        adapter = AStockAdapter(config)

        assert adapter.market_type == MarketType.A_STOCK
        assert adapter.base_url in ["http://test.com", "http://api.wmcloud.com"]

    @patch('requests.get')
    def test_a_stock_adapter_connect(self, mock_get):
        """测试A股适配器连接"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        adapter = AStockAdapter()
        result = adapter.connect()

        assert result is True
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_a_stock_adapter_get_market_data(self, mock_get):
        """测试A股适配器获取市场数据"""
        # 模拟API响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "symbol": "000001",
                    "price": 10.5,
                    "volume": 1000000,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
        mock_get.return_value = mock_response

        adapter = AStockAdapter()
        data = adapter.get_market_data("000001", days=1)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert "symbol" in data.columns


class TestHStockAdapter:
    """港股适配器测试"""

    @patch('requests.get')
    def test_h_stock_adapter_initialization(self, mock_get):
        """测试港股适配器初始化"""
        config = {"api_key": "test_key"}
        adapter = HStockAdapter(config)

        assert adapter.market_type == MarketType.H_STOCK
        assert adapter.api_key == "test_key"

    @patch('requests.get')
    def test_h_stock_adapter_get_quotes(self, mock_get):
        """测试港股适配器获取报价"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "quotes": [
                {
                    "symbol": "00001.HK",
                    "price": 50.0,
                    "change": 1.5
                }
            ]
        }
        mock_get.return_value = mock_response

        adapter = HStockAdapter({"api_key": "test"})
        quotes = adapter.get_quotes(["00001.HK"])

        assert isinstance(quotes, dict)
        assert "00001.HK" in quotes


class TestUSStockAdapter:
    """美股适配器测试"""

    @patch('requests.get')
    def test_us_stock_adapter_initialization(self, mock_get):
        """测试美股适配器初始化"""
        adapter = USStockAdapter()

        assert adapter.market_type == MarketType.US_STOCK

    @patch('requests.get')
    def test_us_stock_adapter_get_market_data(self, mock_get):
        """测试美股适配器获取市场数据"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                "2024-01-01": {
                    "1. open": "100.00",
                    "2. high": "105.00",
                    "3. low": "95.00",
                    "4. close": "102.00",
                    "5. volume": "1000000"
                }
            }
        }
        mock_get.return_value = mock_response

        adapter = USStockAdapter({"api_key": "test"})
        data = adapter.get_market_data("AAPL", days=1)

        assert isinstance(data, pd.DataFrame)


class TestCryptoAdapter:
    """数字货币适配器测试"""

    @patch('requests.get')
    def test_crypto_adapter_initialization(self, mock_get):
        """测试数字货币适配器初始化"""
        adapter = CryptoAdapter()

        assert adapter.market_type == MarketType.CRYPTO

    @patch('requests.get')
    def test_crypto_adapter_get_market_data(self, mock_get):
        """测试数字货币适配器获取市场数据"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "symbol": "BTCUSDT",
                "price": "50000.00",
                "volume": "100.0",
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
        ]
        mock_get.return_value = mock_response

        adapter = CryptoAdapter()
        data = adapter.get_market_data("BTCUSDT", days=1)

        assert isinstance(data, pd.DataFrame)


class TestMarketAdapterManager:
    """市场适配器管理器测试"""

    def test_market_adapter_manager_initialization(self):
        """测试市场适配器管理器初始化"""
        manager = MarketAdapterManager()

        assert hasattr(manager, 'adapters')
        assert hasattr(manager, 'logger')

    def test_register_adapter(self):
        """测试注册适配器"""
        manager = MarketAdapterManager()

        # 创建模拟适配器
        mock_adapter = Mock()
        mock_adapter.market_type = MarketType.A_STOCK

        manager.register_adapter(MarketType.A_STOCK, mock_adapter)

        assert MarketType.A_STOCK in manager.adapters
        assert manager.adapters[MarketType.A_STOCK] == mock_adapter

    def test_get_adapter(self):
        """测试获取适配器"""
        manager = MarketAdapterManager()

        # 注册适配器
        mock_adapter = Mock()
        mock_adapter.market_type = MarketType.A_STOCK
        manager.register_adapter(MarketType.A_STOCK, mock_adapter)

        # 获取适配器
        adapter = manager.get_adapter(MarketType.A_STOCK)
        assert adapter == mock_adapter

        # 获取不存在的适配器
        with pytest.raises(ValueError):
            manager.get_adapter(MarketType.CRYPTO)

    def test_list_available_markets(self):
        """测试列出可用市场"""
        manager = MarketAdapterManager()

        markets = manager.list_available_markets()
        assert isinstance(markets, list)
        assert MarketType.A_STOCK in markets