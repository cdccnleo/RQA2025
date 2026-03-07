#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试加密货币数据加载器

测试目标：提升crypto_loader.py的覆盖率到80%+，确保100%测试通过率
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
import asyncio
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path

from src.data.loader.crypto_loader import (
    CoinGeckoLoader,
    BinanceLoader,
    CryptoDataLoader,
    CryptoData,
    CryptoMarketData
)


class TestCryptoData:
    """测试CryptoData数据类"""

    def test_crypto_data_creation(self):
        """测试创建CryptoData实例"""
        data = CryptoData(
            symbol="BTC",
            name="Bitcoin",
            price=50000.0,
            volume_24h=1000000.0,
            market_cap=1000000000.0,
            price_change_24h=1000.0,
            price_change_percentage_24h=2.0,
            high_24h=51000.0,
            low_24h=49000.0,
            timestamp=datetime.now(),
            source="coingecko"
        )
        assert data.symbol == "BTC"
        assert data.price == 50000.0
        assert isinstance(data.timestamp, datetime)


class TestCryptoMarketData:
    """测试CryptoMarketData数据类"""

    def test_crypto_market_data_creation(self):
        """测试创建CryptoMarketData实例"""
        data = CryptoMarketData(
            total_market_cap=2000000000.0,
            total_volume_24h=50000000.0,
            market_cap_percentage={"BTC": 50.0, "ETH": 30.0},
            market_cap_change_24h=100000.0,
            timestamp=datetime.now()
        )
        assert data.total_market_cap == 2000000000.0
        assert isinstance(data.market_cap_percentage, dict)


class TestCoinGeckoLoader:
    """测试CoinGecko数据加载器"""

    @pytest.fixture
    def coingecko_loader(self, tmp_path):
        """创建CoinGecko数据加载器实例"""
        with patch('src.data.loader.crypto_loader.CacheManager'):
            loader = CoinGeckoLoader(api_key="test_key")
            loader.cache_manager = AsyncMock()
            loader.cache_manager.get = AsyncMock(return_value=None)
            loader.cache_manager.set = AsyncMock()
            return loader

    def test_coingecko_loader_initialization(self, tmp_path):
        """测试CoinGecko加载器初始化"""
        with patch('src.data.loader.crypto_loader.CacheManager'):
            loader = CoinGeckoLoader(api_key="test_key")
            assert loader.api_key == "test_key"
            assert loader.base_url == "https://api.coingecko.com / api / v3"
            assert loader.cache_manager is not None

    def test_coingecko_loader_initialization_no_api_key(self):
        """测试无API密钥的初始化"""
        with patch('src.data.loader.crypto_loader.CacheManager'):
            loader = CoinGeckoLoader()
            assert loader.api_key is None

    def test_coingecko_loader_get_required_config_fields(self, coingecko_loader):
        """测试获取必需配置字段"""
        fields = coingecko_loader.get_required_config_fields()
        assert isinstance(fields, list)
        assert 'cache_dir' in fields
        assert 'max_retries' in fields

    def test_coingecko_loader_validate_config(self, coingecko_loader):
        """测试验证配置"""
        result = coingecko_loader.validate_config()
        assert isinstance(result, bool)

    def test_coingecko_loader_get_metadata(self, coingecko_loader):
        """测试获取元数据"""
        metadata = coingecko_loader.get_metadata()
        assert isinstance(metadata, dict)
        assert metadata["loader_type"] == "coingecko"
        assert metadata["version"] == "1.0.0"

    def test_coingecko_loader_load_not_implemented(self, coingecko_loader):
        """测试load方法抛出NotImplementedError"""
        with pytest.raises(NotImplementedError, match="Use load_data"):
            coingecko_loader.load("2024-01-01", "2024-01-31", "1d")

    @pytest.mark.asyncio
    async def test_coingecko_loader_async_context_manager(self, coingecko_loader):
        """测试异步上下文管理器"""
        async with coingecko_loader as loader:
            assert loader.session is not None
        assert coingecko_loader.session is None or coingecko_loader.session.closed

    @pytest.mark.asyncio
    async def test_coingecko_loader_get_top_coins_from_cache(self, coingecko_loader):
        """测试从缓存获取顶级币种"""
        cached_data = [
            {
                'symbol': 'BTC',
                'name': 'Bitcoin',
                'price': 50000.0,
                'volume_24h': 1000000.0,
                'market_cap': 1000000000.0,
                'price_change_24h': 1000.0,
                'price_change_percentage_24h': 2.0,
                'high_24h': 51000.0,
                'low_24h': 49000.0,
                'timestamp': datetime.now(),
                'source': 'coingecko'
            }
        ]
        coingecko_loader.cache_manager.get = AsyncMock(return_value=cached_data)

        result = await coingecko_loader.get_top_coins(limit=10)
        assert result is not None
        assert len(result) > 0
        assert isinstance(result[0], CryptoData)

    @pytest.mark.asyncio
    async def test_coingecko_loader_get_top_coins_new_data(self, coingecko_loader):
        """测试获取新的顶级币种数据"""
        coingecko_loader.cache_manager.get = AsyncMock(return_value=None)
        coingecko_loader.session = AsyncMock()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[{
            'symbol': 'btc',
            'name': 'Bitcoin',
            'current_price': 50000.0,
            'total_volume': 1000000.0,
            'market_cap': 1000000000.0,
            'price_change_24h': 1000.0,
            'price_change_percentage_24h': 2.0,
            'high_24h': 51000.0,
            'low_24h': 49000.0,
            'last_updated': datetime.now().timestamp() * 1000
        }])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        coingecko_loader.session.get = Mock(return_value=mock_response)

        result = await coingecko_loader.get_top_coins(limit=10)
        assert result is not None
        assert isinstance(result, list)
        coingecko_loader.cache_manager.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_coingecko_loader_get_top_coins_api_error(self, coingecko_loader):
        """测试获取顶级币种API错误"""
        coingecko_loader.cache_manager.get = AsyncMock(return_value=None)
        coingecko_loader.session = AsyncMock()
        
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        coingecko_loader.session.get = Mock(return_value=mock_response)

        result = await coingecko_loader.get_top_coins(limit=10)
        assert result == []

    @pytest.mark.asyncio
    async def test_coingecko_loader_get_coin_detail_from_cache(self, coingecko_loader):
        """测试从缓存获取币种详情"""
        cached_data = {'id': 'bitcoin', 'name': 'Bitcoin'}
        coingecko_loader.cache_manager.get = AsyncMock(return_value=cached_data)

        result = await coingecko_loader.get_coin_detail("bitcoin")
        assert result is not None
        assert result == cached_data

    @pytest.mark.asyncio
    async def test_coingecko_loader_get_coin_detail_new_data(self, coingecko_loader):
        """测试获取新的币种详情"""
        coingecko_loader.cache_manager.get = AsyncMock(return_value=None)
        coingecko_loader.session = AsyncMock()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'id': 'bitcoin', 'name': 'Bitcoin'})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        coingecko_loader.session.get = Mock(return_value=mock_response)

        result = await coingecko_loader.get_coin_detail("bitcoin")
        assert result is not None
        coingecko_loader.cache_manager.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_coingecko_loader_get_market_data_from_cache(self, coingecko_loader):
        """测试从缓存获取市场数据"""
        cached_data = {
            'total_market_cap': 2000000000.0,
            'total_volume_24h': 50000000.0,
            'market_cap_percentage': {'BTC': 50.0},
            'market_cap_change_24h': 100000.0,
            'timestamp': datetime.now()
        }
        coingecko_loader.cache_manager.get = AsyncMock(return_value=cached_data)

        result = await coingecko_loader.get_market_data()
        assert result is not None
        assert isinstance(result, CryptoMarketData)

    @pytest.mark.asyncio
    async def test_coingecko_loader_get_market_data_new_data(self, coingecko_loader):
        """测试获取新的市场数据"""
        coingecko_loader.cache_manager.get = AsyncMock(return_value=None)
        coingecko_loader.session = AsyncMock()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'data': {
                'total_market_cap': {'usd': 2000000000.0},
                'total_volume': {'usd': 50000000.0},
                'market_cap_percentage': {'BTC': 50.0},
                'market_cap_change_percentage_24h_usd': 5.0
            }
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        coingecko_loader.session.get = Mock(return_value=mock_response)

        result = await coingecko_loader.get_market_data()
        assert result is not None
        assert isinstance(result, CryptoMarketData)
        coingecko_loader.cache_manager.set.assert_called_once()


class TestBinanceLoader:
    """测试Binance数据加载器"""

    @pytest.fixture
    def binance_loader(self, tmp_path):
        """创建Binance数据加载器实例"""
        with patch('src.data.loader.crypto_loader.CacheManager'):
            loader = BinanceLoader(api_key="test_key", api_secret="test_secret")
            loader.cache_manager = AsyncMock()
            loader.cache_manager.get = AsyncMock(return_value=None)
            loader.cache_manager.set = AsyncMock()
            return loader

    def test_binance_loader_initialization(self, tmp_path):
        """测试Binance加载器初始化"""
        with patch('src.data.loader.crypto_loader.CacheManager'):
            loader = BinanceLoader(api_key="test_key", api_secret="test_secret")
            assert loader.api_key == "test_key"
            assert loader.api_secret == "test_secret"
            assert loader.base_url == "https://api.binance.com / api / v3"

    def test_binance_loader_get_required_config_fields(self, binance_loader):
        """测试获取必需配置字段"""
        fields = binance_loader.get_required_config_fields()
        assert isinstance(fields, list)
        assert 'cache_dir' in fields

    def test_binance_loader_validate_config(self, binance_loader):
        """测试验证配置"""
        result = binance_loader.validate_config()
        assert isinstance(result, bool)

    def test_binance_loader_get_metadata(self, binance_loader):
        """测试获取元数据"""
        metadata = binance_loader.get_metadata()
        assert isinstance(metadata, dict)
        assert metadata["loader_type"] == "binance"

    def test_binance_loader_load_not_implemented(self, binance_loader):
        """测试load方法抛出NotImplementedError"""
        with pytest.raises(NotImplementedError, match="Use load_data"):
            binance_loader.load("2024-01-01", "2024-01-31", "1d")

    @pytest.mark.asyncio
    async def test_binance_loader_async_context_manager(self, binance_loader):
        """测试异步上下文管理器"""
        async with binance_loader as loader:
            assert loader.session is not None
        assert binance_loader.session is None or binance_loader.session.closed

    @pytest.mark.asyncio
    async def test_binance_loader_get_ticker_24hr_from_cache(self, binance_loader):
        """测试从缓存获取24小时价格统计"""
        cached_data = {'symbol': 'BTCUSDT', 'lastPrice': '50000.0'}
        binance_loader.cache_manager.get = AsyncMock(return_value=cached_data)

        result = await binance_loader.get_ticker_24hr("BTCUSDT")
        assert result is not None
        assert result == cached_data

    @pytest.mark.asyncio
    async def test_binance_loader_get_ticker_24hr_new_data(self, binance_loader):
        """测试获取新的24小时价格统计"""
        binance_loader.cache_manager.get = AsyncMock(return_value=None)
        binance_loader.session = AsyncMock()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'symbol': 'BTCUSDT', 'lastPrice': '50000.0'})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        binance_loader.session.get = Mock(return_value=mock_response)

        result = await binance_loader.get_ticker_24hr("BTCUSDT")
        assert result is not None
        binance_loader.cache_manager.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_binance_loader_get_klines_from_cache(self, binance_loader):
        """测试从缓存获取K线数据"""
        cached_data = [{'open_time': datetime.now(), 'open': 50000.0}]
        binance_loader.cache_manager.get = AsyncMock(return_value=cached_data)

        result = await binance_loader.get_klines("BTCUSDT", "1h", 100)
        assert result is not None
        assert result == cached_data

    @pytest.mark.asyncio
    async def test_binance_loader_get_klines_new_data(self, binance_loader):
        """测试获取新的K线数据"""
        binance_loader.cache_manager.get = AsyncMock(return_value=None)
        binance_loader.session = AsyncMock()
        
        # Binance K线数据格式
        mock_kline = [
            1609459200000,  # open_time
            "50000.0",      # open
            "51000.0",      # high
            "49000.0",      # low
            "50500.0",      # close
            "1000.0",       # volume
            1609545600000,  # close_time
            "50000000.0",   # quote_volume
            100,            # trades
            "500.0",        # taker_buy_base
            "25000000.0"    # taker_buy_quote
        ]
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[mock_kline])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        binance_loader.session.get = Mock(return_value=mock_response)

        result = await binance_loader.get_klines("BTCUSDT", "1h", 100)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0
        assert 'open_time' in result[0]
        binance_loader.cache_manager.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_binance_loader_get_exchange_info_from_cache(self, binance_loader):
        """测试从缓存获取交易所信息"""
        cached_data = {'timezone': 'UTC', 'symbols': []}
        binance_loader.cache_manager.get = AsyncMock(return_value=cached_data)

        result = await binance_loader.get_exchange_info()
        assert result is not None
        assert result == cached_data

    @pytest.mark.asyncio
    async def test_binance_loader_get_exchange_info_new_data(self, binance_loader):
        """测试获取新的交易所信息"""
        binance_loader.cache_manager.get = AsyncMock(return_value=None)
        binance_loader.session = AsyncMock()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'timezone': 'UTC', 'symbols': []})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        binance_loader.session.get = Mock(return_value=mock_response)

        result = await binance_loader.get_exchange_info()
        assert result is not None
        binance_loader.cache_manager.set.assert_called_once()


class TestCryptoDataLoader:
    """测试统一加密货币数据加载器"""

    @pytest.fixture
    def crypto_loader(self, tmp_path):
        """创建统一加密货币数据加载器实例"""
        with patch('src.data.loader.crypto_loader.CacheManager'):
            loader = CryptoDataLoader(
                save_path=str(tmp_path),
                max_retries=3,
                cache_days=1,
                timeout=30
            )
            loader.cache_manager = Mock()
            loader.cache_manager.get = Mock(return_value=None)
            loader.cache_manager.set = Mock()
            loader.coingecko_loader = Mock()
            loader.binance_loader = Mock()
            return loader

    def test_crypto_loader_initialization(self, tmp_path):
        """测试加密货币加载器初始化"""
        with patch('src.data.loader.crypto_loader.CacheManager'):
            loader = CryptoDataLoader(save_path=str(tmp_path))
            assert loader.max_retries == 3
            assert loader.cache_days == 1
            assert loader.timeout == 30
            assert loader.coingecko_loader is None
            assert loader.binance_loader is None

    def test_crypto_loader_get_required_config_fields(self, crypto_loader):
        """测试获取必需配置字段"""
        fields = crypto_loader.get_required_config_fields()
        assert isinstance(fields, list)
        assert 'cache_dir' in fields

    def test_crypto_loader_validate_config(self, crypto_loader):
        """测试验证配置"""
        with patch.object(crypto_loader, '_validate_config', return_value=True, create=True):
            result = crypto_loader.validate_config()
            assert isinstance(result, bool)

    def test_crypto_loader_get_metadata(self, crypto_loader):
        """测试获取元数据"""
        metadata = crypto_loader.get_metadata()
        assert isinstance(metadata, dict)
        assert metadata["loader_type"] == "crypto"
        assert "supported_sources" in metadata

    @pytest.mark.asyncio
    async def test_crypto_loader_initialize(self, crypto_loader):
        """测试初始化"""
        with patch('src.data.loader.crypto_loader.CoinGeckoLoader') as mock_coingecko:
            with patch('src.data.loader.crypto_loader.BinanceLoader') as mock_binance:
                mock_coingecko.return_value = Mock()
                mock_binance.return_value = Mock()
                await crypto_loader.initialize()
                assert crypto_loader.coingecko_loader is not None
                assert crypto_loader.binance_loader is not None

    @pytest.mark.asyncio
    async def test_crypto_loader_get_crypto_data_coingecko(self, crypto_loader):
        """测试从CoinGecko获取加密货币数据"""
        mock_crypto_data = [
            CryptoData(
                symbol="BTC",
                name="Bitcoin",
                price=50000.0,
                volume_24h=1000000.0,
                market_cap=1000000000.0,
                price_change_24h=1000.0,
                price_change_percentage_24h=2.0,
                high_24h=51000.0,
                low_24h=49000.0,
                timestamp=datetime.now(),
                source="coingecko"
            )
        ]
        
        async def mock_get_top_coins(limit):
            return mock_crypto_data
        
        crypto_loader.coingecko_loader = AsyncMock()
        crypto_loader.coingecko_loader.__aenter__ = AsyncMock(return_value=crypto_loader.coingecko_loader)
        crypto_loader.coingecko_loader.__aexit__ = AsyncMock(return_value=None)
        crypto_loader.coingecko_loader.get_top_coins = AsyncMock(return_value=mock_crypto_data)

        result = await crypto_loader.get_crypto_data(symbols=["BTC"], source="coingecko")
        assert result is not None
        assert len(result) > 0
        assert result[0].symbol == "BTC"

    @pytest.mark.asyncio
    async def test_crypto_loader_get_crypto_data_binance(self, crypto_loader):
        """测试从Binance获取加密货币数据"""
        mock_ticker = {
            'lastPrice': '50000.0',
            'volume': '1000000.0',
            'priceChange': '1000.0',
            'priceChangePercent': '2.0',
            'highPrice': '51000.0',
            'lowPrice': '49000.0'
        }
        
        crypto_loader.binance_loader = AsyncMock()
        crypto_loader.binance_loader.__aenter__ = AsyncMock(return_value=crypto_loader.binance_loader)
        crypto_loader.binance_loader.__aexit__ = AsyncMock(return_value=None)
        crypto_loader.binance_loader.get_ticker_24hr = AsyncMock(return_value=mock_ticker)

        result = await crypto_loader.get_crypto_data(symbols=["BTC"], source="binance")
        assert result is not None
        assert len(result) > 0
        assert result[0].symbol == "BTC"

    @pytest.mark.asyncio
    async def test_crypto_loader_get_crypto_data_unsupported_source(self, crypto_loader):
        """测试不支持的数据源"""
        result = await crypto_loader.get_crypto_data(symbols=["BTC"], source="unsupported")
        assert result == []

    @pytest.mark.asyncio
    async def test_crypto_loader_get_market_overview(self, crypto_loader):
        """测试获取市场概览"""
        mock_market_data = CryptoMarketData(
            total_market_cap=2000000000.0,
            total_volume_24h=50000000.0,
            market_cap_percentage={"BTC": 50.0},
            market_cap_change_24h=100000.0,
            timestamp=datetime.now()
        )
        
        crypto_loader.coingecko_loader = AsyncMock()
        crypto_loader.coingecko_loader.__aenter__ = AsyncMock(return_value=crypto_loader.coingecko_loader)
        crypto_loader.coingecko_loader.__aexit__ = AsyncMock(return_value=None)
        crypto_loader.coingecko_loader.get_market_data = AsyncMock(return_value=mock_market_data)

        result = await crypto_loader.get_market_overview()
        assert result is not None
        assert isinstance(result, CryptoMarketData)

    @pytest.mark.asyncio
    async def test_crypto_loader_get_historical_data(self, crypto_loader):
        """测试获取历史数据"""
        mock_klines = [{'open_time': datetime.now(), 'open': 50000.0}]
        
        crypto_loader.binance_loader = AsyncMock()
        crypto_loader.binance_loader.__aenter__ = AsyncMock(return_value=crypto_loader.binance_loader)
        crypto_loader.binance_loader.__aexit__ = AsyncMock(return_value=None)
        crypto_loader.binance_loader.get_klines = AsyncMock(return_value=mock_klines)

        result = await crypto_loader.get_historical_data("BTC", days=30)
        assert result is not None
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_crypto_loader_validate_data_valid(self, crypto_loader):
        """测试验证有效数据"""
        valid_data = [
            CryptoData(
                symbol="BTC",
                name="Bitcoin",
                price=50000.0,
                volume_24h=1000000.0,
                market_cap=1000000000.0,
                price_change_24h=1000.0,
                price_change_percentage_24h=2.0,
                high_24h=51000.0,
                low_24h=49000.0,
                timestamp=datetime.now(),
                source="coingecko"
            )
        ]
        
        result = await crypto_loader.validate_data(valid_data)
        assert result['valid'] is True
        assert result['valid_records'] == 1
        assert result['invalid_records'] == 0

    @pytest.mark.asyncio
    async def test_crypto_loader_validate_data_invalid_price(self, crypto_loader):
        """测试验证无效价格数据"""
        invalid_data = [
            CryptoData(
                symbol="BTC",
                name="Bitcoin",
                price=-100.0,  # 无效：负价格
                volume_24h=1000000.0,
                market_cap=1000000000.0,
                price_change_24h=1000.0,
                price_change_percentage_24h=2.0,
                high_24h=51000.0,
                low_24h=49000.0,
                timestamp=datetime.now(),
                source="coingecko"
            )
        ]
        
        result = await crypto_loader.validate_data(invalid_data)
        assert result['valid'] is False
        assert result['invalid_records'] > 0

    @pytest.mark.asyncio
    async def test_crypto_loader_validate_data_invalid_volume(self, crypto_loader):
        """测试验证无效交易量数据"""
        invalid_data = [
            CryptoData(
                symbol="BTC",
                name="Bitcoin",
                price=50000.0,
                volume_24h=-1000.0,  # 无效：负交易量
                market_cap=1000000000.0,
                price_change_24h=1000.0,
                price_change_percentage_24h=2.0,
                high_24h=51000.0,
                low_24h=49000.0,
                timestamp=datetime.now(),
                source="coingecko"
            )
        ]
        
        result = await crypto_loader.validate_data(invalid_data)
        assert result['valid'] is False
        assert result['invalid_records'] > 0

    @pytest.mark.asyncio
    async def test_crypto_loader_validate_data_price_too_high(self, crypto_loader):
        """测试验证价格异常高"""
        invalid_data = [
            CryptoData(
                symbol="BTC",
                name="Bitcoin",
                price=2000000.0,  # 无效：超过100万美元
                volume_24h=1000000.0,
                market_cap=1000000000.0,
                price_change_24h=1000.0,
                price_change_percentage_24h=2.0,
                high_24h=51000.0,
                low_24h=49000.0,
                timestamp=datetime.now(),
                source="coingecko"
            )
        ]
        
        result = await crypto_loader.validate_data(invalid_data)
        assert result['valid'] is False
        assert result['invalid_records'] > 0

