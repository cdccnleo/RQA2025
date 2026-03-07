"""
crypto_loader.py 边界测试补充
目标：将覆盖率从 67% 提升到 80%+
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
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pandas as pd

from src.data.loader.crypto_loader import (
    CoinGeckoLoader,
    BinanceLoader,
    CryptoDataLoader,
    CryptoData,
    CryptoMarketData
)


@pytest.fixture
def coingecko_loader():
    """创建 CoinGecko 加载器实例"""
    with patch('src.data.loader.crypto_loader.CacheManager'):
        loader = CoinGeckoLoader(api_key="test_key")
        loader.cache_manager = AsyncMock()
        loader.cache_manager.get = AsyncMock(return_value=None)
        loader.cache_manager.set = AsyncMock()
        return loader


@pytest.fixture
def binance_loader():
    """创建 Binance 加载器实例"""
    with patch('src.data.loader.crypto_loader.CacheManager'):
        loader = BinanceLoader(api_key="test_key", api_secret="test_secret")
        loader.cache_manager = AsyncMock()
        loader.cache_manager.get = AsyncMock(return_value=None)
        loader.cache_manager.set = AsyncMock()
        return loader


@pytest.fixture
def crypto_loader():
    """创建 CryptoDataLoader 实例"""
    with patch('src.data.loader.crypto_loader.CacheManager'):
        loader = CryptoDataLoader()
        loader.cache_manager = Mock()
        loader.cache_manager.get = Mock(return_value=None)
        loader.cache_manager.set = Mock()
        return loader


@pytest.mark.asyncio
async def test_coingecko_loader_get_top_coins_exception(coingecko_loader, monkeypatch):
    """测试 CoinGecko 加载器（get_top_coins，异常处理，覆盖 184-186 行）"""
    # Mock session.get 抛出异常
    coingecko_loader.session = AsyncMock()
    coingecko_loader.session.get = Mock(side_effect=Exception("Network error"))
    
    result = await coingecko_loader.get_top_coins(limit=10)
    
    assert result == []


@pytest.mark.asyncio
async def test_coingecko_loader_get_coin_detail_error(coingecko_loader, monkeypatch):
    """测试 CoinGecko 加载器（get_coin_detail，错误状态码，覆盖 218-219 行）"""
    # Mock session.get 返回错误状态码
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    
    coingecko_loader.session = AsyncMock()
    coingecko_loader.session.get = Mock(return_value=mock_response)
    
    result = await coingecko_loader.get_coin_detail("bitcoin")
    
    assert result is None


@pytest.mark.asyncio
async def test_coingecko_loader_get_coin_detail_exception(coingecko_loader, monkeypatch):
    """测试 CoinGecko 加载器（get_coin_detail，异常处理，覆盖 221-223 行）"""
    # Mock session.get 抛出异常
    coingecko_loader.session = AsyncMock()
    coingecko_loader.session.get = Mock(side_effect=Exception("Network error"))
    
    result = await coingecko_loader.get_coin_detail("bitcoin")
    
    assert result is None


@pytest.mark.asyncio
async def test_coingecko_loader_get_market_data_error(coingecko_loader, monkeypatch):
    """测试 CoinGecko 加载器（get_market_data，错误状态码，覆盖 256-257 行）"""
    # Mock session.get 返回错误状态码
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    
    coingecko_loader.session = AsyncMock()
    coingecko_loader.session.get = Mock(return_value=mock_response)
    
    result = await coingecko_loader.get_market_data()
    
    assert result is None


@pytest.mark.asyncio
async def test_coingecko_loader_get_market_data_exception(coingecko_loader, monkeypatch):
    """测试 CoinGecko 加载器（get_market_data，异常处理，覆盖 259-261 行）"""
    # Mock session.get 抛出异常
    coingecko_loader.session = AsyncMock()
    coingecko_loader.session.get = Mock(side_effect=Exception("Network error"))
    
    result = await coingecko_loader.get_market_data()
    
    assert result is None


@pytest.mark.asyncio
async def test_binance_loader_get_ticker_24hr_error(binance_loader, monkeypatch):
    """测试 Binance 加载器（get_ticker_24hr，错误状态码，覆盖 362-363 行）"""
    # Mock session.get 返回错误状态码
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    
    binance_loader.session = AsyncMock()
    binance_loader.session.get = Mock(return_value=mock_response)
    
    result = await binance_loader.get_ticker_24hr("BTCUSDT")
    
    assert result is None


@pytest.mark.asyncio
async def test_binance_loader_get_ticker_24hr_exception(binance_loader, monkeypatch):
    """测试 Binance 加载器（get_ticker_24hr，异常处理，覆盖 365-367 行）"""
    # Mock session.get 抛出异常
    binance_loader.session = AsyncMock()
    binance_loader.session.get = Mock(side_effect=Exception("Network error"))
    
    result = await binance_loader.get_ticker_24hr("BTCUSDT")
    
    assert result is None


@pytest.mark.asyncio
async def test_binance_loader_get_klines_error(binance_loader, monkeypatch):
    """测试 Binance 加载器（get_klines，错误状态码，覆盖 413-414 行）"""
    # Mock session.get 返回错误状态码
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    
    binance_loader.session = AsyncMock()
    binance_loader.session.get = Mock(return_value=mock_response)
    
    result = await binance_loader.get_klines("BTCUSDT", "1h", 100)
    
    assert result == []


@pytest.mark.asyncio
async def test_binance_loader_get_klines_exception(binance_loader, monkeypatch):
    """测试 Binance 加载器（get_klines，异常处理，覆盖 416-418 行）"""
    # Mock session.get 抛出异常
    binance_loader.session = AsyncMock()
    binance_loader.session.get = Mock(side_effect=Exception("Network error"))
    
    result = await binance_loader.get_klines("BTCUSDT", "1h", 100)
    
    assert result == []


@pytest.mark.asyncio
async def test_binance_loader_get_exchange_info_error(binance_loader, monkeypatch):
    """测试 Binance 加载器（get_exchange_info，错误状态码，覆盖 442-443 行）"""
    # Mock session.get 返回错误状态码
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    
    binance_loader.session = AsyncMock()
    binance_loader.session.get = Mock(return_value=mock_response)
    
    result = await binance_loader.get_exchange_info()
    
    assert result is None


@pytest.mark.asyncio
async def test_binance_loader_get_exchange_info_exception(binance_loader, monkeypatch):
    """测试 Binance 加载器（get_exchange_info，异常处理，覆盖 445-447 行）"""
    # Mock session.get 抛出异常
    binance_loader.session = AsyncMock()
    binance_loader.session.get = Mock(side_effect=Exception("Network error"))
    
    result = await binance_loader.get_exchange_info()
    
    assert result is None


def test_crypto_loader_init_with_string_config(tmp_path):
    """测试 CryptoDataLoader（__init__，config 为字符串，覆盖 464 行）"""
    with patch('src.data.loader.crypto_loader.CacheManager'):
        loader = CryptoDataLoader(config=str(tmp_path))
        assert loader is not None


def test_crypto_loader_init_with_path_config(tmp_path):
    """测试 CryptoDataLoader（__init__，config 为 Path，覆盖 464 行）"""
    with patch('src.data.loader.crypto_loader.CacheManager'):
        loader = CryptoDataLoader(config=tmp_path)
        assert loader is not None


@pytest.mark.asyncio
async def test_crypto_loader_get_crypto_data_no_symbols(crypto_loader, monkeypatch):
    """测试 CryptoDataLoader（get_crypto_data，symbols 为 None，覆盖 516 行）"""
    # Mock _get_coingecko_data
    async def mock_get_coingecko_data(symbols):
        return []
    
    monkeypatch.setattr(crypto_loader, '_get_coingecko_data', mock_get_coingecko_data)
    
    result = await crypto_loader.get_crypto_data(symbols=None, source="coingecko")
    
    assert result == []


@pytest.mark.asyncio
async def test_crypto_loader_validate_data_exception(crypto_loader, monkeypatch):
    """测试 CryptoDataLoader（validate_data，异常处理，覆盖 605-607 行）"""
    # 创建一个会导致异常的数据
    # 通过创建一个属性访问会抛出异常的对象
    class InvalidCrypto:
        def __init__(self):
            self.symbol = "BTC"
            self.name = "Bitcoin"
        
        @property
        def price(self):
            raise Exception("Cannot access price")
    
    invalid_crypto = InvalidCrypto()
    
    # 直接测试异常处理路径
    result = await crypto_loader.validate_data([invalid_crypto])
    # 由于异常会被捕获，应该返回验证结果
    assert result is not None
    assert 'errors' in result
    assert result['invalid_records'] > 0


def test_crypto_loader_load_batch_exception(crypto_loader, monkeypatch):
    """测试 CryptoDataLoader（load_batch，异常处理，覆盖 690-692 行）"""
    # Mock load 抛出异常
    def mock_load(*args, **kwargs):
        raise Exception("Load error")
    
    monkeypatch.setattr(crypto_loader, 'load', mock_load)
    
    result = crypto_loader.load_batch(
        symbols=["BTC", "ETH"],
        start_date="2020-01-01",
        end_date="2020-01-31",
        source="coingecko"
    )
    
    # 异常时应该返回 None
    assert "BTC" in result
    assert result["BTC"] is None


@pytest.mark.asyncio
async def test_crypto_loader_load_data_exception(crypto_loader, monkeypatch):
    """测试 CryptoDataLoader（load_data，异常处理，覆盖 742-744 行）"""
    # Mock initialize 抛出异常
    async def mock_initialize():
        raise Exception("Init error")
    
    monkeypatch.setattr(crypto_loader, 'initialize', mock_initialize)
    
    result = await crypto_loader.load_data(symbols=["BTC"], source="coingecko")
    
    # 异常时应该返回包含错误的字典
    assert result is not None
    assert 'data' in result
    assert 'metadata' in result
    assert 'error' in result['metadata']


def test_crypto_loader_normalize_dates_datetime(crypto_loader):
    """测试 CryptoDataLoader（_normalize_dates，datetime 类型，覆盖 760-765 行）"""
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 31)
    
    start_str, end_str = crypto_loader._normalize_dates(start, end)
    
    assert start_str == "2020-01-01"
    assert end_str == "2020-01-31"


def test_crypto_loader_normalize_dates_string(crypto_loader):
    """测试 CryptoDataLoader（_normalize_dates，字符串类型，覆盖 760-765 行）"""
    start = "2020-01-01"
    end = "2020-01-31"
    
    start_str, end_str = crypto_loader._normalize_dates(start, end)
    
    assert start_str == "2020-01-01"
    assert end_str == "2020-01-31"


def test_crypto_loader_fetch_snapshot(crypto_loader, monkeypatch):
    """测试 CryptoDataLoader（_fetch_snapshot，覆盖 774-787 行）"""
    # Mock requests.get
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json = Mock(return_value={"data": [{"timestamp": "2020-01-01", "close": 100}]})
    
    monkeypatch.setattr('src.data.loader.crypto_loader.requests.get', Mock(return_value=mock_response))
    
    result = crypto_loader._fetch_snapshot("BTC", "2020-01-01", "2020-01-31", "coingecko")
    
    assert len(result) > 0


def test_crypto_loader_records_to_dataframe_empty(crypto_loader):
    """测试 CryptoDataLoader（_records_to_dataframe，空记录，覆盖 790-791 行）"""
    result = crypto_loader._records_to_dataframe("BTC", [])
    
    assert result.empty


def test_crypto_loader_records_to_dataframe_with_data(crypto_loader):
    """测试 CryptoDataLoader（_records_to_dataframe，有数据，覆盖 793-824 行）"""
    records = [
        {
            "timestamp": "2020-01-01",
            "open": "100",
            "high": "110",
            "low": "90",
            "close": "105",
            "volume": "1000"
        }
    ]
    
    result = crypto_loader._records_to_dataframe("BTC", records)
    
    assert not result.empty
    assert "open" in result.columns
    assert "high" in result.columns
    assert "low" in result.columns
    assert "close" in result.columns
    assert "volume" in result.columns


def test_crypto_loader_records_to_dataframe_invalid_float(crypto_loader):
    """测试 CryptoDataLoader（_records_to_dataframe，无效浮点数，覆盖 797-802 行）"""
    records = [
        {
            "timestamp": "2020-01-01",
            "open": "invalid",
            "high": "invalid",
            "low": "invalid",
            "close": "invalid",
            "volume": "invalid"
        }
    ]
    
    result = crypto_loader._records_to_dataframe("BTC", records)
    
    # 应该使用 fallback 值
    assert not result.empty


def test_crypto_loader_persist_to_disk_exception(crypto_loader, tmp_path, monkeypatch):
    """测试 CryptoDataLoader（_persist_to_disk，异常处理，覆盖 838-842 行）"""
    df = pd.DataFrame({"open": [100], "high": [110], "low": [90], "close": [105], "volume": [1000]})
    
    # Mock to_csv 抛出异常
    def mock_to_csv(*args, **kwargs):
        raise IOError("Cannot write file")
    
    monkeypatch.setattr('pandas.DataFrame.to_csv', mock_to_csv)
    
    # 应该不会抛出异常，只是记录调试信息
    crypto_loader._persist_to_disk("BTC", "2020-01-01", "2020-01-31", "coingecko", df)


@pytest.mark.asyncio
async def test_get_crypto_data_convenience_function(monkeypatch):
    """测试便捷函数 get_crypto_data（覆盖 848-849 行）"""
    from src.data.loader.crypto_loader import get_crypto_data
    
    # Mock CryptoDataLoader.load_data
    async def mock_load_data(**kwargs):
        return {
            'data': pd.DataFrame(),
            'metadata': {'test': 'data'}
        }
    
    with patch('src.data.loader.crypto_loader.CryptoDataLoader') as MockLoader:
        mock_instance = Mock()
        mock_instance.load_data = mock_load_data
        MockLoader.return_value = mock_instance
        
        result = await get_crypto_data(symbols=["BTC"], source="coingecko")
        
        assert result is not None
        assert 'data' in result
        assert 'metadata' in result


@pytest.mark.asyncio
async def test_get_market_overview_convenience_function(monkeypatch):
    """测试便捷函数 get_market_overview（覆盖 854-856 行）"""
    from src.data.loader.crypto_loader import get_market_overview
    
    # Mock CryptoDataLoader
    mock_loader = AsyncMock()
    mock_loader.initialize = AsyncMock()
    mock_loader.get_market_overview = AsyncMock(return_value=None)
    
    with patch('src.data.loader.crypto_loader.CryptoDataLoader', return_value=mock_loader):
        result = await get_market_overview()
        
        # 应该调用 initialize 和 get_market_overview
        mock_loader.initialize.assert_called_once()
        mock_loader.get_market_overview.assert_called_once()

