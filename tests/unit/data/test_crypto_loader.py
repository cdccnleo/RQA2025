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


import importlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data.loader.crypto_loader import BinanceLoader, CoinGeckoLoader, CryptoData, CryptoDataLoader


class AsyncCacheStub:
    def __init__(self, initial: Optional[Dict[str, Any]] = None):
        self.store = dict(initial or {})
        self.set_calls: List[tuple[str, Any, Any]] = []

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self.store[key] = value
        self.set_calls.append((key, value, ttl))


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None):
        self.status = status
        self._payload = payload or {}

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeSession:
    def __init__(self, *responses: FakeResponse):
        self.responses = list(responses)
        self.calls: List[tuple[str, Optional[Dict[str, Any]]]] = []

    def get(self, url: str, params: Optional[Dict[str, Any]] = None):
        self.calls.append((url, params))
        if not self.responses:
            raise AssertionError("No response queued for FakeSession")
        return self.responses.pop(0)


class SimpleCache:
    def __init__(self):
        self.store: Dict[str, Any] = {}

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self.store[key] = value
        return True


@pytest.fixture
def sync_loader(tmp_path):
    loader = CryptoDataLoader(save_path=tmp_path)
    loader.cache_manager = SimpleCache()
    return loader


@pytest.mark.asyncio
async def test_coingecko_get_top_coins_uses_cache():
    loader = CoinGeckoLoader()
    cached_payload = [{
        "symbol": "BTC",
        "name": "Bitcoin",
        "price": 1.0,
        "volume_24h": 100.0,
        "market_cap": 200.0,
        "price_change_24h": 1.5,
        "price_change_percentage_24h": 2.5,
        "high_24h": 2.0,
        "low_24h": 0.5,
        "timestamp": datetime.now(),
        "source": "coingecko",
    }]
    loader.cache_manager = AsyncCacheStub({"coingecko_top_coins_2": cached_payload})
    loader.session = FakeSession()

    result = await loader.get_top_coins(limit=2)

    assert len(result) == 1
    assert result[0].symbol == "BTC"
    assert loader.session.calls == []


@pytest.mark.asyncio
async def test_coingecko_get_top_coins_fetches_and_caches(monkeypatch):
    loader = CoinGeckoLoader()
    loader.cache_manager = AsyncCacheStub()
    payload = [
        {
            "symbol": "eth",
            "name": "Ethereum",
            "current_price": 2,
            "total_volume": 3,
            "market_cap": 4,
            "price_change_24h": 0.1,
            "price_change_percentage_24h": 1.0,
            "high_24h": 5,
            "low_24h": 1,
            "last_updated": 1700000000000,
        }
    ]
    loader.session = FakeSession(FakeResponse(status=200, payload=payload))

    result = await loader.get_top_coins(limit=1)

    assert result[0].symbol == "ETH"
    cache_value = loader.cache_manager.store["coingecko_top_coins_1"]
    assert cache_value[0]["symbol"] == "ETH"


@pytest.mark.asyncio
async def test_coingecko_get_top_coins_handles_http_error():
    loader = CoinGeckoLoader()
    loader.cache_manager = AsyncCacheStub()
    loader.session = FakeSession(FakeResponse(status=500, payload={}))

    result = await loader.get_top_coins(limit=1)

    assert result == []


@pytest.mark.asyncio
async def test_coingecko_get_coin_detail_handles_error():
    loader = CoinGeckoLoader()
    loader.cache_manager = AsyncCacheStub()
    loader.session = FakeSession(FakeResponse(status=500, payload={}))

    result = await loader.get_coin_detail("bitcoin")

    assert result is None


@pytest.mark.asyncio
async def test_coingecko_get_coin_detail_uses_cache():
    loader = CoinGeckoLoader()
    loader.cache_manager = AsyncCacheStub({"coingecko_coin_detail_btc": {"name": "Bitcoin"}})
    loader.session = FakeSession()

    result = await loader.get_coin_detail("btc")

    assert result["name"] == "Bitcoin"


@pytest.mark.asyncio
async def test_coingecko_get_market_data_fetches_and_caches():
    loader = CoinGeckoLoader()
    loader.cache_manager = AsyncCacheStub()
    payload = {
        "data": {
            "total_market_cap": {"usd": 10},
            "total_volume": {"usd": 5},
            "market_cap_percentage": {"btc": 40},
            "market_cap_change_percentage_24h_usd": 0.5,
        }
    }
    loader.session = FakeSession(FakeResponse(status=200, payload=payload))

    result = await loader.get_market_data()

    assert result.total_market_cap == 10
    cached = loader.cache_manager.store["coingecko_market_data"]
    assert cached["total_volume_24h"] == 5


@pytest.mark.asyncio
async def test_binance_get_ticker_uses_cache():
    loader = BinanceLoader()
    loader.cache_manager = AsyncCacheStub({"binance_ticker_24hr_BTCUSDT": {"lastPrice": "1.2"}})
    loader.session = FakeSession()

    result = await loader.get_ticker_24hr("BTCUSDT")

    assert result["lastPrice"] == "1.2"
    assert loader.session.calls == []


@pytest.mark.asyncio
async def test_binance_get_klines_formats_rows():
    loader = BinanceLoader()
    loader.cache_manager = AsyncCacheStub()
    kline = [
        1700000000000,
        "1.0",
        "2.0",
        "0.5",
        "1.5",
        "100",
        1700003600000,
        "150",
        "10",
        "20",
        "30",
    ]
    loader.session = FakeSession(FakeResponse(status=200, payload=[kline]))

    result = await loader.get_klines("BTCUSDT", "1h", limit=1)

    assert result[0]["trades"] == 10
    cache_key = "binance_klines_BTCUSDT_1h_1"
    assert cache_key in loader.cache_manager.store


@pytest.mark.asyncio
async def test_binance_get_exchange_info_returns_none_on_exception():
    loader = BinanceLoader()
    loader.cache_manager = AsyncCacheStub()

    class RaisingSession(FakeSession):
        def get(self, *args, **kwargs):
            raise RuntimeError("boom")

    loader.session = RaisingSession()
    result = await loader.get_exchange_info()

    assert result is None


@pytest.mark.asyncio
async def test_binance_get_ticker_handles_http_error():
    loader = BinanceLoader()
    loader.cache_manager = AsyncCacheStub()
    loader.session = FakeSession(FakeResponse(status=500, payload={}))

    result = await loader.get_ticker_24hr("BTCUSDT")

    assert result is None


@pytest.mark.asyncio
async def test_coingecko_loader_async_context(monkeypatch):
    created = {}

    class DummySession:
        def __init__(self, *args, **kwargs):
            self.closed = False
            created["session"] = self

        async def close(self):
            self.closed = True

    monkeypatch.setattr("src.data.loader.crypto_loader.aiohttp.ClientSession", DummySession)
    loader = CoinGeckoLoader()

    async with loader as ctx:
        assert ctx is loader

    assert created["session"].closed


@pytest.mark.asyncio
async def test_binance_loader_async_context(monkeypatch):
    created = {}

    class DummySession:
        def __init__(self, *args, **kwargs):
            self.closed = False
            created["session"] = self

        async def close(self):
            self.closed = True

    monkeypatch.setattr("src.data.loader.crypto_loader.aiohttp.ClientSession", DummySession)
    loader = BinanceLoader()

    async with loader as ctx:
        assert ctx is loader

    assert created["session"].closed


def test_coingecko_loader_metadata_helpers():
    loader = CoinGeckoLoader(api_key="demo")
    assert "cache_dir" in loader.get_required_config_fields()
    assert loader.validate_config() is True
    metadata = loader.get_metadata()
    assert metadata["loader_type"] == "coingecko"


def test_binance_loader_metadata_helpers():
    loader = BinanceLoader(api_key="key", api_secret="secret")
    assert "cache_dir" in loader.get_required_config_fields()
    assert loader.validate_config() is True
    metadata = loader.get_metadata()
    assert metadata["loader_type"] == "binance"


def test_coin_gecko_loader_load_not_implemented():
    loader = CoinGeckoLoader()
    with pytest.raises(NotImplementedError):
        loader.load("2024-01-01", "2024-01-02", "1d")


def test_binance_loader_load_not_implemented():
    loader = BinanceLoader()
    with pytest.raises(NotImplementedError):
        loader.load("2024-01-01", "2024-01-02", "1h")


def test_crypto_loader_load_uses_memory_cache(sync_loader, monkeypatch):
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [10.0]})
    key = "BTC_2024-01-01_2024-01-02_coingecko"
    sync_loader.cache_manager.set(key, df)

    called = False

    def _unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("fetch should not be called")

    monkeypatch.setattr(sync_loader, "_fetch_snapshot", _unexpected)
    result = sync_loader.load("BTC", "2024-01-01", "2024-01-02")

    assert_frame_equal(result, df)
    assert called is False


def test_crypto_loader_load_reads_csv_cache(sync_loader):
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10.0],
        },
        index=pd.to_datetime(["2024-01-01"]),
    )
    start, end = sync_loader._normalize_dates("2024-01-01", "2024-01-02")
    csv_path = sync_loader._cache_file_path("BTC", start, end, "coingecko")
    df.to_csv(csv_path)

    result = sync_loader.load("BTC", "2024-01-01", "2024-01-02")

    assert_frame_equal(result, df)
    assert isinstance(sync_loader.cache_manager.get(f"BTC_{start}_{end}_coingecko"), pd.DataFrame)


def test_crypto_loader_load_fetches_and_persists(sync_loader, monkeypatch, tmp_path):
    records = [
        {"timestamp": "2024-01-01T00:00:00Z", "close": 10.0, "volume": 5.0},
        {"timestamp": "2024-01-02T00:00:00Z", "open": 11.0, "close": 12.0, "volume": 6.0},
    ]
    monkeypatch.setattr(sync_loader, "_fetch_snapshot", lambda *_, **__: records)
    persisted: list[Path] = []

    def _persist(symbol, start, end, source, df):
        path = sync_loader._cache_file_path(symbol, start, end, source)
        persisted.append(path)

    monkeypatch.setattr(sync_loader, "_persist_to_disk", _persist)
    result = sync_loader.load("BTC", "2024-01-01", "2024-01-02")

    assert not result.empty
    assert persisted
    key = "BTC_2024-01-01_2024-01-02_coingecko"
    assert key in sync_loader.cache_manager.store


def test_crypto_loader_load_batch_handles_failures(sync_loader, monkeypatch):
    def fake_load(symbol, *args, **kwargs):
        if symbol == "ERR":
            raise ValueError("boom")
        return pd.DataFrame({"close": [1.0]})

    monkeypatch.setattr(sync_loader, "load", fake_load)
    result = sync_loader.load_batch(["OK1", "ERR"], "2024-01-01", "2024-01-02")

    assert isinstance(result["OK1"], pd.DataFrame)
    assert result["ERR"] is None


def test_records_to_dataframe_handles_defaults(sync_loader):
    rows = [
        {"timestamp": "2024-01-01T00:00:00Z", "close": 2.0},
        {"timestamp": "2024-01-02T00:00:00Z", "open": 3.0, "high": 5.0, "low": 1.0, "close": 4.0, "volume": 9.0},
    ]
    df = sync_loader._records_to_dataframe("BTC", rows)

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.iloc[0]["open"] == df.iloc[0]["close"]


def test_fetch_snapshot_invokes_requests(monkeypatch, sync_loader):
    captured: Dict[str, Any] = {}

    class DummyResponse:
        def __init__(self):
            self._data = {"data": [{"price": 1}]}

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    class DummyRequests:
        def get(self, url, params=None, timeout=None):
            captured["url"] = url
            captured["params"] = params
            captured["timeout"] = timeout
            return DummyResponse()

    module = importlib.import_module("src.data.loader.crypto_loader")
    monkeypatch.setattr(module, "requests", DummyRequests())

    data = sync_loader._fetch_snapshot("BTC", "2024-01-01", "2024-01-02", "coingecko")
    assert captured["params"]["symbol"] == "BTC"
    assert data == [{"price": 1}]


def test_cache_file_path_sanitizes_symbol(sync_loader):
    start, end = sync_loader._normalize_dates("2024-01-01", "2024-01-02")
    path = sync_loader._cache_file_path("BTC/USDT", start, end, "binance")
    assert "BTC_USDT" in path.name


def test_normalize_dates_accepts_datetime(sync_loader):
    start, end = sync_loader._normalize_dates(datetime(2024, 1, 1), datetime(2024, 1, 2))
    assert start == "2024-01-01"
    assert end == "2024-01-02"


@pytest.mark.asyncio
async def test_validate_data_flags_invalid_entries():
    loader = CryptoDataLoader()
    data = [
        CryptoData("BTC", "Bitcoin", -1, 10, 0, 0, 0, 0, 0, datetime.now(), "src"),  # price invalid
        CryptoData("ETH", "Ethereum", 10, -5, 0, 0, 0, 0, 0, datetime.now(), "src"),  # volume invalid
        CryptoData("SOL", "Solana", 2_000_000, 1, 0, 0, 0, 0, 0, datetime.now(), "src"),  # too high price
        CryptoData("OK", "Ok", 1, 1, 0, 0, 0, 0, 0, datetime.now(), "src"),
    ]

    result = await loader.validate_data(data)

    assert result["invalid_records"] == 3
    assert result["valid_records"] == 1
    assert result["valid"] is False


@pytest.mark.asyncio
async def test_load_data_pipeline(monkeypatch):
    loader = CryptoDataLoader()

    async def fake_init():
        return None

    async def fake_get_crypto(symbols, source):
        return [
            CryptoData("BTC", "Bitcoin", 1, 2, 3, 0, 0, 1.2, 0.8, datetime.now(), source)
        ]

    async def fake_validate(data):
        return {"valid": True}

    monkeypatch.setattr(loader, "initialize", fake_init)
    monkeypatch.setattr(loader, "get_crypto_data", fake_get_crypto)
    monkeypatch.setattr(loader, "validate_data", fake_validate)

    result = await loader.load_data(symbols=["BTC"], source="coingecko")

    assert result["metadata"]["total_records"] == 1
    assert not result["data"].empty


@pytest.mark.asyncio
async def test_get_crypto_data_unknown_source_returns_empty():
    loader = CryptoDataLoader()
    result = await loader.get_crypto_data(source="unknown")
    assert result == []


@pytest.mark.asyncio
async def test_get_coingecko_data_filters_symbols():
    loader = CryptoDataLoader()

    class StubAsyncLoader:
        def __init__(self, coins):
            self.coins = coins

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get_top_coins(self, limit=100):
            return self.coins

    coins = [
        CryptoData("BTC", "Bitcoin", 1, 1, 1, 0, 0, 1, 1, datetime.now(), "coingecko"),
        CryptoData("ETH", "Ethereum", 1, 1, 1, 0, 0, 1, 1, datetime.now(), "coingecko"),
    ]

    loader.coingecko_loader = StubAsyncLoader(coins)
    result = await loader._get_coingecko_data(["ETH"])

    assert len(result) == 1
    assert result[0].symbol == "ETH"


@pytest.mark.asyncio
async def test_get_binance_data_builds_crypto_objects():
    loader = CryptoDataLoader()

    class StubBinanceLoader:
        def __init__(self, payloads):
            self.payloads = payloads

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get_ticker_24hr(self, symbol):
            return self.payloads.get(symbol)

    payloads = {
        "BTCUSDT": {
            "lastPrice": "10",
            "volume": "100",
            "priceChange": "1",
            "priceChangePercent": "10",
            "highPrice": "12",
            "lowPrice": "8",
        }
    }

    loader.binance_loader = StubBinanceLoader(payloads)
    result = await loader._get_binance_data(["BTC"])

    assert result[0].symbol == "BTC"
    assert result[0].price == 10.0


@pytest.mark.asyncio
async def test_crypto_loader_initialize_sets_subloaders(monkeypatch, tmp_path):
    class DummyCacheManager:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, key):
            return None

        def set(self, key, value, ttl=None):
            return True

    monkeypatch.setattr("src.data.loader.crypto_loader.CacheManager", DummyCacheManager)
    loader = CryptoDataLoader(save_path=tmp_path)

    await loader.initialize()

    assert loader.coingecko_loader is not None
    assert loader.binance_loader is not None

