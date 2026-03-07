"""
ForexDataLoader 单元测试 - 针对真实实现
使用 mock 避免 yfinance 依赖
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


import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

# Mock yfinance before importing forex_loader
sys.modules["yfinance"] = MagicMock()
from src.data.loader.forex_loader import ForexDataLoader, ForexRate


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def loader(temp_dir):
    with patch("src.data.loader.forex_loader.yf"):
        return ForexDataLoader(
            save_path=str(temp_dir),
            max_retries=1,
            cache_days=1,
            frequency="1d",
        )


def _sample_forex_df():
    dates = pd.date_range("2024-01-01", periods=3)
    return pd.DataFrame(
        {
            "Open": [1.08, 1.09, 1.10],
            "High": [1.11, 1.12, 1.13],
            "Low": [1.07, 1.08, 1.09],
            "Close": [1.10, 1.11, 1.12],
            "Volume": [1000000, 1100000, 1200000],
        },
        index=dates,
    )


class TestForexDataLoaderInitialization:
    def test_init_defaults(self, temp_dir):
        with patch("src.data.loader.forex_loader.yf"):
            loader = ForexDataLoader()
            assert loader.save_path.exists()
            assert loader.cache_dir.exists()
            assert loader.max_retries == 3
            assert loader.cache_days == 1

    def test_init_custom_params(self, temp_dir):
        with patch("src.data.loader.forex_loader.yf"):
            loader = ForexDataLoader(
                save_path=str(temp_dir),
                max_retries=5,
                cache_days=7,
                frequency="1h",
            )
            assert loader.save_path == Path(temp_dir)
            assert loader.max_retries == 5
            assert loader.cache_days == 7
            assert loader.frequency == "1h"

    def test_get_metadata(self, loader):
        meta = loader.get_metadata()
        assert meta["loader_type"] == "forex"
        assert "supported_currencies" in meta
        assert "supported_pairs" in meta

    def test_get_supported_currencies(self, loader):
        currencies = loader.get_supported_currencies()
        assert isinstance(currencies, list)
        assert "USD" in currencies

    def test_get_supported_pairs(self, loader):
        pairs = loader.get_supported_pairs()
        assert isinstance(pairs, list)
        assert "EURUSD=X" in pairs

    def test_get_required_config_fields(self, loader):
        fields = loader.get_required_config_fields()
        assert "cache_dir" in fields
        assert "max_retries" in fields

    def test_validate_config_success(self, loader):
        assert loader.validate_config() is True

    def test_validate_config_missing_field(self, temp_dir):
        with patch("src.data.loader.forex_loader.yf"):
            loader = ForexDataLoader(save_path=str(temp_dir))
            loader.runtime_config.pop("cache_dir", None)
            assert loader.validate_config() is False


class TestForexDataLoaderLoad:
    def test_load_uses_memory_cache(self, loader, temp_dir):
        df = _sample_forex_df()
        cache_key = loader._build_cache_key("EURUSD=X", "2024-01-01", "2024-01-03")
        loader.cache_manager.set(cache_key, df.copy())

        result = loader.load("EURUSD=X", "2024-01-01", "2024-01-03")
        assert_frame_equal(result, df)

    def test_load_reads_csv_cache(self, loader, temp_dir):
        df = _sample_forex_df()
        csv_path = loader._cache_file_path("EURUSD=X", "2024-01-01", "2024-01-03")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path)

        with patch.object(loader, "_fetch_forex_data", return_value=None):
            result = loader.load("EURUSD=X", "2024-01-01", "2024-01-03")
            assert not result.empty

    def test_load_fetches_and_persists(self, loader, temp_dir):
        df = _sample_forex_df()
        with patch.object(loader, "_fetch_forex_data", return_value=df):
            result = loader.load("EURUSD=X", "2024-01-01", "2024-01-03", force_refresh=True)
            assert not result.empty
            assert "open" in result.columns

    def test_load_empty_data_returns_empty_df(self, loader):
        with patch.object(loader, "_fetch_forex_data", return_value=None):
            result = loader.load("EURUSD=X", "2024-01-01", "2024-01-03", force_refresh=True)
            assert result.empty

    def test_load_batch_handles_failures(self, loader):
        with patch.object(loader, "load", side_effect=[_sample_forex_df(), Exception("boom")]):
            results = loader.load_batch(["EURUSD=X", "USDJPY=X"], "2024-01-01", "2024-01-03")
            assert "EURUSD=X" in results
            assert results["USDJPY=X"] is None

    def test_load_batch_empty_list(self, loader):
        assert loader.load_batch([], "2024-01-01", "2024-01-03") == {}

    def test_load_data_returns_dict(self, loader):
        df = _sample_forex_df()
        with patch.object(loader, "load", return_value=df):
            result = loader.load_data("EURUSD=X")
            assert "data" in result
            assert "symbol" in result
            assert isinstance(result["data"], pd.DataFrame)


class TestForexDataLoaderHelpers:
    def test_normalize_dates_accepts_datetime(self, loader):
        start, end = loader._normalize_dates(datetime(2024, 1, 1), datetime(2024, 1, 3))
        assert start == "2024-01-01"
        assert end == "2024-01-03"

    def test_normalize_dates_accepts_strings(self, loader):
        start, end = loader._normalize_dates("2024-01-01", "2024-01-03")
        assert start == "2024-01-01"
        assert end == "2024-01-03"

    def test_build_cache_key(self, loader):
        key = loader._build_cache_key("EURUSD=X", "2024-01-01", "2024-01-03")
        assert "EURUSD=X" in key
        assert "2024-01-01" in key

    def test_cache_file_path_sanitizes_symbol(self, loader):
        path = loader._cache_file_path("EUR/USD=X", "2024-01-01", "2024-01-03")
        assert "/" not in path.name
        assert "=" not in path.name

    def test_is_file_cache_valid_missing_file(self, loader, temp_dir):
        path = temp_dir / "missing.csv"
        assert loader._is_file_cache_valid(path) is False

    def test_is_file_cache_valid_expired(self, loader, temp_dir):
        path = temp_dir / "old.csv"
        path.write_text("data")
        import time
        old_time = time.time() - (loader.cache_days + 1) * 86400
        import os
        os.utime(path, (old_time, old_time))
        assert loader._is_file_cache_valid(path) is False

    def test_get_from_cache_memory_hit(self, loader):
        df = _sample_forex_df()
        cache_key = loader._build_cache_key("EURUSD=X", "2024-01-01", "2024-01-03")
        loader.cache_manager.set(cache_key, df.copy())
        result = loader._get_from_cache(cache_key, "EURUSD=X", "2024-01-01", "2024-01-03")
        assert_frame_equal(result, df)

    def test_get_from_cache_file_hit(self, loader, temp_dir):
        df = _sample_forex_df()
        csv_path = loader._cache_file_path("EURUSD=X", "2024-01-01", "2024-01-03")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path)
        cache_key = loader._build_cache_key("EURUSD=X", "2024-01-01", "2024-01-03")
        result = loader._get_from_cache(cache_key, "EURUSD=X", "2024-01-01", "2024-01-03")
        assert not result.empty

    def test_fetch_forex_data_success(self, loader):
        df = _sample_forex_df()
        with patch("src.data.loader.forex_loader.yf") as mock_yf:
            mock_yf.download.return_value = df
            result = loader._fetch_forex_data("EURUSD=X", "2024-01-01", "2024-01-03")
            assert_frame_equal(result, df)

    def test_fetch_forex_data_retries_on_failure(self, loader):
        with patch("src.data.loader.forex_loader.yf") as mock_yf:
            mock_yf.download.side_effect = [Exception("fail"), _sample_forex_df()]
            loader.max_retries = 2
            result = loader._fetch_forex_data("EURUSD=X", "2024-01-01", "2024-01-03")
            assert result is not None

    def test_fetch_forex_data_exhausts_retries(self, loader):
        with patch("src.data.loader.forex_loader.yf") as mock_yf:
            mock_yf.download.side_effect = Exception("fail")
            loader.max_retries = 2
            result = loader._fetch_forex_data("EURUSD=X", "2024-01-01", "2024-01-03")
            assert result is None

    def test_normalize_dataframe_standardizes_columns(self, loader):
        df = pd.DataFrame(
            {
                "Open": [1.0],
                "High": [1.1],
                "Low": [0.9],
                "Close": [1.05],
                "Adj Close": [1.05],
                "Volume": [1000],
            }
        )
        normalized = loader._normalize_dataframe(df)
        assert "open" in normalized.columns
        assert "high" in normalized.columns
        assert "close" in normalized.columns

    def test_normalize_dataframe_fills_missing_columns(self, loader):
        df = pd.DataFrame({"Open": [1.0]})
        normalized = loader._normalize_dataframe(df)
        assert "volume" in normalized.columns

    def test_persist_to_disk_success(self, loader, temp_dir):
        df = _sample_forex_df()
        loader._persist_to_disk("EURUSD=X", "2024-01-01", "2024-01-03", df)
        csv_path = loader._cache_file_path("EURUSD=X", "2024-01-01", "2024-01-03")
        assert csv_path.exists()

    def test_persist_to_disk_handles_error(self, loader, temp_dir):
        df = _sample_forex_df()
        with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("permission denied")):
            loader._persist_to_disk("EURUSD=X", "2024-01-01", "2024-01-03", df)
            # 不应抛出异常

    def test_clear_cache(self, loader, temp_dir):
        csv_path = temp_dir / "cache" / "test.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("data")
        loader.clear_cache()
        assert not csv_path.exists()

    def test_ensure_initialized(self, temp_dir):
        with patch("src.data.loader.forex_loader.yf"):
            loader = ForexDataLoader(save_path=str(temp_dir))
            loader.is_initialized = False
            loader._ensure_initialized()
            assert loader.is_initialized is True


class TestForexRate:
    def test_forex_rate_creation(self):
        rate = ForexRate(
            base_currency="USD",
            quote_currency="EUR",
            symbol="EURUSD=X",
            rate=1.0850,
            timestamp=datetime.now(),
        )
        assert rate.base_currency == "USD"
        assert rate.quote_currency == "EUR"
        assert rate.rate == 1.0850

