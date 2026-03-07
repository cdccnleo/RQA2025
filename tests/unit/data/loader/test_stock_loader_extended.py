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


import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data.loader.stock_loader import StockDataLoader, ak
from src.infrastructure.utils.exceptions import DataLoaderError


@pytest.fixture
def loader(tmp_path, monkeypatch):
    # 禁用时间统计等副作用
    monkeypatch.setattr(StockDataLoader, "_update_stats", lambda self: None)
    return StockDataLoader(save_path=str(tmp_path))


def make_dataframe():
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.5, 10.5],
            "close": [10.5, 11.5],
            "volume": [1000, 1200],
        },
        index=dates,
    )
    return df


def test_load_data_uses_cached_file(loader):
    df = make_dataframe()
    file_path = loader._get_file_path("000001", "2024-01-01", "2024-01-02")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, encoding="utf-8")

    result = loader.load_data(symbol="000001", start_date="2024-01-01", end_date="2024-01-02")
    pd.testing.assert_frame_equal(result, df, check_freq=False)


def test_fetch_raw_data_fallbacks_to_hist(loader, monkeypatch):
    fallback_df = pd.DataFrame({"date": ["20240101"], "open": [10], "close": [11], "high": [11], "low": [9], "volume": [100]})

    def fake_retry(self, func, *args, **kwargs):
        name = getattr(func, "__name__", "")
        if name == "stock_zh_a_daily":
            raise DataLoaderError("daily fail")
        return fallback_df

    monkeypatch.setattr(loader, "_retry_api_call", fake_retry.__get__(loader, StockDataLoader))

    def daily(*args, **kwargs):
        raise AssertionError("should not call daily directly")

    def hist(*args, **kwargs):
        return fallback_df

    monkeypatch.setattr(ak, "stock_zh_a_daily", daily)
    monkeypatch.setattr(ak, "stock_zh_a_hist", hist)

    result = loader._fetch_raw_data("000001", "2024-01-01", "2024-01-02", "hfq")
    pd.testing.assert_frame_equal(result, fallback_df)


def test_retry_api_call_raises_after_empty(loader):
    def empty_df(*args, **kwargs):
        return pd.DataFrame()

    with pytest.raises(DataLoaderError, match="API 返回数据为空"):
        loader._retry_api_call(empty_df)


def test_load_cache_payload_expires(loader, tmp_path):
    cache_file = loader._get_cache_file_path("000001")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    old_time = datetime.now() - timedelta(days=loader.cache_days + 5)
    payload = {
        "data": make_dataframe(),
        "metadata": {"cached_time": old_time.isoformat()},
    }
    with cache_file.open("wb") as fp:
        pickle.dump(payload, fp)

    assert loader._load_cache_payload(cache_file) is None


def test_load_single_stock_force_refresh_saves_cache(loader, monkeypatch):
    df = make_dataframe()

    monkeypatch.setattr(loader, "_load_data_impl", lambda *args, **kwargs: df)
    monkeypatch.setattr(loader, "_validate_data", lambda data: (True, []))

    saved = {}

    def fake_save(self, cache_file: Path, payload):
        saved["path"] = cache_file
        saved["payload"] = payload

    monkeypatch.setattr(StockDataLoader, "_save_cache_payload", fake_save)

    result = loader.load_single_stock(
        symbol="000001",
        start_date="2024-01-01",
        end_date="2024-01-02",
        force_refresh=True,
    )

    assert result["cache_info"]["is_from_cache"] is False
    assert saved["path"].exists() is False  # fake_save 不写文件
    assert saved["payload"]["metadata"]["symbol"] == "000001"

