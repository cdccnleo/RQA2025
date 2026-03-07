"""
补齐 StockDataLoader 边界路径的单元测试，聚焦缓存/重试/行业/股票列表逻辑。
"""
from __future__ import annotations

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

import os
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timedelta
from typing import Any
import pickle

import pandas as pd
import pytest

from src.data.loader.stock_loader import (
    StockDataLoader,
    IndustryLoader,
    StockListLoader,
    DataLoaderError,
)
from requests import RequestException


@pytest.fixture
def stock_loader(tmp_path) -> StockDataLoader:
    return StockDataLoader(save_path=str(tmp_path))


def test_load_data_impl_cache_empty_raises(monkeypatch, stock_loader, tmp_path):
    cache_file = tmp_path / "cached.csv"
    monkeypatch.setattr(stock_loader, "_get_file_path", lambda *args, **kwargs: cache_file)
    monkeypatch.setattr(stock_loader, "_is_cache_valid", lambda path: True)
    monkeypatch.setattr("pandas.read_csv", lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(DataLoaderError, match="缓存数据为空"):
        stock_loader._load_data_impl("000001", "2024-01-01", "2024-01-02")


def test_load_data_impl_api_empty_raises(monkeypatch, stock_loader, tmp_path):
    monkeypatch.setattr(stock_loader, "_is_cache_valid", lambda *args, **kwargs: False)
    monkeypatch.setattr(stock_loader, "_fetch_raw_data", lambda *args, **kwargs: None)

    with pytest.raises(DataLoaderError, match="API 返回数据为空"):
        stock_loader._load_data_impl("000001", "2024-01-01", "2024-01-02", force_refresh=True)


def test_load_data_impl_processes_data_and_marks_trading_day(monkeypatch, stock_loader, tmp_path):
    processed = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [1000],
        },
        index=pd.to_datetime(["2024-01-01"]),
    )

    monkeypatch.setattr(stock_loader, "_is_cache_valid", lambda *args, **kwargs: False)
    monkeypatch.setattr(stock_loader, "_fetch_raw_data", lambda *args, **kwargs: pd.DataFrame({"dummy": [1]}))
    monkeypatch.setattr(stock_loader, "_process_raw_data", lambda *args, **kwargs: processed.copy())
    monkeypatch.setattr(stock_loader, "_get_holidays", lambda *args, **kwargs: [datetime(2024, 1, 1).date()])
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, path, encoding=None: None)

    result = stock_loader._load_data_impl("000001", "2024-01-01", "2024-01-02", force_refresh=True)

    assert "is_trading_day" in result.columns
    assert result["is_trading_day"].iloc[0] == 0


def test_retry_api_call_handles_request_exception(monkeypatch, stock_loader):
    def always_fail(*args, **kwargs):
        raise RequestException("network down")

    with pytest.raises(DataLoaderError, match="network down"):
        stock_loader._retry_api_call(always_fail)


def test_retry_api_call_handles_generic_exception(stock_loader):
    def always_fail(*args, **kwargs):
        raise RuntimeError("boom")

    with pytest.raises(DataLoaderError, match="boom"):
        stock_loader._retry_api_call(always_fail)


def test_retry_api_call_empty_dataframe_raises(stock_loader):
    stock_loader.max_retries = 1

    with pytest.raises(DataLoaderError, match="API 返回数据为空"):
        stock_loader._retry_api_call(lambda *args, **kwargs: pd.DataFrame())


def test_fetch_raw_data_fallbacks_to_hist(monkeypatch, stock_loader):
    def stock_zh_a_daily(*args, **kwargs):
        return "daily"

    def stock_zh_a_hist(*args, **kwargs):
        return "hist"

    fake_ak = SimpleNamespace(stock_zh_a_daily=stock_zh_a_daily, stock_zh_a_hist=stock_zh_a_hist)
    monkeypatch.setattr("src.data.loader.stock_loader.ak", fake_ak)

    call_log = []

    def fake_retry(func, *args, **kwargs):
        if func is stock_zh_a_daily:
            call_log.append("daily")
            raise DataLoaderError("daily fail")
        call_log.append("hist")
        return pd.DataFrame({"open": [1.0], "high": [1.2], "low": [0.9], "close": [1.1], "volume": [100]})

    monkeypatch.setattr(stock_loader, "_retry_api_call", fake_retry)

    df = stock_loader._fetch_raw_data("000001", "2024-01-01", "2024-01-02", "hfq")
    assert not df.empty
    assert call_log == ["daily", "hist"]


def test_industry_loader_load_data_uses_cache(tmp_path):
    loader = IndustryLoader(save_path=str(tmp_path))
    loader._industry_map = None
    loader._setup()
    df = pd.DataFrame({"symbol": ["000001"], "industry": ["科技"]})
    df.to_csv(loader.industry_map_path, index=False, encoding="utf-8")

    mapping = loader.load_data()
    assert mapping == {"000001": "科技"}


def test_industry_loader_load_data_all_fail_raises(monkeypatch, tmp_path):
    loader = IndustryLoader(save_path=str(tmp_path))
    loader._industry_map = None
    loader._setup()
    monkeypatch.setattr(loader, "_is_cache_valid", lambda path: False)
    industry_df = pd.DataFrame({"板块代码": ["BK001"], "板块名称": ["能源"]})
    monkeypatch.setattr(loader, "_fetch_raw_data", lambda: industry_df)

    def failing_components(*args, **kwargs):
        raise ConnectionError("fail")

    monkeypatch.setattr("src.data.loader.stock_loader.ak.stock_board_industry_cons_em", failing_components, raising=False)

    with pytest.raises(DataLoaderError, match="无法获取任何行业映射数据"):
        loader.load_data()


def test_industry_loader_get_industry_unknown_returns_default(tmp_path):
    loader = IndustryLoader(save_path=str(tmp_path))
    loader._industry_map = {"000002": "石油行业"}
    loader.debug_mode = False

    assert loader.get_industry("2") == "能源"
    assert loader.get_industry("999999") == "行业未知"


def test_stock_list_loader_load_data_uses_cache(tmp_path):
    loader = StockListLoader(save_path=str(tmp_path))
    df = pd.DataFrame({"股票代码": ["000001"], "股票名称": ["平安银行"]})
    loader._setup()
    df.to_csv(loader.list_path, index=False, encoding="utf-8")

    result = loader.load_data()
    assert not result.empty


def test_stock_list_loader_load_data_fetches_when_cache_invalid(monkeypatch, tmp_path):
    loader = StockListLoader(save_path=str(tmp_path))
    loader._setup()
    monkeypatch.setattr(loader, "_is_cache_valid", lambda path: False)
    monkeypatch.setattr(loader, "_fetch_raw_data", lambda: pd.DataFrame({"code": ["000001"], "name": ["平安"]}))

    result = loader.load_data()
    assert list(result.columns) == ["股票代码", "股票名称"]


def test_stock_list_loader_get_available_symbols(monkeypatch, tmp_path):
    loader = StockListLoader(save_path=str(tmp_path))
    monkeypatch.setattr(loader, "_fetch_raw_data", lambda: pd.DataFrame({"code": ["000001", "000002"]}))

    symbols = loader.get_available_symbols()
    assert symbols == ["000001", "000002"]


def test_industry_loader_calculate_concentration_success(monkeypatch, tmp_path):
    loader = IndustryLoader(save_path=str(tmp_path))
    loader._industry_map = {}
    loader.stock_loader = SimpleNamespace()

    series = pd.Series(
        [1.0, 1.1, 1.2],
        index=pd.date_range("2024-01-01", periods=3),
        name="close",
    )

    monkeypatch.setattr(
        loader,
        "_get_industry_components",
        lambda code: pd.DataFrame({"symbol": ["000001", "000002"]}),
    )
    monkeypatch.setattr(loader, "_load_stock_data", lambda *args, **kwargs: series.copy())

    result = loader.calculate_industry_concentration(
        "能源",
        start_date="2024-01-01",
        end_date="2024-01-05",
        window=10,
    )

    assert not result.empty
    assert set(result.columns) == {"CR4", "CR8"}


def test_industry_loader_calculate_concentration_no_data(monkeypatch, tmp_path):
    loader = IndustryLoader(save_path=str(tmp_path))
    loader._industry_map = {}

    monkeypatch.setattr(
        loader,
        "_get_industry_components",
        lambda code: pd.DataFrame({"symbol": ["000001"]}),
    )
    monkeypatch.setattr(loader, "_load_stock_data", lambda *args, **kwargs: None)

    result = loader.calculate_industry_concentration(
        "能源",
        start_date="2024-01-01",
        end_date="2024-01-05",
        window=10,
    )

    assert result.empty


def test_industry_loader_get_industry_debug_mode_raises(tmp_path):
    loader = IndustryLoader(save_path=str(tmp_path))
    loader._industry_map = {}
    loader.debug_mode = True

    with pytest.raises(DataLoaderError):
        loader.get_industry("000001")


def test_industry_loader_check_cache_handles_empty_file(tmp_path):
    loader = IndustryLoader(save_path=str(tmp_path))
    loader._setup()
    cache_file = loader.save_path / "industry_cache.csv"
    cache_file.write_text("")

    is_valid, df = loader._check_cache(cache_file)
    assert is_valid is False
    assert df is None


def test_stock_list_loader_is_cache_valid_expired(tmp_path, monkeypatch):
    loader = StockListLoader(save_path=str(tmp_path))
    loader._setup()
    file_path = loader.list_path
    file_path.write_text("symbol,name")

    old_time = datetime(2020, 1, 1).timestamp()
    import os

    os.utime(file_path, (old_time, old_time))
    assert loader._is_cache_valid(file_path) is False


def test_stock_loader_get_holidays_fallback(stock_loader):
    holidays = stock_loader._get_holidays("2024-01-01", "2024-01-02")
    assert isinstance(holidays, list)


def test_stock_loader_handle_exception_raises(stock_loader):
    with pytest.raises(DataLoaderError, match="boom"):
        stock_loader._handle_exception(RuntimeError("boom"), "stage")


def test_stock_data_loader_init_invalid_frequency(tmp_path):
    with pytest.raises(ValueError):
        StockDataLoader(save_path=str(tmp_path), frequency="hourly")


def test_stock_loader_load_wrapper(monkeypatch, stock_loader):
    captured = {}

    def fake_load_data(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(stock_loader, "load_data", fake_load_data)
    result = stock_loader.load("000001", "2024-01-01", "2024-01-02", adjust="pre", foo="bar")

    assert result == "ok"
    assert captured["adjust"] == "pre"
    assert captured["foo"] == "bar"


def test_load_batch_with_external_pool(monkeypatch, tmp_path):
    class DummyFuture:
        def __init__(self, fn, *args, **kwargs):
            self.fn = fn
            self.args = args
            self.kwargs = kwargs

        def result(self):
            return self.fn(*self.args, **self.kwargs)

    class DummyPool:
        __module__ = "custom.pool"

        def submit(self, fn, *args, **kwargs):
            return DummyFuture(fn, *args, **kwargs)

    loader = StockDataLoader(save_path=str(tmp_path), thread_pool=DummyPool())
    monkeypatch.setattr(loader, "load", lambda **kwargs: pd.DataFrame({"close": [1]}))

    result = loader.load_batch(["000001"], "2024-01-01", "2024-01-02")
    assert "000001" in result


def test_load_multiple_stocks_external_pool(monkeypatch, tmp_path):
    class DummyFuture:
        def __init__(self, fn, symbol):
            self.fn = fn
            self.symbol = symbol

        def result(self):
            return self.fn(self.symbol)

    class DummyPool:
        __module__ = "custom.pool"

        def submit(self, fn, symbol):
            return DummyFuture(fn, symbol)

    loader = StockDataLoader(save_path=str(tmp_path), thread_pool=DummyPool())
    monkeypatch.setattr(loader, "_load_single_stock_with_cache", lambda sym: {"symbol": sym})

    result = loader.load_multiple_stocks(["000001", "000002"], max_workers=2)
    assert set(result.keys()) == {"000001", "000002"}


def test_load_cache_payload_expired(monkeypatch, tmp_path):
    loader = StockDataLoader(save_path=str(tmp_path))
    cache_file = loader.cache_dir / "foo.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2024-01-01"])),
        "metadata": {
            "cached_time": (datetime.now() - timedelta(days=loader.cache_days + 5)).isoformat(),
        },
    }
    with cache_file.open("wb") as fp:
        pickle.dump(payload, fp)

    assert loader._load_cache_payload(cache_file) is None


def test_save_cache_payload_writes(tmp_path):
    loader = StockDataLoader(save_path=str(tmp_path))
    cache_file = loader.cache_dir / "bar.pkl"
    payload = {
        "data": pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2024-01-01"])),
        "metadata": {"cached_time": datetime.now()},
    }
    loader._save_cache_payload(cache_file, payload)
    assert cache_file.exists()


def test_industry_loader_partial_failure(monkeypatch, tmp_path):
    loader = IndustryLoader(save_path=str(tmp_path))
    loader._industry_map = None
    loader._setup()
    monkeypatch.setattr(loader, "_is_cache_valid", lambda _: False)
    df = pd.DataFrame({"板块代码": ["BK001", "BK002"], "板块名称": ["成功", "失败"]})
    monkeypatch.setattr(loader, "_fetch_raw_data", lambda: df)

    call_count = {"BK001": 0, "BK002": 0}

    def fake_cons(symbol):
        call_count[symbol] += 1
        if symbol == "BK002":
            raise ConnectionError("boom")
        return pd.DataFrame({"代码": ["1"]})

    monkeypatch.setattr("src.data.loader.stock_loader.ak.stock_board_industry_cons_em", fake_cons, raising=False)

    mapping = loader.load_data()
    assert "000001" in mapping or mapping  # 至少有部分结果
    assert call_count["BK002"] == loader.max_retries + 1


def test_industry_loader_retry_api_call_handles_request(tmp_path):
    loader = IndustryLoader(save_path=str(tmp_path))
    loader.max_retries = 1

    with pytest.raises(RequestException):
        loader._retry_api_call(lambda: (_ for _ in ()).throw(RequestException("err")))


def test_stock_list_loader_get_available_symbols_exception(monkeypatch, tmp_path):
    loader = StockListLoader(save_path=str(tmp_path))

    def raise_error():
        raise RuntimeError("boom")

    monkeypatch.setattr(loader, "_fetch_raw_data", raise_error)
    assert loader.get_available_symbols() == []


def test_stock_list_loader_repr_contains_path(tmp_path):
    loader = StockListLoader(save_path=str(tmp_path))
    assert str(loader.save_path) in repr(loader)

