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
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from requests import RequestException

from src.infrastructure.utils.exceptions import DataLoaderError
from src.data.loader.stock_loader import IndustryLoader, StockListLoader
import src.data.loader.stock_loader as stock_loader_module


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


def test_industry_loader_uses_cache_when_valid(temp_dir):
    loader = IndustryLoader(save_path=str(temp_dir))
    loader._industry_map = None
    loader.debug_mode = False
    cache_file = loader.industry_map_path
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"symbol": ["000001"], "industry": ["金融"]})
    df.to_csv(cache_file, index=False, encoding="utf-8")
    os.utime(cache_file, None)

    mapping = loader.load_data()
    assert mapping["000001"] == "金融"


def test_industry_loader_fetches_and_builds_mapping(temp_dir, monkeypatch):
    loader = IndustryLoader(save_path=str(temp_dir))
    loader._industry_map = None
    loader.debug_mode = False

    industry_df = pd.DataFrame(
        {"板块代码": ["BK001"], "板块名称": ["新能源"]}
    )
    component_df = pd.DataFrame({"代码": ["1", "2"]})

    monkeypatch.setattr(loader, "_fetch_raw_data", lambda: industry_df)
    monkeypatch.setattr(
        "src.data.loader.stock_loader.ak.stock_board_industry_cons_em",
        lambda symbol: component_df if symbol == "BK001" else pd.DataFrame(),
        raising=False,
    )

    mapping = loader.load_data()
    assert mapping["000001"] == "新能源"
    assert mapping["000002"] == "新能源"


def test_industry_loader_calculate_concentration(monkeypatch, temp_dir):
    loader = IndustryLoader(save_path=str(temp_dir))
    loader.debug_mode = False
    loader._industry_map = {}

    components = pd.DataFrame({"symbol": ["000001", "000002"]})
    monkeypatch.setattr(loader, "_get_industry_components", lambda code: components)

    date_index = pd.date_range("2024-01-01", periods=12)

    def fake_load(symbol, start, end):
        base = 1 if symbol == "000001" else 2
        values = [base + i for i in range(len(date_index))]
        return pd.Series(values, index=date_index)

    monkeypatch.setattr(loader, "_load_stock_data", fake_load)

    result = loader.calculate_industry_concentration("新能源", window=12)
    assert not result.empty
    assert {"CR4", "CR8"} == set(result.columns)


def test_industry_loader_get_industry_unknown_defaults(temp_dir):
    loader = IndustryLoader(save_path=str(temp_dir))
    loader._industry_map = {}
    loader.debug_mode = False
    assert loader.get_industry("300000") == "行业未知"


def test_industry_loader_concentration_window_too_small(temp_dir):
    loader = IndustryLoader(save_path=str(temp_dir))
    with pytest.raises(ValueError):
        loader.calculate_industry_concentration("新能源", window=5)


def test_industry_loader_components_missing_raises(temp_dir, monkeypatch):
    loader = IndustryLoader(save_path=str(temp_dir))
    loader._industry_map = {}
    monkeypatch.setattr(loader, "_get_industry_components", lambda code: pd.DataFrame())
    with pytest.raises(DataLoaderError, match="未找到行业"):
        loader.calculate_industry_concentration("未知行业", window=12)


def test_industry_loader_partial_failure_logs_warning(temp_dir, monkeypatch, caplog):
    loader = IndustryLoader(save_path=str(temp_dir), max_retries=1)
    loader._industry_map = None
    loader.debug_mode = False

    industry_df = pd.DataFrame(
        {"板块代码": ["BK1", "BK2"], "板块名称": ["失败行业", "成功行业"]}
    )
    monkeypatch.setattr(loader, "_fetch_raw_data", lambda: industry_df)

    def fake_cons(symbol):
        if symbol == "BK1":
            raise ConnectionError("boom")
        return pd.DataFrame()  # 成功调用但无成分股

    monkeypatch.setattr(
        "src.data.loader.stock_loader.ak.stock_board_industry_cons_em",
        fake_cons,
        raising=False,
    )

    caplog.set_level("WARNING")
    mapping = loader.load_data()
    assert mapping == {}
    assert "部分行业数据获取失败" in caplog.text


def test_industry_loader_all_failures_raise(temp_dir, monkeypatch):
    loader = IndustryLoader(save_path=str(temp_dir), max_retries=1)
    loader._industry_map = None

    industry_df = pd.DataFrame({"板块代码": ["BK1"], "板块名称": ["失败行业"]})
    monkeypatch.setattr(loader, "_fetch_raw_data", lambda: industry_df)

    def always_fail(*_, **__):
        raise ConnectionError("boom")

    monkeypatch.setattr(
        "src.data.loader.stock_loader.ak.stock_board_industry_cons_em",
        always_fail,
        raising=False,
    )

    with pytest.raises(DataLoaderError, match="无法获取任何行业映射数据"):
        loader.load_data()


def test_industry_loader_get_industry_debug_mode_raises(temp_dir):
    loader = IndustryLoader(save_path=str(temp_dir))
    loader._industry_map = {}
    loader.debug_mode = True
    with pytest.raises(DataLoaderError, match="获取行业数据失败"):
        loader.get_industry("300001")


def test_stock_list_loader_uses_cache_when_valid(temp_dir):
    loader = StockListLoader(save_path=temp_dir, cache_days=10)
    cache_file = loader.list_path
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"股票代码": ["000001"], "股票名称": ["平安银行"]})
    df.to_csv(cache_file, index=False, encoding="utf-8")
    os.utime(cache_file, None)

    loaded = loader.load_data()
    assert "股票代码" in loaded.columns


def test_stock_list_loader_fetches_and_saves(monkeypatch, temp_dir):
    loader = StockListLoader(save_path=temp_dir, cache_days=0)
    loader.list_path.unlink(missing_ok=True)

    raw_df = pd.DataFrame({"code": ["000001"], "name": ["平安银行"]})
    monkeypatch.setattr(loader, "_fetch_raw_data", lambda: raw_df)

    loaded = loader.load_data()
    assert loaded.iloc[0]["股票名称"] == "平安银行"
    assert loader.list_path.exists()


def test_stock_list_loader_get_available_symbols_from_symbol_column(monkeypatch, temp_dir):
    loader = StockListLoader(save_path=temp_dir)
    df = pd.DataFrame({"symbol": ["000001", "000002"]})
    monkeypatch.setattr(loader, "_fetch_raw_data", lambda: df)
    assert loader.get_available_symbols() == ["000001", "000002"]


def test_stock_list_loader_load_data_empty_response_raises(monkeypatch, temp_dir):
    loader = StockListLoader(save_path=temp_dir)
    loader.list_path.unlink(missing_ok=True)
    monkeypatch.setattr(loader, "_fetch_raw_data", lambda: pd.DataFrame())

    with pytest.raises(DataLoaderError, match="股票列表为空"):
        loader.load_data()


def test_stock_list_loader_load_data_exception_wrapped(monkeypatch, temp_dir):
    loader = StockListLoader(save_path=temp_dir)
    loader.list_path.unlink(missing_ok=True)

    def boom():
        raise RuntimeError("downstream failure")

    monkeypatch.setattr(loader, "_fetch_raw_data", boom)
    with pytest.raises(DataLoaderError, match="加载股票列表失败"):
        loader.load_data()


def test_stock_list_loader_cache_validation_handles_error(monkeypatch, temp_dir):
    loader = StockListLoader(save_path=temp_dir)
    loader.list_path.parent.mkdir(parents=True, exist_ok=True)
    loader.list_path.write_text("code,name\n000001,平安银行\n", encoding="utf-8")

    def fake_getmtime(_):
        raise OSError("stat failed")

    monkeypatch.setattr("os.path.getmtime", fake_getmtime)
    assert loader._is_cache_valid(loader.list_path) is False


def test_stock_list_loader_retry_api_call_exhausts(monkeypatch, temp_dir):
    loader = StockListLoader(save_path=temp_dir, max_retries=2)
    monkeypatch.setattr("src.data.loader.stock_loader.time.sleep", lambda *_: None)

    def failing(*args, **kwargs):
        raise RequestException("timeout")

    with pytest.raises(RequestException):
        loader._retry_api_call(failing)


def test_stock_list_loader_get_available_symbols_from_code_column(monkeypatch, temp_dir):
    loader = StockListLoader(save_path=temp_dir)
    df = pd.DataFrame({"code": ["000003", "000004"]})
    monkeypatch.setattr(loader, "_fetch_raw_data", lambda: df)
    assert loader.get_available_symbols() == ["000003", "000004"]


def test_stock_list_loader_get_available_symbols_handles_exception(monkeypatch, temp_dir, caplog):
    loader = StockListLoader(save_path=temp_dir)

    def boom():
        raise RuntimeError("network")

    monkeypatch.setattr(loader, "_fetch_raw_data", boom)
    caplog.set_level("ERROR")
    assert loader.get_available_symbols() == []
    assert "获取可用股票代码失败" in caplog.text


def test_stock_list_loader_repr_contains_core_fields(temp_dir):
    loader = StockListLoader(save_path=temp_dir, max_retries=5, cache_days=1)
    loader.frequency = "weekly"
    text = repr(loader)
    assert "StockDataLoader" in text
    assert "max_retries=5" in text
    assert "frequency=weekly" in text


def test_stock_list_loader_fetch_raw_data_uses_retry(monkeypatch, temp_dir):
    loader = StockListLoader(save_path=temp_dir)
    captured = {}

    def fake_retry(func, *args, **kwargs):
        captured["func"] = func
        return pd.DataFrame({"code": ["000001"], "name": ["平安银行"]})

    monkeypatch.setattr(loader, "_retry_api_call", fake_retry)
    result = loader._fetch_raw_data()
    assert "code" in result.columns
    assert captured["func"] is stock_loader_module.ak.stock_info_a_code_name

