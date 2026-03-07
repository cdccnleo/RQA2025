"""
batch_loader 单元测试 - 覆盖真实实现的关键路径
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

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data.loader.batch_loader import BatchDataLoader


@pytest.fixture
def temp_dir(tmp_path) -> Path:
    return tmp_path


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))


class DummyFuture:
    def __init__(self, value: Any):
        self._value = value

    def result(self):
        return self._value


class DummyExecutor:
    def __init__(self, *args, **kwargs):
        self.submitted = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def submit(self, fn, sym):
        value = fn(sym)
        self.submitted.append(sym)
        return DummyFuture(value)


def test_init_uses_default_stock_loader(temp_dir, monkeypatch):
    mock_cls = MagicMock()
    monkeypatch.setattr("src.data.loader.batch_loader.StockDataLoader", mock_cls)

    loader = BatchDataLoader(save_path=str(temp_dir), max_retries=5, cache_days=7, timeout=15, max_workers=2)

    mock_cls.assert_called_once_with(save_path=str(temp_dir), max_retries=5, cache_days=7, timeout=15)
    assert loader.save_path == temp_dir
    assert loader.max_workers == 2
    assert loader.stock_loader is mock_cls.return_value
    assert loader.save_path.exists()


def test_init_uses_provided_stock_loader(temp_dir):
    custom_loader = MagicMock()
    loader = BatchDataLoader(save_path=str(temp_dir), stock_loader=custom_loader)

    assert loader.stock_loader is custom_loader


def test_load_batch_returns_empty_for_no_symbols(temp_dir):
    loader = BatchDataLoader(save_path=str(temp_dir), stock_loader=MagicMock())
    assert loader.load_batch([], "2024-01-01", "2024-01-02") == {}


def test_load_batch_invokes_stock_loader(monkeypatch, temp_dir, sample_df):
    stock_loader = MagicMock()
    stock_loader.load.side_effect = [sample_df, None]
    loader = BatchDataLoader(save_path=str(temp_dir), stock_loader=stock_loader, max_workers=2)

    monkeypatch.setattr("src.data.loader.batch_loader.ThreadPoolExecutor", DummyExecutor)

    result = loader.load_batch(["AAA", "BBB"], "2024-01-01", "2024-01-02", max_workers=1)

    assert_frame_equal(result["AAA"], sample_df)
    assert result["BBB"] is None
    assert stock_loader.load.call_count == 2


def test_load_batch_uses_default_worker_count(monkeypatch, temp_dir, sample_df):
    stock_loader = MagicMock()
    stock_loader.load.return_value = sample_df
    loader = BatchDataLoader(save_path=str(temp_dir), stock_loader=stock_loader, max_workers=3)

    captured = {}

    class CapturingExecutor(DummyExecutor):
        def __init__(self, *args, **kwargs):
            captured["max_workers"] = kwargs.get("max_workers")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("src.data.loader.batch_loader.ThreadPoolExecutor", CapturingExecutor)

    loader.load_batch(["AAA"], "2024-01-01", "2024-01-02")
    assert captured["max_workers"] == 3


def test_load_alias_calls_load_batch(monkeypatch, temp_dir):
    loader = BatchDataLoader(save_path=str(temp_dir), stock_loader=MagicMock())
    spy = MagicMock(return_value={"AAA": None})
    monkeypatch.setattr(loader, "load_batch", spy)

    loader.load(["AAA"], "2024-01-01", "2024-01-02")
    spy.assert_called_once()


def test_validate_rules():
    loader = BatchDataLoader(stock_loader=MagicMock())
    assert loader.validate({"AAA": pd.DataFrame()}) is True
    assert loader.validate({"AAA": None}) is True
    assert loader.validate({"AAA": 123}) is False
    assert loader.validate(123) is False


def test_get_metadata_reports_config():
    loader = BatchDataLoader(max_workers=5, timeout=20, stock_loader=MagicMock())
    meta = loader.get_metadata()
    assert meta["loader_type"] == "BatchDataLoader"
    assert meta["max_workers"] == 5
    assert meta["timeout"] == 20

