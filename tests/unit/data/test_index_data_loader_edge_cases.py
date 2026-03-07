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


import configparser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from unittest.mock import patch

from src.data.loader.index_loader import IndexDataLoader
from src.infrastructure.error import DataLoaderError


pytestmark = [pytest.mark.timeout(30)]


class _ConcreteIndexLoader(IndexDataLoader):
    """避免 BaseDataLoader 抽象限制的轻量实现"""

    def validate_data(self, data: Any) -> bool:  # pragma: no cover - 简化实现
        return True


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def loader(temp_dir):
    return _ConcreteIndexLoader(save_path=str(temp_dir), max_retries=1, cache_days=1)


def _sample_df():
    dates = pd.date_range("2024-01-01", periods=2)
    return pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "close": [1.5, 2.5],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "volume": [100, 120],
        },
        index=dates,
    )


class TestCreateFromConfig:
    def test_accepts_configparser(self, temp_dir):
        parser = configparser.ConfigParser()
        parser["Index"] = {
            "save_path": str(temp_dir),
            "max_retries": "4",
            "cache_days": "7",
        }
        loader = IndexDataLoader.create_from_config(parser)
        assert loader.max_retries == 4
        assert loader.cache_days == 7
        assert loader.save_path == Path(temp_dir)

    def test_accepts_section_proxy(self, temp_dir):
        parser = configparser.ConfigParser()
        parser["Index"] = {
            "save_path": str(temp_dir),
            "max_retries": "5",
            "cache_days": "8",
        }
        loader = IndexDataLoader.create_from_config(parser["Index"])
        assert loader.max_retries == 5
        assert loader.cache_days == 8

    def test_accepts_dict(self, temp_dir):
        cfg = {
            "Index": {
                "save_path": str(temp_dir),
                "max_retries": 2,
                "cache_days": 9,
            }
        }
        loader = IndexDataLoader.create_from_config(cfg)
        assert loader.save_path == Path(temp_dir)
        assert loader.cache_days == 9

    def test_invalid_integer_raises(self, temp_dir):
        cfg = {
            "Index": {
                "save_path": str(temp_dir),
                "max_retries": "not-int",
            }
        }
        with pytest.raises(DataLoaderError, match="无效"):
            IndexDataLoader.create_from_config(cfg)

    def test_unsupported_config_type(self):
        with pytest.raises(ValueError):
            IndexDataLoader.create_from_config(object())

    def test_get_required_fields(self, loader):
        assert loader.get_required_config_fields() == [
            "save_path",
            "max_retries",
            "cache_days",
        ]


class TestLoadDataValidations:
    def test_invalid_index_code(self, loader):
        with pytest.raises(ValueError):
            loader.load_data("UNKNOWN", "2024-01-01", "2024-01-02")

    def test_start_date_after_end_date(self, loader):
        with pytest.raises(ValueError):
            loader.load_data("HS300", "2024-01-02", "2024-01-01")

    def test_raw_data_empty_raises(self, loader):
        with patch.object(loader, "_fetch_raw_data", return_value=pd.DataFrame()):
            with pytest.raises(DataLoaderError, match="API返回的数据为空"):
                loader.load_data("HS300", "2024-01-01", "2024-01-02")

    def test_processed_data_empty_raises(self, loader):
        df = pd.DataFrame({"date": ["2024-01-01"], "open": [1], "close": [1], "high": [1], "low": [1], "volume": [1]})
        with patch.object(loader, "_fetch_raw_data", return_value=df), patch.object(
            loader, "_process_raw_data", return_value=pd.DataFrame()
        ):
            with pytest.raises(DataLoaderError, match="处理后的指数数据为空"):
                loader.load_data("HS300", "2024-01-01", "2024-01-02")

    def test_data_loader_error_rewrapped(self, loader):
        with patch.object(
            loader,
            "_fetch_raw_data",
            side_effect=DataLoaderError("API返回的数据为空"),
        ), patch("time.sleep"):
            with pytest.raises(DataLoaderError, match="加载指数数据失败"):
                loader.load_data("HS300", "2024-01-01", "2024-01-02")

    def test_connection_error_retries_then_fails(self, loader):
        with patch.object(
            loader,
            "_fetch_raw_data",
            side_effect=ConnectionError("boom"),
        ), patch("time.sleep"):
            with pytest.raises(DataLoaderError, match="超过最大重试次数"):
                loader.load_data("HS300", "2024-01-01", "2024-01-02")

    def test_generic_exception_wrapped(self, loader):
        with patch.object(loader, "_fetch_raw_data", side_effect=RuntimeError("boom")):
            with pytest.raises(DataLoaderError, match="加载指数数据失败"):
                loader.load_data("HS300", "2024-01-01", "2024-01-02")


class TestLoadSingleIndex:
    def test_validation_failure_raises(self, loader):
        df = _sample_df()
        with patch.object(loader, "_fetch_raw_data", return_value=df), patch.object(
            loader, "_process_raw_data", return_value=df
        ), patch.object(loader, "_validate_index_data", return_value=(False, ["bad"])):
            with pytest.raises(DataLoaderError, match="数据验证失败"):
                loader.load_single_index("HS300", "2024-01-01", "2024-01-02", force_refresh=True)

    def test_cache_hit_enriches_payload(self, loader, temp_dir):
        cache_file = loader._get_cache_file_path("HS300")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"data": _sample_df(), "metadata": {}, "cache_info": {}}
        with cache_file.open("wb") as fp:
            import pickle

            pickle.dump(payload, fp)

        result = loader.load_single_index("HS300", force_refresh=False)
        assert result["cache_info"]["is_from_cache"]
        assert result["metadata"]["index_code"] == "000300"
        assert result["metadata"]["performance"]["cache_hit"] is True


class TestLoadMultipleIndexes:
    class _ThreadPool:
        __module__ = "custom_pool"

        def __init__(self, results):
            self._results = results

        def submit(self, func, name):
            class _Future:
                def __init__(self, value):
                    self._value = value

                def result(self):
                    return self._value

            return _Future(self._results[name])

    def test_thread_pool_result_filter_and_warning(self, loader):
        loader.thread_pool = self._ThreadPool(
            {
                "HS300": ("HS300", {"data": _sample_df(), "metadata": {}}),
                "SZ50": "unexpected",
            }
        )
        result = loader.load_multiple_indexes(["HS300", "SZ50"], max_workers=2)
        assert "HS300" in result
        assert "SZ50" not in result


class TestCacheValidation:
    def test_missing_required_columns_returns_false(self, loader, temp_dir):
        cache_file = temp_dir / "bad.csv"
        cache_file.write_text("date,open\n2024-01-01,1\n")
        assert loader._is_cache_valid(cache_file) is False

    def test_empty_cache_returns_false(self, loader, temp_dir):
        cache_file = temp_dir / "empty.csv"
        cache_file.write_text("date,open,high,low,close,volume\n")
        assert loader._is_cache_valid(cache_file) is False

    def test_cache_read_error_returns_false(self, loader, temp_dir, monkeypatch):
        cache_file = temp_dir / "error.csv"
        cache_file.write_text("date,open,high,low,close,volume\n2024-01-01,1,1,1,1,1\n")
        monkeypatch.setattr("pandas.read_csv", lambda *_, **__: (_ for _ in ()).throw(ValueError("boom")))
        assert loader._is_cache_valid(cache_file) is False


class TestRawProcessing:
    def test_fetch_raw_data_empty_raises(self, loader):
        with patch.object(loader, "_retry_api_call", return_value=pd.DataFrame()):
            with pytest.raises(DataLoaderError, match="为空或不存在"):
                loader._fetch_raw_data("HS300", "2024-01-01", "2024-01-02")

    def test_process_raw_data_missing_columns(self, loader):
        df = pd.DataFrame({"date": ["2024-01-01"], "开盘": [1]})
        with pytest.raises(DataLoaderError, match="缺少必要列"):
            loader._process_raw_data(df)

    def test_process_raw_data_invalid_date(self, loader):
        df = pd.DataFrame(
            {
                "日期": ["not-date"],
                "开盘": [1],
                "收盘": [1],
                "最高": [1],
                "最低": [1],
                "成交量": [1],
            }
        )
        with pytest.raises(DataLoaderError, match="日期格式解析失败"):
            loader._process_raw_data(df)


class TestCachePayloadUtils:
    def test_merge_with_cache_missing_columns_returns_new(self, loader, temp_dir):
        csv_path = temp_dir / "cache.csv"
        csv_path.write_text("date,open\n2024-01-01,1\n")
        result = loader._merge_with_cache(csv_path, _sample_df())
        assert_frame_equal(result, _sample_df())

    def test_merge_with_cache_read_error_returns_new(self, loader, temp_dir):
        csv_path = temp_dir / "cache.csv"
        csv_path.write_text("not csv")
        result = loader._merge_with_cache(csv_path, _sample_df())
        assert_frame_equal(result, _sample_df())

    def test_load_cache_payload_invalid_pickle(self, loader, temp_dir):
        cache_file = loader._get_cache_file_path("HS300")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(b"not pickle")
        assert loader._load_cache_payload(cache_file) is None

    def test_save_cache_payload_preserves_cached_time(self, loader, temp_dir):
        cache_file = loader._get_cache_file_path("HS300")
        payload = {
            "data": _sample_df(),
            "metadata": {"cached_time": datetime.now()},
        }
        loader._save_cache_payload(cache_file, payload)
        loaded = loader._load_cache_payload(cache_file)
        assert isinstance(loaded, dict)
        assert "data" in loaded


class TestValidationHelpers:
    def test_validate_index_data_none(self, loader):
        valid, errors = loader._validate_index_data(None)
        assert valid is False
        assert "data is None" in errors

    def test_validate_index_data_non_dataframe(self, loader):
        valid, errors = loader._validate_index_data("not df")
        assert valid is False
        assert "data is not a DataFrame" in errors

    def test_validate_index_data_volume_issues(self, loader):
        df = pd.DataFrame(
            {
                "日期": ["2024-01-01"],
                "开盘": [1],
                "收盘": [1],
                "最高": [1],
                "最低": [1],
                "成交量": ["bad"],
            }
        )
        valid, errors = loader._validate_index_data(df)
        assert valid is False
        assert "non-numeric" in errors[0]


class TestMaintenanceUtilities:
    def test_cleanup_handles_missing_dir(self, loader):
        loader.cache_dir = Path("non-existent-dir")
        loader.cleanup()  # 不应抛异常

    def test_cleanup_logs_failure(self, loader, temp_dir):
        cache_file = temp_dir / "cache.pkl"
        cache_file.write_bytes(b"data")
        loader.cache_dir = temp_dir

        with patch.object(Path, "unlink", side_effect=RuntimeError("boom")):
            loader.cleanup()

    def test_handle_exception_wraps(self, loader):
        with pytest.raises(DataLoaderError, match="boom"):
            loader._handle_exception(RuntimeError("boom"), "load")


class TestNormalizationAndMetadata:
    def test_normalize_data_invalid_input(self, loader):
        with pytest.raises(DataLoaderError):
            loader.normalize_data("not df")

    def test_normalize_data_missing_columns(self, loader):
        df = pd.DataFrame({"open": [1]})
        with pytest.raises(DataLoaderError, match="数据缺少必要列"):
            loader.normalize_data(df)

    def test_normalize_data_success_and_inverse(self, loader, temp_dir):
        df = _sample_df()
        scaler_path = temp_dir / "scaler.pkl"
        normalized = loader.normalize_data(df, scaler_path=scaler_path)
        assert isinstance(normalized, pd.DataFrame)
        assert normalized.shape == df.shape
        restored = loader.normalize_data(normalized, scaler_path=scaler_path, inverse=True)
        assert isinstance(restored, pd.DataFrame)

    def test_get_metadata_contains_flags(self, loader):
        meta = loader.get_metadata()
        assert meta["loader_type"] == "IndexDataLoader"
        assert "supported_indices" in meta

    def test_load_proxies_to_load_data(self, loader):
        with patch.object(loader, "load_data", return_value="ok") as mocked:
            assert loader.load("HS300", "2024-01-01", "2024-01-02") == "ok"
            mocked.assert_called_once()

    def test_validate_returns_false_for_invalid_df(self, loader):
        df = pd.DataFrame({"open": [1]})
        assert loader.validate(df) is False

    def test_save_data_success_and_failures(self, loader, temp_dir):
        df = _sample_df()
        path = temp_dir / "hs300.csv"
        assert loader._save_data(df, path) is True
        assert path.exists()

        bad_df = pd.DataFrame({"open": [1]})
        assert loader._save_data(bad_df, path) is False

        invalid_dates = df.copy()
        invalid_dates.index = ["bad", "value"]
        assert loader._save_data(invalid_dates, path) is False

