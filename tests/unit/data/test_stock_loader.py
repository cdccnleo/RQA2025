#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试股票数据加载器

测试目标：提升stock_loader.py的覆盖率到100%
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


import builtins
import configparser
import logging
import os
import pickle
import sys
from types import SimpleNamespace

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
from requests import RequestException

import src.data.loader.stock_loader as stock_loader_module
from src.infrastructure.utils.exceptions import DataLoaderError
from src.data.loader.stock_loader import StockDataLoader


class TestStockDataLoader:
    """测试股票数据加载器"""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """创建临时目录"""
        return tmp_path

    @pytest.fixture
    def loader(self, temp_dir):
        """创建股票数据加载器实例"""
        return StockDataLoader(
            save_path=str(temp_dir),
            max_retries=3,
            cache_days=30,
            frequency='daily',
            adjust_type='none'
        )

    def test_loader_initialization(self, temp_dir):
        """测试加载器初始化"""
        loader = StockDataLoader(save_path=str(temp_dir))

        assert loader.save_path == Path(temp_dir)
        assert loader.max_retries == 3
        assert loader.cache_days == 30
        assert loader.frequency == 'daily'
        assert loader.adjust_type == 'none'

    def test_loader_initialization_invalid_save_path(self):
        """测试无效保存路径的初始化"""
        with pytest.raises(ValueError, match="save_path不能为空"):
            StockDataLoader(save_path="")

    def test_loader_initialization_invalid_max_retries(self, temp_dir):
        """测试无效最大重试次数的初始化"""
        with pytest.raises(ValueError, match="max_retries必须大于0"):
            StockDataLoader(save_path=str(temp_dir), max_retries=0)

    def test_create_from_config_dict(self, temp_dir):
        """测试从字典配置创建加载器"""
        config = {
            "save_path": str(temp_dir),
            "max_retries": 5,
            "cache_days": 60,
            "frequency": "weekly",
            "adjust_type": "pre"
        }

        loader = StockDataLoader.create_from_config(config)

        assert loader.save_path == Path(temp_dir)
        assert loader.max_retries == 5
        assert loader.cache_days == 60
        assert loader.frequency == "weekly"
        assert loader.adjust_type == "pre"

    def test_get_required_config_fields(self, loader):
        """测试获取必需配置字段"""
        fields = loader.get_required_config_fields()

        assert isinstance(fields, list)
        assert "save_path" in fields

    def test_validate_config_valid(self, loader):
        """测试验证有效配置"""
        config = {
            "save_path": "/tmp/data",
            "max_retries": 3,
            "cache_days": 30
        }

        result = loader.validate_config(config)
        assert result == True

    def test_validate_config_invalid(self, loader):
        """测试验证无效配置"""
        invalid_configs = [
            {},  # 空配置
            {"max_retries": 3},  # 缺少save_path
            {"save_path": "", "max_retries": 3},  # 空save_path
        ]

        for config in invalid_configs:
            result = loader.validate_config(config)
            assert result == False

    def test_get_metadata(self, loader):
        """测试获取元数据"""
        metadata = loader.get_metadata()

        assert isinstance(metadata, dict)
        assert "loader_type" in metadata
        assert "supported_data_types" in metadata
        assert "version" in metadata

    @patch('src.data.loader.stock_loader.ak.stock_zh_a_hist')
    def test_load_data_basic(self, mock_ak, loader):
        """测试基本数据加载"""
        # 创建模拟数据
        mock_data = pd.DataFrame({
            '日期': ['2023-01-01', '2023-01-02', '2023-01-03'],
            '开盘': [10.0, 10.5, 11.0],
            '收盘': [10.5, 11.0, 11.5],
            '最高': [11.0, 11.5, 12.0],
            '最低': [9.5, 10.0, 10.5],
            '成交量': [1000, 1200, 1100]
        })
        mock_ak.return_value = mock_data

        result = loader.load_data(
            symbol="000001",
            start_date="2023-01-01",
            end_date="2023-01-03"
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        mock_ak.assert_called_once()

    @patch('src.data.loader.stock_loader.ak.stock_zh_a_hist')
    def test_load_data_with_adjustment(self, mock_ak, loader):
        """测试带复权的数据加载"""
        loader.adjust_type = "pre"

        mock_data = pd.DataFrame({
            '日期': ['2023-01-01'],
            '开盘': [10.0],
            '收盘': [10.5],
            '最高': [11.0],
            '最低': [9.5],
            '成交量': [1000]
        })
        mock_ak.return_value = mock_data

        result = loader.load_data(
            symbol="000001",
            start_date="2023-01-01",
            end_date="2023-01-01"
        )

        assert isinstance(result, pd.DataFrame)
        mock_ak.assert_called_once()

    def test_validate_data_valid(self, loader):
        """测试验证有效数据"""
        valid_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'open': [10.0, 10.5],
            'close': [10.5, 11.0],
            'high': [11.0, 11.5],
            'low': [9.5, 10.0],
            'volume': [1000, 1200]
        })

        result = loader.validate_data(valid_data)
        assert result == True

    def test_validate_data_invalid(self, loader):
        """测试验证无效数据"""
        invalid_data_list = [
            None,  # None数据
            [],  # 空列表
            pd.DataFrame(),  # 空DataFrame
            pd.DataFrame({'invalid': [1, 2, 3]}),  # 缺少必需列
        ]

        for invalid_data in invalid_data_list:
            result = loader.validate_data(invalid_data)
            assert result == False

    def test_get_file_path(self, loader):
        """测试获取文件路径"""
        file_path = loader._get_file_path(
            symbol="000001",
            start_date="2023-01-01",
            end_date="2023-01-03"
        )

        assert isinstance(file_path, Path)
        assert "000001" in str(file_path)
        assert "2023-01-01" in str(file_path)
        assert "2023-01-03" in str(file_path)

    def test_is_cache_valid_file_exists_recent(self, loader, temp_dir):
        """测试缓存验证 - 文件存在且较新"""
        # 创建一个最近的文件
        test_file = temp_dir / "test_stock_2023-01-01_2023-01-03.csv"
        test_file.write_text("test data")

        # 修改文件的修改时间为最近
        import time
        current_time = time.time()
        os.utime(test_file, (current_time, current_time))

        result = loader._is_cache_valid(test_file)
        assert result == True

    def test_is_cache_valid_file_exists_old(self, loader, temp_dir):
        """测试缓存验证 - 文件存在但较旧"""
        # 创建一个旧文件
        test_file = temp_dir / "test_stock_2023-01-01_2023-01-03.csv"
        test_file.write_text("test data")

        # 修改文件的修改时间为很旧
        import time
        old_days = loader.cache_days + 5
        old_time = time.time() - old_days * 24 * 3600
        os.utime(test_file, (old_time, old_time))

        result = loader._is_cache_valid(test_file)
        assert result == False

    def test_is_cache_valid_file_not_exists(self, loader, temp_dir):
        """测试缓存验证 - 文件不存在"""
        test_file = temp_dir / "nonexistent_file.csv"

        result = loader._is_cache_valid(test_file)
        assert result == False

    def test_validate_volume_valid(self, loader):
        """测试成交量验证 - 有效数据"""
        valid_df = pd.DataFrame({
            'volume': [1000, 1200, 1100, 1300]
        })

        result = loader._validate_volume(valid_df)
        assert result == True

    def test_validate_volume_invalid(self, loader):
        """测试成交量验证 - 无效数据"""
        invalid_dfs = [
            pd.DataFrame({'volume': [0, 0, 0]}),  # 全零成交量
            pd.DataFrame({'volume': [-100, 100]}),  # 负成交量
            pd.DataFrame({'other': [1, 2, 3]}),  # 缺少volume列
        ]

        for invalid_df in invalid_dfs:
            result = loader._validate_volume(invalid_df)
            assert result == False

    def test_create_from_config_invalid_type(self):
        """测试 create_from_config 不支持的类型"""
        with pytest.raises(ValueError):
            StockDataLoader.create_from_config(["unexpected"])

    def test_create_from_config_invalid_integer(self, temp_dir):
        """测试 create_from_config 字段为非法整数时抛出异常"""
        config = {"Stock": {"save_path": str(temp_dir), "max_retries": "abc"}}
        with pytest.raises(DataLoaderError):
            StockDataLoader.create_from_config(config)

    def test_create_from_config_with_configparser(self, temp_dir):
        """测试 create_from_config 支持 ConfigParser"""
        parser = configparser.ConfigParser()
        parser["Stock"] = {
            "save_path": str(temp_dir),
            "max_retries": "4",
            "cache_days": "15",
            "frequency": "weekly",
            "adjust_type": "pre",
        }
        loader = StockDataLoader.create_from_config(parser)
        assert loader.max_retries == 4
        assert loader.cache_days == 15
        assert loader.frequency == "weekly"
        assert loader.adjust_type == "pre"

    def test_create_from_config_with_section_proxy(self, temp_dir):
        """测试 create_from_config 支持 SectionProxy"""
        parser = configparser.ConfigParser()
        parser.add_section("Stock")
        parser.set("Stock", "save_path", str(temp_dir))
        parser.set("Stock", "max_retries", "6")
        parser.set("Stock", "cache_days", "10")
        loader = StockDataLoader.create_from_config(parser["Stock"])
        assert loader.max_retries == 6
        assert loader.cache_days == 10

    def test_load_data_impl_uses_cache(self, temp_dir, monkeypatch):
        """测试 load_data_impl 命中缓存"""
        loader = StockDataLoader(save_path=str(temp_dir))
        file_path = loader._get_file_path("600000", "2024-01-01", "2024-01-02")
        df = pd.DataFrame({"close": [10.0]}, index=pd.date_range("2024-01-01", periods=1))
        df.to_csv(file_path, encoding="utf-8")

        def should_not_call(*args, **kwargs):
            raise AssertionError("should not fetch raw data when cache valid")

        monkeypatch.setattr(loader, "_fetch_raw_data", should_not_call)
        result = loader._load_data_impl("600000", "2024-01-01", "2024-01-02")
        assert result.shape == df.shape
        assert result["close"].iloc[0] == pytest.approx(10.0)

    def test_load_data_impl_retries_then_raises(self, temp_dir, monkeypatch):
        """测试 load_data_impl 在多次重试后抛出异常"""
        loader = StockDataLoader(save_path=str(temp_dir), max_retries=1)
        monkeypatch.setattr(loader, "_is_cache_valid", lambda *a, **k: False)
        monkeypatch.setattr(loader, "_fetch_raw_data", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(DataLoaderError):
            loader._load_data_impl("000001", "2024-01-01", "2024-01-02", adjust="none")

    def test_load_data_impl_invalid_date_range(self, loader):
        """测试 load_data_impl 在开始日期大于结束日期时抛出异常"""
        with pytest.raises(ValueError, match="开始日期不能大于结束日期"):
            loader._load_data_impl("000001", "2024-02-10", "2024-01-10")

    def test_load_data_requires_all_arguments(self, loader):
        """测试 load_data 缺少必需参数时抛出异常"""
        with pytest.raises(ValueError):
            loader.load_data(symbol="000001", start_date="2024-01-01")

    def test_load_batch_returns_empty_for_no_symbols(self, loader):
        """测试 load_batch 在空列表时返回空字典"""
        result = loader.load_batch([], "2024-01-01", "2024-01-02")
        assert result == {}

    def test_load_single_stock_returns_cached_payload(self, temp_dir, monkeypatch):
        """测试 load_single_stock 缓存命中"""
        loader = StockDataLoader(save_path=str(temp_dir))
        payload = {"metadata": {}, "cache_info": {}}
        monkeypatch.setattr(loader, "_load_cache_payload", lambda *a, **k: payload)
        result = loader.load_single_stock("000001")
        assert result["metadata"]["symbol"] == "000001"
        assert result["metadata"]["performance"]["cache_hit"] is True
        assert result["cache_info"]["is_from_cache"] is True

    def test_load_single_stock_fetches_and_caches(self, temp_dir, monkeypatch):
        """测试 load_single_stock 在没有缓存时加载并缓存数据"""
        loader = StockDataLoader(save_path=str(temp_dir))
        data = pd.DataFrame({"close": [1.0]}, index=pd.date_range("2024-01-01", periods=1))
        monkeypatch.setattr(loader, "_load_cache_payload", lambda *a, **k: None)
        monkeypatch.setattr(loader, "_load_data_impl", lambda *a, **k: data)
        monkeypatch.setattr(loader, "_validate_data", lambda d: (True, []))

        captured = {}

        def fake_save(path, payload):
            captured["payload"] = payload

        monkeypatch.setattr(loader, "_save_cache_payload", fake_save)
        result = loader.load_single_stock("000002", adjust="pre", force_refresh=True)
        assert result["metadata"]["symbol"] == "000002"
        assert captured["payload"]["cache_info"]["is_from_cache"] is False

    def test_load_single_stock_invalid_data_raises(self, loader, monkeypatch):
        """测试 load_single_stock 在数据校验失败时抛出异常"""
        invalid_df = pd.DataFrame({"close": [1.0, 2.0]})
        monkeypatch.setattr(loader, "_load_data_impl", lambda *a, **k: invalid_df)
        with pytest.raises(DataLoaderError, match="数据验证失败"):
            loader.load_single_stock("000777", "2024-01-01", "2024-01-05", adjust="hfq")

    def test_load_single_stock_with_cache_delegates(self, loader, monkeypatch):
        """测试 _load_single_stock_with_cache 直接调用 load_single_stock"""
        captured = {}

        def fake_loader(symbol, start_date=None, end_date=None, adjust=None, force_refresh=False):
            captured["args"] = (symbol, start_date, end_date, adjust, force_refresh)
            return {"ok": True}

        monkeypatch.setattr(loader, "load_single_stock", fake_loader)
        result = loader._load_single_stock_with_cache(
            "000888",
            start_date="2024-01-01",
            end_date="2024-01-02",
            adjust="qfq",
            force_refresh=True,
        )

        assert result == {"ok": True}
        assert captured["args"] == ("000888", "2024-01-01", "2024-01-02", "qfq", True)

    def test_load_cache_payload_expires(self, temp_dir):
        """测试缓存过期时 _load_cache_payload 返回 None"""
        loader = StockDataLoader(save_path=str(temp_dir))
        cache_file = loader._get_cache_file_path("000003")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "data": pd.DataFrame({"close": [1.0]}, index=pd.date_range("2024-01-01", periods=1)),
            "metadata": {"cached_time": (datetime.now() - timedelta(days=loader.cache_days + 1)).isoformat()},
        }
        with cache_file.open("wb") as fp:
            pickle.dump(payload, fp)
        assert loader._load_cache_payload(cache_file) is None

    def test_process_raw_data_normalizes_columns(self, loader):
        """测试 _process_raw_data 处理中文列并补齐高低价"""
        raw = pd.DataFrame(
            {
                "日期": ["2024-01-01"],
                "开盘": [10],
                "收盘": [12],
                "成交量": ["2000"],
            }
        )
        result = loader._process_raw_data(raw)
        for column in ["open", "high", "low", "close", "volume", "is_trading_day"]:
            assert column in result.columns
        assert result.index.equals(pd.DatetimeIndex(["2024-01-01"]))
        assert result.loc[result.index[0], "high"] == 12
        assert result.loc[result.index[0], "low"] == 10

    def test_process_raw_data_missing_required_raises(self, loader):
        """测试缺少必要列时抛出 DataLoaderError"""
        raw = pd.DataFrame({"日期": ["2024-01-01"], "开盘": [10]})
        with pytest.raises(DataLoaderError, match="必要列"):
            loader._process_raw_data(raw)

    def test_validate_data_detects_negative_volume(self, loader):
        """测试 _validate_data 能识别负数成交量"""
        frame = pd.DataFrame(
            {
                "open": [1.0, 1.1],
                "high": [1.2, 1.3],
                "low": [0.9, 1.0],
                "close": [1.05, 1.15],
                "volume": [1000, -5],
            }
        )
        valid, errors = loader._validate_data(frame)
        assert valid is False
        assert any("negative" in err for err in errors)

    def test_load_data_impl_empty_cache_raises(self, loader, monkeypatch):
        """测试缓存文件为空时 _load_data_impl 抛出异常"""
        monkeypatch.setattr(loader, "_is_cache_valid", lambda *_: True)

        def fake_read_csv(*args, **kwargs):
            return pd.DataFrame(columns=["open", "close"])

        monkeypatch.setattr("src.data.loader.stock_loader.pd.read_csv", fake_read_csv)

        with pytest.raises(DataLoaderError, match="缓存数据为空"):
            loader._load_data_impl(
                symbol="000001",
                start_date="2024-01-01",
                end_date="2024-01-02",
                adjust="hfq",
                force_refresh=False,
            )

    def test_retry_api_call_eventual_success(self, loader, monkeypatch):
        """测试 _retry_api_call 在失败后成功"""
        attempts = {"count": 0}

        def flaky(*args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise RequestException("timeout")
            return pd.DataFrame({"close": [1]})

        monkeypatch.setattr("src.data.loader.stock_loader.time.sleep", lambda *_: None)
        result = loader._retry_api_call(flaky)
        assert isinstance(result, pd.DataFrame)
        assert attempts["count"] == 2

    def test_retry_api_call_request_exception_exhausted(self, loader, monkeypatch):
        """测试 _retry_api_call 在持续网络异常时抛出 DataLoaderError"""
        monkeypatch.setattr("src.data.loader.stock_loader.time.sleep", lambda *_: None)

        def always_fail(*args, **kwargs):
            raise RequestException("timeout")

        with pytest.raises(DataLoaderError, match="timeout"):
            loader._retry_api_call(always_fail)

    def test_load_batch_uses_external_thread_pool(self, loader, monkeypatch):
        """测试 load_batch 能够复用外部线程池"""
        data = pd.DataFrame({"close": [1]}, index=pd.date_range("2024-01-01", periods=1))
        monkeypatch.setattr(loader, "load", lambda **kwargs: data)

        class DummyFuture:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        class DummyPool:
            __module__ = "custom_pool"

            def submit(self, fn, sym):
                return DummyFuture(fn(sym))

        loader.thread_pool = DummyPool()
        results = loader.load_batch(["AAA"], "2024-01-01", "2024-01-02")
        assert isinstance(results["AAA"], pd.DataFrame)

    def test_load_batch_uses_default_executor(self, loader, monkeypatch):
        """测试 load_batch 默认线程池路径"""
        monkeypatch.setattr(loader, "load", lambda **kwargs: pd.DataFrame({"close": [1]}))

        class DummyFuture:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        class DummyPool:
            def __init__(self, max_workers=None):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, sym):
                return DummyFuture(fn(sym))

        monkeypatch.setattr(
            "src.data.loader.stock_loader.concurrent.futures.ThreadPoolExecutor",
            DummyPool,
        )
        monkeypatch.setattr(
            "src.data.loader.stock_loader.concurrent.futures.as_completed",
            lambda futures: futures,
        )

        results = loader.load_batch(["AAA", "BBB"], "2024-01-01", "2024-01-02")
        assert set(results.keys()) == {"AAA", "BBB"}

    def test_load_multiple_stocks_external_pool_warning(self, loader, monkeypatch, caplog):
        """测试 load_multiple_stocks 使用外部线程池并处理异常结果"""
        monkeypatch.setattr(
            loader,
            "_load_single_stock_with_cache",
            lambda sym: {"symbol": sym},
        )

        class DummyFuture:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        class MixedPool:
            __module__ = "custom_pool"

            def submit(self, fn, sym):
                if sym == "good":
                    return DummyFuture(fn(sym))
                return DummyFuture("unexpected")

        loader.thread_pool = MixedPool()
        caplog.set_level(logging.WARNING)
        results = loader.load_multiple_stocks(["good", "bad"], max_workers=2)
        assert results["good"]["symbol"] == "good"
        assert "线程池返回非预期结果" in caplog.text

    def test_load_multiple_stocks_default_executor_handles_errors(self, loader, monkeypatch):
        """测试 load_multiple_stocks 默认执行器处理异常"""
        def fake_load(sym):
            if sym == "bad":
                raise RuntimeError("boom")
            return {"symbol": sym}

        monkeypatch.setattr(loader, "_load_single_stock_with_cache", fake_load)
        results = loader.load_multiple_stocks(["good", "bad"])
        assert results["good"]["symbol"] == "good"
        assert "error" in results["bad"]

    def test_fetch_raw_data_fallbacks_to_hist(self, temp_dir, monkeypatch):
        """测试 _fetch_raw_data 在主接口失败时回退"""
        loader = StockDataLoader(save_path=str(temp_dir))

        def daily_fail(*args, **kwargs):
            raise DataLoaderError("daily fail")

        hist_df = pd.DataFrame(
            {
                "日期": ["2024-01-01"],
                "开盘": [10],
                "收盘": [11],
                "最高": [12],
                "最低": [9],
                "成交量": [1000],
            }
        )

        def hist_success(*args, **kwargs):
            return hist_df

        monkeypatch.setattr(stock_loader_module.ak, "stock_zh_a_daily", daily_fail, raising=False)
        monkeypatch.setattr(stock_loader_module.ak, "stock_zh_a_hist", hist_success, raising=False)
        monkeypatch.setattr(loader, "_retry_api_call", lambda func, *a, **k: func(*a, **k))

        result = loader._fetch_raw_data("000001", "2024-01-01", "2024-01-05", "hfq")
        assert result.equals(hist_df)

    def test_fetch_raw_data_no_available_source_raises(self, temp_dir, monkeypatch):
        """测试当所有 akshare 接口不可用时抛出异常"""
        loader = StockDataLoader(save_path=str(temp_dir))
        monkeypatch.setattr(stock_loader_module.ak, "stock_zh_a_daily", None, raising=False)
        monkeypatch.setattr(stock_loader_module.ak, "stock_zh_a_hist", None, raising=False)

        with pytest.raises(DataLoaderError, match="未找到可用的 akshare 股票行情函数"):
            loader._fetch_raw_data("000001", "2024-01-01", "2024-01-05", "hfq")

    def test_check_cache_handles_missing_and_read_errors(self, temp_dir, monkeypatch):
        """测试 _check_cache 缓存缺失与读取异常路径"""
        loader = StockDataLoader(save_path=str(temp_dir))
        missing = temp_dir / "missing.csv"
        exists, df = loader._check_cache(missing)
        assert exists is False and df is None

        bad_file = temp_dir / "bad.csv"
        bad_file.write_text("bad")

        def raise_error(*args, **kwargs):
            raise ValueError("bad")

        monkeypatch.setattr(stock_loader_module.pd, "read_csv", raise_error, raising=False)
        exists, df = loader._check_cache(bad_file)
        assert exists is False and df is None

    def test_check_cache_zero_size_and_success(self, temp_dir, monkeypatch):
        """测试 _check_cache 处理空文件与有效缓存"""
        loader = StockDataLoader(save_path=str(temp_dir))
        zero_file = temp_dir / "zero.csv"
        zero_file.write_text("")
        exists, df = loader._check_cache(zero_file)
        assert exists is False and df is None

        valid = temp_dir / "valid.csv"
        pd.DataFrame({"close": [1.0]}, index=pd.date_range("2024-01-01", periods=1)).to_csv(valid, encoding="utf-8")
        monkeypatch.setattr(loader, "_is_cache_valid", lambda *a, **k: True)
        exists, df = loader._check_cache(valid)
        assert exists is True
        assert isinstance(df, pd.DataFrame)

    def test_get_holidays_market_calendar(self, loader, monkeypatch):
        """测试 _get_holidays 正常使用交易日历"""
        class FakeCalendar:
            def schedule(self, start_date, end_date):
                trading_idx = pd.DatetimeIndex(["2024-01-02"])
                return pd.DataFrame({"market_open": trading_idx}, index=trading_idx)

        fake_module = SimpleNamespace(get_calendar=lambda name: FakeCalendar())
        monkeypatch.setitem(sys.modules, "pandas_market_calendars", fake_module)

        holidays = loader._get_holidays("2024-01-01", "2024-01-03")
        assert datetime(2024, 1, 1).date() in holidays
        assert datetime(2024, 1, 2).date() not in holidays

    def test_get_holidays_import_error_returns_empty(self, loader, monkeypatch):
        """测试 _get_holidays ImportError 降级"""
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pandas_market_calendars":
                raise ImportError("forced")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        holidays = loader._get_holidays("2024-01-01", "2024-01-05")
        assert holidays == []

    def test_get_holidays_exception_returns_empty(self, loader, monkeypatch):
        """测试 _get_holidays 遇到异常时返回空列表"""
        original_import = builtins.__import__

        class DummyModule:
            @staticmethod
            def get_calendar(name):
                class DummyCalendar:
                    def schedule(self, *args, **kwargs):
                        raise RuntimeError("boom")

                return DummyCalendar()

        def fake_import(name, *args, **kwargs):
            if name == "pandas_market_calendars":
                return DummyModule
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        holidays = loader._get_holidays("2024-01-01", "2024-01-05")
        assert holidays == []

    def test_load_method(self, loader):
        """测试load方法"""
        # load方法应该调用load_data
        with patch.object(loader, 'load_data') as mock_load_data:
            mock_load_data.return_value = pd.DataFrame({'test': [1, 2, 3]})

            result = loader.load(symbol="000001", start_date="2023-01-01", end_date="2023-01-03")

            mock_load_data.assert_called_once_with(
                symbol="000001",
                start_date="2023-01-01",
                end_date="2023-01-03"
            )
            assert isinstance(result, pd.DataFrame)


class TestStockDataLoaderIntegration:
    """测试股票数据加载器集成场景"""

    @pytest.fixture
    def loader(self, tmp_path):
        """创建加载器fixture"""
        return StockDataLoader(save_path=str(tmp_path))

    @patch('src.data.loader.stock_loader.ak.stock_zh_a_hist')
    def test_complete_data_loading_workflow(self, mock_ak, loader):
        """测试完整数据加载工作流程"""
        # 设置模拟数据
        mock_data = pd.DataFrame({
            '日期': ['2023-01-01', '2023-01-02'],
            '开盘': [10.0, 10.5],
            '收盘': [10.5, 11.0],
            '最高': [11.0, 11.5],
            '最低': [9.5, 10.0],
            '成交量': [1000, 1200]
        })
        mock_ak.return_value = mock_data

        # 执行完整工作流程
        data = loader.load_data(
            symbol="000001",
            start_date="2023-01-01",
            end_date="2023-01-02"
        )

        # 验证数据加载
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2

        # 验证数据验证
        is_valid = loader.validate_data(data)
        assert is_valid == True

    @patch('src.data.loader.stock_loader.ak.stock_zh_a_hist')
    def test_error_handling_and_retry(self, mock_ak, loader):
        """测试错误处理和重试机制"""
        # 模拟第一次调用失败，第二次成功
        mock_ak.side_effect = [
            Exception("Network error"),  # 第一次失败
            pd.DataFrame({  # 第二次成功
                '日期': ['2023-01-01'],
                '开盘': [10.0],
                '收盘': [10.5],
                '最高': [11.0],
                '最低': [9.5],
                '成交量': [1000]
            })
        ]

        data = loader.load_data(
            symbol="000001",
            start_date="2023-01-01",
            end_date="2023-01-01"
        )

        # 应该成功获取数据（通过重试）
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1
        assert mock_ak.call_count == 2  # 调用了两次

    def test_cache_functionality(self, loader, tmp_path):
        """测试缓存功能"""
        # 创建缓存文件
        cache_file = tmp_path / "000001_2023-01-01_2023-01-01_daily_none.csv"
        cache_data = pd.DataFrame({
            'date': ['2023-01-01'],
            'open': [10.0],
            'close': [10.5],
            'high': [11.0],
            'low': [9.5],
            'volume': [1000]
        })
        cache_data.to_csv(cache_file, index=False)

        # 设置文件为最近修改
        import time
        current_time = time.time()
        os.utime(cache_file, (current_time, current_time))

        # 检查缓存是否有效
        is_valid = loader._is_cache_valid(cache_file)
        assert is_valid == True

    def test_data_transformation_and_validation(self, loader):
        """测试数据转换和验证"""
        # 创建包含中文列名的原始数据
        raw_data = pd.DataFrame({
            '日期': ['2023-01-01', '2023-01-02'],
            '开盘': [10.0, 10.5],
            '收盘': [10.5, 11.0],
            '最高': [11.0, 11.5],
            '最低': [9.5, 10.0],
            '成交量': [1000, 1200]
        })

        # 验证数据
        is_valid = loader.validate_data(raw_data)
        assert is_valid == True

        # 验证成交量
        volume_valid = loader._validate_volume(raw_data)
        assert volume_valid == True

    def test_configuration_management(self, loader):
        """测试配置管理"""
        # 测试配置验证
        valid_config = {
            "save_path": "/tmp/data",
            "max_retries": 3,
            "cache_days": 30,
            "frequency": "daily",
            "adjust_type": "none"
        }

        is_valid = loader.validate_config(valid_config)
        assert is_valid == True

        # 测试必需字段
        required_fields = loader.get_required_config_fields()
        assert isinstance(required_fields, list)
        assert len(required_fields) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])