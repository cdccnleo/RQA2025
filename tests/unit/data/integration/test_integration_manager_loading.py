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


import types
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data.integration.enhanced_data_integration_modules import integration_manager
from src.data.integration.enhanced_data_integration_modules.config import IntegrationConfig


class _StubParallelManager:
    def __init__(self, results):
        self.results = results
        self.submitted = []

    def submit_task(self, task):
        task_id = f"task_{len(self.submitted)}"
        self.submitted.append(task)
        return task_id

    def execute_tasks(self, timeout=30):
        return {f"task_{idx}": value for idx, value in enumerate(self.results)}


def _make_df(label):
    return pd.DataFrame({"value": [label]})


@pytest.fixture
def integration():
    config = IntegrationConfig()
    instance = integration_manager.EnhancedDataIntegration.__new__(
        integration_manager.EnhancedDataIntegration
    )
    instance.config = config
    instance.cache_strategy = SimpleNamespace(get_hit_rate=lambda: 0.5)
    instance.quality_monitor = SimpleNamespace(get_overall_quality_score=lambda: 0.9)
    instance.parallel_manager = SimpleNamespace()
    instance._thread_pool_manager = SimpleNamespace(get_utilization=lambda: 0.4)
    instance._performance_metrics = {
        "avg_response_time": 0.0,
        "total_requests": 0,
        "cache_hit_rate": 0.0,
        "quality_score": 0.0,
        "memory_usage": 0.0,
        "thread_utilization": 0.0,
    }
    instance._adaptive_cache_config = {
        "hit_rate_threshold": 0.8,
        "memory_threshold": 0.85,
        "response_time_threshold": 1000,
        "quality_threshold": 0.95,
    }
    instance._cache_warming_status = {"is_warming": False, "warmed_items": 0, "total_items": 0}
    instance._update_avg_response_time = lambda *args, **kwargs: None
    instance._get_memory_usage = lambda: 0.42
    instance._cache_data = lambda *args, **kwargs: None
    instance._cache_index_data = lambda *args, **kwargs: None
    instance._cache_financial_data = lambda *args, **kwargs: None
    instance._check_data_quality = lambda data, identifier: SimpleNamespace(
        completeness=0.9,
        accuracy=0.8,
        consistency=0.85,
        timeliness=0.88,
        validity=0.9,
        uniqueness=0.95,
        overall_score=0.9,
        timestamp="2024-01-01T00:00:00",
        data_type=identifier,
        details={"source": "test"},
    )
    return instance


def test_load_stock_data_combines_cache_and_loaded_results(monkeypatch, integration):
    cached_df = _make_df("cached")
    loaded_df = _make_df("loaded")
    cache_calls = []

    def fake_cache(self, symbol, data, start, end, freq):
        cache_calls.append((symbol, freq))

    integration._cache_data = types.MethodType(fake_cache, integration)
    integration._check_cache_for_symbols = types.MethodType(
        lambda self, symbols, *_: {"AAA": cached_df}, integration
    )
    integration._load_data_parallel = types.MethodType(
        lambda self, symbols, *args, **kwargs: {"BBB": loaded_df}, integration
    )

    result = integration.load_stock_data(
        symbols=["AAA", "BBB"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        frequency="1d",
    )

    assert result["success"] is True
    assert set(result["data"].keys()) == {"AAA", "BBB"}
    assert result["stats"]["cached_count"] == 1
    assert result["stats"]["loaded_count"] == 1
    assert cache_calls == [("BBB", "1d")]
    assert "BBB" in result["quality_metrics"]


def test_load_stock_data_returns_error_when_no_results(monkeypatch, integration):
    integration._check_cache_for_symbols = types.MethodType(
        lambda self, *args, **kwargs: {}, integration
    )
    integration._load_data_parallel = types.MethodType(
        lambda self, *args, **kwargs: {}, integration
    )

    result = integration.load_stock_data(
        symbols=["AAA"],
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    assert result["success"] is False
    assert result["error"] == "API Error"
    assert result["stats"]["loaded_count"] == 0


def test_load_stock_data_handles_exception(monkeypatch, integration):
    integration._check_cache_for_symbols = types.MethodType(
        lambda self, *args, **kwargs: {}, integration
    )

    def raise_error(self, *args, **kwargs):
        raise RuntimeError("boom")

    integration._load_data_parallel = types.MethodType(raise_error, integration)
    result = integration.load_stock_data(["AAA"], "2024-01-01", "2024-01-02")

    assert result["success"] is False
    assert result["stats"]["cache_hit_rate"] == 0.0


def test_load_index_data_uses_cache_and_quality(monkeypatch, integration):
    cached_df = _make_df("cached-index")
    loaded_df = _make_df("loaded-index")
    integration._check_cache_for_indices = types.MethodType(
        lambda self, *args, **kwargs: {"000300.SH": cached_df}, integration
    )
    integration._load_index_data_parallel = types.MethodType(
        lambda self, *args, **kwargs: {"000905.SH": loaded_df}, integration
    )
    cache_calls = []
    integration._cache_index_data = types.MethodType(
        lambda self, index, data, *args: cache_calls.append(index), integration
    )

    result = integration.load_index_data(
        ["000300.SH", "000905.SH"],
        "2024-01-01",
        "2024-01-31",
    )

    assert result["success"]
    assert result["stats"]["cached_count"] == 1
    assert cache_calls == ["000905.SH"]


def test_load_data_parallel_handles_missing_loader(monkeypatch, integration):
    integration.stock_loader = None
    integration.parallel_manager = SimpleNamespace()
    results = integration._load_data_parallel(
        ["AAA"],
        "2024-01-01",
        "2024-01-31",
        frequency="1d",
        priority=integration_manager.TaskPriority.NORMAL,
    )
    assert results == {}


def test_load_data_parallel_collects_results(monkeypatch, integration):
    integration.stock_loader = object()
    df_one = _make_df("one")
    df_two = _make_df("two")
    integration.parallel_manager = _StubParallelManager([df_one, df_two])

    results = integration._load_data_parallel(
        ["AAA", "BBB"],
        "2024-01-01",
        "2024-01-31",
        frequency="1d",
        priority=integration_manager.TaskPriority.NORMAL,
    )

    assert results["AAA"].equals(df_one)
    assert results["BBB"].equals(df_two)

