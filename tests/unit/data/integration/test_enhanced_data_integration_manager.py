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
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.integration.enhanced_data_integration_modules.integration_manager import (
    EnhancedDataIntegration,
    IntegrationConfig,
    TaskPriority,
)

MODULE_PATH = "src.data.integration.enhanced_data_integration_modules.integration_manager"
ORIGINAL_START_MONITORING = EnhancedDataIntegration._start_performance_monitoring


class DummyParallelManager:
    def __init__(self):
        self.submitted = []
        self.executed = {}
        self.optimized = False

    def submit_task(self, task):
        self.submitted.append(task.task_id)
        return task.task_id

    def execute_tasks(self, timeout=30):
        return self.executed

    def optimize_task_distribution(self):
        self.optimized = True


class DummyCacheStrategy:
    def __init__(self):
        self.max_size = 100
        self.hit_rate = 0.75
        self.optimized = False
        self.cleaned = False
        self.data = {}

    def get_hit_rate(self):
        return self.hit_rate

    def get_max_size(self):
        return self.max_size

    def set_max_size(self, value):
        self.max_size = value

    def optimize_ttl_strategy(self):
        self.optimized = True

    def cleanup_expired(self):
        self.cleaned = True

    def set(self, key, value, ttl=None):
        self.data[key] = value


class DummyThreadPoolManager:
    def __init__(self, initial_size, max_size, min_size):
        self.current_size = initial_size
        self.max_size = max_size
        self.resize_calls = []

    def resize(self, new_size):
        self.current_size = new_size
        self.resize_calls.append(new_size)

    def get_current_size(self):
        return self.current_size

    def get_max_size(self):
        return self.max_size

    def get_utilization(self):
        return 0.4


class DummyConnectionPoolManager:
    def __init__(self, *_, **__):
        self.connections = []


class DummyMemoryOptimizer:
    def __init__(self, *_, **__):
        self.compress_calls = 0

    def compress_cache_data(self, cache_strategy):
        self.compress_calls += 1


class DummyFinancialOptimizer:
    def __init__(self):
        self.optimized = False

    def optimize_financial_loading(self, *_, **__):
        self.optimized = True


class DummyQualityMonitor:
    def __init__(self):
        self.score = 0.96

    def get_overall_quality_score(self):
        return self.score


class DummyLoader:
    def __init__(self, *_, **__):
        self.load_calls = []

    def load_data(self, *_, **__):
        self.load_calls.append((_, __))
        return pd.DataFrame({"value": [1]})


class DummyDataManagerSingleton:
    @staticmethod
    def get_instance():
        return object()


@pytest.fixture
def integration(monkeypatch):
    module = "src.data.integration.enhanced_data_integration_modules.integration_manager"

    monkeypatch.setattr(f"{module}.create_enhanced_loader", lambda config: DummyParallelManager())
    monkeypatch.setattr(f"{module}.create_enhanced_cache_strategy", lambda config: DummyCacheStrategy())
    monkeypatch.setattr(f"{module}.create_enhanced_quality_monitor", lambda **kwargs: DummyQualityMonitor())
    monkeypatch.setattr(f"{module}.DynamicThreadPoolManager", DummyThreadPoolManager)
    monkeypatch.setattr(f"{module}.ConnectionPoolManager", DummyConnectionPoolManager)
    monkeypatch.setattr(f"{module}.MemoryOptimizer", DummyMemoryOptimizer)
    monkeypatch.setattr(f"{module}.FinancialDataOptimizer", DummyFinancialOptimizer)
    monkeypatch.setattr(f"{module}.DataManagerSingleton", DummyDataManagerSingleton, raising=False)
    monkeypatch.setattr(f"{module}.StockDataLoader", DummyLoader)
    monkeypatch.setattr(f"{module}.IndexDataLoader", DummyLoader)
    monkeypatch.setattr(f"{module}.FinancialDataLoader", DummyLoader)
    monkeypatch.setattr(
        f"{module}.EnhancedDataIntegration._start_performance_monitoring",
        lambda self: None,
    )
    dm_module = importlib.import_module("src.data.data_manager")
    monkeypatch.setattr(dm_module, "DataManagerSingleton", DummyDataManagerSingleton, raising=False)

    return EnhancedDataIntegration(config=IntegrationConfig())


def _quality_metrics(identifier: str):
    return SimpleNamespace(
        completeness=1.0,
        accuracy=0.99,
        consistency=1.0,
        timeliness=1.0,
        validity=1.0,
        uniqueness=1.0,
        overall_score=0.99 if identifier == "BBB" else 0.0,
        timestamp="2024-01-01T00:00:00",
        data_type="stock",
        details={"source": "test"},
    )


def test_load_stock_data_merges_cache_and_loader(integration):
    cached_df = pd.DataFrame({"price": [1]})
    loaded_df = pd.DataFrame({"price": [2]})

    integration._check_cache_for_symbols = lambda *args, **kwargs: {"AAA": cached_df}
    integration._load_data_parallel = lambda *args, **kwargs: {"BBB": loaded_df}

    def fake_quality(data, identifier):
        return _quality_metrics(identifier) if identifier == "BBB" else None

    integration._check_data_quality = fake_quality

    cached_symbols = []
    integration._cache_data = lambda symbol, data, *args, **kwargs: cached_symbols.append(symbol)
    integration._get_memory_usage = lambda: 0.42
    integration._update_avg_response_time = lambda *args, **kwargs: None

    result = integration.load_stock_data(
        ["AAA", "BBB"],
        "2024-01-01",
        "2024-01-05",
    )

    assert result["success"] is True
    assert "AAA" in result["data"] and "BBB" in result["data"]
    assert result["stats"]["loaded_count"] == 1
    assert result["stats"]["cached_count"] == 1
    assert cached_symbols == ["BBB"]
    assert result["quality_metrics"]["BBB"]["overall_quality"] == pytest.approx(0.99, rel=1e-3)
    assert result["quality_metrics"]["AAA"]["overall_quality"] == 0.0


def test_load_stock_data_on_exception_returns_failure(integration):
    def raise_error(*args, **kwargs):
        raise RuntimeError("cache failure")

    integration._check_cache_for_symbols = raise_error
    integration._update_avg_response_time = lambda *args, **kwargs: None

    result = integration.load_stock_data(["AAA"], "2024-01-01", "2024-01-02")

    assert result["success"] is False
    assert result["stats"]["cache_hit_rate"] == 0.0
    assert result["stats"]["loaded_count"] == 0


def test_adaptive_adjustment_invokes_optimizers(integration):
    integration._performance_metrics.update(
        {
            "cache_hit_rate": 0.1,
            "memory_usage": 0.95,
            "avg_response_time": 5000,
        }
    )

    calls = {"cache": 0, "memory": 0, "thread": 0}
    integration._optimize_cache_strategy = lambda: calls.__setitem__("cache", calls["cache"] + 1)
    integration._optimize_memory_usage = lambda: calls.__setitem__("memory", calls["memory"] + 1)
    integration._optimize_thread_pool = lambda: calls.__setitem__("thread", calls["thread"] + 1)

    integration._adaptive_adjustment()

    assert calls == {"cache": 1, "memory": 1, "thread": 1}


def test_optimize_cache_strategy_updates_limits(integration):
    integration.cache_strategy.max_size = 100
    warming_calls = []
    integration._start_cache_warming = lambda: warming_calls.append(True)

    integration._optimize_cache_strategy()

    assert integration.cache_strategy.max_size == 120
    assert integration.cache_strategy.optimized is True
    assert warming_calls == [True]


def test_enterprise_feature_workflow(integration):
    status = integration.get_enterprise_features_status()
    assert status["distributed_manager"]["enabled"] is True

    assert integration.add_distributed_node({"region": "CN"}) is True
    assert len(integration.distributed_manager["nodes"]) == 1

    assert integration.start_realtime_stream_processing({"mode": "live"}) is True
    assert len(integration.realtime_stream["stream_processors"]) == 1

    assert integration.add_monitoring_metric("performance", {"latency": 12}) is True
    assert integration.add_monitoring_metric("unknown", {}) is False

    dashboard = integration.get_monitoring_dashboard_data()
    assert "metrics" in dashboard and "alerts" in dashboard


def test_start_cache_warming_invokes_preload(integration, monkeypatch):
    calls = []

    integration._preload_stock_data = lambda symbol: calls.append(("stock", symbol))
    integration._preload_index_data = lambda index: calls.append(("index", index))

    class ImmediateThread:
        def __init__(self, target, daemon):
            target()

        def start(self):
            return None

    monkeypatch.setattr(
        "src.data.integration.enhanced_data_integration_modules.integration_manager.threading.Thread",
        lambda target, daemon: ImmediateThread(target, daemon),
    )

    integration._start_cache_warming()
    assert any(kind == "stock" for kind, _ in calls)
    assert any(kind == "index" for kind, _ in calls)


def test_load_index_data_combines_cache_and_loaded(integration):
    cached_df = pd.DataFrame({"close": [1]})
    loaded_df = pd.DataFrame({"close": [2]})

    integration._check_cache_for_indices = lambda *args, **kwargs: {"CSI": cached_df}
    integration._load_index_data_parallel = (
        lambda *args, **kwargs: {"HSI": loaded_df}
    )
    cached_symbols = []
    integration._cache_index_data = (
        lambda index, data, *args, **kwargs: cached_symbols.append(index)
    )
    integration._check_data_quality = lambda data, index: {"overall_quality": 1.0}

    result = integration.load_index_data(
        ["CSI", "HSI"], "2024-01-01", "2024-01-05", enable_quality_check=True
    )

    assert result["success"] is True
    assert result["stats"]["loaded_count"] == 1
    assert result["stats"]["cached_count"] == 1
    assert cached_symbols == ["HSI"]
    assert result["quality_metrics"]["HSI"]["overall_quality"] == 1.0


def test_load_financial_data_returns_failure_on_exception(integration):
    integration._check_cache_for_financial = (
        lambda *args, **kwargs: {}
    )

    def failing_loader(*args, **kwargs):
        raise RuntimeError("boom")

    integration._load_financial_data_parallel = failing_loader

    result = integration.load_financial_data(
        ["AAA"], "2024-01-01", "2024-01-05"
    )
    assert result["success"] is False
    assert result["stats"]["loaded_count"] == 0


def test_cache_wrappers_delegate_to_utils(monkeypatch, integration):
    outputs = {}

    monkeypatch.setattr(
        f"{MODULE_PATH}.check_cache_for_symbols",
        lambda cache, symbols, *_args: outputs.setdefault("symbols", symbols) or {"A": pd.DataFrame({"v": [1]})},
    )
    monkeypatch.setattr(
        f"{MODULE_PATH}.check_cache_for_indices",
        lambda cache, indices, *_args: outputs.setdefault("indices", indices) or {"B": pd.DataFrame({"v": [2]})},
    )
    monkeypatch.setattr(
        f"{MODULE_PATH}.check_cache_for_financial",
        lambda symbols, *_args: outputs.setdefault("financial", symbols) or {"C": pd.DataFrame({"v": [3]})},
    )
    cache_calls = []
    monkeypatch.setattr(
        f"{MODULE_PATH}.cache_data",
        lambda *args: cache_calls.append(("stock", args[1])),
    )
    monkeypatch.setattr(
        f"{MODULE_PATH}.cache_index_data",
        lambda *args: cache_calls.append(("index", args[1])),
    )
    monkeypatch.setattr(
        f"{MODULE_PATH}.cache_financial_data",
        lambda *args: cache_calls.append(("financial", args[1])),
    )
    monkeypatch.setattr(
        f"{MODULE_PATH}.check_data_quality",
        lambda data, identifier, *_: {"overall_quality": 1.0, "id": identifier},
    )
    monkeypatch.setattr(
        f"{MODULE_PATH}.update_avg_response_time_util",
        lambda metrics, value: metrics.update({"last": value}),
    )
    sentinel_stats = {"ok": True}
    monkeypatch.setattr(
        f"{MODULE_PATH}.get_integration_stats",
        lambda *_: sentinel_stats,
    )

    assert "A" in integration._check_cache_for_symbols(["A"], "s", "e", "1d")
    assert "B" in integration._check_cache_for_indices(["B"], "s", "e", "1d")
    assert "C" in integration._check_cache_for_financial(["C"], "s", "e", "financial")

    df = pd.DataFrame({"v": [1]})
    integration._cache_data("X", df, "s", "e", "1d")
    integration._cache_index_data("Y", df, "s", "e", "1d")
    integration._cache_financial_data("Z", df, "s", "e", "f")

    quality = integration._check_data_quality(df, "X")
    assert quality["id"] == "X"

    integration._update_avg_response_time(12.3)
    assert integration._performance_metrics["last"] == 12.3
    assert integration.get_integration_stats() is sentinel_stats
    assert len(cache_calls) == 3


def test_start_performance_monitoring_single_iteration(monkeypatch, integration):
    calls = {"update": 0, "adjust": 0, "warming": 0}
    monkeypatch.setattr(integration, "_update_performance_metrics", lambda: calls.__setitem__("update", calls["update"] + 1))
    monkeypatch.setattr(integration, "_adaptive_adjustment", lambda: calls.__setitem__("adjust", calls["adjust"] + 1))
    monkeypatch.setattr(integration, "_check_cache_warming", lambda: calls.__setitem__("warming", calls["warming"] + 1))

    recorded = {}

    class ThreadStub:
        def __init__(self, target, daemon=True):
            self.target = target

        def start(self):
            recorded["target"] = self.target

    monkeypatch.setattr(f"{MODULE_PATH}.threading.Thread", ThreadStub)

    def fake_sleep(_seconds):
        raise StopIteration()

    monkeypatch.setattr(f"{MODULE_PATH}.time.sleep", fake_sleep)

    ORIGINAL_START_MONITORING(integration)
    assert "target" in recorded
    with pytest.raises(StopIteration):
        recorded["target"]()
    assert calls == {"update": 1, "adjust": 1, "warming": 1}


def test_optimize_memory_usage_reduces_cache(integration):
    initial_size = integration.cache_strategy.max_size
    integration._optimize_memory_usage()
    assert integration.cache_strategy.cleaned is True
    assert integration._memory_optimizer.compress_calls == 1
    assert integration.cache_strategy.max_size == int(initial_size * 0.8)


def test_optimize_thread_pool_resizes_and_optimizes(integration):
    integration.parallel_manager.optimized = False
    integration._optimize_thread_pool()
    assert integration._thread_pool_manager.resize_calls
    assert integration.parallel_manager.optimized is True


def test_preload_stock_and_index_data_populates_cache(integration):
    integration.cache_strategy.data.clear()
    integration._preload_stock_data("AAA")
    integration._preload_index_data("CSI")
    assert any(key.startswith("stock_AAA") for key in integration.cache_strategy.data)
    assert any(key.startswith("index_CSI") for key in integration.cache_strategy.data)


def test_start_cache_warming_handles_failure(monkeypatch, integration):
    integration._cache_warming_status["is_warming"] = False
    integration._preload_stock_data = lambda symbol: (_ for _ in ()).throw(RuntimeError("boom"))
    integration._preload_index_data = lambda index: None
    monkeypatch.setattr(
        f"{MODULE_PATH}.threading.Thread",
        lambda target, daemon=True: type("ImmediateThread", (), {"start": lambda self: target()})(),
    )
    integration._start_cache_warming()
    assert integration._cache_warming_status["is_warming"] is False


def test_load_stock_data_without_results_returns_error(integration):
    integration._check_cache_for_symbols = lambda *args, **kwargs: {}
    integration._load_data_parallel = lambda *args, **kwargs: {}
    integration._get_memory_usage = lambda: 0.1
    result = integration.load_stock_data(["AAA"], "2024-01-01", "2024-01-02")
    assert result["success"] is False
    assert result["error"] == "API Error"


def test_load_index_data_exception_returns_failure(monkeypatch, integration):
    integration._check_cache_for_indices = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cache fail"))
    result = integration.load_index_data(["CSI"], "2024-01-01", "2024-01-05")
    assert result["success"] is False
    assert result["data"] == {}


def test_load_financial_data_success_path(integration):
    integration._check_cache_for_financial = lambda *args, **kwargs: {}
    sample_df = pd.DataFrame({"value": [1]})
    integration._load_financial_data_parallel = lambda *args, **kwargs: {"AAA": sample_df}
    integration._check_data_quality = lambda data, identifier: _quality_metrics("BBB")
    result = integration.load_financial_data(["AAA"], "2024-01-01", "2024-01-05")
    assert result["success"] is True
    assert result["quality_metrics"]["AAA"]["overall_quality"] == pytest.approx(0.99, rel=1e-3)


def test_load_data_parallel_collects_results(integration):
    start_date, end_date = "2024-01-01", "2024-01-05"
    task_id = f"stock_BBB_{start_date}_{end_date}"
    integration.parallel_manager.executed = {task_id: pd.DataFrame({"value": [1]})}
    result = integration._load_data_parallel(["BBB"], start_date, end_date, "1d", TaskPriority.HIGH)
    assert "BBB" in result


def test_load_data_parallel_handles_missing_loader(integration):
    integration.stock_loader = None
    result = integration._load_data_parallel(["BBB"], "2024-01-01", "2024-01-05", "1d", TaskPriority.NORMAL)
    assert result == {}


def test_load_index_and_financial_parallel_handles_paths(integration):
    start_date, end_date = "2024-01-01", "2024-01-05"
    index_task = f"index_CSI_{start_date}_{end_date}"
    integration.parallel_manager.executed = {index_task: pd.DataFrame({"value": [2]})}
    result_index = integration._load_index_data_parallel(["CSI"], start_date, end_date, "1d")
    assert "CSI" in result_index

    integration.index_loader = None
    result_index_empty = integration._load_index_data_parallel(["CSI"], start_date, end_date, "1d")
    assert result_index_empty == {}

    integration.index_loader = DummyLoader()
    integration.parallel_manager.executed = {}
    integration.financial_loader = None
    result_fin_empty = integration._load_financial_data_parallel(["AAA"], start_date, end_date, "financial")
    assert result_fin_empty == {}


def test_monitoring_alerts_and_shutdown(monkeypatch, integration):
    thresholds = integration.realtime_stream["alert_system"]["thresholds"]
    integration._avg_response_time = thresholds["response_time"] + 50
    integration._error_count = 5
    integration._total_requests = 5
    integration._start_time = time.time() - 10

    dashboard = integration.get_monitoring_dashboard_data()
    assert dashboard["alerts"]
    assert dashboard["performance_summary"]["uptime"] >= 0

    captured = {}

    def fake_shutdown(parallel, cache, monitor):
        captured["args"] = (parallel, cache, monitor)

    monkeypatch.setattr(
        "src.data.integration.enhanced_data_integration_modules.performance_utils.shutdown",
        fake_shutdown,
    )
    integration.shutdown()
    assert captured["args"][0] is integration.parallel_manager


def test_add_distributed_node_failure_path(integration):
    integration.distributed_manager = None
    assert integration.add_distributed_node({"node_id": "bad"}) is False


def test_start_realtime_stream_processing_failure(integration):
    integration.realtime_stream["stream_processors"] = None
    assert integration.start_realtime_stream_processing({"mode": "fail"}) is False


def test_add_monitoring_metric_exception_path(integration):
    integration.monitoring_dashboard["metrics"] = None
    assert integration.add_monitoring_metric("performance", {"latency": 1}) is False


def test_check_cache_warming_progress_updates(integration):
    integration._cache_warming_status.update({"total_items": 10, "warmed_items": 5})
    integration._check_cache_warming()
    assert integration._cache_warming_status["warming_progress"] == 0.5

