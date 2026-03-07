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

import pandas as pd
import pytest

from src.data.integration.enhanced_data_integration_modules import integration_manager
from src.data.integration.enhanced_data_integration_modules.config import IntegrationConfig


class DummyCacheStrategy:
    def __init__(self, config):
        self._max_size = config["max_size"]
        self.hit_rate = 0.5
        self.ttl_optimized = False
        self.cleanup_calls = 0
        self.set_calls = []

    def get_hit_rate(self):
        return self.hit_rate

    def get_max_size(self):
        return self._max_size

    def set_max_size(self, value):
        self._max_size = value
        self.set_calls.append(value)

    def optimize_ttl_strategy(self):
        self.ttl_optimized = True

    def cleanup_expired(self):
        self.cleanup_calls += 1

    def set(self, key, data, ttl=None):
        self.set_calls.append((key, ttl))

    def get_stats(self):
        return {"hit_rate": self.hit_rate}


class DummyQualityMonitor:
    def get_overall_quality_score(self):
        return 0.92


class DummyParallelManager:
    def __init__(self):
        self.optimize_called = False

    def get_stats(self):
        return {"workers": 2}

    def optimize_task_distribution(self):
        self.optimize_called = True


class DummyThreadPoolManager:
    def __init__(self, initial_size, max_size, min_size):
        self.current_size = initial_size
        self._max_size = max_size
        self.resize_calls = []

    def resize(self, new_size):
        self.current_size = new_size
        self.resize_calls.append(new_size)

    def get_current_size(self):
        return self.current_size

    def get_max_size(self):
        return self._max_size

    def get_utilization(self):
        return 0.55


class DummyConnectionPoolManager:
    def __init__(self, max_size, timeout):
        self.max_size = max_size
        self.timeout = timeout


class DummyMemoryOptimizer:
    def __init__(self, enable_compression, compression_level):
        self.calls = []

    def compress_cache_data(self, cache_strategy):
        self.calls.append(cache_strategy)


class DummyFinancialOptimizer:
    pass


class DummyLoader:
    def load_data(self, identifier, **kwargs):
        return pd.DataFrame({"value": [1]})


@pytest.fixture
def integration():
    config = IntegrationConfig()
    instance = integration_manager.EnhancedDataIntegration.__new__(
        integration_manager.EnhancedDataIntegration
    )
    instance.config = config
    instance.cache_strategy = DummyCacheStrategy(config.cache_strategy)
    instance.quality_monitor = DummyQualityMonitor()
    instance.parallel_manager = DummyParallelManager()
    instance._thread_pool_manager = DummyThreadPoolManager(4, 8, 2)
    instance._memory_optimizer = DummyMemoryOptimizer(True, 3)
    instance._financial_optimizer = DummyFinancialOptimizer()
    instance.stock_loader = DummyLoader()
    instance.index_loader = DummyLoader()
    instance._performance_metrics = {
        "avg_response_time": 0.0,
        "total_requests": 0,
        "cache_hit_rate": 1.0,
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
    instance._cache_warming_status = {
        "is_warming": False,
        "warmed_items": 0,
        "total_items": 0,
        "warming_progress": 0.0,
    }
    return instance


def test_adaptive_adjustment_triggers_all_paths(monkeypatch, integration):
    calls = []

    monkeypatch.setattr(
        integration_manager.EnhancedDataIntegration,
        "_optimize_cache_strategy",
        lambda self: calls.append("cache"),
    )
    monkeypatch.setattr(
        integration_manager.EnhancedDataIntegration,
        "_optimize_memory_usage",
        lambda self: calls.append("memory"),
    )
    monkeypatch.setattr(
        integration_manager.EnhancedDataIntegration,
        "_optimize_thread_pool",
        lambda self: calls.append("thread"),
    )

    integration._performance_metrics.update(
        {
            "cache_hit_rate": 0.1,
            "memory_usage": 0.9,
            "avg_response_time": 1500,
        }
    )

    integration._adaptive_adjustment()
    assert calls == ["cache", "memory", "thread"]


def test_optimize_cache_strategy_updates_cache_and_starts_warming(monkeypatch, integration):
    cache_strategy = integration.cache_strategy
    cache_strategy.hit_rate = 0.3
    start_size = cache_strategy.get_max_size()
    warming_called = {}

    monkeypatch.setattr(
        integration,
        "_start_cache_warming",
        types.MethodType(lambda self: warming_called.setdefault("started", True), integration),
    )

    integration._optimize_cache_strategy()

    assert cache_strategy.get_max_size() == int(start_size * 1.2)
    assert cache_strategy.ttl_optimized is True
    assert warming_called["started"] is True


def test_optimize_memory_usage_cleans_and_compresses(integration):
    cache_strategy = integration.cache_strategy
    start_size = cache_strategy.get_max_size()

    integration._optimize_memory_usage()

    assert cache_strategy.cleanup_calls == 1
    assert integration._memory_optimizer.calls == [cache_strategy]
    assert cache_strategy.get_max_size() == int(start_size * 0.8)


def test_optimize_thread_pool_resizes_and_notifies_parallel_manager(integration):
    integration.parallel_manager.optimize_called = False
    thread_pool = integration._thread_pool_manager
    thread_pool._max_size = 10
    thread_pool.current_size = 4

    integration._optimize_thread_pool()

    assert thread_pool.resize_calls[-1] == 6
    assert integration.parallel_manager.optimize_called is True


def test_start_cache_warming_runs_preload(monkeypatch, integration):
    class ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr(integration_manager.threading, "Thread", ImmediateThread)

    calls = {"stock": 0, "index": 0}

    def fake_stock(self, symbol):
        calls["stock"] += 1

    def fake_index(self, index):
        calls["index"] += 1

    monkeypatch.setattr(
        integration_manager.EnhancedDataIntegration,
        "_preload_stock_data",
        fake_stock,
        raising=False,
    )
    monkeypatch.setattr(
        integration_manager.EnhancedDataIntegration,
        "_preload_index_data",
        fake_index,
        raising=False,
    )

    status = integration._cache_warming_status
    status["total_items"] = 7

    integration._start_cache_warming()
    assert status["is_warming"] is False
    assert status["warmed_items"] == 7
    assert calls == {"stock": 4, "index": 3}

    status["is_warming"] = True
    status["warmed_items"] = 0
    integration._start_cache_warming()
    assert status["warmed_items"] == 0


def test_check_cache_warming_updates_progress(integration):
    status = integration._cache_warming_status
    status.update({"total_items": 10, "warmed_items": 4})
    integration._check_cache_warming()
    assert status["warming_progress"] == 0.4

