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


import logging

import pandas as pd
import pytest

from src.data.integration.enhanced_data_integration_modules import performance_utils as perf_utils


class _DummyMonitor:
    def __init__(self, should_raise=False):
        self.should_raise = should_raise
        self.called_with = []

    def check_quality(self, data, identifier):
        self.called_with.append((len(data), identifier))
        if self.should_raise:
            raise ValueError("boom")
        return {"score": 0.95, "identifier": identifier}


class _DummyParallel:
    def __init__(self, should_raise=False):
        self.should_raise = should_raise
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1
        if self.should_raise:
            raise RuntimeError("parallel failure")

    def get_stats(self):
        return {"workers": 4}


class _DummyCache:
    def __init__(self, should_raise=False):
        self.should_raise = should_raise
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1
        if self.should_raise:
            raise RuntimeError("cache failure")

    def get_stats(self):
        return {"hit_rate": 0.73, "entries": 12}


def test_check_data_quality_handles_monitor_and_failures(caplog):
    df = pd.DataFrame({"value": [1, 2, 3]})
    monitor = _DummyMonitor()
    result = perf_utils.check_data_quality(df, "RQA", quality_monitor=monitor)
    assert result["identifier"] == "RQA"
    assert monitor.called_with == [(3, "RQA")]

    failing_monitor = _DummyMonitor(should_raise=True)
    with caplog.at_level(logging.WARNING):
        assert perf_utils.check_data_quality(df, "RQB", quality_monitor=failing_monitor) is None
        assert "质量检查失败 RQB" in caplog.text

    assert perf_utils.check_data_quality(pd.DataFrame(), "EMPTY") is None
    assert perf_utils.check_data_quality(None, "NONE") is None


def test_update_avg_response_time_uses_ema():
    metrics = {"avg_response_time": 50.0, "total_requests": 5}
    perf_utils.update_avg_response_time(metrics, 100.0)
    assert metrics["total_requests"] == 6
    assert metrics["avg_response_time"] == pytest.approx(55.0)


def test_get_integration_stats_merges_component_data():
    metrics = {
        "avg_response_time": 25.0,
        "total_requests": 10,
        "successful_requests": 9,
        "failed_requests": 1,
        "quality_score": 0.9,
        "memory_usage": 0.4,
    }
    cache = _DummyCache()
    parallel = _DummyParallel()

    stats = perf_utils.get_integration_stats(metrics, cache_strategy=cache, parallel_manager=parallel)

    assert stats["cache_hit_rate"] == 0.73
    assert stats["parallel_stats"]["workers"] == 4
    assert stats["performance_metrics"] is not metrics
    metrics["avg_response_time"] = 5.0
    assert stats["performance_metrics"]["avg_response_time"] == 25.0


def test_monitor_performance_logs_warning(caplog):
    with caplog.at_level(logging.WARNING):
        perf_utils.monitor_performance()
    assert "性能监控功能需要集成上下文对象" in caplog.text


def test_shutdown_closes_components_and_handles_errors(caplog):
    parallel = _DummyParallel()
    cache = _DummyCache()

    perf_utils.shutdown(parallel_manager=parallel, cache_strategy=cache)
    assert parallel.shutdown_calls == 1
    assert cache.cleanup_calls == 1

    failing_parallel = _DummyParallel(should_raise=True)
    failing_cache = _DummyCache(should_raise=True)
    with caplog.at_level(logging.ERROR):
        perf_utils.shutdown(parallel_manager=failing_parallel, cache_strategy=failing_cache, quality_monitor=object())
    assert "关闭并行管理器失败" in caplog.text
    assert "清理缓存失败" in caplog.text

