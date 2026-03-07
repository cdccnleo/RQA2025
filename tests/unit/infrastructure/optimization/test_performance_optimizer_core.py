import time
from collections import deque
from typing import Iterator

import pytest

from src.infrastructure.optimization.performance_optimizer import (
    ComponentFactoryPerformanceOptimizer,
    OptimizationResult,
    PerformanceMetrics,
)


def _metrics_sequence(*pairs) -> Iterator[PerformanceMetrics]:
    for args in pairs:
        yield PerformanceMetrics(*args)


@pytest.fixture()
def optimizer(monkeypatch) -> ComponentFactoryPerformanceOptimizer:
    optimizer = ComponentFactoryPerformanceOptimizer()

    # 避免真实优化逻辑执行，全部替换为 no-op
    for method in [
        "_optimize_async_processing",
        "_optimize_concurrency",
        "_optimize_algorithms",
        "_implement_object_pooling",
        "_optimize_garbage_collection",
        "_reduce_memory_fragmentation",
        "_optimize_connection_pooling",
        "_optimize_cache_strategy",
        "_optimize_batch_operations",
        "_optimize_dictionaries",
        "_optimize_list_operations",
        "_optimize_set_operations",
    ]:
        monkeypatch.setattr(ComponentFactoryPerformanceOptimizer, method, lambda self: None)

    return optimizer


def test_optimize_cpu_usage_handles_zero(monkeypatch, optimizer):
    sequence = _metrics_sequence(
        (time.time(), 50.0, 0.0, 10.0, 100.0, 0.01),
        (time.time(), 45.0, 0.0, 9.0, 120.0, 0.01),
    )
    monkeypatch.setattr(
        ComponentFactoryPerformanceOptimizer,
        "_collect_performance_metrics",
        lambda self: next(sequence),
    )

    result = optimizer.optimize_cpu_usage()
    assert isinstance(result, OptimizationResult)
    assert result.improvement_percentage == 0.0


def test_optimize_io_operations_improvement(monkeypatch, optimizer):
    sequence = _metrics_sequence(
        (time.time(), 50.0, 60.0, 20.0, 150.0, 0.01),
        (time.time(), 48.0, 50.0, 10.0, 200.0, 0.01),
    )
    monkeypatch.setattr(
        ComponentFactoryPerformanceOptimizer,
        "_collect_performance_metrics",
        lambda self: next(sequence),
    )

    result = optimizer.optimize_io_operations()
    assert result.improvement_percentage > 0
    assert result.after_metrics.response_time < result.before_metrics.response_time


def test_optimize_data_structures_zero_before(monkeypatch, optimizer):
    sequence = _metrics_sequence(
        (time.time(), 50.0, 60.0, 15.0, 0.0, 0.01),
        (time.time(), 48.0, 50.0, 10.0, 100.0, 0.01),
    )
    monkeypatch.setattr(
        ComponentFactoryPerformanceOptimizer,
        "_collect_performance_metrics",
        lambda self: next(sequence),
    )

    result = optimizer.optimize_data_structures()
    assert result.improvement_percentage == 100.0


def test_calculate_improvement_edge_cases():
    calc = ComponentFactoryPerformanceOptimizer._calculate_improvement

    assert calc(0, 10, higher_is_better=True) == 100.0
    assert calc(0, 10, higher_is_better=False) == 0.0
    assert calc(100, 90, higher_is_better=False) == pytest.approx(10.0)

