from typing import List

import pytest

from src.infrastructure.cache.core.cache_optimizer import CacheOptimizer, CachePolicy


def test_optimize_cache_size_records_history_and_metrics():
    optimizer = CacheOptimizer()

    monitor_warning = optimizer.monitor_cache_performance(
        {"hit_rate": 0.4, "avg_response_time": 120.0, "memory_usage": 0.85}
    )
    assert monitor_warning["status"] == "critical"

    new_size = optimizer.optimize_cache_size(
        current_size=400,
        metrics_or_hit_rate={"hit_rate": 0.3, "memory_usage": 0.9},
    )
    assert new_size < 400

    optimizer.monitor_cache_performance(
        {"hit_rate": 0.92, "avg_response_time": 80.0, "memory_usage": 0.35}
    )
    enlarged_size = optimizer.optimize_cache_size(
        current_size=new_size,
        metrics_or_hit_rate={"hit_rate": 0.95, "memory_usage": 0.2},
    )
    assert enlarged_size > new_size

    history = optimizer.get_optimization_history()
    assert len(history) == 2
    assert history[0]["reason"] == "low_hit_high_memory"
    assert history[-1]["reason"] in ("high_hit_low_memory", "high_hit_high_memory")

    performance = optimizer.get_performance_metrics()
    assert performance["total_optimizations"] == 2
    assert performance["optimization_success_rate"] > 0

    recommendations = optimizer.get_optimization_recommendations()
    assert recommendations["recommendations"]


@pytest.mark.parametrize(
    "patterns,expected",
    [
        ({"random_access": 10, "sequential_access": 2}, CachePolicy.LFU),
        ({"frequent_access": True}, CachePolicy.LFU),
        ({}, CachePolicy.LRU),
        ({"access_frequencies": {"a": 1, "b": 1000}}, CachePolicy.LFU),
    ],
)
def test_suggest_eviction_policy_handles_patterns(patterns, expected):
    optimizer = CacheOptimizer()
    assert optimizer.suggest_eviction_policy(patterns) == expected


def test_get_cache_recommendations_includes_warnings_and_policy():
    optimizer = CacheOptimizer()
    suggestions = optimizer.get_cache_recommendations(
        {
            "hit_rate": 0.45,
            "avg_response_time": 150.0,
            "memory_usage": 0.95,
            "size": 512,
            "policy": "lfu",
        }
    )
    assert "warnings" in suggestions and suggestions["warnings"]
    assert suggestions["size_optimization"]["suggestion"] == "increase"
    assert suggestions["policy_recommendation"] == "lfu"


def test_analyze_access_patterns_dict_source():
    optimizer = CacheOptimizer()
    analysis = optimizer.analyze_access_patterns(
        {
            "read_operations": 60,
            "write_operations": 40,
            "cache_hits": 80,
            "cache_misses": 20,
            "sequential_access": 5,
            "random_access": 10,
        }
    )
    assert analysis["pattern_type"] == "analyzed"
    assert analysis["total_accesses"] > 0
    assert analysis["read_write_ratio"] == pytest.approx(0.6)
    assert analysis["hit_rate"] == pytest.approx(0.8)
    assert analysis["access_pattern_type"] == "random"


