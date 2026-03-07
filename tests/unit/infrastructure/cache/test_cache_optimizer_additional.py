import math

import pytest

from src.infrastructure.cache.core.cache_optimizer import CacheOptimizer, CachePolicy


@pytest.fixture
def optimizer():
    return CacheOptimizer()


def test_optimize_cache_size_extreme_conditions(optimizer):
    assert optimizer.optimize_cache_size(500, metrics_or_hit_rate=0.0, memory_usage=1.0) == 100
    assert optimizer.optimize_cache_size(500, metrics_or_hit_rate=1.0, memory_usage=0.0) == 10000
    assert optimizer.optimize_cache_size(800, metrics_or_hit_rate=0.4, memory_usage=0.85) == 400
    assert optimizer.optimize_cache_size(800, metrics_or_hit_rate=0.4, memory_usage=0.2) == 1600
    assert optimizer.optimize_cache_size(800, metrics_or_hit_rate=0.95, memory_usage=0.3) == 1600
    assert optimizer.optimize_cache_size(800, metrics_or_hit_rate=0.95, memory_usage=0.9) == 400

    history = optimizer.get_optimization_history()
    assert len(history) >= 6
    assert any(record["reason"] == "critical_underutilization" for record in history)


def test_suggest_eviction_policy_branches(optimizer):
    assert optimizer.suggest_eviction_policy({}) == CachePolicy.LRU
    assert optimizer.suggest_eviction_policy({"random_access": 10, "sequential_access": 1}) == CachePolicy.LFU
    assert optimizer.suggest_eviction_policy({"frequent_access": True}) == CachePolicy.LFU
    assert optimizer.suggest_eviction_policy({"access_frequencies": {"a": 1, "b": 100}}) == CachePolicy.LFU
    assert optimizer.suggest_eviction_policy({"a": 1, "b": 2}) == CachePolicy.LRU


def test_get_cache_recommendations(optimizer):
    metrics = {
        "hit_rate": 0.3,
        "avg_response_time": 200,
        "memory_usage": 0.9,
        "size": 1000,
        "access_pattern": {"random_access": 5, "sequential_access": 1},
    }
    rec = optimizer.get_cache_recommendations(metrics)
    assert rec["size_optimization"]["suggestion"] == "increase"
    assert rec["warnings"]
    assert any("建议调整缓存大小" in warning for warning in rec["warnings"])
    assert rec["policy_recommendation"] == "lfu"


def test_analyze_access_patterns_dict_and_list(optimizer):
    dict_logs = {
        "read_operations": 5,
        "write_operations": 10,
        "cache_hits": 3,
        "cache_misses": 7,
        "sequential_access": 1,
        "random_access": 5,
    }
    dict_result = optimizer.analyze_access_patterns(dict_logs)
    assert dict_result["total_accesses"] == sum(dict_logs.values())
    assert dict_result["access_pattern_type"] in {"random", "sequential"}
    assert dict_result["avg_access_per_key"] >= 0

    list_result = optimizer.analyze_access_patterns([{"key": "a"}, {"key": "a"}, {"key": "b"}])
    assert list_result["total_accesses"] == 3
    assert list_result["unique_keys"] == 2


def test_get_optimization_recommendations_with_history(optimizer):
    optimizer.optimize_cache_size(500, metrics_or_hit_rate=0.6, memory_usage=0.6)
    optimizer.optimize_cache_size(800, metrics_or_hit_rate=0.4, memory_usage=0.8)
    optimizer.metrics_history.append({"hit_rate": 0.4, "memory_usage": 0.8})

    recommendations = optimizer.get_optimization_recommendations()
    assert recommendations["size_trend_analysis"]["history"]
    assert recommendations["hit_rate_trend_analysis"]["history"]
    assert recommendations["overall_recommendations"]


def test_handle_cache_exceptions_decorator():
    from src.infrastructure.cache.core.cache_optimizer import handle_cache_exceptions

    @handle_cache_exceptions(default_return="fallback", log_level="warning")
    def faulty():
        raise RuntimeError("boom")

    assert faulty() == "fallback"


