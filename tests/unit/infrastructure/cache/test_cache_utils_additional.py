import types
import time as real_time
from unittest import mock

import pytest

from src.infrastructure.cache.utils import cache_utils


class FakeTime:
    def __init__(self, start: float = 0.0):
        self._value = start

    def advance(self, delta: float) -> None:
        self._value += delta

    def time(self) -> float:
        return self._value


def test_handle_cache_exceptions_returns_copy_and_reraise():
    captured = []

    @cache_utils.handle_cache_exceptions(default_return={"status": "fallback"})
    def failing():
        captured.append("called")
        raise ValueError("boom")

    result_one = failing()
    result_two = failing()
    assert result_one == {"status": "fallback"}
    assert result_two == {"status": "fallback"}
    assert result_one is not result_two
    assert captured == ["called", "called"]

    decorator = cache_utils.handle_cache_exceptions(default_return=None, reraise=True)

    @decorator
    def failing_reraise():
        raise RuntimeError("propagate")

    with pytest.raises(RuntimeError):
        failing_reraise()


def test_handle_cache_exceptions_direct_usage(monkeypatch):
    calls = {"count": 0}

    @cache_utils.handle_cache_exceptions
    def safe_increment():
        calls["count"] += 1
        return calls["count"]

    assert safe_increment() == 1
    assert safe_increment() == 2


def test_generate_cache_key_allows_empty(monkeypatch):
    monkeypatch.setattr(cache_utils, "_should_allow_empty_key", lambda: True)
    assert cache_utils.generate_cache_key() == ""
    assert cache_utils.generate_cache_key("a", b=1) == "a:b=1"


def test_calculate_hash_uses_sha256(monkeypatch):
    monkeypatch.setattr(cache_utils, "_should_use_sha256", lambda: True)
    value = cache_utils.calculate_hash("abc")
    assert len(value) == 64
    assert value == cache_utils.calculate_hash("abc")


def test_compress_and_decompress_data_variants():
    raw_bytes = b"plain-bytes"
    compressed = cache_utils.compress_data(raw_bytes)
    assert isinstance(compressed, bytes)
    assert cache_utils.decompress_data(compressed) == raw_bytes

    payload = {"key": None}
    compressed_json = cache_utils.compress_data(payload)
    assert cache_utils.decompress_data(compressed_json) == {"key": None}

    none_json = cache_utils.compress_data({"key": "value"})
    with mock.patch("json.loads", return_value=None):
        assert cache_utils.decompress_data(none_json) == "None"

    assert cache_utils.decompress_data(None) is None


def test_cleanup_and_info_helpers():
    now = 1000.0
    cache_data = {
        "a": (now - 10, "expired"),
        "b": (now + 10, "active"),
        "c": "no-expiry",
    }
    expired = cache_utils.cleanup_expired_keys(cache_data, now)
    assert expired == 1
    assert "a" not in cache_data

    info = cache_utils.get_cache_info({"x": b"bytes", "y": 123})
    assert info["total_keys"] == 2
    assert info["avg_key_size"] > 0


def test_format_helpers():
    assert cache_utils.format_stat_line("missing", None) == "Missing: None"
    assert cache_utils.format_stat_line("hit_rate", 0.75).startswith("Hit Rate: 0.750000")
    assert cache_utils.format_stat_line("memory_usage_mb", 12.34).startswith("Memory Usage:")
    assert cache_utils.format_stat_line("custom_value", 5) == "Custom Value: 5"

    stats = {
        "total_requests": 4,
        "hits": 3,
        "misses": 1,
        "avg_response_time": 12,
        "memory_usage": "42MB",
    }
    rendered = cache_utils.format_cache_stats(stats)
    assert "Total Requests: 4" in rendered
    assert "Hit Rate: 0.75 (75%)" in rendered

    assert cache_utils.format_cache_stats({}) == "No cache statistics available"


def test_health_and_score_evaluations():
    assert cache_utils.get_cache_performance_score({}) == 0.0
    assert cache_utils.get_cache_performance_score({"hits": 3, "total_requests": 4}) == 75.0

    healthy_empty = cache_utils.is_cache_healthy({"total_requests": 0})
    assert healthy_empty is True

    unhealthy = cache_utils.is_cache_healthy(
        {"hits": 1, "total_requests": 4, "avg_response_time": 2000},
        thresholds={"min_hit_rate": 0.5, "max_response_time": 1500},
    )
    assert unhealthy is False


def test_parse_cache_config_variants():
    parsed = cache_utils.parse_cache_config("max_size=10,ttl=20,enabled=false,ratio=0.5")
    assert parsed["max_size"] == 10
    assert parsed["ttl"] == 20
    assert parsed["enabled"] is False
    assert pytest.approx(parsed["ratio"], 0.001) == 0.5

    fallback = cache_utils.parse_cache_config("malformed")
    assert fallback["max_size"] == 1000
    assert fallback["enabled"] is True

    clone = cache_utils.parse_cache_config({"enabled": False, "ttl": 1})
    assert clone == {"enabled": False, "ttl": 1}
    assert clone is not cache_utils.parse_cache_config({"enabled": False, "ttl": 1})


def test_prediction_cache_capacity_and_ttl(monkeypatch):
    fake = FakeTime()
    monkeypatch.setattr(real_time, "time", fake.time)
    monkeypatch.setattr(cache_utils.time, "time", fake.time)

    cache = cache_utils.PredictionCache(max_size=1, ttl_seconds=5)
    calls = {"count": 0}

    @cache
    def compute(x):
        calls["count"] += 1
        return x * 10

    assert compute(1) == 10
    assert cache.misses == 1 and cache.hits == 0

    fake.advance(1)
    assert compute(1) == 10
    assert cache.hits == 1

    fake.advance(10)
    assert compute(1) == 10
    assert cache.misses == 2

    fake.advance(1)
    assert compute(2) == 20
    assert cache.get_predicted_keys(10) == [cache._build_key((2,), {})]
    cache.update_prediction(cache._build_key((2,), {}), accessed=True)
    assert 0.5 < cache.predict_access(cache._build_key((2,), {})) <= 1.0
    assert cache.hit_rate > 0


def test_model_cache_ttl_and_capacity(monkeypatch):
    fake = FakeTime()
    monkeypatch.setattr(real_time, "time", fake.time)
    monkeypatch.setattr(cache_utils.time, "time", fake.time)

    calls = {"count": 0}

    def generator(value):
        calls["count"] += 1
        return value * 2

    cached = cache_utils.model_cache(max_size=1, ttl_seconds=2)(generator)

    assert cached(1) == 2
    assert calls["count"] == 1

    fake.advance(1)
    assert cached(1) == 2
    assert calls["count"] == 1

    fake.advance(2)
    assert cached(1) == 2
    assert calls["count"] == 2

    assert cached(2) == 4
    assert calls["count"] == 3

    assert cached(1) == 2
    assert calls["count"] == 4


def test_calculate_ttl_dynamic_adjustments():
    assert cache_utils.calculate_ttl("session:user", 100) == 3600
    assert cache_utils.calculate_ttl("config:item", 100) == 86400
    assert cache_utils.calculate_ttl("other", 100, access_count=10) == 75
    assert cache_utils.calculate_ttl("other", 100, access_count=200) == 150
    assert cache_utils.calculate_ttl("other", 100, hit_rate=0.9) == 100
    assert cache_utils.calculate_ttl("other", 100, hit_rate=0.1) == 80


def test_performance_monitor_metrics():
    monitor = cache_utils.PerformanceMonitor()
    monitor.record_metric("latency", 12.5, tags={"cache": "a"})
    assert monitor.get_metric("latency")["value"] == 12.5
    assert monitor.get_all_metrics() is not monitor.metrics

    monitor.start_operation("refresh")
    duration = monitor.end_operation("refresh")
    assert isinstance(duration, float)

    monitor.reset_metrics()
    assert monitor.get_all_metrics() == {}


def test_cache_statistics_tracks_operations():
    stats = cache_utils.CacheStatistics()
    stats.record_hit()
    stats.record_miss()
    stats.record_set()
    stats.record_delete()
    stats.record_eviction()
    stats.record_error()

    snapshot = stats.get_stats()
    assert snapshot["hits"] == 1
    assert snapshot["misses"] == 1
    assert snapshot["total_requests"] == 2
    assert stats.get_sets() == 1
    assert stats.get_deletes() == 1
    assert stats.get_evictions() == 1
    assert stats.get_errors() == 1
    assert stats.get_miss_rate() == pytest.approx(0.5)

    stats.reset()
    assert stats.get_total_requests() == 0


def test_time_utils_helpers(monkeypatch):
    fake = FakeTime(start=100.0)
    monkeypatch.setattr(real_time, "time", fake.time)

    timestamp = cache_utils.TimeUtils.get_current_timestamp()
    fake.advance(50)
    formatted = cache_utils.TimeUtils.format_timestamp(timestamp)
    from datetime import datetime
    assert formatted == datetime.fromtimestamp(timestamp).isoformat()
    assert cache_utils.TimeUtils.is_expired(timestamp, ttl=10) is True
    assert cache_utils.TimeUtils.is_expired(timestamp, ttl=-1) is False
    assert cache_utils.TimeUtils.calculate_remaining_ttl(timestamp, ttl=120) == 70


def test_expiration_manager_flow(monkeypatch):
    fake = FakeTime(start=200.0)
    monkeypatch.setattr(real_time, "time", fake.time)
    monkeypatch.setattr(cache_utils.time, "time", fake.time)

    manager = cache_utils.ExpirationManager()
    manager.set_expiration("a", ttl=10)
    fake.advance(5)
    manager.set_expiration("b", ttl=2)

    assert manager.is_expired("a") is False
    fake.advance(2.5)
    expired_subset = manager.cleanup_expired(keys=["b", "c"])
    assert expired_subset == ["b"]
    assert manager.is_expired("b") is False

    fake.advance(10)
    expired_all = manager.cleanup_expired()
    assert expired_all == ["a"]

    stats = manager.get_expiration_stats()
    assert stats["total"] == 0
    assert manager.get_remaining_ttl("missing") == 0


def test_thread_safety_manager_lock_cycle():
    manager = cache_utils.ThreadSafetyManager()
    token = manager.acquire_lock("demo")
    assert token is True
    manager.release_lock("demo")


def test_data_validator_and_operations():
    assert cache_utils.DataValidator.validate_data("ok") is True
    assert cache_utils.DataValidator.validate_data(None) is False
    assert cache_utils.DataValidator.sanitize_data({"x": 1}) == {"x": 1}

    class FakeCache:
        def __init__(self):
            self.store = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value):
            self.store[key] = value
            return True

        def clear(self):
            self.store.clear()
            return 0

        def size(self):
            return len(self.store)

    ops = cache_utils.CacheOperations(FakeCache())
    assert ops.set("a", 1) is True
    assert ops.get("a") == 1
    assert ops.delete("missing") is None
    assert ops.clear() == 0
    assert ops.size() == 0


def test_get_performance_monitor_returns_unique_instances():
    first = cache_utils.get_performance_monitor()
    second = cache_utils.get_performance_monitor()
    assert isinstance(first, cache_utils.PerformanceMonitor)
    assert first is not second

