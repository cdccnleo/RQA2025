import time

import pytest

from src.infrastructure.monitoring.core.smart_cache import CacheEntry, SmartCache
from datetime import datetime
import threading


@pytest.fixture
def cache():
    sc = SmartCache(max_size_mb=0.0005, default_ttl_seconds=1)
    yield sc
    sc.shutdown()


def test_set_and_get(cache):
    assert cache.set("key", "value", ttl_seconds=1) is True
    assert cache.get("key") == "value"
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["entries"] == 1


def test_ttl_expiration(cache):
    cache.set("ttl_key", "value", ttl_seconds=0.01)
    time.sleep(0.05)
    assert cache.get("ttl_key") is None
    stats = cache.get_stats()
    assert stats["misses"] >= 1


def test_max_size_evicts(cache):
    cache.set("a", "x" * 400)
    cache.set("b", "y" * 400)
    cache.set("c", "z" * 400)
    assert cache.contains("a") is False
    assert cache.contains("c") is True
    stats = cache.get_stats()
    assert stats["evictions"] >= 1


def test_delete_and_clear(cache):
    cache.set("x", 10)
    cache.delete("x")
    assert cache.contains("x") is False
    cache.set("y", 20)
    cache.clear()
    assert cache.contains("y") is False
    assert cache.get_stats()["entries"] == 0


def test_cache_entry_methods():
    entry = CacheEntry("k", "v", ttl_seconds=1)
    assert entry.is_expired() is False
    entry.access()
    assert entry.access_count == 1
    data = entry.to_dict()
    assert data["key"] == "k"
    assert "size_bytes" in data


def test_get_access_patterns(cache):
    cache.set("pattern", 1)
    cache.get("pattern")
    cache.get("pattern")
    patterns = cache.get_access_patterns()
    assert patterns["total_patterns"] >= 1


def test_get_recommendations(cache):
    cache.set("p", 1)
    cache.get("missing")
    recs = cache.get_recommendations()
    assert isinstance(recs, list)


@pytest.mark.parametrize("hit_rate, utilization, evictions, sets, expect_types", [
    (20.0, 50.0, 0, 10, {"hit_rate_optimization"}),
    (80.0, 96.0, 0, 10, {"size_optimization"}),
    (80.0, 50.0, 20, 100, {"eviction_optimization"}),
])
def test_get_recommendations_branches(cache, monkeypatch, hit_rate, utilization, evictions, sets, expect_types):
    def fake_stats():
        return {
            "hit_rate": hit_rate,
            "utilization_percent": utilization,
            "evictions": evictions,
            "sets": sets,
            "entries": 1,
            "size_bytes": 10,
            "size_mb": 0.01,
            "max_size_mb": 1.0,
            "hits": 1,
            "misses": 1,
            "deletes": 0,
            "uptime_seconds": 1.0,
            "requests_per_second": 1.0,
        }

    monkeypatch.setattr(cache, "get_stats", fake_stats)
    monkeypatch.setattr(cache, "get_access_patterns", lambda: {"patterns": {}, "total_patterns": 0})

    types = {rec["type"] for rec in cache.get_recommendations()}
    assert expect_types <= types


def test_get_recommendations_hot_keys(cache, monkeypatch):
    monkeypatch.setattr(
        cache,
        "get_stats",
        lambda: {
            "hit_rate": 80.0,
            "utilization_percent": 10.0,
            "evictions": 0,
            "sets": 10,
            "entries": 1,
            "size_bytes": 10,
            "size_mb": 0.01,
            "max_size_mb": 1.0,
            "hits": 1,
            "misses": 0,
            "deletes": 0,
            "uptime_seconds": 1.0,
            "requests_per_second": 1.0,
        },
    )

    now_iso = datetime.now().isoformat()

    monkeypatch.setattr(
        cache,
        "get_access_patterns",
        lambda: {
            "total_patterns": 1,
            "patterns": {
                "hot": {"frequency_per_hour": 20.0, "access_count": 5, "last_access": now_iso, "time_span_hours": 0.1}
            },
        },
    )

    recs = cache.get_recommendations()
    assert any(rec["type"] == "access_pattern" for rec in recs)


def test_get_health_status(cache):
    cache.set("health", "value")
    health = cache.get_health_status()
    assert health["status"] in {"healthy", "warning"}


def test_get_health_status_warning_conditions(cache, monkeypatch):
    def fake_stats():
        return {
            "hit_rate": 10.0,
            "utilization_percent": 96.0,
            "evictions": 5,
            "sets": 4,
            "entries": 1,
            "size_bytes": 10,
            "size_mb": 0.01,
            "max_size_mb": 1.0,
            "hits": 1,
            "misses": 10,
            "deletes": 0,
            "uptime_seconds": 1.0,
            "requests_per_second": 1.0,
        }

    monkeypatch.setattr(cache, "get_stats", fake_stats)
    cache.cleanup_thread = None

    health = cache.get_health_status()
    assert health["status"] == "warning"
    assert any("缓存命中率过低" in issue for issue in health["issues"])
    assert any("缓存空间利用率过高" in issue for issue in health["issues"])
    assert any("驱逐过于频繁" in issue for issue in health["issues"])
    assert "清理线程未运行" in health["issues"]


def test_get_health_status_error(cache, monkeypatch):
    monkeypatch.setattr(cache, "get_stats", lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    health = cache.get_health_status()
    assert health["status"] == "error"
    assert health["error"] == "fail"


def test_warmup(cache):
    cache.warmup({"a": 1, "b": 2}, ttl_seconds=1)
    assert cache.get("a") == 1


def test_contains_respects_expiry(cache, monkeypatch):
    cache.set("k", "v", ttl_seconds=0.01)
    time.sleep(0.05)
    assert cache.contains("k") is False


def test_cleanup_thread_catches_exception(monkeypatch):
    sc = SmartCache(max_size_mb=0.0001)

    def fake_sleep(seconds):
        sc.stop_event.set()
        raise RuntimeError("sleep break")

    monkeypatch.setattr("time.sleep", fake_sleep)
    thread = threading.Thread(target=sc._cleanup_expired_entries)
    thread.start()
    thread.join(timeout=0.5)

    sc.shutdown()
