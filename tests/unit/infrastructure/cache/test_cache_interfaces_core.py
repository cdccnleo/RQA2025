import time
from types import SimpleNamespace

import pytest

from src.infrastructure.cache.interfaces.cache_interfaces import (
    AccessPattern,
    CacheEntry,
    CacheEvictionStrategy,
    CacheStats,
    EvictionStrategyImpl,
)


def _make_state(current_size=10, max_size=5):
    return {"current_size": current_size, "max_size": max_size}


def test_eviction_strategy_lru_respects_recent_access(monkeypatch):
    impl = EvictionStrategyImpl(CacheEvictionStrategy.LRU)
    key = "session"
    impl._access_times[key] = time.time() - 400

    assert impl.should_evict(key, object(), _make_state()) is True

    impl._access_times[key] = time.time()
    assert impl.should_evict(key, object(), _make_state()) is False


def test_eviction_strategy_lfu(monkeypatch):
    impl = EvictionStrategyImpl(CacheEvictionStrategy.LFU)
    key = "item"
    impl._access_counts[key] = 0
    assert impl.should_evict(key, object(), _make_state()) is True

    impl._access_counts[key] = 10
    assert impl.should_evict(key, object(), _make_state()) is False


def test_eviction_strategy_fifo_order_tracking():
    impl = EvictionStrategyImpl(CacheEvictionStrategy.FIFO)
    impl.record_insertion("first")
    impl.record_insertion("second")

    assert impl.should_evict("first", object(), _make_state()) is True
    assert impl.should_evict("second", object(), _make_state()) is False


def test_eviction_strategy_ttl_supports_multiple_formats(monkeypatch):
    impl = EvictionStrategyImpl(CacheEvictionStrategy.TTL)
    expired_entry = SimpleNamespace(expiry_time=time.time() - 1)

    assert impl.should_evict("expired", expired_entry, _make_state()) is True

    dict_entry = {"created_at": time.time() - 500, "ttl": 200}
    assert impl.should_evict("dict", dict_entry, _make_state()) is True

    fresh_entry = {"created_at": time.time(), "ttl": 200}
    assert impl.should_evict("fresh", fresh_entry, _make_state()) is False


def test_eviction_strategy_random(monkeypatch):
    impl = EvictionStrategyImpl(CacheEvictionStrategy.RANDOM)

    monkeypatch.setattr("random.random", lambda: 0.9)
    assert impl.should_evict("rand", object(), _make_state()) is True

    monkeypatch.setattr("random.random", lambda: 0.1)
    assert impl.should_evict("rand", object(), _make_state()) is False


def test_eviction_strategy_records_and_removes_keys():
    impl = EvictionStrategyImpl(CacheEvictionStrategy.LRU)
    impl.record_access("key")
    assert "key" in impl._access_times
    assert "key" in impl._access_counts

    impl.record_insertion("key")
    assert "key" in impl._insertion_order

    impl.remove_key("key")
    assert "key" not in impl._access_times
    assert "key" not in impl._insertion_order


def test_cache_entry_and_stats_helpers():
    entry = CacheEntry("token", "value", ttl=60)
    assert entry.key == "token"
    assert entry.ttl == 60
    assert entry.access_count == 0

    stats = CacheStats()
    stats.hits += 1
    stats.evictions += 2
    assert stats.hits == 1
    assert stats.evictions == 2


def test_access_pattern_enum_values():
    assert AccessPattern.SEQUENTIAL.value == "sequential"
    assert AccessPattern.RANDOM.value == "random"

