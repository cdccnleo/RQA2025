#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QueryCacheManager 单元测试，覆盖缓存命中/过期/统计等核心路径。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.infrastructure.utils.components.query_cache_manager import QueryCacheManager


def make_request(query_type="realtime", storage_type="influxdb", params=None):
    return SimpleNamespace(
        query_type=query_type,
        storage_type=storage_type,
        params=params or {"symbol": "TEST"},
    )


def test_cache_result_hits_and_expires(monkeypatch):
    manager = QueryCacheManager(cache_ttl=2)
    base_time = 1_000.0

    monkeypatch.setattr(
        "src.infrastructure.utils.components.query_cache_manager.time.time",
        lambda: base_time,
    )
    request = make_request(params={"page": 1})
    result_obj = {"data": "cached"}
    manager.cache_result(request, result_obj)

    # 命中
    cached = manager.get_cached_result(request)
    assert cached is result_obj
    assert manager.cache_hits == 1

    # 超时后失效
    monkeypatch.setattr(
        "src.infrastructure.utils.components.query_cache_manager.time.time",
        lambda: base_time + 5,
    )
    assert manager.get_cached_result(request) is None
    assert manager.cache_misses == 1


def test_disabled_cache_no_store():
    manager = QueryCacheManager(cache_enabled=False)
    request = make_request()
    manager.cache_result(request, {"data": 1})
    assert manager.get_cached_result(request) is None

    manager.set("key", "value")
    assert manager.get("key") is None


def test_set_get_interfaces_and_clear(monkeypatch):
    manager = QueryCacheManager(cache_ttl=10)
    base_time = 2_000.0
    monkeypatch.setattr(
        "src.infrastructure.utils.components.query_cache_manager.time.time",
        lambda: base_time,
    )

    manager.set("manual-key", {"value": 1})
    assert manager.get("manual-key") == {"value": 1}
    assert manager.cache_hits == 1

    # 过期后 miss
    monkeypatch.setattr(
        "src.infrastructure.utils.components.query_cache_manager.time.time",
        lambda: base_time + 20,
    )
    assert manager.get("manual-key") is None
    assert manager.cache_misses == 1

    manager.clear_cache()
    stats = manager.get_cache_statistics()
    assert stats["cache_size"] == 0
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 0


def test_cleanup_expired_cache(monkeypatch):
    manager = QueryCacheManager(cache_ttl=1)
    t0 = 3_000.0
    monkeypatch.setattr(
        "src.infrastructure.utils.components.query_cache_manager.time.time",
        lambda: t0,
    )

    manager.set("key1", {"v": 1})
    manager.set("key2", {"v": 2})

    monkeypatch.setattr(
        "src.infrastructure.utils.components.query_cache_manager.time.time",
        lambda: t0 + 5,
    )
    cleaned = manager.cleanup_expired_cache()
    assert cleaned == 2
    assert manager.get("key1") is None
    assert manager.get_cache_statistics()["cache_size"] == 0

