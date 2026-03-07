#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CacheStore 行为测试，验证 TTL、统计与清理逻辑。"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.features.store.cache_store import CacheStore

pytestmark = pytest.mark.features


def test_set_get_and_stats_update():
    cache = CacheStore(default_ttl=10)
    assert cache.get("missing") is None

    payload = pd.DataFrame({"value": [1, 2, 3]})
    assert cache.set("asset", payload)
    assert cache.get("asset").equals(payload)

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["sets"] == 1


def test_expired_entries_are_evicted_on_access_and_cleanup():
    cache = CacheStore(default_ttl=1)
    cache.set("temp", 42)

    # 模拟过期
    cache._store["temp"].expires_at = datetime.utcnow() - timedelta(seconds=1)
    assert cache.get("temp") is None

    cache.set("temp2", 13)
    cache._store["temp2"].expires_at = datetime.utcnow() - timedelta(seconds=1)
    assert cache.cleanup_expired() == 1


def test_delete_and_clear_reset_entries():
    cache = CacheStore(default_ttl=None)
    cache.set("foo", 1)
    cache.set("bar", 2)

    assert cache.delete("foo")
    assert cache.get("foo") is None

    assert cache.clear()
    assert cache.get("bar") is None

    stats = cache.stats()
    assert stats["clears"] == 1

