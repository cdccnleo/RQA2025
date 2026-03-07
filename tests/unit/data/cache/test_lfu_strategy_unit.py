#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


import time
from datetime import datetime, timedelta

from src.data.cache.lfu_strategy import LFUStrategy
from src.data.cache.cache_manager import CacheEntry, CacheConfig


def _entry(value, count=0, created_offset=0):
    e = CacheEntry(key=str(value), value=value, ttl=None)
    # simulate access count and created_at offset
    e.access_count = count
    if created_offset != 0:
        e.created_at = (datetime.now() - timedelta(seconds=created_offset)).timestamp()
    return e


def test_evict_none_on_empty_cache():
    strat = LFUStrategy()
    evicted = strat.on_evict({}, CacheConfig())
    assert evicted is None


def test_evict_lowest_access_count_capacity_basic():
    strat = LFUStrategy()
    cache = {
        "a": _entry("a", count=5, created_offset=5),
        "b": _entry("b", count=1, created_offset=3),
        "c": _entry("c", count=2, created_offset=1),
    }
    victim = strat.on_evict(cache, CacheConfig())
    assert victim == "b"


def test_tie_breaker_earliest_created_when_same_frequency():
    strat = LFUStrategy()
    # same access_count for x,y,z; oldest created should be evicted
    cache = {
        "x": _entry("x", count=1, created_offset=10),  # oldest
        "y": _entry("y", count=1, created_offset=5),
        "z": _entry("z", count=1, created_offset=1),
    }
    victim = strat.on_evict(cache, CacheConfig())
    assert victim == "x"


def test_on_set_and_on_get_no_side_effects_but_access_updates_handled_elsewhere():
    strat = LFUStrategy()
    cfg = CacheConfig()
    cache = {}
    e = _entry("k", count=0)
    strat.on_set(cache, "k", e, cfg)
    strat.on_get(cache, "k", e, cfg)
    # behavior is no-op by design; ensure no exception and state unchanged
    assert e.access_count == 0


