#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Object Pool 额外单测，覆盖对象获取/归还、清理与内存优化管理流程。
"""

from __future__ import annotations

import gc
import logging
import time
from types import SimpleNamespace
from typing import Dict, List

import pytest

from src.infrastructure.utils.components.memory_object_pool import (
    GenericObjectPool,
    PooledObjectWrapper,
    MemoryOptimizationManager,
)


@pytest.fixture
def sample_pool() -> GenericObjectPool[Dict[str, int]]:
    """创建一个用于测试的对象池，测试结束后关闭。"""

    counter: List[int] = [0]

    def factory() -> Dict[str, int]:
        counter[0] += 1
        return {"value": counter[0]}

    def reset(obj: Dict[str, int]) -> None:
        obj["value"] = 0

    pool = GenericObjectPool(
        object_factory=factory,
        object_reset=reset,
        max_pool_size=2,
        min_pool_size=1,
        max_idle_time=10,
        cleanup_interval=3600,  # 避免清理线程频繁运行
    )

    try:
        yield pool
    finally:
        pool.shutdown()


def test_get_object_reuse_and_metrics(sample_pool: GenericObjectPool[Dict[str, int]]) -> None:
    with sample_pool.get_object() as obj1:
        obj1["value"] = 42

    wrapper = sample_pool.get_object()
    try:
        assert isinstance(wrapper, PooledObjectWrapper)
        assert wrapper._object["value"] == 0  # reset 后归零
    finally:
        wrapper.return_to_pool()

    assert sample_pool.metrics.objects_reused >= 1
    assert sample_pool.metrics.pool_hits >= 1
    assert sample_pool.metrics.pool_misses >= 0


def test_object_creation_and_pool_exhaustion(sample_pool: GenericObjectPool[Dict[str, int]]) -> None:
    single_pool = GenericObjectPool(
        object_factory=lambda: {"id": time.time()},
        max_pool_size=1,
        min_pool_size=0,
        max_idle_time=5,
        cleanup_interval=3600,
    )
    wrapper = single_pool.get_object()
    with pytest.raises(RuntimeError):
        single_pool.get_object()
    wrapper.return_to_pool()
    single_pool.shutdown()


def test_return_object_when_shutdown(sample_pool: GenericObjectPool[Dict[str, int]]) -> None:
    wrapper = sample_pool.get_object()
    obj = wrapper._object
    sample_pool._shutdown_event.set()
    sample_pool.metrics.current_pool_size = sample_pool.max_pool_size + 1
    sample_pool.return_object(obj)

    assert sample_pool.metrics.objects_destroyed >= 1
    assert sample_pool.metrics.current_pool_size <= sample_pool.max_pool_size


def test_cleanup_expired_objects(sample_pool: GenericObjectPool[Dict[str, int]]) -> None:
    current = time.time()
    sample_pool._pool.clear()
    sample_pool.metrics.current_pool_size = 2
    sample_pool._pool.append({"object": {"value": 1}, "created_time": current, "last_used_time": current - 100})
    sample_pool._pool.append({"object": {"value": 2}, "created_time": current, "last_used_time": current})

    sample_pool._cleanup_expired_objects()
    assert len(sample_pool._pool) == 1
    assert sample_pool.metrics.objects_destroyed >= 1


def test_get_stats_returns_metrics(sample_pool: GenericObjectPool[Dict[str, int]]) -> None:
    stats = sample_pool.get_stats()
    assert stats["max_pool_size"] == sample_pool.max_pool_size
    assert "metrics" in stats
    assert "pool_hits" in stats["metrics"]


def test_pooled_object_wrapper_returns_only_once(sample_pool: GenericObjectPool[Dict[str, int]]) -> None:
    wrapper = sample_pool.get_object()
    obj = wrapper._object
    wrapper.return_to_pool()
    initial_pool_len = len(sample_pool._pool)
    wrapper.return_to_pool()
    assert len(sample_pool._pool) == initial_pool_len  # 再次归还不会重复入池
    assert obj in [entry["object"] for entry in sample_pool._pool]


def test_memory_optimization_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeProcess:
        def __init__(self):
            self._rss = 256 * 1024 * 1024  # 256 MB

        def memory_info(self):
            return SimpleNamespace(rss=self._rss)

    monkeypatch.setattr("psutil.Process", _FakeProcess)
    monkeypatch.setattr(gc, "collect", lambda: 5)

    manager = MemoryOptimizationManager()
    try:
        result = manager.optimize_memory_usage()
        assert result["collected_objects"] == 5
        assert result["current_memory"] == 256.0
        assert "processor_pool_stats" in result
        assert "market_data_pool_stats" in result
    finally:
        manager.data_processor_pool.pool.shutdown()
        manager.market_data_pool.pool.shutdown()

