#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对 memory_object_pool 的核心行为做最小化回归，覆盖初始化、回收、过期清理与内存优化流程。
"""

from __future__ import annotations

import gc
import threading
from collections import deque
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from src.infrastructure.utils.components import memory_object_pool as mop


def _make_pool(**kwargs: Any) -> mop.GenericObjectPool[dict]:
    created: List[dict] = []

    def factory() -> dict:
        item = {"id": len(created)}
        created.append(item)
        return item

    def reset(obj: dict) -> None:
        obj["reset"] = True

    pool = mop.GenericObjectPool(
        object_factory=factory,
        object_reset=reset,
        max_pool_size=kwargs.get("max_pool_size", 4),
        min_pool_size=kwargs.get("min_pool_size", 1),
        max_idle_time=kwargs.get("max_idle_time", 30),
        cleanup_interval=kwargs.get("cleanup_interval", 100),
    )
    return pool


def test_object_pool_initialization_and_reuse() -> None:
    pool = _make_pool()
    try:
        assert pool.metrics.current_pool_size == pool.min_pool_size

        wrapper = pool.get_object()
        obj = wrapper._object  # type: ignore[attr-defined]
        assert obj.get("reset") is True

        wrapper.return_to_pool()
        assert pool.metrics.pool_hits == 1
        assert pool.metrics.objects_reused == 1
    finally:
        pool.shutdown()


def test_object_pool_exhaustion_raises() -> None:
    pool = _make_pool(max_pool_size=1, min_pool_size=0)
    try:
        first = pool.get_object()
        with pytest.raises(RuntimeError):
            pool.get_object()
        first.return_to_pool()
    finally:
        pool.shutdown()


def test_cleanup_expired_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _make_pool(max_idle_time=1, cleanup_interval=100, min_pool_size=0)
    try:
        pool._pool.append(
            {
                "object": {"id": 99},
                "created_time": 0.0,
                "last_used_time": 0.0,
            }
        )
        pool.metrics.current_pool_size = 1

        monkeypatch.setattr(mop.time, "time", lambda: 10.0)
        pool._cleanup_expired_objects()

        assert pool.metrics.current_pool_size == 0
        assert len(pool._pool) == 0
        assert pool.metrics.objects_destroyed >= 1
    finally:
        pool.shutdown()


def test_pooled_object_wrapper_context_returns_to_pool() -> None:
    pool = _make_pool(min_pool_size=0)
    try:
        with pool.get_object() as obj:
            obj["used"] = True
        assert pool.metrics.pool_hits == 0  # first get was miss
        assert pool.metrics.objects_created == 1
        assert len(pool._pool) == 1
    finally:
        pool.shutdown()


def test_return_object_during_shutdown() -> None:
    pool = _make_pool(min_pool_size=0, max_pool_size=1)
    try:
        wrapper = pool.get_object()
        obj = wrapper._object  # type: ignore[attr-defined]
        pool.metrics.current_pool_size = 1
        pool._shutdown_event.set()
        pool.return_object(obj)
        assert pool.metrics.objects_destroyed >= 1
        assert pool.metrics.current_pool_size == 0
    finally:
        pool.shutdown()


def test_cleanup_worker_logs_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _make_pool(cleanup_interval=0.05)
    triggered = threading.Event()

    def boom():
        if not triggered.is_set():
            triggered.set()
            raise RuntimeError("cleanup-failed")

    monkeypatch.setattr(pool, "_cleanup_expired_objects", boom)

    with patch.object(mop.logger, "error") as mock_error:
        try:
            assert triggered.wait(timeout=1.0)
        finally:
            pool.shutdown()

        mock_error.assert_called()
        assert any("cleanup-failed" in str(args[0]) for args, _ in mock_error.call_args_list)


def test_memory_optimization_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeProcess:
        def __init__(self) -> None:
            self._rss = 256 * 1024 * 1024

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=self._rss)

    monkeypatch.setattr(mop.psutil, "Process", _FakeProcess)
    monkeypatch.setattr(mop.gc, "collect", lambda: 5)

    manager = mop.MemoryOptimizationManager()
    try:
        result = manager.optimize_memory_usage()
        assert "memory_saved" in result
        assert result["collected_objects"] == 5
        assert "processor_pool_stats" in result
        assert "market_data_pool_stats" in result
    finally:
        manager.data_processor_pool.pool.shutdown()
        manager.market_data_pool.pool.shutdown()


def test_memory_optimizer_monitor_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeProcess:
        def memory_info(self):
            return SimpleNamespace(rss=128 * 1024 * 1024, vms=256 * 1024 * 1024)

        def memory_percent(self):
            return 12.5

    monkeypatch.setattr(mop.psutil, "Process", _FakeProcess)

    manager = mop.MemoryOptimizationManager()
    try:
        monitor_info = manager.monitor_memory_usage()
        assert monitor_info["rss"] == 128
        assert monitor_info["vms"] == 256
        assert monitor_info["percent"] == 12.5
    finally:
        manager.shutdown()


def test_memory_performance_helpers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    ticker = {"value": 0.0}

    def fake_time() -> float:
        ticker["value"] += 0.001
        return ticker["value"]

    monkeypatch.setattr(mop.time, "time", fake_time)

    class _StatsPool:
        def __init__(self, hit_rate: float, pool_size: int) -> None:
            self._stats = {"metrics": {"hit_rate": hit_rate}, "pool_size": pool_size}

        def get_stats(self) -> Dict[str, Any]:
            return self._stats

    class _StubMarketPool:
        def __init__(self) -> None:
            self.pool = _StatsPool(80.0, 2)

        def get_market_data(self):
            class _Wrapper:
                def __init__(self) -> None:
                    self._obj = {
                        "symbol": "",
                        "price": 0.0,
                        "volume": 0,
                        "timestamp": 0.0,
                        "metadata": {},
                        "indicators": {},
                    }

                def __enter__(self) -> Dict[str, Any]:
                    return self._obj

                def __exit__(self, exc_type, exc_val, exc_tb) -> None:
                    return None

            return _Wrapper()

    class _StubProcessorPool:
        def __init__(self) -> None:
            self.pool = _StatsPool(65.0, 1)

    class _StubManager:
        def __init__(self) -> None:
            self.market_data_pool = _StubMarketPool()
            self.data_processor_pool = _StubProcessorPool()
            self._memory_values = deque(
                [
                    {"rss": 200.0},
                    {"rss": 180.0},
                    {"rss": 170.0},
                ]
            )

        def monitor_memory_usage(self) -> Dict[str, float]:
            if self._memory_values:
                return self._memory_values.popleft()
            return {"rss": 160.0}

        def optimize_memory_usage(self) -> Dict[str, Any]:
            return {
                "collected_objects": 4,
                "current_memory": 150.0,
                "memory_saved": 10.0,
                "processor_pool_stats": self.data_processor_pool.pool.get_stats(),
                "market_data_pool_stats": self.market_data_pool.pool.get_stats(),
                "optimization_timestamp": fake_time(),
            }

    manager = _StubManager()

    traditional = mop._test_traditional_object_creation(manager)
    assert traditional["memory"]["rss"] == 200.0

    pooled = mop._test_pooled_object_creation(manager)
    assert pooled["memory"]["rss"] == 180.0

    comparison = mop._calculate_performance_comparison(traditional, pooled)
    assert "time_improvement" in comparison
    assert "memory_efficiency" in comparison

    mop._print_memory_test_results(
        comparison,
        manager,
        manager.optimize_memory_usage(),
        traditional,
        pooled,
    )
    printed = capsys.readouterr().out
    assert "性能对比结果" in printed
    assert "对象池使用统计" in printed

    summary = mop._prepare_memory_test_results(traditional, pooled, comparison, manager.optimize_memory_usage())
    assert summary["traditional_time"] == traditional["time"]
    assert summary["pooled_memory"] == pooled["memory"]["rss"]

    tmp_dir = tmp_path_factory.mktemp("memory_pool_perf")
    monkeypatch.chdir(tmp_dir)
    monkeypatch.setattr(mop, "MemoryOptimizationManager", _StubManager)
    perf_summary = mop.performance_test()
    assert "traditional_time" in perf_summary
    assert "optimization_result" in perf_summary
    assert not list(tmp_dir.glob("test_file_*.txt"))
    assert not list(tmp_dir.glob("output_*.txt"))

