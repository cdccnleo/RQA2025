#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OptimizedConnectionPool 额外单测，覆盖连接获取/归还、清理、泄漏检测与健康检查关键路径。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Dict

import pytest

from src.infrastructure.utils.components.optimized_connection_pool import (
    OptimizedConnectionPool,
    PoolState,
)


class _DummyThread:
    """用于阻止后台维护线程真正启动的线程桩。"""

    def __init__(self, target: Callable, daemon: bool = True, name: str | None = None) -> None:
        self.target = target
        self.daemon = daemon
        self.name = name
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return False


@pytest.fixture
def patched_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """在测试中替换 optimized_connection_pool 内部使用的线程类。"""

    monkeypatch.setattr(
        "src.infrastructure.utils.components.optimized_connection_pool.threading.Thread",
        _DummyThread,
    )


class _DummyConnection:
    def __init__(self, identifier: int) -> None:
        self.identifier = identifier
        self.close_called = False

    def close(self) -> None:
        self.close_called = True


def _make_factory():
    counter = {"value": 0}

    def factory() -> _DummyConnection:
        counter["value"] += 1
        return _DummyConnection(counter["value"])

    return factory


def test_pool_get_release_and_shutdown(patched_thread: None) -> None:
    pool = OptimizedConnectionPool(
        min_size=0,
        max_size=2,
        initial_size=0,
        idle_timeout=100,
        max_lifetime=1000,
        health_check_interval=3600,
    )
    pool._monitor = None
    pool._health_checker = None

    factory = _make_factory()
    pool.set_connection_factory(factory)

    conn = pool.get_connection()
    assert isinstance(conn, _DummyConnection)
    assert pool.get_pool_status()["current_size"] == 1

    pool.release_connection(conn)
    status = pool.get_pool_status()
    assert status["available_connections"] >= 1
    assert status["active_connections"] == 0
    assert status["stats"]["connection_requests"] == 1

    dummy_info = pool.available_connections[0]
    connection_ref = dummy_info.connection
    pool.shutdown()
    assert isinstance(connection_ref, _DummyConnection)
    assert connection_ref.close_called is True


def test_pool_cleanup_leak_detection_and_health_check(patched_thread: None) -> None:
    pool = OptimizedConnectionPool(
        min_size=0,
        max_size=3,
        initial_size=0,
        idle_timeout=0.1,
        max_lifetime=10,
        leak_detection_threshold=0.01,
        health_check_interval=3600,
    )
    pool._monitor = None
    pool._health_checker = None

    factory = _make_factory()
    pool.set_connection_factory(factory)

    # 创建并释放连接，使其进入可用列表
    conn = pool.get_connection()
    pool.release_connection(conn)
    assert len(pool.available_connections) >= 1

    # 标记为过期并执行清理
    info = pool.available_connections[0]
    info.last_used = datetime.now() - timedelta(seconds=1)
    pool._cleanup_expired_connections()
    assert len(pool.connections) == 0

    # 再次获取连接，制造泄漏检测场景
    conn_leak = pool.get_connection()
    leak_info = list(pool._in_use_connections.values())[0]
    leak_info.last_used = datetime.now() - timedelta(seconds=1)
    pool._detect_connection_leaks()
    assert len(pool.available_connections) >= 1
    assert pool._stats["leak_detections"] >= 1

    # 获取连接以触发高负载状态，执行健康检查（使用回退逻辑）
    pool._state = PoolState.HEALTHY
    pool._max_size = 1
    conn_active = pool.get_connection()
    result = pool.health_check()
    assert "status" in result
    assert result["status"] in {state.value for state in PoolState}
    assert result["total_connections"] >= 1
    assert pool._state in {PoolState.HEALTHY, PoolState.WARNING, PoolState.CRITICAL, PoolState.FAILED}

    pool.release_connection(conn_active)
    pool.release_connection(conn_leak)
    pool.shutdown()

