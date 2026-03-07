#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced_connection_pool 额外单测，覆盖连接获取/归还与维护辅助逻辑。
"""

from __future__ import annotations

import time
from typing import Any, List

import pytest

from src.infrastructure.utils.components import advanced_connection_pool as pool_module


class DummyConn:
    def __init__(self, name: str):
        self.name = name
        self.destroyed = False

    def __hash__(self) -> int:
        return id(self)


@pytest.fixture
def pool(monkeypatch: pytest.MonkeyPatch) -> pool_module.OptimizedConnectionPool:
    """构造一个不启动监控线程的连接池实例。"""

    class _DummyThread:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            return None

    monkeypatch.setattr(pool_module.threading, "Thread", _DummyThread)
    pool = pool_module.OptimizedConnectionPool(
        max_connections=3,
        min_connections=0,
        max_idle_time=0.5,
        max_lifetime=0.5,
        connection_timeout=0.05,
        retry_attempts=2,
    )
    pool.metrics.reset()
    pool._pool.clear()
    pool._active_connections.clear()
    pool.set_connection_factory(lambda: DummyConn("factory"))
    pool.set_connection_validator(lambda conn: True)
    pool.set_connection_destroyer(lambda conn: setattr(conn, "destroyed", True))
    yield pool
    pool.shutdown()


def test_get_connection_prefers_idle_and_updates_metrics(pool: pool_module.OptimizedConnectionPool) -> None:
    conn = DummyConn("idle_conn")
    pool._pool.append(
        {"connection": conn, "created_time": time.time(), "last_used_time": time.time()}
    )
    pool.metrics.idle_connections = 1

    wrapper = pool.get_connection()

    assert isinstance(wrapper, pool_module.ConnectionWrapper)
    assert wrapper.connection.name == "idle_conn"
    assert pool.metrics.active_connections == 1
    assert pool.metrics.connection_hits == 1


def test_get_connection_skips_expired_and_invalid_then_creates_new(
    pool: pool_module.OptimizedConnectionPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_conn = DummyConn("expired")
    bad_conn = DummyConn("bad")

    pool._pool.extend(
        [
            {"connection": old_conn, "created_time": time.time() - 10.0, "last_used_time": time.time()},
            {"connection": bad_conn, "created_time": time.time(), "last_used_time": time.time()},
        ]
    )
    pool.metrics.idle_connections = 2

    validator_calls: List[Any] = []

    def _validator(conn: Any) -> bool:
        validator_calls.append(conn)
        return conn is not bad_conn

    pool.set_connection_validator(_validator)

    new_conn = DummyConn("new")
    pool.set_connection_factory(lambda: new_conn)

    wrapper = pool.get_connection()

    assert wrapper.connection is new_conn
    assert old_conn.destroyed is True
    assert bad_conn.destroyed is True
    assert pool.metrics.destroyed_connections >= 2
    assert validator_calls == [bad_conn]


def test_wait_for_connection_timeout_records_timeout(pool: pool_module.OptimizedConnectionPool, monkeypatch):
    monkeypatch.setattr(pool_module.time, "sleep", lambda _: None)
    start = time.time()
    waited = pool._wait_for_connection(timeout=0.0, start_time=start)
    assert waited is None
    pool._handle_connection_timeout()
    assert pool.metrics.connection_timeouts == 1


def test_return_connection_invalid_triggers_destroy(pool: pool_module.OptimizedConnectionPool):
    conn = DummyConn("active")
    pool._active_connections.add(conn)
    pool.metrics.active_connections = 1

    pool.set_connection_validator(lambda _: False)
    pool.return_connection(conn)

    assert pool.metrics.active_connections == 0
    assert conn.destroyed is True


def test_maintain_min_connections_creates_needed(pool: pool_module.OptimizedConnectionPool):
    created: List[Any] = []

    def factory() -> Any:
        conn = DummyConn(f"conn-{len(created)}")
        created.append(conn)
        return conn

    pool.set_connection_factory(factory)
    pool.min_connections = 2

    pool.maintain_min_connections()

    assert len(pool._pool) == 2
    assert pool.metrics.idle_connections == 2


def test_cleanup_expired_connections_removes_old(pool: pool_module.OptimizedConnectionPool):
    fresh = DummyConn("fresh")
    stale = DummyConn("stale")

    now = time.time()
    pool._pool.extend(
        [
            {"connection": fresh, "created_time": now, "last_used_time": now},
            {"connection": stale, "created_time": now - 10.0, "last_used_time": now - 10.0},
        ]
    )
    pool.cleanup_expired_connections()

    assert len(pool._pool) == 1
    assert pool._pool[0]["connection"] is fresh
    assert stale.destroyed is True


def test_get_stats_returns_expected_structure(pool: pool_module.OptimizedConnectionPool):
    pool._pool.append({"connection": object(), "created_time": time.time(), "last_used_time": time.time()})
    result = pool.get_stats()
    assert result["pool_size"] == 1
    assert "metrics" in result

