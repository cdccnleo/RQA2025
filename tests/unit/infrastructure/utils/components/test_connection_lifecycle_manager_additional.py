#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConnectionLifecycleManager 额外单测，覆盖连接创建、销毁、过期清理与使用标记等关键路径。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pytest

from src.infrastructure.utils.components.connection_lifecycle_manager import (
    ConnectionLifecycleManager,
    ConnectionInfo,
)


class _DummyConnection:
    def __init__(self, should_fail_close: bool = False) -> None:
        self.closed = False
        self._fail = should_fail_close

    def close(self) -> None:
        if self._fail:
            raise RuntimeError("close boom")
        self.closed = True


def _make_info(connection: Optional[_DummyConnection] = None, *, last_used_delta: float = 0.0, created_delta: float = 0.0, use_count: int = 0, is_active: bool = False) -> ConnectionInfo:
    now = datetime.now()
    return ConnectionInfo(
        connection_id="test",
        created_at=now - timedelta(seconds=created_delta),
        last_used=now - timedelta(seconds=last_used_delta),
        use_count=use_count,
        is_active=is_active,
        connection=connection,
    )


def test_create_connection_success_and_failure() -> None:
    manager = ConnectionLifecycleManager(connection_factory=lambda: _DummyConnection())
    info = manager.create_connection()
    assert isinstance(info, ConnectionInfo)
    assert isinstance(info.connection, _DummyConnection)
    assert info.is_active is False

    # 工厂返回 None
    manager.connection_factory = lambda: None  # type: ignore[assignment]
    assert manager.create_connection() is None

    # 工厂抛异常
    def _boom():
        raise RuntimeError("factory error")

    manager.connection_factory = _boom  # type: ignore[assignment]
    assert manager.create_connection() is None


def test_destroy_connection_close_invocation() -> None:
    conn = _DummyConnection()
    info = _make_info(conn)
    manager = ConnectionLifecycleManager()
    assert manager.destroy_connection(info) is True
    assert conn.closed is True

    # close 抛异常时返回 False
    conn_fail = _DummyConnection(should_fail_close=True)
    info_fail = _make_info(conn_fail)
    assert manager.destroy_connection(info_fail) is False


def test_cleanup_expired_connections_respects_timeouts() -> None:
    manager = ConnectionLifecycleManager(idle_timeout=1.0, max_lifetime=5.0, max_usage=3)
    fresh = _make_info(last_used_delta=0.5, created_delta=1.0, use_count=2)
    idle_expired = _make_info(last_used_delta=2.0, created_delta=1.0)
    life_expired = _make_info(last_used_delta=0.5, created_delta=10.0)
    usage_expired = _make_info(last_used_delta=0.5, created_delta=1.0, use_count=5)

    expired = manager.cleanup_expired_connections([fresh, idle_expired, life_expired, usage_expired])
    assert idle_expired in expired
    assert life_expired in expired
    assert usage_expired in expired
    assert fresh not in expired


def test_ensure_min_connections_returns_needed_count() -> None:
    manager = ConnectionLifecycleManager()
    connections = [_make_info() for _ in range(2)]
    needed = manager.ensure_min_connections(connections, min_size=5)
    assert needed == 3
    assert manager.ensure_min_connections(connections, min_size=1) == 0


def test_update_and_release_connection_usage() -> None:
    manager = ConnectionLifecycleManager()
    info = _make_info()
    manager.update_connection_usage(info)
    assert info.use_count == 1
    assert info.is_active is True

    manager.mark_connection_released(info)
    assert info.is_active is False
    assert (datetime.now() - info.last_used).total_seconds() < 1.0

