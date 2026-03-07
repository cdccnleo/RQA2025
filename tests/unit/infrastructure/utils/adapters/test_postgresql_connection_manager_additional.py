#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQLConnectionManager 额外单测，覆盖连接、健康检查与日志兜底分支。
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from src.infrastructure.utils.adapters.postgresql_connection_manager import (
    PostgreSQLConnectionManager,
    ConnectionStatus,
)


class _DummyCursor:
    def __init__(self, version: str = "PostgreSQL 16.0"):
        self.last_query: str = ""
        self.version = version

    def __enter__(self) -> "_DummyCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str) -> None:
        self.last_query = query
        if "fail" in query:
            raise RuntimeError("forced failure")

    def fetchone(self):
        if self.last_query == "SELECT version()":
            return (self.version,)
        return (1,)


class _DummyConnection:
    def __init__(self, cursor: _DummyCursor):
        self.cursor_obj = cursor
        self.closed = False

    def cursor(self) -> _DummyCursor:
        return self.cursor_obj

    def close(self) -> None:
        self.closed = True


def _install_psycopg2(monkeypatch: pytest.MonkeyPatch, connect_impl) -> None:
    module = SimpleNamespace(connect=connect_impl)
    monkeypatch.setitem(sys.modules, "psycopg2", module)


def test_connect_success_and_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _DummyCursor()
    connection = _DummyConnection(cursor)
    _install_psycopg2(monkeypatch, lambda **kwargs: connection)

    manager = PostgreSQLConnectionManager()
    config = {"host": "db.local", "port": 6543, "database": "testdb", "user": "tester"}

    assert manager.connect(config) is True
    assert manager.connected is True
    info = manager.get_connection_info()
    assert info["host"] == "db.local"
    assert info["port"] == 6543
    assert info["database"] == "testdb"

    assert manager.disconnect() is True
    assert connection.closed is True
    assert manager.connected is False


def test_connect_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**_kwargs):
        raise RuntimeError("connect failed")

    _install_psycopg2(monkeypatch, _boom)

    manager = PostgreSQLConnectionManager()
    with pytest.raises(RuntimeError):
        manager.connect({})
    assert manager.connected is False


def test_health_check_connected_path(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _DummyCursor(version="PostgreSQL 15.3")
    connection = _DummyConnection(cursor)
    _install_psycopg2(monkeypatch, lambda **kwargs: connection)

    manager = PostgreSQLConnectionManager()
    manager.connect({"database": "prod"})

    result = manager.health_check()
    assert result.is_healthy is True
    assert result.details["version"] == "PostgreSQL 15.3"
    assert result.details["database"] == "prod"
    assert result.response_time >= 0.0


def test_health_check_not_connected() -> None:
    manager = PostgreSQLConnectionManager()
    result = manager.health_check()
    assert result.is_healthy is False
    assert result.message == "数据库未连接"


def test_health_check_failure_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailCursor(_DummyCursor):
        def execute(self, query: str) -> None:
            raise RuntimeError("cursor boom")

    connection = _DummyConnection(_FailCursor())
    manager = PostgreSQLConnectionManager()
    manager.client = connection
    manager.connected = True

    result = manager.health_check()
    assert result.is_healthy is False
    assert "boom" in result.message


def test_get_connection_status_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _DummyCursor()
    connection = _DummyConnection(cursor)
    manager = PostgreSQLConnectionManager()

    manager.client = connection
    manager.connected = True
    assert manager.get_connection_status() is ConnectionStatus.CONNECTED

    class _BrokenCursor(_DummyCursor):
        def execute(self, query: str) -> None:
            raise ValueError("bad connection")

    broken = _DummyConnection(_BrokenCursor())
    manager.client = broken
    manager.connected = True
    assert manager.get_connection_status() is ConnectionStatus.ERROR
    assert manager.connected is False


def test_get_database_version_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    class _VersionlessCursor(_DummyCursor):
        def fetchone(self):
            return None

    manager = PostgreSQLConnectionManager()
    manager.client = _DummyConnection(_VersionlessCursor())
    assert manager._get_database_version() == "unknown"


def test_safe_log_with_mock_handler() -> None:
    handler = logging.Handler()
    handler.level = "mocked"
    logger = logging.getLogger("src.infrastructure.utils.adapters.postgresql_connection_manager")
    logger.addHandler(handler)
    try:
        manager = PostgreSQLConnectionManager()
        manager._safe_log(logging.INFO, "trigger fallback")
        assert manager._fallback_logger is not None
    finally:
        logger.removeHandler(handler)

