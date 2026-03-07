#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PostgreSQLAdapter 降级、异常与健康检查路径单测。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import pytest

from src.infrastructure.utils.adapters import postgresql_adapter as pg


class _StubErrorHandler:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def handle(self, error: Exception, context: str = "") -> None:
        self.calls.append({"error": error, "context": context})


class _StubConnectionManager:
    def __init__(self, succeed: bool = True) -> None:
        self.succeed = succeed
        self.client: Any = object()
        self.connected: bool = succeed
        self.connection_info = {"host": "stub", "database": "stub_db", "port": 5432}
        self.calls: List[Dict[str, Any]] = []

    def connect(self, config: Dict[str, Any]) -> bool:
        self.calls.append(config)
        return self.succeed


class _StubExecutor:
    def __init__(self) -> None:
        self.client = None

    def set_client(self, client: Any) -> None:
        self.client = client


class _StubWriteManager(_StubExecutor):
    pass


class _FakeCursor:
    def __init__(self, rows: Iterable[Dict[str, Any]] | None = None, error: Optional[BaseException] = None) -> None:
        self.rows = list(rows or [])
        self.error = error
        self.executed_sql: List[str] = []
        self.params: List[Any] = []
        self.rowcount = len(self.rows)
        self.closed = False

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.closed = True
        return False

    def execute(self, sql: str, params: Optional[Any] = None) -> None:
        if self.error:
            raise self.error
        self.executed_sql.append(sql)
        self.params.append(params)

    def fetchall(self) -> List[Dict[str, Any]]:
        return self.rows

    def fetchone(self) -> Optional[Dict[str, Any]]:
        return self.rows[0] if self.rows else None

    def close(self) -> None:
        self.closed = True


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor, autocommit: bool = False, cursor_side_effect: Optional[BaseException] = None) -> None:
        self._cursor = cursor
        self.autocommit = autocommit
        self.closed = False
        self._cursor_side_effect = cursor_side_effect
        self.commits: int = 0
        self.rollbacks: int = 0

    def cursor(self, *args: Any, **kwargs: Any) -> _FakeCursor:
        if self._cursor_side_effect:
            raise self._cursor_side_effect
        return self._cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


def test_connect_uses_connection_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = pg.PostgreSQLAdapter()
    manager = _StubConnectionManager(succeed=True)
    executor = _StubExecutor()
    writer = _StubWriteManager()

    adapter._connection_manager = manager
    adapter._query_executor = executor
    adapter._write_manager = writer
    adapter.COMPONENTS_AVAILABLE = True

    assert adapter.connect({"database": "db1"}) is True
    assert manager.calls[0]["database"] == "db1"
    assert adapter._client is manager.client
    assert executor.client is manager.client
    assert writer.client is manager.client


def test_connect_fallback_raises_and_handles(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = pg.PostgreSQLAdapter()
    handler = _StubErrorHandler()
    adapter._connection_manager = None
    adapter.COMPONENTS_AVAILABLE = False
    adapter._error_handler = handler

    def _boom(**kwargs: Any) -> None:
        raise RuntimeError("connect failed")

    monkeypatch.setattr(pg.psycopg2, "connect", _boom)

    with pytest.raises(RuntimeError, match="connect failed"):
        adapter.connect({"host": "h"})

    assert handler.calls[0]["context"] == "PostgreSQL连接失败"
    assert isinstance(handler.calls[0]["error"], RuntimeError)
    assert adapter._connected is False


def test_execute_query_when_not_connected_returns_empty_success() -> None:
    adapter = pg.PostgreSQLAdapter()
    result = adapter.execute_query("SELECT 1")
    assert result.success is True
    assert result.data == []
    assert getattr(result, "row_count", 0) == 0


def test_execute_query_failure_triggers_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = pg.PostgreSQLAdapter()
    handler = _StubErrorHandler()
    cursor = _FakeCursor(error=ValueError("boom"))
    connection = _FakeConnection(cursor)

    adapter._client = connection
    adapter._error_handler = handler
    adapter._connected = True

    result = adapter.execute_query("SELECT 1")
    assert result.success is False
    assert "boom" in result.error_message
    assert handler.calls[0]["context"] == "PostgreSQL查询失败"


def test_batch_write_failure_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = pg.PostgreSQLAdapter()
    handler = _StubErrorHandler()
    cursor = _FakeCursor(error=RuntimeError("bad write"))
    connection = _FakeConnection(cursor)

    adapter._client = connection
    adapter._error_handler = handler
    adapter._connected = True

    with pytest.raises(RuntimeError, match="bad write"):
        adapter.batch_write([
            {"type": "insert", "table": "t", "columns": ["id"], "values": [1]},
        ])

    assert connection.rollbacks == 1
    assert handler.calls[0]["context"] == "PostgreSQL批量写入失败"


def test_health_check_not_connected_returns_unhealthy() -> None:
    adapter = pg.PostgreSQLAdapter()
    result = adapter.health_check()
    assert result.is_healthy is False
    assert result.message == "数据库未连接"


def test_health_check_failure_handles_and_returns_details() -> None:
    adapter = pg.PostgreSQLAdapter()
    handler = _StubErrorHandler()
    cursor = _FakeCursor(error=RuntimeError("ping fail"))
    connection = _FakeConnection(cursor)

    adapter._client = connection
    adapter._error_handler = handler
    adapter._connected = True

    result = adapter.health_check()
    assert result.is_healthy is False
    assert "ping fail" in result.message
    assert handler.calls[0]["context"] == "PostgreSQL健康检查失败"


def test_connection_status_disconnected_and_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = pg.PostgreSQLAdapter()
    status = adapter.connection_status()
    assert status["connected"] is False
    assert status["status"] == pg.ConnectionStatus.DISCONNECTED.value

    cursor = _FakeCursor(rows=[{"value": 1}])
    connection = _FakeConnection(cursor)
    adapter._client = connection
    adapter._connected = True

    status2 = adapter.connection_status()
    assert status2["connected"] is True
    assert status2["status"] == pg.ConnectionStatus.CONNECTED.value


def test_get_connection_info_reflects_autocommit() -> None:
    adapter = pg.PostgreSQLAdapter()
    connection = _FakeConnection(_FakeCursor(), autocommit=True)
    adapter._client = connection
    adapter._connected = True
    adapter._connection_info = {"host": "h", "database": "d"}

    info = adapter.get_connection_info()
    assert info["connected"] is True
    assert info["autocommit"] is True
    assert info["database_type"] == "postgresql"


def test_generate_connection_string_includes_password_optional() -> None:
    adapter = pg.PostgreSQLAdapter()
    conn_str = adapter._generate_connection_string(
        {"host": "h", "port": 1234, "database": "db", "user": "u", "password": "p"}
    )
    assert "host=h" in conn_str and "password=p" in conn_str


def test_begin_transaction_without_connection_raises() -> None:
    adapter = pg.PostgreSQLAdapter()
    with pytest.raises(RuntimeError, match="数据库未连接"):
        adapter.begin_transaction()
