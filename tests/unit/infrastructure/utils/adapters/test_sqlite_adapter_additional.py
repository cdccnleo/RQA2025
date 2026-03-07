#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLiteAdapter 额外单测，覆盖连接异常、查询/写入失败、批量写入、健康检查与事务控制等关键路径。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.infrastructure.utils.adapters import sqlite_adapter
from src.infrastructure.utils.adapters.sqlite_adapter import SQLiteAdapter


class DummyErrorHandler:
    """记录错误消息，便于断言调用情况。"""

    def __init__(self) -> None:
        self.calls: List[Tuple[Exception, str]] = []

    def handle(self, exc: Exception, message: str) -> None:
        self.calls.append((exc, message))


@pytest.fixture
def error_handler() -> DummyErrorHandler:
    return DummyErrorHandler()


@pytest.fixture
def adapter(tmp_path: Path, error_handler: DummyErrorHandler) -> Tuple[SQLiteAdapter, DummyErrorHandler]:
    db_path = tmp_path / "sqlite" / "test.db"
    adapter = SQLiteAdapter(error_handler=error_handler)
    assert adapter.connect({"path": str(db_path)}) is True
    yield adapter, error_handler
    adapter.close()


def test_connect_success_and_disconnect(tmp_path: Path, error_handler: DummyErrorHandler) -> None:
    db_file = tmp_path / "db" / "data.db"
    adapter = SQLiteAdapter(error_handler=error_handler)
    assert adapter.connect({"path": str(db_file)}) is True
    assert adapter.is_connected() is True
    assert adapter.disconnect() is True
    assert adapter.is_connected() is False


def test_connect_failure_triggers_error_handler(monkeypatch: pytest.MonkeyPatch, error_handler: DummyErrorHandler) -> None:
    def explode_connect(*_: Any, **__: Any) -> sqlite3.Connection:
        raise RuntimeError("connect boom")

    monkeypatch.setattr(sqlite_adapter, "sqlite3", DummySQLiteModule(explode_connect))

    adapter = SQLiteAdapter(error_handler=error_handler)
    assert adapter.connect({"path": ":memory:"}) is False
    assert error_handler.calls
    exc, message = error_handler.calls[0]
    assert isinstance(exc, RuntimeError)
    assert "SQLite连接失败" in message


def test_execute_query_select_and_non_select(adapter: Tuple[SQLiteAdapter, DummyErrorHandler]) -> None:
    sql_adapter, _ = adapter
    sql_adapter.execute_write({"measurement": "m1", "fields": {"v": 1}, "tags": {}})
    result_select = sql_adapter.execute_query("SELECT measurement FROM time_series")
    assert result_select.success is True
    assert result_select.row_count == 1
    assert result_select.data[0]["measurement"] == "m1"

    result_non_select = sql_adapter.execute_query("DELETE FROM time_series")
    assert result_non_select.success is True
    assert result_non_select.row_count == 0


def test_execute_query_not_connected_returns_error(error_handler: DummyErrorHandler) -> None:
    adapter = SQLiteAdapter(error_handler=error_handler)
    result = adapter.execute_query("SELECT 1")
    assert result.success is False
    assert result.error_message == "数据库未连接"


def test_execute_query_failure(adapter: Tuple[SQLiteAdapter, DummyErrorHandler]) -> None:
    sql_adapter, handler = adapter
    bad_result = sql_adapter.execute_query("SELECT * FROM missing_table")
    assert bad_result.success is False
    assert "no such table" in bad_result.error_message


def test_execute_write_success_and_failure(adapter: Tuple[SQLiteAdapter, DummyErrorHandler]) -> None:
    sql_adapter, handler = adapter
    ok = sql_adapter.execute_write({"measurement": "cpu", "fields": {"usage": 0.5}, "tags": {"region": "cn"}})
    assert ok.success is True
    assert ok.affected_rows == 1

    # 通过关闭连接制造失败
    sql_adapter.connection.close()
    sql_adapter._connected = True  # 保持标记，使 SQL 执行阶段报错
    fail = sql_adapter.execute_write({"measurement": "cpu", "fields": {"usage": 0.6}})
    assert fail.success is False
    assert handler.calls


def test_execute_write_not_connected(error_handler: DummyErrorHandler) -> None:
    adapter = SQLiteAdapter(error_handler=error_handler)
    result = adapter.execute_write({"measurement": "cpu", "fields": {"usage": 0.5}})
    assert result.success is False
    assert result.error_message == "数据库未连接"


def test_batch_write_success_and_error(adapter: Tuple[SQLiteAdapter, DummyErrorHandler]) -> None:
    sql_adapter, handler = adapter
    data_list = [
        {"measurement": "cpu", "fields": {"v": 1}},
        {"measurement": "mem", "fields": {"v": 2}},
    ]
    ok = sql_adapter.batch_write(data_list)
    assert ok.success is True
    assert ok.affected_rows == 2

    sql_adapter.connection.close()
    sql_adapter._connected = True
    fail = sql_adapter.batch_write([{"measurement": "io", "fields": {"v": 3}}])
    assert fail.success is False
    assert handler.calls


def test_health_check_paths(adapter: Tuple[SQLiteAdapter, DummyErrorHandler]) -> None:
    sql_adapter, handler = adapter
    healthy = sql_adapter.health_check()
    assert healthy.is_healthy is True
    assert healthy.details["database_path"].endswith(".db")

    sql_adapter.connection.close()
    sql_adapter._connected = True
    unhealthy = sql_adapter.health_check()
    assert unhealthy.is_healthy is False
    assert handler.calls


def test_health_check_not_connected(error_handler: DummyErrorHandler) -> None:
    adapter = SQLiteAdapter(error_handler=error_handler)
    result = adapter.health_check()
    assert result.is_healthy is False
    assert result.details["error"] == "数据库未连接"


def test_connection_status_transitions(adapter: Tuple[SQLiteAdapter, DummyErrorHandler]) -> None:
    sql_adapter, _ = adapter
    assert sql_adapter.connection_status()["connected"] is True

    sql_adapter.connection.close()
    status_after = sql_adapter.connection_status()
    assert status_after["connected"] is False
    assert sql_adapter.is_connected() is False


def test_compatibility_write_and_query(adapter: Tuple[SQLiteAdapter, DummyErrorHandler]) -> None:
    sql_adapter, _ = adapter
    assert sql_adapter.write("trade", {"price": 10}, {"symbol": "AAA"}) is True
    rows = sql_adapter.query("SELECT measurement FROM time_series WHERE measurement='trade'")
    assert rows
    assert rows[0][0] == "trade"


def test_compatibility_methods_errors(adapter: Tuple[SQLiteAdapter, DummyErrorHandler]) -> None:
    sql_adapter, handler = adapter

    class RaisingConnection:
        def execute(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("write explode")

        def cursor(self) -> "RaisingConnection":
            return self

        def fetchall(self) -> List[Any]:
            return []

        def close(self) -> None:
            pass

        def commit(self) -> None:
            raise RuntimeError("write explode")

        def rollback(self) -> None:
            raise RuntimeError("write explode")

    raising_conn = RaisingConnection()
    sql_adapter._conn = raising_conn  # type: ignore[assignment]
    sql_adapter.connection = raising_conn  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        sql_adapter.write("fail", {"v": 1})
    assert handler.calls

    with pytest.raises(RuntimeError):
        sql_adapter.query("SELECT 1")


def test_transaction_commit_and_rollback(adapter: Tuple[SQLiteAdapter, DummyErrorHandler]) -> None:
    sql_adapter, handler = adapter
    assert sql_adapter.begin_transaction() is not None
    assert sql_adapter.commit() is True
    assert sql_adapter.rollback() is True

    sql_adapter.connection.close()
    sql_adapter._connected = True
    assert sql_adapter.commit() is False
    assert sql_adapter.rollback() is False
    assert handler.calls


class DummySQLiteModule:
    """用于替换 sqlite3 模块以制造连接失败。"""

    def __init__(self, connect_func):
        self._connect_func = connect_func

    def connect(self, *args, **kwargs):
        return self._connect_func(*args, **kwargs)

