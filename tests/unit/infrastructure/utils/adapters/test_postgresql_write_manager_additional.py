#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充 PostgreSQLWriteManager 事务失败与参数异常路径覆盖。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.infrastructure.utils.adapters import postgresql_write_manager as pwm


class _DummyCursor:
    def __init__(self, fail_on_execute: bool = False) -> None:
        self.fail_on_execute = fail_on_execute
        self.rowcount = 1
        self.executed: List[Dict[str, Any]] = []

    def __enter__(self) -> "_DummyCursor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.executed.append({"query": query, "params": params})
        if self.fail_on_execute:
            raise RuntimeError("cursor execute failed")


class _DummyClient:
    def __init__(self, cursor: _DummyCursor, fail_commit: bool = False, fail_rollback: bool = False) -> None:
        self._cursor = cursor
        self.fail_commit = fail_commit
        self.fail_rollback = fail_rollback
        self.committed = False
        self.rolled_back = False

    def cursor(self) -> _DummyCursor:
        return self._cursor

    def commit(self) -> None:
        if self.fail_commit:
            raise RuntimeError("commit failed")
        self.committed = True

    def rollback(self) -> None:
        if self.fail_rollback:
            raise RuntimeError("rollback failed")
        self.rolled_back = True


def test_execute_write_without_client() -> None:
    manager = pwm.PostgreSQLWriteManager()
    result = manager.execute_write({"operation": "insert"})
    assert result.success is False
    assert result.error_message == "数据库未连接"


def test_execute_write_unsupported_operation() -> None:
    manager = pwm.PostgreSQLWriteManager(client=_DummyClient(_DummyCursor()))
    result = manager.execute_write({"operation": "unknown"})
    assert result.success is False
    assert "不支持的操作类型" in result.error_message


def test_batch_write_success() -> None:
    cursor = _DummyCursor()
    client = _DummyClient(cursor)
    manager = pwm.PostgreSQLWriteManager(client=client)

    data_list = [
        {"operation": "insert", "table": "users", "values": {"id": 1, "name": "A"}},
        {"operation": "update", "table": "users", "values": {"name": "B"}, "conditions": {"id": 1}},
    ]
    result = manager.batch_write(data_list)

    assert result.success is True
    assert result.affected_rows == 2
    assert client.committed is True


def test_batch_write_rollback_on_error() -> None:
    cursor = _DummyCursor(fail_on_execute=True)
    client = _DummyClient(cursor)
    manager = pwm.PostgreSQLWriteManager(client=client)

    result = manager.batch_write([{ "operation": "insert", "table": "t", "values": {"id": 1}}])

    assert result.success is False
    assert client.rolled_back is True
    assert client.committed is False


def test_batch_write_outer_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def bogus_cursor():
        raise RuntimeError("no cursor")

    client = _DummyClient(_DummyCursor())
    client.cursor = bogus_cursor  # type: ignore[assignment]
    manager = pwm.PostgreSQLWriteManager(client=client)

    result = manager.batch_write([{ "operation": "delete", "table": "t", "conditions": {"id": 1}}])

    assert result.success is False
    assert "no cursor" in result.error_message


def test_execute_insert_failure() -> None:
    cursor = _DummyCursor(fail_on_execute=True)
    client = _DummyClient(cursor, fail_rollback=False)
    manager = pwm.PostgreSQLWriteManager(client=client)

    result = manager.execute_write({"operation": "insert", "table": "t", "values": {"id": 1}})
    assert result.success is False
    assert client.rolled_back is True


def test_execute_update_and_delete() -> None:
    cursor = _DummyCursor()
    client = _DummyClient(cursor)
    manager = pwm.PostgreSQLWriteManager(client=client)

    update_result = manager.execute_write(
        {"operation": "update", "table": "t", "values": {"name": "B"}, "conditions": {"id": 1}}
    )
    delete_result = manager.execute_write(
        {"operation": "delete", "table": "t", "conditions": {"id": 1}}
    )
    assert update_result.success is True
    assert delete_result.success is True
    assert client.committed is True


def test_execute_update_failure() -> None:
    cursor = _DummyCursor(fail_on_execute=True)
    client = _DummyClient(cursor)
    manager = pwm.PostgreSQLWriteManager(client=client)

    result = manager.execute_write(
        {"operation": "update", "table": "t", "values": {"name": "B"}, "conditions": {"id": 1}}
    )
    assert result.success is False
    assert client.rolled_back is True


def test_execute_delete_failure() -> None:
    cursor = _DummyCursor(fail_on_execute=True)
    client = _DummyClient(cursor)
    manager = pwm.PostgreSQLWriteManager(client=client)

    result = manager.execute_write({"operation": "delete", "table": "t", "conditions": {"id": 1}})
    assert result.success is False
    assert client.rolled_back is True

