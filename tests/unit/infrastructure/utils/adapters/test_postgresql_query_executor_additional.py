#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQLQueryExecutor 额外单测，覆盖查询成功/失败、简化查询与 SQL 检查。
"""

from __future__ import annotations

import types
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pytest

from src.infrastructure.utils.adapters.postgresql_query_executor import (
    PostgreSQLQueryExecutor,
)


class _DummyCursor:
    def __init__(
        self,
        rows: Iterable[Sequence[Any]] = (),
        description: Optional[List[Sequence[Any]]] = None,
        raise_on_execute: Optional[BaseException] = None,
    ) -> None:
        self._rows = list(rows)
        self.description = description
        self._raise = raise_on_execute
        self.executed_queries: List[str] = []
        self.executed_params: List[Any] = []

    def __enter__(self) -> "_DummyCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.executed_queries.append(query)
        self.executed_params.append(params)
        if self._raise:
            raise self._raise

    def fetchall(self) -> List[Sequence[Any]]:
        return self._rows


class _DummyConnection:
    def __init__(self, cursor: _DummyCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _DummyCursor:
        return self._cursor


def _extract_error(result) -> Optional[str]:
    if hasattr(result, "error_message"):
        return result.error_message
    if hasattr(result, "error"):
        return result.error
    return None


def _extract_row_count(result) -> Optional[int]:
    return getattr(result, "row_count", None)


def test_execute_query_without_client_returns_error() -> None:
    executor = PostgreSQLQueryExecutor()
    result = executor.execute_query("SELECT 1")

    assert result.success is False
    assert _extract_error(result) == "数据库未连接"
    assert _extract_row_count(result) in (None, 0)
    assert result.data == []
    assert result.execution_time == 0.0


def test_execute_query_success_with_params() -> None:
    rows = [(1, "Alice"), (2, "Bob")]
    description = (("id", None, None, None, None, None, None), ("name", None, None, None, None, None, None))
    cursor = _DummyCursor(rows=rows, description=description)
    executor = PostgreSQLQueryExecutor(client=_DummyConnection(cursor))

    params = {"limit": 10}
    result = executor.execute_query("SELECT id, name FROM users WHERE active = TRUE LIMIT %(limit)s", params)

    assert result.success is True
    assert result.data == [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    assert _extract_error(result) in (None, "")
    assert result.execution_time >= 0.0
    assert cursor.executed_queries == ["SELECT id, name FROM users WHERE active = TRUE LIMIT %(limit)s"]
    assert cursor.executed_params == [params]


def test_execute_query_without_description_returns_empty_dicts() -> None:
    rows = [(1,), (2,)]
    cursor = _DummyCursor(rows=rows, description=None)
    executor = PostgreSQLQueryExecutor(client=_DummyConnection(cursor))

    result = executor.execute_query("SELECT id FROM table")

    assert result.success is True
    assert result.data == [{}, {}]
    assert cursor.executed_params == [None]


def test_execute_query_failure_logs_and_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def _fake_error(msg):
        captured["msg"] = msg

    import logging

    logger = logging.getLogger("src.infrastructure.utils.adapters.postgresql_query_executor")
    monkeypatch.setattr(logger, "error", _fake_error)

    cursor = _DummyCursor(raise_on_execute=ValueError("boom"))
    executor = PostgreSQLQueryExecutor(client=_DummyConnection(cursor))

    result = executor.execute_query("SELECT * FROM table")

    assert result.success is False
    assert "boom" in _extract_error(result)
    assert captured["msg"].startswith("查询执行失败")
    assert result.data == []


def test_execute_query_simple_returns_data_or_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [(42,)]
    description = (("value", None, None, None, None, None, None),)
    cursor = _DummyCursor(rows=rows, description=description)
    executor = PostgreSQLQueryExecutor(client=_DummyConnection(cursor))

    data = executor.execute_query_simple({"query": "SELECT value FROM table"})
    assert data == [{"value": 42}]

    # 失败分支：execute_query 返回失败结果
    class _FailResult(types.SimpleNamespace):
        success = False
        data: List[Dict[str, Any]] = []

    monkeypatch.setattr(executor, "execute_query", lambda query, params=None: _FailResult())
    assert executor.execute_query_simple({"query": "SELECT 1"}) == []


@pytest.mark.parametrize(
    "query,expected",
    [
        ("select * from users", True),
        ("SELECT email FROM users WHERE id=%(id)s", True),
        ("DROP TABLE users", False),
        ("update users set name='x'; delete from users;", False),
        ("-- malicious comment", False),
    ],
)
def test_validate_query_detects_dangerous_patterns(query: str, expected: bool) -> None:
    executor = PostgreSQLQueryExecutor()
    assert executor.validate_query(query) is expected


def test_validate_query_logs_warning_for_dangerous_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: Dict[str, Any] = {}

    def _fake_warning(message: str) -> None:
        warnings["message"] = message

    import logging

    logger = logging.getLogger("src.infrastructure.utils.adapters.postgresql_query_executor")
    monkeypatch.setattr(logger, "warning", _fake_warning)

    executor = PostgreSQLQueryExecutor()
    assert executor.validate_query("DELETE FROM table") is False
    assert "危险SQL关键字" in warnings.get("message", "")

