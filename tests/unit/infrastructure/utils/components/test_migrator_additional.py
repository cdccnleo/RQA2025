#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
额外覆盖 migrator 模块的异常与回滚分支。
"""

from __future__ import annotations

import types
from typing import Any, Dict, List, Optional

import pytest

from src.infrastructure.utils.components import migrator


class _DummyTqdm:
    def __init__(self, total: int, desc: str) -> None:
        self.total = total
        self.desc = desc
        self.updated: List[int] = []

    def __enter__(self) -> "_DummyTqdm":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def update(self, value: int) -> None:
        self.updated.append(value)


class _StubAdapter:
    def __init__(self, responses: List[Any], fail_inserts: int = 0) -> None:
        self.responses = responses
        self.fail_inserts = fail_inserts
        self.queries: List[str] = []
        self.batch_calls: List[Dict[str, Any]] = []

    def execute_query(self, query: str) -> Any:
        self.queries.append(query)
        if not self.responses:
            return types.SimpleNamespace(success=True, data=[])
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def batch_execute(self, query: str, values: List[List[Any]]) -> None:
        self.batch_calls.append({"query": query, "values": values})
        if self.fail_inserts > 0:
            self.fail_inserts -= 1
            raise RuntimeError("insert failed")

    def batch_write(self, points: List[Any]) -> None:
        self.batch_calls.append({"points": points})


@pytest.fixture(autouse=True)
def patch_tqdm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(migrator, "tqdm", _DummyTqdm)


def test_database_migrator_basic_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    count_response = types.SimpleNamespace(success=True, data=[{"count": 2}])
    select_response = types.SimpleNamespace(
        success=True,
        data=[
            {"id": 1, "name": "A"},
            {"id": 2, "name": "B"},
        ],
    )
    source = _StubAdapter([count_response, select_response])
    target = _StubAdapter([])

    db_migrator = migrator.DatabaseMigrator(source, target)
    result = db_migrator.migrate_table("users")

    assert result["success"] is True
    assert result["migrated"] == 2
    assert target.batch_calls[0]["query"].startswith("INSERT INTO users")
    assert len(target.batch_calls[0]["values"]) == 2


def test_database_migrator_retry_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    count_response = types.SimpleNamespace(success=True, data=[{"count": 1}])
    select_response = types.SimpleNamespace(success=True, data=[{"id": 1, "value": "X"}])
    source = _StubAdapter([count_response, select_response])
    target = _StubAdapter([], fail_inserts=1)

    db_migrator = migrator.DatabaseMigrator(source, target)
    db_migrator.retry_count = 2
    result = db_migrator.migrate_table("events")

    assert result["success"] is True
    assert target.fail_inserts == 0
    assert len(target.batch_calls) == 2  # 失败后重试一次


def test_database_migrator_handles_query_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    migrator_instance = migrator.DatabaseMigrator(
        source_adapter=_StubAdapter([RuntimeError("boom")]),
        target_adapter=_StubAdapter([]),
    )
    state = {"migrated": 0, "failed": 0}
    result = migrator_instance._migrate_batch_with_retry("SELECT * FROM t", "t", state)
    assert result == {"processed": 0, "failed": 0}


def test_build_migration_result_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    migrator_instance = migrator.DatabaseMigrator(_StubAdapter([]), _StubAdapter([]))
    state = {"total_count": 5, "migrated": 3, "failed": 2, "start_time": 0}
    result = migrator_instance._build_migration_result("tbl", state)
    assert result["success"] is False
    assert result["total_processed"] == 5


def test_data_migrator_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    count_response = {"data": [{"values": {"_value": 2}}]}
    data_response = {
        "data": [
            {
                "values": {
                    "_measurement": "cpu",
                    "_time": "2024-01-01T00:00:00Z",
                    "host": "srv1",
                    "usage": 0.5,
                }
            },
            {
                "values": {
                    "_measurement": "cpu",
                    "_time": "2024-01-01T00:00:01Z",
                    "host": "srv1",
                    "usage": 0.6,
                }
            },
        ]
    }
    empty_response = {"data": []}

    source = _StubAdapter([count_response, data_response, empty_response])
    target = _StubAdapter([])

    class _PointStub:
        def __init__(self, measurement: str) -> None:
            self.measurement = measurement
            self.tags: Dict[str, str] = {}
            self.fields: Dict[str, Any] = {}

        def tag(self, key: str, value: str) -> "._PointStub":
            self.tags[key] = value
            return self

        def field(self, key: str, value: Any) -> "._PointStub":
            self.fields[key] = value
            return self

    monkeypatch.setattr(migrator, "Point", _PointStub)

    data_migrator = migrator.DataMigrator(source, target)
    result = data_migrator.migrate_measurement("cpu")

    assert result["migrated"] == 2
    assert len(target.batch_calls[0]["points"]) == 2
    first_point = target.batch_calls[0]["points"][0]
    assert first_point.tags["time"]  # _time 字段被转为 tag
    assert first_point.fields["usage"] == 0.5


def test_compare_samples_mismatch() -> None:
    migrator_instance = migrator.DatabaseMigrator(_StubAdapter([]), _StubAdapter([]))
    assert migrator_instance._compare_samples([{"id": 1}], [{"id": 2}]) is False

