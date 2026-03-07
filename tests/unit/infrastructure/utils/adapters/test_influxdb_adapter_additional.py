#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充 InfluxDBAdapter 在异常和断开流程中的测试覆盖。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.infrastructure.utils.adapters import influxdb_adapter as adapter
from src.infrastructure.utils.interfaces.database_interfaces import QueryResult, WriteResult


class _StubWriteAPI:
    def __init__(self, fail_close: bool = False, fail_write: bool = False) -> None:
        self.fail_close = fail_close
        self.fail_write = fail_write
        self.records: List[Any] = []

    def write(self, bucket: str, record: Any) -> None:
        self.records.append((bucket, record))
        if self.fail_write:
            raise RuntimeError("write failed")

    def close(self) -> None:
        if self.fail_close:
            raise RuntimeError("close failed")


class _StubQueryAPI:
    def __init__(self, result: Any = None, fail: bool = False) -> None:
        self.result = result or []
        self.fail = fail
        self.params: List[Dict[str, Any]] = []

    def query(self, query: str, params: Dict[str, Any]) -> Any:
        self.params.append({"query": query, "params": params})
        if self.fail:
            raise RuntimeError("query failed")
        return self.result


class _StubClient:
    def __init__(self, fail_ping: bool = False) -> None:
        self.fail_ping = fail_ping
        self.closed = False

    def write_api(self, write_options: Any = None) -> _StubWriteAPI:
        return self._write_api

    def query_api(self) -> _StubQueryAPI:
        return self._query_api

    def ping(self) -> bool:
        if self.fail_ping:
            raise RuntimeError("ping failed")
        return True

    def close(self) -> None:
        self.closed = True


class _StubErrorHandler:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def handle(self, exc: Exception, message: str) -> None:
        self.calls.append({"exception": exc, "message": message})


@pytest.fixture
def influx(monkeypatch: pytest.MonkeyPatch) -> adapter.InfluxDBAdapter:
    stub_client = _StubClient()
    stub_client._write_api = _StubWriteAPI()
    stub_client._query_api = _StubQueryAPI(result=[])

    def fake_client(*args: Any, **kwargs: Any) -> _StubClient:
        return stub_client

    monkeypatch.setattr(adapter, "InfluxDBClient", fake_client)
    monkeypatch.setattr(adapter, "WriteOptions", lambda *args, **kwargs: object())

    error_handler = _StubErrorHandler()
    influx_adapter = adapter.InfluxDBAdapter(error_handler=error_handler)
    influx_adapter.connect({"url": "http://localhost", "token": "t", "org": "org", "bucket": "b"})
    return influx_adapter


def test_disconnect_handles_write_close_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_client = _StubClient()
    stub_client._write_api = _StubWriteAPI(fail_close=True)
    stub_client._query_api = _StubQueryAPI()

    monkeypatch.setattr(adapter, "InfluxDBClient", lambda *args, **kwargs: stub_client)
    monkeypatch.setattr(adapter, "WriteOptions", lambda *args, **kwargs: object())

    error_handler = _StubErrorHandler()
    influx_adapter = adapter.InfluxDBAdapter(error_handler=error_handler)
    influx_adapter.connect({"url": "x", "token": "y", "org": "z"})

    success = influx_adapter.disconnect()
    assert success is True
    assert influx_adapter.is_connected() is False
    assert stub_client.closed is True
    assert any("关闭失败" in call["message"] for call in error_handler.calls)


def test_execute_query_when_disconnected() -> None:
    influx_adapter = adapter.InfluxDBAdapter(error_handler=_StubErrorHandler())
    result = influx_adapter.execute_query("SELECT 1")
    assert isinstance(result, QueryResult)
    assert result.success is False
    assert result.error_message == "数据库未连接"


def test_execute_query_handles_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_client = _StubClient()
    stub_client._write_api = _StubWriteAPI()
    stub_client._query_api = _StubQueryAPI(fail=True)

    monkeypatch.setattr(adapter, "InfluxDBClient", lambda *args, **kwargs: stub_client)
    monkeypatch.setattr(adapter, "WriteOptions", lambda *args, **kwargs: object())

    error_handler = _StubErrorHandler()
    influx_adapter = adapter.InfluxDBAdapter(error_handler=error_handler)
    influx_adapter.connect({"url": "x", "token": "y", "org": "z"})

    result = influx_adapter.execute_query("SELECT * FROM cpu")
    assert result.success is False
    assert "query failed" in result.error_message
    assert any("查询失败" in call["message"] for call in error_handler.calls)


def test_execute_write_batch_write_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_client = _StubClient()
    stub_client._write_api = _StubWriteAPI(fail_write=True)
    stub_client._query_api = _StubQueryAPI()

    monkeypatch.setattr(adapter, "InfluxDBClient", lambda *args, **kwargs: stub_client)
    monkeypatch.setattr(adapter, "WriteOptions", lambda *args, **kwargs: object())

    error_handler = _StubErrorHandler()
    influx_adapter = adapter.InfluxDBAdapter(error_handler=error_handler)
    influx_adapter.connect({"url": "x", "token": "y", "org": "z"})

    write_result = influx_adapter.execute_write({"measurement": "cpu", "fields": {"usage": 0.5}})
    assert isinstance(write_result, WriteResult)
    assert write_result.success is False
    assert any("写入失败" in call["message"] for call in error_handler.calls)


def test_batch_write_raises_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_client = _StubClient()
    stub_client._write_api = _StubWriteAPI(fail_write=True)
    stub_client._query_api = _StubQueryAPI()

    monkeypatch.setattr(adapter, "InfluxDBClient", lambda *args, **kwargs: stub_client)
    monkeypatch.setattr(adapter, "WriteOptions", lambda *args, **kwargs: object())

    error_handler = _StubErrorHandler()
    influx_adapter = adapter.InfluxDBAdapter(error_handler=error_handler)
    influx_adapter.connect({"url": "x", "token": "y", "org": "z"})

    with pytest.raises(RuntimeError, match="write failed"):
        influx_adapter.batch_write([{"measurement": "cpu", "fields": {"usage": 0.7}}])
    assert any("批量写入失败" in call["message"] for call in error_handler.calls)

