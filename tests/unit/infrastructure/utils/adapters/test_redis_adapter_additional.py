#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedisAdapter 额外单测，覆盖连接、查询、写入、批量、健康检查与事务等关键分支。
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.infrastructure.utils.adapters import redis_adapter
from src.infrastructure.utils.adapters.redis_adapter import (
    RedisAdapter,
    RedisConstants,
    RedisTransaction,
)


class DummyErrorHandler:
    """记录错误调用便于断言。"""

    def __init__(self) -> None:
        self.calls: List[Tuple[Exception, str]] = []

    def handle(self, exc: Exception, message: str) -> None:
        self.calls.append((exc, message))


class FakePipeline:
    """模拟 redis 管道行为。"""

    def __init__(self, client: "FakeRedisClient") -> None:
        self._client = client
        self.operations: List[Tuple[str, Tuple[Any, ...]]] = []
        self.executed = False
        self.discarded = False

    def set(self, key: str, value: str) -> "FakePipeline":
        self.operations.append(("set", (key, value)))
        return self

    def expire(self, key: str, seconds: int) -> "FakePipeline":
        self.operations.append(("expire", (key, seconds)))
        return self

    def delete(self, key: str) -> "FakePipeline":
        self.operations.append(("delete", (key,)))
        return self

    def execute(self) -> None:
        if self._client.pipeline_execute_error:
            raise self._client.pipeline_execute_error
        self.executed = True
        for op, payload in self.operations:
            if op == "set":
                key, value = payload
                self._client.store[key] = value
            elif op == "expire":
                key, seconds = payload
                self._client.expire_calls.append((key, seconds))
            elif op == "delete":
                (key,) = payload
                self._client.store.pop(key, None)

    def discard(self) -> None:
        self.discarded = True


class FakeRedisClient:
    """轻量 redis 客户端桩，支持常用方法及可注入错误。"""

    def __init__(self) -> None:
        self.store: Dict[str, Any] = {}
        self.closed = False
        self.expire_calls: List[Tuple[str, int]] = []
        self.raise_on_get: Optional[Exception] = None
        self.raise_on_set: Optional[Exception] = None
        self.raise_on_ping: Optional[Exception] = None
        self.last_pipeline: Optional[FakePipeline] = None
        self.pipeline_execute_error: Optional[Exception] = None

    def ping(self) -> bool:
        if self.raise_on_ping:
            raise self.raise_on_ping
        return True

    def close(self) -> None:
        self.closed = True

    def get(self, key: str) -> Any:
        if self.raise_on_get:
            raise self.raise_on_get
        return self.store.get(key)

    def set(self, key: str, value: str) -> bool:
        if self.raise_on_set:
            raise self.raise_on_set
        self.store[key] = value
        return True

    def expire(self, key: str, seconds: int) -> None:
        self.expire_calls.append((key, seconds))

    def delete(self, key: str) -> int:
        return 1 if self.store.pop(key, None) is not None else 0

    def exists(self, key: str) -> int:
        return int(key in self.store)

    def keys(self, pattern: str) -> List[str]:
        if pattern == "*":
            return list(self.store.keys())
        return [key for key in self.store if pattern in key]

    def pipeline(self) -> FakePipeline:
        self.last_pipeline = FakePipeline(self)
        return self.last_pipeline

    def info(self) -> Dict[str, Any]:
        return {
            "redis_version": "7.2",
            "used_memory": 1024,
            "connected_clients": 3,
        }


@pytest.fixture
def error_handler() -> DummyErrorHandler:
    return DummyErrorHandler()


@pytest.fixture
def adapter_with_client(error_handler: DummyErrorHandler) -> Tuple[RedisAdapter, FakeRedisClient]:
    adapter = RedisAdapter(error_handler=error_handler)
    client = FakeRedisClient()
    adapter._client = client
    adapter.client = client  # 兼容外部属性
    adapter._connected = True
    return adapter, client


def test_connect_and_disconnect_success(monkeypatch: pytest.MonkeyPatch, error_handler: DummyErrorHandler) -> None:
    captured_kwargs: Dict[str, Any] = {}

    class StubRedis(FakeRedisClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(redis_adapter, "redis", SimpleNamespace(Redis=StubRedis))

    adapter = RedisAdapter(error_handler=error_handler)
    config = {"host": "example.host", "port": 6380, "password": "secret", "decode_responses": False}
    assert adapter.connect(config) is True
    assert adapter.is_connected() is True
    assert captured_kwargs["host"] == "example.host"
    assert captured_kwargs["password"] == "secret"

    assert adapter.disconnect() is True
    assert adapter.is_connected() is False


def test_connect_failure_propagates(monkeypatch: pytest.MonkeyPatch, error_handler: DummyErrorHandler) -> None:
    class ExplodingRedis:
        def __init__(self, **_: Any) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(redis_adapter, "redis", SimpleNamespace(Redis=ExplodingRedis))

    adapter = RedisAdapter(error_handler=error_handler)
    with pytest.raises(RuntimeError):
        adapter.connect({})
    assert error_handler.calls
    exc, message = error_handler.calls[0]
    assert isinstance(exc, RuntimeError)
    assert "Redis连接失败" in message


def test_execute_query_supported_types(adapter_with_client: Tuple[RedisAdapter, FakeRedisClient]) -> None:
    adapter, client = adapter_with_client
    client.store["foo"] = "bar"

    result_get = adapter.execute_query("foo")
    assert result_get.success is True
    assert result_get.data == [{"key": "foo", "value": "bar"}]

    result_exists = adapter.execute_query("foo", {"type": "exists", "key": "foo"})
    assert result_exists.success is True
    assert result_exists.data == [{"key": "foo", "exists": True}]

    client.store["prefix:a"] = "value"
    result_keys = adapter.execute_query("*", {"type": "keys", "pattern": "prefix"})
    assert result_keys.row_count == 1

    unsupported = adapter.execute_query("foo", {"type": "hash_get", "key": "foo"})
    assert unsupported.success is False
    assert "不支持的查询类型" in unsupported.error_message


def test_execute_query_handles_errors(
    adapter_with_client: Tuple[RedisAdapter, FakeRedisClient], error_handler: DummyErrorHandler
) -> None:
    adapter, client = adapter_with_client
    client.raise_on_get = RuntimeError("query failed")

    result = adapter.execute_query("foo")
    assert result.success is False
    assert result.error_message == "query failed"
    assert error_handler.calls


def test_execute_write_paths(adapter_with_client: Tuple[RedisAdapter, FakeRedisClient]) -> None:
    adapter, client = adapter_with_client

    write_set = adapter.execute_write({"type": "set", "key": "k1", "value": {"a": 1}, "expiry": 10})
    assert write_set.success is True
    assert json.loads(client.store["k1"]) == {"a": 1}
    assert client.expire_calls[-1] == ("k1", 10)

    write_delete = adapter.execute_write({"type": "delete", "key": "k1"})
    assert write_delete.affected_rows == RedisConstants.SUCCESS_AFFECTED_ROWS

    missing_key = adapter.execute_write({"type": "set"})
    assert missing_key.affected_rows == 0

    unsupported = adapter.execute_write({"type": "append", "key": "k2"})
    assert unsupported.affected_rows == 0


def test_execute_write_errors(
    adapter_with_client: Tuple[RedisAdapter, FakeRedisClient], error_handler: DummyErrorHandler
) -> None:
    adapter, client = adapter_with_client
    client.raise_on_set = RuntimeError("write boom")

    result = adapter.execute_write({"type": "set", "key": "boom", "value": 1})
    assert result.success is False
    assert result.error_message == "write boom"
    assert error_handler.calls


def test_batch_write_success(adapter_with_client: Tuple[RedisAdapter, FakeRedisClient]) -> None:
    adapter, client = adapter_with_client

    data_list = [
        {"type": "set", "key": "k1", "value": 1, "expiry": 5},
        {"type": "delete", "key": "missing"},
        {"type": "set", "key": None, "value": 2},
    ]
    result = adapter.batch_write(data_list)
    assert result.success is True
    assert result.affected_rows == 2  # 两次有效操作
    assert client.store["k1"] == json.dumps(1)
    assert client.last_pipeline is not None and client.last_pipeline.executed is True


def test_batch_write_error(adapter_with_client: Tuple[RedisAdapter, FakeRedisClient], error_handler: DummyErrorHandler) -> None:
    adapter, client = adapter_with_client
    client.pipeline_execute_error = RuntimeError("batch boom")

    result = adapter.batch_write([{"type": "set", "key": "boom", "value": 1}])
    assert result.success is False
    assert "batch boom" in result.error_message
    assert error_handler.calls


def test_health_check_paths(
    adapter_with_client: Tuple[RedisAdapter, FakeRedisClient], error_handler: DummyErrorHandler
) -> None:
    adapter, client = adapter_with_client

    healthy = adapter.health_check()
    assert healthy.is_healthy is True
    assert healthy.details["redis_version"] == "7.2"

    client.raise_on_ping = RuntimeError("ping fail")
    unhealthy = adapter.health_check()
    assert unhealthy.is_healthy is False
    assert "ping fail" in unhealthy.message
    assert error_handler.calls


def test_connection_status_behaviour(adapter_with_client: Tuple[RedisAdapter, FakeRedisClient]) -> None:
    adapter, client = adapter_with_client
    status = adapter.connection_status()
    assert status["connected"] is True

    client.raise_on_ping = RuntimeError("lost")
    status_after_failure = adapter.connection_status()
    assert status_after_failure["connected"] is False
    assert adapter.is_connected() is False


def test_begin_transaction_and_context(adapter_with_client: Tuple[RedisAdapter, FakeRedisClient]) -> None:
    adapter, client = adapter_with_client

    transaction = adapter.begin_transaction()
    with transaction as pipe:
        pipe.set("k1", "v1")
    assert client.last_pipeline is not None
    assert client.last_pipeline.executed is True

    transaction = adapter.begin_transaction()
    with pytest.raises(RuntimeError):
        with transaction as pipe:
            pipe.set("k2", "v2")
            raise RuntimeError("abort")
    assert client.last_pipeline.discarded is True


def test_generate_connection_string_and_prefixed_key(adapter_with_client: Tuple[RedisAdapter, FakeRedisClient]) -> None:
    adapter, _ = adapter_with_client
    conn_str = adapter._generate_connection_string({"host": "h", "port": 1, "db": 2, "password": "p"})
    assert conn_str == "host=h port=1 db=2 password=p"
    assert adapter._get_prefixed_key("infra:test") == "infra:test"
    assert adapter._get_prefixed_key("raw") == f"{RedisConstants.KEY_PREFIX}raw"


def test_set_get_delete_exists_keys(adapter_with_client: Tuple[RedisAdapter, FakeRedisClient]) -> None:
    adapter, client = adapter_with_client
    data = {"now": datetime(2024, 1, 1)}
    assert adapter.set("custom", data, expiry=30) is True
    stored = adapter.get("custom")
    assert stored == {"now": "2024-01-01T00:00:00"}
    assert adapter.exists("custom") is True
    assert adapter.keys("cust") == ["custom"]
    assert adapter.delete("custom") is True
    assert adapter.get("custom") is None


def test_set_get_delete_error_paths(
    adapter_with_client: Tuple[RedisAdapter, FakeRedisClient], error_handler: DummyErrorHandler
) -> None:
    adapter, client = adapter_with_client
    client.raise_on_set = RuntimeError("set error")
    with pytest.raises(RuntimeError):
        adapter.set("boom", 1)
    assert error_handler.calls

    client.raise_on_set = None
    adapter.set("exists", 1)
    client.raise_on_get = RuntimeError("get error")
    with pytest.raises(RuntimeError):
        adapter.get("exists")

    client.raise_on_get = None
    client.raise_on_set = None
    client.store["exists"] = json.dumps({"v": 1})
    client.raise_on_ping = None
    client.raise_on_set = RuntimeError("delete error")
    client.delete = lambda key: (_ for _ in ()).throw(RuntimeError("delete error"))  # type: ignore
    with pytest.raises(RuntimeError):
        adapter.delete("exists")


def test_begin_transaction_without_connection(error_handler: DummyErrorHandler) -> None:
    adapter = RedisAdapter(error_handler=error_handler)
    with pytest.raises(RuntimeError):
        adapter.begin_transaction()


