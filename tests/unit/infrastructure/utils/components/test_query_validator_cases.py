#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QueryValidator 额外用例，覆盖查询类型、存储类型与参数校验分支。
"""

from __future__ import annotations

from types import SimpleNamespace

from src.infrastructure.utils.components.query_validator import QueryType, QueryValidator, StorageType


def make_request(
    query_type=QueryType.REALTIME,
    storage_type=StorageType.INFLUXDB,
    params=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        query_id="req-1",
        query_type=query_type,
        storage_type=storage_type,
        params=params if params is not None else {"limit": 10},
    )


def test_validate_request_success_and_empty_params():
    validator = QueryValidator()
    request = make_request(params={})
    # 空参数允许通过
    assert validator.validate_request(request) is True


def test_validate_request_invalid_query_type():
    validator = QueryValidator()
    # 使用 SimpleNamespace 模拟 QueryRequest
    bad_request = SimpleNamespace(
        query_id="bad",
        query_type="UNKNOWN",
        storage_type=StorageType.INFLUXDB,
        params={},
    )
    assert validator.validate_request(bad_request) is False


def test_validate_request_invalid_storage_type():
    validator = QueryValidator()
    bad_request = SimpleNamespace(
        query_id="bad",
        query_type=QueryType.REALTIME,
        storage_type="memory",
        params={},
    )
    assert validator.validate_request(bad_request) is False


def test_validate_request_params_not_dict():
    validator = QueryValidator()
    bad_request = make_request(params=["not-dict"])
    assert validator.validate_request(bad_request) is False


def test_validate_requests_batch_behaviour():
    validator = QueryValidator()
    good = make_request()
    bad = SimpleNamespace(
        query_id="bad",
        query_type=QueryType.REALTIME,
        storage_type=StorageType.INFLUXDB,
        params="invalid",
    )

    # 空列表应返回 False
    assert validator.validate_requests([]) is False
    # 只要存在坏请求即失败
    assert validator.validate_requests([good, bad]) is False
    # 全部有效时返回 True
    assert validator.validate_requests([good, make_request(query_type=QueryType.HISTORICAL)]) is True


def test_validate_alias():
    validator = QueryValidator()
    assert validator.validate(make_request()) is True


def test_validate_request_none_logs_error(monkeypatch):
    validator = QueryValidator()
    captured = []

    def fake_log(level, message):
        captured.append((level, message))

    monkeypatch.setattr(
        "src.infrastructure.utils.components.query_validator._safe_log", fake_log
    )

    assert validator.validate_request(None) is False
    assert captured and "不能为空" in captured[0][1]


def test_validate_query_type_exception(monkeypatch):
    class ExplodingRequest(SimpleNamespace):
        @property
        def query_type(self):
            raise RuntimeError("boom")

    validator = QueryValidator()
    captured = []
    monkeypatch.setattr(
        "src.infrastructure.utils.components.query_validator._safe_log",
        lambda level, message: captured.append(message),
    )

    request = ExplodingRequest(storage_type=StorageType.INFLUXDB, params={})
    assert validator.validate_request(request) is False
    assert any("查询类型验证失败" in msg for msg in captured)


def test_validate_storage_type_exception(monkeypatch):
    class ExplodingRequest(SimpleNamespace):
        query_type = QueryType.REALTIME

        @property
        def storage_type(self):
            raise RuntimeError("bad storage")

    validator = QueryValidator()
    captured = []
    monkeypatch.setattr(
        "src.infrastructure.utils.components.query_validator._safe_log",
        lambda level, message: captured.append(message),
    )

    request = ExplodingRequest(params={})
    assert validator.validate_request(request) is False
    assert any("存储类型验证失败" in msg for msg in captured)

