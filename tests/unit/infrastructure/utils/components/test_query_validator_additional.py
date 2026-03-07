#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充 QueryValidator 的异常与日志分支覆盖。
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import List

import pytest

from src.infrastructure.utils.components.query_validator import (
    QueryValidator,
    QueryType,
    StorageType,
    _safe_log,
)


MODULE_LOGGER_NAME = "src.infrastructure.utils.components.query_validator"


def _make_request(**overrides):
    payload = dict(
        query_id="q-1",
        query_type=QueryType.REALTIME,
        storage_type=StorageType.INFLUXDB,
        params={"symbol": "000001"},
    )
    payload.update(overrides)
    return SimpleNamespace(**payload)  # QueryValidator 仅访问属性，使用简单命名空间即可


def test_validate_request_none_logs_error(caplog: pytest.LogCaptureFixture) -> None:
    validator = QueryValidator()
    caplog.set_level(logging.ERROR, logger=MODULE_LOGGER_NAME)

    assert validator.validate_request(None) is False  # type: ignore[arg-type]
    assert any("查询请求不能为空" in rec.message for rec in caplog.records)


def test_validate_request_invalid_query_type(caplog: pytest.LogCaptureFixture) -> None:
    validator = QueryValidator()
    caplog.set_level(logging.ERROR, logger=MODULE_LOGGER_NAME)

    request = _make_request(query_type="invalid-type")
    assert validator.validate_request(request) is False
    assert any("无效的查询类型" in rec.message for rec in caplog.records)


def test_validate_request_invalid_storage_type(caplog: pytest.LogCaptureFixture) -> None:
    validator = QueryValidator()
    caplog.set_level(logging.ERROR, logger=MODULE_LOGGER_NAME)

    request = _make_request(storage_type="unknown-storage")
    assert validator.validate_request(request) is False
    assert any("无效的存储类型" in rec.message for rec in caplog.records)


def test_validate_request_non_dict_params(caplog: pytest.LogCaptureFixture) -> None:
    validator = QueryValidator()
    caplog.set_level(logging.ERROR, logger=MODULE_LOGGER_NAME)

    request = _make_request(params=["not-dict"])  # type: ignore[arg-type]
    assert validator.validate_request(request) is False
    assert any("查询参数必须是字典类型" in rec.message for rec in caplog.records)


def test_validate_requests_empty_list_logs_error(caplog: pytest.LogCaptureFixture) -> None:
    validator = QueryValidator()
    caplog.set_level(logging.ERROR, logger=MODULE_LOGGER_NAME)

    assert validator.validate_requests([]) is False
    assert any("查询请求列表不能为空" in rec.message for rec in caplog.records)


def test_validate_requests_short_circuit_on_failure(caplog: pytest.LogCaptureFixture) -> None:
    validator = QueryValidator()
    caplog.set_level(logging.ERROR, logger=MODULE_LOGGER_NAME)

    good = _make_request()
    bad = _make_request(storage_type="bad-storage")

    assert validator.validate_requests([good, bad, good]) is False
    assert any("无效的存储类型" in rec.message for rec in caplog.records)


def test_safe_log_handles_mocked_handler_level() -> None:
    logger = logging.getLogger(MODULE_LOGGER_NAME)
    records: List[logging.LogRecord] = []
    original_level = logger.level
    logger.setLevel(logging.INFO)

    class DummyHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = DummyHandler()
    handler.level = "INFO"  # 非整数，触发 _safe_log 的容错逻辑
    logger.addHandler(handler)

    try:
        _safe_log(logging.INFO, "fallback-safe-log")
        assert records and records[0].getMessage() == "fallback-safe-log"
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)

