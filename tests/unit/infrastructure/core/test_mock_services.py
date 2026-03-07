#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mock_services 模块测试，覆盖通用 Mock 服务行为。"""

from __future__ import annotations

from typing import Any

import pytest

from src.infrastructure.core import mock_services as ms


def test_base_mock_service_call_tracking() -> None:
    service = ms.BaseMockService()
    assert service.is_healthy() is True

    service.set_failure_mode(False)
    service.set_healthy(False)
    assert service.is_healthy() is False

    service.set_failure_mode(True, RuntimeError("fail"))
    with pytest.raises(RuntimeError, match="fail"):
        service._check_failure_mode()  # type: ignore[attr-defined]

    service.set_failure_mode(False)
    service._record_call("custom", 1, foo="bar")  # type: ignore[attr-defined]
    assert service.call_count == 1
    assert service.get_call_history()[0][1] == "custom"
    service.reset_call_history()
    assert service.call_count == 0


def test_simple_mock_dict_operations() -> None:
    mock = ms.SimpleMockDict(initial_data={"key": "value"})
    assert mock.get("key") == "value"
    assert mock.set("new", 123) is True
    assert mock.exists("new") is True
    assert mock.delete("new") is True
    assert mock.clear() is True
    assert mock.get_stats()["total_keys"] == 0

    mock.set_failure_mode(True, ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        mock.get("any")

    mock.set_failure_mode(False)
    mock.set("again", 1)
    assert mock.get_all_data()["again"] == 1


def test_simple_mock_logger_records_levels(capsys: pytest.CaptureFixture[str]) -> None:
    logger = ms.SimpleMockLogger(enabled_levels=["INFO", "ERROR"])
    logger.debug("debug")  # disabled, no output
    logger.info("info", foo="bar")
    logger.error("error", exc=RuntimeError("oops"))
    logger.critical("critical", exc=RuntimeError("boom"))
    logger.log("CUSTOM", "custom")

    out = capsys.readouterr().out
    assert "[INFO] info" in out
    assert "[ERROR] error" in out

    logs = logger.get_logs()
    assert len(logs) == 2
    # is_enabled_for 应忽略大小写
    assert logger.is_enabled_for("info") is True
    assert logger.is_enabled_for("debug") is False
    logger.clear_logs()
    assert logger.get_logs() == []


def test_simple_mock_monitor_metrics() -> None:
    monitor = ms.SimpleMockMonitor()
    monitor.record_metric("cpu", 0.5, tags={"host": "a"})
    monitor.increment_counter("req")
    monitor.record_histogram("latency", 1.2)
    assert monitor.get_metric_values("cpu") == [0.5]
    assert monitor.get_counter_value("req") == 1
    monitor.reset_metrics()
    assert monitor.get_metric_values("cpu") == []


def test_simple_mock_service_health_dict() -> None:
    service = ms.SimpleMockDict()
    result = service.check_health()
    assert result["service"] == "SimpleMockDict"
    assert result["details"]["failure_mode"] is False
