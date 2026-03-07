#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核心参数对象测试。"""

from __future__ import annotations

from datetime import datetime, timedelta

from src.infrastructure.core import parameter_objects as po


def test_health_check_params_defaults() -> None:
    params = po.HealthCheckParams(service_name="cache")
    assert params.service_name == "cache"
    assert params.timeout == 30
    assert isinstance(params.check_timestamp, datetime)


def test_config_validation_params() -> None:
    params = po.ConfigValidationParams(value=10, expected_type=int, min_value=5, max_value=20)
    assert params.validate() is True

    params.value = 3
    assert params.validate() is False

    params = po.ConfigValidationParams(value="dev", allowed_values=["dev", "prod"])
    assert params.validate() is True

    params.value = "test"
    assert params.validate() is False

    params = po.ConfigValidationParams(value=5, custom_validator=lambda v: v % 2 == 1)
    assert params.validate() is True


def test_resource_usage_params_properties() -> None:
    params = po.ResourceUsageParams(resource_type="cpu", current_usage=40, total_capacity=100)
    assert params.usage_percentage == 40.0
    assert params.is_warning_level is False
    assert params.is_critical_level is False

    params.current_usage = 90
    assert params.is_warning_level is True
    params.current_usage = 97
    assert params.is_critical_level is True


def test_cache_operation_params_defaults() -> None:
    params = po.CacheOperationParams(operation="get")
    assert params.operation == "get"
    assert params.options == {}


def test_alert_params_defaults() -> None:
    params = po.AlertParams(alert_level="warning", alert_message="cpu high", alert_source="monitor")
    assert params.notify_channels == ["log"]
    assert params.context == {}
    assert isinstance(params.timestamp, datetime)


def test_resource_allocation_params_metadata() -> None:
    params = po.ResourceAllocationParams(resource_type="memory", resource_amount=512, requester_id="svc")
    params.metadata["region"] = "cn"
    assert params.metadata["region"] == "cn"


def test_health_check_result_params_defaults() -> None:
    params = po.HealthCheckResultParams(service_name="db", healthy=True, status="ok")
    assert params.version == "1.0.0"
    assert params.details == {}
    assert params.issues == []
    assert params.recommendations == []
    assert isinstance(params.timestamp, datetime)


def test_service_health_report_params_defaults() -> None:
    params = po.ServiceHealthReportParams()
    future = datetime.now() + timedelta(hours=1)
    params.timestamp = future
    assert params.timestamp == future
    assert params.services_filter == []


def test_log_record_params_defaults() -> None:
    params = po.LogRecordParams(level="INFO", message="test")
    assert params.extra == {}
    assert isinstance(params.timestamp, datetime)
