#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parameter_objects 默认分支与降级路径测试。"""

from __future__ import annotations

from datetime import datetime

from src.infrastructure.core.parameter_objects import (
    ConfigValidationParams,
    MonitoringParams,
    ResourceUsageParams,
    ServiceInitializationParams,
)


def test_service_initialization_params_defaults() -> None:
    params = ServiceInitializationParams()
    assert params.config == {}
    assert params.dependencies == []
    assert params.enable_monitoring is True
    assert params.max_retries == 3


def test_monitoring_params_timestamp_and_tags_defaults() -> None:
    before = datetime.now()
    params = MonitoringParams(metric_name="cpu", metric_value=1.0)
    assert params.tags == {}
    assert params.timestamp is not None
    assert params.timestamp >= before


def test_resource_usage_percentage_handles_zero_capacity() -> None:
    params = ResourceUsageParams(resource_type="cpu", current_usage=10, total_capacity=0)
    assert params.usage_percentage == 0.0
    assert params.is_warning_level is False
    assert params.is_critical_level is False


def test_config_validation_expected_type_mismatch() -> None:
    validator = ConfigValidationParams(value="abc", expected_type=int)
    assert validator.validate() is False





