#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 监控枚举和基础类

测试监控相关的枚举类型和基础数据类。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitorStatus(Enum):
    """监控状态枚举"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class HealthStatus(Enum):
    """健康状态枚举"""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class MetricType(Enum):
    """指标类型枚举"""
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricData:
    """指标数据"""
    name: str
    value: Any
    type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertData:
    """告警数据"""
    alert_id: str
    title: str
    description: str
    level: AlertLevel
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "active"


class TestMonitorEnums:
    """监控枚举测试"""

    def test_alert_level_enum(self):
        """测试告警级别枚举"""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.EMERGENCY.value == "emergency"

        # 测试枚举值唯一性
        levels = [level.value for level in AlertLevel]
        assert len(levels) == len(set(levels))

    def test_monitor_status_enum(self):
        """测试监控状态枚举"""
        assert MonitorStatus.STOPPED.value == "stopped"
        assert MonitorStatus.STARTING.value == "starting"
        assert MonitorStatus.RUNNING.value == "running"
        assert MonitorStatus.STOPPING.value == "stopping"
        assert MonitorStatus.ERROR.value == "error"

    def test_health_status_enum(self):
        """测试健康状态枚举"""
        assert HealthStatus.UNKNOWN.value == "unknown"
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    def test_metric_type_enum(self):
        """测试指标类型枚举"""
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"


class TestMetricData:
    """指标数据测试"""

    def test_metric_data_creation(self):
        """测试指标数据创建"""
        metric = MetricData(
            name="cpu_usage",
            value=85.5,
            type=MetricType.GAUGE,
            metadata={"unit": "%", "description": "CPU使用率"}
        )

        assert metric.name == "cpu_usage"
        assert metric.value == 85.5
        assert metric.type == MetricType.GAUGE
        assert metric.metadata["unit"] == "%"
        assert metric.metadata["description"] == "CPU使用率"
        assert isinstance(metric.timestamp, datetime)

    def test_metric_data_default_timestamp(self):
        """测试指标数据默认时间戳"""
        before = datetime.now()
        metric = MetricData("test", 100, MetricType.COUNTER)
        after = datetime.now()

        assert before <= metric.timestamp <= after

    def test_metric_data_default_metadata(self):
        """测试指标数据默认元数据"""
        metric = MetricData("test", 100, MetricType.COUNTER)

        assert isinstance(metric.metadata, dict)
        assert len(metric.metadata) == 0

    def test_metric_data_with_custom_metadata(self):
        """测试指标数据自定义元数据"""
        metadata = {"source": "system", "interval": 60}
        metric = MetricData("response_time", 150.5, MetricType.GAUGE, metadata=metadata)

        assert metric.metadata == metadata
        assert metric.metadata["source"] == "system"
        assert metric.metadata["interval"] == 60

    def test_metric_data_string_representation(self):
        """测试指标数据的字符串表示"""
        metric = MetricData("cpu", 75.0, MetricType.GAUGE)

        str_repr = str(metric)
        assert "cpu" in str_repr
        assert "75.0" in str_repr
        assert "gauge" in str_repr

    def test_metric_data_equality(self):
        """测试指标数据相等性"""
        metric1 = MetricData("cpu", 80.0, MetricType.GAUGE)
        metric2 = MetricData("cpu", 80.0, MetricType.GAUGE)
        metric3 = MetricData("cpu", 90.0, MetricType.GAUGE)
        metric4 = MetricData("memory", 80.0, MetricType.GAUGE)

        assert metric1 == metric2
        assert metric1 != metric3
        assert metric1 != metric4

    def test_metric_data_different_types(self):
        """测试不同类型指标数据的创建"""
        gauge_metric = MetricData("temperature", 25.5, MetricType.GAUGE)
        counter_metric = MetricData("requests", 1000, MetricType.COUNTER)
        histogram_metric = MetricData("response_times", [100, 200, 300], MetricType.HISTOGRAM)

        assert gauge_metric.type == MetricType.GAUGE
        assert counter_metric.type == MetricType.COUNTER
        assert histogram_metric.type == MetricType.HISTOGRAM

        assert isinstance(gauge_metric.value, float)
        assert isinstance(counter_metric.value, int)
        assert isinstance(histogram_metric.value, list)


class TestAlertData:
    """告警数据测试"""

    def test_alert_data_creation(self):
        """测试告警数据创建"""
        alert = AlertData(
            alert_id="alert_001",
            title="高CPU使用率",
            description="CPU使用率超过90%",
            level=AlertLevel.CRITICAL,
            source="cpu_monitor",
            status="acknowledged"
        )

        assert alert.alert_id == "alert_001"
        assert alert.title == "高CPU使用率"
        assert alert.description == "CPU使用率超过90%"
        assert alert.level == AlertLevel.CRITICAL
        assert alert.source == "cpu_monitor"
        assert alert.status == "acknowledged"
        assert isinstance(alert.timestamp, datetime)

    def test_alert_data_default_values(self):
        """测试告警数据默认值"""
        alert = AlertData(
            alert_id="alert_002",
            title="测试告警",
            description="测试描述",
            level=AlertLevel.WARNING
        )

        assert alert.source == ""
        assert alert.status == "active"
        assert isinstance(alert.timestamp, datetime)

    def test_alert_data_different_levels(self):
        """测试不同告警级别的创建"""
        levels = [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL, AlertLevel.EMERGENCY]

        for level in levels:
            alert = AlertData(
                alert_id=f"alert_{level.value}",
                title=f"{level.value} alert",
                description=f"Test {level.value} alert",
                level=level
            )
            assert alert.level == level
            assert alert.status == "active"

    def test_alert_data_string_representation(self):
        """测试告警数据的字符串表示"""
        alert = AlertData(
            alert_id="alert_123",
            title="磁盘空间不足",
            description="可用磁盘空间低于10%",
            level=AlertLevel.ERROR,
            source="disk_monitor"
        )

        str_repr = str(alert)
        assert "alert_123" in str_repr
        assert "磁盘空间不足" in str_repr
        assert "ERROR" in str_repr

    def test_alert_data_equality(self):
        """测试告警数据相等性"""
        alert1 = AlertData("alert_001", "Title", "Description", AlertLevel.WARNING, "source1")
        alert2 = AlertData("alert_001", "Title", "Description", AlertLevel.WARNING, "source1")
        alert3 = AlertData("alert_002", "Title", "Description", AlertLevel.WARNING, "source1")

        assert alert1 == alert2
        assert alert1 != alert3

    def test_alert_data_with_timestamp(self):
        """测试告警数据自定义时间戳"""
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        alert = AlertData(
            alert_id="alert_001",
            title="测试",
            description="测试",
            level=AlertLevel.INFO,
            timestamp=custom_time
        )

        assert alert.timestamp == custom_time

    def test_alert_data_status_transitions(self):
        """测试告警状态转换"""
        alert = AlertData("alert_001", "Title", "Desc", AlertLevel.ERROR)

        # 初始状态
        assert alert.status == "active"

        # 模拟状态变化（这里只是测试数据类本身）
        alert.status = "acknowledged"
        assert alert.status == "acknowledged"

        alert.status = "resolved"
        assert alert.status == "resolved"


class TestMonitorDataValidation:
    """监控数据验证测试"""

    def test_metric_data_validation(self):
        """测试指标数据验证"""
        # 有效的指标数据
        valid_metric = MetricData("cpu", 85.5, MetricType.GAUGE)
        assert valid_metric.name == "cpu"
        assert valid_metric.value == 85.5

        # 测试边界值
        zero_metric = MetricData("zero", 0, MetricType.COUNTER)
        assert zero_metric.value == 0

        negative_metric = MetricData("negative", -10, MetricType.GAUGE)
        assert negative_metric.value == -10

    def test_alert_data_validation(self):
        """测试告警数据验证"""
        # 有效的告警数据
        valid_alert = AlertData(
            alert_id="valid_001",
            title="有效告警",
            description="这是一个有效的告警",
            level=AlertLevel.WARNING
        )
        assert valid_alert.alert_id == "valid_001"
        assert valid_alert.title == "有效告警"

    def test_enum_value_uniqueness(self):
        """测试枚举值唯一性"""
        # AlertLevel
        alert_values = [level.value for level in AlertLevel]
        assert len(alert_values) == len(set(alert_values))

        # MonitorStatus
        status_values = [status.value for status in MonitorStatus]
        assert len(status_values) == len(set(status_values))

        # HealthStatus
        health_values = [health.value for health in HealthStatus]
        assert len(health_values) == len(set(health_values))

        # MetricType
        metric_values = [metric.value for metric in MetricType]
        assert len(metric_values) == len(set(metric_values))

    def test_enum_iteration(self):
        """测试枚举迭代"""
        # 确保可以迭代所有枚举值
        alert_levels = list(AlertLevel)
        assert len(alert_levels) == 5

        monitor_statuses = list(MonitorStatus)
        assert len(monitor_statuses) == 5

        health_statuses = list(HealthStatus)
        assert len(health_statuses) == 4

        metric_types = list(MetricType)
        assert len(metric_types) == 4


if __name__ == "__main__":
    pytest.main([__file__])
