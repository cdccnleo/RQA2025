#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Resource Alert数据类深度测试"""

import pytest
from datetime import datetime
from dataclasses import is_dataclass, asdict


# ============================================================================
# AlertRule测试
# ============================================================================

def test_alert_rule_creation():
    """测试AlertRule创建"""
    from src.infrastructure.resource.models.alert_dataclasses import AlertRule
    from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
    
    rule = AlertRule(
        name="test_rule",
        alert_type=AlertType.TEST_TIMEOUT,
        alert_level=AlertLevel.WARNING,
        condition="time > 5",
        threshold=5.0
    )
    
    assert rule.name == "test_rule"
    assert rule.alert_type == AlertType.TEST_TIMEOUT
    assert rule.alert_level == AlertLevel.WARNING
    assert rule.condition == "time > 5"
    assert rule.threshold == 5.0
    assert rule.enabled is True  # 默认值
    assert rule.cooldown == 300  # 默认值
    assert rule.last_triggered is None  # 默认值


def test_alert_rule_with_all_fields():
    """测试AlertRule使用所有字段"""
    from src.infrastructure.resource.models.alert_dataclasses import AlertRule
    from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
    
    trigger_time = datetime(2025, 1, 1, 12, 0, 0)
    rule = AlertRule(
        name="full_rule",
        alert_type=AlertType.SYSTEM_ERROR,
        alert_level=AlertLevel.CRITICAL,
        condition="error_count > 10",
        threshold=10.0,
        enabled=False,
        cooldown=600,
        last_triggered=trigger_time
    )
    
    assert rule.name == "full_rule"
    assert rule.enabled is False
    assert rule.cooldown == 600
    assert rule.last_triggered == trigger_time


def test_alert_rule_is_dataclass():
    """测试AlertRule是dataclass"""
    from src.infrastructure.resource.models.alert_dataclasses import AlertRule
    
    assert is_dataclass(AlertRule)


def test_alert_rule_to_dict():
    """测试AlertRule转为字典"""
    from src.infrastructure.resource.models.alert_dataclasses import AlertRule
    from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
    
    rule = AlertRule(
        name="test",
        alert_type=AlertType.TEST_TIMEOUT,
        alert_level=AlertLevel.WARNING,
        condition="x > 1",
        threshold=1.0
    )
    
    rule_dict = asdict(rule)
    assert isinstance(rule_dict, dict)
    assert rule_dict['name'] == "test"
    assert rule_dict['threshold'] == 1.0


# ============================================================================
# Alert测试
# ============================================================================

def test_alert_creation():
    """测试Alert创建"""
    from src.infrastructure.resource.models.alert_dataclasses import Alert
    from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
    
    timestamp = datetime.now()
    alert = Alert(
        id="alert_001",
        alert_type=AlertType.TEST_FAILURE,
        alert_level=AlertLevel.ERROR,
        message="Test failed",
        details={"test_name": "test_example"},
        timestamp=timestamp,
        source="test_runner"
    )
    
    assert alert.id == "alert_001"
    assert alert.alert_type == AlertType.TEST_FAILURE
    assert alert.alert_level == AlertLevel.ERROR
    assert alert.message == "Test failed"
    assert alert.details["test_name"] == "test_example"
    assert alert.timestamp == timestamp
    assert alert.source == "test_runner"
    assert alert.resolved is False  # 默认值
    assert alert.resolved_at is None  # 默认值


def test_alert_with_resolution():
    """测试Alert包含解决信息"""
    from src.infrastructure.resource.models.alert_dataclasses import Alert
    from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
    
    timestamp = datetime.now()
    resolved_time = datetime.now()
    
    alert = Alert(
        id="alert_002",
        alert_type=AlertType.SYSTEM_ERROR,
        alert_level=AlertLevel.CRITICAL,
        message="System error",
        details={},
        timestamp=timestamp,
        source="monitor",
        resolved=True,
        resolved_at=resolved_time
    )
    
    assert alert.resolved is True
    assert alert.resolved_at == resolved_time


def test_alert_is_dataclass():
    """测试Alert是dataclass"""
    from src.infrastructure.resource.models.alert_dataclasses import Alert
    
    assert is_dataclass(Alert)


# ============================================================================
# AlertPerformanceMetrics测试
# ============================================================================

def test_alert_performance_metrics_defaults():
    """测试AlertPerformanceMetrics默认值"""
    from src.infrastructure.resource.models.alert_dataclasses import AlertPerformanceMetrics
    
    metrics = AlertPerformanceMetrics()
    
    assert metrics.cpu_usage == 0.0
    assert metrics.memory_usage == 0.0
    assert metrics.disk_usage == 0.0
    assert metrics.network_latency == 0.0
    assert metrics.test_execution_time == 0.0
    assert metrics.test_success_rate == 0.0
    assert metrics.active_threads == 0
    assert isinstance(metrics.timestamp, datetime)


def test_alert_performance_metrics_custom_values():
    """测试AlertPerformanceMetrics自定义值"""
    from src.infrastructure.resource.models.alert_dataclasses import AlertPerformanceMetrics
    
    custom_time = datetime(2025, 1, 1, 0, 0, 0)
    metrics = AlertPerformanceMetrics(
        cpu_usage=75.5,
        memory_usage=80.2,
        disk_usage=60.0,
        network_latency=50.5,
        test_execution_time=3.5,
        test_success_rate=95.5,
        active_threads=10,
        timestamp=custom_time
    )
    
    assert metrics.cpu_usage == 75.5
    assert metrics.memory_usage == 80.2
    assert metrics.disk_usage == 60.0
    assert metrics.network_latency == 50.5
    assert metrics.test_execution_time == 3.5
    assert metrics.test_success_rate == 95.5
    assert metrics.active_threads == 10
    assert metrics.timestamp == custom_time


def test_alert_performance_metrics_is_dataclass():
    """测试AlertPerformanceMetrics是dataclass"""
    from src.infrastructure.resource.models.alert_dataclasses import AlertPerformanceMetrics
    
    assert is_dataclass(AlertPerformanceMetrics)


# ============================================================================
# TestExecutionInfo测试
# ============================================================================

def test_test_execution_info_minimal():
    """测试TestExecutionInfo最小创建"""
    from src.infrastructure.resource.models.alert_dataclasses import TestExecutionInfo
    
    start = datetime.now()
    info = TestExecutionInfo(
        test_id="test_001",
        test_name="test_example",
        start_time=start
    )
    
    assert info.test_id == "test_001"
    assert info.test_name == "test_example"
    assert info.start_time == start
    assert info.end_time is None  # 默认值
    assert info.status == "running"  # 默认值
    assert info.execution_time is None  # 默认值
    assert info.error_message is None  # 默认值
    assert info.performance_metrics is None  # 默认值


def test_test_execution_info_complete():
    """测试TestExecutionInfo完整信息"""
    from src.infrastructure.resource.models.alert_dataclasses import (
        TestExecutionInfo,
        AlertPerformanceMetrics
    )
    
    start = datetime(2025, 1, 1, 10, 0, 0)
    end = datetime(2025, 1, 1, 10, 0, 5)
    metrics = AlertPerformanceMetrics(cpu_usage=50.0)
    
    info = TestExecutionInfo(
        test_id="test_002",
        test_name="test_complete",
        start_time=start,
        end_time=end,
        status="passed",
        execution_time=5.0,
        error_message=None,
        performance_metrics=metrics
    )
    
    assert info.test_id == "test_002"
    assert info.end_time == end
    assert info.status == "passed"
    assert info.execution_time == 5.0
    assert info.performance_metrics == metrics


def test_test_execution_info_with_error():
    """测试TestExecutionInfo包含错误"""
    from src.infrastructure.resource.models.alert_dataclasses import TestExecutionInfo
    
    start = datetime.now()
    info = TestExecutionInfo(
        test_id="test_003",
        test_name="test_failed",
        start_time=start,
        status="failed",
        error_message="Assertion failed"
    )
    
    assert info.status == "failed"
    assert info.error_message == "Assertion failed"


def test_test_execution_info_is_dataclass():
    """测试TestExecutionInfo是dataclass"""
    from src.infrastructure.resource.models.alert_dataclasses import TestExecutionInfo
    
    assert is_dataclass(TestExecutionInfo)


# ============================================================================
# 向后兼容性测试
# ============================================================================

def test_performance_metrics_alias():
    """测试PerformanceMetrics别名"""
    from src.infrastructure.resource.models.alert_dataclasses import (
        PerformanceMetrics,
        AlertPerformanceMetrics
    )
    
    # PerformanceMetrics应该是AlertPerformanceMetrics的别名
    assert PerformanceMetrics is AlertPerformanceMetrics


def test_performance_metrics_alias_usage():
    """测试使用PerformanceMetrics别名"""
    from src.infrastructure.resource.models.alert_dataclasses import PerformanceMetrics
    
    metrics = PerformanceMetrics(cpu_usage=60.0)
    assert metrics.cpu_usage == 60.0


# ============================================================================
# 综合测试
# ============================================================================

def test_alert_with_performance_metrics():
    """测试Alert包含PerformanceMetrics"""
    from src.infrastructure.resource.models.alert_dataclasses import (
        Alert,
        AlertPerformanceMetrics
    )
    from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
    
    metrics = AlertPerformanceMetrics(cpu_usage=90.0, memory_usage=85.0)
    timestamp = datetime.now()
    
    alert = Alert(
        id="perf_alert",
        alert_type=AlertType.PERFORMANCE_DEGRADATION,
        alert_level=AlertLevel.WARNING,
        message="High resource usage",
        details={"metrics": asdict(metrics)},
        timestamp=timestamp,
        source="performance_monitor"
    )
    
    assert alert.details["metrics"]["cpu_usage"] == 90.0
    assert alert.details["metrics"]["memory_usage"] == 85.0


def test_test_execution_with_all_components():
    """测试TestExecutionInfo包含所有组件"""
    from src.infrastructure.resource.models.alert_dataclasses import (
        TestExecutionInfo,
        AlertPerformanceMetrics
    )
    
    metrics = AlertPerformanceMetrics(
        test_execution_time=10.5,
        test_success_rate=100.0
    )
    
    start = datetime(2025, 1, 1, 9, 0, 0)
    end = datetime(2025, 1, 1, 9, 0, 10)
    
    info = TestExecutionInfo(
        test_id="comprehensive_test",
        test_name="full_test",
        start_time=start,
        end_time=end,
        status="completed",
        execution_time=10.5,
        performance_metrics=metrics
    )
    
    assert info.execution_time == metrics.test_execution_time
    assert info.performance_metrics.test_success_rate == 100.0

