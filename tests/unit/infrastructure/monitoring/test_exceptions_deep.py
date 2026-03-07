#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Monitoring异常深度测试"""

import pytest


# ============================================================================
# 基础异常测试
# ============================================================================

def test_monitoring_exception_basic():
    """测试MonitoringException基础创建"""
    from src.infrastructure.monitoring.core.exceptions import MonitoringException
    
    error = MonitoringException("Test error")
    assert error.message == "Test error"
    assert error.monitor_type is None
    assert error.component_name is None
    assert error.details == {}


def test_monitoring_exception_full():
    """测试MonitoringException完整参数"""
    from src.infrastructure.monitoring.core.exceptions import MonitoringException
    
    error = MonitoringException(
        "Error occurred",
        monitor_type="system",
        component_name="cpu_monitor",
        details={"cpu_usage": 95}
    )
    
    assert error.message == "Error occurred"
    assert error.monitor_type == "system"
    assert error.component_name == "cpu_monitor"
    assert error.details["cpu_usage"] == 95


# ============================================================================
# 配置错误测试
# ============================================================================

def test_monitor_configuration_error():
    """测试MonitorConfigurationError"""
    from src.infrastructure.monitoring.core.exceptions import MonitorConfigurationError
    
    error = MonitorConfigurationError(
        "Invalid config",
        config_key="interval",
        expected_value=30,
        actual_value=0
    )
    
    assert error.config_key == "interval"
    assert error.expected_value == 30
    assert error.actual_value == 0


# ============================================================================
# 指标收集错误测试
# ============================================================================

def test_metric_collection_error():
    """测试MetricCollectionError"""
    from src.infrastructure.monitoring.core.exceptions import MetricCollectionError
    
    error = MetricCollectionError(
        "Failed to collect metric",
        metric_name="cpu_usage",
        collection_method="psutil"
    )
    
    assert error.metric_name == "cpu_usage"
    assert error.collection_method == "psutil"


# ============================================================================
# 告警处理错误测试
# ============================================================================

def test_alert_processing_error():
    """测试AlertProcessingError"""
    from src.infrastructure.monitoring.core.exceptions import AlertProcessingError
    
    error = AlertProcessingError(
        "Alert processing failed",
        alert_id="alert_001",
        alert_rule="cpu_high"
    )
    
    assert error.alert_id == "alert_001"
    assert error.alert_rule == "cpu_high"


# ============================================================================
# 通知错误测试
# ============================================================================

def test_notification_error():
    """测试NotificationError"""
    from src.infrastructure.monitoring.core.exceptions import NotificationError
    
    error = NotificationError(
        "Notification failed",
        notification_type="email",
        recipient="admin@example.com"
    )
    
    assert error.notification_type == "email"
    assert error.recipient == "admin@example.com"


# ============================================================================
# 健康检查错误测试
# ============================================================================

def test_health_check_error():
    """测试HealthCheckError"""
    from src.infrastructure.monitoring.core.exceptions import HealthCheckError
    
    error = HealthCheckError(
        "Health check failed",
        check_type="http",
        check_target="api.example.com"
    )
    
    assert error.check_type == "http"
    assert error.check_target == "api.example.com"


# ============================================================================
# 阈值超限错误测试
# ============================================================================

def test_threshold_exceeded_error():
    """测试ThresholdExceededError"""
    from src.infrastructure.monitoring.core.exceptions import ThresholdExceededError
    
    error = ThresholdExceededError(
        "Threshold exceeded",
        metric_name="memory_usage",
        threshold_value=80.0,
        actual_value=95.5
    )
    
    assert error.metric_name == "memory_usage"
    assert error.threshold_value == 80.0
    assert error.actual_value == 95.5


# ============================================================================
# 连接错误测试
# ============================================================================

def test_monitor_connection_error():
    """测试MonitorConnectionError"""
    from src.infrastructure.monitoring.core.exceptions import MonitorConnectionError
    
    error = MonitorConnectionError(
        "Connection failed",
        target_host="192.168.1.1",
        target_port=8080
    )
    
    assert error.target_host == "192.168.1.1"
    assert error.target_port == 8080


# ============================================================================
# 数据处理错误测试
# ============================================================================

def test_data_processing_error():
    """测试DataProcessingError"""
    from src.infrastructure.monitoring.core.exceptions import DataProcessingError
    
    error = DataProcessingError(
        "Data processing failed",
        data_type="metrics",
        processing_step="aggregation"
    )
    
    assert error.data_type == "metrics"
    assert error.processing_step == "aggregation"


# ============================================================================
# 存储错误测试
# ============================================================================

def test_storage_error():
    """测试StorageError"""
    from src.infrastructure.monitoring.core.exceptions import StorageError
    
    error = StorageError(
        "Storage operation failed",
        storage_type="database",
        operation="write"
    )
    
    assert error.storage_type == "database"
    assert error.operation == "write"


# ============================================================================
# 告警规则错误测试
# ============================================================================

def test_alert_rule_error():
    """测试AlertRuleError"""
    from src.infrastructure.monitoring.core.exceptions import AlertRuleError
    
    error = AlertRuleError(
        "Rule validation failed",
        rule_id="rule_123",
        rule_condition="cpu > 90"
    )
    
    assert error.rule_id == "rule_123"
    assert error.rule_condition == "cpu > 90"


# ============================================================================
# 装饰器测试
# ============================================================================

def test_handle_monitoring_exception_decorator():
    """测试handle_monitoring_exception装饰器"""
    from src.infrastructure.monitoring.core.exceptions import (
        handle_monitoring_exception,
        MonitoringException
    )
    
    @handle_monitoring_exception(operation="test_operation")
    def test_func():
        return "success"
    
    result = test_func()
    assert result == "success"


def test_handle_monitoring_exception_catches_error():
    """测试装饰器捕获错误"""
    from src.infrastructure.monitoring.core.exceptions import (
        handle_monitoring_exception,
        MonitoringException
    )
    
    @handle_monitoring_exception(operation="test_operation")
    def test_func():
        raise ValueError("Test error")
    
    with pytest.raises(Exception):  # 装饰器会将异常包装为MonitoringException
        test_func()


def test_handle_metric_collection_exception_decorator():
    """测试handle_metric_collection_exception装饰器"""
    from src.infrastructure.monitoring.core.exceptions import (
        handle_metric_collection_exception,
        MetricCollectionError
    )
    
    @handle_metric_collection_exception(metric_name="cpu", collection_method="psutil")
    def collect_metric():
        return {"cpu": 50}
    
    result = collect_metric()
    assert result["cpu"] == 50


def test_handle_metric_collection_exception_catches_error():
    """测试指标收集装饰器捕获错误"""
    from src.infrastructure.monitoring.core.exceptions import (
        handle_metric_collection_exception,
        MetricCollectionError
    )
    
    @handle_metric_collection_exception(metric_name="memory", collection_method="api")
    def collect_metric():
        raise RuntimeError("Collection failed")
    
    with pytest.raises(MetricCollectionError) as exc_info:
        collect_metric()
    
    assert exc_info.value.metric_name == "memory"
    assert exc_info.value.collection_method == "api"


def test_all_exceptions_inherit_monitoring_exception():
    """测试所有异常都继承自MonitoringException"""
    from src.infrastructure.monitoring.core.exceptions import (
        MonitoringException,
        MonitorConfigurationError,
        MetricCollectionError,
        AlertProcessingError,
        NotificationError,
        HealthCheckError
    )
    
    assert issubclass(MonitorConfigurationError, MonitoringException)
    assert issubclass(MetricCollectionError, MonitoringException)
    assert issubclass(AlertProcessingError, MonitoringException)
    assert issubclass(NotificationError, MonitoringException)
    assert issubclass(HealthCheckError, MonitoringException)

