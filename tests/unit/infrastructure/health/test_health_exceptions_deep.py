#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Health核心异常深度测试"""

import pytest
from datetime import datetime


# ============================================================================
# HealthInfrastructureError基础异常测试
# ============================================================================

def test_health_infrastructure_error_basic():
    """测试HealthInfrastructureError基础创建"""
    from src.infrastructure.health.core.exceptions import HealthInfrastructureError
    
    error = HealthInfrastructureError("Test error")
    assert error.message == "Test error"
    assert error.error_code == "HEALTH_INFRA_ERROR"
    assert error.details == {}
    assert error.timestamp is not None


def test_health_infrastructure_error_with_code():
    """测试HealthInfrastructureError包含错误码"""
    from src.infrastructure.health.core.exceptions import HealthInfrastructureError
    
    error = HealthInfrastructureError("Error", error_code="CUSTOM_ERROR")
    assert error.error_code == "CUSTOM_ERROR"


def test_health_infrastructure_error_with_details():
    """测试HealthInfrastructureError包含详情"""
    from src.infrastructure.health.core.exceptions import HealthInfrastructureError
    
    details = {"service": "database", "status": "down"}
    error = HealthInfrastructureError("Error", details=details)
    assert error.details == details


def test_health_infrastructure_error_to_dict():
    """测试HealthInfrastructureError转字典"""
    from src.infrastructure.health.core.exceptions import HealthInfrastructureError
    
    error = HealthInfrastructureError("Test", error_code="TEST_ERROR")
    error_dict = error.to_dict()
    
    assert isinstance(error_dict, dict)
    assert error_dict["error_type"] == "HealthInfrastructureError"
    assert error_dict["message"] == "Test"
    assert error_dict["error_code"] == "TEST_ERROR"
    assert "timestamp" in error_dict


def test_health_infrastructure_error_timestamp_format():
    """测试HealthInfrastructureError时间戳格式"""
    from src.infrastructure.health.core.exceptions import HealthInfrastructureError
    
    error = HealthInfrastructureError("Test")
    # 验证时间戳是ISO格式
    assert isinstance(error.timestamp, str)
    assert "T" in error.timestamp or "-" in error.timestamp


# ============================================================================
# LoadBalancerError测试
# ============================================================================

def test_load_balancer_error():
    """测试LoadBalancerError"""
    from src.infrastructure.health.core.exceptions import LoadBalancerError
    
    backend_status = {"server1": "healthy", "server2": "down"}
    error = LoadBalancerError(
        "Load balancer error",
        service_name="api-service",
        backend_status=backend_status
    )
    
    assert error.error_code == "LOAD_BALANCER_ERROR"
    assert error.details["service_name"] == "api-service"
    assert error.details["backend_status"] == backend_status


def test_load_balancer_error_minimal():
    """测试LoadBalancerError最小参数"""
    from src.infrastructure.health.core.exceptions import LoadBalancerError
    
    error = LoadBalancerError("Error")
    assert error.error_code == "LOAD_BALANCER_ERROR"


# ============================================================================
# HealthCheckError测试
# ============================================================================

def test_health_check_error():
    """测试HealthCheckError"""
    from src.infrastructure.health.core.exceptions import HealthCheckError
    
    last_result = {"status": "unhealthy", "response_time": 5000}
    error = HealthCheckError(
        "Health check failed",
        component="database",
        check_type="ping",
        last_result=last_result
    )
    
    assert error.error_code == "HEALTH_CHECK_ERROR"
    assert error.details["component"] == "database"
    assert error.details["check_type"] == "ping"
    assert error.details["last_result"] == last_result


def test_health_check_error_minimal():
    """测试HealthCheckError最小参数"""
    from src.infrastructure.health.core.exceptions import HealthCheckError
    
    error = HealthCheckError("Check failed")
    assert error.error_code == "HEALTH_CHECK_ERROR"


# ============================================================================
# MonitoringError测试
# ============================================================================

def test_monitoring_error():
    """测试MonitoringError"""
    from src.infrastructure.health.core.exceptions import MonitoringError
    
    metrics = {"cpu": 90.5, "memory": 85.0}
    error = MonitoringError(
        "Monitoring error",
        monitor_type="system",
        metrics=metrics,
        alert_triggered=True
    )
    
    assert error.error_code == "MONITORING_ERROR"
    assert error.details["monitor_type"] == "system"
    assert error.details["metrics"] == metrics
    assert error.details["alert_triggered"] is True


def test_monitoring_error_no_alert():
    """测试MonitoringError未触发告警"""
    from src.infrastructure.health.core.exceptions import MonitoringError
    
    error = MonitoringError("Error", alert_triggered=False)
    assert error.details["alert_triggered"] is False


# ============================================================================
# ConfigurationError测试
# ============================================================================

def test_configuration_error():
    """测试ConfigurationError"""
    from src.infrastructure.health.core.exceptions import ConfigurationError
    
    error = ConfigurationError(
        "Config error",
        config_key="health.check.interval",
        config_value=30
    )
    
    assert error.error_code == "CONFIGURATION_ERROR"
    assert error.details["config_key"] == "health.check.interval"
    assert error.details["config_value"] == 30


def test_configuration_error_minimal():
    """测试ConfigurationError最小参数"""
    from src.infrastructure.health.core.exceptions import ConfigurationError
    
    error = ConfigurationError("Config error")
    assert error.error_code == "CONFIGURATION_ERROR"


# ============================================================================
# ValidationError测试
# ============================================================================

def test_validation_error():
    """测试ValidationError"""
    from src.infrastructure.health.core.exceptions import ValidationError
    
    error = ValidationError(
        "Validation failed",
        field="email",
        value="invalid",
        validation_rule="email_format"
    )
    
    assert error.error_code == "VALIDATION_ERROR"
    assert error.details["field"] == "email"
    assert error.details["value"] == "invalid"
    assert error.details["validation_rule"] == "email_format"


def test_validation_error_minimal():
    """测试ValidationError最小参数"""
    from src.infrastructure.health.core.exceptions import ValidationError
    
    error = ValidationError("Validation failed")
    assert error.error_code == "VALIDATION_ERROR"


# ============================================================================
# AsyncOperationError测试
# ============================================================================

def test_async_operation_error():
    """测试AsyncOperationError"""
    from src.infrastructure.health.core.exceptions import AsyncOperationError
    
    coroutine_info = {"name": "check_health", "state": "pending"}
    error = AsyncOperationError(
        "Async operation failed",
        operation="health_check",
        timeout=30.0,
        coroutine_info=coroutine_info
    )
    
    assert error.error_code == "ASYNC_OPERATION_ERROR"
    assert error.details["operation"] == "health_check"
    assert error.details["timeout"] == 30.0
    assert error.details["coroutine_info"] == coroutine_info


def test_async_operation_error_minimal():
    """测试AsyncOperationError最小参数"""
    from src.infrastructure.health.core.exceptions import AsyncOperationError
    
    error = AsyncOperationError("Async error")
    assert error.error_code == "ASYNC_OPERATION_ERROR"


# ============================================================================
# 异常继承关系测试
# ============================================================================

def test_all_errors_inherit_health_infrastructure_error():
    """测试所有异常都继承自HealthInfrastructureError"""
    from src.infrastructure.health.core.exceptions import (
        HealthInfrastructureError,
        LoadBalancerError,
        HealthCheckError,
        MonitoringError,
        ConfigurationError,
        ValidationError,
        AsyncOperationError
    )
    
    assert issubclass(LoadBalancerError, HealthInfrastructureError)
    assert issubclass(HealthCheckError, HealthInfrastructureError)
    assert issubclass(MonitoringError, HealthInfrastructureError)
    assert issubclass(ConfigurationError, HealthInfrastructureError)
    assert issubclass(ValidationError, HealthInfrastructureError)
    assert issubclass(AsyncOperationError, HealthInfrastructureError)


def test_all_errors_inherit_exception():
    """测试所有异常都继承自Exception"""
    from src.infrastructure.health.core.exceptions import HealthInfrastructureError
    
    assert issubclass(HealthInfrastructureError, Exception)


# ============================================================================
# to_dict()功能测试
# ============================================================================

def test_load_balancer_error_to_dict():
    """测试LoadBalancerError的to_dict"""
    from src.infrastructure.health.core.exceptions import LoadBalancerError
    
    error = LoadBalancerError("Error", service_name="api")
    error_dict = error.to_dict()
    
    assert error_dict["error_type"] == "LoadBalancerError"
    assert error_dict["error_code"] == "LOAD_BALANCER_ERROR"


def test_health_check_error_to_dict():
    """测试HealthCheckError的to_dict"""
    from src.infrastructure.health.core.exceptions import HealthCheckError
    
    error = HealthCheckError("Error", component="db")
    error_dict = error.to_dict()
    
    assert error_dict["error_type"] == "HealthCheckError"
    assert "timestamp" in error_dict


# ============================================================================
# 异常使用场景测试
# ============================================================================

def test_exception_can_be_raised():
    """测试异常可以被raise"""
    from src.infrastructure.health.core.exceptions import HealthCheckError
    
    with pytest.raises(HealthCheckError) as exc_info:
        raise HealthCheckError("Test", component="test")
    
    assert exc_info.value.details["component"] == "test"


def test_exception_catching_by_base_class():
    """测试通过基类捕获异常"""
    from src.infrastructure.health.core.exceptions import (
        HealthInfrastructureError,
        HealthCheckError
    )
    
    try:
        raise HealthCheckError("Test")
    except HealthInfrastructureError as e:
        assert isinstance(e, HealthCheckError)


def test_exception_with_none_values():
    """测试异常处理None值"""
    from src.infrastructure.health.core.exceptions import ValidationError
    
    error = ValidationError("Error", field="test", value=None)
    # None值不应导致错误
    assert error.details.get("value") is None


def test_multiple_exceptions_independent():
    """测试多个异常实例独立"""
    from src.infrastructure.health.core.exceptions import HealthCheckError
    
    error1 = HealthCheckError("Error1", component="comp1")
    error2 = HealthCheckError("Error2", component="comp2")
    
    assert error1.details["component"] == "comp1"
    assert error2.details["component"] == "comp2"
    assert error1.timestamp != error2.timestamp or error1 is not error2

