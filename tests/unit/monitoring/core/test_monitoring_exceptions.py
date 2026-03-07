"""
监控服务层核心异常测试
测试监控相关的异常类和错误处理机制
"""

import pytest
from pathlib import Path
import sys

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 导入异常类
from src.monitoring.core.exceptions import (
    MonitoringException,
    MetricsCollectionError,
    AlertProcessingError,
    ConfigurationError,
    HealthCheckError,
# ResourceExhaustionError, DataPersistenceError
)


class TestMonitoringExceptions:
    """监控异常测试"""

    def test_monitoring_exception_basic(self):
        """测试基础监控异常"""
        message = "Monitoring operation failed"
        error_code = 500

        exception = MonitoringException(message, error_code)

        assert str(exception) == message
        assert exception.error_code == error_code
        assert exception.message == message

    def test_metrics_collection_error(self):
        """测试指标收集异常"""
        message = "Failed to collect CPU metrics"
        metric_name = "cpu_usage"

        exception = MetricsCollectionError(message, metric_name)

        assert "指标收集失败" in str(exception)
        assert metric_name in str(exception)
        assert exception.metric_name == metric_name

    def test_alert_processing_error(self):
        """测试告警处理异常"""
        message = "Failed to send alert notification"
        alert_id = "alert_001"

        exception = AlertProcessingError(message, alert_id)

        assert "告警处理失败" in str(exception)
        assert alert_id in str(exception)
        assert exception.alert_id == alert_id

    def test_configuration_error(self):
        """测试配置异常"""
        message = "Invalid alert threshold"
        config_key = "alert.cpu_threshold"

        exception = ConfigurationError(message, config_key)

        assert "配置错误" in str(exception)
        assert config_key in str(exception)
        assert exception.config_key == config_key

    def test_health_check_error(self):
        """测试健康检查异常"""
        message = "Database connection failed"
        component = "database_service"

        exception = HealthCheckError(message, component)

        assert "健康检查失败" in str(exception)
        assert component in str(exception)
        assert exception.component == component

    def test_resource_exhaustion_error(self):
        """测试资源耗尽异常"""
        message = "Memory usage above threshold"
        resource_type = "memory"

        exception = ResourceExhaustionError(message, resource_type)

        assert "资源耗尽" in str(exception)
        assert resource_type in str(exception)
        assert exception.resource_type == resource_type

    def test_data_persistence_error(self):
        """测试数据持久化异常"""
        message = "Failed to save metrics data"
        data_type = "metrics"

        exception = DataPersistenceError(message, data_type)

        assert "数据持久化失败" in str(exception)
        assert data_type in str(exception)
        assert exception.data_type == data_type

    def test_exception_inheritance(self):
        """测试异常继承关系"""
        base_exception = MonitoringException("test")
        assert isinstance(base_exception, Exception)

        metrics_error = MetricsCollectionError("test", "cpu")
        assert isinstance(metrics_error, MonitoringException)

        alert_error = AlertProcessingError("test", "alert_001")
        assert isinstance(alert_error, MonitoringException)

        assert issubclass(MetricsCollectionError, MonitoringException)
        assert issubclass(AlertProcessingError, MonitoringException)

    def test_exception_with_default_values(self):
        """测试异常默认值"""
        # 测试没有额外参数的异常
        exception = MonitoringException("test")
        assert exception.error_code == -1

        # 测试有默认参数的异常
        metrics_error = MetricsCollectionError("test")
        assert metrics_error.metric_name is None

        alert_error = AlertProcessingError("test")
        assert alert_error.alert_id is None
