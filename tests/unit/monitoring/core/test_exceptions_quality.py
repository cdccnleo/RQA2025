#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控异常质量测试
测试覆盖监控层的异常类
"""

import sys
import importlib
from pathlib import Path
import pytest

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    core_exceptions_module = importlib.import_module('src.monitoring.core.exceptions')
    MonitoringException = getattr(core_exceptions_module, 'MonitoringException', None)
    MetricsCollectionError = getattr(core_exceptions_module, 'MetricsCollectionError', None)
    AlertProcessingError = getattr(core_exceptions_module, 'AlertProcessingError', None)
    ConfigurationError = getattr(core_exceptions_module, 'ConfigurationError', None)
    HealthCheckError = getattr(core_exceptions_module, 'HealthCheckError', None)
    ResourceExhaustionError = getattr(core_exceptions_module, 'ResourceExhaustionError', None)
    DataPersistenceError = getattr(core_exceptions_module, 'DataPersistenceError', None)
    handle_monitoring_exception = getattr(core_exceptions_module, 'handle_monitoring_exception', None)
    validate_metric_data = getattr(core_exceptions_module, 'validate_metric_data', None)
    validate_config_key = getattr(core_exceptions_module, 'validate_config_key', None)
    
    if MonitoringException is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestMonitoringException:
    """MonitoringException测试类"""

    def test_monitoring_exception_creation(self):
        """测试监控异常创建"""
        exception = MonitoringException("Test error")
        assert str(exception) == "Test error"
        assert isinstance(exception, Exception)

    def test_monitoring_exception_with_error_code(self):
        """测试监控异常（带错误码）"""
        exception = MonitoringException("Test error", error_code=1001)
        assert str(exception) == "Test error"
        assert exception.error_code == 1001


class TestMetricsCollectionError:
    """MetricsCollectionError测试类"""

    def test_metrics_collection_error(self):
        """测试指标收集错误"""
        error = MetricsCollectionError("Collection failed", "test_metric")
        assert "Collection failed" in str(error)
        assert isinstance(error, MonitoringException)
        assert error.metric_name == "test_metric"


class TestAlertProcessingError:
    """AlertProcessingError测试类"""

    def test_alert_processing_error(self):
        """测试告警处理错误"""
        error = AlertProcessingError("Alert processing failed", "test_alert")
        assert "Alert processing failed" in str(error)
        assert isinstance(error, MonitoringException)
        assert error.alert_id == "test_alert"


class TestConfigurationError:
    """ConfigurationError测试类"""

    def test_configuration_error(self):
        """测试配置错误"""
        error = ConfigurationError("Config failed", "test_key")
        assert "Config failed" in str(error)
        assert isinstance(error, MonitoringException)

class TestHealthCheckError:
    """HealthCheckError测试类"""

    def test_health_check_error(self):
        """测试健康检查错误"""
        error = HealthCheckError("Health check failed", "test_component")
        assert "Health check failed" in str(error)
        assert isinstance(error, MonitoringException)

class TestResourceExhaustionError:
    """ResourceExhaustionError测试类"""

    def test_resource_exhaustion_error(self):
        """测试资源耗尽错误"""
        error = ResourceExhaustionError("Resource exhausted", "memory")
        assert "Resource exhausted" in str(error)
        assert isinstance(error, MonitoringException)

class TestDataPersistenceError:
    """DataPersistenceError测试类"""

    def test_data_persistence_error(self):
        """测试数据持久化错误"""
        error = DataPersistenceError("Persistence failed", "metrics")
        assert "Persistence failed" in str(error)
        assert isinstance(error, MonitoringException)

class TestExceptionUtilities:
    """异常工具函数测试类"""

    def test_validate_metric_data(self):
        """测试验证指标数据"""
        # 正常情况
        validate_metric_data('test_metric', 100.0, float)
        
        # None值应该抛出异常
        with pytest.raises(MetricsCollectionError):
            validate_metric_data('test_metric', None)
        
        # 类型错误应该抛出异常
        with pytest.raises(MetricsCollectionError):
            validate_metric_data('test_metric', "not_a_float", float)

    def test_validate_config_key(self):
        """测试验证配置键"""
        config = {'key1': 'value1', 'key2': None}
        
        # 必需键存在
        validate_config_key(config, 'key1', required=True)
        
        # 必需键不存在应该抛出异常
        with pytest.raises(ConfigurationError):
            validate_config_key(config, 'missing_key', required=True)
        
        # 键值为None应该抛出异常
        with pytest.raises(ConfigurationError):
            validate_config_key(config, 'key2', required=True)

    def test_handle_monitoring_exception_decorator(self):
        """测试监控异常处理装饰器"""
        @handle_monitoring_exception
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
        
        @handle_monitoring_exception
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(MonitoringException):
            failing_func()

