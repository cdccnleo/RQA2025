#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控层异常类覆盖率测试
专注提升exceptions.py的测试覆盖率
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
    exceptions_module = importlib.import_module('src.monitoring.core.exceptions')
    MonitoringException = getattr(exceptions_module, 'MonitoringException', None)
    MetricsCollectionError = getattr(exceptions_module, 'MetricsCollectionError', None)
    AlertProcessingError = getattr(exceptions_module, 'AlertProcessingError', None)
    ConfigurationError = getattr(exceptions_module, 'ConfigurationError', None)
    HealthCheckError = getattr(exceptions_module, 'HealthCheckError', None)
    ResourceExhaustionError = getattr(exceptions_module, 'ResourceExhaustionError', None)
    DataPersistenceError = getattr(exceptions_module, 'DataPersistenceError', None)
    handle_monitoring_exception = getattr(exceptions_module, 'handle_monitoring_exception', None)
    validate_metric_data = getattr(exceptions_module, 'validate_metric_data', None)
    validate_config_key = getattr(exceptions_module, 'validate_config_key', None)
    
    if MonitoringException is None:
        pytest.skip("监控异常模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("监控异常模块导入失败", allow_module_level=True)


class TestMonitoringException:
    """测试MonitoringException基础异常类"""

    def test_initialization(self):
        """测试初始化"""
        error = MonitoringException("Test error message")
        assert "Test error message" in str(error)
        assert isinstance(error, Exception)
        assert hasattr(error, 'error_code')

    def test_with_error_code(self):
        """测试带错误码"""
        error = MonitoringException("Test error", error_code=1001)
        assert error.error_code == 1001

    def test_inheritance(self):
        """测试继承关系"""
        error = MonitoringException("Test")
        assert isinstance(error, Exception)


class TestMetricsCollectionError:
    """测试MetricsCollectionError"""

    def test_initialization(self):
        """测试初始化"""
        error = MetricsCollectionError("Collection error")
        assert "Collection error" in str(error)
        assert isinstance(error, MonitoringException)

    def test_with_metric_name(self):
        """测试带指标名"""
        error = MetricsCollectionError("Error", metric_name="test_metric")
        assert "test_metric" in str(error)
        assert error.metric_name == "test_metric"


class TestAlertProcessingError:
    """测试AlertProcessingError"""

    def test_initialization(self):
        """测试初始化"""
        error = AlertProcessingError("Alert error")
        assert "Alert error" in str(error)
        assert isinstance(error, MonitoringException)

    def test_with_alert_id(self):
        """测试带告警ID"""
        error = AlertProcessingError("Error", alert_id="test_alert_123")
        assert "test_alert_123" in str(error)
        assert error.alert_id == "test_alert_123"


class TestConfigurationError:
    """测试ConfigurationError"""

    def test_initialization(self):
        """测试初始化"""
        error = ConfigurationError("Config error")
        assert "Config error" in str(error)
        assert isinstance(error, MonitoringException)

    def test_with_config_key(self):
        """测试带配置键"""
        error = ConfigurationError("Invalid config", config_key="test_key")
        assert "test_key" in str(error)
        assert error.config_key == "test_key"


class TestHealthCheckError:
    """测试HealthCheckError"""

    def test_initialization(self):
        """测试初始化"""
        error = HealthCheckError("Health check error")
        assert "Health check error" in str(error)
        assert isinstance(error, MonitoringException)

    def test_with_component(self):
        """测试带组件名"""
        error = HealthCheckError("Error", component="test_component")
        assert "test_component" in str(error)
        assert error.component == "test_component"


class TestResourceExhaustionError:
    """测试ResourceExhaustionError"""

    def test_initialization(self):
        """测试初始化"""
        error = ResourceExhaustionError("Resource error")
        assert "Resource error" in str(error)
        assert isinstance(error, MonitoringException)

    def test_with_resource_type(self):
        """测试带资源类型"""
        error = ResourceExhaustionError("Error", resource_type="memory")
        assert "memory" in str(error)
        assert error.resource_type == "memory"


class TestDataPersistenceError:
    """测试DataPersistenceError"""

    def test_initialization(self):
        """测试初始化"""
        error = DataPersistenceError("Persistence error")
        assert "Persistence error" in str(error)
        assert isinstance(error, MonitoringException)

    def test_with_data_type(self):
        """测试带数据类型"""
        error = DataPersistenceError("Error", data_type="metrics")
        assert "metrics" in str(error)
        assert error.data_type == "metrics"


class TestExceptionUsage:
    """测试异常使用场景"""

    def test_exception_raising(self):
        """测试抛出异常"""
        with pytest.raises(MonitoringException):
            raise MonitoringException("Test error")

    def test_exception_chaining(self):
        """测试异常链"""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise MonitoringException("Wrapped error") from e
        except MonitoringException as e:
            assert isinstance(e, MonitoringException)
            assert e.__cause__ is not None

    def test_all_exception_types(self):
        """测试所有异常类型"""
        exceptions = [
            MonitoringException("Base error"),
            ConfigurationError("Config error"),
            MetricsCollectionError("Collection error"),
            AlertProcessingError("Alert error"),
            HealthCheckError("Health error"),
            ResourceExhaustionError("Resource error"),
            DataPersistenceError("Persistence error")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, MonitoringException)
            assert len(str(exc)) > 0


class TestExceptionUtilities:
    """测试异常工具函数"""

    def test_validate_metric_data_valid(self):
        """测试验证有效指标数据"""
        # 应该不抛出异常
        validate_metric_data("test_metric", 10.0, float)

    def test_validate_metric_data_none(self):
        """测试验证None值"""
        with pytest.raises(MetricsCollectionError):
            validate_metric_data("test_metric", None)

    def test_validate_metric_data_wrong_type(self):
        """测试验证错误类型"""
        with pytest.raises(MetricsCollectionError):
            validate_metric_data("test_metric", "not_a_float", float)

    def test_validate_config_key_exists(self):
        """测试验证存在的配置键"""
        config = {"test_key": "test_value"}
        # 应该不抛出异常
        validate_config_key(config, "test_key")

    def test_validate_config_key_required_missing(self):
        """测试验证必需的配置键缺失"""
        config = {}
        with pytest.raises(ConfigurationError):
            validate_config_key(config, "required_key", required=True)

    def test_validate_config_key_none_value(self):
        """测试验证配置键值为None"""
        config = {"test_key": None}
        with pytest.raises(ConfigurationError):
            validate_config_key(config, "test_key")

    def test_handle_monitoring_exception_decorator(self):
        """测试异常处理装饰器"""
        @handle_monitoring_exception
        def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(MonitoringException):
            test_func()

    def test_handle_monitoring_exception_preserves_monitoring_exception(self):
        """测试装饰器保留监控异常"""
        @handle_monitoring_exception
        def test_func():
            raise MonitoringException("Monitoring error")
        
        with pytest.raises(MonitoringException) as exc_info:
            test_func()
        assert "Monitoring error" in str(exc_info.value)

