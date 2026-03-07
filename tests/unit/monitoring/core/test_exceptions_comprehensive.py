#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控层异常类综合测试
补充exceptions.py中所有异常类的完整测试，覆盖边界情况
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
    
    if MonitoringException is None:
        pytest.skip("监控异常模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("监控异常模块导入失败", allow_module_level=True)


class TestMonitoringExceptionComprehensive:
    """测试MonitoringException基础异常类-综合测试"""

    def test_initialization_default_error_code(self):
        """测试初始化默认错误码"""
        error = MonitoringException("Test error")
        assert error.error_code == -1  # 默认错误码

    def test_initialization_custom_error_code(self):
        """测试初始化自定义错误码"""
        error = MonitoringException("Test error", error_code=2001)
        assert error.error_code == 2001
        assert error.message == "Test error"

    def test_str_representation(self):
        """测试字符串表示"""
        error = MonitoringException("Test error message")
        assert "Test error message" in str(error)

    def test_exception_attributes(self):
        """测试异常属性"""
        error = MonitoringException("Test", error_code=1001)
        assert hasattr(error, 'error_code')
        assert hasattr(error, 'message')
        assert error.error_code == 1001
        assert error.message == "Test"


class TestMetricsCollectionErrorComprehensive:
    """测试MetricsCollectionError-综合测试"""

    def test_initialization_without_metric_name(self):
        """测试初始化不带指标名"""
        error = MetricsCollectionError("Collection failed")
        assert isinstance(error, MonitoringException)
        assert error.metric_name is None

    def test_initialization_with_metric_name(self):
        """测试初始化带指标名"""
        error = MetricsCollectionError("Collection failed", metric_name="cpu_usage")
        assert error.metric_name == "cpu_usage"
        assert "cpu_usage" in str(error)

    def test_inheritance(self):
        """测试继承关系"""
        error = MetricsCollectionError("Error", "metric")
        assert isinstance(error, MonitoringException)
        assert isinstance(error, Exception)


class TestAlertProcessingErrorComprehensive:
    """测试AlertProcessingError-综合测试"""

    def test_initialization_without_alert_id(self):
        """测试初始化不带告警ID"""
        error = AlertProcessingError("Processing failed")
        assert isinstance(error, MonitoringException)
        assert error.alert_id is None

    def test_initialization_with_alert_id(self):
        """测试初始化带告警ID"""
        error = AlertProcessingError("Processing failed", alert_id="alert_123")
        assert error.alert_id == "alert_123"
        assert "alert_123" in str(error)

    def test_inheritance(self):
        """测试继承关系"""
        error = AlertProcessingError("Error", "alert_1")
        assert isinstance(error, MonitoringException)
        assert isinstance(error, Exception)


class TestConfigurationErrorComprehensive:
    """测试ConfigurationError-综合测试"""

    def test_initialization_without_config_key(self):
        """测试初始化不带配置键"""
        error = ConfigurationError("Config failed")
        assert isinstance(error, MonitoringException)
        assert error.config_key is None

    def test_initialization_with_config_key(self):
        """测试初始化带配置键"""
        error = ConfigurationError("Config failed", config_key="database_url")
        assert error.config_key == "database_url"
        assert "database_url" in str(error)

    def test_inheritance(self):
        """测试继承关系"""
        error = ConfigurationError("Error", "key1")
        assert isinstance(error, MonitoringException)
        assert isinstance(error, Exception)


class TestHealthCheckErrorComprehensive:
    """测试HealthCheckError-综合测试"""

    def test_initialization_without_component(self):
        """测试初始化不带组件名"""
        error = HealthCheckError("Health check failed")
        assert isinstance(error, MonitoringException)
        assert error.component is None

    def test_initialization_with_component(self):
        """测试初始化带组件名"""
        error = HealthCheckError("Health check failed", component="Database")
        assert error.component == "Database"
        assert "Database" in str(error)

    def test_inheritance(self):
        """测试继承关系"""
        error = HealthCheckError("Error", "Component1")
        assert isinstance(error, MonitoringException)
        assert isinstance(error, Exception)


class TestResourceExhaustionErrorComprehensive:
    """测试ResourceExhaustionError-综合测试"""

    def test_initialization_without_resource_type(self):
        """测试初始化不带资源类型"""
        error = ResourceExhaustionError("Resource exhausted")
        assert isinstance(error, MonitoringException)
        assert error.resource_type is None

    def test_initialization_with_resource_type(self):
        """测试初始化带资源类型"""
        error = ResourceExhaustionError("Resource exhausted", resource_type="Memory")
        assert error.resource_type == "Memory"
        assert "Memory" in str(error)

    def test_inheritance(self):
        """测试继承关系"""
        error = ResourceExhaustionError("Error", "CPU")
        assert isinstance(error, MonitoringException)
        assert isinstance(error, Exception)


class TestDataPersistenceErrorComprehensive:
    """测试DataPersistenceError-综合测试"""

    def test_initialization_without_data_type(self):
        """测试初始化不带数据类型"""
        error = DataPersistenceError("Persistence failed")
        assert isinstance(error, MonitoringException)
        assert error.data_type is None

    def test_initialization_with_data_type(self):
        """测试初始化带数据类型"""
        error = DataPersistenceError("Persistence failed", data_type="Metrics")
        assert error.data_type == "Metrics"
        assert "Metrics" in str(error)

    def test_inheritance(self):
        """测试继承关系"""
        error = DataPersistenceError("Error", "Data1")
        assert isinstance(error, MonitoringException)
        assert isinstance(error, Exception)


class TestExceptionHierarchy:
    """测试异常层次结构"""

    def test_all_exceptions_inherit_from_monitoring_exception(self):
        """测试所有异常都继承自MonitoringException"""
        exceptions = [
            MetricsCollectionError("Error"),
            AlertProcessingError("Error"),
            ConfigurationError("Error"),
            HealthCheckError("Error"),
            ResourceExhaustionError("Error"),
            DataPersistenceError("Error")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, MonitoringException)
            assert isinstance(exc, Exception)

    def test_all_exceptions_have_error_code(self):
        """测试所有异常都有error_code属性"""
        exceptions = [
            MetricsCollectionError("Error"),
            AlertProcessingError("Error"),
            ConfigurationError("Error"),
            HealthCheckError("Error"),
            ResourceExhaustionError("Error"),
            DataPersistenceError("Error")
        ]
        
        for exc in exceptions:
            assert hasattr(exc, 'error_code')
            assert exc.error_code == -1  # 默认错误码



