#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层异常测试

测试目标：提升utils/core/exceptions.py的真实覆盖率
实际导入和使用src.infrastructure.utils.core.exceptions模块
"""

import pytest
from unittest.mock import patch


class TestInfrastructureError:
    """测试基础设施基础异常类"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import InfrastructureError
        
        exc = InfrastructureError("Test error")
        assert str(exc) == "Test error"
        assert exc.message == "Test error"
        assert exc.error_code == "INFRASTRUCTURE_ERROR"
        assert exc.details == {}
    
    def test_init_with_error_code(self):
        """测试使用错误码初始化"""
        from src.infrastructure.utils.core.exceptions import InfrastructureError
        
        exc = InfrastructureError("Test error", error_code="CUSTOM_ERROR")
        assert exc.error_code == "CUSTOM_ERROR"
    
    def test_init_with_details(self):
        """测试使用详细信息初始化"""
        from src.infrastructure.utils.core.exceptions import InfrastructureError
        
        details = {"key": "value", "count": 123}
        exc = InfrastructureError("Test error", details=details)
        assert exc.details == details
    
    def test_to_dict(self):
        """测试转换为字典"""
        from src.infrastructure.utils.core.exceptions import InfrastructureError
        
        exc = InfrastructureError("Test error", error_code="TEST_ERROR", details={"key": "value"})
        result = exc.to_dict()
        
        assert result["error_type"] == "InfrastructureError"
        assert result["error_code"] == "TEST_ERROR"
        assert result["message"] == "Test error"
        assert result["details"] == {"key": "value"}


class TestConfigurationError:
    """测试配置异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import ConfigurationError
        
        exc = ConfigurationError("Invalid config")
        assert "Invalid config" in str(exc)
        assert exc.error_code == "CONFIG_ERROR"
        assert exc.details.get("config_key") is None
    
    def test_init_with_config_key(self):
        """测试使用配置键初始化"""
        from src.infrastructure.utils.core.exceptions import ConfigurationError
        
        exc = ConfigurationError("Invalid config", config_key="database.host")
        assert exc.details.get("config_key") == "database.host"


class TestDataProcessingError:
    """测试数据处理异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import DataProcessingError
        
        exc = DataProcessingError("Processing failed")
        assert exc.error_code == "DATA_PROCESSING_ERROR"
        assert exc.details.get("data_source") is None
        assert exc.details.get("operation") is None
    
    def test_init_with_data_source(self):
        """测试使用数据源初始化"""
        from src.infrastructure.utils.core.exceptions import DataProcessingError
        
        exc = DataProcessingError("Processing failed", data_source="database")
        assert exc.details.get("data_source") == "database"
    
    def test_init_with_operation(self):
        """测试使用操作初始化"""
        from src.infrastructure.utils.core.exceptions import DataProcessingError
        
        exc = DataProcessingError("Processing failed", operation="transform")
        assert exc.details.get("operation") == "transform"


class TestConnectionError:
    """测试连接异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import ConnectionError
        
        exc = ConnectionError("Connection failed")
        assert exc.error_code == "CONNECTION_ERROR"
        assert exc.details.get("host") is None
        assert exc.details.get("port") is None
    
    def test_init_with_host_and_port(self):
        """测试使用主机和端口初始化"""
        from src.infrastructure.utils.core.exceptions import ConnectionError
        
        exc = ConnectionError("Connection failed", host="localhost", port=5432)
        assert exc.details.get("host") == "localhost"
        assert exc.details.get("port") == 5432


class TestServiceDiscoveryError:
    """测试服务发现异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import ServiceDiscoveryError
        
        exc = ServiceDiscoveryError("Service not found")
        assert exc.error_code == "SERVICE_DISCOVERY_ERROR"
        assert exc.details.get("service_name") is None
    
    def test_init_with_service_name(self):
        """测试使用服务名称初始化"""
        from src.infrastructure.utils.core.exceptions import ServiceDiscoveryError
        
        exc = ServiceDiscoveryError("Service not found", service_name="user-service")
        assert exc.details.get("service_name") == "user-service"


class TestHealthCheckError:
    """测试健康检查异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import HealthCheckError
        
        exc = HealthCheckError("Health check failed")
        assert exc.error_code == "HEALTH_CHECK_ERROR"
        assert exc.details.get("service_name") is None
        assert exc.details.get("check_type") is None
    
    def test_init_with_service_name(self):
        """测试使用服务名称初始化"""
        from src.infrastructure.utils.core.exceptions import HealthCheckError
        
        exc = HealthCheckError("Health check failed", service_name="database")
        assert exc.details.get("service_name") == "database"
    
    def test_init_with_check_type(self):
        """测试使用检查类型初始化"""
        from src.infrastructure.utils.core.exceptions import HealthCheckError
        
        exc = HealthCheckError("Health check failed", check_type="liveness")
        assert exc.details.get("check_type") == "liveness"


class TestEventBusError:
    """测试事件总线异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import EventBusError
        
        exc = EventBusError("Event publish failed")
        assert exc.error_code == "EVENT_BUS_ERROR"
        assert exc.details.get("event_type") is None
        assert exc.details.get("subscriber") is None
    
    def test_init_with_event_type(self):
        """测试使用事件类型初始化"""
        from src.infrastructure.utils.core.exceptions import EventBusError
        
        exc = EventBusError("Event publish failed", event_type="user.created")
        assert exc.details.get("event_type") == "user.created"
    
    def test_init_with_subscriber(self):
        """测试使用订阅者初始化"""
        from src.infrastructure.utils.core.exceptions import EventBusError
        
        exc = EventBusError("Event publish failed", subscriber="notification-service")
        assert exc.details.get("subscriber") == "notification-service"


class TestMonitoringError:
    """测试监控异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import MonitoringError
        
        exc = MonitoringError("Metric collection failed")
        assert exc.error_code == "MONITORING_ERROR"
        assert exc.details.get("metric_name") is None
    
    def test_init_with_metric_name(self):
        """测试使用指标名称初始化"""
        from src.infrastructure.utils.core.exceptions import MonitoringError
        
        exc = MonitoringError("Metric collection failed", metric_name="cpu.usage")
        assert exc.details.get("metric_name") == "cpu.usage"


class TestLoggingError:
    """测试日志异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import LoggingError
        
        exc = LoggingError("Log write failed")
        assert exc.error_code == "LOGGING_ERROR"
        assert exc.details.get("logger_name") is None
    
    def test_init_with_logger_name(self):
        """测试使用日志器名称初始化"""
        from src.infrastructure.utils.core.exceptions import LoggingError
        
        exc = LoggingError("Log write failed", logger_name="app.logger")
        assert exc.details.get("logger_name") == "app.logger"


class TestDataLoaderError:
    """测试数据加载器异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import DataLoaderError
        
        exc = DataLoaderError("Data load failed")
        assert exc.error_code == "DATA_LOADER_ERROR"
        assert exc.details.get("loader_name") is None
        assert exc.details.get("data_source") is None
    
    def test_init_with_loader_name(self):
        """测试使用加载器名称初始化"""
        from src.infrastructure.utils.core.exceptions import DataLoaderError
        
        exc = DataLoaderError("Data load failed", loader_name="csv_loader")
        assert exc.details.get("loader_name") == "csv_loader"
    
    def test_init_with_data_source(self):
        """测试使用数据源初始化"""
        from src.infrastructure.utils.core.exceptions import DataLoaderError
        
        exc = DataLoaderError("Data load failed", data_source="/data/file.csv")
        assert exc.details.get("data_source") == "/data/file.csv"


class TestCacheError:
    """测试缓存异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import CacheError
        
        exc = CacheError("Cache operation failed")
        assert exc.error_code == "CACHE_ERROR"
        assert exc.details.get("cache_key") is None
        assert exc.details.get("operation") is None
    
    def test_init_with_cache_key(self):
        """测试使用缓存键初始化"""
        from src.infrastructure.utils.core.exceptions import CacheError
        
        exc = CacheError("Cache operation failed", cache_key="user:123")
        assert exc.details.get("cache_key") == "user:123"
    
    def test_init_with_operation(self):
        """测试使用操作初始化"""
        from src.infrastructure.utils.core.exceptions import CacheError
        
        exc = CacheError("Cache operation failed", operation="get")
        assert exc.details.get("operation") == "get"


class TestSecurityError:
    """测试安全异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import SecurityError
        
        exc = SecurityError("Security violation")
        assert exc.error_code == "SECURITY_ERROR"
        assert exc.details.get("security_context") is None
    
    def test_init_with_security_context(self):
        """测试使用安全上下文初始化"""
        from src.infrastructure.utils.core.exceptions import SecurityError
        
        exc = SecurityError("Security violation", security_context="api.v1.users")
        assert exc.details.get("security_context") == "api.v1.users"


class TestResourceLimitError:
    """测试资源限制异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import ResourceLimitError
        
        exc = ResourceLimitError("Resource limit exceeded")
        assert exc.error_code == "RESOURCE_LIMIT_ERROR"
        assert exc.details.get("resource_type") is None
        assert exc.details.get("current_usage") is None
        assert exc.details.get("limit") is None
    
    def test_init_with_resource_type(self):
        """测试使用资源类型初始化"""
        from src.infrastructure.utils.core.exceptions import ResourceLimitError
        
        exc = ResourceLimitError("Resource limit exceeded", resource_type="memory")
        assert exc.details.get("resource_type") == "memory"
    
    def test_init_with_usage_and_limit(self):
        """测试使用使用量和限制初始化"""
        from src.infrastructure.utils.core.exceptions import ResourceLimitError
        
        exc = ResourceLimitError("Resource limit exceeded", current_usage=150.0, limit=100.0)
        assert exc.details.get("current_usage") == 150.0
        assert exc.details.get("limit") == 100.0


class TestValidationError:
    """测试验证异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import ValidationError
        
        exc = ValidationError("Validation failed")
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.details.get("field") is None
        assert exc.details.get("value") is None
    
    def test_init_with_field(self):
        """测试使用字段初始化"""
        from src.infrastructure.utils.core.exceptions import ValidationError
        
        exc = ValidationError("Validation failed", field="email")
        assert exc.details.get("field") == "email"
    
    def test_init_with_value(self):
        """测试使用值初始化"""
        from src.infrastructure.utils.core.exceptions import ValidationError
        
        exc = ValidationError("Validation failed", value="invalid@")
        assert exc.details.get("value") == "invalid@"


class TestTimeoutError:
    """测试超时异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import TimeoutError
        
        exc = TimeoutError("Operation timed out")
        assert exc.error_code == "TIMEOUT_ERROR"
        assert exc.details.get("timeout_seconds") is None
    
    def test_init_with_timeout_seconds(self):
        """测试使用超时秒数初始化"""
        from src.infrastructure.utils.core.exceptions import TimeoutError
        
        exc = TimeoutError("Operation timed out", timeout_seconds=30.0)
        assert exc.details.get("timeout_seconds") == 30.0


class TestDataVersionError:
    """测试数据版本异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import DataVersionError
        
        exc = DataVersionError("Version mismatch")
        assert exc.error_code == "DATA_VERSION_ERROR"
        assert exc.details.get("version") is None
        assert exc.details.get("operation") is None
    
    def test_init_with_version(self):
        """测试使用版本初始化"""
        from src.infrastructure.utils.core.exceptions import DataVersionError
        
        exc = DataVersionError("Version mismatch", version="1.0.0")
        assert exc.details.get("version") == "1.0.0"
    
    def test_init_with_operation(self):
        """测试使用操作初始化"""
        from src.infrastructure.utils.core.exceptions import DataVersionError
        
        exc = DataVersionError("Version mismatch", operation="update")
        assert exc.details.get("operation") == "update"


class TestDependencyError:
    """测试依赖关系异常"""
    
    def test_init_with_message(self):
        """测试使用消息初始化"""
        from src.infrastructure.utils.core.exceptions import DependencyError
        
        exc = DependencyError("Dependency not found")
        assert exc.error_code == "DEPENDENCY_ERROR"
        assert exc.details.get("dependency_name") is None
    
    def test_init_with_dependency_name(self):
        """测试使用依赖名称初始化"""
        from src.infrastructure.utils.core.exceptions import DependencyError
        
        exc = DependencyError("Dependency not found", dependency_name="redis")
        assert exc.details.get("dependency_name") == "redis"


class TestExceptionLogging:
    """测试异常日志记录"""
    
    @patch('src.infrastructure.utils.core.exceptions.logging.error')
    def test_exception_logs_error(self, mock_logging_error):
        """测试异常自动记录日志"""
        from src.infrastructure.utils.core.exceptions import InfrastructureError
        
        exc = InfrastructureError("Test error", error_code="TEST_ERROR")
        
        # 验证日志被调用
        mock_logging_error.assert_called_once()
        call_args = mock_logging_error.call_args
        assert "基础设施异常" in call_args[0][0]
        assert "TEST_ERROR" in call_args[0][0]
        
        # 验证extra参数
        assert call_args[1]["extra"]["error_code"] == "TEST_ERROR"
        assert call_args[1]["extra"]["error_type"] == "InfrastructureError"
    
    @patch('src.infrastructure.utils.core.exceptions.logging.error')
    def test_exception_logging_failure_does_not_crash(self, mock_logging_error):
        """测试日志记录失败不会导致程序崩溃"""
        from src.infrastructure.utils.core.exceptions import InfrastructureError
        
        # 模拟日志记录失败
        mock_logging_error.side_effect = Exception("Logging failed")
        
        # 应该不会抛出异常
        exc = InfrastructureError("Test error")
        assert exc.message == "Test error"
