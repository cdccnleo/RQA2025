"""
适配器服务层核心异常测试
测试适配器相关的异常类和错误处理机制
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
from src.adapters.core.exceptions import (
    AdapterException,
    ConnectionError,
    DataSourceError,
    DataTransformationError,
    ValidationError,
    ConfigurationError,
    ResourceExhaustionError,
    TimeoutError,
    AuthenticationError,
    RateLimitError
)


class TestAdapterExceptions:
    """适配器异常测试"""

    def test_adapter_exception_basic(self):
        """测试基础适配器异常"""
        message = "Adapter operation failed"
        error_code = 500

        exception = AdapterException(message, error_code)

        assert str(exception) == message
        assert exception.error_code == error_code
        assert exception.message == message

    def test_connection_error(self):
        """测试连接异常"""
        message = "Connection refused"
        adapter_name = "database_adapter"

        exception = ConnectionError(message, adapter_name)

        assert "连接失败" in str(exception)
        assert adapter_name in str(exception)
        assert exception.adapter_name == adapter_name

    def test_data_source_error(self):
        """测试数据源异常"""
        message = "Data source unavailable"
        source_name = "market_data_api"

        exception = DataSourceError(message, source_name)

        assert "数据源错误" in str(exception)
        assert source_name in str(exception)
        assert exception.source_name == source_name

    def test_data_transformation_error(self):
        """测试数据转换异常"""
        message = "Invalid data format"
        field_name = "price"

        exception = DataTransformationError(message, field_name)

        assert "数据转换失败" in str(exception)
        assert field_name in str(exception)
        assert exception.field_name == field_name

    def test_validation_error(self):
        """测试验证异常"""
        message = "Invalid input data"
        validation_rule = "format_check"

        exception = ValidationError(message, validation_rule)

        assert "数据验证失败" in str(exception)
        assert validation_rule in str(exception)
        assert exception.validation_rule == validation_rule

    def test_configuration_error(self):
        """测试配置异常"""
        message = "Missing configuration"
        config_key = "adapter.endpoint"

        exception = ConfigurationError(message, config_key)

        assert "配置错误" in str(exception)
        assert config_key in str(exception)
        assert exception.config_key == config_key

    def test_resource_exhaustion_error(self):
        """测试资源耗尽异常"""
        message = "Resource exhausted"
        resource_type = "memory"

        exception = ResourceExhaustionError(message, resource_type)

        assert "资源耗尽" in str(exception)
        assert resource_type in str(exception)
        assert exception.resource_type == resource_type

    def test_timeout_error(self):
        """测试超时异常"""
        message = "Operation timed out"
        timeout_seconds = 30

        exception = TimeoutError(message, timeout_seconds)

        assert "操作超时" in str(exception)
        assert str(timeout_seconds) in str(exception)
        assert exception.timeout_seconds == timeout_seconds

    def test_authentication_error(self):
        """测试认证异常"""
        message = "Invalid credentials"
        auth_method = "password"

        exception = AuthenticationError(message, auth_method)

        assert "认证失败" in str(exception)
        assert auth_method in str(exception)
        assert exception.auth_method == auth_method

    def test_rate_limit_error(self):
        """测试速率限制异常"""
        message = "Too many requests"
        limit_type = "per_minute"

        exception = RateLimitError(message, limit_type)

        assert "速率限制" in str(exception)
        assert limit_type in str(exception)
        assert exception.limit_type == limit_type

    def test_exception_inheritance(self):
        """测试异常继承关系"""
        base_exception = AdapterException("test")
        assert isinstance(base_exception, Exception)

        conn_error = ConnectionError("test", "adapter")
        assert isinstance(conn_error, AdapterException)

        data_error = DataSourceError("test", "source")
        assert isinstance(data_error, AdapterException)

        assert issubclass(ConnectionError, AdapterException)
        assert issubclass(DataSourceError, AdapterException)

    def test_exception_with_default_values(self):
        """测试异常默认值"""
        # 测试没有额外参数的异常
        exception = AdapterException("test")
        assert exception.error_code == -1

        # 测试有默认参数的异常
        conn_error = ConnectionError("test")
        assert conn_error.adapter_name is None

        data_error = DataSourceError("test")
        assert data_error.source_name is None
