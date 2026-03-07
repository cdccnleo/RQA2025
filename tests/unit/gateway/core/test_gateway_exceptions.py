"""
网关服务层核心异常测试
测试网关相关的异常类和错误处理机制
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
from src.gateway.core.exceptions import (
    GatewayException,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    RoutingError,
    UpstreamError,
    RequestValidationError,
    CircuitBreakerError,
    TimeoutError,
    ResourceExhaustionError,
    ConfigurationError,
    WebSocketError
)


class TestGatewayExceptions:
    """网关异常测试"""

    def test_gateway_exception_basic(self):
        """测试基础网关异常"""
        message = "Gateway operation failed"
        error_code = 500
        status_code = 502

        exception = GatewayException(message, error_code, status_code)

        assert str(exception) == message
        assert exception.error_code == error_code
        assert exception.status_code == status_code
        assert exception.message == message

    def test_authentication_error(self):
        """测试认证异常"""
        message = "Invalid credentials"
        user_id = "user_001"

        exception = AuthenticationError(message, user_id)

        assert "认证失败" in str(exception)
        assert user_id in str(exception)
        assert exception.user_id == user_id
        assert exception.status_code == 401

    def test_authorization_error(self):
        """测试授权异常"""
        message = "Insufficient permissions"
        resource = "/admin/users"

        exception = AuthorizationError(message, resource)

        assert "授权失败" in str(exception)
        assert resource in str(exception)
        assert exception.resource == resource
        assert exception.status_code == 403

    def test_rate_limit_error(self):
        """测试速率限制异常"""
        message = "Too many requests"
        client_ip = "192.168.1.100"

        exception = RateLimitError(message, client_ip)

        assert "速率限制" in str(exception)
        assert client_ip in str(exception)
        assert exception.client_ip == client_ip
        assert exception.status_code == 429

    def test_routing_error(self):
        """测试路由异常"""
        message = "Service not found"
        route_path = "/api/v1/users"

        exception = RoutingError(message, route_path)

        assert "路由失败" in str(exception)
        assert route_path in str(exception)
        assert exception.route_path == route_path
        assert exception.status_code == 404

    def test_upstream_error(self):
        """测试上游异常"""
        message = "Backend service error"
        service_name = "api-service"

        exception = UpstreamError(message, service_name)

        assert "上游服务错误" in str(exception)
        assert service_name in str(exception)
        assert exception.service_name == service_name
        assert exception.status_code == 502

    def test_request_validation_error(self):
        """测试请求验证异常"""
        message = "Invalid request format"
        field = "email"

        exception = RequestValidationError(message, field)

        assert "请求验证失败" in str(exception)
        assert field in str(exception)
        assert exception.field == field
        assert exception.status_code == 400

    def test_timeout_error(self):
        """测试超时异常"""
        message = "Gateway timeout"
        timeout_seconds = 30

        exception = TimeoutError(message, timeout_seconds)

        assert "请求超时" in str(exception)
        assert str(timeout_seconds) in str(exception)
        assert exception.timeout_seconds == timeout_seconds
        assert exception.status_code == 504

    def test_resource_exhaustion_error(self):
        """测试资源耗尽异常"""
        message = "Too many connections"
        resource_type = "connections"

        exception = ResourceExhaustionError(message, resource_type)

        assert "资源耗尽" in str(exception)
        assert resource_type in str(exception)
        assert exception.resource_type == resource_type
        assert exception.status_code == 503

    def test_configuration_error(self):
        """测试配置异常"""
        message = "Missing configuration"
        config_key = "api.endpoint"

        exception = ConfigurationError(message, config_key)

        assert "配置错误" in str(exception)
        assert config_key in str(exception)
        assert exception.config_key == config_key
        assert exception.status_code == 500

    def test_circuit_breaker_error(self):
        """测试熔断器异常"""
        message = "Circuit breaker triggered"
        service_name = "api_circuit"

        exception = CircuitBreakerError(message, service_name)

        assert "服务熔断" in str(exception)
        assert service_name in str(exception)
        assert exception.service_name == service_name
        assert exception.status_code == 503

    def test_configuration_error(self):
        """测试配置异常"""
        message = "Missing gateway config"
        config_key = "api.endpoint"

        exception = ConfigurationError(message, config_key)

        assert "配置错误" in str(exception)
        assert config_key in str(exception)
        assert exception.config_key == config_key
        assert exception.status_code == 500

    def test_websocket_error(self):
        """测试WebSocket异常"""
        message = "WebSocket connection failed"
        connection_id = "ws_12345"

        exception = WebSocketError(message, connection_id)

        assert "WebSocket错误" in str(exception)
        assert connection_id in str(exception)
        assert exception.connection_id == connection_id
        assert exception.status_code == 1011

    def test_exception_inheritance(self):
        """测试异常继承关系"""
        base_exception = GatewayException("test")
        assert isinstance(base_exception, Exception)

        auth_error = AuthenticationError("test", "user")
        assert isinstance(auth_error, GatewayException)

        rate_limit_error = RateLimitError("test", "ip")
        assert isinstance(rate_limit_error, GatewayException)

        assert issubclass(AuthenticationError, GatewayException)
        assert issubclass(RateLimitError, GatewayException)

    def test_exception_with_default_values(self):
        """测试异常默认值"""
        # 测试没有额外参数的异常
        exception = GatewayException("test")
        assert exception.error_code == -1
        assert exception.status_code == 500

        # 测试有默认参数的异常
        auth_error = AuthenticationError("test")
        assert auth_error.user_id is None
        assert auth_error.status_code == 401

        routing_error = RoutingError("test")
        assert routing_error.route_path is None
        assert routing_error.status_code == 404

    def test_exception_status_codes(self):
        """测试异常状态码正确性"""
        # HTTP状态码验证
        auth_error = AuthenticationError("test", "user")
        assert auth_error.status_code == 401

        authz_error = AuthorizationError("test", "resource")
        assert authz_error.status_code == 403

        rate_error = RateLimitError("test", "ip")
        assert rate_error.status_code == 429

        routing_error = RoutingError("test", "path")
        assert routing_error.status_code == 404

        validation_error = RequestValidationError("test", "field")
        assert validation_error.status_code == 400

        timeout_error = TimeoutError("test", 30)
        assert timeout_error.status_code == 504
