#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网关层异常测试

测试目标：提升exceptions.py的覆盖率
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

# 动态导入网关异常模块
try:
    exceptions_module = importlib.import_module('src.gateway.core.exceptions')
    GatewayException = getattr(exceptions_module, 'GatewayException', None)
    AuthenticationError = getattr(exceptions_module, 'AuthenticationError', None)
    AuthorizationError = getattr(exceptions_module, 'AuthorizationError', None)
    RateLimitError = getattr(exceptions_module, 'RateLimitError', None)
    RoutingError = getattr(exceptions_module, 'RoutingError', None)
    UpstreamError = getattr(exceptions_module, 'UpstreamError', None)
    RequestValidationError = getattr(exceptions_module, 'RequestValidationError', None)
    CircuitBreakerError = getattr(exceptions_module, 'CircuitBreakerError', None)
    TimeoutError = getattr(exceptions_module, 'TimeoutError', None)
    
    ResourceExhaustionError = getattr(exceptions_module, 'ResourceExhaustionError', None)
    ConfigurationError = getattr(exceptions_module, 'ConfigurationError', None)
    WebSocketError = getattr(exceptions_module, 'WebSocketError', None)
    handle_gateway_exception = getattr(exceptions_module, 'handle_gateway_exception', None)
    validate_request_data = getattr(exceptions_module, 'validate_request_data', None)
    validate_request_size = getattr(exceptions_module, 'validate_request_size', None)
    check_rate_limit = getattr(exceptions_module, 'check_rate_limit', None)
    get_error_response = getattr(exceptions_module, 'get_error_response', None)
    
    if GatewayException is None:
        pytest.skip("网关异常模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("网关异常模块导入失败", allow_module_level=True)


class TestGatewayExceptions:
    """测试网关异常类"""
    
    def test_gateway_exception(self):
        """测试基础网关异常"""
        exc = GatewayException("测试错误", error_code=1001, status_code=400)
        assert str(exc) == "测试错误"
        assert exc.error_code == 1001
        assert exc.message == "测试错误"
        assert exc.status_code == 400
    
    def test_authentication_error(self):
        """测试认证错误"""
        exc = AuthenticationError("认证失败", user_id="user123")
        assert "user123" in str(exc)
        assert exc.user_id == "user123"
        assert exc.status_code == 401
    
    def test_authorization_error(self):
        """测试授权错误"""
        exc = AuthorizationError("授权失败", resource="api/data")
        assert "api/data" in str(exc)
        assert exc.resource == "api/data"
        assert exc.status_code == 403
    
    def test_rate_limit_error(self):
        """测试速率限制错误"""
        exc = RateLimitError("超过速率限制", client_ip="192.168.1.1")
        assert "192.168.1.1" in str(exc)
        assert exc.client_ip == "192.168.1.1"
        assert exc.status_code == 429
    
    def test_routing_error(self):
        """测试路由错误"""
        exc = RoutingError("路由失败", route_path="/api/data")
        assert "/api/data" in str(exc)
        assert exc.route_path == "/api/data"
        assert exc.status_code == 404
    
    def test_upstream_error(self):
        """测试上游服务错误"""
        exc = UpstreamError("上游服务错误", service_name="service1")
        assert "service1" in str(exc)
        assert exc.service_name == "service1"
        assert exc.status_code == 502
    
    def test_request_validation_error(self):
        """测试请求验证错误"""
        exc = RequestValidationError("请求验证失败", field="username")
        assert "username" in str(exc)
        assert exc.field == "username"
        assert exc.status_code == 400
    
    def test_circuit_breaker_error(self):
        """测试熔断器错误"""
        exc = CircuitBreakerError("服务熔断", service_name="service1")
        assert "service1" in str(exc)
        assert exc.service_name == "service1"
        assert exc.status_code == 503
    
    def test_timeout_error(self):
        """测试超时错误"""
        exc = TimeoutError("请求超时", timeout_seconds=30)
        assert "30" in str(exc)
        assert exc.timeout_seconds == 30
        assert exc.status_code == 504
    
    def test_resource_exhaustion_error(self):
        """测试资源耗尽错误"""
        exc = ResourceExhaustionError("资源耗尽", resource_type="memory")
        assert "memory" in str(exc)
        assert exc.resource_type == "memory"
        assert exc.status_code == 503
    
    def test_configuration_error(self):
        """测试配置错误"""
        exc = ConfigurationError("配置错误", config_key="api_key")
        assert "api_key" in str(exc)
        assert exc.config_key == "api_key"
        assert exc.status_code == 500
    
    def test_websocket_error(self):
        """测试WebSocket错误"""
        exc = WebSocketError("WebSocket错误", connection_id="conn123")
        assert "conn123" in str(exc)
        assert exc.connection_id == "conn123"
        assert exc.status_code == 1011


class TestExceptionDecorators:
    """测试异常装饰器"""
    
    def test_handle_gateway_exception_success(self):
        """测试异常处理装饰器 - 成功"""
        @handle_gateway_exception
        def test_func():
            return "success"
        
        assert test_func() == "success"
    
    def test_handle_gateway_exception_gateway_error(self):
        """测试异常处理装饰器 - 网关异常"""
        @handle_gateway_exception
        def test_func():
            raise GatewayException("网关错误")
        
        with pytest.raises(GatewayException) as exc_info:
            test_func()
        assert "网关错误" in str(exc_info.value)
    
    def test_handle_gateway_exception_general_error(self):
        """测试异常处理装饰器 - 通用异常"""
        @handle_gateway_exception
        def test_func():
            raise ValueError("通用错误")
        
        with pytest.raises(GatewayException) as exc_info:
            test_func()
        assert "意外网关错误" in str(exc_info.value)


class TestValidationFunctions:
    """测试验证函数"""
    
    def test_validate_request_data_success(self):
        """测试请求数据验证 - 成功"""
        data = {"username": "test", "password": "123456"}
        validate_request_data(data, ["username", "password"])
    
    def test_validate_request_data_empty(self):
        """测试请求数据验证 - 空数据"""
        with pytest.raises(RequestValidationError) as exc_info:
            validate_request_data({}, ["username"])
        assert "不能为空" in str(exc_info.value)
    
    def test_validate_request_data_missing_fields(self):
        """测试请求数据验证 - 缺少字段"""
        data = {"username": "test"}
        with pytest.raises(RequestValidationError) as exc_info:
            validate_request_data(data, ["username", "password"])
        assert "缺少必需字段" in str(exc_info.value)
    
    def test_validate_request_size_success(self):
        """测试请求大小验证 - 成功"""
        validate_request_size(1000, 10000)
    
    def test_validate_request_size_exceed(self):
        """测试请求大小验证 - 超过限制"""
        with pytest.raises(RequestValidationError) as exc_info:
            validate_request_size(20000, 10000)
        assert "超过限制" in str(exc_info.value)
    
    def test_check_rate_limit_success(self):
        """测试速率限制检查 - 成功"""
        result = check_rate_limit(50, 100, 60, "client1")
        assert result is True
    
    def test_check_rate_limit_exceed(self):
        """测试速率限制检查 - 超过限制"""
        with pytest.raises(RateLimitError) as exc_info:
            check_rate_limit(100, 100, 60, "client1")
        assert "超过速率限制" in str(exc_info.value)
    
    def test_get_error_response(self):
        """测试获取错误响应"""
        exc = GatewayException("测试错误", error_code=1001, status_code=400)
        response = get_error_response(exc)
        
        assert "error" in response
        assert response["error"]["code"] == 1001
        assert response["error"]["message"] == "测试错误"
        assert response["status_code"] == 400
        assert "timestamp" in response

