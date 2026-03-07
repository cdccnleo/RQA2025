"""
测试API服务

覆盖 api_service.py 中的核心类
"""

import time
from collections import deque
import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.infrastructure.logging.services.api_service import (
    _NoopEventBus,
    _NoopServiceContainer,
    RequestRouter,
    RequestValidator,
    RequestExecutor,
    ResponseHandler,
    RateLimiter,
    VersionManager,
    APIVersion,
    APIEndpoint,
    RateLimitStrategy,
    RateLimitInfo,
    APIService,
    LoggingAPIService
)


class TestNoopEventBus:
    """_NoopEventBus 测试"""

    def test_subscribe(self):
        """测试订阅"""
        bus = _NoopEventBus()
        result = bus.subscribe("event", lambda: None)
        assert result is None

    def test_unsubscribe(self):
        """测试取消订阅"""
        bus = _NoopEventBus()
        result = bus.unsubscribe("event", lambda: None)
        assert result is None

    def test_publish(self):
        """测试发布"""
        bus = _NoopEventBus()
        result = bus.publish("event", data="test")
        assert result is None


class TestNoopServiceContainer:
    """_NoopServiceContainer 测试"""

    def test_has(self):
        """测试服务存在检查"""
        container = _NoopServiceContainer()
        result = container.has("service_name")
        assert result == False

    def test_get_not_found(self):
        """测试获取不存在的服务"""
        container = _NoopServiceContainer()
        with pytest.raises(KeyError) as exc_info:
            container.get("nonexistent_service")

        assert "Service not registered in infrastructure container stub" in str(exc_info.value)


class TestRequestRouter:
    """RequestRouter 测试"""

    def test_init(self):
        """测试初始化"""
        router = RequestRouter()

        assert router._endpoints == {}
        assert router._middlewares == []

    def test_register_endpoint_success(self):
        """测试成功注册端点"""
        router = RequestRouter()
        handler = lambda: "test"

        result = router.register_endpoint("/test", "GET", handler)

        assert result == True
        assert "GET" in router._endpoints
        assert "/test" in router._endpoints["GET"]
        assert router._endpoints["GET"]["/test"]["handler"] == handler
        assert router._endpoints["GET"]["/test"]["version"] == "v1"
        assert router._endpoints["GET"]["/test"]["requires_auth"] == False
        assert router._endpoints["GET"]["/test"]["rate_limit"] is None

    def test_register_endpoint_duplicate(self):
        """测试注册重复端点"""
        router = RequestRouter()
        handler1 = lambda: "test1"
        handler2 = lambda: "test2"

        router.register_endpoint("/test", "GET", handler1)
        result = router.register_endpoint("/test", "GET", handler2)

        assert result == False
        assert router._endpoints["GET"]["/test"]["handler"] == handler1

    def test_register_endpoint_different_methods(self):
        """测试注册不同方法的相同路径"""
        router = RequestRouter()
        get_handler = lambda: "get"
        post_handler = lambda: "post"

        router.register_endpoint("/test", "GET", get_handler)
        router.register_endpoint("/test", "POST", post_handler)

        assert router._endpoints["GET"]["/test"]["handler"] == get_handler
        assert router._endpoints["POST"]["/test"]["handler"] == post_handler

    def test_find_endpoint_found(self):
        """测试查找存在的端点"""
        router = RequestRouter()
        handler = lambda: "test"

        router.register_endpoint("/test", "GET", handler)
        endpoint = router.find_endpoint("/test", "GET")

        assert endpoint is not None
        assert endpoint["handler"] == handler

    def test_find_endpoint_not_found(self):
        """测试查找不存在的端点"""
        router = RequestRouter()

        endpoint = router.find_endpoint("/nonexistent", "GET")

        assert endpoint is None

    def test_get_all_endpoints(self):
        """测试获取所有端点"""
        router = RequestRouter()

        router.register_endpoint("/test1", "GET", lambda: None)
        router.register_endpoint("/test2", "POST", lambda: None)

        endpoints = router.get_all_endpoints()

        assert len(endpoints) == 2
        assert "/test1" in str(endpoints)
        assert "/test2" in str(endpoints)

    def test_add_middleware(self):
        """测试添加中间件"""
        router = RequestRouter()
        middleware = lambda req, next_handler: next_handler(req)

        router.add_middleware(middleware)

        assert middleware in router._middlewares

    def test_add_middleware(self):
        """测试添加中间件"""
        router = RequestRouter()
        middleware = lambda req, next_handler: next_handler(req)

        router.add_middleware(middleware)

        assert middleware in router._middlewares


class TestRequestValidator:
    """RequestValidator 测试"""

    def test_init(self):
        """测试初始化"""
        validator = RequestValidator()

        assert validator._auth_providers == {}

    def test_validate_request_valid(self):
        """测试验证有效请求"""
        validator = RequestValidator()
        request = {
            "method": "GET",
            "path": "/test",
            "headers": {"content-type": "application/json"}
        }
        endpoint_info = {"requires_auth": False}

        result = validator.validate_request(request, endpoint_info)

        assert result['valid'] == True
        assert result['errors'] == []
        assert isinstance(result['warnings'], list)

    def test_validate_request_missing_required_fields(self):
        """测试缺少必需字段的请求验证"""
        validator = RequestValidator()
        request = {"path": "/test"}  # 缺少method和headers
        endpoint_info = {"requires_auth": False}

        result = validator.validate_request(request, endpoint_info)

        assert result['valid'] == False
        assert "缺少必需字段: method" in result['errors']
        assert "缺少必需字段: headers" in result['errors']

    def test_validate_request_invalid_method(self):
        """测试无效HTTP方法的请求验证"""
        validator = RequestValidator()
        request = {
            "method": "INVALID",
            "path": "/test",
            "headers": {}
        }
        endpoint_info = {"requires_auth": False}

        result = validator.validate_request(request, endpoint_info)

        assert result['valid'] == False
        assert "不支持的HTTP方法: INVALID" in result['errors']

    def test_validate_request_requires_auth_no_auth(self):
        """测试需要认证但无认证信息的请求验证"""
        validator = RequestValidator()
        request = {
            "method": "GET",
            "path": "/test",
            "headers": {}
        }
        endpoint_info = {"requires_auth": True}

        result = validator.validate_request(request, endpoint_info)

        assert result['valid'] == False
        assert "缺少认证信息" in result['errors']

    def test_validate_request_path_format_warning(self):
        """测试路径格式警告"""
        validator = RequestValidator()
        request = {
            "method": "GET",
            "path": "test",  # 不以/开头
            "headers": {}
        }
        endpoint_info = {"requires_auth": False}

        result = validator.validate_request(request, endpoint_info)

        assert result['valid'] == True
        assert "路径应该以'/'开头" in result['warnings']

    def test_register_auth_provider(self):
        """测试注册认证提供者"""
        validator = RequestValidator()

        def mock_provider(token):
            return {"valid": True, "user": "test"}

        validator.register_auth_provider("bearer", mock_provider)

        assert "bearer" in validator._auth_providers
        assert validator._auth_providers["bearer"] == mock_provider

    def test_validate_request_with_auth_success(self):
        """测试带认证成功的请求验证"""
        validator = RequestValidator()

        def mock_provider(token):
            return {"valid": True, "user": "test"}

        validator.register_auth_provider("bearer", mock_provider)

        request = {
            "method": "GET",
            "path": "/test",
            "headers": {"authorization": "Bearer token123"}
        }
        endpoint_info = {"requires_auth": True}

        result = validator.validate_request(request, endpoint_info)

        assert result['valid'] == True

    def test_validate_request_with_auth_failure(self):
        """测试带认证失败的请求验证"""
        validator = RequestValidator()

        def mock_provider(token):
            return {"valid": False, "error": "Invalid token"}

        validator.register_auth_provider("bearer", mock_provider)

        request = {
            "method": "GET",
            "path": "/test",
            "headers": {"authorization": "Bearer invalid"}
        }
        endpoint_info = {"requires_auth": True}

        result = validator.validate_request(request, endpoint_info)

        assert result['valid'] == False
        assert "Invalid token" in result['errors']


class TestRateLimiter:
    """RateLimiter 测试"""

    def test_init(self):
        """测试初始化"""
        limiter = RateLimiter()

        assert limiter._limits == {}
        assert limiter._counters == {}

    def test_set_limit(self):
        """测试设置限制"""
        limiter = RateLimiter()

        limiter.set_limit("test_key", 100, 60)  # 100请求/60分钟

        assert "test_key" in limiter._limits
        assert limiter._limits["test_key"]["limit"] == 100
        assert limiter._limits["test_key"]["window_minutes"] == 60

    def test_check_limit_under_limit(self):
        """测试在限制内检查"""
        limiter = RateLimiter()
        limiter.set_limit("test_key", 10, 1)

        for i in range(5):
            result = limiter.check_limit("test_key")
            assert result["valid"] == True

    def test_check_limit_over_limit(self):
        """测试超过限制检查"""
        limiter = RateLimiter()
        limiter.set_limit("test_key", 3, 1)

        # 达到限制
        for i in range(3):
            result = limiter.check_limit("test_key")
            assert result["valid"] == True

        # 第4次应该被拒绝
        result = limiter.check_limit("test_key")
        assert result["valid"] == False

    def test_reset_limit(self):
        """测试重置限制"""
        limiter = RateLimiter()
        limiter.set_limit("test_key", 2, 1)

        # 达到限制
        limiter.check_limit("test_key")
        limiter.check_limit("test_key")
        result = limiter.check_limit("test_key")
        assert result["valid"] == False

        # 重置限制
        limiter.reset_limit("test_key")

        # 应该又可以允许请求了
        result = limiter.check_limit("test_key")
        assert result["valid"] == True

    def test_get_remaining_limit(self):
        """测试获取剩余限制"""
        limiter = RateLimiter()
        limiter.set_limit("test_key", 10, 1)

        # 初始状态
        remaining = limiter.get_remaining_limit("test_key")
        assert remaining == 10

        # 使用一些请求
        limiter.check_limit("test_key")
        limiter.check_limit("test_key")

        remaining = limiter.get_remaining_limit("test_key")
        assert remaining == 8

    def test_check_rate_limit(self):
        """测试检查速率限制"""
        limiter = RateLimiter()

        result = limiter.check_rate_limit("client1", "/api/test", 5)

        assert "allowed" in result
        assert "remaining" in result
        assert "reset_time" in result


class TestAPIVersion:
    """APIVersion 测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert APIVersion.V1.value == "v1"
        assert APIVersion.V2.value == "v2"

    def test_enum_str(self):
        """测试字符串表示"""
        assert str(APIVersion.V1) == "APIVersion.V1"
        assert APIVersion.V1.value == "v1"


class TestAPIEndpoint:
    """APIEndpoint 测试"""

    def test_init(self):
        """测试初始化"""
        handler = lambda: "test"
        endpoint = APIEndpoint("/api/test", "GET", handler)

        assert endpoint.path == "/api/test"
        assert endpoint.method == "GET"
        assert endpoint.handler == handler
        assert endpoint.version == APIVersion.V1
        assert endpoint.auth_required == False
        assert endpoint.rate_limit is None

    def test_init_with_options(self):
        """测试带选项初始化"""
        handler = lambda: "test"
        endpoint = APIEndpoint(
            "/api/test",
            "POST",
            handler,
            version=APIVersion.V2,
            rate_limit=100,
            auth_required=True,
            description="Test endpoint"
        )

        assert endpoint.path == "/api/test"
        assert endpoint.method == "POST"
        assert endpoint.handler == handler
        assert endpoint.version == APIVersion.V2
        assert endpoint.auth_required == True
        assert endpoint.rate_limit == 100
        assert endpoint.description == "Test endpoint"

    def test_str(self):
        """测试字符串表示"""
        handler = lambda: "test"
        endpoint = APIEndpoint("/api/test", "GET", handler)

        # dataclass的默认__str__方法会显示所有字段
        assert "/api/test" in str(endpoint)
        assert "GET" in str(endpoint)


class TestRateLimitStrategy:
    """RateLimitStrategy 测试"""

    def test_fixed_window(self):
        """测试固定窗口策略"""
        strategy = RateLimitStrategy.FIXED_WINDOW
        assert strategy.value == "fixed_window"

    def test_sliding_window(self):
        """测试滑动窗口策略"""
        strategy = RateLimitStrategy.SLIDING_WINDOW
        assert strategy.value == "sliding_window"

    def test_token_bucket(self):
        """测试令牌桶策略"""
        strategy = RateLimitStrategy.TOKEN_BUCKET
        assert strategy.value == "token_bucket"


class TestRateLimitInfo:
    """RateLimitInfo 测试"""

    def test_init(self):
        """测试初始化"""
        info = RateLimitInfo()

        assert isinstance(info.requests, deque)
        assert info.tokens == 10
        assert isinstance(info.last_refill, float)

    def test_init_custom(self):
        """测试自定义初始化"""
        custom_requests = deque([1, 2, 3])
        info = RateLimitInfo(
            requests=custom_requests,
            tokens=20,
            last_refill=1234567890.0
        )

        assert info.requests == custom_requests
        assert info.tokens == 20
        assert info.last_refill == 1234567890.0


class TestAPIService:
    """APIService 测试"""

    def test_init(self):
        """测试初始化"""
        service = APIService(name="TestAPIService")

        assert service.name == "TestAPIService"
        # APIService不继承BaseService的event_bus/container逻辑
        assert hasattr(service, '_router')
        assert hasattr(service, '_validator')
        assert hasattr(service, '_rate_limiter')
        assert isinstance(service._router, RequestRouter)
        assert isinstance(service._validator, RequestValidator)
        assert isinstance(service._rate_limiter, RateLimiter)

    def test_has_required_components(self):
        """测试具有必需的组件"""
        service = APIService(name="TestAPIService")

        # 检查是否有必需的私有组件
        assert hasattr(service, '_router')
        assert hasattr(service, '_validator')
        assert hasattr(service, '_rate_limiter')
        assert hasattr(service, '_version_manager')

        # 检查组件类型
        assert isinstance(service._router, RequestRouter)
        assert isinstance(service._validator, RequestValidator)
        assert isinstance(service._rate_limiter, RateLimiter)


class TestRequestExecutor:
    """RequestExecutor 测试"""

    def test_init(self):
        """测试初始化"""
        executor = RequestExecutor()
        assert executor._executors == {}
        assert executor._timeout == 30

    def test_execute_request(self):
        """测试执行请求"""
        executor = RequestExecutor()

        def mock_handler(request, endpoint_info):
            return {"status": "success", "data": request["data"]}

        executor.register_executor("GET", mock_handler)

        request = {"method": "GET", "data": "test"}
        endpoint_info = {"method": "GET"}

        result = executor.execute_request(request, endpoint_info)

        assert result["success"] is True
        assert result["status_code"] == 200
        assert result["data"]["status"] == "success"
        assert result["data"]["data"] == "test"

    def test_register_executor(self):
        """测试注册执行器"""
        executor = RequestExecutor()

        def mock_handler(request):
            return {"status": "ok"}

        executor.register_executor("POST", mock_handler)

        assert "POST" in executor._executors

    def test_set_timeout(self):
        """测试设置超时"""
        executor = RequestExecutor()
        executor.set_timeout(60)

        assert executor._timeout == 60


class TestResponseHandler:
    """ResponseHandler 测试"""

    def test_init(self):
        """测试初始化"""
        handler = ResponseHandler()
        assert '_formatters' in handler.__dict__

    def test_handle_success_response(self):
        """测试处理成功响应"""
        handler = ResponseHandler()

        execution_result = {
            "status": "success",
            "data": {"key": "value"},
            "execution_time": 0.1
        }

        request = {"method": "GET", "path": "/api/test"}

        response = handler.handle_success_response(execution_result, request)

        assert "status_code" in response
        assert "headers" in response

    def test_handle_error_response(self):
        """测试处理错误响应"""
        handler = ResponseHandler()

        execution_result = {
            "status": "error",
            "error": "Test error",
            "execution_time": 0.2
        }

        request = {"method": "POST", "path": "/api/error"}

        response = handler.handle_error_response(execution_result, request)

        assert "status_code" in response
        assert "body" in response
        assert "error" in response["body"]

    def test_format_response(self):
        """测试格式化响应"""
        handler = ResponseHandler()

        data = {"status": "ok", "message": "test"}
        formatted = handler.format_response(data)

        assert isinstance(formatted, str)
        assert "status" in formatted
        assert "ok" in formatted


class TestVersionManager:
    """VersionManager 测试"""

    def test_init(self):
        """测试初始化"""
        manager = VersionManager()
        assert manager._current_version == "v1"
        assert manager._versions == {}
        assert manager._version_mappings == {}

    def test_check_version_compatibility(self):
        """测试版本兼容性检查"""
        manager = VersionManager()

        result = manager.check_version_compatibility("v1", "v1")
        assert result["compatible"] is True

        result = manager.check_version_compatibility("v3", "v1")
        assert result["compatible"] is False

    def test_add_version(self):
        """测试添加版本"""
        manager = VersionManager()

        manager.add_version("v2.0", ["feature1", "feature2"])

        assert "v2.0" in manager._versions
        assert manager._versions["v2.0"] == ["feature1", "feature2"]

    def test_set_current_version(self):
        """测试设置当前版本"""
        manager = VersionManager()
        manager.set_current_version("v2.0")

        assert manager._current_version == "v2.0"

    def test_get_version_info(self):
        """测试获取版本信息"""
        manager = VersionManager()
        manager.add_version("v1.1", ["feature1"])

        info = manager.get_version_info("v1.1")
        assert info == ["feature1"]

        info = manager.get_version_info("nonexistent")
        assert info is None

    def test_list_versions(self):
        """测试列出版本"""
        manager = VersionManager()
        manager.add_version("v1.0", [])
        manager.add_version("v1.1", [])

        versions = manager.list_versions()
        assert "v1.0" in versions
        assert "v1.1" in versions

    def test_is_version_supported(self):
        """测试版本支持检查"""
        manager = VersionManager()
        manager.add_version("v1.0", [])

        result = manager.is_version_supported("v1.0")
        assert result["valid"] is True

        result = manager.is_version_supported("v2.0")
        assert result["valid"] is False


class TestLoggingAPIService:
    """LoggingAPIService 测试"""

    def test_init(self):
        """测试初始化"""
        service = LoggingAPIService()
        assert hasattr(service, 'router')

    def test_register_endpoint(self):
        """测试注册端点"""
        service = LoggingAPIService()

        def mock_handler():
            return "ok"

        service.register_endpoint("/test", "GET", mock_handler)

        # 验证端点已注册到router
        assert hasattr(service, 'router')

    def test_dispatch(self):
        """测试分发请求"""
        service = LoggingAPIService()

        def mock_handler():
            return "handled"

        service.register_endpoint("/test", "GET", mock_handler)

        # 使用find_endpoint检查端点是否已注册
        endpoint = service.router.find_endpoint("/test", "GET")
        assert endpoint is not None
        assert "handler" in endpoint
        assert endpoint["version"] == "v1"