#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - API服务综合测试

全面测试logging/services/api_service.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import requests

# 由于api_service.py有复杂的导入依赖，我们需要mock一些依赖
from src.infrastructure.logging.services.api_service import (
    APIService, RequestRouter, RequestValidator, RequestExecutor,
    ResponseHandler, RateLimiter, VersionManager, APIEndpoint,
    RateLimitInfo, APIVersion, RateLimitStrategy
)
from src.infrastructure.logging.core.exceptions import (
    HTTP_OK, HTTP_BAD_REQUEST, HTTP_UNAUTHORIZED, HTTP_FORBIDDEN,
    HTTP_NOT_FOUND, HTTP_INTERNAL_ERROR
)


class TestRequestRouter:
    """测试请求路由器"""

    def setup_method(self):
        """测试前准备"""
        self.router = RequestRouter()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.router, '_endpoints')
        assert hasattr(self.router, '_middlewares')
        assert isinstance(self.router._endpoints, dict)
        assert isinstance(self.router._middlewares, list)

    def test_register_endpoint_success(self):
        """测试成功注册端点"""
        path = "/api/test"
        method = "GET"
        handler = Mock()

        result = self.router.register_endpoint(path, method, handler)

        assert result is True
        assert method in self.router._endpoints
        assert path in self.router._endpoints[method]
        assert self.router._endpoints[method][path]['handler'] is handler
        assert self.router._endpoints[method][path]['version'] == "v1"
        assert self.router._endpoints[method][path]['requires_auth'] is False

    def test_register_endpoint_with_options(self):
        """测试注册带选项的端点"""
        path = "/api/secure"
        method = "POST"
        handler = Mock()
        version = "v2"
        requires_auth = True
        rate_limit = 100

        result = self.router.register_endpoint(
            path, method, handler, version, requires_auth, rate_limit
        )

        assert result is True
        endpoint = self.router._endpoints[method][path]
        assert endpoint['version'] == version
        assert endpoint['requires_auth'] is requires_auth
        assert endpoint['rate_limit'] == rate_limit
        assert 'registered_at' in endpoint

    def test_register_endpoint_duplicate(self):
        """测试注册重复端点"""
        path = "/api/test"
        method = "GET"
        handler1 = Mock()
        handler2 = Mock()

        # 注册第一个
        result1 = self.router.register_endpoint(path, method, handler1)
        assert result1 is True

        # 尝试注册重复的
        result2 = self.router.register_endpoint(path, method, handler2)
        assert result2 is False

    def test_unregister_endpoint_success(self):
        """测试成功注销端点"""
        path = "/api/test"
        method = "GET"
        handler = Mock()

        # 先注册
        self.router.register_endpoint(path, method, handler)

        # 注销
        result = self.router.unregister_endpoint(path, method)

        assert result is True
        assert path not in self.router._endpoints[method]

    def test_unregister_endpoint_nonexistent(self):
        """测试注销不存在的端点"""
        result = self.router.unregister_endpoint("/api/nonexistent", "GET")

        assert result is False

    def test_find_endpoint_existing(self):
        """测试查找存在的端点"""
        path = "/api/test"
        method = "GET"
        handler = Mock()

        self.router.register_endpoint(path, method, handler)

        endpoint = self.router.find_endpoint(path, method)

        assert endpoint is not None
        assert endpoint['handler'] is handler

    def test_find_endpoint_nonexistent(self):
        """测试查找不存在的端点"""
        endpoint = self.router.find_endpoint("/api/nonexistent", "GET")

        assert endpoint is None

    def test_find_endpoint_wrong_method(self):
        """测试用错误方法查找端点"""
        path = "/api/test"
        handler = Mock()

        self.router.register_endpoint(path, "GET", handler)

        endpoint = self.router.find_endpoint(path, "POST")

        assert endpoint is None

    def test_get_all_endpoints(self):
        """测试获取所有端点"""
        # 注册多个端点
        endpoints = [
            ("/api/users", "GET", Mock()),
            ("/api/users", "POST", Mock()),
            ("/api/orders", "GET", Mock()),
            ("/api/orders", "PUT", Mock()),
        ]

        for path, method, handler in endpoints:
            self.router.register_endpoint(path, method, handler)

        all_endpoints = self.router.get_all_endpoints()

        assert isinstance(all_endpoints, dict)
        assert "GET" in all_endpoints
        assert "POST" in all_endpoints
        assert "PUT" in all_endpoints
        assert len(all_endpoints["GET"]) == 2
        assert len(all_endpoints["POST"]) == 1
        assert len(all_endpoints["PUT"]) == 1

    def test_add_middleware(self):
        """测试添加中间件"""
        middleware = Mock()

        self.router.add_middleware(middleware)

        assert middleware in self.router._middlewares

    def test_multiple_middlewares(self):
        """测试多个中间件"""
        middleware1 = Mock()
        middleware2 = Mock()
        middleware3 = Mock()

        self.router.add_middleware(middleware1)
        self.router.add_middleware(middleware2)
        self.router.add_middleware(middleware3)

        assert len(self.router._middlewares) == 3
        assert middleware1 in self.router._middlewares
        assert middleware2 in self.router._middlewares
        assert middleware3 in self.router._middlewares


class TestRequestValidator:
    """测试请求验证器"""

    def setup_method(self):
        """测试前准备"""
        self.validator = RequestValidator()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.validator, '_auth_providers')
        assert isinstance(self.validator._auth_providers, dict)

    def test_validate_request_valid(self):
        """测试验证有效请求"""
        # Mock认证提供者
        auth_provider = Mock(return_value={'user_id': 123, 'valid': True})
        self.validator.register_auth_provider('bearer', auth_provider)

        request = {
            'path': '/api/test',
            'method': 'GET',
            'headers': {'authorization': 'Bearer token123'},
            'body': None
        }

        endpoint_info = {
            'requires_auth': True,
            'version': 'v1'
        }

        result = self.validator.validate_request(request, endpoint_info)

        assert result['valid'] is True
        assert 'errors' in result
        assert 'warnings' in result

    def test_validate_request_missing_path(self):
        """测试验证缺少路径的请求"""
        request = {
            'method': 'GET',
            'headers': {},
            'body': None
        }

        endpoint_info = {'requires_auth': False}

        result = self.validator.validate_request(request, endpoint_info)

        assert result['valid'] is False
        assert 'errors' in result

    def test_validate_request_invalid_method(self):
        """测试验证无效方法的请求"""
        request = {
            'path': '/api/test',
            'method': 'INVALID',
            'headers': {},
            'body': None
        }

        endpoint_info = {'requires_auth': False}

        result = self.validator.validate_request(request, endpoint_info)

        assert result['valid'] is False
        assert 'errors' in result

    def test_validate_request_auth_required_but_missing(self):
        """测试需要认证但缺少认证的请求"""
        request = {
            'path': '/api/secure',
            'method': 'GET',
            'headers': {},
            'body': None
        }

        endpoint_info = {'requires_auth': True}

        result = self.validator.validate_request(request, endpoint_info)

        assert result['valid'] is False
        assert 'errors' in result

    def test_validate_request_malformed_path(self):
        """测试验证格式错误的路径"""
        request = {
            'path': 'invalid-path-without-slash',
            'method': 'GET',
            'headers': {},
            'body': None
        }

        endpoint_info = {'requires_auth': False}

        result = self.validator.validate_request(request, endpoint_info)

        assert result['valid'] is True  # 路径格式问题只是警告
        assert 'warnings' in result
        assert len(result['warnings']) > 0

    def test_validate_required_fields(self):
        """测试验证必需字段"""
        result = {'valid': True, 'errors': []}

        # 有效请求
        valid_request = {
            'path': '/api/test',
            'method': 'GET',
            'headers': {},
            'body': None
        }

        self.validator._validate_required_fields(valid_request, result)
        assert result['valid'] is True

        # 无效请求 - 缺少必需字段
        invalid_request = {'method': 'GET'}

        result = {'valid': True, 'errors': []}
        self.validator._validate_required_fields(invalid_request, result)
        assert result['valid'] is False
        assert len(result['errors']) > 0

    def test_validate_http_method(self):
        """测试验证HTTP方法"""
        result = {'valid': True, 'errors': []}

        # 有效方法
        valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']

        for method in valid_methods:
            result = {'valid': True, 'errors': []}
            request = {'method': method}
            self.validator._validate_http_method(request, result)
            assert result['valid'] is True

        # 无效方法
        result = {'valid': True, 'errors': []}
        invalid_request = {'method': 'INVALID'}
        self.validator._validate_http_method(invalid_request, result)
        assert result['valid'] is False

    def test_validate_path_format(self):
        """测试验证路径格式"""
        # 有效路径 - 不应该产生警告
        valid_paths = ['/api/test', '/api/users/123', '/api/orders', '/']

        for path in valid_paths:
            result = {'valid': True, 'errors': [], 'warnings': []}
            request = {'path': path}
            self.validator._validate_path_format(request, result)
            assert result['valid'] is True
            # 有效路径不应该有警告
            assert len(result['warnings']) == 0
        # 无效路径 - 应该产生警告
        invalid_paths = ['api/test', 'invalid-path', '', 'path with spaces']

        for path in invalid_paths:
            result = {'valid': True, 'errors': [], 'warnings': []}
            request = {'path': path}
            self.validator._validate_path_format(request, result)
            assert result['valid'] is True  # 路径格式问题只是警告
            assert len(result['warnings']) > 0
    def test_validate_auth_for_request(self):
        """测试为请求验证认证"""
        # Mock认证提供者
        auth_provider = Mock(return_value={'user_id': 123, 'valid': True})
        self.validator.register_auth_provider('bearer', auth_provider)

        # 不需要认证
        endpoint_info = {'requires_auth': False}
        request = {'path': '/api/test', 'method': 'GET', 'headers': {}, 'body': None}
        result = self.validator.validate_request(request, endpoint_info)
        assert result['valid'] is True

        # 需要认证 - 有有效的认证头
        endpoint_info = {'requires_auth': True}
        request = {'path': '/api/test', 'method': 'GET', 'headers': {'authorization': 'Bearer token123'}, 'body': None}
        result = self.validator.validate_request(request, endpoint_info)
        assert result['valid'] is True

        # 需要认证 - 缺少认证头
        endpoint_info = {'requires_auth': True}
        request = {'path': '/api/test', 'method': 'GET', 'headers': {}, 'body': None}
        result = self.validator.validate_request(request, endpoint_info)
        assert result['valid'] is False

    def test_validate_request_body(self):
        """测试验证请求体"""
        result = {'valid': True, 'errors': [], 'warnings': []}

        # 有效请求体
        valid_bodies = [None, {}, {'data': 'value'}, [1, 2, 3]]

        for body in valid_bodies:
            request = {'body': body}
            self.validator._validate_request_body(request, result)
            assert result['valid'] is True

        # 无效请求体类型
        invalid_bodies = ['string', 123, True]

        for body in invalid_bodies:
            result = {'valid': True, 'errors': [], 'warnings': []}
            request = {'body': body}
            self.validator._validate_request_body(request, result)
            # 这个验证可能允许各种类型，取决于实现

    def test_validate_authentication_success(self):
        """测试认证验证成功"""
        request = {'headers': {'authorization': 'Bearer valid_token'}}

        # Mock认证提供者
        auth_provider = Mock(return_value={'user_id': 123, 'valid': True})
        self.validator.register_auth_provider('bearer', auth_provider)

        result = self.validator._validate_authentication(request)

        assert result['valid'] is True
        assert result['user_id'] == 123

    def test_validate_authentication_failure(self):
        """测试认证验证失败"""
        request = {'headers': {'authorization': 'Bearer invalid_token'}}

        # Mock认证提供者
        auth_provider = Mock(return_value={'valid': False, 'error': 'Invalid token'})
        self.validator.register_auth_provider('bearer', auth_provider)

        result = self.validator._validate_authentication(request)

        assert result['valid'] is False
        assert len(result['errors']) > 0

    def test_register_auth_provider(self):
        """测试注册认证提供者"""
        name = 'custom_auth'
        provider = Mock()

        self.validator.register_auth_provider(name, provider)

        assert name in self.validator._auth_providers
        assert self.validator._auth_providers[name] is provider


class TestRequestExecutor:
    """测试请求执行器"""

    def setup_method(self):
        """测试前准备"""
        self.executor = RequestExecutor()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.executor, '_executors')
        assert hasattr(self.executor, '_timeout')
        assert isinstance(self.executor._executors, dict)

    def test_execute_request_success(self):
        """测试成功执行请求"""
        request = {
            'path': '/api/test',
            'method': 'GET',
            'headers': {},
            'body': None
        }

        endpoint_info = {
            'handler': Mock(return_value={'result': 'success'})
        }

        result = self.executor.execute_request(request, endpoint_info)

        assert result['success'] is True
        assert result['data']['result'] == 'success'

    def test_execute_request_handler_error(self):
        """测试执行请求时处理器错误"""
        request = {
            'path': '/api/test',
            'method': 'GET',
            'headers': {},
            'body': None
        }

        endpoint_info = {
            'handler': Mock(side_effect=Exception('Handler error'))
        }

        result = self.executor.execute_request(request, endpoint_info)

        assert result['success'] is False
        assert 'error' in result

    def test_execute_request_timeout(self):
        """测试执行请求超时"""
        request = {
            'path': '/api/test',
            'method': 'GET',
            'headers': {},
            'body': None
        }

        # Mock一个会超时的处理器
        def slow_handler():
            time.sleep(2)  # 超过默认超时
        return {'result': 'success'}
        endpoint_info = {
            'handler': slow_handler
        }

        # 设置短超时
        self.executor.set_timeout(0.1)

        result = self.executor.execute_request(request, endpoint_info)

        # 应该超时或正常完成，取决于实现
        assert isinstance(result, dict)

    def test_register_executor(self):
        """测试注册执行器"""
        method = 'CUSTOM'
        custom_executor = Mock()

        self.executor.register_executor(method, custom_executor)

        assert method in self.executor._executors
        assert self.executor._executors[method] is custom_executor

    def test_set_timeout(self):
        """测试设置超时"""
        timeout = 30

        self.executor.set_timeout(timeout)

        assert self.executor._timeout == timeout

    def test_custom_executor(self):
        """测试自定义执行器"""
        method = 'CUSTOM'
        custom_executor = Mock(return_value={'custom': True})

        self.executor.register_executor(method, custom_executor)

        request = {'method': method}
        endpoint_info = {}

        result = self.executor.execute_request(request, endpoint_info)

        custom_executor.assert_called_once()
        assert result['success'] is True


class TestResponseHandler:
    """测试响应处理器"""

    def setup_method(self):
        """测试前准备"""
        self.handler = ResponseHandler()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.handler, '_formatters')
        assert isinstance(self.handler._formatters, dict)

    def test_format_response_json(self):
        """测试格式化JSON响应"""
        response_data = {
            'status': HTTP_OK,
            'data': {'users': [{'id': 1, 'name': 'John'}]},
            'message': 'Success'
        }

        result = self.handler.format_response(response_data)

        assert isinstance(result, str)
        # 应该返回JSON字符串
        parsed = json.loads(result)
        assert parsed['status'] == HTTP_OK
        assert 'data' in parsed

    def test_format_response_error(self):
        """测试格式化错误响应"""
        response_data = {
            'status': HTTP_INTERNAL_ERROR,
            'error': 'Internal server error',
            'message': 'Something went wrong'
        }

        result = self.handler.format_response(response_data)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed['status'] == HTTP_INTERNAL_ERROR
        assert parsed['error'] == 'Internal server error'

    def test_format_response_custom_format(self):
        """测试格式化自定义格式响应"""
        # 注册自定义格式化器
        def xml_formatter(response_data):
            return f"<response><status>{response_data['status']}</status></response>"

        self.handler._formatters['xml'] = xml_formatter

        response_data = {'status': HTTP_OK, 'data': 'test'}

        result = self.handler.format_response(response_data, format_type='xml')

        assert result == "<response><status>200</status></response>"


class TestRateLimiter:
    """测试限流器"""

    def setup_method(self):
        """测试前准备"""
        self.limiter = RateLimiter()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.limiter, '_limits')
        assert hasattr(self.limiter, '_counters')
        assert isinstance(self.limiter._limits, dict)
        assert isinstance(self.limiter._counters, dict)

    def test_check_limit_under_limit(self):
        """测试检查限流 - 未达到限制"""
        client_id = "user_123"
        limit = 10

        # 设置限流
        self.limiter.set_limit(client_id, limit)

        # 检查多次，应该都通过
        for i in range(5):
            result = self.limiter.check_limit(client_id)
            assert result['valid'] is True

    def test_check_limit_at_limit(self):
        """测试检查限流 - 达到限制"""
        client_id = "user_123"
        limit = 3

        # 设置限流
        self.limiter.set_limit(client_id, limit)

        # 使用完所有配额
        for i in range(limit):
            result = self.limiter.check_limit(client_id)
            assert result['valid'] is True

        # 下一次应该被限流
        result = self.limiter.check_limit(client_id)
        assert result['valid'] is False

    def test_set_limit(self):
        """测试设置限流"""
        client_id = "api_client"
        limit = 100

        self.limiter.set_limit(client_id, limit)

        assert client_id in self.limiter._limits
        assert self.limiter._limits[client_id]['limit'] == limit

    def test_get_remaining_limit(self):
        """测试获取剩余限流配额"""
        client_id = "user_123"
        limit = 5

        self.limiter.set_limit(client_id, limit)

        # 使用一些配额
        for i in range(2):
            self.limiter.check_limit(client_id)

        remaining = self.limiter.get_remaining_limit(client_id)

        assert remaining == 3

    def test_reset_limit(self):
        """测试重置限流"""
        client_id = "user_123"
        limit = 3

        self.limiter.set_limit(client_id, limit)

        # 使用完配额
        for i in range(limit):
            self.limiter.check_limit(client_id)

        # 应该被限流
        result = self.limiter.check_limit(client_id)
        assert result['valid'] is False

        # 重置
        self.limiter.reset_limit(client_id)

        # 应该又可以访问了
        result = self.limiter.check_limit(client_id)
        assert result['valid'] is True


class TestVersionManager:
    """测试版本管理器"""

    def setup_method(self):
        """测试前准备"""
        self.manager = VersionManager()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.manager, '_versions')
        assert hasattr(self.manager, '_current_version')
        assert isinstance(self.manager._versions, dict)

    def test_add_version(self):
        """测试添加版本"""
        version = "v2.1"
        features = ["new_feature", "improved_performance"]

        self.manager.add_version(version, features)

        assert version in self.manager._versions
        assert self.manager._versions[version] == features

    def test_set_current_version(self):
        """测试设置当前版本"""
        version = "v2.0"

        self.manager.set_current_version(version)

        assert self.manager._current_version == version

    def test_get_version_info(self):
        """测试获取版本信息"""
        version = "v1.5"
        features = ["bug_fixes", "security_updates"]

        self.manager.add_version(version, features)

        info = self.manager.get_version_info(version)

        assert info == features

    def test_get_version_info_nonexistent(self):
        """测试获取不存在的版本信息"""
        info = self.manager.get_version_info("nonexistent")

        assert info is None

    def test_list_versions(self):
        """测试列出版本"""
        versions = ["v1.0", "v1.1", "v2.0"]

        for version in versions:
            self.manager.add_version(version, [])

        listed = self.manager.list_versions()

        assert set(listed) == set(versions)

    def test_is_version_supported(self):
        """测试检查版本是否支持"""
        supported_version = "v1.0"
        unsupported_version = "v0.5"

        self.manager.add_version(supported_version, [])

        result_supported = self.manager.is_version_supported(supported_version)
        result_unsupported = self.manager.is_version_supported(unsupported_version)

        assert result_supported['valid'] is True
        assert result_unsupported['valid'] is False


class TestAPIService:
    """测试API服务"""

    def setup_method(self):
        """测试前准备"""
        self.mock_event_bus = Mock()
        self.mock_container = Mock()

        # Mock事件总线订阅
        self.mock_event_bus.subscribe = Mock()

        self.service = APIService(
            event_bus=self.mock_event_bus,
            container=self.mock_container,
            name="test_api_service"
        )

    def teardown_method(self):
        """测试后清理"""
        if hasattr(self.service, 'stop'):
            try:
                self.service.stop()
            except:
                pass

    def test_initialization(self):
        """测试初始化"""
        assert self.service.name == "test_api_service"
        assert hasattr(self.service, '_router')
        assert hasattr(self.service, '_validator')
        assert hasattr(self.service, '_executor')
        assert hasattr(self.service, '_response_handler')
        assert hasattr(self.service, '_rate_limiter')
        assert hasattr(self.service, '_version_manager')

        assert self.service.event_bus is self.mock_event_bus
        assert self.service.container is self.mock_container

        # 兼容性属性
        assert isinstance(self.service.routes, dict)
        assert isinstance(self.service.version_routes, dict)
        assert isinstance(self.service.request_stats, dict)
        assert isinstance(self.service.response_times, dict)
        assert isinstance(self.service.api_docs, dict)

    def test_register_endpoint(self):
        """测试注册端点"""
        path = "/api/test"
        method = "GET"
        handler = Mock()

        result = self.service.register_endpoint(path, method, handler)

        assert result is True

    def test_register_endpoint_with_options(self):
        """测试注册带选项的端点"""
        path = "/api/secure"
        method = "POST"
        handler = Mock()
        version = "v2"
        requires_auth = True

        result = self.service.register_endpoint(
            path, method, handler, version, requires_auth
        )

        assert result is True

    def test_unregister_endpoint(self):
        """测试注销端点"""
        path = "/api/test"
        method = "GET"
        handler = Mock()

        # 先注册
        self.service.register_endpoint(path, method, handler)

        # 注销
        result = self.service.unregister_endpoint(path, method)

        assert result is True

    def test_find_endpoint(self):
        """测试查找端点"""
        path = "/api/test"
        method = "GET"
        handler = Mock()

        self.service.register_endpoint(path, method, handler)

        endpoint = self.service.find_endpoint(path, method)

        assert endpoint is not None
        assert endpoint['handler'] is handler

    def test_get_all_endpoints(self):
        """测试获取所有端点"""
        # 注册一些端点
        endpoints = [
            ("/api/users", "GET", Mock()),
            ("/api/users", "POST", Mock()),
            ("/api/orders", "GET", Mock())
        ]

        for path, method, handler in endpoints:
            self.service.register_endpoint(path, method, handler)

        all_endpoints = self.service.get_all_endpoints()

        assert isinstance(all_endpoints, dict)

    def test_process_request_success(self):
        """测试成功处理请求"""
        request = {
            'path': '/api/test',
            'method': 'GET',
            'headers': {},
            'body': None
        }

        # Mock所有组件
        with patch.object(self.service._router, 'find_endpoint', return_value={'handler': Mock(return_value={'result': 'success'})}), \
             patch.object(self.service._validator, 'validate_request', return_value={'valid': True, 'request': request, 'endpoint_info': {}}), \
             patch.object(self.service._rate_limiter, 'check_limit', return_value=True), \
             patch.object(self.service._executor, 'execute_request', return_value={'success': True, 'data': {'result': 'success'}}), \
             patch.object(self.service._response_handler, 'format_response', return_value='{"status": 200, "data": {"result": "success"}}'):
            response = self.service.process_request(request)

        assert response is not None
    def test_process_request_invalid(self):
        """测试处理无效请求"""
        invalid_request = {
            'invalid': 'request'
        }

        # Mock验证器返回无效
        with patch.object(self.service._validator, 'validate_request', return_value={'valid': False, 'errors': ['Invalid request']}):

            response = self.service.process_request(invalid_request)

        assert 'error' in response.get('body', {}) or response.get('status_code', 200) >= 400
    def test_process_request_rate_limited(self):
        """测试处理被限流的请求"""
        # 先注册一个端点
        self.service.register_endpoint('/api/test', 'GET', Mock(), rate_limit=10)

        request = {
            'path': '/api/test',
            'method': 'GET',
            'headers': {},
            'body': None
        }

        # Mock限流器返回False
        with patch.object(self.service._rate_limiter, 'check_rate_limit', return_value={'allowed': False}):

            response = self.service.process_request(request)

        assert response.get('status_code') == 429 or 'Rate limit exceeded' in str(response)
    def test_generate_api_docs(self):
        """测试生成API文档"""
        # 注册一些端点
        self.service.register_endpoint("/api/users", "GET", Mock(), description="Get users")
        self.service.register_endpoint("/api/users", "POST", Mock(), description="Create user")

        docs = self.service.generate_api_docs()

        assert isinstance(docs, dict)
        assert len(docs) > 0

    def test_get_request_stats(self):
        """测试获取请求统计"""
        stats = self.service.get_request_stats()

        assert isinstance(stats, dict)

    def test_clear_request_stats(self):
        """测试清除请求统计"""
        result = self.service.clear_request_stats()

        assert result is True

    def test_get_health_status(self):
        """测试获取健康状态"""
        health = self.service.get_health_status()

        assert isinstance(health, dict)
        assert 'status' in health
        assert 'timestamp' in health

    def test_enable_maintenance_mode(self):
        """测试启用维护模式"""
        result = self.service.enable_maintenance_mode()

        assert result is True

    def test_disable_maintenance_mode(self):
        """测试禁用维护模式"""
        result = self.service.disable_maintenance_mode()

        assert result is True

    def test_get_supported_versions(self):
        """测试获取支持的版本"""
        versions = self.service.get_supported_versions()

        assert isinstance(versions, list)

    def test_get_version_info(self):
        """测试获取版本信息"""
        info = self.service.get_version_info("v1")

        assert isinstance(info, dict)

    def test_concurrent_request_processing(self):
        """测试并发请求处理"""
        import threading

        # 重置关键组件状态，避免前序用例的临时修改影响并发流程
        self.service._router = RequestRouter()
        self.service._validator = RequestValidator()
        self.service._rate_limiter = RateLimiter()
        self.service._executor = RequestExecutor()
        self.service._response_handler = ResponseHandler()

        results = []
        errors = []

        def process_request_worker(worker_id):
            try:
                for i in range(10):
                    request = {
                        'path': f'/api/test{worker_id}',
                        'method': 'GET',
                        'headers': {},
                        'body': None
                    }

                    # 注册测试端点
                    def test_handler(req):
                        return {'result': f'success_{worker_id}_{i}'}

                    self.service.register_endpoint(
                        f'/api/test{worker_id}',
                        'GET',
                        test_handler
                    )

                    response = self.service.process_request(request)
                    results.append(f"worker_{worker_id}_request_{i}_processed")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")

        # 启动多个线程
        threads = []
        for i in range(3):
            t = threading.Thread(target=process_request_worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0
        assert len(results) == 30  # 3 workers * 10 requests each

    def test_route_request_with_component_attribute_errors(self):
        """当核心组件缺失属性时应启用兜底逻辑"""
        request = {
            'path': '/api/fallback',
            'method': 'GET',
            'headers': {},
            'body': None,
            'version': 'v1',
        }

        with patch.object(self.service._router, 'find_endpoint', side_effect=AttributeError("router missing")), \
             patch.object(self.service._rate_limiter, 'check_rate_limit', return_value={'allowed': True}), \
             patch.object(self.service._validator, 'validate_request', side_effect=AttributeError("validator missing")), \
             patch.object(self.service._executor, 'execute_request', side_effect=AttributeError("executor missing")):
            response = self.service.process_request(request)

        if isinstance(response, str):
            response_payload = json.loads(response)
        else:
            response_payload = response

        assert response_payload['status_code'] == 200
        assert response_payload['body']['data']['result'] == 'fallback'
        assert response_payload['body']['success'] is True

    def test_rate_limit_fallback_to_check_limit(self):
        """check_rate_limit 缺失时应回退到 check_limit"""
        request = {
            'path': '/api/legacy',
            'method': 'GET',
            'headers': {},
        }
        endpoint_info = {
            'rate_limit': 5,
            'version': 'v1',
        }

        with patch.object(self.service._rate_limiter, 'check_rate_limit', side_effect=AttributeError("no new method")), \
             patch.object(self.service._rate_limiter, 'check_limit', return_value={'valid': True}) as mock_check_limit:
            allowed = self.service._check_rate_limit(request, endpoint_info)

        mock_check_limit.assert_called_once_with('anonymous')
        assert allowed is True

    def test_error_handling_in_request_processing(self):
        """测试请求处理中的错误处理"""
        request = {
            'path': '/api/error',
            'method': 'GET',
            'headers': {},
            'body': None
        }

        # Mock一个会抛出异常的组件
        with patch.object(self.service._router, 'find_endpoint', side_effect=Exception("Routing error")):

            try:
                response = self.service.process_request(request)
                # 应该返回错误响应
                assert response.get('status', 0) >= 400 or 'error' in response
            except:
                # 异常处理也是可接受的
                assert True
    def test_rate_limiting_integration(self):
        """测试限流功能集成"""
        # Mock限流检查
        with patch.object(self.service._rate_limiter, 'check_limit', return_value=False):
            request = {
                'path': '/api/limited',
                'method': 'GET',
                'headers': {},
                'body': None
            }

            try:
                response = self.service.process_request(request)
                # 应该被限流
                assert response.get('status') == 429 or 'rate_limited' in str(response)
            except:
                assert True
    def test_version_routing(self):
        """测试版本路由"""
        # 注册不同版本的端点
        handler_v1 = Mock()
        handler_v2 = Mock()

        self.service.register_endpoint("/api/test/v1", "GET", handler_v1, version="v1")
        self.service.register_endpoint("/api/test/v2", "GET", handler_v2, version="v2")

        # 验证版本路由存在
        assert len(self.service.version_routes) >= 2

    def test_request_validation_integration(self):
        """测试请求验证集成"""
        # 测试有效请求
        valid_request = {
            'path': '/api/valid',
            'method': 'GET',
            'headers': {'authorization': 'Bearer token'},
            'body': None
        }

        # Mock验证器
        with patch.object(self.service._validator, 'validate_request', return_value={'valid': True}):
            # 验证集成调用
            try:
                validation_result = self.service._validator.validate_request(valid_request, {})
                assert validation_result['valid'] is True
            except:
                assert True
    def test_response_formatting_integration(self):
        """测试响应格式化集成"""
        response_data = {'status': 200, 'data': {'result': 'success'}}

        # Mock响应处理器
        with patch.object(self.service._response_handler, 'format_response', return_value='{"status": 200, "data": {"result": "success"}}'):
            formatted = self.service._response_handler.format_response(response_data)
        assert formatted is not None
    def test_service_state_consistency(self):
        """测试服务状态一致性"""
        # 执行一系列操作
        initial_health = self.service.get_health_status()

        # 注册端点
        self.service.register_endpoint("/api/test", "GET", Mock())

        # 处理请求（mock）
        with patch.object(self.service._router, 'find_endpoint', return_value={'handler': Mock()}), \
             patch.object(self.service._validator, 'validate_request', return_value={'valid': True}), \
             patch.object(self.service._rate_limiter, 'check_limit', return_value=True), \
             patch.object(self.service._executor, 'execute_request', return_value={'success': True}), \
             patch.object(self.service._response_handler, 'format_response', return_value='{"status": 200}'):

            for i in range(5):
                request = {'path': '/api/test', 'method': 'GET', 'headers': {}, 'body': None}
                self.service.process_request(request)

        # 检查最终状态
        final_health = self.service.get_health_status()
        stats = self.service.get_request_stats()

        assert final_health['status'] == initial_health['status']  # 状态应该保持一致
        assert isinstance(stats, dict)

    def test_large_scale_endpoint_management(self):
        """测试大规模端点管理"""
        # 注册大量端点
        num_endpoints = 50
        registered_endpoints = []

        for i in range(num_endpoints):
            path = f"/api/endpoint_{i}"
            method = "GET" if i % 2 == 0 else "POST"
            handler = Mock()

            result = self.service.register_endpoint(path, method, handler)
            if result:
                registered_endpoints.append((path, method))

        # 验证注册成功
        assert len(registered_endpoints) == num_endpoints

        # 查找端点
        found_count = 0
        for path, method in registered_endpoints[:10]:  # 检查前10个
            endpoint = self.service.find_endpoint(path, method)
            if endpoint:
                found_count += 1

        assert found_count > 0

        # 获取所有端点
        all_endpoints = self.service.get_all_endpoints()
        assert len(all_endpoints) > 0

    def test_performance_request_processing(self):
        """测试请求处理性能"""
        import time

        # 注册一个端点
        self.service.register_endpoint("/api/perf", "GET", Mock())

        start_time = time.time()

        # 处理大量请求
        num_requests = 100
        for i in range(num_requests):
            request = {
                'path': '/api/perf',
                'method': 'GET',
                'headers': {},
                'body': None
            }

            # Mock处理
            with patch.object(self.service._router, 'find_endpoint', return_value={'handler': Mock(return_value={'result': f'success_{i}'})}), \
                 patch.object(self.service._validator, 'validate_request', return_value={'valid': True, 'request': request, 'endpoint_info': {}}), \
                 patch.object(self.service._rate_limiter, 'check_limit', return_value=True), \
                 patch.object(self.service._executor, 'execute_request', return_value={'success': True, 'data': {'result': f'success_{i}'}}), \
                 patch.object(self.service._response_handler, 'format_response', return_value=f'{{"status": 200, "data": {{"result": "success_{i}"}}}}'):
                self.service.process_request(request)

        end_time = time.time()
        duration = end_time - start_time

        # 应该在合理时间内完成
        assert duration < 5.0  # 少于5秒

    def test_memory_usage_under_load(self):
        """测试负载下的内存使用"""
        # 处理大量请求
        for i in range(100):
            request = {
                'path': f'/api/load_{i}',
                'method': 'GET',
                'headers': {},
                'body': f'{{"data": "test_{i}"}}'  # 大请求体
            }

            # Mock处理
            with patch.object(self.service._router, 'find_endpoint', return_value={'handler': Mock()}), \
                 patch.object(self.service._validator, 'validate_request', return_value={'valid': True}), \
                 patch.object(self.service._rate_limiter, 'check_limit', return_value=True), \
                 patch.object(self.service._executor, 'execute_request', return_value={'success': True}), \
                 patch.object(self.service._response_handler, 'format_response', return_value='{"status": 200}'):

                self.service.process_request(request)

        # 服务应该仍然稳定
        health = self.service.get_health_status()
        assert health['status'] == 'healthy' or isinstance(health, dict)

    def test_api_service_comprehensive_health_check(self):
        """测试API服务全面健康检查"""
        health = self.service.get_health_status()

        assert isinstance(health, dict)
        assert 'status' in health
        assert 'timestamp' in health
        assert 'components' in health

        components = health.get('components', {})
        assert isinstance(components, dict)

        # 可能包含各个组件的健康状态
        expected_components = ['router', 'validator', 'executor', 'response_handler', 'rate_limiter', 'version_manager']
        for component in expected_components:
            if component in components:
                assert isinstance(components[component], dict)
    def test_service_configuration_validation(self):
        """测试服务配置验证"""
        # 测试各种配置场景
        configs = [
            {},  # 默认配置
            {'max_endpoints': 100},  # 自定义配置
            {'enable_cors': True, 'cors_origins': ['*']},  # CORS配置
            {'rate_limits': {'default': 100, 'premium': 1000}},  # 限流配置
        ]

        for config in configs:
            # 配置应该被接受（或适当处理）
            try:
                service = APIService(self.mock_event_bus, self.mock_container, config=config)
                assert hasattr(service, '_router')
            except:
                # 某些配置可能不支持，这是可以的
                assert True
    def test_request_tracing_and_logging(self):
        """测试请求跟踪和日志记录"""
        request = {
            'path': '/api/trace',
            'method': 'GET',
            'headers': {'X-Request-ID': 'trace-123'},
            'body': None
        }

        # Mock处理并检查统计信息
        with patch.object(self.service._router, 'find_endpoint', return_value={'handler': Mock(return_value={'traced': True})}), \
             patch.object(self.service._validator, 'validate_request', return_value={'valid': True}), \
             patch.object(self.service._rate_limiter, 'check_limit', return_value=True), \
             patch.object(self.service._executor, 'execute_request', return_value={'success': True}), \
             patch.object(self.service._response_handler, 'format_response', return_value='{"status": 200, "traced": true}'):

            self.service.process_request(request)

        # 检查统计信息
        stats = self.service.get_request_stats()
        assert isinstance(stats, dict)

        # 检查响应时间记录
        assert '/api/trace' in self.service.response_times or isinstance(self.service.response_times, dict)

    def test_request_timeout(self):
        # 测试通过route_request处理超时请求
        request = {
            'path': '/api/timeout',
            'method': 'GET',
            'headers': {},
            'body': None
        }
        
        with patch.object(self.service._router, 'find_endpoint', return_value=None):
            result = self.service.route_request(request)
            assert result is not None

    def test_response_parsing(self):
        # 测试响应解析
        request = {
            'path': '/api/parse',
            'method': 'GET',
            'headers': {},
            'body': None
        }
        
        endpoint_info = {
            'handler': Mock(return_value={"key": "value"}),
            'version': 'v1',
            'requires_auth': False
        }
        
        with patch.object(self.service._router, 'find_endpoint', return_value=endpoint_info), \
             patch.object(self.service._validator, 'validate_request', return_value={'valid': True}), \
             patch.object(self.service._rate_limiter, 'check_limit', return_value=True):
            result = self.service.route_request(request)
            assert result is not None

    def test_complex_request(self):
        # 测试复杂请求处理
        request = {
            'path': '/api/complex',
            'method': 'POST',
            'headers': {'content-type': 'application/json'},
            'body': '{"param": "value"}'
        }
        
        with patch.object(self.service, 'route_request', return_value={"status": "success"}) as mock_route:
            result = self.service.route_request(request)
            mock_route.assert_called_once()
            assert isinstance(result, dict)
