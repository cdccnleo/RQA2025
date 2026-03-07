#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网关类型定义测试

测试目标：提升gateway_types.py的覆盖率
"""

import sys
import importlib
from pathlib import Path
import pytest
from datetime import datetime

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入网关类型模块
try:
    gateway_types_module = importlib.import_module('src.gateway.api.gateway_types')
    HttpMethod = getattr(gateway_types_module, 'HttpMethod', None)
    ServiceStatus = getattr(gateway_types_module, 'ServiceStatus', None)
    RateLimitType = getattr(gateway_types_module, 'RateLimitType', None)
    ServiceEndpoint = getattr(gateway_types_module, 'ServiceEndpoint', None)
    RateLimitRule = getattr(gateway_types_module, 'RateLimitRule', None)
    RouteRule = getattr(gateway_types_module, 'RouteRule', None)
    ApiRequest = getattr(gateway_types_module, 'ApiRequest', None)
    ApiResponse = getattr(gateway_types_module, 'ApiResponse', None)
    
    if HttpMethod is None:
        pytest.skip("网关类型模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("网关类型模块导入失败", allow_module_level=True)


class TestHttpMethod:
    """测试HTTP方法枚举"""
    
    def test_http_method_values(self):
        """测试HTTP方法值"""
        assert HttpMethod.GET.value == "GET"
        assert HttpMethod.POST.value == "POST"
        assert HttpMethod.PUT.value == "PUT"
        assert HttpMethod.DELETE.value == "DELETE"
        assert HttpMethod.PATCH.value == "PATCH"
        assert HttpMethod.OPTIONS.value == "OPTIONS"
        assert HttpMethod.HEAD.value == "HEAD"


class TestServiceStatus:
    """测试服务状态枚举"""
    
    def test_service_status_values(self):
        """测试服务状态值"""
        assert ServiceStatus.HEALTHY.value == "healthy"
        assert ServiceStatus.UNHEALTHY.value == "unhealthy"
        assert ServiceStatus.MAINTENANCE.value == "maintenance"
        assert ServiceStatus.DOWN.value == "down"


class TestRateLimitType:
    """测试限流类型枚举"""
    
    def test_rate_limit_type_values(self):
        """测试限流类型值"""
        assert RateLimitType.IP.value == "ip"
        assert RateLimitType.USER.value == "user"
        assert RateLimitType.GLOBAL.value == "global"
        assert RateLimitType.API_KEY.value == "api_key"


class TestServiceEndpoint:
    """测试服务端点数据类"""
    
    def test_service_endpoint_basic(self):
        """测试基本服务端点"""
        endpoint = ServiceEndpoint(
            service_name="test_service",
            path="/api/data",
            method=HttpMethod.GET,
            upstream_url="http://localhost:8000"
        )
        assert endpoint.service_name == "test_service"
        assert endpoint.path == "/api/data"
        assert endpoint.method == HttpMethod.GET
        assert endpoint.upstream_url == "http://localhost:8000"
        assert endpoint.timeout == 30
        assert endpoint.retries == 3
        assert endpoint.weight == 1
    
    def test_service_endpoint_with_health_check(self):
        """测试带健康检查的服务端点"""
        endpoint = ServiceEndpoint(
            service_name="test_service",
            health_check_url="http://localhost:8000/health"
        )
        assert endpoint.health_check_url == "http://localhost:8000/health"


class TestRateLimitRule:
    """测试限流规则数据类"""
    
    def test_rate_limit_rule_basic(self):
        """测试基本限流规则"""
        rule = RateLimitRule(
            limit_type=RateLimitType.IP,
            limit=100,
            window=60
        )
        assert rule.limit_type == RateLimitType.IP
        assert rule.limit == 100
        assert rule.window == 60
    
    def test_rate_limit_rule_with_key(self):
        """测试带键的限流规则"""
        rule = RateLimitRule(
            limit_type=RateLimitType.USER,
            limit=50,
            window=30,
            key="user123"
        )
        assert rule.limit_type == RateLimitType.USER
        assert rule.limit == 50
        assert rule.window == 30
        assert rule.key == "user123"


class TestRouteRule:
    """测试路由规则数据类"""
    
    def test_route_rule_basic(self):
        """测试基本路由规则"""
        rule = RouteRule(
            path="/api/data",
            method=HttpMethod.GET,
            service_name="service1"
        )
        assert rule.path == "/api/data"
        assert rule.method == HttpMethod.GET
        assert rule.service_name == "service1"
        assert rule.strip_prefix is True
        assert rule.auth_required is True
        assert rule.cors_enabled is True
        assert rule.cache_enabled is False
    
    def test_route_rule_with_options(self):
        """测试带选项的路由规则"""
        rate_limits = [RateLimitRule(RateLimitType.IP, 100, 60)]
        rule = RouteRule(
            path="/api/data",
            method=HttpMethod.POST,
            service_name="service1",
            strip_prefix=False,
            auth_required=False,
            cors_enabled=False,
            cache_enabled=True,
            cache_ttl=600,
            rate_limits=rate_limits
        )
        assert rule.strip_prefix is False
        assert rule.auth_required is False
        assert rule.cors_enabled is False
        assert rule.cache_enabled is True
        assert rule.cache_ttl == 600
        assert len(rule.rate_limits) == 1


class TestApiRequest:
    """测试API请求数据类"""
    
    def test_api_request_basic(self):
        """测试基本API请求"""
        request = ApiRequest(
            id="req123",
            method=HttpMethod.POST,
            path="/api/data",
            headers={"Content-Type": "application/json"},
            query_params={"key": "value"}
        )
        assert request.id == "req123"
        assert request.method == HttpMethod.POST
        assert request.path == "/api/data"
        assert request.headers == {"Content-Type": "application/json"}
        assert request.query_params == {"key": "value"}
        assert request.body is None
    
    def test_api_request_with_body(self):
        """测试带请求体的API请求"""
        request = ApiRequest(
            id="req456",
            method=HttpMethod.POST,
            path="/api/data",
            headers={"Content-Type": "application/json"},
            query_params={},
            body=b'{"key": "value"}',
            client_ip="192.168.1.1",
            user_id="user123"
        )
        assert request.body == b'{"key": "value"}'
        assert request.client_ip == "192.168.1.1"
        assert request.user_id == "user123"


class TestApiResponse:
    """测试API响应数据类"""
    
    def test_api_response_basic(self):
        """测试基本API响应"""
        response = ApiResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b'{"result": "success"}',
            processing_time=0.1
        )
        assert response.status_code == 200
        assert response.headers == {"Content-Type": "application/json"}
        assert response.body == b'{"result": "success"}'
        assert response.processing_time == 0.1
        assert response.cached is False
    
    def test_api_response_with_options(self):
        """测试带选项的API响应"""
        response = ApiResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b'{"result": "success"}',
            processing_time=0.2,
            upstream_url="http://localhost:8000",
            cached=True
        )
        assert response.upstream_url == "http://localhost:8000"
        assert response.cached is True

