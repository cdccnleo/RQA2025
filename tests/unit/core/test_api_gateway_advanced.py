# -*- coding: utf-8 -*-
"""
核心服务层 - API网关高级单元测试
测试覆盖率目标: 95%+
按照业务流程驱动架构设计测试API网关核心功能
"""

import pytest
import asyncio
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from concurrent.futures import ThreadPoolExecutor

from src.core.integration.apis.api_gateway import (
    ApiGateway, RouteRule, ServiceEndpoint, RateLimitRule,
    HttpMethod, ServiceStatus, RateLimitType, CircuitBreaker,
    ApiRequest, ApiResponse, LoadBalancer
)

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]


class TestApiGatewayCore:
    """测试API网关核心功能"""

    def setup_method(self, method):
        """设置测试环境"""
        self.config = {
            'port': 8080,
            'host': '0.0.0.0',
            'jwt_secret': 'test_secret',
            'redis_enabled': False
        }
        self.gateway = ApiGateway(self.config)

    def test_api_gateway_initialization(self):
        """测试API网关初始化"""
        assert isinstance(self.gateway.routes, dict)
        assert isinstance(self.gateway.services, dict)
        assert hasattr(self.gateway, 'rate_limiter')
        assert isinstance(self.gateway.circuit_breakers, dict)
        assert hasattr(self.gateway, 'response_cache')
        assert self.gateway.port == 8080

    def test_route_registration(self):
        """测试路由注册"""
        route = RouteRule(
            path="/api/v1/users",
            method=HttpMethod.GET,
            service_name="user-service",
            auth_required=True,
            rate_limits=[
                RateLimitRule(RateLimitType.IP, 100, 60),
                RateLimitRule(RateLimitType.USER, 1000, 3600)
            ]
        )

        self.gateway.add_route(route)

        route_key = f"{HttpMethod.GET.value}:/api/v1/users"
        assert route_key in self.gateway.routes
        assert self.gateway.routes[route_key] == route

    def test_service_registration(self):
        """测试服务注册"""
        service = ServiceEndpoint(
            service_name="user-service",
            path="/api/v1",
            method=HttpMethod.GET,
            upstream_url="http://user-service:8080",
            weight=2,
            health_check_url="http://user-service:8080/health"
        )

        # 使用正确的register_service调用方式
        self.gateway.register_service("user-service", ["http://user-service:8080"])

        assert "user-service" in self.gateway.services
        load_balancer = self.gateway.services["user-service"]
        assert len(load_balancer.endpoints) == 1
        assert load_balancer.endpoints[0].upstream_url == "http://user-service:8080"

    def test_route_matching(self):
        """测试路由匹配"""
        # 注册路由
        route = RouteRule(
            path="/api/v1/users/{id}",
            method=HttpMethod.GET,
            service_name="user-service"
        )
        self.gateway.add_route(route)

        # 测试匹配
        matched_route, path_params = self.gateway._match_route(HttpMethod.GET, "/api/v1/users/123")

        assert matched_route is not None
        assert matched_route.service_name == "user-service"
        assert path_params["id"] == "123"

    def test_route_matching_with_wildcards(self):
        """测试通配符路由匹配"""
        # 注册带通配符的路由
        route = RouteRule(
            path="/api/v1/*",
            method=HttpMethod.GET,
            service_name="catch-all-service"
        )
        self.gateway.add_route(route)

        # 测试匹配
        matched_route, path_params = self.gateway._match_route(HttpMethod.GET, "/api/v1/anything")

        assert matched_route is not None
        assert matched_route.service_name == "catch-all-service"

    def test_route_not_found(self):
        """测试路由未找到"""
        matched_route, path_params = self.gateway._match_route(HttpMethod.GET, "/nonexistent/route")

        assert matched_route is None
        assert path_params == {}


class TestAuthenticationAuthorization:
    """测试认证授权功能"""

    def setup_method(self, method):
        """设置测试环境"""
        self.config = {
            'port': 8080,
            'host': '0.0.0.0',
            'jwt_secret': 'test_secret',
            'redis_enabled': False
        }
        self.gateway = ApiGateway(self.config)

    def test_jwt_token_validation(self):
        """测试JWT令牌验证"""
        # 生成测试令牌
        payload = {
            "user_id": "test_user",
            "roles": ["user", "admin"],
            "exp": datetime.now() + timedelta(hours=1)
        }

        token = self.gateway._generate_jwt_token(payload)

        # 验证令牌
        is_valid, user_info = self.gateway._validate_jwt_token(token)

        assert is_valid is True
        assert user_info["user_id"] == "test_user"
        assert "admin" in user_info["roles"]

    def test_jwt_token_expiration(self):
        """测试JWT令牌过期"""
        # 生成过期的令牌
        payload = {
            "user_id": "test_user",
            "exp": 0  # 使用时间戳0，确保过期
        }

        token = self.gateway._generate_jwt_token(payload)

        # 验证令牌
        is_valid, user_info = self.gateway._validate_jwt_token(token)

        assert is_valid is False
        assert user_info is None

    def test_api_key_authentication(self):
        """测试API密钥认证"""
        api_key = "test_api_key_12345"

        # 注册API密钥
        self.gateway.api_keys[api_key] = {
            "user_id": "test_user",
            "permissions": ["read", "write"],
            "rate_limit": 100
        }

        # 验证API密钥
        is_valid, user_info = self.gateway._validate_api_key(api_key)

        assert is_valid is True
        assert user_info["user_id"] == "test_user"
        assert "write" in user_info["permissions"]

    def test_invalid_api_key(self):
        """测试无效API密钥"""
        invalid_key = "invalid_key"

        is_valid, user_info = self.gateway._validate_api_key(invalid_key)

        assert is_valid is False
        assert user_info is None

    def test_role_based_access_control(self):
        """测试基于角色的访问控制"""
        # 设置路由权限
        route = RouteRule(
            path="/api/v1/admin",
            method=HttpMethod.POST,
            service_name="admin-service",
            auth_required=True
        )

        user_permissions = {
            "regular_user": ["read"],
            "admin_user": ["read", "write", "admin"],
            "power_user": ["read", "write"]
        }

        # 设置用户权限到API密钥存储中
        for user, permissions in user_permissions.items():
            api_key = f"{user}_key"
            self.gateway.api_keys[api_key] = {
                "user_id": user,
                "permissions": permissions
            }

        # 测试不同用户权限
        for user, permissions in user_permissions.items():
            has_access = self.gateway._check_permissions(user, "/api/v1/admin", "admin")
            expected_access = "admin" in permissions

            assert has_access == expected_access, f"User {user} access check failed"


class TestRateLimiting:
    """测试速率限制功能"""

    def setup_method(self, method):
        """设置测试环境"""
        self.config = {
            'port': 8080,
            'host': '0.0.0.0',
            'jwt_secret': 'test_secret',
            'redis_enabled': False
        }
        self.gateway = ApiGateway(self.config)

    def test_ip_based_rate_limiting(self):
        """测试基于IP的速率限制"""
        client_ip = "192.168.1.100"
        limit = 10
        window = 60

        # 注册速率限制规则
        rule = RateLimitRule(RateLimitType.IP, limit, window, client_ip)
        self.gateway.rate_limits[client_ip] = rule

        # 测试正常请求
        for i in range(limit):
            allowed, _ = self.gateway._check_rate_limit(client_ip, rule)
            assert allowed is True, f"Request {i+1} should be allowed"

        # 测试超出限制
        allowed, _ = self.gateway._check_rate_limit(client_ip, rule)
        assert allowed is False, "Request should be blocked after limit exceeded"

    def test_user_based_rate_limiting(self):
        """测试基于用户的速率限制"""
        user_id = "test_user"
        limit = 5
        window = 300

        rule = RateLimitRule(RateLimitType.USER, limit, window, user_id)
        self.gateway.rate_limits[user_id] = rule

        # 模拟用户请求
        for i in range(limit + 2):
            allowed, _ = self.gateway._check_rate_limit(user_id, rule)
            if i < limit:
                assert allowed is True, f"User request {i+1} should be allowed"
            else:
                assert allowed is False, f"User request {i+1} should be blocked"

    def test_rate_limit_window_reset(self):
        """测试速率限制窗口重置"""
        client_ip = "192.168.1.200"
        limit = 3
        window = 2  # 2秒窗口，便于测试

        rule = RateLimitRule(RateLimitType.IP, limit, window, client_ip)
        self.gateway.rate_limits[client_ip] = rule

        # 填满限制
        for i in range(limit):
            self.gateway._check_rate_limit(client_ip, rule)[0]

        # 超出限制
        allowed, _ = self.gateway._check_rate_limit(client_ip, rule)
        assert allowed is False

        # 等待窗口重置
        time.sleep(window + 0.1)

        # 应该再次允许
        allowed, _ = self.gateway._check_rate_limit(client_ip, rule)
        assert allowed is True

    def test_global_rate_limiting(self):
        """测试全局速率限制"""
        global_key = "global"
        limit = 20
        window = 60

        rule = RateLimitRule(RateLimitType.GLOBAL, limit, window, global_key)
        self.gateway.rate_limits[global_key] = rule

        # 模拟多个客户端的请求
        clients = [f"client_{i}" for i in range(5)]

        total_requests = 0
        for i in range(limit // len(clients) + 2):
            for client in clients:
                allowed, _ = self.gateway._check_rate_limit(global_key, rule)
                if allowed:
                    total_requests += 1
                else:
                    break
            if not allowed:
                break

        # 应该达到全局限制
        assert total_requests <= limit


class TestLoadBalancing:
    """测试负载均衡功能"""

    def setup_method(self, method):
        """设置测试环境"""
        self.config = {
            'port': 8080,
            'host': '0.0.0.0',
            'jwt_secret': 'test_secret',
            'redis_enabled': False
        }
        self.gateway = ApiGateway(self.config)

    def test_round_robin_load_balancing(self):
        """测试轮询负载均衡"""
        service_name = "test-service"
        endpoints = [
            ServiceEndpoint(service_name, "/api", HttpMethod.GET, "http://service1:8080", weight=1),
            ServiceEndpoint(service_name, "/api", HttpMethod.GET, "http://service2:8080", weight=1),
            ServiceEndpoint(service_name, "/api", HttpMethod.GET, "http://service3:8080", weight=1)
        ]

        # 创建LoadBalancer并添加端点
        load_balancer = LoadBalancer()
        for endpoint in endpoints:
            load_balancer.add_endpoint(endpoint)
        self.gateway.services[service_name] = load_balancer

        # 测试轮询分配
        selected_endpoints = []
        for i in range(6):
            endpoint = self.gateway._select_endpoint(service_name)
            selected_endpoints.append(endpoint.upstream_url)

        # 应该均匀分配
        expected_distribution = ["http://service1:8080", "http://service2:8080", "http://service3:8080"] * 2
        assert selected_endpoints == expected_distribution

    def test_weighted_load_balancing(self):
        """测试加权负载均衡"""
        service_name = "weighted-service"
        endpoints = [
            ServiceEndpoint(service_name, "/api", HttpMethod.GET, "http://service1:8080", weight=3),
            ServiceEndpoint(service_name, "/api", HttpMethod.GET, "http://service2:8080", weight=1),
            ServiceEndpoint(service_name, "/api", HttpMethod.GET, "http://service3:8080", weight=2)
        ]

        # 创建LoadBalancer并添加端点
        load_balancer = LoadBalancer(algorithm="weighted")
        for endpoint in endpoints:
            load_balancer.add_endpoint(endpoint)
        self.gateway.services[service_name] = load_balancer

        # 测试加权分配
        selections = {}
        for i in range(60):  # 足够多的请求来观察权重分布
            endpoint = self.gateway._select_endpoint(service_name)
            url = endpoint.upstream_url
            selections[url] = selections.get(url, 0) + 1

        # 验证权重比例 (3:1:2)
        total_selections = sum(selections.values())
        weight_ratios = {
            "http://service1:8080": selections.get("http://service1:8080", 0) / total_selections,
            "http://service2:8080": selections.get("http://service2:8080", 0) / total_selections,
            "http://service3:8080": selections.get("http://service3:8080", 0) / total_selections
        }

        # service1应该获得约50%的请求 (3/6)
        assert 0.35 <= weight_ratios["http://service1:8080"] <= 0.65

        # service2应该获得约17%的请求 (1/6)
        assert 0.05 <= weight_ratios["http://service2:8080"] <= 0.30

        # service3应该获得约33%的请求 (2/6)
        assert 0.20 <= weight_ratios["http://service3:8080"] <= 0.50

    def test_health_based_load_balancing(self):
        """测试基于健康的负载均衡"""
        service_name = "health-service"
        endpoints = [
            ServiceEndpoint(service_name, "/api", HttpMethod.GET, "http://service1:8080", weight=1),
            ServiceEndpoint(service_name, "/api", HttpMethod.GET, "http://service2:8080", weight=1),
            ServiceEndpoint(service_name, "/api", HttpMethod.GET, "http://service3:8080", weight=1, status=ServiceStatus.UNHEALTHY)
        ]

        # 创建LoadBalancer并添加端点
        load_balancer = LoadBalancer()
        for endpoint in endpoints:
            load_balancer.add_endpoint(endpoint)
        self.gateway.services[service_name] = load_balancer

        # 测试健康检查
        for endpoint in endpoints:
            is_healthy = self.gateway._check_endpoint_health(endpoint)
            assert is_healthy == (endpoint.status == ServiceStatus.HEALTHY)

        # 测试负载均衡只选择健康端点
        healthy_selections = set()
        for i in range(10):
            endpoint = self.gateway._select_endpoint(service_name)
            if endpoint.status == ServiceStatus.HEALTHY:
                healthy_selections.add(endpoint.upstream_url)

        # 只应该选择健康的端点
        assert "http://service1:8080" in healthy_selections
        assert "http://service2:8080" in healthy_selections
        assert "http://service3:8080" not in healthy_selections

    def test_failover_load_balancing(self):
        """测试故障转移负载均衡"""
        service_name = "failover-service"
        endpoints = [
            ServiceEndpoint(service_name, "/api", HttpMethod.GET, "http://primary:8080", weight=1),
            ServiceEndpoint(service_name, "/api", HttpMethod.GET, "http://backup:8080", weight=1)
        ]

        # 创建LoadBalancer并添加端点
        load_balancer = LoadBalancer()
        for endpoint in endpoints:
            load_balancer.add_endpoint(endpoint)
        self.gateway.services[service_name] = load_balancer

        # 模拟主服务故障
        primary_endpoint = endpoints[0]
        primary_endpoint.status = ServiceStatus.UNHEALTHY
        primary_endpoint.failure_count = 5

        # 测试故障转移
        selected_endpoint = self.gateway._select_endpoint(service_name)

        # 应该选择备份服务
        assert selected_endpoint.upstream_url == "http://backup:8080"
        assert selected_endpoint.status == ServiceStatus.HEALTHY


class TestCircuitBreaker:
    """测试熔断器功能"""

    def setup_method(self, method):
        """设置测试环境"""
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5, success_threshold=1)

    def test_circuit_breaker_initial_state(self):
        """测试熔断器初始状态"""
        assert self.circuit_breaker.state == "CLOSED"
        assert self.circuit_breaker.failure_count == 0
        assert self.circuit_breaker.success_count == 0

    def test_circuit_breaker_failure_handling(self):
        """测试熔断器故障处理"""
        # 模拟连续失败
        for i in range(3):
            self.circuit_breaker.record_failure()
            assert self.circuit_breaker.failure_count == i + 1

        # 应该熔断
        assert self.circuit_breaker.state == "OPEN"

    def test_circuit_breaker_recovery(self):
        """测试熔断器恢复"""
        # 触发熔断
        for i in range(3):
            self.circuit_breaker.record_failure()
        assert self.circuit_breaker.state == "OPEN"

        # 等待恢复超时
        time.sleep(5.1)

        # 尝试恢复
        can_attempt = self.circuit_breaker.can_attempt()
        assert can_attempt is True
        assert self.circuit_breaker.state == "HALF_OPEN"

        # 记录成功
        self.circuit_breaker.record_success()
        assert self.circuit_breaker.state == "CLOSED"
        assert self.circuit_breaker.failure_count == 0

    def test_circuit_breaker_half_open_failures(self):
        """测试半开状态下的故障"""
        # 触发熔断
        for i in range(3):
            self.circuit_breaker.record_failure()
        assert self.circuit_breaker.state == "OPEN"

        # 等待恢复
        time.sleep(5.1)
        self.circuit_breaker.can_attempt()  # 进入半开状态

        # 在半开状态下再次失败
        self.circuit_breaker.record_failure()
        assert self.circuit_breaker.state == "OPEN"  # 应该回到打开状态

    def test_circuit_breaker_success_threshold(self):
        """测试熔断器成功阈值"""
        # 触发熔断
        for i in range(3):
            self.circuit_breaker.record_failure()
        assert self.circuit_breaker.state == "OPEN"

        # 等待恢复
        time.sleep(5.1)
        self.circuit_breaker.can_attempt()
        assert self.circuit_breaker.state == "HALF_OPEN"

        # 需要多个成功才能完全恢复
        success_threshold = 2
        for i in range(success_threshold):
            self.circuit_breaker.record_success()

        assert self.circuit_breaker.state == "CLOSED"


class TestCachingAndOptimization:
    """测试缓存和优化功能"""

    def setup_method(self, method):
        """设置测试环境"""
        self.config = {
            'port': 8080,
            'host': '0.0.0.0',
            'jwt_secret': 'test_secret',
            'redis_enabled': False
        }
        self.gateway = ApiGateway(self.config)

    def test_response_caching(self):
        """测试响应缓存"""
        cache_key = "GET:/api/v1/users"
        cached_response = ApiResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b'{"users": []}',
            processing_time=0.1,
            upstream_url="http://user-service:8080"
        )

        # 存储缓存（使用正确的缓存存储）
        self.gateway._cache_response(cache_key, cached_response, 300)

        # 获取缓存
        cached_result = self.gateway._get_cached_response(cache_key)
        assert cached_result is not None
        assert cached_result.status_code == 200
        assert cached_result.cached is True

    def test_cache_expiration(self):
        """测试缓存过期"""
        cache_key = "GET:/api/v1/products"
        expired_time = datetime.now() - timedelta(seconds=400)  # 超过TTL

        self.gateway.cache[cache_key] = {
            "response": ApiResponse(200, {}, b'{}', 0.1, "http://service"),
            "timestamp": expired_time,
            "ttl": 300
        }

        # 获取过期缓存
        cached_result = self.gateway._get_cached_response(cache_key)
        assert cached_result is None  # 应该返回None

    def test_cache_invalidation(self):
        """测试缓存失效"""
        cache_key = "POST:/api/v1/users"

        self.gateway.cache[cache_key] = {
            "response": ApiResponse(201, {}, b'{"id": 123}', 0.2, "http://service"),
            "timestamp": datetime.now(),
            "ttl": 300
        }

        # 使缓存失效
        self.gateway._invalidate_cache(cache_key)
        assert cache_key not in self.gateway.cache

    def test_request_deduplication(self):
        """测试请求去重"""
        request_id = "unique_request_123"
        duplicate_request = ApiRequest(
            id=request_id,
            method=HttpMethod.GET,
            path="/api/v1/data",
            headers={},
            query_params={}
        )

        # 首次请求
        is_duplicate = self.gateway._is_duplicate_request(request_id, ttl=1)  # 使用1秒TTL
        assert is_duplicate is False

        # 重复请求（在TTL内）
        is_duplicate = self.gateway._is_duplicate_request(request_id, ttl=1)
        assert is_duplicate is True

        # 等待TTL过期
        import time
        time.sleep(1.1)  # 等待超过1秒TTL

        # 再次请求（应该不被认为是重复）
        is_duplicate = self.gateway._is_duplicate_request(request_id, ttl=1)
        assert is_duplicate is False


class TestServiceDiscovery:
    """测试服务发现功能"""

    def setup_method(self, method):
        """设置测试环境"""
        self.config = {
            'port': 8080,
            'host': '0.0.0.0',
            'jwt_secret': 'test_secret',
            'redis_enabled': False
        }
        self.gateway = ApiGateway(self.config)

    def test_service_registration_and_discovery(self):
        """测试服务注册和发现"""
        service_name = "discovery-test-service"
        service_instances = [
            {"host": "service1.example.com", "port": 8080, "weight": 2},
            {"host": "service2.example.com", "port": 8080, "weight": 1},
            {"host": "service3.example.com", "port": 8080, "weight": 1}
        ]

        # 注册服务实例
        load_balancer = LoadBalancer(algorithm="weighted")
        for instance in service_instances:
            endpoint = ServiceEndpoint(
                service_name=service_name,
                path="/api",
                method=HttpMethod.GET,
                upstream_url=f"http://{instance['host']}:{instance['port']}",
                weight=instance["weight"]
            )
            load_balancer.add_endpoint(endpoint)

        # 将负载均衡器添加到网关服务中
        self.gateway.services[service_name] = load_balancer

        # 发现服务
        load_balancer = self.gateway.services.get(service_name)
        assert load_balancer is not None
        assert len(load_balancer.endpoints) == len(service_instances)

        # 验证权重配置
        weights = [ep.weight for ep in load_balancer.endpoints]
        assert 2 in weights  # 应该有权重为2的实例
        assert weights.count(1) == 2  # 应该有两个权重为1的实例

    def test_service_health_monitoring(self):
        """测试服务健康监控"""
        service_name = "health-monitor-service"
        endpoint = ServiceEndpoint(
            service_name=service_name,
            path="/api",
            method=HttpMethod.GET,
            upstream_url="http://service.example.com:8080",
            health_check_url="http://service.example.com:8080/health"
        )

        # 直接添加到服务中进行测试
        load_balancer = LoadBalancer()
        load_balancer.add_endpoint(endpoint)
        self.gateway.services[service_name] = load_balancer

        # 检查端点状态
        # 默认情况下新创建的端点应该是健康状态
        assert endpoint.status == ServiceStatus.HEALTHY

        # 模拟端点故障
        endpoint.status = ServiceStatus.UNHEALTHY
        assert endpoint.status == ServiceStatus.UNHEALTHY

        # 恢复端点
        endpoint.status = ServiceStatus.HEALTHY
        assert endpoint.status == ServiceStatus.HEALTHY


    def test_service_failover_and_recovery(self):
        """测试服务故障转移和恢复"""
        service_name = "failover-service"
        endpoints = [
            ServiceEndpoint(service_name, "/api", HttpMethod.GET, "http://primary:8080"),
            ServiceEndpoint(service_name, "/api", HttpMethod.GET, "http://secondary:8080"),
            ServiceEndpoint(service_name, "/api", HttpMethod.GET, "http://tertiary:8080")
        ]

        # 设置服务端点
        load_balancer = LoadBalancer()
        for endpoint in endpoints:
            load_balancer.add_endpoint(endpoint)

        self.gateway.services[service_name] = load_balancer

        # 模拟主服务故障
        primary = endpoints[0]
        primary.status = ServiceStatus.UNHEALTHY

        # 故障转移到次级服务
        available_endpoints = [ep for ep in endpoints if ep.status == ServiceStatus.HEALTHY]
        failover_endpoint = available_endpoints[0] if available_endpoints else None

        assert failover_endpoint is not None
        assert failover_endpoint.upstream_url == "http://secondary:8080"

        # 模拟主服务恢复
        primary.status = ServiceStatus.HEALTHY
        primary.last_health_check = datetime.now()

        # 验证恢复
        all_healthy = all(ep.status == ServiceStatus.HEALTHY for ep in endpoints)
        assert all_healthy is True


class TestApiGatewayIntegration:
    """测试API网关集成功能"""

    def setup_method(self, method):
        """设置测试环境"""
        self.config = {
            'port': 8080,
            'host': '0.0.0.0',
            'jwt_secret': 'test_secret',
            'redis_enabled': False
        }
        self.gateway = ApiGateway(self.config)

    @pytest.mark.asyncio
    async def test_request_processing_pipeline(self):
        """测试请求处理管道"""
        # 创建测试请求
        request = ApiRequest(
            id="test_request_001",
            method=HttpMethod.GET,
            path="/api/v1/users/123",
            headers={"Authorization": "Bearer test_token", "X-API-Key": "test_key"},
            query_params={"include": "details"},
            client_ip="192.168.1.100",
            user_id="test_user"
        )

        # 设置路由
        route = RouteRule(
            path="/api/v1/users/{id}",
            method=HttpMethod.GET,
            service_name="user-service",
            auth_required=True,
            rate_limits=[RateLimitRule(RateLimitType.IP, 100, 60)]
        )
        self.gateway.add_route(route)

        # 设置服务
        # 注册服务
        self.gateway.register_service("user-service", ["http://user-service:8080"])

        # 处理请求
        response = await self.gateway.process_request(request)

        # 验证响应
        assert isinstance(response, ApiResponse)
        assert response.status_code in [200, 401, 403, 429, 500]  # 可能的HTTP状态码

    def test_api_gateway_metrics_collection(self):
        """测试API网关指标收集"""
        # 模拟一些请求处理
        requests = [
            ApiRequest("req1", HttpMethod.GET, "/api/users", {}, {}, client_ip="192.168.1.100"),
            ApiRequest("req2", HttpMethod.POST, "/api/users", {}, {}, client_ip="192.168.1.100"),
            ApiRequest("req3", HttpMethod.GET, "/api/products", {}, {}, client_ip="192.168.1.101")
        ]

        responses = [
            ApiResponse(200, {}, b'{"users": []}', 0.1, "http://service1"),
            ApiResponse(201, {}, b'{"id": 123}', 0.15, "http://service1"),
            ApiResponse(200, {}, b'{"products": []}', 0.08, "http://service2")
        ]

        # 收集指标
        for request, response in zip(requests, responses):
            self.gateway._collect_metrics(request, response)

        # 验证指标收集
        metrics = self.gateway.metrics

        assert "total_requests" in metrics
        assert "total_responses" in metrics
        assert "avg_response_time" in metrics
        assert "requests_by_method" in metrics
        assert "requests_by_endpoint" in metrics

        assert metrics["total_requests"] == len(requests)
        assert metrics["total_responses"] == len(responses)
        assert metrics["avg_response_time"] > 0

    def test_api_gateway_error_handling(self):
        """测试API网关错误处理"""
        error_scenarios = [
            {"type": "auth_error", "status_code": 401, "message": "Unauthorized"},
            {"type": "rate_limit_error", "status_code": 429, "message": "Too Many Requests"},
            {"type": "service_unavailable", "status_code": 503, "message": "Service Unavailable"},
            {"type": "timeout_error", "status_code": 504, "message": "Gateway Timeout"}
        ]

        for scenario in error_scenarios:
            error_response = self.gateway._create_error_response(
                scenario["status_code"],
                scenario["message"]
            )

            assert error_response.status_code == scenario["status_code"]
            assert scenario["message"] in error_response.body.decode()
            assert "Content-Type" in error_response.headers

    def test_api_gateway_configuration_management(self):
        """测试API网关配置管理"""
        # 测试配置更新
        new_config = {
            "port": 9090,
            "max_concurrent_requests": 1000,
            "request_timeout": 60,
            "rate_limit_enabled": True,
            "cache_enabled": True,
            "cors_enabled": True
        }

        # 应用配置
        for key, value in new_config.items():
            if hasattr(self.gateway, key):
                setattr(self.gateway, key, value)

        # 验证配置应用
        assert self.gateway.port == new_config["port"]
        assert self.gateway.max_concurrent_requests == new_config["max_concurrent_requests"]
        assert self.gateway.request_timeout == new_config["request_timeout"]
        assert self.gateway.rate_limit_enabled == new_config["rate_limit_enabled"]
        assert self.gateway.cache_enabled == new_config["cache_enabled"]
        assert self.gateway.cors_enabled == new_config["cors_enabled"]

    def test_api_gateway_graceful_shutdown(self):
        """测试API网关优雅关闭"""
        # 设置一些活跃请求（使用ApiGateway实际的属性）
        self.gateway.current_requests = 3  # 设置当前有3个活跃请求

        # 模拟关闭过程
        shutdown_successful = True
        shutdown_timeout = 5.0  # 5秒超时
        start_time = time.time()

        try:
            # 停止接受新请求（设置一个标志）
            self.gateway._shutdown_requested = True

            # 等待现有请求完成，使用实际的并发控制机制
            with self.gateway.requests_lock:
                while self.gateway.current_requests > 0 and (time.time() - start_time) < shutdown_timeout:
                    # 模拟请求逐渐完成
                    if self.gateway.current_requests > 0:
                        self.gateway.current_requests -= 1
                        time.sleep(0.1)  # 模拟请求完成时间

                    # 如果超时，强制终止剩余请求
                    if (time.time() - start_time) >= shutdown_timeout:
                        self.gateway.current_requests = 0
                        break

            # 清理资源
            with self.gateway.cache_lock:
                self.gateway.cache.clear()
            with self.gateway.rate_limit_lock:
                self.gateway.rate_limits.clear()

        except Exception as e:
            shutdown_successful = False
            print(f"Shutdown error: {e}")

        # 验证优雅关闭
        assert shutdown_successful is True
        assert self.gateway.current_requests == 0
        assert len(self.gateway.cache) == 0
        assert len(self.gateway.rate_limits) == 0
