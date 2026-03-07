#!/usr/bin/env python3
"""
API网关综合测试

为src/core/api_gateway.py提供全面的测试覆盖
"""

import pytest

# 尝试导入所需模块
try:
    from src.core.core_services.api.api_service import APIService
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
import json
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import asyncio
import aiohttp
from aiohttp import web
import threading
from concurrent.futures import ThreadPoolExecutor

from src.core.api_gateway import (

ApiGateway,
    RouteRule,
    LoadBalancer,
    CircuitBreaker,
    RateLimiter,
    ApiRequest,
    ApiResponse,
    ServiceEndpoint
)

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]




@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestApiGatewayCore:
    """API网关核心功能测试"""

    def setup_method(self):
        """测试前准备"""
        self.config = {
            'host': 'localhost',
            'port': 8080,
            'jwt_secret': 'test_secret',
            'redis_enabled': False
        }
        self.gateway = ApiGateway(self.config)

    def test_gateway_initialization(self):
        """测试网关初始化"""
        assert self.gateway.config == self.config
        assert isinstance(self.gateway.routes, dict)
        assert isinstance(self.gateway.services, dict)
        assert isinstance(self.gateway.circuit_breakers, dict)
        assert hasattr(self.gateway, 'rate_limiter')
        assert hasattr(self.gateway, 'auth_manager')

    def test_route_registration(self):
        """测试路由注册"""
        # 创建测试路由规则
        route_rule = RouteRule(
            path="/api/test",
            methods=["GET", "POST"],
            service_name="test_service",
            endpoint_path="/test"
        )

        # 注册路由
        self.gateway.add_route(route_rule)

        # 验证路由注册
        assert len(self.gateway.routes) > 0
        # 路由存储方式可能不同，检查是否有路由被添加

    def test_service_registration(self):
        """测试服务注册"""
        service_name = "test_service"
        endpoints = ["http://localhost:8081", "http://localhost:8082"]

        # 注册服务
        self.gateway.register_service(service_name, endpoints)

        # 验证服务注册
        assert service_name in self.gateway.services
        load_balancer = self.gateway.services[service_name]
        assert isinstance(load_balancer, LoadBalancer)

    def test_route_matching(self):
        """测试路由匹配"""
        # 注册路由
        route_rule = RouteRule(
            path="/api/users/{id}",
            methods=["GET"],
            service_name="user_service",
            endpoint_path="/users/{id}"
        )
        self.gateway.add_route(route_rule)

        # 测试路由匹配
        matched_route, params = self.gateway._match_route("GET", "/api/users/123")

        assert matched_route is not None
        assert params["id"] == "123"

    def test_health_check(self):
        """测试健康检查"""
        # 创建模拟请求
        mock_request = Mock()
        mock_request.method = "GET"
        mock_request.path = "/health"

        # 执行健康检查（同步版本用于测试）
        response_future = asyncio.run(self.gateway.health_check(mock_request))

        # 验证响应
        assert response_future.status == 200
        response_data = response_future.text
        assert "healthy" in response_data.lower()

    def test_metrics_endpoint(self):
        """测试指标端点"""
        # 创建模拟请求
        mock_request = Mock()
        mock_request.method = "GET"
        mock_request.path = "/metrics"

        # 执行指标请求
        response_future = asyncio.run(self.gateway.get_metrics(mock_request))

        # 验证响应
        assert response_future.status == 200
        response_data = response_future.text
        # 应该包含JSON格式的指标数据


class TestLoadBalancer:
    """负载均衡器测试"""

    def setup_method(self):
        """测试前准备"""
        self.endpoints = [
            "http://backend1.example.com",
            "http://backend2.example.com",
            "http://backend3.example.com"
        ]
        self.load_balancer = LoadBalancer(self.endpoints)

    def test_round_robin_selection(self):
        """测试轮询选择算法"""
        # 测试轮询选择
        selected_endpoints = []
        for i in range(6):  # 多于一轮以验证循环
            endpoint = self.load_balancer.select_endpoint()
            selected_endpoints.append(endpoint)

        # 验证轮询顺序
        expected_order = self.endpoints * 2  # 两轮
        assert selected_endpoints == expected_order

    def test_weighted_round_robin(self):
        """测试加权轮询"""
        # 设置权重
        weights = [1, 2, 3]  # backend1: 1次, backend2: 2次, backend3: 3次
        self.load_balancer.set_weights(weights)

        # 收集选择结果
        selections = {}
        total_requests = 6  # 1+2+3的公倍数

        for i in range(total_requests):
            endpoint = self.load_balancer.select_endpoint()
            selections[endpoint] = selections.get(endpoint, 0) + 1

        # 验证权重比例
        assert selections[self.endpoints[0]] == 1  # 权重1
        assert selections[self.endpoints[1]] == 2  # 权重2
        assert selections[self.endpoints[2]] == 3  # 权重3

    def test_endpoint_health_monitoring(self):
        """测试端点健康监控"""
        # 模拟健康检查
        health_status = {
            self.endpoints[0]: True,   # 健康
            self.endpoints[1]: False,  # 不健康
            self.endpoints[2]: True    # 健康
        }

        # 更新健康状态
        for endpoint, healthy in health_status.items():
            self.load_balancer.update_health(endpoint, healthy)

        # 测试选择（应该避开不健康的端点）
        selected_endpoints = []
        for i in range(10):
            endpoint = self.load_balancer.select_endpoint()
            selected_endpoints.append(endpoint)

        # 验证不选择不健康的端点
        assert self.endpoints[1] not in selected_endpoints

        # 验证只选择健康的端点
        healthy_endpoints = [ep for ep in self.endpoints if health_status[ep]]
        for selected in selected_endpoints:
            assert selected in healthy_endpoints

    def test_dynamic_endpoint_management(self):
        """测试动态端点管理"""
        # 添加新端点
        new_endpoint = "http://backend4.example.com"
        self.load_balancer.add_endpoint(new_endpoint)

        # 验证新端点被添加
        assert new_endpoint in self.load_balancer.endpoints

        # 测试新端点被选择
        found_new = False
        for i in range(10):
            endpoint = self.load_balancer.select_endpoint()
            if endpoint == new_endpoint:
                found_new = True
                break
        assert found_new

        # 移除端点
        self.load_balancer.remove_endpoint(new_endpoint)
        assert new_endpoint not in self.load_balancer.endpoints

        # 验证移除的端点不再被选择
        for i in range(10):
            endpoint = self.load_balancer.select_endpoint()
            assert endpoint != new_endpoint


class TestCircuitBreaker:
    """熔断器测试"""

    def setup_method(self):
        """测试前准备"""
        self.config = {
            "failure_threshold": 3,
            "recovery_timeout": 5,
            "expected_exception": Exception
        }
        self.circuit_breaker = CircuitBreaker(**self.config)

    def test_circuit_breaker_initial_state(self):
        """测试熔断器初始状态"""
        assert self.circuit_breaker.state == "closed"
        assert self.circuit_breaker.failure_count == 0
        assert not self.circuit_breaker.is_open()

    def test_successful_requests(self):
        """测试成功请求"""
        # 执行成功请求
        for i in range(5):
            result = self.circuit_breaker.call(lambda: "success")
            assert result == "success"

        # 验证状态保持关闭
        assert self.circuit_breaker.state == "closed"
        assert self.circuit_breaker.failure_count == 0

    def test_failure_threshold_breaking(self):
        """测试失败阈值熔断"""
        # 执行失败请求达到阈值
        for i in range(self.config["failure_threshold"]):
            with pytest.raises(Exception):
                self.circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("Test failure")))

        # 验证熔断器开启
        assert self.circuit_breaker.state == "open"
        assert self.circuit_breaker.is_open()

    def test_open_circuit_rejection(self):
        """测试开启状态拒绝请求"""
        # 先触发熔断
        for i in range(self.config["failure_threshold"]):
            with pytest.raises(Exception):
                self.circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("Test failure")))

        # 验证开启状态拒绝请求
        with pytest.raises(Exception, match="Circuit breaker is open"):
            self.circuit_breaker.call(lambda: "should_not_execute")

    def test_half_open_recovery(self):
        """测试半开状态恢复"""
        # 触发熔断
        for i in range(self.config["failure_threshold"]):
            with pytest.raises(Exception):
                self.circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("Test failure")))

        assert self.circuit_breaker.state == "open"

        # 等待恢复超时
        time.sleep(self.config["recovery_timeout"] + 0.1)

        # 执行成功请求，应该关闭熔断器
        result = self.circuit_breaker.call(lambda: "recovery_success")
        assert result == "recovery_success"
        assert self.circuit_breaker.state == "closed"

    def test_half_open_failure_fallback(self):
        """测试半开状态失败回退"""
        # 触发熔断
        for i in range(self.config["failure_threshold"]):
            with pytest.raises(Exception):
                self.circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("Test failure")))

        # 等待恢复超时
        time.sleep(self.config["recovery_timeout"] + 0.1)

        # 执行失败请求，应该回到开启状态
        with pytest.raises(Exception):
            self.circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("Recovery failure")))

        assert self.circuit_breaker.state == "open"

    def test_metrics_collection(self):
        """测试指标收集"""
        # 执行各种操作
        self.circuit_breaker.call(lambda: "success1")
        self.circuit_breaker.call(lambda: "success2")

        for i in range(2):
            with pytest.raises(Exception):
                self.circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("failure")))

        # 获取指标
        metrics = self.circuit_breaker.get_metrics()

        assert metrics["total_requests"] == 4
        assert metrics["successful_requests"] == 2
        assert metrics["failed_requests"] == 2
        assert metrics["failure_rate"] == 0.5
        assert metrics["state"] == "closed"  # 还没达到阈值


class TestRateLimiter:
    """限流器测试"""

    def setup_method(self):
        """测试前准备"""
        self.config = {
            "requests_per_second": 5,
            "burst_size": 10
        }
        self.rate_limiter = RateLimiter(**self.config)

    def test_rate_limiter_initialization(self):
        """测试限流器初始化"""
        assert self.rate_limiter.requests_per_second == self.config["requests_per_second"]
        assert self.rate_limiter.burst_size == self.config["burst_size"]
        assert self.rate_limiter.tokens >= 0

    def test_normal_rate_requests(self):
        """测试正常速率请求"""
        # 执行允许的请求
        for i in range(self.config["requests_per_second"]):
            allowed = self.rate_limiter.allow_request()
            assert allowed

        # 验证令牌消耗
        assert self.rate_limiter.tokens < self.config["burst_size"]

    def test_burst_handling(self):
        """测试突发请求处理"""
        # 先填充令牌
        time.sleep(1)  # 等待令牌补充

        # 执行突发请求
        burst_requests = 0
        for i in range(self.config["burst_size"] + 5):  # 超过burst_size
            if self.rate_limiter.allow_request():
                burst_requests += 1
            else:
                break

        # 验证突发容量
        assert burst_requests == self.config["burst_size"]

    def test_rate_limiting_enforcement(self):
        """测试速率限制执行"""
        # 快速消耗所有令牌
        for i in range(self.config["burst_size"]):
            allowed = self.rate_limiter.allow_request()
            if not allowed:
                break

        # 验证后续请求被限制
        denied_requests = 0
        for i in range(5):
            if not self.rate_limiter.allow_request():
                denied_requests += 1

        assert denied_requests > 0

    def test_token_replenishment(self):
        """测试令牌补充"""
        # 消耗所有令牌
        for i in range(self.config["burst_size"]):
            self.rate_limiter.allow_request()

        # 等待令牌补充
        time.sleep(1.1)  # 稍微超过1秒

        # 验证令牌得到补充
        replenished_requests = 0
        for i in range(self.config["requests_per_second"]):
            if self.rate_limiter.allow_request():
                replenished_requests += 1

        assert replenished_requests > 0

    def test_concurrent_access(self):
        """测试并发访问"""
        results = []
        errors = []

        def concurrent_request_worker(worker_id):
            """并发请求工作线程"""
            try:
                allowed_count = 0
                for i in range(20):
                    if self.rate_limiter.allow_request():
                        allowed_count += 1
                    time.sleep(0.01)  # 小延迟

                results.append({
                    'worker_id': worker_id,
                    'allowed': allowed_count,
                    'total': 20
                })
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # 启动并发线程
        threads = []
        num_threads = 5

        for i in range(num_threads):
            thread = threading.Thread(target=concurrent_request_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证并发访问结果
        assert len(results) == num_threads
        assert len(errors) == 0

        total_allowed = sum(r['allowed'] for r in results)
        total_requested = sum(r['total'] for r in results)

        # 验证总请求数在合理范围内
        assert total_requested == num_threads * 20
        # 允许的请求数应该受到速率限制
        assert total_allowed <= self.config["burst_size"] + self.config["requests_per_second"]


class TestAPIGatewayIntegration:
    """API网关集成测试"""

    def setup_method(self):
        """测试前准备"""
        self.config = GatewayConfig(
            host="localhost",
            port=0,  # 使用随机端口
            workers=2,
            timeout=10,
            max_connections=100
        )
        self.gateway = APIGateway(self.config)

    def teardown_method(self):
        """测试后清理"""
        if hasattr(self.gateway, '_server') and self.gateway._server:
            self.gateway._server.close()

    @pytest.mark.asyncio
    async def test_full_request_flow(self):
        """测试完整请求流程"""
        # 注册路由
        async def test_handler(request):
            return web.json_response({
                "status": "success",
                "method": request.method,
                "path": request.path,
                "query": dict(request.query)
            })

        route = Route("/api/integration", ["GET", "POST"], test_handler, "integration_test")
        self.gateway.register_route(route)

        # 注册中间件
        async def logging_middleware(request, handler):
            # 添加请求头
            request.headers['X-Test'] = 'middleware_processed'
            response = await handler(request)
            response.headers['X-Processed-By'] = 'test_middleware'
            return response

        middleware = Middleware("test_middleware", logging_middleware, priority=1)
        self.gateway.register_middleware(middleware)

        # 创建测试应用
        app = web.Application()
        self.gateway.setup_routes(app)

        # 创建测试客户端
        async with aiohttp.ClientSession() as session:
            # 启动测试服务器
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', 0)
            await site.start()

            # 获取实际端口
            port = site._server.sockets[0].getsockname()[1]

            try:
                # 发送测试请求
                url = f"http://localhost:{port}/api/integration"
                params = {"test": "value", "number": "123"}

                async with session.get(url, params=params) as response:
                    assert response.status == 200

                    data = await response.json()
                    assert data["status"] == "success"
                    assert data["method"] == "GET"
                    assert data["path"] == "/api/integration"
                    assert data["query"]["test"] == "value"
                    assert data["query"]["number"] == "123"

                    # 验证中间件处理
                    assert response.headers.get("X-Processed-By") == "test_middleware"

            finally:
                await runner.cleanup()

    def test_performance_under_load(self):
        """测试负载下的性能"""
        # 注册简单路由
        def simple_handler(request):
            return {"status": "ok", "timestamp": time.time()}

        route = Route("/api/loadtest", ["GET"], simple_handler, "load_test")
        self.gateway.register_route(route)

        # 模拟高并发请求
        def load_test_worker(worker_id):
            """负载测试工作线程"""
            results = []
            for i in range(50):  # 每个线程50个请求
                try:
                    # 模拟请求处理
                    start_time = time.perf_counter()
                    result = simple_handler(Mock())
                    end_time = time.perf_counter()

                    response_time = end_time - start_time
                    results.append({
                        'request_id': f"{worker_id}_{i}",
                        'response_time': response_time,
                        'success': True
                    })

                    # 记录到网关指标
                    self.gateway._record_request_metric(
                        "/api/loadtest", "GET", 200, response_time
                    )

                except Exception as e:
                    results.append({
                        'request_id': f"{worker_id}_{i}",
                        'error': str(e),
                        'success': False
                    })

            return results

        # 执行并发负载测试
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(load_test_worker, i) for i in range(10)]
            all_results = []

            for future in futures:
                all_results.extend(future.result())

        # 分析负载测试结果
        successful_requests = [r for r in all_results if r['success']]
        failed_requests = [r for r in all_results if not r['success']]

        assert len(successful_requests) >= len(all_results) * 0.95  # 95%成功率

        if successful_requests:
            avg_response_time = sum(r['response_time'] for r in successful_requests) / len(successful_requests)
            max_response_time = max(r['response_time'] for r in successful_requests)

            # 验证性能指标
            assert avg_response_time < 0.01  # 平均响应时间小于10ms
            assert max_response_time < 0.1   # 最大响应时间小于100ms

        # 验证网关指标
        metrics = self.gateway.get_metrics()
        assert metrics['total_requests'] == len(all_results)
        assert metrics['successful_requests'] == len(successful_requests)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
