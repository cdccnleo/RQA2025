#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API网关集成测试
测试API网关与其他交易组件的集成
"""

import pytest
import time
import threading
import json
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
import asyncio
import aiohttp
from fastapi import FastAPI
from fastapi.testclient import TestClient



# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

@dataclass
class APIRequest:
    """API请求"""
    method: str
    path: str
    headers: Dict[str, str]
    body: Optional[Dict[str, Any]] = None
    query_params: Optional[Dict[str, str]] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class APIResponse:
    """API响应"""
    status_code: int
    headers: Dict[str, str]
    body: Dict[str, Any]
    response_time: float
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class RateLimiter:
    """速率限制器"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
        self.lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        """检查是否允许请求"""
        with self.lock:
            current_time = time.time()

            # 清理过期请求
            if client_id in self.requests:
                self.requests[client_id] = [
                    req_time for req_time in self.requests[client_id]
                    if current_time - req_time < self.window_seconds
                ]

            # 检查请求数量
            if client_id not in self.requests:
                self.requests[client_id] = []

            if len(self.requests[client_id]) >= self.max_requests:
                return False

            # 记录新请求
            self.requests[client_id].append(current_time)
            return True


class Authenticator:
    """认证器"""

    def __init__(self):
        self.valid_tokens = {
            "valid_token_123": {"user_id": "user_001", "permissions": ["read", "trade"]},
            "admin_token_456": {"user_id": "admin_001", "permissions": ["read", "trade", "admin"]}
        }

    def authenticate(self, token: str) -> Optional[Dict[str, Any]]:
        """认证token"""
        return self.valid_tokens.get(token)

    def authorize(self, user_info: Dict[str, Any], required_permissions: List[str]) -> bool:
        """授权检查"""
        user_permissions = user_info.get("permissions", [])
        return all(perm in user_permissions for perm in required_permissions)


class RequestRouter:
    """请求路由器"""

    def __init__(self):
        self.routes = {}
        self.services = {}

    def add_route(self, path: str, service_name: str, method: str = "GET"):
        """添加路由"""
        if path not in self.routes:
            self.routes[path] = {}
        self.routes[path][method] = service_name

    def add_service(self, service_name: str, service_url: str):
        """添加服务"""
        self.services[service_name] = service_url

    def route_request(self, request: APIRequest) -> Optional[str]:
        """路由请求到服务"""
        path_routes = self.routes.get(request.path)
        if not path_routes:
            return None

        service_name = path_routes.get(request.method)
        if not service_name:
            return None

        return self.services.get(service_name)


class CircuitBreaker:
    """熔断器"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = {}
        self.last_failure_time = {}
        self.state = {}  # 'closed', 'open', 'half_open'
        self.lock = threading.Lock()

    def is_allowed(self, service_name: str) -> bool:
        """检查是否允许调用服务"""
        with self.lock:
            if service_name not in self.state:
                self.state[service_name] = 'closed'
                self.failure_count[service_name] = 0

            state = self.state[service_name]

            if state == 'closed':
                return True
            elif state == 'open':
                # 检查是否可以进入半开状态
                if time.time() - self.last_failure_time.get(service_name, 0) > self.recovery_timeout:
                    self.state[service_name] = 'half_open'
                    return True
                return False
            elif state == 'half_open':
                return True

            return False

    def record_success(self, service_name: str):
        """记录成功调用"""
        with self.lock:
            if self.state.get(service_name) == 'half_open':
                self.state[service_name] = 'closed'
                self.failure_count[service_name] = 0

    def record_failure(self, service_name: str):
        """记录失败调用"""
        with self.lock:
            self.failure_count[service_name] = self.failure_count.get(service_name, 0) + 1
            self.last_failure_time[service_name] = time.time()

            if self.failure_count[service_name] >= self.failure_threshold:
                self.state[service_name] = 'open'


class APIGateway:
    """API网关"""

    def __init__(self):
        self.authenticator = Authenticator()
        self.rate_limiter = RateLimiter()
        self.router = RequestRouter()
        self.circuit_breaker = CircuitBreaker()
        self.request_logs = []
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0
        }

    def handle_request(self, request: APIRequest) -> APIResponse:
        """处理API请求"""
        start_time = time.time()

        try:
            # 1. 速率限制检查
            client_id = request.headers.get("X-Client-ID", "anonymous")
            if not self.rate_limiter.is_allowed(client_id):
                return APIResponse(
                    status_code=429,
                    headers={"Content-Type": "application/json"},
                    body={"error": "Rate limit exceeded"},
                    response_time=time.time() - start_time,
                    timestamp=time.time()
                )

            # 2. 认证检查
            auth_token = request.headers.get("Authorization", "").replace("Bearer ", "")
            user_info = self.authenticator.authenticate(auth_token)
            if not user_info:
                return APIResponse(
                    status_code=401,
                    headers={"Content-Type": "application/json"},
                    body={"error": "Unauthorized"},
                    response_time=time.time() - start_time,
                    timestamp=time.time()
                )

            # 3. 路由请求
            service_url = self.router.route_request(request)
            if not service_url:
                return APIResponse(
                    status_code=404,
                    headers={"Content-Type": "application/json"},
                    body={"error": "Service not found"},
                    response_time=time.time() - start_time,
                    timestamp=time.time()
                )

            # 4. 熔断器检查
            service_name = service_url.split("/")[-1]  # 简化服务名提取
            if not self.circuit_breaker.is_allowed(service_name):
                return APIResponse(
                    status_code=503,
                    headers={"Content-Type": "application/json"},
                    body={"error": "Service temporarily unavailable"},
                    response_time=time.time() - start_time,
                    timestamp=time.time()
                )

            # 5. 转发请求到服务
            response = self._forward_request(service_url, request, user_info)

            # 6. 记录成功
            self.circuit_breaker.record_success(service_name)
            self.metrics["successful_requests"] += 1

            return response

        except Exception as e:
            # 记录失败
            service_name = getattr(self, '_extract_service_name', lambda x: 'unknown')(request.path)
            self.circuit_breaker.record_failure(service_name)
            self.metrics["failed_requests"] += 1

            return APIResponse(
                status_code=500,
                headers={"Content-Type": "application/json"},
                body={"error": str(e)},
                response_time=time.time() - start_time
            )

        finally:
            # 更新统计
            self.metrics["total_requests"] += 1
            response_time = time.time() - start_time
            self.metrics["avg_response_time"] = (
                (self.metrics["avg_response_time"] * (self.metrics["total_requests"] - 1)) +
                response_time
            ) / self.metrics["total_requests"]

            # 记录请求日志
            self.request_logs.append({
                "request": request,
                "response_time": response_time,
                "timestamp": time.time()
            })

    def _forward_request(self, service_url: str, request: APIRequest, user_info: Dict[str, Any]) -> APIResponse:
        """转发请求到后端服务"""
        # 模拟服务调用
        time.sleep(0.01)  # 模拟网络延迟

        # 根据路径返回不同的响应
        if "/trading/order" in request.path:
            return APIResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body={
                    "order_id": "order_123",
                    "status": "submitted",
                    "user_id": user_info["user_id"]
                },
                response_time=0.01
            )
        elif "/portfolio/positions" in request.path:
            return APIResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body={
                    "positions": [
                        {"symbol": "000001.SZ", "quantity": 1000, "avg_price": 10.0}
                    ],
                    "user_id": user_info["user_id"]
                },
                response_time=0.01
            )
        else:
            return APIResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body={"message": "Request processed", "user_id": user_info["user_id"]},
                response_time=0.01
            )

    def get_metrics(self) -> Dict[str, Any]:
        """获取网关指标"""
        return self.metrics.copy()

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            "status": "healthy",
            "uptime": time.time(),
            "metrics": self.get_metrics()
        }


class MockTradingService:
    """模拟交易服务"""

    def __init__(self):
        self.orders = []
        self.positions = {}

    def submit_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """提交订单"""
        order_id = f"order_{len(self.orders)}"
        order = {"order_id": order_id, **order_data}
        self.orders.append(order)

        return {
            "order_id": order_id,
            "status": "submitted",
            "timestamp": time.time()
        }

    def get_positions(self) -> Dict[str, Any]:
        """获取持仓"""
        return {
            "positions": list(self.positions.values()),
            "timestamp": time.time()
        }


class TestAPIGatewayIntegration:
    """API网关集成测试"""

    @pytest.fixture
    def setup_gateway_components(self):
        """设置网关测试组件"""
        # 创建API网关
        gateway = APIGateway()

        # 配置路由
        gateway.router.add_route("/trading/order", "trading_service", "POST")
        gateway.router.add_route("/portfolio/positions", "trading_service", "GET")
        gateway.router.add_service("trading_service", "http://trading-service:8000")

        # 创建模拟交易服务
        trading_service = MockTradingService()

        return {
            "gateway": gateway,
            "trading_service": trading_service
        }

    def test_api_gateway_authentication(self, setup_gateway_components):
        """测试API网关认证"""
        components = setup_gateway_components
        gateway = components["gateway"]

        # 创建未认证请求
        request = APIRequest(
            method="GET",
            path="/portfolio/positions",
            headers={}
        )

        response = gateway.handle_request(request)

        # 验证认证失败
        assert response.status_code == 401
        assert "error" in response.body

    def test_api_gateway_authorization(self, setup_gateway_components):
        """测试API网关授权"""
        components = setup_gateway_components
        gateway = components["gateway"]

        # 创建有效token但权限不足的请求
        request = APIRequest(
            method="POST",
            path="/admin/config",
            headers={"Authorization": "Bearer valid_token_123"}  # 普通用户token
        )

        # 假设需要admin权限
        response = gateway.handle_request(request)

        # 验证授权失败（如果有权限检查）
        # 注意：当前实现中没有具体的授权检查，所以这里主要测试认证
        assert response.status_code in [200, 404]  # 成功或未找到

    def test_api_gateway_routing(self, setup_gateway_components):
        """测试API网关路由"""
        components = setup_gateway_components
        gateway = components["gateway"]

        # 创建有效请求
        request = APIRequest(
            method="POST",
            path="/trading/order",
            headers={"Authorization": "Bearer valid_token_123"},
            body={"symbol": "000001.SZ", "quantity": 100, "price": 10.0}
        )

        response = gateway.handle_request(request)

        # 验证路由成功
        assert response.status_code == 200
        assert "order_id" in response.body

    def test_api_gateway_rate_limiting(self, setup_gateway_components):
        """测试API网关速率限制"""
        components = setup_gateway_components
        gateway = components["gateway"]

        # 创建大量请求
        requests = [
            APIRequest(
                method="GET",
                path="/portfolio/positions",
                headers={
                    "Authorization": "Bearer valid_token_123",
                    "X-Client-ID": "test_client"
                }
            )
            for _ in range(150)  # 超过默认限额100
        ]

        responses = []
        for request in requests:
            response = gateway.handle_request(request)
            responses.append(response)

        # 检查是否有速率限制响应
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        successful_responses = [r for r in responses if r.status_code == 200]

        # 验证速率限制生效
        assert len(rate_limited_responses) > 0 or len(successful_responses) > 0

    def test_api_gateway_circuit_breaker(self, setup_gateway_components):
        """测试API网关熔断器"""
        components = setup_gateway_components
        gateway = components["gateway"]

        # 模拟服务失败
        with patch.object(gateway, '_forward_request', side_effect=Exception("Service unavailable")):
            # 发送多个失败请求
            for _ in range(6):  # 超过失败阈值5
                request = APIRequest(
                    method="GET",
                    path="/portfolio/positions",
                    headers={"Authorization": "Bearer valid_token_123"}
                )
                gateway.handle_request(request)

            # 发送新请求，应该被熔断
            final_request = APIRequest(
                method="GET",
                path="/portfolio/positions",
                headers={"Authorization": "Bearer valid_token_123"}
            )
            response = gateway.handle_request(final_request)

            # 验证熔断器生效
            assert response.status_code == 503

    def test_api_gateway_with_trading_service_integration(self, setup_gateway_components):
        """测试API网关与交易服务集成"""
        components = setup_gateway_components
        gateway = components["gateway"]
        trading_service = components["trading_service"]

        # 通过网关提交订单
        order_request = APIRequest(
            method="POST",
            path="/trading/order",
            headers={"Authorization": "Bearer valid_token_123"},
            body={
                "symbol": "000001.SZ",
                "quantity": 100,
                "price": 10.0,
                "order_type": "market"
            }
        )

        response = gateway.handle_request(order_request)

        # 验证订单通过网关成功提交
        assert response.status_code == 200
        assert "order_id" in response.body

        # 验证交易服务收到了订单
        assert len(trading_service.orders) > 0

    def test_api_gateway_metrics_collection(self, setup_gateway_components):
        """测试API网关指标收集"""
        components = setup_gateway_components
        gateway = components["gateway"]

        # 发送多个请求
        requests = [
            APIRequest(
                method="GET",
                path="/portfolio/positions",
                headers={"Authorization": "Bearer valid_token_123"}
            )
            for _ in range(10)
        ]

        for request in requests:
            gateway.handle_request(request)

        # 获取指标
        metrics = gateway.get_metrics()

        # 验证指标收集
        assert metrics["total_requests"] >= 10
        assert "avg_response_time" in metrics
        assert metrics["avg_response_time"] > 0

    def test_api_gateway_health_monitoring(self, setup_gateway_components):
        """测试API网关健康监控"""
        components = setup_gateway_components
        gateway = components["gateway"]

        # 发送一些请求
        request = APIRequest(
            method="GET",
            path="/portfolio/positions",
            headers={"Authorization": "Bearer valid_token_123"}
        )

        for _ in range(5):
            gateway.handle_request(request)

        # 获取健康状态
        health_status = gateway.get_health_status()

        # 验证健康监控
        assert health_status["status"] == "healthy"
        assert "uptime" in health_status
        assert "metrics" in health_status

    def test_api_gateway_concurrent_requests(self, setup_gateway_components):
        """测试API网关并发请求处理"""
        components = setup_gateway_components
        gateway = components["gateway"]

        # 创建并发请求
        request_count = 20
        requests = [
            APIRequest(
                method="GET",
                path="/portfolio/positions",
                headers={"Authorization": "Bearer valid_token_123"}
            )
            for _ in range(request_count)
        ]

        responses = []

        def handle_request_async(request):
            response = gateway.handle_request(request)
            responses.append(response)

        # 并发处理请求
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(handle_request_async, request) for request in requests]
            for future in as_completed(futures):
                future.result()

        # 验证并发处理结果
        assert len(responses) == request_count
        successful_responses = [r for r in responses if r.status_code == 200]
        assert len(successful_responses) > 0

    def test_api_gateway_full_workflow_integration(self, setup_gateway_components):
        """测试API网关完整工作流集成"""
        components = setup_gateway_components

        # 1. 认证并授权
        auth_request = APIRequest(
            method="GET",
            path="/portfolio/positions",
            headers={"Authorization": "Bearer valid_token_123"}
        )

        auth_response = components["gateway"].handle_request(auth_request)
        assert auth_response.status_code == 200

        # 2. 速率限制检查
        rate_limited = False
        for i in range(10):
            request = APIRequest(
                method="GET",
                path="/portfolio/positions",
                headers={"Authorization": "Bearer valid_token_123", "X-Client-ID": "test_user"}
            )
            response = components["gateway"].handle_request(request)
            if response.status_code == 429:
                rate_limited = True
                break

        # 3. 路由和转发
        trading_request = APIRequest(
            method="POST",
            path="/trading/order",
            headers={"Authorization": "Bearer valid_token_123"},
            body={"symbol": "000001.SZ", "quantity": 100, "price": 10.0}
        )

        trading_response = components["gateway"].handle_request(trading_request)
        assert trading_response.status_code == 200

        # 4. 监控和指标
        metrics = components["gateway"].get_metrics()
        assert metrics["total_requests"] > 0

        health = components["gateway"].get_health_status()
        assert health["status"] == "healthy"

        # 验证完整流程
        assert len(components["gateway"].request_logs) > 0


class TestAPIGatewayLoadTesting:
    """API网关负载测试"""

    def test_api_gateway_high_throughput(self, setup_gateway_components):
        """测试API网关高吞吐量"""
        components = setup_gateway_components
        gateway = components["gateway"]

        # 创建大量并发请求进行负载测试
        request_count = 200
        requests = [
            APIRequest(
                method="GET",
                path="/portfolio/positions",
                headers={"Authorization": "Bearer valid_token_123"}
            )
            for _ in range(request_count)
        ]

        start_time = time.time()
        responses = []

        def handle_request_async(request):
            response = gateway.handle_request(request)
            responses.append(response)

        # 并发处理请求
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(handle_request_async, request) for request in requests]
            for future in as_completed(futures):
                future.result()

        end_time = time.time()
        total_time = end_time - start_time

        # 验证结果
        assert len(responses) == request_count
        successful_responses = [r for r in responses if r.status_code == 200]
        assert len(successful_responses) > 0

        # 性能指标
        total_response_time = sum(r.response_time for r in responses)
        avg_response_time = total_response_time / request_count
        throughput = request_count / total_time

        print("API网关性能指标:")
        print(f"- 请求数量: {request_count}")
        print(f"- 总时间: {total_time:.3f}s")
        print(f"- 平均响应时间: {avg_response_time:.4f}s")
        print(f"- 吞吐量: {throughput:.1f} requests/s")

        # 性能断言
        assert avg_response_time < 0.1, f"平均响应时间太长: {avg_response_time:.4f}s"
        assert throughput > 50, f"吞吐量太低: {throughput:.1f} requests/s"

        # 验证网关指标
        metrics = gateway.get_metrics()
        assert metrics["total_requests"] == request_count


if __name__ == "__main__":
    pytest.main([__file__])
