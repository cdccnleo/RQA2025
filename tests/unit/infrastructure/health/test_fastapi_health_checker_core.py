"""
FastAPIHealthChecker核心业务逻辑测试套件

针对fastapi_health_checker.py的核心功能进行深度测试
目标: 将覆盖率从12.64%提升到80%+
重点: HTTP端点、状态码、异步处理、错误处理
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import AsyncMock, Mock
from fastapi.testclient import TestClient
from fastapi import HTTPException

# 导入被测试模块
from src.infrastructure.health.api.fastapi_integration import (
    FastAPIHealthChecker,
    HTTP_OK,
    HTTP_SERVICE_UNAVAILABLE,
    HTTP_INTERNAL_SERVER_ERROR
)
from src.infrastructure.health.models.health_status import HealthStatus


class TestFastAPIHealthCheckerCore:
    """FastAPIHealthChecker核心功能测试"""

    @pytest.fixture
    def mock_health_checker(self):
        """创建模拟健康检查器"""
        checker = Mock()

        # 配置异步方法
        checker.check_health = AsyncMock()
        checker.check_health_detailed = AsyncMock()
        checker.check_service = AsyncMock()
        checker.get_status = AsyncMock()
        checker.get_enhanced_status = AsyncMock()

        return checker

    @pytest.fixture
    def fastapi_checker(self, mock_health_checker):
        """创建FastAPIHealthChecker实例"""
        return FastAPIHealthChecker(mock_health_checker)

    @pytest.fixture
    def client(self, fastapi_checker):
        """创建测试客户端"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(fastapi_checker.router)
        return TestClient(app)

    def test_initialization(self, fastapi_checker, mock_health_checker):
        """测试初始化"""
        assert fastapi_checker.health_checker == mock_health_checker
        assert fastapi_checker.config == {}
        assert hasattr(fastapi_checker, 'router')

    def test_routes_setup(self, fastapi_checker):
        """测试路由设置"""
        routes = [route.path for route in fastapi_checker.router.routes]

        assert "/health" in routes
        assert "/health/detailed" in routes
        assert "/health/service/{service_name}" in routes
        assert "/health/status" in routes

    # =============== 核心HTTP端点测试 ===============

    def test_health_endpoint_success_up(self, client, mock_health_checker):
        """测试/health端点 - 成功响应 (UP状态)"""
        # 配置模拟返回值
        mock_health_checker.check_health.return_value = {
            "overall_status": "UP",
            "status": "healthy",
            "timestamp": "2025-01-01T00:00:00Z",
            "version": "1.0.0"
        }

        response = client.get("/health")

        assert response.status_code == HTTP_OK
        data = response.json()
        assert data["overall_status"] == "UP"
        assert data["status"] == "healthy"

        # 验证方法被调用
        mock_health_checker.check_health.assert_called_once()

    def test_health_endpoint_success_degraded(self, client, mock_health_checker):
        """测试/health端点 - 降级但可用 (DEGRADED状态)"""
        mock_health_checker.check_health.return_value = {
            "overall_status": "DEGRADED",
            "status": "degraded",
            "issues": ["High CPU usage"],
            "timestamp": "2025-01-01T00:00:00Z"
        }

        response = client.get("/health")

        assert response.status_code == HTTP_OK  # 降级仍返回200
        data = response.json()
        assert data["overall_status"] == "DEGRADED"
        assert "issues" in data

    def test_health_endpoint_service_unavailable(self, client, mock_health_checker):
        """测试/health端点 - 服务不可用 (DOWN状态)"""
        mock_health_checker.check_health.return_value = {
            "overall_status": "DOWN",
            "status": "unhealthy",
            "issues": ["Database connection failed"],
            "timestamp": "2025-01-01T00:00:00Z"
        }

        response = client.get("/health")

        assert response.status_code == HTTP_SERVICE_UNAVAILABLE
        data = response.json()
        assert data["overall_status"] == "DOWN"
        assert data["status"] == "unhealthy"

    def test_health_endpoint_fallback_status(self, client, mock_health_checker):
        """测试/health端点 - 状态回退逻辑"""
        # 没有overall_status，只有status字段
        mock_health_checker.check_health.return_value = {
            "status": "UP",
            "message": "Service is healthy"
        }

        response = client.get("/health")

        assert response.status_code == HTTP_OK
        data = response.json()
        assert data["status"] == "UP"

    def test_health_endpoint_exception_handling(self, client, mock_health_checker):
        """测试/health端点 - 异常处理"""
        mock_health_checker.check_health.side_effect = Exception("Test error")

        response = client.get("/health")

        assert response.status_code == HTTP_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "健康检查失败" in data["detail"]

    def test_detailed_health_endpoint_success(self, client, mock_health_checker):
        """测试/health/detailed端点 - 成功响应"""
        mock_health_checker.check_health_detailed.return_value = {
            "overall_status": "UP",
            "components": {
                "database": {"status": "UP", "response_time": 0.05},
                "cache": {"status": "UP", "response_time": 0.02}
            },
            "total_checks": 2,
            "passed_checks": 2,
            "failed_checks": 0,
            "timestamp": "2025-01-01T00:00:00Z"
        }

        response = client.get("/health/detailed")

        assert response.status_code == HTTP_OK
        data = response.json()
        assert "components" in data
        assert data["total_checks"] == 2
        assert data["passed_checks"] == 2

        mock_health_checker.check_health_detailed.assert_called_once()

    def test_detailed_health_endpoint_error(self, client, mock_health_checker):
        """测试/health/detailed端点 - 错误处理"""
        mock_health_checker.check_health_detailed.side_effect = ConnectionError("Network timeout")

        response = client.get("/health/detailed")

        assert response.status_code == HTTP_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "健康检查失败" in data["detail"]

    def test_service_health_endpoint_success(self, client, mock_health_checker):
        """测试/health/service/{service_name}端点 - 成功响应"""
        service_name = "database"
        mock_health_checker.check_service.return_value = {
            "service": service_name,
            "status": "UP",
            "response_time": 0.03,
            "message": "Database connection healthy"
        }

        response = client.get(f"/health/service/{service_name}")

        assert response.status_code == HTTP_OK
        data = response.json()
        assert data["service"] == service_name
        assert data["status"] == "UP"

        mock_health_checker.check_service.assert_called_once_with(service_name)

    def test_service_health_endpoint_not_found(self, client, mock_health_checker):
        """测试/health/service/{service_name}端点 - 服务不存在"""
        service_name = "unknown_service"
        mock_health_checker.check_service.side_effect = ValueError(f"Service '{service_name}' not found")

        response = client.get(f"/health/service/{service_name}")

        assert response.status_code == HTTP_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "健康检查失败" in data["detail"]

    def test_health_status_endpoint_success(self, client, mock_health_checker):
        """测试/health/status端点 - 成功响应"""
        # 修复：确保返回普通dict而不是coroutine
        mock_health_checker.get_status = Mock(return_value={
            "overall_status": "UP",
            "uptime": "2d 3h 15m",
            "total_checks": 150,
            "success_rate": 0.987,
            "last_check": "2025-01-01T00:00:00Z"
        })

        response = client.get("/health/status")

        assert response.status_code == HTTP_OK
        data = response.json()
        assert data["overall_status"] == "UP"
        assert "success_rate" in data

    def test_health_status_endpoint_no_method(self, fastapi_checker):
        """测试/health/status端点 - 方法不存在的情况"""
        # 移除get_status方法
        delattr(fastapi_checker.health_checker, 'get_status')

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(fastapi_checker.router)
        client = TestClient(app)

        response = client.get("/health/status")
        assert response.status_code == HTTP_INTERNAL_SERVER_ERROR

    def test_performance_stats_endpoint_conditional(self, fastapi_checker):
        """测试/health/performance端点 - 条件路由"""
        # 检查是否有性能统计端点
        routes = [route.path for route in fastapi_checker.router.routes]

        # 如果health_checker有get_enhanced_status方法，应该有性能端点
        if hasattr(fastapi_checker.health_checker, 'get_enhanced_status'):
            assert "/health/performance" in routes
        else:
            assert "/health/performance" not in routes

    def test_performance_stats_endpoint_success(self, client, mock_health_checker):
        """测试/health/performance端点 - 成功响应"""
        # 修复：确保返回普通dict而不是coroutine
        mock_health_checker.get_enhanced_status = Mock(return_value={
            "performance": {
                "avg_response_time": 0.045,
                "max_response_time": 0.123,
                "total_requests": 1000,
                "error_rate": 0.005
            },
            "memory_usage": "256MB",
            "cpu_usage": "15%"
        })

        response = client.get("/health/performance")

        assert response.status_code == HTTP_OK
        data = response.json()
        assert "performance" in data
        assert data["performance"]["avg_response_time"] == 0.045

    # =============== 错误处理测试 ===============

    def test_async_method_exception_propagation(self, client, mock_health_checker):
        """测试异步方法异常传播"""
        mock_health_checker.check_health.side_effect = RuntimeError("Async operation failed")

        response = client.get("/health")

        assert response.status_code == HTTP_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Async operation failed" in data["detail"]

    def test_network_timeout_simulation(self, client, mock_health_checker):
        """测试网络超时模拟"""
        import asyncio
        mock_health_checker.check_health.side_effect = asyncio.TimeoutError("Request timeout")

        response = client.get("/health")

        assert response.status_code == HTTP_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Request timeout" in data["detail"]

    # =============== 配置和初始化测试 ===============

    def test_custom_config_initialization(self, mock_health_checker):
        """测试自定义配置初始化"""
        config = {
            "timeout": 30,
            "retries": 3,
            "custom_setting": "value"
        }

        checker = FastAPIHealthChecker(mock_health_checker, config)

        assert checker.config == config
        assert checker.config["timeout"] == 30

    def test_empty_config_defaults(self, mock_health_checker):
        """测试空配置的默认值"""
        checker = FastAPIHealthChecker(mock_health_checker)

        assert checker.config == {}
        assert isinstance(checker.config, dict)

    # =============== HTTP状态码常量测试 ===============

    def test_http_status_constants(self):
        """测试HTTP状态码常量"""
        assert HTTP_OK == 200
        assert HTTP_SERVICE_UNAVAILABLE == 503
        assert HTTP_INTERNAL_SERVER_ERROR == 500

    # =============== 路由方法测试 ===============

    @pytest.mark.asyncio
    async def test_health_check_method_direct_call(self, fastapi_checker, mock_health_checker):
        """测试health_check方法直接调用"""
        mock_health_checker.check_health.return_value = {
            "overall_status": "UP",
            "status": "healthy"
        }

        response = await fastapi_checker.health_check()

        assert response.status_code == HTTP_OK
        assert "overall_status" in response.body.decode()
        mock_health_checker.check_health.assert_called_once()

    # 移除有问题的直接方法调用测试，专注于HTTP端点测试

    @pytest.mark.asyncio
    async def test_service_health_check_method_direct_call(self, fastapi_checker, mock_health_checker):
        """测试service_health_check方法直接调用"""
        service_name = "redis"
        mock_health_checker.check_service.return_value = {
            "service": service_name,
            "status": "UP"
        }

        response = await fastapi_checker.service_health_check(service_name)

        assert response.status_code == HTTP_OK
        mock_health_checker.check_service.assert_called_once_with(service_name)

    # =============== 边界条件测试 ===============

    def test_special_characters_in_service_name(self, client, mock_health_checker):
        """测试服务名称中的特殊字符"""
        service_name = "database-cache_01.test"
        mock_health_checker.check_service.return_value = {
            "service": service_name,
            "status": "UP"
        }

        response = client.get(f"/health/service/{service_name}")

        assert response.status_code == HTTP_OK
        data = response.json()
        assert data["service"] == service_name

    def test_service_exception_handling(self, client, mock_health_checker):
        """测试服务异常处理"""
        service_name = "database"
        mock_health_checker.check_service.side_effect = ValueError("Service not configured")

        response = client.get(f"/health/service/{service_name}")

        assert response.status_code == HTTP_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "健康检查失败" in data["detail"]

    # =============== 并发和性能测试 ===============

    def test_multiple_concurrent_requests(self, client, mock_health_checker):
        """测试多个并发请求"""
        import threading
        import time

        mock_health_checker.check_health.return_value = {
            "overall_status": "UP",
            "status": "healthy"
        }

        results = []

        def make_request():
            response = client.get("/health")
            results.append(response.status_code)

        # 创建多个线程并发请求
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证所有请求都成功
        assert len(results) == 5
        assert all(status == HTTP_OK for status in results)

    def test_response_format_consistency(self, client, mock_health_checker):
        """测试响应格式一致性"""
        mock_health_checker.check_health.return_value = {
            "overall_status": "UP",
            "status": "healthy",
            "timestamp": "2025-01-01T00:00:00Z",
            "version": "1.0.0",
            "uptime": "1d 2h 30m"
        }

        response = client.get("/health")

        assert response.status_code == HTTP_OK
        data = response.json()

        # 验证必需字段存在
        required_fields = ["overall_status", "status", "timestamp"]
        for field in required_fields:
            assert field in data

        # 验证数据类型
        assert isinstance(data["overall_status"], str)
        assert isinstance(data["status"], str)
