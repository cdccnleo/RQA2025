"""
FastAPI健康检查器简单测试套件

针对fastapi_health_checker.py的基本功能进行测试
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestFastAPIHealthCheckerSimple:
    """FastAPI健康检查器简单测试"""

    def test_constants(self):
        """测试常量定义"""
        from src.infrastructure.health.api.fastapi_integration import (
            HTTP_OK, HTTP_SERVICE_UNAVAILABLE, HTTP_INTERNAL_SERVER_ERROR, HTTP_NOT_IMPLEMENTED
        )

        assert HTTP_OK == 200
        assert HTTP_SERVICE_UNAVAILABLE == 503
        assert HTTP_INTERNAL_SERVER_ERROR == 500
        assert HTTP_NOT_IMPLEMENTED == 501

    def test_get_router_function(self):
        """测试get_router模块函数"""
        from src.infrastructure.health.api.fastapi_integration import get_router

        # 创建mock health checker
        mock_checker = Mock()
        mock_checker.check_health = AsyncMock(return_value={'status': 'UP'})

        router = get_router(mock_checker)
        assert router is not None

        # 检查路由
        routes = [route.path for route in router.routes]
        expected_routes = ['/health', '/health/detailed', '/health/service/{service_name}', '/health/status']
        for route in expected_routes:
            assert route in routes

    def test_include_in_app_function(self):
        """测试include_in_app模块函数"""
        from src.infrastructure.health.api.fastapi_integration import include_in_app

        # 创建mock app和health checker
        app = FastAPI()
        mock_checker = Mock()
        mock_checker.check_health = AsyncMock(return_value={'status': 'UP'})

        # 包含路由
        include_in_app(app, mock_checker)

        # 检查路由是否被添加
        routes = [route.path for route in app.routes]
        expected_routes = ['/health', '/health/detailed', '/health/service/{service_name}', '/health/status']
        for route in expected_routes:
            assert route in routes

    def test_fastapi_integration_basic(self):
        """测试基本的FastAPI集成"""
        from src.infrastructure.health.api.fastapi_integration import get_router

        # 创建mock health checker
        mock_checker = Mock()
        mock_checker.check_health = AsyncMock(return_value={
            'status': 'UP',
            'overall_status': 'UP',
            'timestamp': '2024-01-01T00:00:00',
            'response_time': 0.1
        })

        # 获取路由
        router = get_router(mock_checker)

        # 创建测试app
        app = FastAPI()
        app.include_router(router)

        # 创建测试客户端
        client = TestClient(app)

        # 测试基本健康检查端点
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data or 'overall_status' in data

    def test_fastapi_integration_detailed(self):
        """测试详细健康检查端点"""
        from src.infrastructure.health.api.fastapi_integration import get_router

        # 创建mock health checker
        mock_checker = Mock()
        mock_checker.check_health_detailed = AsyncMock(return_value={
            'status': 'healthy',
            'database': {'status': 'healthy'},
            'cache': {'status': 'healthy'},
            'system': {'status': 'healthy'}
        })

        # 获取路由
        router = get_router(mock_checker)

        # 创建测试app
        app = FastAPI()
        app.include_router(router)

        # 创建测试客户端
        client = TestClient(app)

        # 测试详细健康检查端点
        response = client.get('/health/detailed')
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data

    def test_fastapi_integration_service(self):
        """测试服务特定健康检查端点"""
        from src.infrastructure.health.api.fastapi_integration import get_router

        # 创建mock health checker
        mock_checker = Mock()
        mock_checker.check_service = AsyncMock(return_value={
            'status': 'UP',
            'service': 'test_service',
            'response_time': 0.05
        })

        # 获取路由
        router = get_router(mock_checker)

        # 创建测试app
        app = FastAPI()
        app.include_router(router)

        # 创建测试客户端
        client = TestClient(app)

        # 测试服务健康检查端点
        response = client.get('/health/service/database')
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data

    def test_error_handling_invalid_service(self):
        """测试无效服务名的错误处理"""
        from src.infrastructure.health.api.fastapi_integration import get_router

        # 创建mock health checker - 抛出异常
        mock_checker = Mock()
        mock_checker.check_service = AsyncMock(side_effect=ValueError("Service not found"))

        # 获取路由
        router = get_router(mock_checker)

        # 创建测试app
        app = FastAPI()
        app.include_router(router)

        # 创建测试客户端
        client = TestClient(app)

        # 测试错误处理
        response = client.get('/health/service/invalid_service')
        assert response.status_code == 500  # 应该返回内部服务器错误

    def test_module_level_functions(self):
        """测试模块级函数"""
        from src.infrastructure.health.api.fastapi_integration import get_router, include_in_app

        assert callable(get_router)
        assert callable(include_in_app)

        # 测试函数签名
        import inspect
        get_router_sig = inspect.signature(get_router)
        include_in_app_sig = inspect.signature(include_in_app)

        assert 'health_checker' in get_router_sig.parameters
        assert 'app' in include_in_app_sig.parameters
        assert 'health_checker' in include_in_app_sig.parameters
