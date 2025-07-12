import unittest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from src.infrastructure.web import app_factory
from src.infrastructure.error import ErrorHandler

class TestAppFactory(unittest.TestCase):
    """Web服务模块单元测试"""

    def setUp(self):
        self.test_config = {
            "app": {
                "title": "Test API",
                "version": "1.0.0"
            },
            "middleware": {
                "enable_cors": True,
                "enable_https_redirect": False
            }
        }

    def test_app_creation(self):
        """测试应用工厂创建应用"""
        app = app_factory.create_app(self.test_config)

        self.assertIsInstance(app, FastAPI)
        self.assertEqual(app.title, "Test API")
        self.assertEqual(app.version, "1.0.0")

    def test_default_routes(self):
        """测试默认路由注册"""
        app = app_factory.create_app(self.test_config)
        client = TestClient(app)

        # 测试健康检查端点
        response = client.get("/health")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json(), {"status": "healthy"})

        # 测试版本端点
        response = client.get("/version")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json(), {"version": "1.0.0"})

    @patch('src.infrastructure.web.app_factory.ErrorHandler')
    def test_error_handling(self, mock_error_handler):
        """测试错误处理集成"""
        # 设置模拟错误处理器
        mock_handler = MagicMock()
        mock_error_handler.return_value = mock_handler

        # 创建测试应用并模拟错误路由
        app = app_factory.create_app(self.test_config)

        @app.get("/error")
        def raise_error():
            raise ValueError("Test error")

        client = TestClient(app)
        response = client.get("/error")

        # 验证错误处理被调用
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        mock_handler.handle.assert_called_once()

    @patch('src.infrastructure.monitoring.ApplicationMonitor')
    def test_performance_monitoring(self, mock_monitor):
        """测试性能监控集成"""
        # 设置模拟监控器
        mock_monitor_instance = MagicMock()
        mock_monitor.return_value = mock_monitor_instance

        # 创建测试应用
        app = app_factory.create_app(self.test_config)

        @app.get("/test")
        def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)
        response = client.get("/test")

        # 验证监控器被调用
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_monitor_instance.record_metric.assert_called()

    def test_cors_middleware(self):
        """测试CORS中间件"""
        # 启用CORS配置
        self.test_config["middleware"]["enable_cors"] = True
        app = app_factory.create_app(self.test_config)
        client = TestClient(app)

        # 测试OPTIONS预检请求
        response = client.options(
            "/health",
            headers={
                "Origin": "http://test.com",
                "Access-Control-Request-Method": "GET"
            }
        )

        # 验证CORS头
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("access-control-allow-origin", response.headers)

    def test_route_registration(self):
        """测试自定义路由注册"""
        # 创建测试路由
        def test_router(app: FastAPI):
            @app.get("/custom")
            def custom_endpoint():
                return {"custom": "route"}

        # 注册路由并创建应用
        app_factory.register_router(test_router)
        app = app_factory.create_app(self.test_config)
        client = TestClient(app)

        # 测试自定义路由
        response = client.get("/custom")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json(), {"custom": "route"})

if __name__ == '__main__':
    unittest.main()
