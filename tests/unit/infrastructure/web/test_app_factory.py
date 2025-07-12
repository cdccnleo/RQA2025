import pytest
from fastapi.testclient import TestClient
from src.infrastructure.web import app_factory
from unittest.mock import MagicMock, patch

class TestAppFactory:
    @pytest.fixture
    def test_client(self):
        """提供测试用的FastAPI客户端"""
        app = app_factory.create_app()
        return TestClient(app)

    def test_root_endpoint(self, test_client):
        """测试根端点"""
        response = test_client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "RQA2025 API"}

    def test_health_check(self, test_client):
        """测试健康检查端点"""
        with patch('src.infrastructure.monitoring.ApplicationMonitor') as mock_monitor:
            mock_monitor.return_value.run_health_checks.return_value = {"db": True}
            response = test_client.get("/health")

            assert response.status_code == 200
            assert response.json() == {"status": "healthy", "checks": {"db": True}}
            mock_monitor.return_value.run_health_checks.assert_called_once()

    def test_metrics_endpoint(self, test_client):
        """测试指标端点"""
        with patch('src.infrastructure.monitoring.ApplicationMonitor') as mock_monitor:
            mock_monitor.return_value.get_metrics.return_value = {"cpu_usage": 30}
            response = test_client.get("/metrics")

            assert response.status_code == 200
            assert response.json() == {"cpu_usage": 30}
            mock_monitor.return_value.get_metrics.assert_called_once()

    def test_request_monitoring(self, test_client):
        """测试请求监控中间件"""
        with patch('src.infrastructure.monitoring.ApplicationMonitor') as mock_monitor:
            test_client.get("/")

            # 验证监控器是否记录了请求
            mock_monitor.return_value.record_metric.assert_called()
            args, _ = mock_monitor.return_value.record_metric.call_args
            assert "request_duration" in args[0]

    def test_error_handling(self, test_client):
        """测试错误处理中间件"""
        # 模拟路由抛出异常
        @test_client.app.get("/error")
        def raise_error():
            raise ValueError("test error")

        with patch('src.infrastructure.error.ErrorHandler') as mock_handler:
            response = test_client.get("/error")

            assert response.status_code == 500
            assert "test error" in response.text
            mock_handler.return_value.handle.assert_called_once()
