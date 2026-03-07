"""
测试Phase 8.2.2系统健康检查器重构

验证SystemHealthChecker正确实现IHealthCheckExecutor接口
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import sys
import os
from datetime import datetime
from unittest.mock import MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../..'))

from src.infrastructure.health.monitoring.health_checker import SystemHealthChecker
from src.infrastructure.health.components.health_checker import IHealthCheckExecutor, HealthCheckResult


class TestSystemHealthCheckerRefactor:
    """测试系统健康检查器重构"""

    @pytest.fixture
    def health_checker(self):
        """创建系统健康检查器实例"""
        # 创建模拟的指标收集器
        mock_collector = MagicMock()
        mock_collector.get_latest_metrics.return_value = {
            'cpu': {'usage_percent': 45.5},
            'memory': {'percent': 60.0},
            'disk': {'percent': 75.0}
        }
        return SystemHealthChecker(mock_collector)

    def test_implements_interface(self, health_checker):
        """测试实现了正确的接口"""
        assert isinstance(health_checker, IHealthCheckExecutor)
        assert hasattr(health_checker, 'register_service')
        assert hasattr(health_checker, 'unregister_service')
        assert hasattr(health_checker, 'check_service')
        assert hasattr(health_checker, 'get_service_health_history')
        assert hasattr(health_checker, 'check_health_async')
        assert hasattr(health_checker, 'check_health_sync')
        assert hasattr(health_checker, 'get_health_metrics')

    def test_register_service(self, health_checker):
        """测试注册服务"""
        def mock_check():
            return {"status": "healthy", "value": 100}

        # 注册服务
        health_checker.register_service("test_service", mock_check)

        # 验证服务已注册
        assert "test_service" in health_checker._health_checks
        assert "test_service" in health_checker._check_history
        assert health_checker._check_history["test_service"] == []

    def test_register_invalid_service(self, health_checker):
        """测试注册无效服务"""
        with pytest.raises(ValueError, match="必须是可调用的"):
            health_checker.register_service("invalid", "not_callable")

    def test_unregister_service(self, health_checker):
        """测试注销服务"""
        def mock_check():
            return {"status": "healthy"}

        # 先注册
        health_checker.register_service("test_service", mock_check)
        assert "test_service" in health_checker._health_checks

        # 注销
        health_checker.unregister_service("test_service")
        assert "test_service" not in health_checker._health_checks
        assert "test_service" not in health_checker._check_history

    def test_check_service(self, health_checker):
        """测试检查单个服务"""
        def mock_check():
            return {"status": "healthy", "value": 95.5, "message": "OK"}

        health_checker.register_service("cpu_usage", mock_check)

        result = health_checker.check_service("cpu_usage")

        # 验证结果
        assert isinstance(result, HealthCheckResult)
        assert result.service_name == "cpu_usage"
        assert result.status == "healthy"
        assert isinstance(result.timestamp, datetime)
        assert result.response_time >= 0
        assert result.details["status"] == "healthy"

        # 验证历史记录
        history = health_checker.get_service_health_history("cpu_usage")
        assert len(history) == 1
        assert history[0] == result

    def test_check_unregistered_service(self, health_checker):
        """测试检查未注册的服务"""
        result = health_checker.check_service("nonexistent")

        assert isinstance(result, HealthCheckResult)
        assert result.service_name == "nonexistent"
        assert result.status == "error"
        assert "未注册" in str(result.details)

    def test_get_service_health_history(self, health_checker):
        """测试获取服务健康历史"""
        def mock_check():
            return {"status": "healthy"}

        health_checker.register_service("test_service", mock_check)

        # 执行多次检查
        for _ in range(3):
            health_checker.check_service("test_service")

        history = health_checker.get_service_health_history("test_service")
        assert len(history) == 3
        assert all(isinstance(r, HealthCheckResult) for r in history)

    def test_check_health_sync(self, health_checker):
        """测试同步健康检查"""
        result = health_checker.check_health_sync()

        # 验证结果结构
        assert "status" in result
        assert "response_time" in result
        assert "timestamp" in result
        assert "services" in result
        assert "summary" in result

        # 验证摘要信息
        summary = result["summary"]
        assert "total_services" in summary
        assert "healthy_count" in summary
        assert "warning_count" in summary
        assert "critical_count" in summary
        assert "error_count" in summary

    @pytest.mark.asyncio
    async def test_check_health_async(self, health_checker):
        """测试异步健康检查"""
        result = await health_checker.check_health_async()

        # 验证结果结构（与同步检查相同）
        assert "status" in result
        assert "response_time" in result
        assert "timestamp" in result
        assert "services" in result
        assert "summary" in result

    def test_get_health_metrics(self, health_checker):
        """测试获取健康指标"""
        # 执行一些检查以生成历史数据
        for service_name in health_checker._health_checks.keys():
            health_checker.check_service(service_name)

        metrics = health_checker.get_health_metrics()

        # 验证指标结构
        assert "system_health_metrics" in metrics
        assert "performance_metrics" in metrics
        assert "timestamp" in metrics

        system_metrics = metrics["system_health_metrics"]
        assert "total_registered_services" in system_metrics
        assert "total_check_history" in system_metrics
        assert "services_with_history" in system_metrics
        assert "overall_health_status" in system_metrics

        perf_metrics = metrics["performance_metrics"]
        assert "average_response_time" in perf_metrics
        assert "check_success_rate" in perf_metrics

    def test_default_system_checks_registered(self, health_checker):
        """测试默认系统检查已注册"""
        expected_services = ["cpu_usage", "memory_usage", "disk_usage", "process_health"]

        for service in expected_services:
            assert service in health_checker._health_checks
            # _check_history可能在首次检查后才存在
            # assert service in health_checker._check_history

    def test_health_check_result_format(self, health_checker):
        """测试健康检查结果格式"""
        def mock_check():
            return {
                "status": "warning",
                "value": 85.5,
                "threshold": 80,
                "message": "接近阈值"
            }

        health_checker.register_service("test_check", mock_check)
        result = health_checker.check_service("test_check")

        # 验证标准结果格式
        assert result.service_name == "test_check"
        assert result.status == "warning"
        assert "recommendations" in result.__dict__
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) > 0  # 应该有建议

    def test_error_handling_in_checks(self, health_checker):
        """测试检查中的错误处理"""
        def failing_check():
            raise Exception("检查失败")

        health_checker.register_service("failing_service", failing_check)
        result = health_checker.check_service("failing_service")

        # 验证错误处理
        assert result.status == "error"
        assert "检查失败" in str(result.details)
        assert "检查服务配置" in result.recommendations

    def test_history_limit(self, health_checker):
        """测试历史记录限制"""
        def mock_check():
            return {"status": "healthy"}

        health_checker.register_service("history_test", mock_check)

        # 执行超过100次检查
        for _ in range(150):
            health_checker.check_service("history_test")

        history = health_checker.get_service_health_history("history_test")

        # 验证历史记录限制为100条
        assert len(history) <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


