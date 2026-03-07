"""
测试 basic_health_checker 模块 - 提升零覆盖率

专注于测试 BasicHealthChecker 类的所有核心功能。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from src.infrastructure.health.monitoring.basic_health_checker import (
    BasicHealthChecker,
    ServiceHealthProfile,
    IHealthChecker
)
from src.infrastructure.health.models.health_status import HealthStatus


class TestServiceHealthProfile:
    """测试服务健康档案"""

    def test_default_values(self):
        """测试默认值"""
        profile = ServiceHealthProfile(name="test_service")
        assert profile.name == "test_service"
        assert profile.check_count == 0
        assert profile.success_count == 0
        assert profile.failure_count == 0
        assert profile.last_check_time is None
        assert profile.average_response_time == 0.0
        assert profile.status == "unknown"

    def test_custom_values(self):
        """测试自定义值"""
        now = datetime.now()
        profile = ServiceHealthProfile(
            name="custom_service",
            check_count=10,
            success_count=8,
            failure_count=2,
            last_check_time=now,
            average_response_time=0.5,
            status="healthy"
        )
        assert profile.name == "custom_service"
        assert profile.check_count == 10
        assert profile.success_count == 8
        assert profile.failure_count == 2
        assert profile.last_check_time == now
        assert profile.average_response_time == 0.5
        assert profile.status == "healthy"


class TestBasicHealthChecker:
    """测试基础健康检查器"""

    def test_initialization_default(self):
        """测试默认初始化"""
        checker = BasicHealthChecker()
        assert checker.config == {}
        assert len(checker._checkers) == 0
        assert len(checker._services) == 0

    def test_initialization_with_config(self):
        """测试带配置的初始化"""
        config = {"timeout": 10, "retries": 3}
        checker = BasicHealthChecker(config=config)
        assert checker.config == config

    def test_register_service_success(self):
        """测试成功注册服务"""
        checker = BasicHealthChecker()
        check_func = lambda: True
        
        checker.register_service("test_service", check_func)
        
        assert "test_service" in checker._checkers
        assert "test_service" in checker._services
        assert checker._checkers["test_service"] == check_func

    def test_register_service_not_callable_raises_error(self):
        """测试注册非可调用对象抛出错误"""
        checker = BasicHealthChecker()
        
        with pytest.raises(ValueError, match="must be callable"):
            checker.register_service("test_service", "not_callable")

    def test_register_multiple_services(self):
        """测试注册多个服务"""
        checker = BasicHealthChecker()
        
        checker.register_service("service1", lambda: True)
        checker.register_service("service2", lambda: False)
        checker.register_service("service3", lambda: True)
        
        assert len(checker._checkers) == 3
        assert "service1" in checker._checkers
        assert "service2" in checker._checkers
        assert "service3" in checker._checkers

    def test_unregister_service_success(self):
        """测试成功注销服务"""
        checker = BasicHealthChecker()
        checker.register_service("test_service", lambda: True)
        
        checker.unregister_service("test_service")
        
        assert "test_service" not in checker._checkers

    def test_unregister_nonexistent_service(self):
        """测试注销不存在的服务"""
        checker = BasicHealthChecker()
        # 应该不抛出错误
        checker.unregister_service("nonexistent_service")

    def test_check_service_healthy(self):
        """测试检查健康服务"""
        checker = BasicHealthChecker()
        checker.register_service("healthy_service", lambda: True)
        
        result = checker.check_service("healthy_service")
        
        assert result["status"] == "up"
        assert "response_time" in result
        assert "timestamp" in result
        assert "details" in result
        assert result["details"]["result"] is True

    def test_check_service_unhealthy(self):
        """测试检查不健康服务"""
        checker = BasicHealthChecker()
        checker.register_service("unhealthy_service", lambda: False)
        
        result = checker.check_service("unhealthy_service")
        
        assert result["status"] == "unhealthy"
        assert "response_time" in result
        assert "timestamp" in result

    def test_check_service_not_found(self):
        """测试检查不存在的服务"""
        checker = BasicHealthChecker()
        
        result = checker.check_service("nonexistent_service")
        
        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_check_service_with_exception(self):
        """测试检查服务时发生异常"""
        checker = BasicHealthChecker()
        
        def failing_check():
            raise RuntimeError("Service check failed")
        
        checker.register_service("failing_service", failing_check)
        
        result = checker.check_service("failing_service")
        
        assert result["status"] == "error"
        assert "Service check failed" in result["error"]

    def test_check_health_all_healthy(self):
        """测试所有服务健康"""
        checker = BasicHealthChecker()
        checker.register_service("service1", lambda: True)
        checker.register_service("service2", lambda: True)
        
        result = checker.check_health()
        
        assert result["overall_status"] == "healthy"
        assert len(result["services"]) == 2
        assert "timestamp" in result

    def test_check_health_some_unhealthy(self):
        """测试部分服务不健康（返回False的服务状态为unhealthy，不触发overall为unhealthy）"""
        checker = BasicHealthChecker()
        checker.register_service("healthy", lambda: True)
        checker.register_service("unhealthy", lambda: False)
        
        result = checker.check_health()
        
        # 注意：根据实际实现，只有error和down状态才会导致overall_status为unhealthy
        # unhealthy状态不会触发overall_status变化
        assert result["overall_status"] == "healthy"
        assert len(result["services"]) == 2
        assert result["services"]["unhealthy"]["status"] == "unhealthy"

    def test_check_health_with_errors(self):
        """测试带错误的健康检查"""
        checker = BasicHealthChecker()
        checker.register_service("healthy", lambda: True)
        
        def error_check():
            raise Exception("Error occurred")
        
        checker.register_service("error_service", error_check)
        
        result = checker.check_health()
        
        assert result["overall_status"] == "unhealthy"
        assert result["services"]["error_service"]["status"] == "error"

    def test_check_health_no_services(self):
        """测试没有注册服务时的健康检查"""
        checker = BasicHealthChecker()
        
        result = checker.check_health()
        
        assert result["overall_status"] == "healthy"
        assert len(result["services"]) == 0

    def test_generate_status_report(self):
        """测试生成状态报告"""
        checker = BasicHealthChecker()
        checker.register_service("service1", lambda: True)
        
        report = checker.generate_status_report()
        
        assert "overall_status" in report
        assert "services" in report
        assert "timestamp" in report

    def test_check_component(self):
        """测试检查组件"""
        checker = BasicHealthChecker()
        checker.register_service("component1", lambda: True)
        
        result = checker.check_component("component1")
        
        assert result["status"] == "up"

    def test_perform_health_check(self):
        """测试执行健康检查（兼容方法）"""
        checker = BasicHealthChecker()
        checker.register_service("service1", lambda: True)
        
        result = checker.perform_health_check()
        
        assert "healthy" in result
        assert result["healthy"] is True
        assert result["status"] == "healthy"
        assert "services" in result
        assert "timestamp" in result

    def test_perform_health_check_unhealthy(self):
        """测试执行健康检查（带错误服务触发unhealthy）"""
        checker = BasicHealthChecker()
        
        def error_check():
            raise Exception("Service down")
        
        checker.register_service("error_service", error_check)
        
        result = checker.perform_health_check()
        
        # 有error状态的服务会导致overall为unhealthy
        assert result["healthy"] is False
        assert result["status"] == "unhealthy"

    def test_validate_service_exists_true(self):
        """测试验证服务存在（存在）"""
        checker = BasicHealthChecker()
        checker.register_service("existing_service", lambda: True)
        
        assert checker._validate_service_exists("existing_service") is True

    def test_validate_service_exists_false(self):
        """测试验证服务存在（不存在）"""
        checker = BasicHealthChecker()
        
        assert checker._validate_service_exists("nonexistent") is False

    def test_execute_service_check(self):
        """测试执行服务检查"""
        checker = BasicHealthChecker()
        check_func = MagicMock(return_value=True)
        checker.register_service("test", check_func)
        
        result, response_time = checker._execute_service_check("test")
        
        assert result is True
        assert response_time >= 0
        check_func.assert_called_once()

    def test_create_success_check_result(self):
        """测试创建成功检查结果"""
        checker = BasicHealthChecker()
        checker.register_service("test", lambda: True)
        
        result = checker._create_success_check_result("test", True, 0.1)
        
        assert result["status"] == "up"
        assert result["response_time"] == 0.1
        assert "timestamp" in result
        assert result["details"]["result"] is True

    def test_create_error_check_result(self):
        """测试创建错误检查结果"""
        checker = BasicHealthChecker()
        checker.register_service("test", lambda: True)
        error = RuntimeError("Test error")
        
        result = checker._create_error_check_result("test", error)
        
        assert result["status"] == "error"
        assert "Test error" in result["error"]
        assert "timestamp" in result

    def test_response_time_measurement(self):
        """测试响应时间测量"""
        checker = BasicHealthChecker()
        
        def slow_check():
            import time
            time.sleep(0.01)  # 10ms 延迟
            return True
        
        checker.register_service("slow_service", slow_check)
        result = checker.check_service("slow_service")
        
        assert result["response_time"] >= 0.01

    def test_multiple_checks_on_same_service(self):
        """测试对同一服务进行多次检查"""
        checker = BasicHealthChecker()
        checker.register_service("test", lambda: True)
        
        result1 = checker.check_service("test")
        result2 = checker.check_service("test")
        
        assert result1["status"] == result2["status"]
        assert "timestamp" in result1
        assert "timestamp" in result2


class TestIHealthCheckerInterface:
    """测试健康检查器接口"""

    def test_interface_methods_exist(self):
        """测试接口方法存在"""
        assert hasattr(IHealthChecker, 'check_health')
        assert hasattr(IHealthChecker, 'register_service')
        assert hasattr(IHealthChecker, 'unregister_service')

    def test_basic_checker_implements_interface(self):
        """测试 BasicHealthChecker 实现了接口"""
        checker = BasicHealthChecker()
        assert isinstance(checker, IHealthChecker)

