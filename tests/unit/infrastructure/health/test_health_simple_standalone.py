#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
健康检查系统简化独立测试

测试基本的健康检查功能，不依赖复杂的导入链
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from typing import Dict, Any, Optional, Callable
from enum import Enum


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthResult:
    """健康检查结果"""

    def __init__(self,
                 name: str,
                 status: HealthStatus,
                 message: str = "",
                 details: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[float] = None):
        self.name = name
        self.status = status
        self.message = message
        self.details = details or {}
        self.timestamp = timestamp or time.time()

    def is_healthy(self) -> bool:
        """检查是否健康"""
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }


class HealthCheckError(Exception):
    """健康检查异常"""
    pass


class HealthChecker:
    """简化的健康检查器"""

    def __init__(self, name: str = "health_checker"):
        self.name = name
        self._checks: Dict[str, Callable[[], HealthResult]] = {}

    def add_check(self, name: str, check_func: Callable[[], HealthResult]) -> None:
        """添加健康检查"""
        self._checks[name] = check_func

    def remove_check(self, name: str) -> bool:
        """移除健康检查"""
        if name in self._checks:
            del self._checks[name]
            return True
        return False

    def perform_check(self, name: str) -> Optional[HealthResult]:
        """执行指定的健康检查"""
        if name not in self._checks:
            return None

        try:
            return self._checks[name]()
        except Exception as e:
            return HealthResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                details={"error": str(e)}
            )

    def perform_all_checks(self) -> Dict[str, HealthResult]:
        """执行所有健康检查"""
        results = {}
        for name in self._checks:
            results[name] = self.perform_check(name)
        return results

    def get_overall_status(self) -> HealthStatus:
        """获取整体健康状态"""
        if not self._checks:
            return HealthStatus.UNKNOWN

        results = self.perform_all_checks()

        # 如果有任何检查不健康，则整体不健康
        for result in results.values():
            if not result.is_healthy():
                return HealthStatus.UNHEALTHY

        # 如果所有检查都健康，则整体健康
        return HealthStatus.HEALTHY

    def get_summary(self) -> Dict[str, Any]:
        """获取健康检查摘要"""
        results = self.perform_all_checks()
        total_checks = len(results)
        healthy_checks = sum(1 for r in results.values() if r.is_healthy())

        return {
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "unhealthy_checks": total_checks - healthy_checks,
            "overall_status": self.get_overall_status().value,
            "results": {name: result.to_dict() for name, result in results.items()}
        }


# 预定义的健康检查函数
def check_database_connection() -> HealthResult:
    """检查数据库连接"""
    try:
        # 模拟数据库连接检查
        # 在实际实现中，这里会连接数据库
        time.sleep(0.01)  # 模拟网络延迟
        return HealthResult(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connection is healthy",
            details={"latency_ms": 10}
        )
    except Exception as e:
        return HealthResult(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {str(e)}"
        )


def check_memory_usage() -> HealthResult:
    """检查内存使用情况"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        usage_percent = memory.percent

        if usage_percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"Memory usage too high: {usage_percent}%"
        elif usage_percent > 80:
            status = HealthStatus.DEGRADED
            message = f"Memory usage high: {usage_percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage normal: {usage_percent}%"

        return HealthResult(
            name="memory",
            status=status,
            message=message,
            details={
                "usage_percent": usage_percent,
                "total_mb": memory.total / 1024 / 1024,
                "available_mb": memory.available / 1024 / 1024
            }
        )
    except ImportError:
        # 如果没有psutil，使用简化的检查
        return HealthResult(
            name="memory",
            status=HealthStatus.HEALTHY,
            message="Memory check skipped (psutil not available)",
            details={"note": "psutil not available"}
        )
    except Exception as e:
        return HealthResult(
            name="memory",
            status=HealthStatus.UNHEALTHY,
            message=f"Memory check failed: {str(e)}"
        )


def check_disk_space() -> HealthResult:
    """检查磁盘空间"""
    try:
        import psutil
        disk = psutil.disk_usage('/')
        free_percent = 100 - disk.percent

        if free_percent < 5:
            status = HealthStatus.UNHEALTHY
            message = f"Disk space critically low: {free_percent}% free"
        elif free_percent < 10:
            status = HealthStatus.DEGRADED
            message = f"Disk space low: {free_percent}% free"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk space sufficient: {free_percent}% free"

        return HealthResult(
            name="disk",
            status=status,
            message=message,
            details={
                "free_percent": free_percent,
                "total_gb": disk.total / 1024 / 1024 / 1024,
                "free_gb": disk.free / 1024 / 1024 / 1024
            }
        )
    except ImportError:
        return HealthResult(
            name="disk",
            status=HealthStatus.HEALTHY,
            message="Disk check skipped (psutil not available)",
            details={"note": "psutil not available"}
        )
    except Exception as e:
        return HealthResult(
            name="disk",
            status=HealthStatus.UNHEALTHY,
            message=f"Disk check failed: {str(e)}"
        )


class TestHealthChecker:
    """测试简化健康检查器"""

    def setup_method(self, method):
        """测试前准备"""
        self.checker = HealthChecker("test_checker")

    def test_initialization(self):
        """测试初始化"""
        assert self.checker.name == "test_checker"
        assert self.checker._checks == {}

    def test_add_and_remove_check(self):
        """测试添加和移除检查"""
        def dummy_check():
            return HealthResult("dummy", HealthStatus.HEALTHY, "OK")

        # 添加检查
        self.checker.add_check("dummy", dummy_check)
        assert "dummy" in self.checker._checks

        # 执行检查
        result = self.checker.perform_check("dummy")
        assert result is not None
        assert result.name == "dummy"
        assert result.is_healthy()

        # 移除检查
        removed = self.checker.remove_check("dummy")
        assert removed is True
        assert "dummy" not in self.checker._checks

        # 再次移除应该失败
        removed = self.checker.remove_check("dummy")
        assert removed is False

    def test_perform_check_nonexistent(self):
        """测试执行不存在的检查"""
        result = self.checker.perform_check("nonexistent")
        assert result is None

    def test_perform_all_checks_empty(self):
        """测试执行空检查列表"""
        results = self.checker.perform_all_checks()
        assert results == {}

    def test_perform_all_checks_with_checks(self):
        """测试执行多个检查"""
        def check1():
            return HealthResult("check1", HealthStatus.HEALTHY, "OK1")

        def check2():
            return HealthResult("check2", HealthStatus.UNHEALTHY, "Failed")

        self.checker.add_check("check1", check1)
        self.checker.add_check("check2", check2)

        results = self.checker.perform_all_checks()

        assert len(results) == 2
        assert results["check1"].is_healthy()
        assert not results["check2"].is_healthy()

    def test_get_overall_status_empty(self):
        """测试获取空检查器的整体状态"""
        status = self.checker.get_overall_status()
        assert status == HealthStatus.UNKNOWN

    def test_get_overall_status_all_healthy(self):
        """测试所有检查都健康时的整体状态"""
        def healthy_check():
            return HealthResult("healthy", HealthStatus.HEALTHY, "OK")

        self.checker.add_check("healthy1", healthy_check)
        self.checker.add_check("healthy2", healthy_check)

        status = self.checker.get_overall_status()
        assert status == HealthStatus.HEALTHY

    def test_get_overall_status_with_unhealthy(self):
        """测试有不健康检查时的整体状态"""
        def healthy_check():
            return HealthResult("healthy", HealthStatus.HEALTHY, "OK")

        def unhealthy_check():
            return HealthResult("unhealthy", HealthStatus.UNHEALTHY, "Failed")

        self.checker.add_check("healthy", healthy_check)
        self.checker.add_check("unhealthy", unhealthy_check)

        status = self.checker.get_overall_status()
        assert status == HealthStatus.UNHEALTHY

    def test_get_summary(self):
        """测试获取摘要"""
        def healthy_check():
            return HealthResult("healthy", HealthStatus.HEALTHY, "OK")

        def unhealthy_check():
            return HealthResult("unhealthy", HealthStatus.UNHEALTHY, "Failed")

        self.checker.add_check("healthy", healthy_check)
        self.checker.add_check("unhealthy", unhealthy_check)

        summary = self.checker.get_summary()

        assert summary["total_checks"] == 2
        assert summary["healthy_checks"] == 1
        assert summary["unhealthy_checks"] == 1
        assert summary["overall_status"] == HealthStatus.UNHEALTHY.value
        assert len(summary["results"]) == 2

    def test_health_result_properties(self):
        """测试健康结果属性"""
        result = HealthResult(
            name="test",
            status=HealthStatus.HEALTHY,
            message="Test message",
            details={"key": "value"}
        )

        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Test message"
        assert result.details == {"key": "value"}
        assert result.is_healthy()
        assert isinstance(result.timestamp, float)

    def test_health_result_to_dict(self):
        """测试健康结果转换为字典"""
        result = HealthResult(
            name="test",
            status=HealthStatus.DEGRADED,
            message="Test message",
            details={"key": "value"}
        )

        data = result.to_dict()

        assert data["name"] == "test"
        assert data["status"] == HealthStatus.DEGRADED.value
        assert data["message"] == "Test message"
        assert data["details"] == {"key": "value"}
        assert isinstance(data["timestamp"], float)

    def test_check_exception_handling(self):
        """测试检查函数异常处理"""
        def failing_check():
            raise Exception("Test exception")

        self.checker.add_check("failing", failing_check)

        result = self.checker.perform_check("failing")
        assert result is not None
        assert result.status == HealthStatus.UNHEALTHY
        assert "Test exception" in result.message
        assert not result.is_healthy()

    def test_predefined_checks(self):
        """测试预定义的检查函数"""
        # 数据库检查
        db_result = check_database_connection()
        assert db_result.name == "database"
        assert db_result.status in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]

        # 内存检查
        mem_result = check_memory_usage()
        assert mem_result.name == "memory"
        assert mem_result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]

        # 磁盘检查
        disk_result = check_disk_space()
        assert disk_result.name == "disk"
        assert disk_result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]

    def test_health_status_enum(self):
        """测试健康状态枚举"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


if __name__ == '__main__':
    pytest.main([__file__])
