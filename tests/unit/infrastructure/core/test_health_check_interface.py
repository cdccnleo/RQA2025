#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""健康检查接口测试。"""

from __future__ import annotations

from typing import Dict, Any

import pytest

from src.infrastructure.core import health_check_interface as hc


class _DemoService(hc.HealthCheckInterface):
    def __init__(self, name: str = "demo", healthy: bool = True):
        self._name = name
        self._healthy = healthy
        self._version = "1.2.3"

    def health_check(self) -> Dict[str, Any]:
        return {
            "service": self._name,
            "healthy": self._healthy,
            "status": "healthy" if self._healthy else "unhealthy",
            "timestamp": hc.datetime.now().isoformat(),
            "version": self._version,
        }

    @property
    def service_name(self) -> str:
        return self._name

    @property
    def service_version(self) -> str:
        return self._version


def test_health_check_result_to_dict() -> None:
    result = hc.HealthCheckResult(
        service_name="cache",
        healthy=False,
        status="unhealthy",
        details={"latency": 1.2},
        issues=["timeout"],
        recommendations=["restart"],
    )
    data = result.to_dict()
    assert data["service"] == "cache"
    assert data["healthy"] is False
    assert data["details"]["latency"] == 1.2


def test_infrastructure_health_checker_register_and_check() -> None:
    checker = hc.InfrastructureHealthChecker()
    checker.register_service(_DemoService("svc1", healthy=True))
    checker.register_service(_DemoService("svc2", healthy=False))

    assert checker.get_service_list() == ["svc1", "svc2"]
    svc_info = checker.get_service_info("svc1")
    assert svc_info["name"] == "svc1"
    assert svc_info["version"] == "1.2.3"

    single = checker.check_service("svc1")
    assert single["healthy"] is True
    missing = checker.check_service("unknown")
    assert missing is None

    summary = checker.check_all_services()
    assert summary["overall_healthy"] is False
    assert summary["services"]["svc2"]["healthy"] is False

    checker.unregister_service("svc1")
    assert checker.get_service_list() == ["svc2"]
    assert checker.get_service_info("missing") is None


def test_health_checker_handles_service_exception() -> None:
    class _FailService(_DemoService):
        def health_check(self) -> Dict[str, Any]:  # type: ignore[override]
            raise RuntimeError("boom")

    checker = hc.InfrastructureHealthChecker()
    checker.register_service(_FailService("bad"))
    summary = checker.check_all_services()
    service_result = summary["services"]["bad"]
    assert service_result["status"] == "error"
    assert service_result["error"] == "boom"
    assert summary["overall_healthy"] is False


def test_global_health_checker_helpers() -> None:
    checker = hc.get_infrastructure_health_checker()
    checker.unregister_service("global")
    hc.register_infrastructure_service(_DemoService("global"))
    result = hc.check_infrastructure_health()
    assert "services" in result
    checker.unregister_service("global")


def test_infrastructure_health_checker_service_management() -> None:
    """测试基础设施健康检查器的服务管理功能"""
    checker = hc.InfrastructureHealthChecker()

    # 测试初始状态
    assert checker.get_service_list() == []

    # 注册服务
    service1 = _DemoService("service1", True)
    service2 = _DemoService("service2", False)
    checker.register_service(service1)
    checker.register_service(service2)

    # 检查服务列表
    services = checker.get_service_list()
    assert len(services) == 2
    assert "service1" in services
    assert "service2" in services

    # 检查服务信息
    info1 = checker.get_service_info("service1")
    assert info1 is not None
    assert info1["name"] == "service1"
    assert info1["has_health_check"] == True

    info2 = checker.get_service_info("service2")
    assert info2 is not None
    assert info2["name"] == "service2"
    assert info2["has_health_check"] == True

    # 检查不存在的服务
    assert checker.get_service_info("nonexistent") is None

    # 注销服务
    checker.unregister_service("service1")
    assert checker.get_service_list() == ["service2"]

    # 再次注销不存在的服务（应该不报错）
    checker.unregister_service("nonexistent")


def test_infrastructure_health_checker_check_service() -> None:
    """测试基础设施健康检查器的单个服务检查"""
    checker = hc.InfrastructureHealthChecker()

    # 检查不存在的服务
    result = checker.check_service("nonexistent")
    assert result is None

    # 注册并检查服务
    service = _DemoService("test_service", True)
    checker.register_service(service)

    result = checker.check_service("test_service")
    assert result is not None
    assert result["service"] == "test_service"
    assert result["healthy"] == True


def test_infrastructure_health_checker_check_all_with_exceptions() -> None:
    """测试基础设施健康检查器处理服务异常"""
    checker = hc.InfrastructureHealthChecker()

    # 创建一个会抛出异常的服务
    class FailingService(hc.HealthCheckInterface):
        @property
        def service_name(self) -> str:
            return "failing"

        @property
        def service_version(self) -> str:
            return "1.0.0"

        def health_check(self) -> Dict[str, Any]:
            raise Exception("Service check failed")

    # 注册正常和异常服务
    checker.register_service(_DemoService("normal", True))
    checker.register_service(FailingService())

    result = checker.check_all_services()

    # 整体结果应该是不健康的
    assert result["overall_healthy"] == False
    assert result["total_services"] == 2
    assert result["healthy_services"] == 1
    assert result["unhealthy_services"] == 1

    # 检查异常服务的错误信息
    failing_result = result["services"]["failing"]
    assert failing_result["healthy"] == False
    assert failing_result["status"] == "error"
    assert "Service check failed" in failing_result["error"]


def test_health_check_result_initialization() -> None:
    """测试HealthCheckResult的初始化"""
    # 测试默认值
    result = hc.HealthCheckResult("test", True, "ok")
    assert result.service_name == "test"
    assert result.healthy == True
    assert result.status == "ok"
    assert result.version == "1.0.0"
    assert result.details == {}
    assert result.issues == []
    assert result.recommendations == []
    assert result.timestamp is not None

    # 测试自定义值
    custom_details = {"key": "value"}
    custom_issues = ["issue1"]
    custom_recommendations = ["fix1"]

    result = hc.HealthCheckResult(
        "test2", False, "error",
        version="2.0.0",
        details=custom_details,
        issues=custom_issues,
        recommendations=custom_recommendations
    )
    assert result.service_name == "test2"
    assert result.healthy == False
    assert result.status == "error"
    assert result.version == "2.0.0"
    assert result.details == custom_details
    assert result.issues == custom_issues
    assert result.recommendations == custom_recommendations
