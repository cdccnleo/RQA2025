#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""依赖检查器测试"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from src.infrastructure.health.components.dependency_checker import DependencyService, DependencyChecker


class TestDependencyService:
    """测试DependencyService类"""

    def test_init_with_defaults(self):
        """测试默认参数初始化"""
        service = DependencyService()

        assert service.name == ""
        assert callable(service.check_func)
        assert service.config == {}
        assert service.last_check_result is None
        assert service.last_check_time is None
        assert service.check_count == 0
        assert service.error_count == 0

    def test_init_with_parameters(self):
        """测试带参数初始化"""
        def custom_check():
            return {"status": "healthy"}

        config = {"timeout": 30, "retries": 3}
        service = DependencyService(name="test_service", check_func=custom_check, config=config)

        assert service.name == "test_service"
        assert service.check_func == custom_check
        assert service.config == config

    def test_default_check_func(self):
        """测试默认检查函数"""
        service = DependencyService()
        result = service.check_func()

        assert result == {"status": "unknown"}


class TestDependencyChecker:
    """测试DependencyChecker类"""

    def test_init(self):
        """测试初始化"""
        checker = DependencyChecker()

        assert isinstance(checker.dependencies, list)
        assert len(checker.dependencies) == 0

    def test_add_dependency_check(self):
        """测试添加依赖检查"""
        from src.infrastructure.health.components.parameter_objects import DependencyConfig

        checker = DependencyChecker()

        def check_func():
            return {"status": "healthy"}

        config = {"timeout": 30}

        dep_config = DependencyConfig(
            name="test_dep",
            check_func=check_func,
            config=config
        )

        result = checker.add_dependency_check(dep_config)

        assert result is True
        assert len(checker.dependencies) == 1

        service = checker.dependencies[0]
        assert service.name == "test_dep"
        assert service.check_func == check_func
        assert service.config == config

    def test_add_dependency_check_duplicate_name(self):
        """测试添加重复名称的依赖检查"""
        checker = DependencyChecker()

        def check_func1():
            return {"status": "healthy"}

        def check_func2():
            return {"status": "healthy"}

        # 添加第一个
        checker.add_dependency_check("test_dep", check_func1)
        assert len(checker.dependencies) == 1

        # 添加同名第二个，应该失败
        result = checker.add_dependency_check("test_dep", check_func2)
        assert result is False
        assert len(checker.dependencies) == 1

    def test_add_dependency_check_legacy(self):
        """测试遗留的添加依赖检查方法"""
        checker = DependencyChecker()

        def check_func():
            return {"status": "healthy"}

        result = checker.add_dependency_check_legacy(
            name="legacy_dep",
            check_func=check_func,
            config={"legacy": True}
        )

        assert result is True
        assert len(checker.dependencies) == 1

        service = checker.dependencies[0]
        assert service.name == "legacy_dep"
        assert service.check_func == check_func

    def test_remove_dependency_check(self):
        """测试移除依赖检查"""
        checker = DependencyChecker()

        def check_func():
            return {"status": "healthy"}

        # 添加依赖
        checker.add_dependency_check("test_dep", check_func)
        assert len(checker.dependencies) == 1

        # 移除存在的依赖
        result = checker.remove_dependency_check("test_dep")
        assert result is True
        assert len(checker.dependencies) == 0

        # 移除不存在的依赖
        result = checker.remove_dependency_check("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_all_dependencies_empty(self):
        """测试检查所有依赖（空列表）"""
        checker = DependencyChecker()

        results = await checker.check_all_dependencies()

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_check_all_dependencies_with_services(self):
        """测试检查所有依赖（有服务）"""
        checker = DependencyChecker()

        # Mock检查函数
        mock_check1 = Mock(return_value={"status": "healthy", "service": "dep1"})
        mock_check2 = Mock(return_value={"status": "unhealthy", "service": "dep2"})

        from src.infrastructure.health.components.parameter_objects import DependencyConfig

        dep_config1 = DependencyConfig(name="dep1", check_func=mock_check1)
        dep_config2 = DependencyConfig(name="dep2", check_func=mock_check2)

        checker.add_dependency_check(dep_config1)
        checker.add_dependency_check(dep_config2)

        results = await checker.check_all_dependencies()

        assert len(results) == 2
        assert results[0]["name"] == "dep1"
        assert results[1]["name"] == "dep2"

        # 验证调用次数
        mock_check1.assert_called_once()
        mock_check2.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_single_dependency(self):
        """测试检查单个依赖"""
        checker = DependencyChecker()

        def check_func():
            return {"status": "healthy", "details": "test"}

        service = DependencyService("test_service", check_func)
        result = await checker._check_single_dependency(service)

        assert result["name"] == "test_service"
        assert result["status"] == "healthy"
        assert "response_time_ms" in result
        assert result["response_time_ms"] >= 0
        assert "timestamp" in result
        assert "response_time_ms" in result

    @pytest.mark.asyncio
    async def test_check_single_dependency_with_exception(self):
        """测试检查单个依赖（异常情况）"""
        checker = DependencyChecker()

        def failing_check():
            raise Exception("Check failed")

        service = DependencyService("failing_service", failing_check)
        result = await checker._check_single_dependency(service)

        assert result["name"] == "failing_service"
        assert result["status"] == "error"
        assert "Check failed" in result["error"]

    def test_evaluate_health_status(self):
        """测试健康状态评估"""
        checker = DependencyChecker()

        # 测试健康状态
        assert checker._evaluate_health_status({"status": "healthy"}) is True
        assert checker._evaluate_health_status({"status": "ok"}) is True

        # 测试不健康状态
        assert checker._evaluate_health_status({"status": "unhealthy"}) is False
        assert checker._evaluate_health_status({"status": "error"}) is False
        assert checker._evaluate_health_status({"status": "critical"}) is False

        # 测试未知状态
        assert checker._evaluate_health_status({"status": "unknown"}) is False
        assert checker._evaluate_health_status({}) is False
        assert checker._evaluate_health_status(None) is False

    def test_check_dependencies_health_no_dependencies(self):
        """测试健康检查（无依赖）"""
        checker = DependencyChecker()

        result = checker.check_dependencies_health()

        assert result["status"] == "success"
        assert result["total_count"] == 0
        assert result["healthy_count"] == 0

    def test_check_dependencies_health_with_dependencies(self):
        """测试健康检查（有依赖）"""
        checker = DependencyChecker()

        # 添加健康的依赖
        def healthy_check():
            return {"status": "healthy"}

        # 添加不健康的依赖
        def unhealthy_check():
            return {"status": "unhealthy"}

        checker.add_dependency_check("healthy_dep", healthy_check)
        checker.add_dependency_check("unhealthy_dep", unhealthy_check)

        result = checker.check_dependencies_health()

        assert result["status"] == "success"  # 方法总是返回success状态
        assert result["total_count"] == 2
        assert result["healthy_count"] == 1
        assert result["unhealthy_count"] == 1

    def test_get_dependencies_summary(self):
        """测试获取依赖摘要"""
        checker = DependencyChecker()

        def check_func():
            return {"status": "healthy"}

        checker.add_dependency_check("test_dep", check_func, {"timeout": 30})

        summary = checker.get_dependencies_summary()

        assert summary["total_dependencies"] == 1
        assert len(summary["dependencies"]) == 1

        dep_info = summary["dependencies"][0]
        assert dep_info["name"] == "test_dep"
        assert dep_info["check_count"] == 0
        assert dep_info["error_count"] == 0

    def test_get_dependency_by_name(self):
        """测试按名称获取依赖"""
        checker = DependencyChecker()

        def check_func():
            return {"status": "healthy"}

        checker.add_dependency_check("test_dep", check_func)

        # 获取存在的依赖
        service = checker.get_dependency_by_name("test_dep")
        assert service is not None
        assert service.name == "test_dep"

        # 获取不存在的依赖
        service = checker.get_dependency_by_name("nonexistent")
        assert service is None

    def test_create_no_dependencies_response(self):
        """测试创建无依赖响应"""
        checker = DependencyChecker()

        response = checker._create_no_dependencies_response()

        assert response["status"] == "success"
        assert "message" in response

    def test_create_health_check_response(self):
        """测试创建健康检查响应"""
        checker = DependencyChecker()

        results = [
            {"service": "dep1", "status": "healthy"},
            {"service": "dep2", "status": "unhealthy"}
        ]

        response = checker._create_health_check_response(results)

        assert response["status"] == "success"
        assert response["total_count"] == 2
        assert response["healthy_count"] == 1
        assert response["unhealthy_count"] == 1
        assert len(response["dependencies"]) == 2

    def test_create_error_response(self):
        """测试创建错误响应"""
        checker = DependencyChecker()

        response = checker._create_error_response("Test error message")

        assert response["status"] == "error"
        assert "Test error message" in response["message"]
        assert "timestamp" in response

    def test_check_all_dependencies_sync(self):
        """测试同步检查所有依赖"""
        checker = DependencyChecker()

        def check_func():
            return {"status": "healthy", "service": "test"}

        checker.add_dependency_check("test_dep", check_func)

        results = checker._check_all_dependencies_sync()

        assert len(results) == 1
        assert results[0]["name"] == "test_dep"
        assert results[0]["status"] == "healthy"

    def test_check_single_dependency_sync(self):
        """测试同步检查单个依赖"""
        checker = DependencyChecker()

        def check_func():
            return {"status": "healthy"}

        service = DependencyService("test_service", check_func)

        result = checker._check_single_dependency_sync(service)

        assert result["name"] == "test_service"
        assert result["status"] == "healthy"
        assert "timestamp" in result
