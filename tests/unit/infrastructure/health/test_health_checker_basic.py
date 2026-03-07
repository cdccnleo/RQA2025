#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 健康检查器组件基础功能测试
测试AsyncHealthCheckerComponent和HealthChecker的基本可用API
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import asyncio

from infrastructure.health.components.health_checker import (
    AsyncHealthCheckerComponent, HealthChecker
)
from infrastructure.health.models.health_result import HealthCheckResult, CheckType
from infrastructure.health.models.health_status import HealthStatus


class TestAsyncHealthCheckerComponent:
    """异步健康检查器组件测试"""

    @pytest.fixture
    def async_checker(self):
        """异步检查器fixture"""
        return AsyncHealthCheckerComponent("test_service")

    @pytest.mark.asyncio
    async def test_check_health_async_basic(self, async_checker):
        """测试基本的异步健康检查"""
        result = await async_checker.check_health_async()

        assert isinstance(result, HealthCheckResult)
        assert result.service_name == "test_service"
        assert result.check_type == CheckType.BASIC
        assert hasattr(result, 'status')
        assert hasattr(result, 'message')
        assert hasattr(result, 'details')

    @pytest.mark.asyncio
    async def test_check_health_async_with_registered_check(self, async_checker):
        """测试注册检查函数后的异步健康检查"""
        async def test_check():
            return HealthCheckResult(
                service_name="test_component",
                status=HealthStatus.UP,
                check_type=CheckType.BASIC,
                message="Component is healthy"
            )

        async_checker.register_check_function("test_check", test_check)

        result = await async_checker.check_health_async()

        assert isinstance(result, HealthCheckResult)
        assert result.service_name == "test_service"
        assert "checks_count" in result.details

    def test_check_health_sync_basic(self, async_checker):
        """测试基本的同步健康检查"""
        result = async_checker.check_health_sync()

        assert isinstance(result, HealthCheckResult)
        assert result.service_name == "test_service"
        assert result.check_type == CheckType.BASIC

    def test_register_check_function(self, async_checker):
        """测试注册检查函数"""
        async def test_check():
            return HealthCheckResult("test", HealthStatus.UP, CheckType.BASIC, response_time=0.0)

        async_checker.register_check_function("test_check", test_check)

        # 验证内部状态
        assert "test_check" in async_checker.check_functions
        assert async_checker.check_functions["test_check"] == test_check

    def test_unregister_check_function(self, async_checker):
        """测试取消注册检查函数"""
        async def test_check():
            return HealthCheckResult("test", HealthStatus.UP, CheckType.BASIC, response_time=0.0)

        async_checker.register_check_function("test_check", test_check)

        # 确认已注册
        assert "test_check" in async_checker.check_functions

        # 取消注册
        result = async_checker.unregister_check_function("test_check")
        assert result is True

        # 确认已移除
        assert "test_check" not in async_checker.check_functions

    def test_unregister_nonexistent_check_function(self, async_checker):
        """测试取消注册不存在的检查函数"""
        result = async_checker.unregister_check_function("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_multiple_check_functions(self, async_checker):
        """测试多个检查函数"""
        async def check1():
            return HealthCheckResult("comp1", HealthStatus.UP, CheckType.BASIC, response_time=0.0)

        async def check2():
            return HealthCheckResult("comp2", HealthStatus.DEGRADED, CheckType.BASIC, response_time=0.0)

        async_checker.register_check_function("check1", check1)
        async_checker.register_check_function("check2", check2)

        result = await async_checker.check_health_async()

        assert isinstance(result, HealthCheckResult)
        assert result.details["checks_count"] == 2

    @pytest.mark.asyncio
    async def test_exception_in_check_function(self, async_checker):
        """测试检查函数中的异常"""
        async def failing_check():
            raise ValueError("Test exception")

        async_checker.register_check_function("failing", failing_check)

        result = await async_checker.check_health_async()

        # 应该不会崩溃，而是返回错误结果
        assert isinstance(result, HealthCheckResult)


class TestHealthChecker:
    """同步健康检查器测试"""

    @pytest.fixture
    def sync_checker(self):
        """同步检查器fixture"""
        return HealthChecker("sync_test_service")

    def test_check_health_basic(self, sync_checker):
        """测试基本的同步健康检查"""
        result = sync_checker.check_health()

        assert isinstance(result, HealthCheckResult)
        assert result.service_name == "sync_test_service"
        assert result.check_type == CheckType.BASIC

    def test_register_sync_check_function(self, sync_checker):
        """测试注册同步检查函数"""
        def sync_check():
            return HealthCheckResult(
                service_name="sync_test",
                status=HealthStatus.UP,
                check_type=CheckType.BASIC,
                response_time=0.0
            )

        sync_checker.register_check_function("sync_check", sync_check)

        # HealthChecker是AsyncHealthCheckerComponent的包装器
        assert "sync_check" in sync_checker.async_checker.check_functions

    def test_unregister_sync_check_function(self, sync_checker):
        """测试取消注册同步检查函数"""
        def sync_check():
            return HealthCheckResult(
                "sync",
                HealthStatus.UP,
                CheckType.BASIC,
                response_time=0.0
            )

        sync_checker.register_check_function("sync_check", sync_check)

        result = sync_checker.unregister_check_function("sync_check")
        assert result is True
        assert "sync_check" not in sync_checker.async_checker.check_functions
