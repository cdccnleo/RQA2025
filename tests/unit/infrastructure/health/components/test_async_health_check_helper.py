#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""异步健康检查助手测试"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from src.infrastructure.health.components.async_health_check_helper import AsyncHealthCheckHelper


class TestAsyncHealthCheckHelper:
    """测试异步健康检查助手"""

    def test_init_with_default_health_checker(self):
        """测试使用默认健康检查器的初始化"""
        helper = AsyncHealthCheckHelper()

        assert helper.health_checker is not None
        assert hasattr(helper.health_checker, 'check_health_async')
        assert hasattr(helper.health_checker, 'check_health')
        assert hasattr(helper.health_checker, 'check_service_async')

    def test_init_with_custom_health_checker(self):
        """测试使用自定义健康检查器的初始化"""
        custom_checker = Mock()
        helper = AsyncHealthCheckHelper(health_checker=custom_checker)

        assert helper.health_checker is custom_checker

    @pytest.mark.asyncio
    async def test_check_database_async(self):
        """测试异步数据库检查"""
        helper = AsyncHealthCheckHelper()

        # Mock数据库连接检查
        with patch('asyncio.sleep'), \
             patch('time.time') as mock_time:

            mock_time.return_value = 1000.0

            result = await helper.check_database_async()

            assert isinstance(result, dict)
            assert 'status' in result
            assert 'timestamp' in result
            assert 'database_status' in result
            assert result['check_type'] == 'async_database_check'

    @pytest.mark.asyncio
    async def test_check_service_async(self):
        """测试异步服务检查"""
        helper = AsyncHealthCheckHelper()

        service_name = "test_service"

        # Mock服务检查
        with patch.object(helper.health_checker, 'check_service_async',
                         new_callable=AsyncMock) as mock_check:

            mock_check.return_value = {"status": "healthy", "service": service_name}

            result = await helper.check_service_async(service_name)

            assert result["status"] == "healthy"
            assert result["service"] == service_name
            mock_check.assert_called_once_with(service_name)

    def test_create_comprehensive_check_tasks(self):
        """测试创建综合检查任务"""
        helper = AsyncHealthCheckHelper()

        # 测试默认服务名（应该有2个服务 + 1个数据库检查）
        tasks = helper.create_comprehensive_check_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) == 3  # 2 services + 1 database

        # 测试自定义服务名（N个服务 + 1个数据库检查）
        service_names = ["service1", "service2", "service3"]
        tasks = helper.create_comprehensive_check_tasks(service_names)
        assert isinstance(tasks, list)
        assert len(tasks) == len(service_names) + 1  # +1 for database check

    @pytest.mark.asyncio
    async def test_create_comprehensive_check_tasks_execution(self):
        """测试综合检查任务的执行"""
        helper = AsyncHealthCheckHelper()

        service_names = ["service1", "service2"]

        # Mock所有异步操作
        with patch.object(helper, 'check_database_async', new_callable=AsyncMock) as mock_db, \
             patch.object(helper, 'check_service_async', new_callable=AsyncMock) as mock_service, \
             patch('asyncio.gather', new_callable=AsyncMock) as mock_gather:

            mock_db.return_value = {"status": "healthy", "database_check": {}}
            mock_service.return_value = {"status": "healthy", "service": "test"}
            mock_gather.return_value = [
                {"status": "healthy", "database_check": {}},
                {"status": "healthy", "service": "service1"},
                {"status": "healthy", "service": "service2"}
            ]

            tasks = helper.create_comprehensive_check_tasks(service_names)

            # 验证任务创建
            assert len(tasks) == len(service_names) + 1  # +1 for database check

            # 执行任务
            results = await asyncio.gather(*tasks)

            assert len(results) == len(service_names) + 1

    def test_analyze_comprehensive_results(self):
        """测试综合结果分析"""
        helper = AsyncHealthCheckHelper()

        # 测试结果数据
        results = [
            {"status": "healthy", "service": "service1"},
            {"status": "unhealthy", "service": "service2"},
            {"status": "healthy", "database_check": {}}
        ]

        summary, counts = helper.analyze_comprehensive_results(results)

        assert isinstance(summary, dict)
        assert isinstance(counts, dict)
        assert 'healthy_count' in counts
        assert 'unhealthy_count' in counts
        assert 'critical_count' in counts
        assert counts['healthy_count'] >= 1
        assert counts['unhealthy_count'] >= 1

    def test_determine_comprehensive_status(self):
        """测试综合状态确定"""
        helper = AsyncHealthCheckHelper()

        # 测试全部健康
        counts = {"healthy": 3, "unhealthy": 0, "unknown": 0}
        status = helper.determine_comprehensive_status(counts)
        assert status == "healthy"

        # 测试部分不健康
        counts = {"healthy_count": 2, "unhealthy_count": 1, "critical_count": 0}
        status = helper.determine_comprehensive_status(counts)
        assert status == "warning"

        # 测试全部不健康（没有healthy_count，只有unhealthy_count）
        counts = {"healthy_count": 0, "unhealthy_count": 3, "critical_count": 0}
        status = helper.determine_comprehensive_status(counts)
        assert status == "warning"

        # 测试未知状态（没有unhealthy和critical，只有healthy）
        counts = {"healthy_count": 0, "unhealthy_count": 0, "critical_count": 0}
        status = helper.determine_comprehensive_status(counts)
        assert status == "healthy"

    def test_create_comprehensive_success_response(self):
        """测试创建成功的综合响应"""
        helper = AsyncHealthCheckHelper()

        import time
        start_time = time.time()
        tasks = [AsyncMock(), AsyncMock()]
        counts = {"healthy_count": 2, "unhealthy_count": 0, "critical_count": 0}
        component_results = {"comp1": {"status": "healthy"}}

        response = helper.create_comprehensive_success_response(
            "healthy", start_time, tasks, counts, component_results
        )

        assert response["status"] == "healthy"
        assert "timestamp" in response
        assert "execution_time" in response
        assert response["components_checked"] == len(tasks)
        assert response["healthy_components"] == counts["healthy_count"]

    def test_create_comprehensive_error_response(self):
        """测试创建错误的综合响应"""
        helper = AsyncHealthCheckHelper()

        error = Exception("Test error")

        response = helper.create_comprehensive_error_response(error)

        assert response["status"] == "critical"
        assert "error" in response
        assert "timestamp" in response
        assert "Test error" in str(response["error"])

    @pytest.mark.asyncio
    async def test_comprehensive_health_check_async(self):
        """测试异步综合健康检查"""
        helper = AsyncHealthCheckHelper()

        service_names = ["service1"]

        # Mock所有内部方法
        with patch.object(helper, 'create_comprehensive_check_tasks') as mock_create_tasks, \
             patch('asyncio.gather', new_callable=AsyncMock) as mock_gather, \
             patch.object(helper, 'analyze_comprehensive_results') as mock_analyze, \
             patch.object(helper, 'determine_comprehensive_status') as mock_determine, \
             patch.object(helper, 'create_comprehensive_success_response') as mock_success_response:

            # 设置mock返回值
            mock_create_tasks.return_value = [AsyncMock(), AsyncMock()]
            mock_gather.return_value = [{"status": "healthy"}]
            mock_analyze.return_value = ({"summary": "test"}, {"healthy": 1})
            mock_determine.return_value = "healthy"
            mock_success_response.return_value = {"status": "healthy", "final": True}

            result = await helper.comprehensive_health_check_async(service_names)

            assert result["status"] == "healthy"
            assert result["final"] is True

    @pytest.mark.asyncio
    async def test_comprehensive_health_check_async_with_error(self):
        """测试异步综合健康检查的错误处理"""
        helper = AsyncHealthCheckHelper()

        # Mock抛出异常
        with patch.object(helper, 'create_comprehensive_check_tasks') as mock_create_tasks, \
             patch('asyncio.gather', side_effect=Exception("Test exception")), \
             patch.object(helper, 'create_comprehensive_error_response') as mock_error_response:

            mock_create_tasks.return_value = [AsyncMock()]
            mock_error_response.return_value = {"status": "error", "error": "Test exception"}

            result = await helper.comprehensive_health_check_async()

            assert result["status"] == "error"
            assert "error" in result
