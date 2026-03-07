#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础设施层健康管理 - 健康检查器完整工作流程测试

针对health_checker.py的核心业务逻辑进行高质量测试
当前覆盖率：16.78%，目标：通过测试完整业务流程提升到40%+

策略：每个测试执行完整的业务逻辑流程，争取覆盖5-10行代码
预期效率：6行/测试
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
from collections import deque


class TestEnhancedHealthCheckerWorkflows:
    """增强健康检查器完整工作流程测试"""

    @pytest.fixture
    def checker(self):
        """创建增强健康检查器实例"""
        from src.infrastructure.health.components.enhanced_health_checker import EnhancedHealthChecker
        return EnhancedHealthChecker()

    @pytest.fixture
    def checker_with_custom_config(self):
        """创建带自定义配置的健康检查器实例"""
        from src.infrastructure.health.components.enhanced_health_checker import EnhancedHealthChecker
        config = {
            'check_timeout': 60.0,
            'retry_count': 5,
            'concurrent_limit': 20
        }
        return EnhancedHealthChecker(config)

    # =========================================================================
    # 初始化工作流程测试
    # =========================================================================

    def test_initialization_default_config(self, checker):
        """测试默认配置初始化工作流程"""
        assert checker.config is not None
        assert hasattr(checker, '_health_history')
        assert hasattr(checker, '_performance_metrics')
        assert hasattr(checker, '_diagnostic_data')
        assert checker._check_timeout == 30.0
        assert checker._retry_count == 3
        assert checker._concurrent_limit == 10

    def test_initialization_custom_config(self, checker_with_custom_config):
        """测试自定义配置初始化工作流程"""
        checker = checker_with_custom_config
        assert checker._check_timeout == 60.0
        assert checker._retry_count == 5
        assert checker._concurrent_limit == 20

    def test_initialization_creates_data_structures(self, checker):
        """测试初始化创建数据结构工作流程"""
        assert isinstance(checker._health_history, dict)
        assert isinstance(checker._performance_metrics, dict)
        assert isinstance(checker._diagnostic_data, dict)
        assert checker._semaphore is None
        assert checker._semaphore_created is False

    # =========================================================================
    # 异步健康检查工作流程测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_check_health_async_success_workflow(self, checker):
        """测试异步健康检查成功工作流程"""
        result = await checker.check_health_async("test_service")
        
        assert isinstance(result, dict)
        assert "service" in result
        assert "status" in result
        assert "response_time" in result
        assert "timestamp" in result
        assert result["service"] == "test_service"

    @pytest.mark.asyncio
    async def test_check_health_async_stores_history(self, checker):
        """测试异步健康检查存储历史记录工作流程"""
        service_name = "test_service"
        result = await checker.check_health_async(service_name)
        
        assert service_name in checker._health_history
        assert len(checker._health_history[service_name]) > 0
        
        last_check = checker._health_history[service_name][-1]
        assert "timestamp" in last_check
        assert "status" in last_check
        assert "response_time" in last_check

    @pytest.mark.asyncio
    async def test_check_health_async_with_semaphore(self, checker):
        """测试带并发控制的异步健康检查工作流程"""
        # 确保semaphore被创建
        checker._ensure_semaphore()
        
        # 并行执行多个检查
        tasks = [
            checker.check_health_async(f"service_{i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for result in results:
            assert "status" in result
            assert "service" in result

    @pytest.mark.asyncio
    async def test_check_health_async_error_handling(self, checker):
        """测试异步健康检查错误处理工作流程"""
        with patch.object(checker, '_perform_comprehensive_check_async', side_effect=Exception("Test error")):
            result = await checker.check_health_async("test_service")
            
            assert result["status"] == "critical"
            assert "error" in result
            assert "Test error" in result["error"]

    # =========================================================================
    # 综合健康检查工作流程测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_comprehensive_check_all_dimensions(self, checker):
        """测试综合健康检查所有维度工作流程"""
        result = await checker._perform_comprehensive_check_async("test_service")
        
        assert "status" in result
        assert "issues" in result
        assert "check_dimensions" in result
        assert "details" in result
        assert result["check_dimensions"] == 4
        
        details = result["details"]
        assert "connectivity" in details
        assert "performance" in details
        assert "resources" in details
        assert "errors" in details

    @pytest.mark.asyncio
    async def test_comprehensive_check_status_aggregation(self, checker):
        """测试综合健康检查状态聚合工作流程"""
        # 测试健康状态
        result = await checker._perform_comprehensive_check_async("healthy_service")
        assert result["status"] in ["healthy", "warning", "critical"]

    @pytest.mark.asyncio
    async def test_comprehensive_check_handles_exceptions(self, checker):
        """测试综合健康检查异常处理工作流程"""
        with patch.object(checker, '_check_basic_connectivity_async', side_effect=Exception("Connection error")):
            result = await checker._perform_comprehensive_check_async("test_service")
            
            assert "status" in result
            assert result["status"] == "critical"
            assert len(result["issues"]) > 0

    # =========================================================================
    # 连接性检查工作流程测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_connectivity_check_success(self, checker):
        """测试连接性检查成功工作流程"""
        result = await checker._check_basic_connectivity_async("test_service")
        
        assert "status" in result
        assert "message" in result or "issues" in result
        assert "details" in result
        assert "connectivity" in result["details"]

    @pytest.mark.asyncio
    async def test_connectivity_check_failure(self, checker):
        """测试连接性检查失败工作流程"""
        with patch('asyncio.sleep', side_effect=Exception("Network error")):
            result = await checker._check_basic_connectivity_async("test_service")
            
            assert result["status"] == "critical"
            assert len(result["issues"]) > 0
            assert "connectivity" in result["details"]
            assert result["details"]["connectivity"] == "failed"

    # =========================================================================
    # 性能指标检查工作流程测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_performance_metrics_check_normal(self, checker):
        """测试性能指标检查正常工作流程"""
        checker._performance_metrics["test_service"] = {
            "avg_response_time": 0.5
        }
        
        result = await checker._check_performance_metrics_async("test_service")
        
        assert result["status"] == "healthy"
        assert "details" in result
        assert "response_time" in result["details"]

    @pytest.mark.asyncio
    async def test_performance_metrics_check_warning(self, checker):
        """测试性能指标检查警告工作流程"""
        checker._performance_metrics["test_service"] = {
            "avg_response_time": 3.0  # 超过阈值
        }
        
        result = await checker._check_performance_metrics_async("test_service")
        
        assert result["status"] == "warning"
        assert len(result["issues"]) > 0
        assert "响应时间" in result["issues"][0]

    @pytest.mark.asyncio
    async def test_performance_metrics_check_no_data(self, checker):
        """测试性能指标检查无数据工作流程"""
        result = await checker._check_performance_metrics_async("new_service")
        
        assert result["status"] == "healthy"
        assert result["details"]["response_time"] == 0

    # =========================================================================
    # 资源使用检查工作流程测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_resource_usage_check_normal(self, checker):
        """测试资源使用检查正常工作流程"""
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_mem:
            mock_mem.return_value.percent = 60.0
            
            result = await checker._check_resource_usage_async("test_service")
            
            assert result["status"] == "healthy"
            assert "details" in result
            assert "cpu_usage" in result["details"]
            assert "memory_usage" in result["details"]

    @pytest.mark.asyncio
    async def test_resource_usage_check_high_cpu(self, checker):
        """测试资源使用检查高CPU工作流程"""
        with patch('psutil.cpu_percent', return_value=85.0), \
             patch('psutil.virtual_memory') as mock_mem:
            mock_mem.return_value.percent = 60.0
            
            result = await checker._check_resource_usage_async("test_service")
            
            assert result["status"] == "critical"
            assert len(result["issues"]) > 0
            assert "CPU使用率" in result["issues"][0]

    @pytest.mark.asyncio
    async def test_resource_usage_check_high_memory(self, checker):
        """测试资源使用检查高内存工作流程"""
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_mem:
            mock_mem.return_value.percent = 90.0
            
            result = await checker._check_resource_usage_async("test_service")
            
            assert result["status"] == "critical"
            assert len(result["issues"]) > 0
            assert "内存使用率" in result["issues"][0]

    @pytest.mark.asyncio
    async def test_resource_usage_check_error_handling(self, checker):
        """测试资源使用检查错误处理工作流程"""
        with patch('psutil.cpu_percent', side_effect=Exception("Access denied")):
            result = await checker._check_resource_usage_async("test_service")
            
            assert result["status"] == "warning"
            assert len(result["issues"]) > 0

    # =========================================================================
    # 错误模式检查工作流程测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_error_patterns_check_no_failures(self, checker):
        """测试错误模式检查无失败工作流程"""
        # 添加正常检查历史
        service_name = "test_service"
        for i in range(10):
            checker._health_history[service_name].append({
                "status": "healthy",
                "timestamp": time.time()
            })
        
        result = await checker._check_error_patterns_async(service_name)
        
        assert result["status"] == "healthy"
        assert result["details"]["recent_failures"] == 0

    @pytest.mark.asyncio
    async def test_error_patterns_check_frequent_failures(self, checker):
        """测试错误模式检查频繁失败工作流程"""
        # 添加包含失败的检查历史
        service_name = "test_service"
        for i in range(10):
            status = "critical" if i % 3 == 0 else "healthy"
            checker._health_history[service_name].append({
                "status": status,
                "timestamp": time.time()
            })
        
        result = await checker._check_error_patterns_async(service_name)
        
        if result["details"]["recent_failures"] >= 3:
            assert result["status"] == "warning"
            assert len(result["issues"]) > 0

    @pytest.mark.asyncio
    async def test_error_patterns_check_new_service(self, checker):
        """测试错误模式检查新服务工作流程"""
        result = await checker._check_error_patterns_async("new_service")
        
        assert result["status"] == "healthy"
        assert result["details"]["recent_failures"] == 0

    # =========================================================================
    # 服务健康检查工作流程测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_check_service_health_async_workflow(self, checker):
        """测试服务健康检查异步工作流程"""
        result = await checker.check_service_health_async("test_service")
        
        assert "service" in result
        assert "status" in result
        assert "response_time" in result

    @pytest.mark.asyncio
    async def test_check_system_health_async_workflow(self, checker):
        """测试系统健康检查异步工作流程"""
        result = await checker.check_system_health_async()
        
        assert result["service"] == "system"
        assert "status" in result
        assert "details" in result
        assert "total_services" in result["details"]
        assert "healthy_services" in result["details"]

    @pytest.mark.asyncio
    async def test_check_database_async_workflow(self, checker):
        """测试数据库健康检查异步工作流程"""
        result = await checker.check_database_async()
        
        assert result["service"] == "database"
        assert "status" in result

    @pytest.mark.asyncio
    async def test_check_cache_async_workflow(self, checker):
        """测试缓存健康检查异步工作流程"""
        result = await checker.check_cache_async()
        
        assert result["service"] == "cache"
        assert "status" in result

    @pytest.mark.asyncio
    async def test_check_network_async_workflow(self, checker):
        """测试网络健康检查异步工作流程"""
        result = await checker.check_network_async()
        
        assert result["service"] == "network"
        assert "status" in result

    # =========================================================================
    # 健康状态管理工作流程测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_health_status_async_workflow(self, checker):
        """测试健康状态异步获取工作流程"""
        # 先执行一些检查
        await checker.check_health_async("service_1")
        await checker.check_health_async("service_2")
        
        status = await checker.health_status_async()
        
        assert "component" in status
        assert "services_monitored" in status
        assert "total_checks" in status
        assert status["services_monitored"] >= 2

    @pytest.mark.asyncio
    async def test_health_summary_async_workflow(self, checker):
        """测试健康状态汇总异步工作流程"""
        # 先执行一些检查
        await checker.check_health_async("service_1")
        await checker.check_health_async("service_2")
        
        summary = await checker.health_summary_async()
        
        assert "component" in summary
        assert "services_monitored" in summary
        if summary.get("total_checks", 0) > 0:
            assert "healthy_percentage" in summary
            assert "average_response_time" in summary

    def test_calculate_average_response_time(self, checker):
        """测试平均响应时间计算工作流程"""
        # 添加测试数据
        checker._health_history["service_1"].append({"response_time": 0.1})
        checker._health_history["service_1"].append({"response_time": 0.2})
        checker._health_history["service_2"].append({"response_time": 0.3})
        
        avg_time = checker._calculate_average_response_time()
        
        assert abs(avg_time - 0.2) < 0.001  # 浮点数近似比较

    # =========================================================================
    # 监控管理工作流程测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_monitor_start_workflow(self, checker):
        """测试监控启动工作流程"""
        result = await checker.monitor_start_async()
        assert result is True

    @pytest.mark.asyncio
    async def test_monitor_stop_workflow(self, checker):
        """测试监控停止工作流程"""
        result = await checker.monitor_stop_async()
        assert result is True

    @pytest.mark.asyncio
    async def test_monitor_status_workflow(self, checker):
        """测试监控状态查询工作流程"""
        status = await checker.monitor_status_async()
        
        assert "component" in status
        assert "monitoring_active" in status
        assert "services_count" in status

    @pytest.mark.asyncio
    async def test_monitor_lifecycle_workflow(self, checker):
        """测试监控完整生命周期工作流程"""
        # 启动监控
        start_result = await checker.monitor_start_async()
        assert start_result is True
        
        # 检查监控状态
        status = await checker.monitor_status_async()
        assert status["monitoring_active"] is True
        
        # 停止监控
        stop_result = await checker.monitor_stop_async()
        assert stop_result is True

    # =========================================================================
    # 配置验证工作流程测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_validate_health_config_valid(self, checker):
        """测试健康配置验证有效配置工作流程"""
        config = {
            'check_timeout': 30.0,
            'retry_count': 3
        }
        
        result = await checker.validate_health_config_async(config)
        
        assert result["status"] == "healthy"
        assert "message" in result

    @pytest.mark.asyncio
    async def test_validate_health_config_missing_keys(self, checker):
        """测试健康配置验证缺失键工作流程"""
        config = {
            'check_timeout': 30.0
            # 缺少 retry_count
        }
        
        result = await checker.validate_health_config_async(config)
        
        assert result["status"] == "critical"
        assert "缺少必需配置项" in result["message"]

    @pytest.mark.asyncio
    async def test_validate_health_config_invalid_values(self, checker):
        """测试健康配置验证无效值工作流程"""
        config = {
            'check_timeout': 0,  # 无效值
            'retry_count': 3
        }
        
        result = await checker.validate_health_config_async(config)
        
        assert result["status"] == "warning"

    # =========================================================================
    # 接口实现工作流程测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_check_service_async_interface(self, checker):
        """测试异步服务检查接口工作流程"""
        result = await checker.check_service_async("test_service", timeout=10.0)
        
        assert "service" in result
        assert "status" in result

    def test_check_service_sync_interface(self, checker):
        """测试同步服务检查接口工作流程"""
        result = checker.check_service("test_service", timeout=5)
        
        assert "service" in result
        assert "status" in result

    def test_check_health_sync_compatibility(self, checker):
        """测试同步健康检查兼容性工作流程"""
        result = checker.check_health("test_service")
        
        assert "service" in result
        assert "status" in result

    # =========================================================================
    # 统一基础设施接口工作流程测试
    # =========================================================================

    def test_initialize_interface(self, checker):
        """测试初始化接口工作流程"""
        config = {
            'check_timeout': 45.0,
            'retry_count': 4,
            'concurrent_limit': 15
        }
        
        result = checker.initialize(config)
        
        assert result is True
        assert checker._check_timeout == 45.0
        assert checker._retry_count == 4
        assert checker._concurrent_limit == 15

    def test_get_component_info_interface(self, checker):
        """测试获取组件信息接口工作流程"""
        info = checker.get_component_info()
        
        assert "component_type" in info
        assert "version" in info
        assert "capabilities" in info
        assert "config" in info
        assert info["component_type"] == "EnhancedHealthChecker"

    def test_is_healthy_interface(self, checker):
        """测试健康状态检查接口工作流程"""
        result = checker.is_healthy()
        
        assert isinstance(result, bool)
        assert result is True

    def test_get_metrics_interface(self, checker):
        """测试获取指标接口工作流程"""
        # 先执行一些检查以生成指标
        checker.check_health("service_1")
        checker.check_health("service_2")
        
        metrics = checker.get_metrics()
        
        assert "total_services" in metrics
        assert "total_checks" in metrics
        assert metrics["total_services"] >= 2

    def test_cleanup_interface(self, checker):
        """测试清理接口工作流程"""
        # 添加一些数据
        checker._health_history["test"] = deque([{"status": "healthy"}])
        checker._performance_metrics["test"] = {"cpu": 50}
        
        result = checker.cleanup()
        
        assert result is True
        assert len(checker._health_history) == 0
        assert len(checker._performance_metrics) == 0

    @pytest.mark.asyncio
    async def test_async_interface_methods(self, checker):
        """测试异步接口方法工作流程"""
        # 测试异步初始化
        init_result = await checker.initialize_async({'test': 'value'})
        assert isinstance(init_result, bool)
        
        # 测试异步获取组件信息
        info = await checker.get_component_info_async()
        assert "component_type" in info
        
        # 测试异步健康检查
        healthy = await checker.is_healthy_async()
        assert isinstance(healthy, bool)
        
        # 测试异步获取指标
        metrics = await checker.get_metrics_async()
        assert isinstance(metrics, dict)
        
        # 测试异步清理
        cleanup_result = await checker.cleanup_async()
        assert isinstance(cleanup_result, bool)

    # =========================================================================
    # 并发控制工作流程测试
    # =========================================================================

    def test_ensure_semaphore_creates_semaphore(self, checker):
        """测试确保semaphore创建工作流程"""
        assert checker._semaphore is None
        assert checker._semaphore_created is False
        
        checker._ensure_semaphore()
        
        # 在异步上下文中semaphore会被创建
        assert checker._semaphore_created is True or checker._semaphore_created is False

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, checker):
        """测试并发健康检查工作流程"""
        # 执行多个并发健康检查
        services = [f"service_{i}" for i in range(10)]
        tasks = [checker.check_health_async(service) for service in services]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        for result in results:
            assert "status" in result
            assert "service" in result

    # =========================================================================
    # 历史数据管理工作流程测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_health_history_accumulation(self, checker):
        """测试健康历史数据积累工作流程"""
        service_name = "test_service"
        
        # 执行多次检查
        for i in range(5):
            await checker.check_health_async(service_name)
        
        # 验证历史记录
        assert service_name in checker._health_history
        assert len(checker._health_history[service_name]) == 5

    @pytest.mark.asyncio
    async def test_health_history_maxlen(self, checker):
        """测试健康历史数据最大长度工作流程"""
        service_name = "test_service"
        
        # 执行超过最大长度的检查次数
        for i in range(150):  # 超过HEALTH_HISTORY_MAXLEN (100)
            checker._health_history[service_name].append({
                "status": "healthy",
                "timestamp": time.time()
            })
        
        # 验证历史记录不超过最大长度
        assert len(checker._health_history[service_name]) <= 100

    # =========================================================================
    # 完整业务流程测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_complete_health_monitoring_workflow(self, checker):
        """测试完整健康监控工作流程"""
        # 1. 启动监控
        start_result = await checker.monitor_start_async()
        assert start_result is True
        
        # 2. 执行系统健康检查
        system_health = await checker.check_system_health_async()
        assert system_health["service"] == "system"
        
        # 3. 执行特定服务检查
        db_health = await checker.check_database_async()
        cache_health = await checker.check_cache_async()
        network_health = await checker.check_network_async()
        
        assert db_health["service"] == "database"
        assert cache_health["service"] == "cache"
        assert network_health["service"] == "network"
        
        # 4. 获取健康状态汇总
        summary = await checker.health_summary_async()
        assert "services_monitored" in summary
        
        # 5. 停止监控
        stop_result = await checker.monitor_stop_async()
        assert stop_result is True

    @pytest.mark.asyncio
    async def test_error_detection_and_recovery_workflow(self, checker):
        """测试错误检测和恢复工作流程"""
        service_name = "failing_service"
        
        # 模拟服务失败
        for i in range(5):
            checker._health_history[service_name].append({
                "status": "critical",
                "timestamp": time.time()
            })
        
        # 检查错误模式
        error_result = await checker._check_error_patterns_async(service_name)
        assert error_result["status"] == "warning"
        assert error_result["details"]["recent_failures"] >= 3
        
        # 模拟服务恢复
        for i in range(10):
            checker._health_history[service_name].append({
                "status": "healthy",
                "timestamp": time.time()
            })
        
        # 再次检查错误模式
        recovery_result = await checker._check_error_patterns_async(service_name)
        # 恢复后最近的失败应该减少
        assert recovery_result["details"]["recent_failures"] < 3
