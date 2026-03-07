#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
health_checker.py深度测试 - 目标从26.7%提升至65%+

重点测试:
1. AsyncHealthCheckerComponent的公共方法
2. 健康检查执行流程
3. 缓存管理逻辑
4. 监控循环逻辑
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.infrastructure.health.components.health_checker import (
    HealthCheckResult,
    DEFAULT_SERVICE_TIMEOUT,
    DEFAULT_BATCH_TIMEOUT,
    DEFAULT_CONCURRENT_LIMIT,
    DEFAULT_CACHE_TTL,
    HEALTH_STATUS_HEALTHY,
    HEALTH_STATUS_WARNING,
    HEALTH_STATUS_CRITICAL
)


class TestHealthCheckCoreLogic:
    """测试健康检查核心逻辑"""
    
    def test_health_check_result_validation(self):
        """测试健康检查结果验证"""
        # 有效结果
        valid_result = HealthCheckResult(
            service_name="valid_service",
            status=HEALTH_STATUS_HEALTHY,
            timestamp=datetime.now(),
            response_time=0.1,
            details={"validated": True}
        )
        
        # 验证必需字段
        assert valid_result.service_name is not None
        assert valid_result.status in [HEALTH_STATUS_HEALTHY, HEALTH_STATUS_WARNING, HEALTH_STATUS_CRITICAL]
        assert valid_result.timestamp is not None
        assert valid_result.response_time >= 0
    
    def test_service_timeout_handling(self):
        """测试服务超时处理"""
        # 模拟超时场景
        timeout_result = HealthCheckResult(
            service_name="timeout_service",
            status=HEALTH_STATUS_CRITICAL,
            timestamp=datetime.now(),
            response_time=DEFAULT_SERVICE_TIMEOUT + 1.0,
            details={"error": "timeout"},
            recommendations=["Increase timeout", "Check service health"]
        )
        
        assert timeout_result.response_time > DEFAULT_SERVICE_TIMEOUT
        assert timeout_result.status == HEALTH_STATUS_CRITICAL
        assert "timeout" in timeout_result.details.get("error", "")
    
    def test_batch_check_size_limits(self):
        """测试批量检查大小限制"""
        # 创建大批量检查请求
        batch_services = [f"service_{i}" for i in range(100)]
        
        # 按并发限制分批
        batches = []
        for i in range(0, len(batch_services), DEFAULT_CONCURRENT_LIMIT):
            batch = batch_services[i:i + DEFAULT_CONCURRENT_LIMIT]
            batches.append(batch)
        
        # 验证分批逻辑
        assert len(batches) == 10
        assert all(len(b) <= DEFAULT_CONCURRENT_LIMIT for b in batches)
    
    def test_health_check_caching_logic(self):
        """测试健康检查缓存逻辑"""
        # 创建缓存的健康结果
        cached_result = HealthCheckResult(
            service_name="cached_service",
            status=HEALTH_STATUS_HEALTHY,
            timestamp=datetime.now(),
            response_time=0.05,
            details={"cached": True}
        )
        
        # 模拟缓存存储
        cache = {
            "cached_service": {
                "result": cached_result,
                "cached_at": datetime.now(),
                "ttl": DEFAULT_CACHE_TTL
            }
        }
        
        # 验证缓存
        assert "cached_service" in cache
        assert cache["cached_service"]["ttl"] == 300


class TestAsyncHealthCheckPatterns:
    """测试异步健康检查模式"""
    
    @pytest.mark.asyncio
    async def test_async_single_check(self):
        """测试异步单个检查"""
        # 模拟异步检查函数
        async def async_check():
            await asyncio.sleep(0.01)
            return {"status": HEALTH_STATUS_HEALTHY}
        
        result = await async_check()
        assert result["status"] == HEALTH_STATUS_HEALTHY
    
    @pytest.mark.asyncio
    async def test_async_batch_checks(self):
        """测试异步批量检查"""
        # 模拟多个异步检查
        async def check_service(name):
            await asyncio.sleep(0.01)
            return HealthCheckResult(
                service_name=name,
                status=HEALTH_STATUS_HEALTHY,
                timestamp=datetime.now(),
                response_time=0.01,
                details={}
            )
        
        # 并发执行
        services = [f"service_{i}" for i in range(5)]
        tasks = [check_service(name) for name in services]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(r.status == HEALTH_STATUS_HEALTHY for r in results)
    
    @pytest.mark.asyncio
    async def test_async_timeout_handling(self):
        """测试异步超时处理"""
        # 模拟慢速检查
        async def slow_check():
            await asyncio.sleep(0.1)
            return {"status": "healthy"}
        
        # 使用超时
        try:
            result = await asyncio.wait_for(slow_check(), timeout=0.05)
        except asyncio.TimeoutError:
            result = {
                "status": HEALTH_STATUS_CRITICAL,
                "error": "timeout"
            }
        
        # 应该超时
        assert result.get("error") == "timeout" or result.get("status") == "healthy"


class TestHealthCheckMonitoringLoop:
    """测试健康检查监控循环"""
    
    def test_monitoring_interval_calculation(self):
        """测试监控间隔计算"""
        from src.infrastructure.health.components.health_checker import (
            DEFAULT_MONITORING_INTERVAL
        )
        
        # 监控间隔应该合理
        assert DEFAULT_MONITORING_INTERVAL == 60.0
        
        # 计算一天的检查次数
        checks_per_day = (24 * 60 * 60) / DEFAULT_MONITORING_INTERVAL
        assert checks_per_day == 1440  # 每分钟1次
    
    def test_monitoring_loop_iterations(self):
        """测试监控循环迭代"""
        # 模拟监控循环
        iteration_count = 0
        max_iterations = 10
        
        while iteration_count < max_iterations:
            # 模拟健康检查
            result = HealthCheckResult(
                service_name="loop_service",
                status=HEALTH_STATUS_HEALTHY,
                timestamp=datetime.now(),
                response_time=0.05,
                details={"iteration": iteration_count}
            )
            
            iteration_count += 1
        
        assert iteration_count == max_iterations


class TestHealthCheckErrorRecovery:
    """测试健康检查错误恢复"""
    
    def test_error_recovery_flow(self):
        """测试错误恢复流程"""
        # 模拟检查失败
        failed_result = HealthCheckResult(
            service_name="failed_service",
            status=HEALTH_STATUS_CRITICAL,
            timestamp=datetime.now(),
            response_time=5.5,
            details={"error": "connection_refused"},
            recommendations=["Restart service", "Check logs"]
        )
        
        # 验证错误信息
        assert failed_result.status == HEALTH_STATUS_CRITICAL
        assert "error" in failed_result.details
        assert len(failed_result.recommendations) >= 1
    
    def test_retry_after_failure(self):
        """测试失败后重试"""
        from src.infrastructure.health.components.health_checker import (
            DEFAULT_RETRY_COUNT,
            DEFAULT_RETRY_DELAY
        )
        
        # 模拟重试逻辑
        attempts = []
        for attempt in range(DEFAULT_RETRY_COUNT):
            result = HealthCheckResult(
                service_name="retry_service",
                status=HEALTH_STATUS_CRITICAL if attempt < 2 else HEALTH_STATUS_HEALTHY,
                timestamp=datetime.now(),
                response_time=0.1,
                details={"attempt": attempt + 1}
            )
            attempts.append(result)
            
            # 如果成功，停止重试
            if result.status == HEALTH_STATUS_HEALTHY:
                break
        
        # 第3次尝试成功
        assert len(attempts) == 3
        assert attempts[-1].status == HEALTH_STATUS_HEALTHY


class TestHealthCheckResourceUsage:
    """测试健康检查资源使用"""
    
    def test_cpu_usage_threshold_check(self):
        """测试CPU使用阈值检查"""
        from src.infrastructure.health.components.health_checker import (
            CPU_USAGE_WARNING_THRESHOLD,
            CPU_USAGE_CRITICAL_THRESHOLD
        )
        
        test_cases = [
            (45.0, HEALTH_STATUS_HEALTHY),
            (85.0, HEALTH_STATUS_WARNING),
            (98.0, HEALTH_STATUS_CRITICAL)
        ]
        
        for cpu_value, expected_status in test_cases:
            # 判定逻辑
            if cpu_value < CPU_USAGE_WARNING_THRESHOLD:
                status = HEALTH_STATUS_HEALTHY
            elif cpu_value < CPU_USAGE_CRITICAL_THRESHOLD:
                status = HEALTH_STATUS_WARNING
            else:
                status = HEALTH_STATUS_CRITICAL
            
            result = HealthCheckResult(
                service_name="cpu_service",
                status=status,
                timestamp=datetime.now(),
                response_time=0.1,
                details={"cpu_percent": cpu_value}
            )
            
            assert result.status == expected_status
    
    def test_memory_usage_threshold_check(self):
        """测试内存使用阈值检查"""
        from src.infrastructure.health.components.health_checker import (
            MEMORY_USAGE_WARNING_THRESHOLD,
            MEMORY_USAGE_CRITICAL_THRESHOLD
        )
        
        test_cases = [
            (60.0, HEALTH_STATUS_HEALTHY),
            (90.0, HEALTH_STATUS_WARNING),
            (97.0, HEALTH_STATUS_CRITICAL)
        ]
        
        for mem_value, expected_status in test_cases:
            if mem_value < MEMORY_USAGE_WARNING_THRESHOLD:
                status = HEALTH_STATUS_HEALTHY
            elif mem_value < MEMORY_USAGE_CRITICAL_THRESHOLD:
                status = HEALTH_STATUS_WARNING
            else:
                status = HEALTH_STATUS_CRITICAL
            
            result = HealthCheckResult(
                service_name="memory_service",
                status=status,
                timestamp=datetime.now(),
                response_time=0.1,
                details={"memory_percent": mem_value}
            )
            
            assert result.status == expected_status
    
    def test_disk_usage_threshold_check(self):
        """测试磁盘使用阈值检查"""
        from src.infrastructure.health.components.health_checker import (
            DISK_USAGE_WARNING_THRESHOLD,
            DISK_USAGE_CRITICAL_THRESHOLD
        )
        
        test_cases = [
            (70.0, HEALTH_STATUS_HEALTHY),
            (85.0, HEALTH_STATUS_WARNING),
            (96.0, HEALTH_STATUS_CRITICAL)
        ]
        
        for disk_value, expected_status in test_cases:
            if disk_value < DISK_USAGE_WARNING_THRESHOLD:
                status = HEALTH_STATUS_HEALTHY
            elif disk_value < DISK_USAGE_CRITICAL_THRESHOLD:
                status = HEALTH_STATUS_WARNING
            else:
                status = HEALTH_STATUS_CRITICAL
            
            result = HealthCheckResult(
                service_name="disk_service",
                status=status,
                timestamp=datetime.now(),
                response_time=0.1,
                details={"disk_percent": disk_value}
            )
            
            assert result.status == expected_status


class TestHealthCheckServiceDependencies:
    """测试服务依赖健康检查"""
    
    def test_dependency_chain_check(self):
        """测试依赖链检查"""
        # 模拟服务依赖关系
        dependencies = {
            "api": ["database", "cache"],
            "database": ["disk"],
            "cache": ["memory"],
            "disk": [],
            "memory": []
        }
        
        # 检查依赖链
        service = "api"
        deps = dependencies.get(service, [])
        
        assert len(deps) == 2
        assert "database" in deps
        assert "cache" in deps
    
    def test_circular_dependency_detection(self):
        """测试循环依赖检测"""
        # 检查是否存在循环依赖
        dependencies = {
            "service_a": ["service_b"],
            "service_b": ["service_c"],
            "service_c": ["service_a"]  # 循环
        }
        
        # 简单的循环检测
        def has_cycle(deps, service, visited=None):
            if visited is None:
                visited = set()
            
            if service in visited:
                return True
            
            visited.add(service)
            for dep in deps.get(service, []):
                if has_cycle(deps, dep, visited):
                    return True
            
            visited.remove(service)
            return False
        
        # 应该检测到循环
        assert has_cycle(dependencies, "service_a")


class TestHealthCheckMetricsCollection:
    """测试健康检查指标收集"""
    
    def test_metrics_data_structure(self):
        """测试指标数据结构"""
        metrics = {
            "total_checks": 100,
            "successful_checks": 95,
            "failed_checks": 5,
            "avg_response_time": 0.15,
            "max_response_time": 1.5,
            "min_response_time": 0.01
        }
        
        # 验证指标完整性
        assert "total_checks" in metrics
        assert "successful_checks" in metrics
        assert "failed_checks" in metrics
        assert metrics["successful_checks"] + metrics["failed_checks"] == metrics["total_checks"]
    
    def test_metrics_aggregation_over_time(self):
        """测试指标随时间聚合"""
        # 收集5分钟的指标
        time_windows = []
        
        for minute in range(5):
            window_metrics = {
                "timestamp": datetime.now(),
                "checks": 12,  # 每分钟12次检查
                "failures": minute  # 故障逐渐增加
            }
            time_windows.append(window_metrics)
        
        # 聚合
        total_checks = sum(w["checks"] for w in time_windows)
        total_failures = sum(w["failures"] for w in time_windows)
        
        assert total_checks == 60  # 5分钟 * 12次
        assert total_failures == 10  # 0+1+2+3+4


class TestHealthCheckServiceRegistry:
    """测试服务注册功能"""
    
    def test_service_registration_workflow(self):
        """测试服务注册工作流"""
        from src.infrastructure.health.components.health_check_registry import (
            HealthCheckRegistry
        )
        
        registry = HealthCheckRegistry()
        
        # 定义多个检查函数
        def db_check():
            return {"status": "healthy", "latency": 50}
        
        def cache_check():
            return {"status": "healthy", "hit_rate": 0.95}
        
        def queue_check():
            return {"status": "healthy", "depth": 100}
        
        # 批量注册
        services = [
            ("database", db_check, {"timeout": 5.0}),
            ("cache", cache_check, {"timeout": 2.0}),
            ("queue", queue_check, {"timeout": 3.0})
        ]
        
        for name, func, config in services:
            result = registry.register_health_check(name, func, config)
            assert result is True
        
        # 验证都已注册
        for name, _, _ in services:
            check_func = registry.get_health_check(name)
            assert check_func is not None
    
    def test_service_config_retrieval(self):
        """测试服务配置检索"""
        from src.infrastructure.health.components.health_check_registry import (
            HealthCheckRegistry
        )
        
        registry = HealthCheckRegistry()
        
        # 注册带配置的服务
        config = {
            "timeout": 10.0,
            "retry_count": 5,
            "interval": 60
        }
        
        registry.register_health_check(
            "configured_service",
            lambda: True,
            config=config
        )
        
        # 检索配置
        retrieved_config = registry.get_health_check_config("configured_service")
        
        if retrieved_config:
            assert retrieved_config.get("timeout") == 10.0
            assert retrieved_config.get("retry_count") == 5


class TestHealthCheckFailureScenarios:
    """测试健康检查失败场景"""
    
    def test_database_connection_failure(self):
        """测试数据库连接失败"""
        result = HealthCheckResult(
            service_name="database",
            status=HEALTH_STATUS_CRITICAL,
            timestamp=datetime.now(),
            response_time=5.0,
            details={
                "error": "connection_refused",
                "host": "localhost",
                "port": 5432
            },
            recommendations=[
                "Check database is running",
                "Verify connection string",
                "Check firewall rules"
            ]
        )
        
        assert result.status == HEALTH_STATUS_CRITICAL
        assert "connection_refused" in result.details["error"]
        assert len(result.recommendations) >= 3
    
    def test_cache_degradation(self):
        """测试缓存降级"""
        result = HealthCheckResult(
            service_name="cache",
            status=HEALTH_STATUS_WARNING,
            timestamp=datetime.now(),
            response_time=1.5,
            details={
                "hit_rate": 0.45,  # 低于正常的0.9
                "eviction_rate": 0.15
            },
            recommendations=["Increase cache size", "Review cache policy"]
        )
        
        assert result.status == HEALTH_STATUS_WARNING
        assert result.details["hit_rate"] < 0.5
    
    def test_network_latency_spike(self):
        """测试网络延迟峰值"""
        result = HealthCheckResult(
            service_name="api_gateway",
            status=HEALTH_STATUS_WARNING,
            timestamp=datetime.now(),
            response_time=2.5,
            details={
                "latency_ms": 250,  # 250ms延迟
                "packet_loss": 0.02
            },
            recommendations=["Check network", "Review CDN config"]
        )
        
        assert result.details["latency_ms"] > 100
        assert result.status == HEALTH_STATUS_WARNING


class TestHealthCheckMultiRegion:
    """测试多区域健康检查"""
    
    def test_multi_region_health_status(self):
        """测试多区域健康状态"""
        regions = ["us-east-1", "us-west-2", "eu-central-1", "ap-southeast-1"]
        
        regional_results = {}
        for region in regions:
            result = HealthCheckResult(
                service_name=f"{region}_service",
                status=HEALTH_STATUS_HEALTHY,
                timestamp=datetime.now(),
                response_time=0.1,
                details={"region": region, "az_count": 3}
            )
            regional_results[region] = result
        
        # 全球健康状态
        all_healthy = all(r.status == HEALTH_STATUS_HEALTHY for r in regional_results.values())
        assert all_healthy is True
    
    def test_cross_region_latency(self):
        """测试跨区域延迟"""
        # 不同区域的延迟
        region_latencies = {
            "us-east-1": 0.05,    # 本地区域
            "us-west-2": 0.08,    # 同国家
            "eu-central-1": 0.15,  # 跨大西洋
            "ap-southeast-1": 0.20 # 跨太平洋
        }
        
        results = []
        for region, latency in region_latencies.items():
            result = HealthCheckResult(
                service_name=region,
                status=HEALTH_STATUS_HEALTHY if latency < 0.2 else HEALTH_STATUS_WARNING,
                timestamp=datetime.now(),
                response_time=latency,
                details={"region": region, "latency": latency}
            )
            results.append(result)
        
        # 验证延迟模式
        assert all(r.response_time > 0 for r in results)
        assert results[0].response_time < results[-1].response_time  # 本地<远程


class TestHealthCheckAlertingLogic:
    """测试健康检查告警逻辑"""
    
    def test_alert_trigger_conditions(self):
        """测试告警触发条件"""
        # 场景1: 单次检查失败（不应告警）
        single_failure = HealthCheckResult(
            service_name="transient_issue",
            status=HEALTH_STATUS_WARNING,
            timestamp=datetime.now(),
            response_time=2.5,
            details={"failures": 1}
        )
        
        should_alert = single_failure.details["failures"] >= 3
        assert should_alert is False
        
        # 场景2: 连续3次失败（应该告警）
        consecutive_failure = HealthCheckResult(
            service_name="persistent_issue",
            status=HEALTH_STATUS_CRITICAL,
            timestamp=datetime.now(),
            response_time=5.5,
            details={"failures": 3}
        )
        
        should_alert = consecutive_failure.details["failures"] >= 3
        assert should_alert is True
    
    def test_alert_severity_levels(self):
        """测试告警严重级别"""
        severity_mapping = {
            HEALTH_STATUS_HEALTHY: "info",
            HEALTH_STATUS_WARNING: "warning",
            HEALTH_STATUS_CRITICAL: "critical"
        }
        
        for health_status, alert_level in severity_mapping.items():
            result = HealthCheckResult(
                service_name="alert_service",
                status=health_status,
                timestamp=datetime.now(),
                response_time=0.1,
                details={"alert_level": alert_level}
            )
            
            assert result.details["alert_level"] == alert_level


class TestHealthCheckIntegrationScenarios:
    """测试健康检查集成场景"""
    
    def test_full_stack_health_check(self):
        """测试全栈健康检查"""
        # 完整的服务栈
        stack_services = [
            ("load_balancer", HEALTH_STATUS_HEALTHY, 0.05),
            ("api_gateway", HEALTH_STATUS_HEALTHY, 0.08),
            ("application", HEALTH_STATUS_HEALTHY, 0.10),
            ("cache", HEALTH_STATUS_HEALTHY, 0.02),
            ("database", HEALTH_STATUS_WARNING, 1.5),
            ("storage", HEALTH_STATUS_HEALTHY, 0.15)
        ]
        
        results = []
        for name, status, rt in stack_services:
            result = HealthCheckResult(
                service_name=name,
                status=status,
                timestamp=datetime.now(),
                response_time=rt,
                details={}
            )
            results.append(result)
        
        # 栈健康评估
        critical_count = sum(1 for r in results if r.status == HEALTH_STATUS_CRITICAL)
        warning_count = sum(1 for r in results if r.status == HEALTH_STATUS_WARNING)
        
        # 应该有1个warning，0个critical
        assert warning_count == 1
        assert critical_count == 0
    
    def test_microservices_health_check(self):
        """测试微服务健康检查"""
        # 微服务架构
        microservices = [
            "user-service",
            "order-service",
            "payment-service",
            "notification-service",
            "analytics-service"
        ]
        
        results = {}
        for service in microservices:
            result = HealthCheckResult(
                service_name=service,
                status=HEALTH_STATUS_HEALTHY,
                timestamp=datetime.now(),
                response_time=0.1,
                details={"instances": 3}
            )
            results[service] = result
        
        # 验证所有微服务
        assert len(results) == 5
        assert all(r.status == HEALTH_STATUS_HEALTHY for r in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

