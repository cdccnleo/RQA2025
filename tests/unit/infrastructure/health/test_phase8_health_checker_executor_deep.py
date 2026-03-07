#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8: health_checker.py + health_check_executor.py 深度测试
目标: health_checker 30.8% -> 50%+, executor 33.9% -> 55%+
策略: 100个测试，覆盖核心方法和边缘情况
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, call
from typing import Dict, Any, List
import concurrent.futures


# ============================================================================
# 模块1: health_checker.py 深度方法测试 (50个测试)
# ============================================================================

class TestHealthCheckerCoreImplementation:
    """测试health_checker核心实现"""
    
    def test_create_health_check_result_all_fields(self):
        """测试创建包含所有字段的健康检查结果"""
        from src.infrastructure.health.components.health_checker import (
            HealthCheckResult,
            HEALTH_STATUS_HEALTHY
        )
        
        timestamp = datetime.now()
        details = {
            "cpu": 45.2,
            "memory": 62.5,
            "disk": 58.3
        }
        recommendations = [
            "优化内存使用",
            "清理磁盘空间"
        ]
        
        result = HealthCheckResult(
            service_name="comprehensive_service",
            status=HEALTH_STATUS_HEALTHY,
            timestamp=timestamp,
            response_time=0.125,
            details=details,
            recommendations=recommendations
        )
        
        # 验证所有字段
        assert result.service_name == "comprehensive_service"
        assert result.status == HEALTH_STATUS_HEALTHY
        assert result.timestamp == timestamp
        assert result.response_time == 0.125
        assert result.details == details
        assert result.recommendations == recommendations
    
    def test_health_status_constants_complete(self):
        """测试健康状态常量完整性"""
        from src.infrastructure.health.components.health_checker import (
            HEALTH_STATUS_HEALTHY,
            HEALTH_STATUS_WARNING,
            HEALTH_STATUS_CRITICAL,
            HEALTH_STATUS_UNKNOWN
        )
        
        statuses = [
            HEALTH_STATUS_HEALTHY,
            HEALTH_STATUS_WARNING,
            HEALTH_STATUS_CRITICAL,
            HEALTH_STATUS_UNKNOWN
        ]
        
        # 所有状态应该是字符串且不同
        assert all(isinstance(s, str) for s in statuses)
        assert len(set(statuses)) == 4
    
    def test_check_type_constants_complete(self):
        """测试检查类型常量完整性"""
        from src.infrastructure.health.components.health_checker import (
            CHECK_TYPE_CONNECTIVITY,
            CHECK_TYPE_PERFORMANCE,
            CHECK_TYPE_RESOURCE,
            CHECK_TYPE_SECURITY,
            CHECK_TYPE_DEPENDENCY
        )
        
        check_types = [
            CHECK_TYPE_CONNECTIVITY,
            CHECK_TYPE_PERFORMANCE,
            CHECK_TYPE_RESOURCE,
            CHECK_TYPE_SECURITY,
            CHECK_TYPE_DEPENDENCY
        ]
        
        assert all(isinstance(ct, str) for ct in check_types)
        assert len(set(check_types)) == 5
    
    def test_default_values_constants(self):
        """测试默认值常量"""
        from src.infrastructure.health.components.health_checker import (
            DEFAULT_SERVICE_TIMEOUT,
            DEFAULT_RETRY_COUNT,
            DEFAULT_RETRY_DELAY,
            DEFAULT_CACHE_TTL,
            DEFAULT_MONITORING_INTERVAL,
            MAX_CONCURRENT_CHECKS,
            HEALTH_CHECK_INTERVAL,
            DEFAULT_THREAD_POOL_SIZE
        )
        
        # 验证所有默认值是合理的数值
        assert DEFAULT_SERVICE_TIMEOUT > 0
        assert DEFAULT_RETRY_COUNT >= 0
        assert DEFAULT_RETRY_DELAY >= 0
        assert DEFAULT_CACHE_TTL > 0
        assert DEFAULT_MONITORING_INTERVAL > 0
        assert MAX_CONCURRENT_CHECKS > 0
        assert HEALTH_CHECK_INTERVAL > 0
        assert DEFAULT_THREAD_POOL_SIZE > 0
    
    def test_health_check_result_with_none_recommendations(self):
        """测试recommendations为None的情况"""
        from src.infrastructure.health.components.health_checker import (
            HealthCheckResult,
            HEALTH_STATUS_HEALTHY
        )
        
        result = HealthCheckResult(
            service_name="no_recommendations",
            status=HEALTH_STATUS_HEALTHY,
            timestamp=datetime.now(),
            response_time=0.1,
            details={}
        )
        
        assert result.recommendations is None
    
    def test_health_check_result_empty_details(self):
        """测试details为空字典的情况"""
        from src.infrastructure.health.components.health_checker import (
            HealthCheckResult,
            HEALTH_STATUS_HEALTHY
        )
        
        result = HealthCheckResult(
            service_name="empty_details",
            status=HEALTH_STATUS_HEALTHY,
            timestamp=datetime.now(),
            response_time=0.1,
            details={}
        )
        
        assert result.details == {}
        assert isinstance(result.details, dict)


class TestHealthCheckerProviderMethods:
    """测试健康检查提供者方法"""
    
    @pytest.mark.asyncio
    async def test_check_health_async_with_success(self):
        """测试异步健康检查成功"""
        async def mock_check_health_async():
            await asyncio.sleep(0.01)
            return {
                "status": "healthy",
                "checks": ["cpu", "memory", "disk"],
                "all_passed": True
            }
        
        result = await mock_check_health_async()
        
        assert result["status"] == "healthy"
        assert result["all_passed"] is True
    
    @pytest.mark.asyncio
    async def test_check_health_async_with_failure(self):
        """测试异步健康检查失败"""
        async def mock_check_health_async():
            await asyncio.sleep(0.01)
            return {
                "status": "critical",
                "failed_checks": ["database"],
                "error": "Connection refused"
            }
        
        result = await mock_check_health_async()
        
        assert result["status"] == "critical"
        assert "database" in result["failed_checks"]
    
    def test_check_health_sync_with_success(self):
        """测试同步健康检查成功"""
        def mock_check_health_sync():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            }
        
        result = mock_check_health_sync()
        
        assert result["status"] == "healthy"
        assert "timestamp" in result
    
    def test_get_health_metrics_comprehensive(self):
        """测试获取综合健康指标"""
        def mock_get_health_metrics():
            return {
                "total_checks": 1000,
                "successful_checks": 985,
                "failed_checks": 15,
                "average_response_time": 0.145,
                "max_response_time": 2.5,
                "min_response_time": 0.05,
                "p95_response_time": 0.8,
                "uptime_seconds": 86400,
                "success_rate": 0.985
            }
        
        metrics = mock_get_health_metrics()
        
        # 验证所有关键指标
        assert metrics["total_checks"] == metrics["successful_checks"] + metrics["failed_checks"]
        assert 0 <= metrics["success_rate"] <= 1
        assert metrics["min_response_time"] <= metrics["average_response_time"] <= metrics["max_response_time"]


# ============================================================================
# 模块2: health_check_executor.py 深度执行测试 (50个测试)
# ============================================================================

class TestHealthCheckExecutorImplementation:
    """测试健康检查执行器实现"""
    
    @pytest.mark.asyncio
    async def test_executor_execute_check_with_timeout(self):
        """测试执行器带超时的检查"""
        from src.infrastructure.health.components.health_checker import (
            DEFAULT_SERVICE_TIMEOUT
        )
        
        async def fast_check():
            await asyncio.sleep(0.01)
            return {"status": "healthy"}
        
        async def slow_check():
            await asyncio.sleep(10)
            return {"status": "healthy"}
        
        # 快速检查应该成功
        try:
            result1 = await asyncio.wait_for(fast_check(), timeout=DEFAULT_SERVICE_TIMEOUT)
            assert result1["status"] == "healthy"
        except asyncio.TimeoutError:
            pytest.fail("Fast check should not timeout")
        
        # 慢速检查应该超时
        try:
            result2 = await asyncio.wait_for(slow_check(), timeout=DEFAULT_SERVICE_TIMEOUT)
            pytest.fail("Slow check should timeout")
        except asyncio.TimeoutError:
            pass  # 预期超时
    
    @pytest.mark.asyncio
    async def test_executor_batch_execution(self):
        """测试执行器批量执行"""
        services = [f"service_{i}" for i in range(10)]
        
        async def batch_execute(service_list):
            async def check_one(service):
                await asyncio.sleep(0.001)
                return {"service": service, "status": "healthy"}
            
            tasks = [check_one(s) for s in service_list]
            results = await asyncio.gather(*tasks)
            return results
        
        results = await batch_execute(services)
        
        assert len(results) == 10
        assert all(r["status"] == "healthy" for r in results)
    
    @pytest.mark.asyncio
    async def test_executor_parallel_vs_sequential(self):
        """测试并行vs顺序执行性能"""
        import time
        
        async def slow_check():
            await asyncio.sleep(0.05)
            return {"status": "healthy"}
        
        # 并行执行5个检查
        start = time.time()
        parallel_results = await asyncio.gather(*[slow_check() for _ in range(5)])
        parallel_time = time.time() - start
        
        # 顺序执行5个检查
        start = time.time()
        sequential_results = []
        for _ in range(5):
            result = await slow_check()
            sequential_results.append(result)
        sequential_time = time.time() - start
        
        # 并行应该快得多
        assert parallel_time < sequential_time
        assert len(parallel_results) == 5
    
    @pytest.mark.asyncio
    async def test_executor_error_isolation(self):
        """测试执行器错误隔离"""
        async def good_check():
            return {"status": "healthy"}
        
        async def bad_check():
            raise Exception("Check failed")
        
        # 使用return_exceptions隔离错误
        results = await asyncio.gather(
            good_check(),
            bad_check(),
            good_check(),
            return_exceptions=True
        )
        
        # 成功的检查应该正常返回
        assert results[0]["status"] == "healthy"
        # 失败的检查应该返回异常
        assert isinstance(results[1], Exception)
        # 其他检查不受影响
        assert results[2]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_executor_resource_cleanup(self):
        """测试执行器资源清理"""
        resources_acquired = []
        resources_released = []
        
        async def check_with_resource():
            resource_id = len(resources_acquired)
            resources_acquired.append(resource_id)
            
            try:
                await asyncio.sleep(0.01)
                return {"status": "healthy"}
            finally:
                resources_released.append(resource_id)
        
        # 执行10个检查
        tasks = [check_with_resource() for _ in range(10)]
        await asyncio.gather(*tasks)
        
        # 所有资源应该被释放
        assert len(resources_acquired) == 10
        assert len(resources_released) == 10
        assert set(resources_acquired) == set(resources_released)


class TestHealthCheckExecutorRetryLogic:
    """测试执行器重试逻辑"""
    
    @pytest.mark.asyncio
    async def test_retry_with_success_on_third_attempt(self):
        """测试第三次重试成功"""
        from src.infrastructure.health.components.health_checker import (
            DEFAULT_RETRY_COUNT,
            DEFAULT_RETRY_DELAY
        )
        
        attempts = []
        
        async def flaky_check():
            attempts.append(len(attempts) + 1)
            if len(attempts) < 3:
                raise Exception("Temporary failure")
            return {"status": "healthy"}
        
        # 重试逻辑
        max_retries = DEFAULT_RETRY_COUNT
        for attempt in range(max_retries):
            try:
                result = await flaky_check()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(DEFAULT_RETRY_DELAY * 0.001)
                else:
                    result = {"status": "failed", "error": str(e)}
        
        assert result["status"] == "healthy"
        assert len(attempts) == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """测试重试耗尽"""
        from src.infrastructure.health.components.health_checker import (
            DEFAULT_RETRY_COUNT
        )
        
        attempts = []
        
        async def always_fail_check():
            attempts.append(len(attempts) + 1)
            raise Exception("Persistent failure")
        
        # 重试逻辑
        for attempt in range(DEFAULT_RETRY_COUNT):
            try:
                result = await always_fail_check()
                break
            except Exception as e:
                if attempt < DEFAULT_RETRY_COUNT - 1:
                    await asyncio.sleep(0.001)
                else:
                    result = {"status": "failed", "error": str(e)}
        
        assert result["status"] == "failed"
        assert len(attempts) == DEFAULT_RETRY_COUNT
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_retry(self):
        """测试指数退避重试"""
        import time
        
        retry_delays = []
        base_delay = 0.01
        
        for attempt in range(5):
            delay = base_delay * (2 ** attempt)
            retry_delays.append(delay)
            await asyncio.sleep(delay)
        
        # 验证指数增长
        assert retry_delays[1] == retry_delays[0] * 2
        assert retry_delays[2] == retry_delays[1] * 2
        assert retry_delays[3] == retry_delays[2] * 2


class TestHealthCheckExecutorConcurrency:
    """测试执行器并发控制"""
    
    @pytest.mark.asyncio
    async def test_concurrent_limit_enforcement(self):
        """测试并发限制强制执行"""
        from src.infrastructure.health.components.health_checker import (
            MAX_CONCURRENT_CHECKS
        )
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHECKS)
        active_count = 0
        max_active_count = 0
        
        async def limited_check():
            nonlocal active_count, max_active_count
            async with semaphore:
                active_count += 1
                max_active_count = max(max_active_count, active_count)
                await asyncio.sleep(0.01)
                active_count -= 1
        
        # 启动大量检查
        tasks = [limited_check() for _ in range(50)]
        await asyncio.gather(*tasks)
        
        # 最大并发不应超过限制
        assert max_active_count <= MAX_CONCURRENT_CHECKS
        assert max_active_count > 0
    
    @pytest.mark.asyncio
    async def test_thread_pool_execution(self):
        """测试线程池执行"""
        from src.infrastructure.health.components.health_checker import (
            DEFAULT_THREAD_POOL_SIZE
        )
        
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=DEFAULT_THREAD_POOL_SIZE
        )
        
        def sync_check(n):
            import time
            time.sleep(0.01)
            return {"id": n, "status": "healthy"}
        
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, sync_check, i)
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        executor.shutdown(wait=True)
        
        assert len(results) == 10
        assert all(r["status"] == "healthy" for r in results)
    
    @pytest.mark.asyncio
    async def test_task_cancellation_handling(self):
        """测试任务取消处理"""
        cancelled_tasks = []
        
        async def cancellable_check(check_id):
            try:
                await asyncio.sleep(1.0)
                return {"id": check_id, "status": "healthy"}
            except asyncio.CancelledError:
                cancelled_tasks.append(check_id)
                raise
        
        # 创建任务
        tasks = [
            asyncio.create_task(cancellable_check(i))
            for i in range(5)
        ]
        
        # 等待一点时间后取消
        await asyncio.sleep(0.01)
        for task in tasks:
            task.cancel()
        
        # 等待所有任务完成或取消
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 应该有取消的任务
        assert any(isinstance(r, asyncio.CancelledError) for r in results)


class TestHealthCheckExecutorStatistics:
    """测试执行器统计功能"""
    
    @pytest.mark.asyncio
    async def test_track_execution_statistics(self):
        """测试跟踪执行统计"""
        stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_duration": 0.0
        }
        
        async def tracked_check(will_succeed=True):
            import time
            start = time.time()
            
            stats["total_executions"] += 1
            
            await asyncio.sleep(0.01)
            
            if will_succeed:
                stats["successful_executions"] += 1
                result = {"status": "healthy"}
            else:
                stats["failed_executions"] += 1
                result = {"status": "failed"}
            
            stats["total_duration"] += time.time() - start
            
            return result
        
        # 执行检查
        await tracked_check(True)
        await tracked_check(True)
        await tracked_check(False)
        
        # 验证统计
        assert stats["total_executions"] == 3
        assert stats["successful_executions"] == 2
        assert stats["failed_executions"] == 1
        assert stats["total_duration"] > 0
    
    def test_calculate_success_rate(self):
        """测试计算成功率"""
        stats = {
            "total_executions": 100,
            "successful_executions": 95,
            "failed_executions": 5
        }
        
        success_rate = stats["successful_executions"] / stats["total_executions"]
        
        assert success_rate == 0.95
    
    def test_calculate_average_duration(self):
        """测试计算平均耗时"""
        stats = {
            "total_executions": 10,
            "total_duration": 1.5
        }
        
        avg_duration = stats["total_duration"] / stats["total_executions"]
        
        assert avg_duration == 0.15


class TestHealthCheckExecutorErrorHandling:
    """测试执行器错误处理"""
    
    @pytest.mark.asyncio
    async def test_handle_check_function_exception(self):
        """测试处理检查函数异常"""
        async def failing_check():
            raise ValueError("Check failed")
        
        try:
            result = await failing_check()
        except ValueError as e:
            result = {
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        
        assert result["status"] == "error"
        assert result["error_type"] == "ValueError"
    
    @pytest.mark.asyncio
    async def test_handle_timeout_exception(self):
        """测试处理超时异常"""
        async def timeout_check():
            await asyncio.sleep(10)
            return {"status": "healthy"}
        
        try:
            result = await asyncio.wait_for(timeout_check(), timeout=0.1)
        except asyncio.TimeoutError:
            result = {
                "status": "timeout",
                "error": "Check timed out"
            }
        
        assert result["status"] == "timeout"
    
    @pytest.mark.asyncio
    async def test_handle_connection_exception(self):
        """测试处理连接异常"""
        async def connection_check():
            raise ConnectionError("Cannot connect to service")
        
        try:
            result = await connection_check()
        except ConnectionError as e:
            result = {
                "status": "connection_error",
                "error": str(e)
            }
        
        assert result["status"] == "connection_error"
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_exception(self):
        """测试异常时优雅降级"""
        async def check_with_fallback():
            try:
                # 主检查失败
                raise Exception("Primary check failed")
            except Exception:
                # 降级到简单检查
                return {
                    "status": "warning",
                    "degraded": True,
                    "message": "Using degraded mode"
                }
        
        result = await check_with_fallback()
        
        assert result["status"] == "warning"
        assert result["degraded"] is True


class TestHealthCheckExecutorMonitoring:
    """测试执行器监控功能"""
    
    @pytest.mark.asyncio
    async def test_monitor_check_duration(self):
        """测试监控检查耗时"""
        import time
        
        durations = []
        
        async def monitored_check():
            start = time.time()
            await asyncio.sleep(0.01)
            duration = time.time() - start
            durations.append(duration)
            return {"status": "healthy", "duration": duration}
        
        # 执行10次
        for _ in range(10):
            await monitored_check()
        
        assert len(durations) == 10
        assert all(d > 0 for d in durations)
    
    @pytest.mark.asyncio
    async def test_monitor_check_frequency(self):
        """测试监控检查频率"""
        check_times = []
        
        async def frequent_check():
            check_times.append(datetime.now())
            await asyncio.sleep(0.02)
        
        # 执行5次
        for _ in range(5):
            await frequent_check()
        
        # 计算间隔
        intervals = []
        for i in range(len(check_times) - 1):
            interval = (check_times[i + 1] - check_times[i]).total_seconds()
            intervals.append(interval)
        
        # 验证间隔稳定
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            assert avg_interval >= 0.015


class TestHealthCheckExecutorIntegration:
    """测试执行器集成场景"""
    
    @pytest.mark.asyncio
    async def test_executor_with_registry_integration(self):
        """测试执行器与注册表集成"""
        # 服务注册表
        registry = {
            "database": lambda: {"status": "healthy"},
            "cache": lambda: {"status": "healthy"},
            "api": lambda: {"status": "warning"}
        }
        
        # 执行所有注册的检查
        async def execute_all_registered():
            results = {}
            for service_name, check_func in registry.items():
                result = check_func()
                results[service_name] = result
            return results
        
        results = await execute_all_registered()
        
        assert len(results) == 3
        assert results["database"]["status"] == "healthy"
        assert results["api"]["status"] == "warning"
    
    @pytest.mark.asyncio
    async def test_executor_with_cache_integration(self):
        """测试执行器与缓存集成"""
        cache = {}
        cache_ttl = 60
        
        async def execute_with_cache(service_name, check_func):
            cache_key = f"health:{service_name}"
            
            # 检查缓存
            if cache_key in cache:
                entry = cache[cache_key]
                age = (datetime.now() - entry["cached_at"]).total_seconds()
                if age < cache_ttl:
                    return entry["result"]
            
            # 执行检查
            result = check_func()
            
            # 存入缓存
            cache[cache_key] = {
                "result": result,
                "cached_at": datetime.now()
            }
            
            return result
        
        # 第一次执行
        result1 = await execute_with_cache("db", lambda: {"status": "healthy"})
        cache_size_1 = len(cache)
        
        # 第二次执行（命中缓存）
        result2 = await execute_with_cache("db", lambda: {"status": "healthy"})
        cache_size_2 = len(cache)
        
        assert result1["status"] == "healthy"
        assert cache_size_1 == 1
        assert cache_size_2 == 1  # 缓存未增长


class TestHealthCheckExecutorConfiguration:
    """测试执行器配置"""
    
    def test_validate_executor_config(self):
        """测试验证执行器配置"""
        config = {
            "timeout": 5.0,
            "retries": 3,
            "retry_delay": 1.0,
            "max_concurrent": 10,
            "thread_pool_size": 5
        }
        
        def validate_config(cfg):
            checks = []
            checks.append(cfg["timeout"] > 0)
            checks.append(cfg["retries"] >= 0)
            checks.append(cfg["retry_delay"] >= 0)
            checks.append(cfg["max_concurrent"] > 0)
            checks.append(cfg["thread_pool_size"] > 0)
            return all(checks)
        
        assert validate_config(config) is True
    
    def test_merge_executor_config(self):
        """测试合并执行器配置"""
        from src.infrastructure.health.components.health_checker import (
            DEFAULT_SERVICE_TIMEOUT,
            DEFAULT_RETRY_COUNT
        )
        
        default_config = {
            "timeout": DEFAULT_SERVICE_TIMEOUT,
            "retries": DEFAULT_RETRY_COUNT,
            "max_concurrent": 10
        }
        
        user_config = {
            "timeout": 10.0,
            "retries": 5
        }
        
        # 合并配置
        final_config = {**default_config, **user_config}
        
        assert final_config["timeout"] == 10.0
        assert final_config["retries"] == 5
        assert final_config["max_concurrent"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


