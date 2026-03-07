#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 服务健康监控深度测试
测试SystemHealthChecker的健康检查、指标收集和服务监控功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import asyncio
import time
from datetime import datetime
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import pytest

from infrastructure.health.monitoring.health_checker import SystemHealthChecker
from infrastructure.health.components.health_checker import HealthCheckResult


class TestSystemHealthCheckerInitialization:
    """SystemHealthChecker初始化测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        checker = SystemHealthChecker()

        assert checker.metrics_collector is None
        assert isinstance(checker._service_checks, dict)
        assert isinstance(checker._health_history, dict)
        assert checker._max_history_size == 100
        assert len(checker._service_checks) > 0  # 应该有系统检查

    def test_initialization_with_metrics_collector(self):
        """测试带指标收集器的初始化"""
        mock_collector = MagicMock()
        checker = SystemHealthChecker(mock_collector)

        assert checker.metrics_collector == mock_collector


class TestSystemHealthCheckerServiceManagement:
    """SystemHealthChecker服务管理测试"""

    @pytest.fixture
    def checker(self):
        """SystemHealthChecker fixture"""
        return SystemHealthChecker()

    def test_register_service(self, checker):
        """测试注册服务"""
        def dummy_check():
            return {"status": "healthy", "message": "OK"}

        checker.register_service("test_service", dummy_check)

        assert "test_service" in checker._service_checks
        assert checker._service_checks["test_service"] == dummy_check

    def test_register_duplicate_service(self, checker):
        """测试注册重复服务"""
        def check1():
            return {"status": "healthy"}

        def check2():
            return {"status": "unhealthy"}

        checker.register_service("duplicate_service", check1)
        checker.register_service("duplicate_service", check2)  # 应该覆盖

        assert checker._service_checks["duplicate_service"] == check2

    def test_unregister_service(self, checker):
        """测试注销服务"""
        def dummy_check():
            return {"status": "healthy"}

        checker.register_service("test_service", dummy_check)
        assert "test_service" in checker._service_checks

        checker.unregister_service("test_service")
        assert "test_service" not in checker._service_checks

    def test_unregister_nonexistent_service(self, checker):
        """测试注销不存在的服务"""
        # 不应该抛出异常
        checker.unregister_service("nonexistent_service")


class TestSystemHealthCheckerServiceChecks:
    """SystemHealthChecker服务检查测试"""

    @pytest.fixture
    def checker(self):
        """SystemHealthChecker fixture"""
        return SystemHealthChecker()

    def test_check_service_success(self, checker):
        """测试服务检查成功"""
        def healthy_check():
            return {
                "status": "healthy",
                "message": "Service is running",
                "response_time": 0.1,
                "details": {"connections": 10}
            }

        checker.register_service("healthy_service", healthy_check)
        result = checker.check_service("healthy_service")

        assert isinstance(result, HealthCheckResult)
        assert result.service_name == "healthy_service"
        assert result.status == "healthy"
        assert result.message == "Service is running"
        assert result.response_time == 0.1
        assert result.details["connections"] == 10

    def test_check_service_failure(self, checker):
        """测试服务检查失败"""
        def failing_check():
            raise Exception("Service unavailable")

        checker.register_service("failing_service", failing_check)
        result = checker.check_service("failing_service")

        assert isinstance(result, HealthCheckResult)
        assert result.service_name == "failing_service"
        assert result.status == "unhealthy"
        assert "Service unavailable" in result.message
        assert result.response_time > 0

    def test_check_service_timeout(self, checker):
        """测试服务检查超时"""
        def slow_check():
            time.sleep(10)  # 超过默认超时时间5秒
            return {"status": "healthy"}

        checker.register_service("slow_service", slow_check)

        start_time = time.time()
        result = checker.check_service("slow_service", timeout=0.1)
        end_time = time.time()

        # 应该在超时时间内完成
        assert end_time - start_time < 1.0
        assert result.status == "unhealthy"
        assert "timeout" in result.message.lower()

    def test_check_nonexistent_service(self, checker):
        """测试检查不存在的服务"""
        result = checker.check_service("nonexistent_service")

        assert isinstance(result, HealthCheckResult)
        assert result.service_name == "nonexistent_service"
        assert result.status == "unhealthy"
        assert "not registered" in result.message.lower()


class TestSystemHealthCheckerSystemChecks:
    """SystemHealthChecker系统检查测试"""

    @pytest.fixture
    def checker(self):
        """SystemHealthChecker fixture"""
        return SystemHealthChecker()

    @patch('psutil.cpu_percent')
    def test_check_cpu_usage_normal(self, mock_cpu_percent, checker):
        """测试CPU使用率检查 - 正常"""
        mock_cpu_percent.return_value = 45.5

        result = checker._check_cpu_usage()

        assert result["status"] == "healthy"
        assert result["cpu_percent"] == 45.5
        assert "正常" in result["message"]

    @patch('psutil.cpu_percent')
    def test_check_cpu_usage_high(self, mock_cpu_percent, checker):
        """测试CPU使用率检查 - 过高"""
        mock_cpu_percent.return_value = 95.5

        result = checker._check_cpu_usage()

        assert result["status"] == "warning"
        assert result["cpu_percent"] == 95.5
        assert "过高" in result["message"]

    @patch('psutil.cpu_percent')
    def test_check_cpu_usage_critical(self, mock_cpu_percent, checker):
        """测试CPU使用率检查 - 严重过高"""
        mock_cpu_percent.return_value = 98.5

        result = checker._check_cpu_usage()

        assert result["status"] == "critical"
        assert result["cpu_percent"] == 98.5
        assert "严重过高" in result["message"]

    @patch('psutil.cpu_percent', side_effect=Exception("psutil error"))
    def test_check_cpu_usage_exception(self, mock_cpu_percent, checker):
        """测试CPU使用率检查 - 异常"""
        result = checker._check_cpu_usage()

        assert result["status"] == "unknown"
        assert "无法获取" in result["message"]

    @patch('psutil.virtual_memory')
    def test_check_memory_usage_normal(self, mock_memory, checker):
        """测试内存使用率检查 - 正常"""
        mock_memory.return_value.percent = 60.0
        mock_memory.return_value.available = 4 * 1024 * 1024 * 1024  # 4GB

        result = checker._check_memory_usage()

        assert result["status"] == "healthy"
        assert result["memory_percent"] == 60.0
        assert result["available_gb"] == 4.0

    @patch('psutil.virtual_memory')
    def test_check_memory_usage_high(self, mock_memory, checker):
        """测试内存使用率检查 - 过高"""
        mock_memory.return_value.percent = 92.0
        mock_memory.return_value.available = 0.5 * 1024 * 1024 * 1024  # 0.5GB

        result = checker._check_memory_usage()

        assert result["status"] == "warning"
        assert result["memory_percent"] == 92.0

    @patch('psutil.disk_usage')
    def test_check_disk_usage_normal(self, mock_disk, checker):
        """测试磁盘使用率检查 - 正常"""
        mock_disk.return_value.percent = 45.0
        mock_disk.return_value.free = 100 * 1024 * 1024 * 1024  # 100GB

        result = checker._check_disk_usage()

        assert result["status"] == "healthy"
        assert result["disk_percent"] == 45.0
        assert result["free_gb"] == 100.0

    @patch('psutil.disk_usage')
    def test_check_disk_usage_critical(self, mock_disk, checker):
        """测试磁盘使用率检查 - 严重不足"""
        mock_disk.return_value.percent = 96.0
        mock_disk.return_value.free = 5 * 1024 * 1024 * 1024  # 5GB

        result = checker._check_disk_usage()

        assert result["status"] == "critical"
        assert result["disk_percent"] == 96.0

    @patch('psutil.process_iter')
    def test_check_process_health_normal(self, mock_process_iter, checker):
        """测试进程健康检查 - 正常"""
        # Mock一些进程
        mock_process1 = MagicMock()
        mock_process1.info = {'pid': 1, 'name': 'systemd', 'status': 'running'}

        mock_process2 = MagicMock()
        mock_process2.info = {'pid': 2, 'name': 'python', 'status': 'running'}

        mock_process_iter.return_value = [mock_process1, mock_process2]

        result = checker._check_process_health()

        assert result["status"] == "healthy"
        assert result["total_processes"] == 2
        assert len(result["critical_processes"]) == 0

    @patch('psutil.process_iter')
    def test_check_process_health_with_critical(self, mock_process_iter, checker):
        """测试进程健康检查 - 有关键进程异常"""
        # Mock进程，其中一个关键进程不在运行
        mock_process = MagicMock()
        mock_process.info = {'pid': 1, 'name': 'sshd', 'status': 'stopped'}

        mock_process_iter.return_value = [mock_process]

        result = checker._check_process_health()

        assert result["status"] == "warning"
        assert "sshd" in str(result["critical_processes"])


class TestSystemHealthCheckerAsyncOperations:
    """SystemHealthChecker异步操作测试"""

    @pytest.fixture
    def checker(self):
        """SystemHealthChecker fixture"""
        return SystemHealthChecker()

    @pytest.mark.asyncio
    async def test_check_health_async(self, checker):
        """测试异步健康检查"""
        with patch.object(checker, '_check_cpu_usage') as mock_cpu, \
             patch.object(checker, '_check_memory_usage') as mock_memory, \
             patch.object(checker, '_check_disk_usage') as mock_disk, \
             patch.object(checker, '_check_process_health') as mock_process:

            # 设置mock返回值
            mock_cpu.return_value = {"status": "healthy", "cpu_percent": 50.0}
            mock_memory.return_value = {"status": "healthy", "memory_percent": 60.0}
            mock_disk.return_value = {"status": "healthy", "disk_percent": 40.0}
            mock_process.return_value = {"status": "healthy", "total_processes": 100}

            result = await checker.check_health_async()

            assert isinstance(result, dict)
            assert "timestamp" in result
            assert "overall_status" in result
            assert "checks" in result
            assert "cpu" in result["checks"]
            assert "memory" in result["checks"]
            assert "disk" in result["checks"]
            assert "process" in result["checks"]

    def test_check_health_sync(self, checker):
        """测试同步健康检查"""
        with patch.object(checker, 'check_health_async') as mock_async:
            mock_async.return_value = asyncio.Future()
            mock_async.return_value.set_result({
                "timestamp": datetime.now(),
                "overall_status": "healthy",
                "checks": {"cpu": {"status": "healthy"}}
            })

            result = checker.check_health_sync()

            assert isinstance(result, dict)
            assert "overall_status" in result


class TestSystemHealthCheckerMetrics:
    """SystemHealthChecker指标测试"""

    @pytest.fixture
    def checker(self):
        """SystemHealthChecker fixture"""
        return SystemHealthChecker()

    def test_get_health_metrics(self, checker):
        """测试获取健康指标"""
        # 执行一些检查以生成历史数据
        checker.check_service("cpu_usage")
        checker.check_service("memory_usage")

        metrics = checker.get_health_metrics()

        assert isinstance(metrics, dict)
        assert "total_checks" in metrics
        assert "services_checked" in metrics
        assert "overall_health_score" in metrics
        assert "check_distribution" in metrics

    def test_get_service_health_history(self, checker):
        """测试获取服务健康历史"""
        # 执行多次检查
        for i in range(5):
            checker.check_service("cpu_usage")
            time.sleep(0.01)  # 确保时间戳不同

        history = checker.get_service_health_history("cpu_usage")

        assert isinstance(history, list)
        assert len(history) <= checker._max_history_size

        # 历史记录应该按时间倒序排列（最新的在前）
        if len(history) > 1:
            assert history[0].timestamp >= history[1].timestamp

    def test_get_service_health_history_empty(self, checker):
        """测试获取空的服务健康历史"""
        history = checker.get_service_health_history("nonexistent_service")

        assert isinstance(history, list)
        assert len(history) == 0


class TestSystemHealthCheckerStatusCalculation:
    """SystemHealthChecker状态计算测试"""

    @pytest.fixture
    def checker(self):
        """SystemHealthChecker fixture"""
        return SystemHealthChecker()

    def test_calculate_overall_status_all_healthy(self, checker):
        """测试总体状态计算 - 全部健康"""
        results = {
            "cpu": HealthCheckResult("cpu", "healthy", "OK", 0.1),
            "memory": HealthCheckResult("memory", "healthy", "OK", 0.1),
            "disk": HealthCheckResult("disk", "healthy", "OK", 0.1)
        }

        status = checker._calculate_overall_status(results)

        assert status == "healthy"

    def test_calculate_overall_status_with_warnings(self, checker):
        """测试总体状态计算 - 包含警告"""
        results = {
            "cpu": HealthCheckResult("cpu", "healthy", "OK", 0.1),
            "memory": HealthCheckResult("memory", "warning", "High usage", 0.1),
            "disk": HealthCheckResult("disk", "healthy", "OK", 0.1)
        }

        status = checker._calculate_overall_status(results)

        assert status == "warning"

    def test_calculate_overall_status_with_critical(self, checker):
        """测试总体状态计算 - 包含严重问题"""
        results = {
            "cpu": HealthCheckResult("cpu", "healthy", "OK", 0.1),
            "memory": HealthCheckResult("memory", "critical", "Out of memory", 0.1),
            "disk": HealthCheckResult("disk", "healthy", "OK", 0.1)
        }

        status = checker._calculate_overall_status(results)

        assert status == "critical"

    def test_calculate_overall_status_mixed(self, checker):
        """测试总体状态计算 - 混合状态"""
        results = {
            "cpu": HealthCheckResult("cpu", "warning", "High CPU", 0.1),
            "memory": HealthCheckResult("memory", "critical", "Out of memory", 0.1),
            "disk": HealthCheckResult("disk", "unhealthy", "Disk error", 0.1)
        }

        # 严重问题优先级最高
        status = checker._calculate_overall_status(results)

        assert status == "critical"

    def test_generate_recommendations_healthy(self, checker):
        """测试生成建议 - 健康状态"""
        check_result = {
            "status": "healthy",
            "cpu_percent": 50.0,
            "memory_percent": 60.0
        }

        recommendations = checker._generate_recommendations(check_result)

        assert isinstance(recommendations, list)
        assert len(recommendations) == 0  # 健康状态没有建议

    def test_generate_recommendations_with_issues(self, checker):
        """测试生成建议 - 有问题状态"""
        check_result = {
            "status": "warning",
            "cpu_percent": 95.0,
            "memory_percent": 85.0
        }

        recommendations = checker._generate_recommendations(check_result)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("CPU" in rec for rec in recommendations)
        assert any("内存" in rec for rec in recommendations)


class TestSystemHealthCheckerIntegration:
    """SystemHealthChecker集成测试"""

    @pytest.fixture
    def checker(self):
        """SystemHealthChecker fixture"""
        return SystemHealthChecker()

    def test_comprehensive_health_check(self, checker):
        """测试综合健康检查"""
        # 执行完整健康检查
        result = checker.check_health_sync()

        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "overall_status" in result
        assert "checks" in result

        # 检查必要的检查项
        checks = result["checks"]
        assert "cpu" in checks
        assert "memory" in checks
        assert "disk" in checks
        assert "process" in checks

        # 验证每个检查都有必要字段
        for check_name, check_data in checks.items():
            assert "status" in check_data
            assert "message" in check_data
            assert isinstance(check_data["status"], str)

    def test_service_registration_and_monitoring(self, checker):
        """测试服务注册和监控"""
        service_name = "integration_test_service"

        # 注册服务
        def test_check():
            return {
                "status": "healthy",
                "message": "Integration test service is running",
                "response_time": 0.05,
                "details": {"version": "1.0.0"}
            }

        checker.register_service(service_name, test_check)

        # 执行检查
        check_result = checker.check_service(service_name)

        assert check_result.service_name == service_name
        assert check_result.status == "healthy"
        assert "Integration test" in check_result.message

        # 验证历史记录
        history = checker.get_service_health_history(service_name)
        assert len(history) == 1
        assert history[0].status == "healthy"

        # 再次检查
        check_result2 = checker.check_service(service_name)
        history = checker.get_service_health_history(service_name)
        assert len(history) == 2

    def test_system_under_load_simulation(self, checker):
        """测试系统负载模拟"""
        # 模拟高负载情况
        with patch('psutil.cpu_percent', return_value=85.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:

            # 设置内存高使用率
            mock_memory.return_value.percent = 88.0
            mock_memory.return_value.available = 1 * 1024 * 1024 * 1024  # 1GB

            # 设置磁盘高使用率
            mock_disk.return_value.percent = 91.0
            mock_disk.return_value.free = 10 * 1024 * 1024 * 1024  # 10GB

            result = checker.check_health_sync()

            assert result["overall_status"] in ["warning", "critical"]

            # 检查各个子系统状态
            checks = result["checks"]
            assert checks["cpu"]["status"] in ["warning", "critical"]
            assert checks["memory"]["status"] in ["warning", "critical"]
            assert checks["disk"]["status"] in ["warning", "critical"]

    def test_error_recovery_and_resilience(self, checker):
        """测试错误恢复和弹性"""
        service_name = "resilient_service"

        # 注册一个有时会失败的服务
        failure_count = 0

        def unreliable_check():
            nonlocal failure_count
            failure_count += 1
            if failure_count % 3 == 0:  # 每第三次失败
                raise Exception("Simulated service failure")
            return {
                "status": "healthy",
                "message": "Service is working",
                "response_time": 0.1
            }

        checker.register_service(service_name, unreliable_check)

        # 执行多次检查
        results = []
        for i in range(9):
            result = checker.check_service(service_name)
            results.append(result.status)

        # 统计结果
        healthy_count = sum(1 for status in results if status == "healthy")
        unhealthy_count = sum(1 for status in results if status == "unhealthy")

        # 应该有健康的检查（2/3）
        assert healthy_count >= 4  # 至少4次成功（9次中的2/3）
        assert unhealthy_count >= 2  # 至少2次失败

        # 系统应该能够处理这些失败而不崩溃
        history = checker.get_service_health_history(service_name)
        assert len(history) == 9

    def test_performance_under_concurrent_load(self, checker):
        """测试并发负载下的性能"""
        import threading
        import time

        # 注册多个服务
        for i in range(10):
            service_name = f"perf_service_{i}"
            def create_check(service_id=i):
                def check():
                    time.sleep(0.01)  # 模拟检查时间
                    return {"status": "healthy", "message": f"Service {service_id} OK"}
                return check

            checker.register_service(service_name, create_check())

        # 并发检查所有服务
        results = {}
        exceptions = []

        def check_worker(service_name: str):
            try:
                start_time = time.time()
                result = checker.check_service(service_name)
                end_time = time.time()
                results[service_name] = (result, end_time - start_time)
            except Exception as e:
                exceptions.append(f"{service_name}: {e}")

        threads = []
        for i in range(10):
            service_name = f"perf_service_{i}"
            t = threading.Thread(target=check_worker, args=(service_name,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join(timeout=5.0)

        # 不应该有异常
        assert len(exceptions) == 0, f"Concurrent check exceptions: {exceptions}"

        # 所有服务都应该被检查
        assert len(results) == 10

        # 检查性能
        total_time = sum(duration for _, duration in results.values())
        avg_time = total_time / len(results)

        # 平均检查时间应该在合理范围内
        assert avg_time < 1.0, f"Average check time too high: {avg_time}s"
