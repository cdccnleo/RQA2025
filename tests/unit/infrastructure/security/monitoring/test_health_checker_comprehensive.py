#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 HealthChecker综合测试

测试健康检查器的所有功能，包括：
- 系统健康状态检查
- 内存使用监控
- CPU使用监控
- 磁盘空间监控
- 网络连接检查
- 服务可用性检查
- 健康报告生成
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import psutil
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open
from src.infrastructure.security.monitoring.health_checker import HealthChecker, HealthStatus


class TestHealthCheckerComprehensive:
    """HealthChecker综合测试"""

    @pytest.fixture
    def health_checker(self):
        """创建HealthChecker实例"""
        return HealthChecker()

    def test_initialization(self, health_checker):
        """测试初始化"""
        assert health_checker is not None
        assert hasattr(health_checker, 'check_memory')
        assert hasattr(health_checker, 'check_cpu')
        assert hasattr(health_checker, 'check_disk')
        assert hasattr(health_checker, 'check_network')
        assert hasattr(health_checker, 'overall_health_check')

    def test_memory_check_normal(self, health_checker):
        """测试正常内存检查"""
        result = health_checker.check_memory()

        assert isinstance(result, dict)
        assert "status" in result
        assert "message" in result
        assert "details" in result

        # 正常情况下应该是healthy或warning
        assert result["status"] in ["healthy", "warning", "critical"]

        # 检查详细信息
        details = result["details"]
        assert "total_mb" in details
        assert "used_mb" in details
        assert "available_mb" in details
        assert "usage_percent" in details

    @patch('psutil.virtual_memory')
    def test_memory_check_high_usage(self, mock_memory, health_checker):
        """测试高内存使用情况"""
        # 模拟95%的内存使用
        mock_mem = MagicMock()
        mock_mem.total = 1000 * 1024 * 1024  # 1000MB
        mock_mem.available = 50 * 1024 * 1024  # 50MB
        mock_mem.percent = 95.0
        mock_memory.return_value = mock_mem

        result = health_checker.check_memory()

        assert result["status"] == "critical"
        assert "high" in result["message"].lower() or "critical" in result["message"].lower()

    @patch('psutil.virtual_memory')
    def test_memory_check_critical_usage(self, mock_memory, health_checker):
        """测试临界内存使用情况"""
        # 模拟98%的内存使用
        mock_mem = MagicMock()
        mock_mem.total = 1000 * 1024 * 1024  # 1000MB
        mock_mem.available = 20 * 1024 * 1024  # 20MB
        mock_mem.percent = 98.0
        mock_memory.return_value = mock_mem

        result = health_checker.check_memory()

        assert result["status"] == "critical"
        assert "critical" in result["message"].lower()

    def test_cpu_check_normal(self, health_checker):
        """测试正常CPU检查"""
        result = health_checker.check_cpu()

        assert isinstance(result, dict)
        assert "status" in result
        assert "message" in result
        assert "details" in result

        assert result["status"] in ["healthy", "warning", "critical"]

        details = result["details"]
        assert "cpu_percent" in details
        assert "cpu_count" in details
        assert isinstance(details["cpu_percent"], (int, float))
        assert details["cpu_count"] > 0

    @patch('psutil.cpu_percent')
    def test_cpu_check_high_usage(self, mock_cpu_percent, health_checker):
        """测试高CPU使用情况"""
        mock_cpu_percent.return_value = 95.0

        result = health_checker.check_cpu()

        assert result["status"] == "critical"
        assert "high" in result["message"].lower() or "critical" in result["message"].lower()

    @patch('psutil.cpu_percent')
    def test_cpu_check_warning_usage(self, mock_cpu_percent, health_checker):
        """测试警告级CPU使用情况"""
        mock_cpu_percent.return_value = 75.0

        result = health_checker.check_cpu()

        assert result["status"] == "warning"

    def test_disk_check_normal(self, health_checker):
        """测试正常磁盘检查"""
        result = health_checker.check_disk()

        assert isinstance(result, dict)
        assert "status" in result
        assert "message" in result
        assert "details" in result

        assert result["status"] in ["healthy", "warning", "critical"]

        details = result["details"]
        assert "total_gb" in details
        assert "used_gb" in details
        assert "free_gb" in details
        assert "usage_percent" in details

    @patch('psutil.disk_usage')
    def test_disk_check_low_space(self, mock_disk_usage, health_checker):
        """测试磁盘空间不足情况"""
        # 模拟只剩下5%的磁盘空间
        mock_disk = MagicMock()
        mock_disk.total = 1000 * 1024 * 1024 * 1024  # 1000GB
        mock_disk.used = 950 * 1024 * 1024 * 1024   # 950GB
        mock_disk.free = 50 * 1024 * 1024 * 1024    # 50GB
        mock_disk.percent = 95.0
        mock_disk_usage.return_value = mock_disk

        result = health_checker.check_disk()

        assert result["status"] == "critical"
        assert "low" in result["message"].lower() or "critical" in result["message"].lower()

    @patch('psutil.disk_usage')
    def test_disk_check_warning_space(self, mock_disk_usage, health_checker):
        """测试磁盘空间警告情况"""
        # 模拟只剩下15%的磁盘空间
        mock_disk = MagicMock()
        mock_disk.total = 1000 * 1024 * 1024 * 1024  # 1000GB
        mock_disk.used = 850 * 1024 * 1024 * 1024   # 850GB
        mock_disk.free = 150 * 1024 * 1024 * 1024   # 150GB
        mock_disk.percent = 85.0
        mock_disk_usage.return_value = mock_disk

        result = health_checker.check_disk()

        assert result["status"] == "warning"

    def test_network_check_basic(self, health_checker):
        """测试基本网络检查"""
        result = health_checker.check_network()

        assert isinstance(result, dict)
        assert "status" in result
        assert "message" in result
        assert "details" in result

        # 网络检查可能因环境而异，但应该有合理的响应
        assert result["status"] in ["healthy", "warning", "critical", "unknown"]

    @patch('psutil.net_if_stats')
    def test_network_check_no_interfaces(self, mock_net_if_stats, health_checker):
        """测试无网络接口的情况"""
        mock_net_if_stats.return_value = {}

        result = health_checker.check_network()

        assert result["status"] in ["warning", "critical"]
        assert "network" in result["message"].lower()

    @patch('psutil.net_if_stats')
    def test_network_check_interfaces_down(self, mock_net_if_stats, health_checker):
        """测试网络接口关闭的情况"""
        # 模拟所有接口都关闭
        mock_interfaces = {
            'eth0': MagicMock(isup=False),
            'wlan0': MagicMock(isup=False)
        }
        mock_net_if_stats.return_value = mock_interfaces

        result = health_checker.check_network()

        assert result["status"] in ["warning", "critical"]

    def test_overall_health_check(self, health_checker):
        """测试整体健康检查"""
        result = health_checker.overall_health_check()

        assert isinstance(result, dict)
        assert "overall_status" in result
        assert "timestamp" in result
        assert "checks" in result

        assert result["overall_status"] in ["healthy", "warning", "critical", "unknown"]

        checks = result["checks"]
        assert "memory" in checks
        assert "cpu" in checks
        assert "disk" in checks
        assert "network" in checks

        # 验证每个检查都有正确的结构
        for check_name, check_result in checks.items():
            assert "status" in check_result
            assert "message" in check_result

    @patch('src.infrastructure.security.monitoring.health_checker.HealthChecker.check_memory')
    @patch('src.infrastructure.security.monitoring.health_checker.HealthChecker.check_cpu')
    @patch('src.infrastructure.security.monitoring.health_checker.HealthChecker.check_disk')
    @patch('src.infrastructure.security.monitoring.health_checker.HealthChecker.check_network')
    def test_overall_health_check_critical_system(self, mock_network, mock_disk, mock_cpu, mock_memory, health_checker):
        """测试整体健康检查在系统临界状态下的表现"""
        # 模拟所有检查都返回critical
        mock_memory.return_value = {"status": "critical", "message": "Memory critical", "details": {}}
        mock_cpu.return_value = {"status": "critical", "message": "CPU critical", "details": {}}
        mock_disk.return_value = {"status": "critical", "message": "Disk critical", "details": {}}
        mock_network.return_value = {"status": "critical", "message": "Network critical", "details": {}}

        result = health_checker.overall_health_check()

        assert result["overall_status"] == "critical"

    @patch('src.infrastructure.security.monitoring.health_checker.HealthChecker.check_memory')
    @patch('src.infrastructure.security.monitoring.health_checker.HealthChecker.check_cpu')
    @patch('src.infrastructure.security.monitoring.health_checker.HealthChecker.check_disk')
    @patch('src.infrastructure.security.monitoring.health_checker.HealthChecker.check_network')
    def test_overall_health_check_mixed_status(self, mock_network, mock_disk, mock_cpu, mock_memory, health_checker):
        """测试整体健康检查在混合状态下的表现"""
        # 模拟混合状态：一些healthy，一些warning
        mock_memory.return_value = {"status": "healthy", "message": "Memory OK", "details": {}}
        mock_cpu.return_value = {"status": "warning", "message": "CPU high", "details": {}}
        mock_disk.return_value = {"status": "healthy", "message": "Disk OK", "details": {}}
        mock_network.return_value = {"status": "healthy", "message": "Network OK", "details": {}}

        result = health_checker.overall_health_check()

        # 整体状态应该反映最严重的单个检查状态
        assert result["overall_status"] == "warning"

    def test_health_check_error_handling(self, health_checker):
        """测试健康检查的错误处理"""
        # 测试在异常情况下的健壮性

        # 使用patch来模拟异常
        with patch('psutil.virtual_memory', side_effect=Exception("Mock error")):
            result = health_checker.check_memory()
            # 应该返回错误状态而不是崩溃
            assert result["status"] == "unknown" or "error" in result["status"]

        with patch('psutil.cpu_percent', side_effect=Exception("Mock error")):
            result = health_checker.check_cpu()
            assert result["status"] == "unknown" or "error" in result["status"]

        with patch('psutil.disk_usage', side_effect=Exception("Mock error")):
            result = health_checker.check_disk()
            assert result["status"] == "unknown" or "error" in result["status"]

    def test_memory_thresholds(self, health_checker):
        """测试内存阈值设置"""
        # 测试不同的内存使用百分比
        test_cases = [
            (50.0, "healthy"),      # 50% - 健康
            (75.0, "warning"),      # 75% - 警告
            (85.0, "warning"),      # 85% - 警告
            (95.0, "critical"),     # 95% - 严重
        ]

        for usage_percent, expected_status in test_cases:
            with patch('psutil.virtual_memory') as mock_memory:
                mock_mem = MagicMock()
                mock_mem.percent = usage_percent
                mock_mem.total = 1000 * 1024 * 1024
                mock_mem.available = 1000 * 1024 * 1024 * (1 - usage_percent / 100)
                mock_memory.return_value = mock_mem

                result = health_checker.check_memory()
                assert result["status"] == expected_status, f"Expected {expected_status} for {usage_percent}% usage, got {result['status']}"

    def test_cpu_thresholds(self, health_checker):
        """测试CPU阈值设置"""
        test_cases = [
            (30.0, "healthy"),      # 30% - 健康
            (60.0, "healthy"),      # 60% - 健康
            (75.0, "warning"),      # 75% - 警告
            (90.0, "critical"),     # 90% - 严重
        ]

        for cpu_percent, expected_status in test_cases:
            with patch('psutil.cpu_percent', return_value=cpu_percent):
                result = health_checker.check_cpu()
                assert result["status"] == expected_status, f"Expected {expected_status} for {cpu_percent}% CPU, got {result['status']}"

    def test_disk_thresholds(self, health_checker):
        """测试磁盘阈值设置"""
        test_cases = [
            (50.0, "healthy"),      # 50% - 健康
            (75.0, "warning"),      # 75% - 警告
            (85.0, "warning"),      # 85% - 警告
            (95.0, "critical"),     # 95% - 严重
        ]

        for usage_percent, expected_status in test_cases:
            with patch('psutil.disk_usage') as mock_disk:
                mock_d = MagicMock()
                mock_d.percent = usage_percent
                mock_d.total = 1000 * 1024 * 1024 * 1024
                mock_d.used = mock_d.total * (usage_percent / 100)
                mock_d.free = mock_d.total - mock_d.used
                mock_disk.return_value = mock_d

                result = health_checker.check_disk()
                assert result["status"] == expected_status, f"Expected {expected_status} for {usage_percent}% disk usage, got {result['status']}"

    def test_health_report_detailed_structure(self, health_checker):
        """测试健康报告的详细结构"""
        result = health_checker.overall_health_check()

        # 验证基本结构
        assert "overall_status" in result
        assert "timestamp" in result
        assert "checks" in result
        assert "summary" in result

        # 验证时间戳格式
        timestamp = result["timestamp"]
        assert isinstance(timestamp, datetime)

        # 验证摘要信息
        summary = result["summary"]
        assert "total_checks" in summary
        assert "healthy_checks" in summary
        assert "warning_checks" in summary
        assert "critical_checks" in summary

        # 验证总数计算正确
        checks = result["checks"]
        total_checks = len(checks)
        assert summary["total_checks"] == total_checks

    def test_concurrent_health_checks(self, health_checker):
        """测试并发健康检查"""
        import threading
        import queue

        results = queue.Queue()
        errors = queue.Queue()

        def worker(worker_id):
            try:
                # 执行多种健康检查
                memory_result = health_checker.check_memory()
                cpu_result = health_checker.check_cpu()
                disk_result = health_checker.check_disk()
                network_result = health_checker.check_network()
                overall_result = health_checker.overall_health_check()

                results.put({
                    'worker': worker_id,
                    'memory': memory_result['status'],
                    'cpu': cpu_result['status'],
                    'disk': disk_result['status'],
                    'network': network_result['status'],
                    'overall': overall_result['overall_status']
                })
            except Exception as e:
                errors.put(f"worker_{worker_id}: {e}")

        # 启动多个线程
        threads = []
        num_threads = 5
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join(timeout=10)

        # 验证结果
        assert errors.empty()

        result_count = 0
        while not results.empty():
            result = results.get()
            result_count += 1
            # 验证每个结果都有合理的状态
            for key in ['memory', 'cpu', 'disk', 'network', 'overall']:
                assert result[key] in ['healthy', 'warning', 'critical', 'unknown']

        assert result_count == num_threads

    def test_health_check_performance(self, health_checker):
        """测试健康检查性能"""
        import time

        # 测试单个检查的性能
        start_time = time.time()
        for _ in range(10):
            health_checker.check_memory()
            health_checker.check_cpu()
            health_checker.check_disk()
            health_checker.check_network()
        single_checks_time = time.time() - start_time

        # 测试整体健康检查的性能
        start_time = time.time()
        for _ in range(10):
            health_checker.overall_health_check()
        overall_check_time = time.time() - start_time

        # 性能断言（在合理的时间范围内）
        assert single_checks_time < 5.0  # 40个单个检查应该在5秒内完成
        assert overall_check_time < 3.0  # 10个整体检查应该在3秒内完成

    def test_health_status_persistence(self, health_checker):
        """测试健康状态持久性"""
        # 执行多次检查，验证结果的一致性
        results = []
        for _ in range(5):
            result = health_checker.overall_health_check()
            results.append(result['overall_status'])
            time.sleep(0.1)  # 小延迟

        # 在短时间内，健康状态应该相对稳定（除非系统状态变化很大）
        # 至少不应该每次都完全不同
        unique_results = set(results)
        assert len(unique_results) <= 3  # 最多3种不同的状态

    def test_health_check_data_accuracy(self, health_checker):
        """测试健康检查数据准确性"""
        # 获取真实的系统信息
        real_memory = psutil.virtual_memory()
        real_cpu = psutil.cpu_percent(interval=0.1)
        real_disk = psutil.disk_usage('/')

        # 执行检查
        memory_result = health_checker.check_memory()
        cpu_result = health_checker.check_cpu()
        disk_result = health_checker.check_disk()

        # 验证数据准确性（允许小误差）
        memory_details = memory_result['details']
        cpu_details = cpu_result['details']
        disk_details = disk_result['details']

        # 内存检查准确性
        assert abs(memory_details['usage_percent'] - real_memory.percent) < 5.0

        # CPU检查准确性（允许较大误差，因为CPU使用率变化快）
        assert abs(cpu_details['cpu_percent'] - real_cpu) < 20.0

        # 磁盘检查准确性
        assert abs(disk_details['usage_percent'] - real_disk.percent) < 5.0

    def test_health_check_comprehensive_reporting(self, health_checker):
        """测试健康检查综合报告"""
        result = health_checker.overall_health_check()

        # 验证报告的完整性
        required_fields = [
            'overall_status', 'timestamp', 'checks', 'summary'
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # 验证checks字段包含所有必需的检查
        checks = result['checks']
        required_checks = ['memory', 'cpu', 'disk', 'network']

        for check_name in required_checks:
            assert check_name in checks, f"Missing required check: {check_name}"

        # 验证summary字段的完整性
        summary = result['summary']
        required_summary_fields = [
            'total_checks', 'healthy_checks', 'warning_checks', 'critical_checks'
        ]

        for field in required_summary_fields:
            assert field in summary, f"Missing required summary field: {field}"

        # 验证计数正确性
        total_from_checks = len(checks)
        total_from_summary = summary['total_checks']
        assert total_from_checks == total_from_summary, "Check count mismatch"

        # 验证状态分布
        status_counts = {'healthy': 0, 'warning': 0, 'critical': 0, 'unknown': 0}
        for check_result in checks.values():
            status = check_result['status']
            if status in status_counts:
                status_counts[status] += 1

        assert status_counts['healthy'] == summary['healthy_checks']
        assert status_counts['warning'] == summary['warning_checks']
        assert status_counts['critical'] == summary['critical_checks']

    def test_health_checker_resilience(self, health_checker):
        """测试健康检查器的韧性"""
        # 测试在各种异常情况下的表现

        # 1. 测试在极端资源情况下
        with patch('psutil.virtual_memory', side_effect=OSError("No memory info")):
            result = health_checker.check_memory()
            assert result['status'] == 'unknown'

        # 2. 测试在权限不足情况下
        with patch('psutil.cpu_percent', side_effect=PermissionError("No permission")):
            result = health_checker.check_cpu()
            assert result['status'] == 'unknown'

        # 3. 测试在系统调用失败情况下
        with patch('psutil.disk_usage', side_effect=FileNotFoundError("Path not found")):
            result = health_checker.check_disk()
            assert result['status'] == 'unknown'

        # 验证整体健康检查在部分组件失败时仍能工作
        with patch('src.infrastructure.security.monitoring.health_checker.HealthChecker.check_memory',
                  return_value={'status': 'unknown', 'message': 'Failed', 'details': {}}):
            result = health_checker.overall_health_check()
            assert 'overall_status' in result
            assert result['overall_status'] != 'unknown'  # 应该有合理的整体状态

    def test_add_check(self, health_checker):
        """测试添加健康检查"""
        from src.infrastructure.security.monitoring.health_checker import HealthCheck

        def dummy_check():
            return HealthStatus.HEALTHY, "Dummy check passed"

        check = HealthCheck(
            name="dummy_check",
            description="A dummy health check",
            check_function=dummy_check
        )

        health_checker.add_check(check)

        assert "dummy_check" in health_checker.checks
        assert health_checker.checks["dummy_check"] == check

    def test_remove_check(self, health_checker):
        """测试移除健康检查"""
        from src.infrastructure.security.monitoring.health_checker import HealthCheck

        def dummy_check():
            return HealthStatus.HEALTHY, "Dummy check passed"

        check = HealthCheck(
            name="dummy_check",
            description="A dummy health check",
            check_function=dummy_check
        )

        health_checker.add_check(check)
        assert "dummy_check" in health_checker.checks

        health_checker.remove_check("dummy_check")
        assert "dummy_check" not in health_checker.checks

    def test_remove_nonexistent_check(self, health_checker):
        """测试移除不存在的检查"""
        # 应该不会抛出异常
        health_checker.remove_check("nonexistent_check")

    def test_is_worse_status(self, health_checker):
        """测试状态比较"""
        from src.infrastructure.security.monitoring.health_checker import HealthStatus

        # 测试状态严重程度比较
        assert health_checker._is_worse_status(HealthStatus.CRITICAL, HealthStatus.HEALTHY) == True
        assert health_checker._is_worse_status(HealthStatus.HEALTHY, HealthStatus.CRITICAL) == False
        assert health_checker._is_worse_status(HealthStatus.WARNING, HealthStatus.WARNING) == False
        assert health_checker._is_worse_status(HealthStatus.UNHEALTHY, HealthStatus.DEGRADED) == True

    def test_check_process_health(self, health_checker):
        """测试进程健康检查"""
        result = health_checker._check_process_health()

        assert isinstance(result, tuple)
        assert len(result) == 2
        status, message = result

        # 进程健康检查应该返回合理的状态
        from src.infrastructure.security.monitoring.health_checker import HealthStatus
        assert status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
        assert isinstance(message, str)

    def test_collect_system_metrics(self, health_checker):
        """测试系统指标收集"""
        metrics = health_checker._collect_system_metrics()

        assert isinstance(metrics, dict)
        # 应该包含基本的系统指标
        expected_keys = ['cpu_percent', 'memory_percent', 'disk_usage', 'timestamp']
        for key in expected_keys:
            assert key in metrics or any(key in k for k in metrics.keys())

    def test_generate_recommendations(self, health_checker):
        """测试生成健康建议"""
        # 创建一个模拟的健康状态
        from src.infrastructure.security.monitoring.health_checker import SystemHealth, HealthStatus

        mock_health = SystemHealth(
            overall_status=HealthStatus.CRITICAL,
            timestamp=datetime.now(),
            checks={
                'cpu_usage': MagicMock(last_status=HealthStatus.CRITICAL, last_message="High CPU usage"),
                'memory_usage': MagicMock(last_status=HealthStatus.UNHEALTHY, last_message="High memory usage")
            },
            recommendations=[]
        )

        recommendations = health_checker._generate_recommendations(mock_health)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # 验证建议内容
        for rec in recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0

    def test_shutdown_method(self, health_checker):
        """测试shutdown方法"""
        # 应该能够正常关闭而不抛出异常
        health_checker.shutdown()

        # 验证监控停止标志被设置
        assert health_checker._stop_monitoring.is_set()

    def test_health_check_with_disabled_background_monitoring(self):
        """测试禁用后台监控的健康检查器"""
        checker = HealthChecker(enable_background_monitoring=False)

        # 应该正常工作
        result = checker.overall_health_check()
        assert 'overall_status' in result

        # 监控线程不应该启动
        assert checker._monitor_thread is None or not checker.enable_background_monitoring

    def test_health_check_timeout_handling(self, health_checker):
        """测试健康检查超时处理"""
        from src.infrastructure.security.monitoring.health_checker import HealthCheck

        def slow_check():
            import time
            time.sleep(0.1)  # 稍微慢一点但不超时
            return HealthStatus.HEALTHY, "Slow check completed"

        slow_health_check = HealthCheck(
            name="slow_check",
            description="A slow health check",
            check_function=slow_check,
            timeout_seconds=0.05  # 很短的超时时间
        )

        # 添加检查
        health_checker.add_check(slow_health_check)

        # 运行健康检查 - 可能会超时
        health = health_checker.run_health_check("slow_check")

        # 无论是否超时，都应该有结果
        assert "slow_check" in health.checks

    def test_concurrent_health_checks(self, health_checker):
        """测试并发健康检查"""
        import threading
        import queue

        results = queue.Queue()
        errors = queue.Queue()

        def worker(worker_id):
            try:
                for i in range(5):
                    result = health_checker.overall_health_check()
                    results.put(f"worker_{worker_id}_check_{i}: {result['overall_status']}")
            except Exception as e:
                errors.put(f"worker_{worker_id}: {e}")

        # 启动多个线程
        threads = []
        num_threads = 3
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join(timeout=10)

        # 验证结果
        assert errors.empty()
        result_count = 0
        while not results.empty():
            results.get()
            result_count += 1

        assert result_count == num_threads * 5

    def test_health_checker_memory_efficiency(self, health_checker):
        """测试健康检查器的内存效率"""
        # 执行多次健康检查
        for i in range(100):
            health_checker.overall_health_check()

        # 应该能够处理大量检查而不出现内存问题
        # 这是一个基本的健全性测试
        result = health_checker.overall_health_check()
        assert 'overall_status' in result

    def test_custom_health_check_function(self, health_checker):
        """测试自定义健康检查函数"""
        from src.infrastructure.security.monitoring.health_checker import HealthCheck

        call_count = 0

        def custom_check():
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return HealthStatus.HEALTHY, "Even check"
            else:
                return HealthStatus.DEGRADED, "Odd check"

        custom_health_check = HealthCheck(
            name="custom_check",
            description="A custom health check",
            check_function=custom_check
        )

        health_checker.add_check(custom_health_check)

        # 运行多次检查
        for i in range(4):
            health_checker.run_health_check("custom_check")

        # 验证检查被调用
        assert call_count == 4

    def test_health_check_intervals(self, health_checker):
        """测试健康检查间隔"""
        from src.infrastructure.security.monitoring.health_checker import HealthCheck

        call_count = 0

        def interval_check():
            nonlocal call_count
            call_count += 1
            return HealthStatus.HEALTHY, f"Call {call_count}"

        interval_check_obj = HealthCheck(
            name="interval_check",
            description="Test interval checking",
            check_function=interval_check,
            interval_seconds=0.1  # 很短的间隔
        )

        health_checker.add_check(interval_check_obj)

        # 第一次应该运行
        health_checker.run_health_check("interval_check")
        first_calls = call_count

        # 立即再次运行 - 不应该运行（间隔太短）
        health_checker.run_health_check("interval_check")
        second_calls = call_count

        # 等待间隔时间后再次运行
        import time
        time.sleep(0.15)
        health_checker.run_health_check("interval_check")
        third_calls = call_count

        # 验证间隔控制
        assert first_calls == 1
        assert second_calls == 1  # 不应该增加
        assert third_calls == 2   # 应该增加

    def test_health_status_enum_values(self):
        """测试健康状态枚举值"""
        from src.infrastructure.security.monitoring.health_checker import HealthStatus

        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.CRITICAL.value == "critical"

    def test_health_check_to_dict(self):
        """测试健康检查对象的to_dict方法"""
        from src.infrastructure.security.monitoring.health_checker import HealthCheck

        def test_check():
            return HealthStatus.HEALTHY, "Test message"

        check = HealthCheck(
            name="test_check",
            description="Test description",
            check_function=test_check,
            timeout_seconds=10.0,
            interval_seconds=60.0
        )

        check_dict = check.to_dict()

        assert check_dict['name'] == "test_check"
        assert check_dict['description'] == "Test description"
        assert check_dict['timeout_seconds'] == 10.0
        assert check_dict['interval_seconds'] == 60.0
        assert 'last_check_time' in check_dict
        assert 'enabled' in check_dict

    def test_health_checker_with_no_checks(self):
        """测试没有健康检查的健康检查器"""
        checker = HealthChecker(enable_background_monitoring=False)

        # 移除所有检查
        for name in list(checker.checks.keys()):
            checker.remove_check(name)

        # 运行健康检查
        health = checker.run_health_check()

        # 应该正常工作，即使没有检查
        assert isinstance(health, object)  # SystemHealth对象
        assert len(health.checks) == 0

    def test_health_checker_large_scale_operation(self, health_checker):
        """测试大规模操作"""
        # 添加多个健康检查
        from src.infrastructure.security.monitoring.health_checker import HealthCheck

        for i in range(20):
            def make_check(i):
                def check_func():
                    return HealthStatus.HEALTHY, f"Check {i} passed"
                return check_func

            check = HealthCheck(
                name=f"bulk_check_{i}",
                description=f"Bulk check {i}",
                check_function=make_check(i)
            )
            health_checker.add_check(check)

        # 执行整体健康检查
        result = health_checker.overall_health_check()

        # 应该能够处理大量检查
        assert 'overall_status' in result
        assert 'checks' in result
        assert 'summary' in result

    def test_health_checker_error_recovery(self, health_checker):
        """测试错误恢复"""
        from src.infrastructure.security.monitoring.health_checker import HealthCheck

        error_count = 0

        def failing_check():
            nonlocal error_count
            error_count += 1
            if error_count == 1:
                raise Exception("Simulated failure")
            return HealthStatus.HEALTHY, "Recovered"

        failing_health_check = HealthCheck(
            name="failing_check",
            description="A check that fails initially",
            check_function=failing_check
        )

        health_checker.add_check(failing_health_check)

        # 第一次运行 - 应该失败但不崩溃
        health1 = health_checker.run_health_check("failing_check")
        assert "failing_check" in health1.checks

        # 第二次运行 - 应该恢复
        health2 = health_checker.run_health_check("failing_check")
        assert "failing_check" in health2.checks

        # 验证错误计数
        assert error_count == 2

    def test_health_checker_memory_check_detailed_info(self, health_checker):
        """测试内存检查的详细信息返回"""
        result = health_checker.check_memory()

        # 验证详细信息包含预期字段
        assert 'details' in result
        details = result['details']
        assert 'total_mb' in details
        assert 'used_mb' in details
        assert 'available_mb' in details
        assert 'usage_percent' in details

        # 验证数据类型和范围
        assert isinstance(details['total_mb'], (int, float))
        assert isinstance(details['usage_percent'], (int, float))
        assert 0 <= details['usage_percent'] <= 100

    def test_health_checker_cpu_check_detailed_info(self, health_checker):
        """测试CPU检查的详细信息返回"""
        result = health_checker.check_cpu()

        # 验证详细信息包含预期字段
        assert 'details' in result
        details = result['details']
        assert 'cpu_percent' in details

        # 验证数据类型和范围
        assert isinstance(details['cpu_percent'], (int, float))
        assert 0 <= details['cpu_percent'] <= 100

    def test_health_checker_disk_check_detailed_info(self, health_checker):
        """测试磁盘检查的详细信息返回"""
        result = health_checker.check_disk()

        # 验证详细信息包含预期字段
        assert 'details' in result
        details = result['details']
        assert 'total_gb' in details
        assert 'used_gb' in details
        assert 'free_gb' in details
        assert 'usage_percent' in details

        # 验证数据类型和范围
        assert isinstance(details['usage_percent'], (int, float))
        assert 0 <= details['usage_percent'] <= 100

    def test_health_checker_network_check_detailed_info(self, health_checker):
        """测试网络检查的详细信息返回"""
        result = health_checker.check_network()

        # 验证详细信息包含预期字段
        assert 'details' in result
        details = result['details']
        assert 'bytes_sent' in details
        assert 'bytes_recv' in details
        assert 'packets_sent' in details
        assert 'packets_recv' in details

        # 验证数据类型
        assert isinstance(details['bytes_sent'], (int, float))
        assert isinstance(details['bytes_recv'], (int, float))

    def test_health_checker_overall_health_check_structure(self, health_checker):
        """测试整体健康检查返回的详细结构"""
        result = health_checker.overall_health_check()

        # 验证基本结构
        assert 'overall_status' in result
        assert 'timestamp' in result
        assert 'checks' in result
        assert 'summary' in result

        # 验证checks结构
        checks = result['checks']
        assert 'cpu' in checks
        assert 'memory' in checks
        assert 'disk' in checks
        assert 'network' in checks

        # 验证每个check都有status和message
        for check_name, check_data in checks.items():
            assert 'status' in check_data
            assert 'message' in check_data

        # 验证summary结构
        summary = result['summary']
        assert 'total_checks' in summary
        assert 'healthy_checks' in summary
        assert 'warning_checks' in summary
        assert 'critical_checks' in summary

        # 验证summary计算
        assert summary['total_checks'] == len(checks)

    def test_health_checker_is_worse_status_method(self, health_checker):
        """测试_is_worse_status方法"""
        from src.infrastructure.security.monitoring.health_checker import HealthStatus

        # 测试不同状态的比较
        assert health_checker._is_worse_status(HealthStatus.CRITICAL, HealthStatus.HEALTHY)
        assert health_checker._is_worse_status(HealthStatus.UNHEALTHY, HealthStatus.WARNING)
        assert not health_checker._is_worse_status(HealthStatus.WARNING, HealthStatus.CRITICAL)
        assert not health_checker._is_worse_status(HealthStatus.HEALTHY, HealthStatus.HEALTHY)

    def test_health_checker_generate_recommendations_for_all_statuses(self, health_checker):
        """测试为所有健康状态生成推荐"""
        from src.infrastructure.security.monitoring.health_checker import HealthStatus

        test_statuses = [
            HealthStatus.CRITICAL,
            HealthStatus.UNHEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.WARNING,
            HealthStatus.HEALTHY
        ]

        for status in test_statuses:
            recommendations = health_checker.generate_recommendations(status)

            # 验证返回列表
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0

            # 验证每条推荐都是字符串
            for rec in recommendations:
                assert isinstance(rec, str)
                assert len(rec.strip()) > 0

    def test_health_checker_collect_system_metrics_comprehensive(self, health_checker):
        """测试收集系统指标的完整性"""
        metrics = health_checker._collect_system_metrics()

        # 验证基本结构
        assert isinstance(metrics, dict)

        # 验证包含预期指标（可能因系统而异，但应该有基本指标）
        expected_keys = ['cpu_percent', 'memory_percent', 'disk_usage']
        for key in expected_keys:
            assert key in metrics or any(k.startswith(key.split('_')[0]) for k in metrics.keys())

    def test_health_checker_health_status_enum_values(self, health_checker):
        """测试健康状态枚举值的使用"""
        from src.infrastructure.security.monitoring.health_checker import HealthStatus

        # 验证所有枚举值都可以访问
        statuses = [
            HealthStatus.HEALTHY,
            HealthStatus.WARNING,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
            HealthStatus.CRITICAL
        ]

        for status in statuses:
            assert status.value in ['healthy', 'warning', 'degraded', 'unhealthy', 'critical']

    def test_health_checker_to_dict_method(self, health_checker):
        """测试HealthCheck的to_dict方法"""
        from src.infrastructure.security.monitoring.health_checker import HealthCheck
        from datetime import datetime

        # 创建HealthCheck实例
        check = HealthCheck(
            status=HealthStatus.HEALTHY,
            message="Test message",
            timestamp=datetime.now(),
            details={"test": "data"}
        )

        # 测试to_dict方法
        result = check.to_dict()

        assert isinstance(result, dict)
        assert 'status' in result
        assert 'message' in result
        assert 'timestamp' in result
        assert 'details' in result

    def test_health_checker_threshold_based_statuses(self, health_checker):
        """测试基于阈值的状态判断"""
        # 测试内存阈值
        memory_result = health_checker.check_memory()
        assert memory_result['status'] in ['healthy', 'warning', 'degraded', 'unhealthy', 'critical']

        # 测试CPU阈值
        cpu_result = health_checker.check_cpu()
        assert cpu_result['status'] in ['healthy', 'warning', 'degraded', 'unhealthy', 'critical']

        # 测试磁盘阈值
        disk_result = health_checker.check_disk()
        assert disk_result['status'] in ['healthy', 'warning', 'degraded', 'unhealthy', 'critical']

    def test_health_checker_error_handling_in_checks(self, health_checker, mocker):
        """测试检查方法中的错误处理"""
        # 模拟psutil抛出异常
        mock_psutil = mocker.patch('src.infrastructure.security.monitoring.health_checker.psutil')
        mock_psutil.virtual_memory.side_effect = Exception("Mock error")

        # 测试内存检查的错误处理
        result = health_checker.check_memory()
        assert 'status' in result
        assert 'message' in result
        assert 'details' in result

        # 即使出错也应该返回有效结构
        assert result['status'] in ['critical', 'unhealthy']

    def test_health_checker_large_scale_operation_resilience(self, health_checker):
        """测试大规模操作的韧性"""
        import time

        start_time = time.time()

        # 执行多次健康检查
        for _ in range(10):
            result = health_checker.overall_health_check()
            assert 'overall_status' in result
            assert 'timestamp' in result

        end_time = time.time()

        # 验证性能在合理范围内（每秒至少能处理几次）
        duration = end_time - start_time
        operations_per_second = 10 / duration
        assert operations_per_second > 0.5  # 至少每秒0.5次操作
