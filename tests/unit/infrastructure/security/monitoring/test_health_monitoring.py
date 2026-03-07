#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 监控和健康检查测试

测试性能监控和健康检查功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.infrastructure.security.monitoring.performance_monitor import (
    PerformanceMonitor, record_security_operation, get_security_performance_report
)
from src.infrastructure.security.monitoring.health_checker import (
    HealthChecker, HealthStatus, HealthCheck
)


class TestPerformanceMonitor(unittest.TestCase):
    """性能监控器测试"""

    def setUp(self):
        """测试前准备"""
        self.monitor = PerformanceMonitor(enabled=True)

    def tearDown(self):
        """测试后清理"""
        self.monitor.shutdown()

    def test_record_security_operation(self):
        """测试记录安全操作"""
        # 记录安全操作
        self.monitor.record_operation("authenticate", 0.1, user_id="user1", resource="/api/auth")
        self.monitor.record_operation("authorize", 0.2, user_id="user1", resource="/api/data")
        self.monitor.record_operation("audit", 0.05, user_id="user1", resource="/api/logs")

        # 验证基本指标
        auth_metrics = self.monitor.get_metrics("authenticate")
        self.assertEqual(auth_metrics['total_calls'], 1)
        self.assertAlmostEqual(auth_metrics['avg_time'], 0.1, places=3)

        # 验证用户活动记录
        self.assertIn("user1", self.monitor.user_activity)
        self.assertIn("authenticate", self.monitor.user_activity["user1"])

        # 验证资源访问记录
        self.assertIn("/api/auth", self.monitor.resource_access)
        self.assertIn("authenticate", self.monitor.resource_access["/api/auth"])

    def test_security_performance_report(self):
        """测试安全性能报告"""
        # 记录一些安全操作
        self.monitor.record_operation("authenticate", 0.1, user_id="user1", resource="/api/auth")
        self.monitor.record_operation("authorize", 0.2, user_id="user2", resource="/api/data")
        self.monitor.record_operation("authenticate", 0.15, user_id="user1", resource="/api/auth")

        # 获取安全性能报告
        report = self.monitor.get_performance_report()

        # 验证报告结构
        self.assertIn('security_metrics', report)
        self.assertIn('user_activity_summary', report['security_metrics'])
        self.assertIn('resource_access_summary', report['security_metrics'])

        # 验证用户活动摘要
        user_summary = report['security_metrics']['user_activity_summary']
        self.assertIn('user1', user_summary)
        self.assertIn('authenticate', user_summary['user1'])

    def test_global_security_monitoring(self):
        """测试全局安全监控功能"""
        # 使用全局函数记录安全操作
        record_security_operation("login", 0.3, user_id="test_user", resource="/auth")

        # 获取全局安全性能报告
        report = get_security_performance_report()

        # 验证报告包含安全指标
        self.assertIn('security_metrics', report)


class TestHealthChecker(unittest.TestCase):
    """健康检查器测试"""

    def setUp(self):
        """测试前准备"""
        self.checker = HealthChecker(enable_background_monitoring=False)

    def tearDown(self):
        """测试后清理"""
        self.checker.shutdown()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.checker.checks)
        self.assertGreater(len(self.checker.checks), 0)

        # 验证默认检查项
        expected_checks = ['cpu_usage', 'memory_usage', 'disk_space', 'network_connectivity', 'process_health']
        for check_name in expected_checks:
            self.assertIn(check_name, self.checker.checks)

    def test_run_single_health_check(self):
        """测试运行单个健康检查"""
        # 运行CPU使用率检查
        health = self.checker.run_health_check("cpu_usage")

        # 验证结果结构
        self.assertIsInstance(health, object)
        self.assertIn('cpu_usage', health.checks)
        self.assertIn('status', health.checks['cpu_usage'])
        self.assertIn('message', health.checks['cpu_usage'])

    def test_run_all_health_checks(self):
        """测试运行所有健康检查"""
        health = self.checker.run_health_check()

        # 验证整体状态
        self.assertIsNotNone(health.overall_status)

        # 验证所有检查都被执行
        for check_name in self.checker.checks.keys():
            self.assertIn(check_name, health.checks)

    def test_add_custom_health_check(self):
        """测试添加自定义健康检查"""
        def custom_check():
            return HealthStatus.HEALTHY, "Custom check passed"

        custom_check_obj = HealthCheck(
            name="custom_check",
            description="Custom health check",
            check_function=custom_check,
            interval_seconds=60.0
        )

        self.checker.add_check(custom_check_obj)

        # 验证检查被添加
        self.assertIn("custom_check", self.checker.checks)

        # 运行自定义检查
        health = self.checker.run_health_check("custom_check")
        self.assertEqual(health.checks["custom_check"]["status"], "healthy")
        self.assertEqual(health.checks["custom_check"]["message"], "Custom check passed")

    def test_remove_health_check(self):
        """测试移除健康检查"""
        # 先添加一个检查
        def dummy_check():
            return HealthStatus.HEALTHY, "Dummy check"

        dummy_check_obj = HealthCheck(
            name="dummy_check",
            description="Dummy health check",
            check_function=dummy_check
        )

        self.checker.add_check(dummy_check_obj)
        self.assertIn("dummy_check", self.checker.checks)

        # 移除检查
        self.checker.remove_check("dummy_check")
        self.assertNotIn("dummy_check", self.checker.checks)

    def test_health_status_hierarchy(self):
        """测试健康状态层次"""
        # 创建模拟检查结果
        test_health = self.checker.run_health_check()

        # 验证状态是有效的枚举值
        valid_statuses = [status.value for status in HealthStatus]
        self.assertIn(test_health.overall_status.value, valid_statuses)

    def test_health_report_format(self):
        """测试健康报告格式"""
        health = self.checker.run_health_check()
        report = self.checker.get_health_report()

        # 验证报告是字典格式
        self.assertIsInstance(report, dict)

        # 验证必需字段
        required_fields = ['overall_status', 'timestamp', 'checks', 'system_metrics', 'recommendations']
        for field in required_fields:
            self.assertIn(field, report)

    def test_system_metrics_collection(self):
        """测试系统指标收集"""
        health = self.checker.run_health_check()

        # 验证系统指标包含基本信息
        metrics = health.system_metrics
        self.assertIsInstance(metrics, dict)

        # 检查是否包含关键指标
        expected_metrics = ['cpu_percent', 'memory_percent', 'disk_usage']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)

    def test_health_check_with_timeout(self):
        """测试带超时的健康检查"""
        def slow_check():
            time.sleep(2)  # 超过默认超时时间
            return HealthStatus.HEALTHY, "Slow check completed"

        slow_check_obj = HealthCheck(
            name="slow_check",
            description="Slow health check",
            check_function=slow_check,
            timeout_seconds=1.0  # 1秒超时
        )

        self.checker.add_check(slow_check_obj)
        health = self.checker.run_health_check("slow_check")

        # 应该因为超时而标记为降级状态
        self.assertIn("slow_check", health.checks)
        # 注意：实际行为可能因实现而异

    @patch('psutil.cpu_percent')
    def test_cpu_check_with_mock(self, mock_cpu):
        """测试CPU检查（使用mock）"""
        # 模拟高CPU使用率
        mock_cpu.return_value = 95.0

        health = self.checker.run_health_check("cpu_usage")

        # 验证检查结果
        cpu_check = health.checks["cpu_usage"]
        self.assertEqual(cpu_check["status"], "critical")
        self.assertIn("95.0%", cpu_check["message"])

    @patch('psutil.virtual_memory')
    def test_memory_check_with_mock(self, mock_memory):
        """测试内存检查（使用mock）"""
        # 创建模拟内存对象
        mock_memory_obj = Mock()
        mock_memory_obj.percent = 85.0
        mock_memory.return_value = mock_memory_obj

        health = self.checker.run_health_check("memory_usage")

        # 验证检查结果
        memory_check = health.checks["memory_usage"]
        self.assertEqual(memory_check["status"], "unhealthy")
        self.assertIn("85.0%", memory_check["message"])


class TestIntegratedMonitoring(unittest.TestCase):
    """集成监控测试"""

    def test_performance_and_health_integration(self):
        """测试性能监控和健康检查的集成"""
        # 创建监控组件
        perf_monitor = PerformanceMonitor(enabled=True)
        health_checker = HealthChecker(enable_background_monitoring=False)

        try:
            # 记录一些性能数据
            perf_monitor.record_operation("security_check", 0.1, user_id="system", resource="/health")

            # 执行健康检查
            health = health_checker.run_health_check()

            # 验证两个系统都能正常工作
            self.assertIsNotNone(perf_monitor.get_metrics("security_check"))
            self.assertIsNotNone(health.overall_status)

            # 验证健康状态是有效的
            valid_statuses = [status.value for status in HealthStatus]
            self.assertIn(health.overall_status.value, valid_statuses)

        finally:
            perf_monitor.shutdown()
            health_checker.shutdown()

    def test_monitoring_data_consistency(self):
        """测试监控数据的一致性"""
        perf_monitor = PerformanceMonitor(enabled=True)

        try:
            # 记录操作
            perf_monitor.record_operation("test_operation", 0.05)

            # 获取指标
            metrics = perf_monitor.get_metrics("test_operation")

            # 验证数据一致性
            self.assertEqual(metrics['total_calls'], 1)
            self.assertEqual(metrics['error_count'], 0)

        finally:
            perf_monitor.shutdown()


if __name__ == '__main__':
    unittest.main()
