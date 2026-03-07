#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 性能监控面板基础测试
验证PerformanceMonitorDashboard的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class TestPerformanceMonitorDashboard(unittest.TestCase):
    """测试性能监控面板"""

    def setUp(self):
        """测试前准备"""
        self.monitor = None

    def tearDown(self):
        """测试后清理"""
        if self.monitor and hasattr(self.monitor, 'stop'):
            try:
                self.monitor.stop()
            except:
                pass

    # ==================== 基础功能测试 ====================

    def test_performance_monitor_initialization(self):
        """测试性能监控面板初始化"""
        from src.infrastructure.config.monitoring.performance_monitor_dashboard import PerformanceMonitorDashboard

        # 测试默认初始化
        monitor = PerformanceMonitorDashboard()
        self.assertIsInstance(monitor, PerformanceMonitorDashboard)
        self.assertIsInstance(monitor._metrics, list)
        self.assertIsInstance(monitor._system_resources, list)
        self.assertIsInstance(monitor._operation_stats, dict)
        self.assertTrue(hasattr(monitor, '_monitor_thread'))

        # 测试自定义配置初始化
        monitor_custom = PerformanceMonitorDashboard(
            storage_path="/tmp/test_monitoring",
            retention_days=7,
            enable_system_monitoring=False
        )
        self.assertEqual(str(monitor_custom.storage_path), "\\tmp\\test_monitoring")
        self.assertEqual(monitor_custom.retention_days, 7)
        self.assertFalse(monitor_custom.enable_system_monitoring)

    def test_start_and_stop_monitoring(self):
        """测试启动和停止监控"""
        from src.infrastructure.config.monitoring.performance_monitor_dashboard import PerformanceMonitorDashboard

        monitor = PerformanceMonitorDashboard(enable_system_monitoring=False)

        # 测试启动监控
        monitor.start_monitoring()
        self.assertTrue(monitor._monitoring_active)

        # 测试停止监控
        monitor.stop_monitoring()
        self.assertFalse(monitor._monitoring_active)

    # ==================== 性能指标记录测试 ====================

    def test_record_operation(self):
        """测试记录操作"""
        from src.infrastructure.config.monitoring.performance_monitor_dashboard import PerformanceMonitorDashboard

        monitor = PerformanceMonitorDashboard(enable_system_monitoring=False)
        monitor.start_monitoring()

        # 记录操作
        result = monitor.record_operation(
            operation_type="config_load",
            duration=500.0,  # 毫秒
            success=True,
            metadata={"source": "test.json"}
        )

        self.assertTrue(result)

        # 验证指标已记录
        self.assertTrue(len(monitor._metrics) > 0)

        monitor.stop_monitoring()

    def test_record_failed_operation(self):
        """测试记录失败操作"""
        from src.infrastructure.config.monitoring.performance_monitor_dashboard import PerformanceMonitorDashboard

        monitor = PerformanceMonitorDashboard(enable_system_monitoring=False)
        monitor.start_monitoring()

        # 记录失败操作
        monitor.record_operation(
            operation_type="config_save",
            duration=2000.0,
            success=False,
            metadata={"error": "permission denied"}
        )

        # 验证指标已记录
        self.assertTrue(len(monitor._metrics) > 0)

        monitor.stop_monitoring()

    # ==================== 统计信息测试 ====================

    def test_get_operation_stats(self):
        """测试获取操作统计"""
        from src.infrastructure.config.monitoring.performance_monitor_dashboard import PerformanceMonitorDashboard

        monitor = PerformanceMonitorDashboard(enable_system_monitoring=False)
        monitor.start_monitoring()

        # 记录一些操作
        monitor.record_operation("test_op", 100.0, True)
        monitor.record_operation("test_op", 200.0, True)
        monitor.record_operation("test_op", 500.0, False)

        # 获取统计
        stats = monitor.get_operation_stats()

        self.assertIsInstance(stats, dict)
        if "test_op" in stats:
            op_stats = stats["test_op"]
            self.assertEqual(op_stats["count"], 3)
            self.assertEqual(op_stats["success_count"], 2)
            self.assertEqual(op_stats["error_count"], 1)

        monitor.stop_monitoring()

    # ==================== 系统资源监控测试 ====================

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_collect_system_resources(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """测试收集系统资源"""
        from src.infrastructure.config.monitoring.performance_monitor_dashboard import PerformanceMonitorDashboard

        # Mock系统资源数据
        mock_cpu.return_value = 45.2
        mock_memory.return_value.percent = 67.8
        mock_disk.return_value.percent = 23.4
        mock_net_io.return_value.bytes_sent = 1000000
        mock_net_io.return_value.bytes_recv = 2000000

        monitor = PerformanceMonitorDashboard(enable_system_monitoring=True)
        monitor.start_monitoring()

        # 等待一次收集周期
        time.sleep(0.1)

        # 验证系统资源已收集
        self.assertTrue(len(monitor._system_resources) > 0)

        monitor.stop_monitoring()

    # ==================== 健康检查测试 ====================

    def test_health_check(self):
        """测试健康检查"""
        from src.infrastructure.config.monitoring.performance_monitor_dashboard import PerformanceMonitorDashboard

        monitor = PerformanceMonitorDashboard(enable_system_monitoring=False)
        monitor.start_monitoring()

        # 执行健康检查
        health = monitor.health_check()

        self.assertIsInstance(health, dict)
        self.assertIn("status", health)
        self.assertIn("timestamp", health)

        monitor.stop_monitoring()

    # ==================== 边界情况测试 ====================

    def test_empty_operation_type(self):
        """测试空操作类型"""
        from src.infrastructure.config.monitoring.performance_monitor_dashboard import PerformanceMonitorDashboard

        monitor = PerformanceMonitorDashboard(enable_system_monitoring=False)
        monitor.start_monitoring()

        # 记录空操作类型的指标
        result = monitor.record_operation("", 100.0, True)
        self.assertTrue(result)

        monitor.stop_monitoring()

    def test_zero_duration(self):
        """测试零持续时间"""
        from src.infrastructure.config.monitoring.performance_monitor_dashboard import PerformanceMonitorDashboard

        monitor = PerformanceMonitorDashboard(enable_system_monitoring=False)
        monitor.start_monitoring()

        # 记录零持续时间的指标
        result = monitor.record_operation("zero_duration", 0.0, True)
        self.assertTrue(result)

        monitor.stop_monitoring()

    # ==================== 性能测试 ====================

    def test_performance_under_load(self):
        """测试负载下的性能"""
        from src.infrastructure.config.monitoring.performance_monitor_dashboard import PerformanceMonitorDashboard

        monitor = PerformanceMonitorDashboard(enable_system_monitoring=False)
        monitor.start_monitoring()

        # 测试大量操作记录的性能
        start_time = time.time()

        for i in range(1000):
            monitor.record_operation(
                operation_type=f"perf_test_{i % 10}",
                duration=1.0 + (i % 100) * 0.1,
                success=(i % 100 != 0),
                metadata={"index": i}
            )

        end_time = time.time()
        total_time = end_time - start_time

        # 验证性能（1000个操作应该在合理时间内完成）
        self.assertLess(total_time, 2.0)  # 2秒内完成

        monitor.stop_monitoring()


if __name__ == '__main__':
    unittest.main()
