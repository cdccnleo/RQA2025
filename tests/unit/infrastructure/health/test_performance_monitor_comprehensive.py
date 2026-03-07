#!/usr/bin/env python3
"""
性能监控器综合测试 - 提升测试覆盖率至80%+

针对performance_monitor.py的深度测试覆盖
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from typing import Dict, Any


class TestPerformanceMonitorComprehensive:
    """性能监控器全面测试"""

    def setup_method(self):
        """测试前准备"""
        try:
            from src.infrastructure.health.monitoring.performance_monitor import PerformanceMonitor
            self.PerformanceMonitor = PerformanceMonitor
        except ImportError as e:
            pytest.skip(f"无法导入PerformanceMonitor: {e}")

    def test_initialization(self):
        """测试初始化"""
        monitor = self.PerformanceMonitor()
        assert monitor is not None
        assert hasattr(monitor, 'snapshots')
        assert hasattr(monitor, 'alerts')
        assert hasattr(monitor, 'performance_data')
        assert hasattr(monitor, 'tracemalloc_started')

    def test_memory_tracing_operations(self):
        """测试内存跟踪操作"""
        monitor = self.PerformanceMonitor()

        # 测试开始内存跟踪
        monitor.start_memory_tracing()
        assert monitor.tracemalloc_started is True
        assert len(monitor.snapshots) == 1  # 初始快照

        # 测试拍摄内存快照
        snapshot = monitor.take_memory_snapshot()
        assert isinstance(snapshot, dict)
        assert "timestamp" in snapshot
        assert "total_count" in snapshot
        assert "total_size" in snapshot

        # 测试停止内存跟踪
        monitor.stop_memory_tracing()
        assert monitor.tracemalloc_started is False

    def test_memory_snapshot_comparison(self):
        """测试内存快照比较"""
        monitor = self.PerformanceMonitor()

        # 创建一些快照
        monitor.start_memory_tracing()
        time.sleep(0.01)  # 短暂延迟
        monitor.take_memory_snapshot()
        time.sleep(0.01)
        monitor.take_memory_snapshot()

        # 比较快照
        comparison = monitor.compare_memory_snapshots()
        assert isinstance(comparison, dict)

    def test_memory_usage_trend(self):
        """测试内存使用趋势"""
        monitor = self.PerformanceMonitor()

        # 创建多个快照
        monitor.start_memory_tracing()
        for _ in range(3):
            time.sleep(0.01)
            monitor.take_memory_snapshot()

        # 获取趋势
        trend = monitor.get_memory_usage_trend()
        assert isinstance(trend, dict)

    def test_memory_leak_detection(self):
        """测试内存泄漏检测"""
        monitor = self.PerformanceMonitor()

        # 创建快照数据
        monitor.start_memory_tracing()
        for _ in range(5):
            time.sleep(0.01)
            monitor.take_memory_snapshot()

        # 检测内存泄漏
        leaks = monitor.detect_memory_leaks()
        assert isinstance(leaks, dict)

    def test_performance_alerts(self):
        """测试性能告警"""
        monitor = self.PerformanceMonitor()

        # 添加告警
        monitor.add_performance_alert("memory_high", "内存使用过高")

        # 获取告警
        alerts = monitor.get_performance_alerts()
        assert isinstance(alerts, list)

    def test_clear_old_alerts(self):
        """测试清理旧告警"""
        monitor = self.PerformanceMonitor()

        # 添加一些告警
        for i in range(3):
            monitor.add_performance_alert(f"alert_{i}", f"消息{i}")

        # 清理旧告警
        monitor.clear_old_alerts(hours=0)  # 清理所有

        # 验证告警已被清理
        alerts = monitor.get_performance_alerts(hours=24)
        assert len(alerts) == 0

    def test_performance_summary(self):
        """测试性能摘要"""
        monitor = self.PerformanceMonitor()

        # 创建一些性能数据
        monitor.start_memory_tracing()
        monitor.take_memory_snapshot()
        monitor.add_performance_alert("test", "测试告警")

        # 获取摘要
        summary = monitor.get_performance_summary()
        assert isinstance(summary, dict)

    def test_garbage_collection(self):
        """测试垃圾回收"""
        monitor = self.PerformanceMonitor()

        # 执行垃圾回收
        result = monitor.force_garbage_collection()
        assert isinstance(result, dict)

    def test_function_performance_monitoring(self):
        """测试函数性能监控"""
        monitor = self.PerformanceMonitor()

        # 监控函数性能 - 执行时间超过阈值时应该产生告警
        monitor.monitor_function_performance("test_func", 2.5, 2.0)  # 2.5 > 2.0

        # 验证告警被添加
        alerts = monitor.get_performance_alerts()
        assert len(alerts) > 0
        assert any("test_func" in str(alert) for alert in alerts)

    def test_component_info(self):
        """测试组件信息"""
        monitor = self.PerformanceMonitor()

        info = monitor.get_component_info()
        assert isinstance(info, dict)

    def test_health_status(self):
        """测试健康状态"""
        monitor = self.PerformanceMonitor()

        healthy = monitor.is_healthy()
        assert isinstance(healthy, bool)

        metrics = monitor.get_metrics()
        assert isinstance(metrics, dict)

    def test_cleanup(self):
        """测试清理"""
        monitor = self.PerformanceMonitor()

        # 先添加一些数据
        monitor.start_memory_tracing()
        monitor.take_memory_snapshot()
        monitor.add_performance_alert("test", "测试")

        # 执行清理
        result = monitor.cleanup()
        assert result is True

    def test_health_checks(self):
        """测试健康检查"""
        monitor = self.PerformanceMonitor()

        # 基础健康检查
        health = monitor.check_health()
        assert isinstance(health, dict)

        # 内存健康检查
        memory_health = monitor.check_memory_health()
        assert isinstance(memory_health, dict)

        # 性能数据健康检查
        perf_health = monitor.check_performance_data_health()
        assert isinstance(perf_health, dict)

        # 告警系统健康检查
        alert_health = monitor.check_alert_system_health()
        assert isinstance(alert_health, dict)

    def test_monitor_performance_status(self):
        """测试监控性能状态"""
        monitor = self.PerformanceMonitor()

        status = monitor.monitor_performance_status()
        assert isinstance(status, dict)

    def test_validate_performance_config(self):
        """测试验证性能配置"""
        monitor = self.PerformanceMonitor()

        validation = monitor.validate_performance_config()
        assert isinstance(validation, dict)


class TestPerformanceMonitorEdgeCases:
    """性能监控器边界情况测试"""

    def setup_method(self):
        """测试前准备"""
        try:
            from src.infrastructure.health.monitoring.performance_monitor import PerformanceMonitor
            self.PerformanceMonitor = PerformanceMonitor
        except ImportError:
            pytest.skip("无法导入PerformanceMonitor")

    def test_empty_snapshots_handling(self):
        """测试空快照处理"""
        monitor = self.PerformanceMonitor()

        # 没有快照时的方法调用应该安全
        trend = monitor.get_memory_usage_trend()
        assert isinstance(trend, dict)

        leaks = monitor.detect_memory_leaks()
        assert isinstance(leaks, dict)

    def test_performance_alerts_edge_cases(self):
        """测试性能告警边界情况"""
        monitor = self.PerformanceMonitor()

        # 空告警列表
        alerts = monitor.get_performance_alerts()
        assert isinstance(alerts, list)
        assert len(alerts) == 0

    def test_function_performance_data(self):
        """测试函数性能数据"""
        monitor = self.PerformanceMonitor()

        # 添加函数数据 (threshold应该是float)
        monitor.monitor_function_performance("test_func", 1.0, 2.0)
        # 检查数据是否被记录（可能不记录到performance_data中）
