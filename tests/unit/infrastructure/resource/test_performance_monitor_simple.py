#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控器简单测试

测试PerformanceMonitorLegacy的基本功能
"""

import pytest
from unittest.mock import Mock


class TestPerformanceMonitorLegacy:
    """性能监控器遗留类测试"""

    def test_performance_monitor_initialization(self):
        """测试性能监控器初始化"""
        try:
            from src.infrastructure.resource.monitoring.performance.performance_monitor import PerformanceMonitorLegacy

            monitor = PerformanceMonitorLegacy()

            # 测试基本属性
            assert hasattr(monitor, 'config')
            assert hasattr(monitor, 'logger')
            assert hasattr(monitor, 'error_handler')
            assert hasattr(monitor, 'collector')
            assert hasattr(monitor, 'storage')
            assert hasattr(monitor, 'analyzer')

        except ImportError:
            pytest.skip("PerformanceMonitorLegacy not available")

    def test_performance_monitor_initialization_with_config(self):
        """测试带配置的性能监控器初始化"""
        try:
            from src.infrastructure.resource.monitoring.performance.performance_monitor import PerformanceMonitorLegacy, PerformanceConfig

            config = PerformanceConfig()
            config.collection_interval = 30

            monitor = PerformanceMonitorLegacy(config)

            # 验证配置被正确设置
            assert monitor.config == config

        except ImportError:
            pytest.skip("PerformanceMonitorLegacy initialization with config not available")

    def test_get_current_metrics(self):
        """测试获取当前指标"""
        try:
            from src.infrastructure.resource.monitoring.performance.performance_monitor import PerformanceMonitorLegacy

            monitor = PerformanceMonitorLegacy()

            # 测试获取当前指标
            metrics = monitor.get_current_metrics()
            # 可能返回None或实际指标
            assert metrics is None or hasattr(metrics, 'cpu_percent')

        except ImportError:
            pytest.skip("Get current metrics not available")

    def test_start_stop_monitoring(self):
        """测试启动和停止监控"""
        try:
            from src.infrastructure.resource.monitoring.performance.performance_monitor import PerformanceMonitorLegacy

            monitor = PerformanceMonitorLegacy()

            # 测试启动监控
            monitor.start_monitoring()

            # 测试停止监控
            monitor.stop_monitoring()

        except ImportError:
            pytest.skip("Start/stop monitoring not available")

    def test_get_metrics_history(self):
        """测试获取指标历史"""
        try:
            from src.infrastructure.resource.monitoring.performance.performance_monitor import PerformanceMonitorLegacy

            monitor = PerformanceMonitorLegacy()

            # 测试获取指标历史
            history = monitor.get_metrics_history()
            assert isinstance(history, list)

        except ImportError:
            pytest.skip("Get metrics history not available")

    def test_get_performance_stats(self):
        """测试获取性能统计"""
        try:
            from src.infrastructure.resource.monitoring.performance.performance_monitor import PerformanceMonitorLegacy

            monitor = PerformanceMonitorLegacy()

            # 测试获取性能统计
            stats = monitor.get_performance_stats()
            assert isinstance(stats, dict)

        except ImportError:
            pytest.skip("Get performance stats not available")

    def test_analyze_performance(self):
        """测试性能分析"""
        try:
            from src.infrastructure.resource.monitoring.performance.performance_monitor import PerformanceMonitorLegacy

            monitor = PerformanceMonitorLegacy()

            # 测试性能分析
            analysis = monitor.analyze_performance()
            assert isinstance(analysis, dict)

        except ImportError:
            pytest.skip("Analyze performance not available")

    def test_check_performance_thresholds(self):
        """测试性能阈值检查"""
        try:
            from src.infrastructure.resource.monitoring.performance.performance_monitor import PerformanceMonitorLegacy

            monitor = PerformanceMonitorLegacy()

            # 测试性能阈值检查
            alerts = monitor.check_performance_thresholds()
            assert isinstance(alerts, list)

        except ImportError:
            pytest.skip("Check performance thresholds not available")

    def test_generate_performance_report(self):
        """测试生成性能报告"""
        try:
            from src.infrastructure.resource.monitoring.performance.performance_monitor import PerformanceMonitorLegacy

            monitor = PerformanceMonitorLegacy()

            # 测试生成性能报告
            report = monitor.generate_performance_report()
            assert isinstance(report, dict)

        except ImportError:
            pytest.skip("Generate performance report not available")