#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统监控器简单测试

测试SystemMonitorFacade的基本功能
"""

import pytest


class TestSystemMonitorFacade:
    """系统监控器门面测试"""

    def test_system_monitor_facade_initialization(self):
        """测试系统监控器门面初始化"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade

            monitor = SystemMonitorFacade()

            # 测试基本属性
            assert hasattr(monitor, 'config')
            assert hasattr(monitor, 'info_collector')
            assert hasattr(monitor, 'metrics_calculator')
            assert hasattr(monitor, 'alert_manager')
            assert hasattr(monitor, 'monitor_engine')

        except ImportError:
            pytest.skip("SystemMonitorFacade not available")

    def test_facade_initialization_with_config(self):
        """测试带配置的门面初始化"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade, SystemMonitorConfig

            config = SystemMonitorConfig()
            config.check_interval = 30

            monitor = SystemMonitorFacade(config)

            # 验证配置被正确设置
            assert monitor.config == config

        except ImportError:
            pytest.skip("SystemMonitorFacade initialization with config not available")

    def test_system_info_retrieval(self):
        """测试系统信息检索"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade

            monitor = SystemMonitorFacade()

            # 测试系统信息获取
            system_info = monitor.get_system_info()
            assert isinstance(system_info, dict)

        except ImportError:
            pytest.skip("System info retrieval not available")

    def test_system_resources_retrieval(self):
        """测试系统资源检索"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade

            monitor = SystemMonitorFacade()

            # 测试系统资源获取
            system_resources = monitor.get_system_resources()
            assert isinstance(system_resources, dict)

        except ImportError:
            pytest.skip("System resources retrieval not available")

    def test_monitoring_lifecycle(self):
        """测试监控生命周期"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade

            monitor = SystemMonitorFacade()

            # 测试启动和停止监控
            monitor.start_monitoring()
            monitor.stop_monitoring()

        except ImportError:
            pytest.skip("Monitoring lifecycle not available")

    def test_stats_retrieval(self):
        """测试统计信息检索"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade

            monitor = SystemMonitorFacade()

            # 测试获取统计信息
            stats = monitor.get_stats(current=True)
            assert isinstance(stats, dict)

        except ImportError:
            pytest.skip("Stats retrieval not available")

    def test_performance_report(self):
        """测试性能报告"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade

            monitor = SystemMonitorFacade()

            # 测试获取性能报告
            performance_report = monitor.get_performance_report()
            assert performance_report is not None

        except ImportError:
            pytest.skip("Performance report not available")

    def test_alerts_history(self):
        """测试告警历史"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade

            monitor = SystemMonitorFacade()

            # 测试获取告警历史
            alerts_history = monitor.get_alerts_history()
            assert isinstance(alerts_history, list)

        except ImportError:
            pytest.skip("Alerts history not available")