# -*- coding: utf-8 -*-
"""
监控系统单元测试
测试 MonitoringSystem 的核心功能
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.monitoring.monitoring_system import MonitoringSystem


class TestMonitoringSystem:
    """监控系统测试"""

    @patch('src.monitoring.monitoring_system.PerformanceAnalyzer')
    @patch('src.monitoring.monitoring_system.MonitorComponents')
    @patch('src.monitoring.monitoring_system.IntelligentAlertSystem')
    def test_monitoring_system_initialization(self, mock_alert_system, mock_monitor_components, mock_performance_analyzer):
        """测试监控系统初始化"""
        # Mock 依赖类
        mock_performance_analyzer.return_value = Mock()
        mock_monitor_components.return_value = Mock()
        mock_alert_system.return_value = Mock()

        config = {"monitoring_interval": 30, "alert_threshold": 0.8}
        system = MonitoringSystem(config)

        assert system.config == config
        assert hasattr(system, '_components')
        assert hasattr(system, '_monitors')
        assert hasattr(system, '_performance_analyzer')
        assert hasattr(system, '_intelligent_alert_system')
        assert system._initialized is False

    @patch('src.monitoring.monitoring_system.PerformanceAnalyzer')
    @patch('src.monitoring.monitoring_system.MonitorComponents')
    @patch('src.monitoring.monitoring_system.IntelligentAlertSystem')
    def test_initialize_monitoring(self, mock_alert_system, mock_monitor_components, mock_performance_analyzer):
        """测试初始化监控"""
        # Mock 依赖类
        mock_performance_analyzer.return_value = Mock()
        mock_monitor_components.return_value = Mock()
        mock_alert_system.return_value = Mock()

        system = MonitoringSystem()

        config = {
            "monitoring_interval": 30,
            "alert_threshold": 0.8,
            "components": ["cpu_monitor", "memory_monitor"]
        }

        result = system.initialize_monitoring(config)

        assert result is True
        assert system._initialized is True
        assert system.config["monitoring_interval"] == 30

    @patch('src.monitoring.monitoring_system.PerformanceAnalyzer')
    @patch('src.monitoring.monitoring_system.MonitorComponents')
    @patch('src.monitoring.monitoring_system.IntelligentAlertSystem')
    def test_start_monitoring(self, mock_alert_system, mock_monitor_components, mock_performance_analyzer):
        """测试启动监控"""
        # Mock 依赖类
        mock_performance_analyzer.return_value = Mock()
        mock_monitor_components.return_value = Mock()
        mock_alert_system.return_value = Mock()

        system = MonitoringSystem()

        # 先初始化
        system._initialized = True

        # Mock 监控器
        mock_monitor = Mock()
        mock_monitor.start_monitoring.return_value = True
        system._monitors = {"cpu_monitor": mock_monitor}

        result = system.start_monitoring()

        assert result is True
        mock_monitor.start_monitoring.assert_called_once()

    @patch('src.monitoring.monitoring_system.PerformanceAnalyzer')
    @patch('src.monitoring.monitoring_system.MonitorComponents')
    @patch('src.monitoring.monitoring_system.IntelligentAlertSystem')
    def test_stop_monitoring(self, mock_alert_system, mock_monitor_components, mock_performance_analyzer):
        """测试停止监控"""
        # Mock 依赖类
        mock_performance_analyzer.return_value = Mock()
        mock_monitor_components.return_value = Mock()
        mock_alert_system.return_value = Mock()

        system = MonitoringSystem()

        # Mock 监控器
        mock_monitor = Mock()
        mock_monitor.stop_monitoring.return_value = True
        system._monitors = {"cpu_monitor": mock_monitor}

        result = system.stop_monitoring()

        assert result is True
        mock_monitor.stop_monitoring.assert_called_once()

    @patch('src.monitoring.monitoring_system.PerformanceAnalyzer')
    @patch('src.monitoring.monitoring_system.MonitorComponents')
    @patch('src.monitoring.monitoring_system.IntelligentAlertSystem')
    def test_get_system_status(self, mock_alert_system, mock_monitor_components, mock_performance_analyzer):
        """测试获取系统状态"""
        # Mock 依赖类
        mock_performance_analyzer.return_value = Mock()
        mock_monitor_components.return_value = Mock()
        mock_alert_system.return_value = Mock()

        system = MonitoringSystem()

        # Mock 组件
        mock_component = Mock()
        mock_component.get_status.return_value = {"status": "healthy", "cpu": 75}
        system._components = {"cpu_monitor": mock_component}

        status = system.get_system_status()

        assert isinstance(status, dict)
        assert "overall_status" in status
        assert "components" in status
        assert "timestamp" in status

    @patch('src.monitoring.monitoring_system.PerformanceAnalyzer')
    @patch('src.monitoring.monitoring_system.MonitorComponents')
    @patch('src.monitoring.monitoring_system.IntelligentAlertSystem')
    def test_collect_metrics(self, mock_alert_system, mock_monitor_components, mock_performance_analyzer):
        """测试收集指标"""
        # Mock 依赖类
        mock_performance_analyzer.return_value = Mock()
        mock_monitor_components.return_value = Mock()
        mock_alert_system.return_value = Mock()

        system = MonitoringSystem()

        # Mock 监控器
        mock_monitor = Mock()
        mock_monitor.collect_metrics.return_value = {"cpu_usage": 75, "memory_usage": 80}
        system._monitors = {"system_monitor": mock_monitor}

        metrics = system.collect_metrics()

        assert isinstance(metrics, dict)
        assert "timestamp" in metrics
        assert "monitors" in metrics

    @patch('src.monitoring.monitoring_system.PerformanceAnalyzer')
    @patch('src.monitoring.monitoring_system.MonitorComponents')
    @patch('src.monitoring.monitoring_system.IntelligentAlertSystem')
    def test_register_monitor(self, mock_alert_system, mock_monitor_components, mock_performance_analyzer):
        """测试注册监控器"""
        # Mock 依赖类
        mock_performance_analyzer.return_value = Mock()
        mock_monitor_components.return_value = Mock()
        mock_alert_system.return_value = Mock()

        system = MonitoringSystem()

        mock_monitor = Mock()
        mock_monitor.monitor_type = "cpu_monitor"

        result = system.register_monitor(mock_monitor)

        assert result is True
        assert "cpu_monitor" in system._monitors
        assert system._monitors["cpu_monitor"] == mock_monitor

    @patch('src.monitoring.monitoring_system.PerformanceAnalyzer')
    @patch('src.monitoring.monitoring_system.MonitorComponents')
    @patch('src.monitoring.monitoring_system.IntelligentAlertSystem')
    def test_unregister_monitor(self, mock_alert_system, mock_monitor_components, mock_performance_analyzer):
        """测试注销监控器"""
        # Mock 依赖类
        mock_performance_analyzer.return_value = Mock()
        mock_monitor_components.return_value = Mock()
        mock_alert_system.return_value = Mock()

        system = MonitoringSystem()

        # 先注册监控器
        mock_monitor = Mock()
        mock_monitor.monitor_type = "cpu_monitor"
        system._monitors["cpu_monitor"] = mock_monitor

        result = system.unregister_monitor("cpu_monitor")

        assert result is True
        assert "cpu_monitor" not in system._monitors

    @patch('src.monitoring.monitoring_system.PerformanceAnalyzer')
    @patch('src.monitoring.monitoring_system.MonitorComponents')
    @patch('src.monitoring.monitoring_system.IntelligentAlertSystem')
    def test_get_monitor(self, mock_alert_system, mock_monitor_components, mock_performance_analyzer):
        """测试获取监控器"""
        # Mock 依赖类
        mock_performance_analyzer.return_value = Mock()
        mock_monitor_components.return_value = Mock()
        mock_alert_system.return_value = Mock()

        system = MonitoringSystem()

        # 添加监控器
        mock_monitor = Mock()
        mock_monitor.monitor_type = "cpu_monitor"
        system._monitors["cpu_monitor"] = mock_monitor

        retrieved_monitor = system.get_monitor("cpu_monitor")

        assert retrieved_monitor == mock_monitor

        # 测试获取不存在的监控器
        result = system.get_monitor("non_existent")
        assert result is None

    @patch('src.monitoring.monitoring_system.PerformanceAnalyzer')
    @patch('src.monitoring.monitoring_system.MonitorComponents')
    @patch('src.monitoring.monitoring_system.IntelligentAlertSystem')
    def test_get_all_monitors(self, mock_alert_system, mock_monitor_components, mock_performance_analyzer):
        """测试获取所有监控器"""
        # Mock 依赖类
        mock_performance_analyzer.return_value = Mock()
        mock_monitor_components.return_value = Mock()
        mock_alert_system.return_value = Mock()

        system = MonitoringSystem()

        # 添加多个监控器
        mock_monitor1 = Mock()
        mock_monitor1.monitor_type = "cpu_monitor"
        mock_monitor2 = Mock()
        mock_monitor2.monitor_type = "memory_monitor"

        system._monitors = {
            "cpu_monitor": mock_monitor1,
            "memory_monitor": mock_monitor2
        }

        all_monitors = system.get_all_monitors()

        assert isinstance(all_monitors, dict)
        assert len(all_monitors) == 2
        assert "cpu_monitor" in all_monitors
        assert "memory_monitor" in all_monitors

    @patch('src.monitoring.monitoring_system.PerformanceAnalyzer')
    @patch('src.monitoring.monitoring_system.MonitorComponents')
    @patch('src.monitoring.monitoring_system.IntelligentAlertSystem')
    def test_start_all_monitors(self, mock_alert_system, mock_monitor_components, mock_performance_analyzer):
        """测试启动所有监控器"""
        # Mock 依赖类
        mock_performance_analyzer.return_value = Mock()
        mock_monitor_components.return_value = Mock()
        mock_alert_system.return_value = Mock()

        system = MonitoringSystem()

        # Mock 监控器
        mock_monitor1 = Mock()
        mock_monitor1.start_monitoring.return_value = True
        mock_monitor2 = Mock()
        mock_monitor2.start_monitoring.return_value = True

        system._monitors = {
            "cpu_monitor": mock_monitor1,
            "memory_monitor": mock_monitor2
        }

        result = system.start_all_monitors()

        assert result is True
        mock_monitor1.start_monitoring.assert_called_once()
        mock_monitor2.start_monitoring.assert_called_once()

    @patch('src.monitoring.monitoring_system.PerformanceAnalyzer')
    @patch('src.monitoring.monitoring_system.MonitorComponents')
    @patch('src.monitoring.monitoring_system.IntelligentAlertSystem')
    def test_get_performance_analyzer(self, mock_alert_system, mock_monitor_components, mock_performance_analyzer):
        """测试获取性能分析器"""
        mock_perf_analyzer = Mock()
        mock_performance_analyzer.return_value = mock_perf_analyzer
        mock_monitor_components.return_value = Mock()
        mock_alert_system.return_value = Mock()

        system = MonitoringSystem()

        analyzer = system.get_performance_analyzer()

        assert analyzer == mock_perf_analyzer

    @patch('src.monitoring.monitoring_system.PerformanceAnalyzer')
    @patch('src.monitoring.monitoring_system.MonitorComponents')
    @patch('src.monitoring.monitoring_system.IntelligentAlertSystem')
    def test_get_monitor_components(self, mock_alert_system, mock_monitor_components, mock_performance_analyzer):
        """测试获取监控组件"""
        mock_performance_analyzer.return_value = Mock()
        mock_monitor_comps = Mock()
        mock_monitor_components.return_value = mock_monitor_comps
        mock_alert_system.return_value = Mock()

        system = MonitoringSystem()

        components = system.get_monitor_components()

        assert components == mock_monitor_comps

    @patch('src.monitoring.monitoring_system.PerformanceAnalyzer')
    @patch('src.monitoring.monitoring_system.MonitorComponents')
    @patch('src.monitoring.monitoring_system.IntelligentAlertSystem')
    def test_get_intelligent_alert_system(self, mock_alert_system, mock_monitor_components, mock_performance_analyzer):
        """测试获取智能告警系统"""
        mock_performance_analyzer.return_value = Mock()
        mock_monitor_components.return_value = Mock()
        mock_alert_sys = Mock()
        mock_alert_system.return_value = mock_alert_sys

        system = MonitoringSystem()

        alert_system = system.get_intelligent_alert_system()

        assert alert_system == mock_alert_sys
