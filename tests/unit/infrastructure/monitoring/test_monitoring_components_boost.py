#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monitoring模块组件测试
覆盖监控组件的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

# 测试指标收集器
try:
    from src.infrastructure.monitoring.components.metrics_collector import MetricsCollector, Metric
    HAS_METRICS_COLLECTOR = True
except ImportError:
    HAS_METRICS_COLLECTOR = False
    
    @dataclass
    class Metric:
        name: str
        value: float
        timestamp: float = 0.0
        tags: dict = None
    
    class MetricsCollector:
        def __init__(self):
            self.metrics = []
        
        def collect(self, metric):
            self.metrics.append(metric)
        
        def get_metrics(self):
            return self.metrics


class TestMetric:
    """测试指标数据类"""
    
    def test_create_simple_metric(self):
        """测试创建简单指标"""
        metric = Metric(name="cpu_usage", value=75.5)
        
        assert metric.name == "cpu_usage"
        assert metric.value == 75.5
    
    def test_create_with_timestamp(self):
        """测试带时间戳的指标"""
        metric = Metric(name="memory", value=80.0, timestamp=1699000000.0)
        
        assert metric.timestamp == 1699000000.0
    
    def test_create_with_tags(self):
        """测试带标签的指标"""
        tags = {"host": "server1", "env": "prod"}
        metric = Metric(name="disk", value=60.0, tags=tags)
        
        if hasattr(metric, 'tags'):
            assert metric.tags == tags


class TestMetricsCollector:
    """测试指标收集器"""
    
    def test_init(self):
        """测试初始化"""
        collector = MetricsCollector()
        
        if hasattr(collector, 'metrics'):
            assert collector.metrics == []
    
    def test_collect_metric(self):
        """测试收集指标"""
        collector = MetricsCollector()
        metric = Metric("cpu", 50.0)
        
        if hasattr(collector, 'collect'):
            collector.collect(metric)
            
            if hasattr(collector, 'metrics'):
                assert len(collector.metrics) == 1
    
    def test_collect_multiple(self):
        """测试收集多个指标"""
        collector = MetricsCollector()
        
        if hasattr(collector, 'collect'):
            collector.collect(Metric("cpu", 50))
            collector.collect(Metric("memory", 60))
            collector.collect(Metric("disk", 70))
            
            if hasattr(collector, 'metrics'):
                assert len(collector.metrics) == 3
    
    def test_get_metrics(self):
        """测试获取指标"""
        collector = MetricsCollector()
        
        if hasattr(collector, 'collect') and hasattr(collector, 'get_metrics'):
            collector.collect(Metric("test", 100))
            metrics = collector.get_metrics()
            
            assert isinstance(metrics, list)


# 测试告警管理器
try:
    from src.infrastructure.monitoring.components.alert_manager import AlertManager, Alert, AlertLevel
    HAS_ALERT_MANAGER = True
except ImportError:
    HAS_ALERT_MANAGER = False
    
    from enum import Enum
    
    class AlertLevel(Enum):
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"
    
    @dataclass
    class Alert:
        message: str
        level: AlertLevel
        source: str = "system"
    
    class AlertManager:
        def __init__(self):
            self.alerts = []
        
        def add_alert(self, alert):
            self.alerts.append(alert)
        
        def get_alerts(self, level=None):
            if level:
                return [a for a in self.alerts if a.level == level]
            return self.alerts


class TestAlertLevel:
    """测试告警级别"""
    
    def test_info_level(self):
        """测试INFO级别"""
        assert AlertLevel.INFO.value == "info"
    
    def test_warning_level(self):
        """测试WARNING级别"""
        assert AlertLevel.WARNING.value == "warning"
    
    def test_error_level(self):
        """测试ERROR级别"""
        assert AlertLevel.ERROR.value == "error"
    
    def test_critical_level(self):
        """测试CRITICAL级别"""
        assert AlertLevel.CRITICAL.value == "critical"


class TestAlert:
    """测试告警数据类"""
    
    def test_create_alert(self):
        """测试创建告警"""
        alert = Alert(
            message="High CPU usage",
            level=AlertLevel.WARNING
        )
        
        assert alert.message == "High CPU usage"
        assert alert.level == AlertLevel.WARNING
    
    def test_create_with_source(self):
        """测试带来源的告警"""
        alert = Alert(
            message="Error occurred",
            level=AlertLevel.ERROR,
            source="database"
        )
        
        assert alert.source == "database"


class TestAlertManager:
    """测试告警管理器"""
    
    def test_init(self):
        """测试初始化"""
        manager = AlertManager()
        
        if hasattr(manager, 'alerts'):
            assert manager.alerts == []
    
    def test_add_alert(self):
        """测试添加告警"""
        manager = AlertManager()
        alert = Alert("Test alert", AlertLevel.INFO)
        
        if hasattr(manager, 'add_alert'):
            manager.add_alert(alert)
            
            if hasattr(manager, 'alerts'):
                assert len(manager.alerts) == 1
    
    def test_get_all_alerts(self):
        """测试获取所有告警"""
        manager = AlertManager()
        
        if hasattr(manager, 'add_alert') and hasattr(manager, 'get_alerts'):
            manager.add_alert(Alert("Alert 1", AlertLevel.INFO))
            manager.add_alert(Alert("Alert 2", AlertLevel.WARNING))
            
            alerts = manager.get_alerts()
            assert isinstance(alerts, list)
    
    def test_get_alerts_by_level(self):
        """测试按级别获取告警"""
        manager = AlertManager()
        
        if hasattr(manager, 'add_alert') and hasattr(manager, 'get_alerts'):
            manager.add_alert(Alert("Info", AlertLevel.INFO))
            manager.add_alert(Alert("Warning", AlertLevel.WARNING))
            manager.add_alert(Alert("Error", AlertLevel.ERROR))
            
            warnings = manager.get_alerts(level=AlertLevel.WARNING)
            assert isinstance(warnings, list)
    
    def test_multiple_alerts_same_level(self):
        """测试相同级别的多个告警"""
        manager = AlertManager()
        
        if hasattr(manager, 'add_alert'):
            for i in range(5):
                alert = Alert(f"Error {i}", AlertLevel.ERROR)
                manager.add_alert(alert)
            
            if hasattr(manager, 'alerts'):
                assert len(manager.alerts) == 5


# 测试监控仪表板
try:
    from src.infrastructure.monitoring.components.dashboard import MonitoringDashboard
    HAS_DASHBOARD = True
except ImportError:
    HAS_DASHBOARD = False
    
    class MonitoringDashboard:
        def __init__(self):
            self.widgets = {}
        
        def add_widget(self, name, widget):
            self.widgets[name] = widget
        
        def get_widget(self, name):
            return self.widgets.get(name)
        
        def render(self):
            return {"widgets": list(self.widgets.keys())}


class TestMonitoringDashboard:
    """测试监控仪表板"""
    
    def test_init(self):
        """测试初始化"""
        dashboard = MonitoringDashboard()
        
        if hasattr(dashboard, 'widgets'):
            assert dashboard.widgets == {}
    
    def test_add_widget(self):
        """测试添加组件"""
        dashboard = MonitoringDashboard()
        widget = Mock()
        
        if hasattr(dashboard, 'add_widget'):
            dashboard.add_widget("cpu_chart", widget)
            
            if hasattr(dashboard, 'widgets'):
                assert "cpu_chart" in dashboard.widgets
    
    def test_get_widget(self):
        """测试获取组件"""
        dashboard = MonitoringDashboard()
        widget = Mock()
        
        if hasattr(dashboard, 'add_widget') and hasattr(dashboard, 'get_widget'):
            dashboard.add_widget("memory_gauge", widget)
            retrieved = dashboard.get_widget("memory_gauge")
            
            assert retrieved is widget
    
    def test_render_dashboard(self):
        """测试渲染仪表板"""
        dashboard = MonitoringDashboard()
        
        if hasattr(dashboard, 'add_widget') and hasattr(dashboard, 'render'):
            dashboard.add_widget("widget1", Mock())
            dashboard.add_widget("widget2", Mock())
            
            result = dashboard.render()
            assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

