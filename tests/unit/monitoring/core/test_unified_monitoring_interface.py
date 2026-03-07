#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一监控系统接口测试
测试unified_monitoring_interface.py中的枚举、数据类和接口定义
"""

import pytest
from datetime import datetime
from abc import ABC

import sys
import importlib
from pathlib import Path
import pytest

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    core_unified_monitoring_interface_module = importlib.import_module('src.monitoring.core.unified_monitoring_interface')
    MonitorType = getattr(core_unified_monitoring_interface_module, 'MonitorType', None)
    AlertLevel = getattr(core_unified_monitoring_interface_module, 'AlertLevel', None)
    AlertStatus = getattr(core_unified_monitoring_interface_module, 'AlertStatus', None)
    MetricType = getattr(core_unified_monitoring_interface_module, 'MetricType', None)
    HealthStatus = getattr(core_unified_monitoring_interface_module, 'HealthStatus', None)
    Metric = getattr(core_unified_monitoring_interface_module, 'Metric', None)
    Alert = getattr(core_unified_monitoring_interface_module, 'Alert', None)
    HealthCheck = getattr(core_unified_monitoring_interface_module, 'HealthCheck', None)
    PerformanceMetrics = getattr(core_unified_monitoring_interface_module, 'PerformanceMetrics', None)
    MonitoringConfig = getattr(core_unified_monitoring_interface_module, 'MonitoringConfig', None)
    IMonitor = getattr(core_unified_monitoring_interface_module, 'IMonitor', None)
    IAlertManager = getattr(core_unified_monitoring_interface_module, 'IAlertManager', None)
    IHealthChecker = getattr(core_unified_monitoring_interface_module, 'IHealthChecker', None)
    IPerformanceMonitor = getattr(core_unified_monitoring_interface_module, 'IPerformanceMonitor', None)
    IMonitoringDashboard = getattr(core_unified_monitoring_interface_module, 'IMonitoringDashboard', None)
    
    IMonitoringSystem = getattr(core_unified_monitoring_interface_module, 'IMonitoringSystem', None)
    
    if MonitorType is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestEnums:
    """测试枚举类"""

    def test_monitor_type_enum(self):
        """测试MonitorType枚举"""
        assert MonitorType.SYSTEM.value == "system"
        assert MonitorType.APPLICATION.value == "application"
        assert MonitorType.BUSINESS.value == "business"
        assert MonitorType.PERFORMANCE.value == "performance"
        assert MonitorType.SECURITY.value == "security"
        assert MonitorType.INFRASTRUCTURE.value == "infrastructure"

    def test_alert_level_enum(self):
        """测试AlertLevel枚举"""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.EMERGENCY.value == "emergency"

    def test_alert_status_enum(self):
        """测试AlertStatus枚举"""
        assert AlertStatus.ACTIVE.value == "active"
        assert AlertStatus.ACKNOWLEDGED.value == "acknowledged"
        assert AlertStatus.RESOLVED.value == "resolved"
        assert AlertStatus.SUPPRESSED.value == "suppressed"

    def test_metric_type_enum(self):
        """测试MetricType枚举"""
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"
        assert MetricType.TIMER.value == "timer"

    def test_health_status_enum(self):
        """测试HealthStatus枚举"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestMetric:
    """测试Metric数据类"""

    def test_metric_creation(self):
        """测试Metric创建"""
        timestamp = datetime.now()
        metric = Metric(
            name="cpu_usage",
            value=75.5,
            timestamp=timestamp,
            labels={"host": "server1"},
            metric_type=MetricType.GAUGE
        )
        
        assert metric.name == "cpu_usage"
        assert metric.value == 75.5
        assert metric.timestamp == timestamp
        assert metric.labels == {"host": "server1"}
        assert metric.metric_type == MetricType.GAUGE
        assert metric.unit is None
        assert metric.description is None

    def test_metric_with_optional_fields(self):
        """测试Metric带可选字段"""
        timestamp = datetime.now()
        metric = Metric(
            name="memory_usage",
            value=60.0,
            timestamp=timestamp,
            labels={},
            metric_type=MetricType.GAUGE,
            unit="percent",
            description="Memory usage percentage"
        )
        
        assert metric.unit == "percent"
        assert metric.description == "Memory usage percentage"


class TestAlert:
    """测试Alert数据类"""

    def test_alert_creation(self):
        """测试Alert创建"""
        now = datetime.now()
        alert = Alert(
            alert_id="alert_001",
            title="High CPU Usage",
            description="CPU usage exceeded threshold",
            level=AlertLevel.WARNING,
            status=AlertStatus.ACTIVE,
            source="monitoring_system",
            component="cpu_monitor",
            created_at=now,
            updated_at=now
        )
        
        assert alert.alert_id == "alert_001"
        assert alert.title == "High CPU Usage"
        assert alert.level == AlertLevel.WARNING
        assert alert.status == AlertStatus.ACTIVE
        assert alert.tags == []
        assert alert.metadata is None

    def test_alert_with_optional_fields(self):
        """测试Alert带可选字段"""
        now = datetime.now()
        alert = Alert(
            alert_id="alert_002",
            title="Critical Error",
            description="System failure",
            level=AlertLevel.CRITICAL,
            status=AlertStatus.RESOLVED,
            source="system",
            component="database",
            created_at=now,
            updated_at=now,
            resolved_at=now,
            acknowledged_at=now,
            acknowledged_by="admin",
            tags=["critical", "database"],
            metadata={"error_code": "DB001"}
        )
        
        assert alert.resolved_at == now
        assert alert.acknowledged_by == "admin"
        assert alert.tags == ["critical", "database"]
        assert alert.metadata == {"error_code": "DB001"}


class TestHealthCheck:
    """测试HealthCheck数据类"""

    def test_health_check_creation(self):
        """测试HealthCheck创建"""
        timestamp = datetime.now()
        health_check = HealthCheck(
            component="database",
            status=HealthStatus.HEALTHY,
            timestamp=timestamp
        )
        
        assert health_check.component == "database"
        assert health_check.status == HealthStatus.HEALTHY
        assert health_check.timestamp == timestamp
        assert health_check.response_time is None
        assert health_check.message is None
        assert health_check.details == {}

    def test_health_check_with_optional_fields(self):
        """测试HealthCheck带可选字段"""
        timestamp = datetime.now()
        health_check = HealthCheck(
            component="api_service",
            status=HealthStatus.DEGRADED,
            timestamp=timestamp,
            response_time=1.5,
            message="Response time increased",
            details={"avg_response_time": 1.5}
        )
        
        assert health_check.response_time == 1.5
        assert health_check.message == "Response time increased"
        assert health_check.details == {"avg_response_time": 1.5}


class TestPerformanceMetrics:
    """测试PerformanceMetrics数据类"""

    def test_performance_metrics_creation(self):
        """测试PerformanceMetrics创建"""
        timestamp = datetime.now()
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            cpu_usage=75.0,
            memory_usage=60.0,
            disk_usage=50.0,
            network_io={"bytes_sent": 1000, "bytes_recv": 2000}
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_usage == 75.0
        assert metrics.memory_usage == 60.0
        assert metrics.disk_usage == 50.0
        assert metrics.network_io == {"bytes_sent": 1000, "bytes_recv": 2000}
        assert metrics.response_time is None
        assert metrics.throughput is None
        assert metrics.error_rate is None
        assert metrics.active_connections is None


class TestMonitoringConfig:
    """测试MonitoringConfig数据类"""

    def test_monitoring_config_creation(self):
        """测试MonitoringConfig创建"""
        config = MonitoringConfig(
            monitor_type=MonitorType.SYSTEM
        )
        
        assert config.monitor_type == MonitorType.SYSTEM
        assert config.enabled is True
        assert config.interval == 60
        assert config.timeout == 30
        assert config.retries == 3
        assert config.alert_thresholds == {}
        assert config.notification_channels == []

    def test_monitoring_config_with_custom_values(self):
        """测试MonitoringConfig自定义值"""
        config = MonitoringConfig(
            monitor_type=MonitorType.PERFORMANCE,
            enabled=False,
            interval=30,
            timeout=15,
            retries=5,
            alert_thresholds={"cpu": 80.0},
            notification_channels=["email", "sms"]
        )
        
        assert config.enabled is False
        assert config.interval == 30
        assert config.timeout == 15
        assert config.retries == 5
        assert config.alert_thresholds == {"cpu": 80.0}
        assert config.notification_channels == ["email", "sms"]


class TestInterfaces:
    """测试接口类"""

    def test_imonitor_is_abstract(self):
        """测试IMonitor是抽象类"""
        assert ABC in IMonitor.__bases__
        
        # 尝试实例化应该失败
        with pytest.raises(TypeError):
            IMonitor()

    def test_ialert_manager_is_abstract(self):
        """测试IAlertManager是抽象类"""
        assert ABC in IAlertManager.__bases__
        
        with pytest.raises(TypeError):
            IAlertManager()

    def test_ihealth_checker_is_abstract(self):
        """测试IHealthChecker是抽象类"""
        assert ABC in IHealthChecker.__bases__
        
        with pytest.raises(TypeError):
            IHealthChecker()

    def test_iperformance_monitor_is_abstract(self):
        """测试IPerformanceMonitor是抽象类"""
        assert ABC in IPerformanceMonitor.__bases__
        
        with pytest.raises(TypeError):
            IPerformanceMonitor()

    def test_imonitoring_dashboard_is_abstract(self):
        """测试IMonitoringDashboard是抽象类"""
        assert ABC in IMonitoringDashboard.__bases__
        
        with pytest.raises(TypeError):
            IMonitoringDashboard()

    def test_imonitoring_system_is_abstract(self):
        """测试IMonitoringSystem是抽象类"""
        assert ABC in IMonitoringSystem.__bases__
        
        with pytest.raises(TypeError):
            IMonitoringSystem()



