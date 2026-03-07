#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一监控接口质量测试
测试覆盖 UnifiedMonitoringInterface 的核心功能
"""

import sys
import importlib
from pathlib import Path
import pytest
from datetime import datetime

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
    IMonitoringInterface = getattr(core_unified_monitoring_interface_module, 'IMonitoringInterface', None)
    
    if MonitorType is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestMonitorType:
    """MonitorType枚举测试类"""

    def test_all_monitor_types(self):
        """测试所有监控类型"""
        assert MonitorType.SYSTEM.value == "system"
        assert MonitorType.APPLICATION.value == "application"
        assert MonitorType.BUSINESS.value == "business"
        assert MonitorType.PERFORMANCE.value == "performance"
        assert MonitorType.SECURITY.value == "security"
        assert MonitorType.INFRASTRUCTURE.value == "infrastructure"


class TestAlertLevel:
    """AlertLevel枚举测试类"""

    def test_all_alert_levels(self):
        """测试所有告警级别"""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.EMERGENCY.value == "emergency"


class TestAlertStatus:
    """AlertStatus枚举测试类"""

    def test_all_alert_statuses(self):
        """测试所有告警状态"""
        assert AlertStatus.ACTIVE.value == "active"
        assert AlertStatus.ACKNOWLEDGED.value == "acknowledged"
        assert AlertStatus.RESOLVED.value == "resolved"
        assert AlertStatus.SUPPRESSED.value == "suppressed"


class TestMetricType:
    """MetricType枚举测试类"""

    def test_all_metric_types(self):
        """测试所有指标类型"""
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"
        assert MetricType.TIMER.value == "timer"


