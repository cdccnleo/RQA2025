#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingMonitorDashboard告警功能测试
补充告警相关方法的测试覆盖率
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

try:
    from flask import Flask
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

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
    trading_trading_monitor_dashboard_module = importlib.import_module('src.monitoring.trading.trading_monitor_dashboard')

    # 从导入的模块中获取需要的类
    TradingMonitorDashboard = getattr(trading_trading_monitor_dashboard_module, 'TradingMonitorDashboard', None)
    TradingStatus = getattr(trading_trading_monitor_dashboard_module, 'TradingStatus', None)

    if TradingMonitorDashboard is None or TradingStatus is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


@pytest.mark.skipif(not FLASK_AVAILABLE, reason="Flask not available")
class TestTradingMonitorDashboardAlerts:
    """测试告警相关功能"""

    @pytest.fixture
    def dashboard(self):
        """创建dashboard实例"""
        dashboard = TradingMonitorDashboard({'update_interval': 0.1})
        # 添加一些告警
        dashboard.current_status.alerts = [
            {
                'id': 'alert_1',
                'level': 'error',
                'message': 'Critical alert',
                'timestamp': datetime.now()
            },
            {
                'id': 'alert_2',
                'level': 'warning',
                'message': 'Warning alert',
                'timestamp': datetime.now()
            },
            {
                'id': 'alert_3',
                'level': 'info',
                'message': 'Info alert',
                'timestamp': datetime.now()
            }
        ]
        return dashboard

    def test_get_trading_alerts_data(self, dashboard):
        """测试获取交易告警数据"""
        alerts_data = dashboard._get_trading_alerts_data()
        
        assert isinstance(alerts_data, dict)
        assert 'active_alerts' in alerts_data
        assert 'alert_summary' in alerts_data
        assert 'alert_trends' in alerts_data

    def test_calculate_alert_summary(self, dashboard):
        """测试计算告警汇总"""
        alerts = dashboard.current_status.alerts
        summary = dashboard._calculate_alert_summary(alerts)
        
        assert isinstance(summary, dict)
        assert 'total_alerts' in summary
        assert 'error_count' in summary
        assert 'warning_count' in summary

    def test_calculate_alert_summary_empty(self, dashboard):
        """测试空告警列表的汇总"""
        dashboard.current_status.alerts = []
        summary = dashboard._calculate_alert_summary([])
        
        assert isinstance(summary, dict)
        assert summary['total_alerts'] == 0

    def test_calculate_alert_summary_only_errors(self, dashboard):
        """测试只有错误告警的汇总"""
        dashboard.current_status.alerts = [
            {'id': 'alert_1', 'level': 'error', 'message': 'Error 1'},
            {'id': 'alert_2', 'level': 'error', 'message': 'Error 2'}
        ]
        summary = dashboard._calculate_alert_summary(dashboard.current_status.alerts)
        
        assert summary['error_count'] == 2
        assert summary['warning_count'] == 0
        assert summary['alert_severity'] == 'high'

    def test_calculate_alert_summary_only_warnings(self, dashboard):
        """测试只有警告告警的汇总"""
        dashboard.current_status.alerts = [
            {'id': 'alert_1', 'level': 'warning', 'message': 'Warning 1'},
            {'id': 'alert_2', 'level': 'warning', 'message': 'Warning 2'}
        ]
        summary = dashboard._calculate_alert_summary(dashboard.current_status.alerts)
        
        assert summary['error_count'] == 0
        assert summary['warning_count'] == 2
        assert summary['alert_severity'] == 'medium'

    def test_calculate_alert_trends(self, dashboard):
        """测试计算告警趋势"""
        trends = dashboard._calculate_alert_trends()
        
        assert isinstance(trends, dict)
        assert 'alert_trend' in trends
        assert 'alert_frequency' in trends

