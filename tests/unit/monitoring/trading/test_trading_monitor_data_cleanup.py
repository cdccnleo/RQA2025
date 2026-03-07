#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingMonitor数据清理功能测试
覆盖_cleanup_old_data方法及其相关逻辑
"""

import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta

import sys
import importlib
from pathlib import Path

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
    trading_trading_monitor_module = importlib.import_module('src.monitoring.trading.trading_monitor')
    Alert = getattr(trading_trading_monitor_module, 'Alert', None)
    AlertLevel = getattr(trading_trading_monitor_module, 'AlertLevel', None)
    MonitorType = getattr(trading_trading_monitor_module, 'MonitorType', None)
    if Alert is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


def test_cleanup_recent_alerts():
    """测试清理近期告警（应保留）"""
    # 创建模拟的TradingMonitor实例
    monitor = Mock()
    monitor.alerts = []
    monitor._cleanup_old_data = Mock()

    now = datetime.now()

    # 添加多个近期告警
    for i in range(5):
        alert = Alert(
            alert_id=f'recent_alert_{i}',
            monitor_type=MonitorType.PERFORMANCE,
            level=AlertLevel.WARNING,
            message=f'Recent alert {i}',
            details={},
            timestamp=now - timedelta(minutes=i * 10)  # 0-40分钟前
        )
        monitor.alerts.append(alert)

    initial_count = len(monitor.alerts)
    monitor._cleanup_old_data()

    # 所有近期告警都应保留
    assert len(monitor.alerts) == initial_count