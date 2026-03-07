#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingMonitor摘要和工具方法测试
补充get_performance_summary、get_strategy_summary、get_risk_summary等方法
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from collections import deque

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
    trading_trading_monitor_module = importlib.import_module('src.monitoring.trading.trading_monitor')
    TradingMonitor = getattr(trading_trading_monitor_module, 'TradingMonitor', None)
    PerformanceMetrics = getattr(trading_trading_monitor_module, 'PerformanceMetrics', None)
    StrategyMetrics = getattr(trading_trading_monitor_module, 'StrategyMetrics', None)
    RiskMetrics = getattr(trading_trading_monitor_module, 'RiskMetrics', None)
    Alert = getattr(trading_trading_monitor_module, 'Alert', None)

    if TradingMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestTradingMonitorSummaryMethods:
    """测试TradingMonitor摘要方法"""

    @pytest.fixture
    def monitor(self):
        """创建监控实例"""
        return TradingMonitor()

    def test_get_performance_summary_empty(self, monitor):
        """测试空性能摘要"""
        summary = monitor.get_performance_summary()
        assert summary == {}

    def test_get_performance_summary_with_data(self, monitor):
        """测试有数据的性能摘要"""
        # 添加性能指标
        for i in range(15):
            metrics = PerformanceMetrics(
                timestamp=datetime.now() - timedelta(seconds=i),
                cpu_usage=50.0 + i,
                memory_usage=60.0 + i,
                gpu_usage=None,
                disk_usage=70.0,
                network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
                response_time=1.0 + i * 0.1
            )
            monitor.performance_history.append(metrics)
        
        summary = monitor.get_performance_summary()
        
        assert isinstance(summary, dict)
        assert 'cpu_usage_avg' in summary
        assert 'memory_usage_avg' in summary
        assert 'response_time_avg' in summary
        assert 'data_points' in summary
        assert summary['data_points'] == 15

    def test_get_strategy_summary_empty(self, monitor):
        """测试空策略摘要"""
        summary = monitor.get_strategy_summary()
        assert summary == {}

    def test_get_strategy_summary_with_data(self, monitor):
        """测试有数据的策略摘要"""
        strategy_name = "test_strategy"
        strategy_history = deque(maxlen=1000)
        
        # 添加策略指标
        for i in range(15):
            metrics = StrategyMetrics(
                strategy_name=strategy_name,
                timestamp=datetime.now() - timedelta(seconds=i),
                total_signals=10 + i,
                profitable_signals=7 + i,
                win_rate=0.6 + i * 0.01,
                total_pnl=100.0 + i * 10,
                sharpe_ratio=1.5 + i * 0.1,
                max_drawdown=0.1,
                total_trades=10 + i
            )
            strategy_history.append(metrics)
        
        monitor.strategy_history[strategy_name] = strategy_history
        
        summary = monitor.get_strategy_summary()
        
        assert isinstance(summary, dict)
        assert strategy_name in summary
        assert 'win_rate_avg' in summary[strategy_name]
        assert 'total_pnl' in summary[strategy_name]
        assert 'sharpe_ratio_avg' in summary[strategy_name]
        assert 'max_drawdown_max' in summary[strategy_name]
        assert 'data_points' in summary[strategy_name]

    def test_get_strategy_summary_multiple_strategies(self, monitor):
        """测试多个策略的摘要"""
        # 添加第一个策略
        strategy1 = "strategy_1"
        history1 = deque(maxlen=1000)
        for i in range(5):
            metrics = StrategyMetrics(
                strategy_name=strategy1,
                timestamp=datetime.now(),
                total_signals=10,
                profitable_signals=7,
                win_rate=0.7,
                total_pnl=100.0,
                sharpe_ratio=1.5,
                max_drawdown=0.1,
                total_trades=10
            )
            history1.append(metrics)
        monitor.strategy_history[strategy1] = history1
        
        # 添加第二个策略
        strategy2 = "strategy_2"
        history2 = deque(maxlen=1000)
        for i in range(5):
            metrics = StrategyMetrics(
                strategy_name=strategy2,
                timestamp=datetime.now(),
                total_signals=8,
                profitable_signals=5,
                win_rate=0.625,
                total_pnl=80.0,
                sharpe_ratio=1.2,
                max_drawdown=0.08,
                total_trades=8
            )
            history2.append(metrics)
        monitor.strategy_history[strategy2] = history2
        
        summary = monitor.get_strategy_summary()
        
        assert len(summary) == 2
        assert strategy1 in summary
        assert strategy2 in summary

    def test_get_strategy_summary_empty_history(self, monitor):
        """测试策略历史为空的情况"""
        strategy_name = "empty_strategy"
        monitor.strategy_history[strategy_name] = deque(maxlen=1000)
        
        summary = monitor.get_strategy_summary()
        
        # 空历史不应该出现在摘要中
        assert strategy_name not in summary

    def test_get_risk_summary_empty(self, monitor):
        """测试空风险摘要"""
        summary = monitor.get_risk_summary()
        assert summary == {}

    def test_get_risk_summary_with_data(self, monitor):
        """测试有数据的风险摘要"""
        # 添加风险指标
        for i in range(15):
            metrics = RiskMetrics(
                timestamp=datetime.now() - timedelta(seconds=i),
                portfolio_value=100000.0 + i * 1000,
                position_value=50000.0 + i * 500,
                total_exposure=45000.0 + i * 450,
                margin_usage=0.5 + i * 0.01,
                var_95=5000.0 + i * 100,
                concentration_ratio=0.3 + i * 0.01,
                leverage_ratio=1.5 + i * 0.1
            )
            monitor.risk_history.append(metrics)
        
        summary = monitor.get_risk_summary()
        
        assert isinstance(summary, dict)
        assert 'portfolio_value_avg' in summary
        assert 'margin_usage_avg' in summary
        assert 'concentration_ratio_avg' in summary
        assert 'var_95_avg' in summary
        assert 'data_points' in summary
        assert summary['data_points'] == 15

    def test_get_alert_summary_empty(self, monitor):
        """测试空告警摘要"""
        summary = monitor.get_alert_summary()
        assert summary == {}

    def test_get_alert_summary_with_alerts(self, monitor):
        """测试有告警的摘要"""
        # 创建告警
        alert1 = Alert(
            alert_id="alert1",
            monitor_type=MonitorType.PERFORMANCE,
            level=AlertLevel.WARNING,
            message="Test alert 1",
            details={},
            timestamp=datetime.now()
        )
        
        alert2 = Alert(
            alert_id="alert2",
            monitor_type=MonitorType.PERFORMANCE,
            level=AlertLevel.CRITICAL,
            message="Test alert 2",
            details={},
            timestamp=datetime.now()
        )
        
        alert3 = Alert(
            alert_id="alert3",
            monitor_type=MonitorType.STRATEGY,
            level=AlertLevel.WARNING,
            message="Test alert 3",
            details={},
            timestamp=datetime.now()
        )
        
        monitor.alerts = [alert1, alert2, alert3]
        
        summary = monitor.get_alert_summary()
        
        assert isinstance(summary, dict)
        assert 'performance_warning' in summary or f"{MonitorType.PERFORMANCE.value}_{AlertLevel.WARNING.value}" in summary
        assert summary.get('performance_warning', summary.get(f"{MonitorType.PERFORMANCE.value}_{AlertLevel.WARNING.value}", 0)) >= 1

    def test_cleanup_old_data(self, monitor):
        """测试清理过期数据"""
        # 设置保留时间为1小时
        monitor.metrics_retention = 3600
        
        # 添加新告警和旧告警
        old_alert = Alert(
            alert_id="old_alert",
            monitor_type=MonitorType.PERFORMANCE,
            level=AlertLevel.WARNING,
            message="Old alert",
            details={},
            timestamp=datetime.now() - timedelta(hours=2)  # 2小时前
        )
        
        new_alert = Alert(
            alert_id="new_alert",
            monitor_type=MonitorType.PERFORMANCE,
            level=AlertLevel.WARNING,
            message="New alert",
            details={},
            timestamp=datetime.now() - timedelta(minutes=30)  # 30分钟前
        )
        
        monitor.alerts = [old_alert, new_alert]
        
        monitor._cleanup_old_data()
        
        # 旧告警应该被清理，新告警应该保留
        assert len(monitor.alerts) == 1
        assert monitor.alerts[0].alert_id == "new_alert"

    def test_process_alerts(self, monitor):
        """测试处理告警"""
        # _process_alerts目前是空实现，只需验证可以调用
        monitor._process_alerts()
        assert True  # 方法可以正常调用

    def test_set_alert_callback(self, monitor):
        """测试设置告警回调"""
        def callback(alert):
            pass
        
        monitor.set_alert_callback(callback)
        
        assert monitor.on_alert == callback

    def test_get_all_alerts(self, monitor):
        """测试获取所有告警"""
        alert1 = Alert(
            alert_id="alert1",
            monitor_type=MonitorType.PERFORMANCE,
            level=AlertLevel.WARNING,
            message="Alert 1",
            details={},
            timestamp=datetime.now(),
            resolved=False
        )
        
        alert2 = Alert(
            alert_id="alert2",
            monitor_type=MonitorType.PERFORMANCE,
            level=AlertLevel.WARNING,
            message="Alert 2",
            details={},
            timestamp=datetime.now(),
            resolved=True
        )
        
        monitor.alerts = [alert1, alert2]
        
        # 获取所有告警
        all_alerts = monitor.get_all_alerts()
        assert len(all_alerts) == 2
        
        # 获取未解决的告警
        unresolved = monitor.get_all_alerts(resolved=False)
        assert len(unresolved) == 1
        assert unresolved[0].alert_id == "alert1"
        
        # 获取已解决的告警
        resolved = monitor.get_all_alerts(resolved=True)
        assert len(resolved) == 1
        assert resolved[0].alert_id == "alert2"

    def test_resolve_alert(self, monitor):
        """测试解决告警"""
        alert = Alert(
            alert_id="test_alert",
            monitor_type=MonitorType.PERFORMANCE,
            level=AlertLevel.WARNING,
            message="Test alert",
            details={},
            timestamp=datetime.now(),
            resolved=False
        )
        
        monitor.alerts = [alert]
        
        monitor.resolve_alert("test_alert")
        
        assert alert.resolved == True
        assert alert.resolved_time is not None

    def test_resolve_alert_not_found(self, monitor):
        """测试解决不存在的告警"""
        monitor.alerts = []
        
        # 应该不抛出异常
        monitor.resolve_alert("nonexistent_alert")
        assert True

    def test_resolve_alert_already_resolved(self, monitor):
        """测试解决已解决的告警"""
        alert = Alert(
            alert_id="test_alert",
            monitor_type=MonitorType.PERFORMANCE,
            level=AlertLevel.WARNING,
            message="Test alert",
            details={},
            timestamp=datetime.now(),
            resolved=True,
            resolved_time=datetime.now()
        )
        
        monitor.alerts = [alert]
        original_resolved_time = alert.resolved_time
        
        monitor.resolve_alert("test_alert")
        
        # 已解决的告警不应该再次更新
        assert alert.resolved_time == original_resolved_time

