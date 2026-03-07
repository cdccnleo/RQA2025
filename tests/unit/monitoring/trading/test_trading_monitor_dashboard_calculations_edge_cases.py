#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingMonitorDashboard计算方法边界情况测试
补充_calculate_position_summary、_calculate_pnl_analysis、_calculate_order_distribution等方法的边界情况测试
"""

import pytest
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
    from src.monitoring.trading.trading_monitor_dashboard import TradingMonitorDashboard, TradingStatus
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


@pytest.mark.skipif(not FLASK_AVAILABLE, reason="Flask not available")
class TestTradingMonitorDashboardCalculationsEdgeCases:
    """测试计算方法边界情况"""

    @pytest.fixture
    def dashboard(self):
        """创建dashboard实例"""
        return TradingMonitorDashboard({'update_interval': 0.1})

    def test_calculate_order_distribution_empty_orders(self, dashboard):
        """测试空订单时的订单分布"""
        dashboard.current_status.orders = {}
        
        distribution = dashboard._calculate_order_distribution()
        
        assert distribution == {}

    def test_calculate_order_distribution_single_order(self, dashboard):
        """测试单个订单时的订单分布"""
        dashboard.current_status.orders = {'executed': 1}
        
        distribution = dashboard._calculate_order_distribution()
        
        assert distribution == {'executed': 100.0}

    def test_calculate_order_distribution_multiple_orders(self, dashboard):
        """测试多个订单时的订单分布"""
        dashboard.current_status.orders = {
            'executed': 100,
            'pending': 20,
            'cancelled': 10,
            'rejected': 5
        }
        
        distribution = dashboard._calculate_order_distribution()
        
        total = sum(dashboard.current_status.orders.values())
        assert distribution['executed'] == (100 / total) * 100
        assert distribution['pending'] == (20 / total) * 100
        assert abs(sum(distribution.values()) - 100.0) < 0.01  # 总和应该约等于100%

    def test_calculate_execution_stats_empty_orders(self, dashboard):
        """测试空订单时的执行统计"""
        dashboard.current_status.orders = {}
        
        stats = dashboard._calculate_execution_stats()
        
        assert stats == {}

    def test_calculate_execution_stats_zero_total(self, dashboard):
        """测试总订单数为0时的执行统计"""
        dashboard.current_status.orders = {
            'executed': 0,
            'cancelled': 0,
            'rejected': 0
        }
        
        stats = dashboard._calculate_execution_stats()
        
        assert stats == {}

    def test_calculate_execution_stats_all_executed(self, dashboard):
        """测试所有订单都执行成功时的统计"""
        dashboard.current_status.orders = {
            'executed': 100,
            'pending': 0,
            'cancelled': 0,
            'rejected': 0
        }
        
        stats = dashboard._calculate_execution_stats()
        
        assert stats['execution_rate'] == 1.0
        assert stats['cancel_rate'] == 0.0
        assert stats['reject_rate'] == 0.0
        assert stats['success_rate'] == 1.0

    def test_calculate_execution_stats_all_rejected(self, dashboard):
        """测试所有订单都被拒绝时的统计"""
        dashboard.current_status.orders = {
            'executed': 0,
            'pending': 0,
            'cancelled': 0,
            'rejected': 50
        }
        
        stats = dashboard._calculate_execution_stats()
        
        assert stats['execution_rate'] == 0.0
        assert stats['reject_rate'] == 1.0
        assert stats['success_rate'] == 0.0

    def test_calculate_position_summary_empty_positions(self, dashboard):
        """测试空持仓时的持仓汇总"""
        summary = dashboard._calculate_position_summary({})
        
        assert summary['total_positions'] == 0
        assert summary['total_value'] == 0
        assert summary['total_pnl'] == 0

    def test_calculate_position_summary_single_position(self, dashboard):
        """测试单个持仓时的持仓汇总"""
        positions = {
            'AAPL': {
                'size': 100,
                'avg_price': 150.0,
                'current_price': 155.0
            }
        }
        
        summary = dashboard._calculate_position_summary(positions)
        
        assert summary['total_positions'] == 1
        assert summary['total_value'] == 100 * 155.0
        assert summary['total_pnl'] == 100 * (155.0 - 150.0)

    def test_calculate_position_summary_profitable_positions(self, dashboard):
        """测试盈利持仓时的持仓汇总"""
        positions = {
            'AAPL': {'size': 100, 'avg_price': 150.0, 'current_price': 155.0},
            'GOOGL': {'size': 50, 'avg_price': 100.0, 'current_price': 105.0}
        }
        
        summary = dashboard._calculate_position_summary(positions)
        
        assert summary['total_pnl'] > 0

    def test_calculate_position_summary_losing_positions(self, dashboard):
        """测试亏损持仓时的持仓汇总"""
        positions = {
            'AAPL': {'size': 100, 'avg_price': 155.0, 'current_price': 150.0},
            'GOOGL': {'size': 50, 'avg_price': 105.0, 'current_price': 100.0}
        }
        
        summary = dashboard._calculate_position_summary(positions)
        
        assert summary['total_pnl'] < 0

    def test_calculate_position_summary_zero_pnl(self, dashboard):
        """测试盈亏为零时的持仓汇总"""
        positions = {
            'AAPL': {'size': 100, 'avg_price': 150.0, 'current_price': 150.0}
        }
        
        summary = dashboard._calculate_position_summary(positions)
        
        assert summary['total_pnl'] == 0

    def test_calculate_pnl_analysis_empty_positions(self, dashboard):
        """测试空持仓时的盈亏分析"""
        analysis = dashboard._calculate_pnl_analysis({})
        
        assert analysis['profitable_positions'] == 0
        assert analysis['losing_positions'] == 0
        assert analysis['neutral_positions'] == 0
        assert analysis['total_pnl'] == 0
        assert analysis['win_rate'] == 0.0

    def test_calculate_pnl_analysis_all_profitable(self, dashboard):
        """测试所有持仓都盈利时的盈亏分析"""
        positions = {
            'AAPL': {'size': 100, 'avg_price': 150.0, 'current_price': 155.0},
            'GOOGL': {'size': 50, 'avg_price': 100.0, 'current_price': 105.0}
        }
        
        analysis = dashboard._calculate_pnl_analysis(positions)
        
        assert analysis['profitable_positions'] == 2
        assert analysis['losing_positions'] == 0
        assert analysis['win_rate'] == 1.0

    def test_calculate_pnl_analysis_all_losing(self, dashboard):
        """测试所有持仓都亏损时的盈亏分析"""
        positions = {
            'AAPL': {'size': 100, 'avg_price': 155.0, 'current_price': 150.0},
            'GOOGL': {'size': 50, 'avg_price': 105.0, 'current_price': 100.0}
        }
        
        analysis = dashboard._calculate_pnl_analysis(positions)
        
        assert analysis['profitable_positions'] == 0
        assert analysis['losing_positions'] == 2
        assert analysis['win_rate'] == 0.0

    def test_calculate_pnl_analysis_mixed_positions(self, dashboard):
        """测试混合持仓（有盈利有亏损）时的盈亏分析"""
        positions = {
            'AAPL': {'size': 100, 'avg_price': 150.0, 'current_price': 155.0},  # 盈利
            'GOOGL': {'size': 50, 'avg_price': 105.0, 'current_price': 100.0},  # 亏损
            'MSFT': {'size': 200, 'avg_price': 100.0, 'current_price': 100.0}   # 中性
        }
        
        analysis = dashboard._calculate_pnl_analysis(positions)
        
        assert analysis['profitable_positions'] == 1
        assert analysis['losing_positions'] == 1
        assert analysis['neutral_positions'] == 1

    def test_calculate_pnl_analysis_neutral_positions(self, dashboard):
        """测试所有持仓都中性时的盈亏分析"""
        positions = {
            'AAPL': {'size': 100, 'avg_price': 150.0, 'current_price': 150.0},
            'GOOGL': {'size': 50, 'avg_price': 100.0, 'current_price': 100.0}
        }
        
        analysis = dashboard._calculate_pnl_analysis(positions)
        
        assert analysis['profitable_positions'] == 0
        assert analysis['losing_positions'] == 0
        assert analysis['neutral_positions'] == 2

    def test_calculate_position_risk_metrics_empty_positions(self, dashboard):
        """测试空持仓时的风险指标"""
        risk_metrics = dashboard._calculate_position_risk_metrics({})
        
        assert risk_metrics['concentration_risk'] == 0
        assert risk_metrics['current_exposure'] == 0

    def test_calculate_position_risk_metrics_multiple_positions(self, dashboard):
        """测试多个持仓时的风险指标"""
        positions = {
            'AAPL': {'size': 100, 'current_price': 150.0},
            'GOOGL': {'size': 50, 'current_price': 100.0}
        }
        
        risk_metrics = dashboard._calculate_position_risk_metrics(positions)
        
        assert risk_metrics['concentration_risk'] == 2
        expected_exposure = 100 * 150.0 + 50 * 100.0
        assert risk_metrics['current_exposure'] == expected_exposure

    def test_calculate_metrics_summary_empty_history(self, dashboard):
        """测试空历史数据时的指标汇总"""
        summary = dashboard._calculate_metrics_summary([])
        
        assert summary == {}

    def test_calculate_metrics_summary_single_entry(self, dashboard):
        """测试单个历史数据条目时的指标汇总"""
        history = [{
            'metrics': {
                'order_latency': 5.0,
                'order_throughput': 100.0,
                'execution_rate': 0.95
            }
        }]
        
        summary = dashboard._calculate_metrics_summary(history)
        
        assert 'order_latency' in summary
        assert summary['order_latency']['current'] == 5.0
        assert summary['order_latency']['average'] == 5.0

    def test_calculate_metrics_summary_trend_up(self, dashboard):
        """测试上升趋势时的指标汇总"""
        history = [
            {'metrics': {'order_latency': 5.0}},
            {'metrics': {'order_latency': 7.0}},
            {'metrics': {'order_latency': 10.0}}
        ]
        
        summary = dashboard._calculate_metrics_summary(history)
        
        assert summary['order_latency']['trend'] == 'up'

    def test_calculate_metrics_summary_trend_down(self, dashboard):
        """测试下降趋势时的指标汇总"""
        history = [
            {'metrics': {'order_latency': 10.0}},
            {'metrics': {'order_latency': 7.0}},
            {'metrics': {'order_latency': 5.0}}
        ]
        
        summary = dashboard._calculate_metrics_summary(history)
        
        assert summary['order_latency']['trend'] == 'down'

    def test_calculate_metrics_summary_trend_stable(self, dashboard):
        """测试稳定趋势时的指标汇总"""
        history = [
            {'metrics': {'order_latency': 5.0}},
            {'metrics': {'order_latency': 5.0}},
            {'metrics': {'order_latency': 5.0}}
        ]
        
        summary = dashboard._calculate_metrics_summary(history)
        
        assert summary['order_latency']['trend'] == 'stable'

    def test_calculate_metrics_summary_missing_metric(self, dashboard):
        """测试缺少某个指标时的指标汇总"""
        history = [
            {'metrics': {'order_latency': 5.0}}  # 缺少其他指标
        ]
        
        summary = dashboard._calculate_metrics_summary(history)
        
        # 缺少的指标应该使用默认值0
        assert 'order_latency' in summary
        assert summary['order_latency']['current'] == 5.0


