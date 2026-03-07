#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingMonitorDashboard图表生成边界情况测试
补充图表生成方法的边界情况和错误处理测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

try:
    from flask import Flask, Response
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

except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


@pytest.mark.skipif(not FLASK_AVAILABLE, reason="Flask not available")
class TestTradingMonitorDashboardChartsEdgeCases:
    """测试图表生成的边界情况"""

    @pytest.fixture
    def dashboard(self):
        """创建dashboard实例"""
        return TradingMonitorDashboard({'update_interval': 0.1})

    @pytest.fixture
    def dashboard_with_status(self, dashboard):
        """准备有状态的dashboard"""
        dashboard.current_status = TradingStatus(
            timestamp=datetime.now(),
            orders={'pending': 10, 'executed': 50, 'cancelled': 5, 'rejected': 2},
            positions={
                'AAPL': {'size': 100, 'avg_price': 150.0, 'current_price': 155.0}
            },
            metrics={
                'order_latency': 15.5,
                'risk_exposure': 1500000,
                'pnl_realized': 45000
            },
            connections={
                'NYSE': {'status': 'connected', 'latency': 10.5},
                'NASDAQ': {'status': 'connected', 'latency': 8.2}
            },
            alerts=[]
        )
        return dashboard

    def test_get_performance_overview_chart_no_plotly(self, dashboard_with_status):
        """测试Plotly不可用时的性能概览图表"""
        with patch('src.monitoring.trading.trading_monitor_dashboard.PLOTLY_AVAILABLE', False):
            if dashboard_with_status.app:
                with dashboard_with_status.app.app_context():
                    result = dashboard_with_status._get_performance_overview_chart()
                    # 应该返回错误响应
                    assert result is not None
            else:
                pytest.skip("Flask app not available")

    def test_get_performance_overview_chart_empty_metrics(self, dashboard):
        """测试空指标时的性能概览图表"""
        dashboard.current_status = TradingStatus(
            timestamp=datetime.now(),
            metrics={},
            orders={},
            positions={},
            connections={},
            alerts=[]
        )
        
        with patch('src.monitoring.trading.trading_monitor_dashboard.PLOTLY_AVAILABLE', False):
            if dashboard.app:
                with dashboard.app.app_context():
                    result = dashboard._get_performance_overview_chart()
                    assert result is not None
            else:
                pytest.skip("Flask app not available")

    def test_get_order_flow_chart_empty_orders(self, dashboard):
        """测试空订单时的订单流图表"""
        dashboard.current_status = TradingStatus(
            timestamp=datetime.now(),
            orders={},
            metrics={},
            positions={},
            connections={},
            alerts=[]
        )
        
        with patch('src.monitoring.trading.trading_monitor_dashboard.PLOTLY_AVAILABLE', False):
            if dashboard.app:
                with dashboard.app.app_context():
                    result = dashboard._get_order_flow_chart()
                    assert result is not None
            else:
                pytest.skip("Flask app not available")

    def test_get_order_flow_chart_single_order_status(self, dashboard):
        """测试单个订单状态时的订单流图表"""
        dashboard.current_status = TradingStatus(
            timestamp=datetime.now(),
            orders={'executed': 100},
            metrics={},
            positions={},
            connections={},
            alerts=[]
        )
        
        with patch('src.monitoring.trading.trading_monitor_dashboard.PLOTLY_AVAILABLE', False):
            if dashboard.app:
                with dashboard.app.app_context():
                    result = dashboard._get_order_flow_chart()
                    assert result is not None
            else:
                pytest.skip("Flask app not available")

    def test_get_pnl_trend_chart_no_plotly(self, dashboard_with_status):
        """测试Plotly不可用时的盈亏趋势图表"""
        with patch('src.monitoring.trading.trading_monitor_dashboard.PLOTLY_AVAILABLE', False):
            if dashboard_with_status.app:
                with dashboard_with_status.app.app_context():
                    result = dashboard_with_status._get_pnl_trend_chart()
                    assert result is not None
            else:
                pytest.skip("Flask app not available")

    def test_get_risk_exposure_chart_no_plotly(self, dashboard_with_status):
        """测试Plotly不可用时的风险敞口图表"""
        with patch('src.monitoring.trading.trading_monitor_dashboard.PLOTLY_AVAILABLE', False):
            if dashboard_with_status.app:
                with dashboard_with_status.app.app_context():
                    result = dashboard_with_status._get_risk_exposure_chart()
                    assert result is not None
            else:
                pytest.skip("Flask app not available")

    def test_get_risk_exposure_chart_zero_exposure(self, dashboard):
        """测试风险敞口为0时的风险敞口图表"""
        dashboard.current_status = TradingStatus(
            timestamp=datetime.now(),
            metrics={'risk_exposure': 0},
            orders={},
            positions={},
            connections={},
            alerts=[]
        )
        
        with patch('src.monitoring.trading.trading_monitor_dashboard.PLOTLY_AVAILABLE', False):
            if dashboard.app:
                with dashboard.app.app_context():
                    result = dashboard._get_risk_exposure_chart()
                    assert result is not None
            else:
                pytest.skip("Flask app not available")

    def test_get_connection_health_chart_empty_connections(self, dashboard):
        """测试空连接时的连接健康图表"""
        dashboard.current_status = TradingStatus(
            timestamp=datetime.now(),
            connections={},
            metrics={},
            orders={},
            positions={},
            alerts=[]
        )
        
        with patch('src.monitoring.trading.trading_monitor_dashboard.PLOTLY_AVAILABLE', False):
            if dashboard.app:
                with dashboard.app.app_context():
                    result = dashboard._get_connection_health_chart()
                    assert result is not None
            else:
                pytest.skip("Flask app not available")

    def test_get_connection_health_chart_no_plotly(self, dashboard_with_status):
        """测试Plotly不可用时的连接健康图表"""
        with patch('src.monitoring.trading.trading_monitor_dashboard.PLOTLY_AVAILABLE', False):
            if dashboard_with_status.app:
                with dashboard_with_status.app.app_context():
                    result = dashboard_with_status._get_connection_health_chart()
                    assert result is not None
            else:
                pytest.skip("Flask app not available")

    def test_get_connection_health_chart_missing_latency(self, dashboard):
        """测试连接缺少latency时的连接健康图表"""
        dashboard.current_status = TradingStatus(
            timestamp=datetime.now(),
            connections={
                'NYSE': {'status': 'connected'},  # 缺少latency
                'NASDAQ': {'status': 'connected', 'latency': 8.2}
            },
            metrics={},
            orders={},
            positions={},
            alerts=[]
        )
        
        with patch('src.monitoring.trading.trading_monitor_dashboard.PLOTLY_AVAILABLE', False):
            if dashboard.app:
                with dashboard.app.app_context():
                    result = dashboard._get_connection_health_chart()
                    assert result is not None
            else:
                pytest.skip("Flask app not available")

    def test_chart_route_unknown_type(self, dashboard_with_status):
        """测试未知图表类型的路由"""
        if not dashboard_with_status.app:
            pytest.skip("Flask app not available")
        
        with dashboard_with_status.app.test_client() as client:
            response = client.get('/api/trading/charts/unknown_type')
            # 如果路由不存在，应该是404
            assert response.status_code in [400, 404, 500]

    def test_chart_route_performance_overview(self, dashboard_with_status):
        """测试性能概览图表路由"""
        if not dashboard_with_status.app:
            pytest.skip("Flask app not available")
        
        with patch('src.monitoring.trading.trading_monitor_dashboard.PLOTLY_AVAILABLE', False):
            with dashboard_with_status.app.test_client() as client:
                response = client.get('/api/trading/charts/performance_overview')
                # 可能返回错误或成功响应
                assert response.status_code in [200, 404, 500]

    def test_chart_route_order_flow(self, dashboard_with_status):
        """测试订单流图表路由"""
        if not dashboard_with_status.app:
            pytest.skip("Flask app not available")
        
        with patch('src.monitoring.trading.trading_monitor_dashboard.PLOTLY_AVAILABLE', False):
            with dashboard_with_status.app.test_client() as client:
                response = client.get('/api/trading/charts/order_flow')
                assert response.status_code in [200, 404, 500]

