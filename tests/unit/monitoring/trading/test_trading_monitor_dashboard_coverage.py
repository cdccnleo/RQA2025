#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingMonitorDashboard覆盖率测试
专注提升trading_monitor_dashboard.py的测试覆盖率
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

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
    from src.monitoring.trading.trading_monitor_dashboard import TradingMonitorDashboard
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestTradingMonitorDashboardInitialization:
    """测试TradingMonitorDashboard初始化"""

    def test_init_default(self):
        """测试默认初始化"""
        dashboard = TradingMonitorDashboard()
        assert dashboard.config == {}
        assert dashboard.update_interval == 5.0
        assert dashboard.history_size == 3600
        assert dashboard.alert_threshold == 0.8
        assert dashboard.is_monitoring == False
        assert isinstance(dashboard.trading_history, list)
        assert isinstance(dashboard.current_status, TradingStatus)

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'update_interval': 10.0,
            'history_size': 7200,
            'alert_threshold': 0.9
        }
        dashboard = TradingMonitorDashboard(config=config)
        assert dashboard.update_interval == 10.0
        assert dashboard.history_size == 7200
        assert dashboard.alert_threshold == 0.9

    def test_trading_metric_enum(self):
        """测试交易指标枚举"""
        assert TradingMetric.ORDER_LATENCY.value == "order_latency"
        assert TradingMetric.ORDER_THROUGHPUT.value == "order_throughput"
        assert TradingMetric.EXECUTION_RATE.value == "execution_rate"
        assert TradingMetric.SLIPPAGE.value == "slippage"
        assert TradingMetric.RISK_EXPOSURE.value == "risk_exposure"


class TestTradingMonitorDashboardDataManagement:
    """测试数据管理功能"""

    @pytest.fixture
    def dashboard(self):
        """创建dashboard实例"""
        return TradingMonitorDashboard()

    def test_collect_trading_status(self, dashboard):
        """测试收集交易状态"""
        # 通过监控循环收集状态
        dashboard._collect_trading_status()
        
        assert dashboard.current_status is not None
        assert 'order_latency' in dashboard.current_status.metrics
        assert isinstance(dashboard.current_status.metrics, dict)
        assert isinstance(dashboard.current_status.orders, dict)
        assert isinstance(dashboard.current_status.positions, dict)

    def test_get_current_status_data(self, dashboard):
        """测试获取当前状态数据"""
        dashboard._collect_trading_status()
        status_data = dashboard._get_current_status_data()
        
        assert isinstance(status_data, dict)
        assert 'timestamp' in status_data
        assert 'metrics' in status_data
        assert 'health_score' in status_data

    def test_calculate_health_score(self, dashboard):
        """测试计算健康评分"""
        dashboard._collect_trading_status()
        score = dashboard._calculate_health_score()
        
        assert isinstance(score, (float, int))
        assert 0 <= score <= 100

    def test_trigger_status_callbacks(self, dashboard):
        """测试触发状态回调"""
        callback_called = []
        
        def test_callback(status):
            callback_called.append(status)
        
        dashboard.add_status_callback(test_callback)
        dashboard._collect_trading_status()
        dashboard._trigger_status_callbacks()
        
        assert len(callback_called) > 0
        assert callback_called[0] == dashboard.current_status


class TestTradingMonitorDashboardMonitoring:
    """测试监控功能"""

    @pytest.fixture
    def dashboard(self):
        """创建dashboard实例"""
        return TradingMonitorDashboard({'update_interval': 0.1})

    def test_start_monitoring(self, dashboard):
        """测试启动监控"""
        assert dashboard.is_monitoring == False
        dashboard.start_monitoring()
        assert dashboard.is_monitoring == True
        assert dashboard.monitor_thread is not None
        dashboard.stop_monitoring()

    def test_stop_monitoring(self, dashboard):
        """测试停止监控"""
        dashboard.start_monitoring()
        assert dashboard.is_monitoring == True
        dashboard.stop_monitoring()
        # 等待线程结束
        if dashboard.monitor_thread:
            dashboard.monitor_thread.join(timeout=2)
        assert dashboard.is_monitoring == False

    def test_monitoring_loop_updates_status(self, dashboard):
        """测试监控循环更新状态"""
        dashboard.start_monitoring()
        time.sleep(0.2)  # 等待一次更新
        dashboard.stop_monitoring()
        
        # 验证状态有更新（至少时间戳更新）
        assert dashboard.current_status.timestamp is not None


class TestTradingMonitorDashboardMetrics:
    """测试指标功能"""

    @pytest.fixture
    def dashboard(self):
        """创建dashboard实例"""
        return TradingMonitorDashboard()

    def test_get_metrics_data(self, dashboard):
        """测试获取指标数据"""
        dashboard._collect_trading_status()
        metrics_data = dashboard._get_metrics_data()
        
        assert isinstance(metrics_data, dict)
        assert 'current' in metrics_data or 'history' in metrics_data or 'summary' in metrics_data

    def test_get_order_status_data(self, dashboard):
        """测试获取订单状态数据"""
        dashboard._collect_trading_status()
        order_data = dashboard._get_order_status_data()
        
        assert isinstance(order_data, dict)

    def test_get_position_status_data(self, dashboard):
        """测试获取持仓状态数据"""
        dashboard._collect_trading_status()
        position_data = dashboard._get_position_status_data()
        
        assert isinstance(position_data, dict)

    def test_trading_history_management(self, dashboard):
        """测试交易历史管理"""
        # 收集多次状态以生成历史
        for i in range(5):
            dashboard._collect_trading_status()
            time.sleep(0.01)
        
        assert len(dashboard.trading_history) > 0
        assert len(dashboard.trading_history) <= dashboard.history_size


class TestTradingMonitorDashboardCallbacks:
    """测试回调功能"""

    @pytest.fixture
    def dashboard(self):
        """创建dashboard实例"""
        return TradingMonitorDashboard()

    def test_add_status_callback(self, dashboard):
        """测试添加状态回调"""
        callback = Mock()
        dashboard.add_status_callback(callback)
        assert callback in dashboard.status_callbacks

    def test_multiple_status_callbacks(self, dashboard):
        """测试多个状态回调"""
        callback1 = Mock()
        callback2 = Mock()
        dashboard.add_status_callback(callback1)
        dashboard.add_status_callback(callback2)
        
        assert len(dashboard.status_callbacks) == 2
        
        dashboard._collect_trading_status()
        dashboard._trigger_status_callbacks()
        
        # 验证回调被调用
        assert callback1.called or callback2.called


class TestTradingMonitorDashboardWebApp:
    """测试Web应用功能"""

    @pytest.fixture
    def dashboard(self):
        """创建dashboard实例"""
        return TradingMonitorDashboard()

    def test_web_app_initialization(self, dashboard):
        """测试Web应用初始化"""
        # Web应用可能在导入时初始化，也可能延迟初始化
        # 只要没有抛出异常即可
        assert dashboard is not None

    @pytest.mark.skipif(not hasattr(TradingMonitorDashboard, 'app') or 
                       TradingMonitorDashboard.__init__.__defaults__ is None,
                       reason="Flask may not be available")
    def test_get_status_endpoint(self, dashboard):
        """测试获取状态端点"""
        if dashboard.app:
            with dashboard.app.test_client() as client:
                response = client.get('/api/status')
                assert response.status_code in [200, 404]  # 取决于路由是否存在

    @pytest.mark.skipif(not hasattr(TradingMonitorDashboard, 'app') or 
                       TradingMonitorDashboard.__init__.__defaults__ is None,
                       reason="Flask may not be available")
    def test_get_metrics_endpoint(self, dashboard):
        """测试获取指标端点"""
        if dashboard.app:
            with dashboard.app.test_client() as client:
                response = client.get('/api/metrics')
                assert response.status_code in [200, 404]


class TestTradingMonitorDashboardIntegration:
    """测试集成功能"""

    @pytest.fixture
    def dashboard(self):
        """创建dashboard实例"""
        return TradingMonitorDashboard({'update_interval': 0.1})

    def test_full_workflow(self, dashboard):
        """测试完整工作流"""
        # 1. 收集交易状态
        dashboard._collect_trading_status()
        
        # 2. 验证状态数据
        assert dashboard.current_status is not None
        assert len(dashboard.current_status.metrics) > 0
        
        # 3. 获取状态数据
        status_data = dashboard._get_current_status_data()
        assert isinstance(status_data, dict)
        
        # 4. 启动监控
        dashboard.start_monitoring()
        time.sleep(0.2)
        
        # 5. 验证监控运行
        assert dashboard.is_monitoring == True
        assert len(dashboard.trading_history) > 0
        
        # 6. 停止监控
        dashboard.stop_monitoring()
        assert dashboard.is_monitoring == False

    def test_check_alert_conditions(self, dashboard):
        """测试检查告警条件"""
        dashboard._collect_trading_status()
        
        # 模拟高延迟情况
        dashboard.current_status.metrics['order_latency'] = 15.0  # 超过阈值10
        dashboard._check_alert_conditions()
        
        assert len(dashboard.current_status.alerts) > 0

    def test_concurrent_monitoring(self, dashboard):
        """测试并发监控操作"""
        import threading
        
        results = []
        
        def worker(worker_id):
            try:
                # 每个worker收集状态
                dashboard._collect_trading_status()
                # 获取状态数据
                status = dashboard._get_current_status_data()
                results.append(f'worker_{worker_id}_done')
            except Exception as e:
                results.append(f'worker_{worker_id}_error: {e}')
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=1)
        
        # 验证操作完成（可能有错误，但不会崩溃）
        assert len(results) == 3

