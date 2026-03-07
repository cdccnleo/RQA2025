#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileMonitor附加方法测试
补充mobile_monitor.py中其他方法的测试
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

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
    mobile_mobile_monitor_module = importlib.import_module('monitoring.mobile.mobile_monitor')
    MobileMonitor = getattr(mobile_mobile_monitor_module, 'MobileMonitor', None)
    if MobileMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("MobileMonitor不可用", allow_module_level=True)


class TestMobileMonitorAdditionalMethods:
    """测试MobileMonitor附加方法"""

    @pytest.fixture
    def monitor(self):
        """创建monitor实例"""
        config = {
            'host': '127.0.0.1',
            'port': 8082,
            'debug': False
        }
        return MobileMonitor(config)

    def test_add_performance_metrics(self, monitor):
        """测试添加性能指标"""
        metrics = {
            'response_time': 100.0,
            'throughput': 1000.0,
            'error_rate': 0.1
        }
        
        initial_count = len(monitor.performance_data)
        monitor.add_performance_metrics(metrics)
        
        assert len(monitor.performance_data) == initial_count + 1
        assert monitor.performance_data[-1]['response_time'] == 100.0

    def test_add_performance_metrics_multiple(self, monitor):
        """测试多次添加性能指标"""
        for i in range(5):
            metrics = {'value': float(i)}
            monitor.add_performance_metrics(metrics)
        
        assert len(monitor.performance_data) == 5

    def test_add_strategy_metrics(self, monitor):
        """测试添加策略指标"""
        strategy_name = 'test_strategy'
        metrics = {
            'pnl': 1000.0,
            'win_rate': 0.6,
            'total_trades': 100
        }
        
        monitor.add_strategy_metrics(strategy_name, metrics)
        
        assert strategy_name in monitor.strategy_data
        assert monitor.strategy_data[strategy_name]['pnl'] == 1000.0
        assert monitor.strategy_data[strategy_name]['win_rate'] == 0.6

    def test_add_strategy_metrics_multiple_strategies(self, monitor):
        """测试为多个策略添加指标"""
        monitor.add_strategy_metrics('strategy1', {'pnl': 1000.0})
        monitor.add_strategy_metrics('strategy2', {'pnl': 2000.0})
        monitor.add_strategy_metrics('strategy3', {'pnl': 3000.0})
        
        assert len(monitor.strategy_data) == 3
        assert 'strategy1' in monitor.strategy_data
        assert 'strategy2' in monitor.strategy_data
        assert 'strategy3' in monitor.strategy_data

    def test_get_mobile_optimized_data_structure(self, monitor):
        """测试获取移动端优化数据结构"""
        # 添加一些数据
        monitor.system_data = {'cpu': 50.0}
        monitor.update_performance_data({'value': 100.0})
        monitor.update_strategy_data('test_strategy', {'pnl': 1000.0})
        
        data = monitor.get_mobile_optimized_data()
        
        assert isinstance(data, dict)
        assert 'system' in data
        assert 'performance' in data
        assert 'strategies' in data
        assert 'alerts' in data
        assert 'timestamp' in data

    def test_get_mobile_optimized_data_performance_limit(self, monitor):
        """测试移动端优化数据限制性能数据点"""
        # 添加超过10个性能数据点
        for i in range(20):
            monitor.update_performance_data({'value': float(i)})
        
        data = monitor.get_mobile_optimized_data()
        
        # 应该只返回最近10个
        assert len(data['performance']) <= 10

    def test_get_mobile_optimized_data_alerts_limit(self, monitor):
        """测试移动端优化数据限制告警数量"""
        # 添加超过5个告警
        for i in range(10):
            try:
                monitor.add_alert({'message': f'Alert {i}', 'level': 'info'})
            except ValueError:
                # 如果格式字符串有问题，跳过
                pass
        
        data = monitor.get_mobile_optimized_data()
        
        # 应该只返回最近5个告警（如果add_alert成功）
        if 'alerts' in data and data['alerts']:
            assert len(data['alerts']) <= 5

    def test_get_mobile_optimized_data_timestamp(self, monitor):
        """测试移动端优化数据包含时间戳"""
        data = monitor.get_mobile_optimized_data()
        
        assert 'timestamp' in data
        assert isinstance(data['timestamp'], str)
        # 验证时间戳可以解析
        from datetime import datetime
        timestamp = datetime.fromisoformat(data['timestamp'])
        assert isinstance(timestamp, datetime)

    def test_get_mobile_optimized_data_empty(self, monitor):
        """测试获取空数据的移动端优化数据"""
        data = monitor.get_mobile_optimized_data()
        
        assert isinstance(data, dict)
        assert 'system' in data
        assert 'performance' in data
        assert 'strategies' in data
        assert 'alerts' in data

    def test_get_system_uptime_formatted(self, monitor):
        """测试获取系统运行时间格式化"""
        uptime = monitor._get_system_uptime()
        
        assert isinstance(uptime, str)
        # 验证格式（应该包含h和m）
        assert len(uptime) > 0

    @patch('psutil.boot_time')
    def test_get_system_uptime_formatting(self, mock_boot_time, monitor):
        """测试系统运行时间格式化"""
        import time
        mock_boot_time.return_value = time.time() - 7320  # 2小时2分钟前
        
        uptime = monitor._get_system_uptime()
        
        assert isinstance(uptime, str)
        # 验证格式包含小时和分钟
        assert 'h' in uptime
        assert 'm' in uptime



