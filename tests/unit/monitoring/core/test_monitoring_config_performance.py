#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig性能测试补充
补充simulate_api_performance_test和test_concurrency_performance的完整测试
"""

import sys
import importlib
from pathlib import Path
import pytest
import time
import threading
import secrets
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

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
    core_monitoring_config_module = importlib.import_module('src.monitoring.core.monitoring_config')
    MonitoringSystem = getattr(core_monitoring_config_module, 'MonitoringSystem', None)
    monitoring = getattr(core_monitoring_config_module, 'monitoring', None)
    collect_system_metrics = getattr(core_monitoring_config_module, 'collect_system_metrics', None)
    simulate_api_performance_test = getattr(core_monitoring_config_module, 'simulate_api_performance_test', None)
    test_concurrency_performance = getattr(core_monitoring_config_module, 'test_concurrency_performance', None)
    
    if MonitoringSystem is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestSimulateApiPerformanceTest:
    """测试模拟API性能测试函数"""

    def test_simulate_api_performance_test_normal_response(self):
        """测试正常响应情况"""
        # Mock secrets和time.sleep以避免实际等待
        # secrets是在函数内部导入的，需要mock 'secrets'模块
        with patch('secrets.random', return_value=0.5):
            with patch('secrets.uniform', return_value=0.1):
                with patch('time.sleep'):
                    with patch('builtins.print'):
                        # 重置monitoring实例
                        monitoring.metrics = {}
                        monitoring.traces = []
                        monitoring.alerts = []
                        
                        result = simulate_api_performance_test()
                        
                        assert isinstance(result, dict)
                        assert 'avg_response_time' in result
                        assert 'p95_response_time' in result
                        assert 'total_requests' in result
                        assert result['total_requests'] == 100

    def test_simulate_api_performance_test_slow_response(self):
        """测试慢响应情况（secrets.random() >= 0.9）"""
        call_count = {'count': 0}
        
        def mock_random():
            call_count['count'] += 1
            # 第一次调用返回>=0.9触发慢响应分支
            if call_count['count'] == 1:
                return 0.95
            return 0.5
        
        with patch('secrets.random', side_effect=mock_random):
            with patch('secrets.uniform', return_value=1.0):
                with patch('time.sleep'):
                    with patch('builtins.print'):
                        monitoring.metrics = {}
                        monitoring.traces = []
                        monitoring.alerts = []
                        
                        result = simulate_api_performance_test()
                        
                        assert isinstance(result, dict)
                        assert result['total_requests'] == 100

    def test_simulate_api_performance_test_mixed_responses(self):
        """测试混合响应（正常+慢响应）"""
        random_values = [0.5, 0.95, 0.8, 0.92, 0.3] * 20
        random_iter = iter(random_values)
        
        def mock_random():
            return next(random_iter, 0.5)
        
        with patch('secrets.random', side_effect=mock_random):
            def mock_uniform(a, b):
                # 慢响应范围
                if a == 0.5:
                    return 1.0
                # 正常响应范围
                return 0.1
            
            with patch('secrets.uniform', side_effect=mock_uniform):
                with patch('time.sleep'):
                    with patch('builtins.print'):
                        monitoring.metrics = {}
                        monitoring.traces = []
                        monitoring.alerts = []
                        
                        result = simulate_api_performance_test()
                        
                        assert isinstance(result, dict)
                        assert result['total_requests'] == 100


class TestConcurrencyPerformance:
    """测试并发性能测试函数"""

    def test_test_concurrency_performance_basic(self):
        """测试基本并发性能"""
        with patch('time.sleep'):
            with patch('secrets.uniform', return_value=0.2):
                with patch('builtins.print'):
                    # 重置monitoring实例
                    monitoring.metrics = {}
                    monitoring.traces = []
                    monitoring.alerts = []
                    
                    result = test_concurrency_performance()
                    
                    assert isinstance(result, dict)
                    assert 'concurrent_requests' in result
                    assert 'avg_response_time' in result
                    assert 'max_response_time' in result
                    assert result['concurrent_requests'] == 50

    def test_test_concurrency_performance_worker_execution(self):
        """测试worker函数执行"""
        with patch('time.sleep'):
            with patch('secrets.uniform', return_value=0.1):
                with patch('builtins.print'):
                    monitoring.metrics = {}
                    monitoring.traces = []
                    monitoring.alerts = []
                    
                    result = test_concurrency_performance()
                    
                    # 验证有追踪被创建
                    assert len(monitoring.traces) == 50
                    
                    # 验证所有追踪都已结束
                    for trace in monitoring.traces:
                        assert trace['end_time'] is not None
                        assert trace['duration'] is not None


class TestCollectSystemMetrics:
    """测试收集系统指标函数"""

    def test_collect_system_metrics_with_network(self):
        """测试收集系统指标（包含网络指标）"""
        try:
            with patch('psutil.cpu_percent', return_value=50.0):
                with patch('psutil.virtual_memory') as mock_memory:
                    mock_memory.return_value = type('obj', (object,), {
                        'percent': 60.0
                    })()
                    with patch('psutil.disk_usage') as mock_disk:
                        mock_disk.return_value = type('obj', (object,), {
                            'percent': 70.0
                        })()
                        with patch('psutil.net_io_counters') as mock_network:
                            # Mock网络指标（不为None）
                            mock_network.return_value = type('obj', (object,), {
                                'bytes_sent': 1000000,
                                'bytes_recv': 2000000
                            })()
                            
                            monitoring.metrics = {}
                            result = collect_system_metrics()
                            
                            assert isinstance(result, dict)
                            assert 'cpu_percent' in result
                            assert 'memory_percent' in result
                            assert 'disk_percent' in result
                            
                            # 验证网络指标被记录
                            assert 'network_bytes_sent' in monitoring.metrics
                            assert 'network_bytes_recv' in monitoring.metrics
        except ImportError:
            pytest.skip("psutil not available")

    def test_collect_system_metrics_without_network(self):
        """测试收集系统指标（无网络指标）"""
        try:
            with patch('psutil.cpu_percent', return_value=50.0):
                with patch('psutil.virtual_memory') as mock_memory:
                    mock_memory.return_value = type('obj', (object,), {
                        'percent': 60.0
                    })()
                    with patch('psutil.disk_usage') as mock_disk:
                        mock_disk.return_value = type('obj', (object,), {
                            'percent': 70.0
                        })()
                        with patch('psutil.net_io_counters', return_value=None):
                            monitoring.metrics = {}
                            result = collect_system_metrics()
                            
                            assert isinstance(result, dict)
                            # 网络指标不应该被记录
        except ImportError:
            pytest.skip("psutil not available")


class TestCheckAlertsApiSlow:
    """测试API慢响应告警"""

    def test_check_alerts_api_slow(self):
        """测试API响应时间过慢告警"""
        monitoring_system = MonitoringSystem()
        
        # 记录API响应时间超过1000ms
        monitoring_system.record_metric('api_response_time', 1500.0)
        
        alerts = monitoring_system.check_alerts()
        
        # 应该产生API慢响应告警
        api_slow_alerts = [a for a in alerts if a.get('type') == 'api_slow']
        assert len(api_slow_alerts) > 0
        assert api_slow_alerts[0]['severity'] == 'warning'

    def test_check_alerts_api_normal(self):
        """测试API响应时间正常（不触发告警）"""
        monitoring_system = MonitoringSystem()
        
        # 记录API响应时间小于1000ms
        monitoring_system.record_metric('api_response_time', 500.0)
        
        alerts = monitoring_system.check_alerts()
        
        # 不应该产生API慢响应告警
        api_slow_alerts = [a for a in alerts if a.get('type') == 'api_slow']
        assert len(api_slow_alerts) == 0

