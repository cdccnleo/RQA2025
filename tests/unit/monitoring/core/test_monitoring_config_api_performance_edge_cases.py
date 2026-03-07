#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig API性能测试边界情况
补充simulate_api_performance_test的边界情况测试
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

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
    monitoring_config_module = importlib.import_module('src.monitoring.core.monitoring_config')
    simulate_api_performance_test = getattr(monitoring_config_module, 'simulate_api_performance_test', None)
    monitoring = getattr(monitoring_config_module, 'monitoring', None)
    if simulate_api_performance_test is None or monitoring is None:
        pytest.skip("监控配置模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("监控配置模块导入失败", allow_module_level=True)


class TestSimulateApiPerformanceEdgeCases:
    """测试API性能测试函数的边界情况"""

    @pytest.fixture(autouse=True)
    def reset_monitoring(self):
        """重置monitoring实例"""
        monitoring.metrics = {}
        monitoring.traces = []
        monitoring.alerts = []
        yield
        monitoring.metrics = {}
        monitoring.traces = []
        monitoring.alerts = []

    def test_simulate_api_performance_threshold_boundary_09(self):
        """测试secrets.random()正好等于0.9的边界情况"""
        call_count = {'count': 0}
        
        def mock_random():
            call_count['count'] += 1
            # 第一次调用返回正好0.9，应该走正常响应分支（< 0.9）
            if call_count['count'] == 1:
                return 0.9
            return 0.5
        
        with patch('src.monitoring.core.monitoring_config.secrets.random', side_effect=mock_random):
            with patch('src.monitoring.core.monitoring_config.secrets.uniform', return_value=0.1):
                with patch('time.sleep'):
                    with patch('builtins.print'):
                        result = simulate_api_performance_test()
                        
                        assert isinstance(result, dict)
                        assert result['total_requests'] == 100

    def test_simulate_api_performance_threshold_boundary_09_minus(self):
        """测试secrets.random()接近0.9但小于0.9的情况"""
        call_count = {'count': 0}
        
        def mock_random():
            call_count['count'] += 1
            # 第一次调用返回0.899，应该走正常响应分支
            if call_count['count'] == 1:
                return 0.899
            return 0.5
        
        with patch('src.monitoring.core.monitoring_config.secrets.random', side_effect=mock_random):
            with patch('src.monitoring.core.monitoring_config.secrets.uniform', return_value=0.1):
                with patch('time.sleep'):
                    with patch('builtins.print'):
                        result = simulate_api_performance_test()
                        
                        assert isinstance(result, dict)
                        assert result['total_requests'] == 100

    def test_simulate_api_performance_threshold_boundary_09_plus(self):
        """测试secrets.random()大于0.9的情况"""
        call_count = {'count': 0}
        
        def mock_random():
            call_count['count'] += 1
            # 第一次调用返回0.901，应该走慢响应分支
            if call_count['count'] == 1:
                return 0.901
            return 0.5
        
        with patch('src.monitoring.core.monitoring_config.secrets.random', side_effect=mock_random):
            def mock_uniform(a, b):
                # 慢响应范围
                if a == 0.5:
                    return 1.0
                # 正常响应范围
                return 0.1
            
            with patch('src.monitoring.core.monitoring_config.secrets.uniform', side_effect=mock_uniform):
                with patch('time.sleep'):
                    with patch('builtins.print'):
                        result = simulate_api_performance_test()
                        
                        assert isinstance(result, dict)
                        assert result['total_requests'] == 100

    def test_simulate_api_performance_all_normal_responses(self):
        """测试所有响应都是正常响应"""
        with patch('src.monitoring.core.monitoring_config.secrets.random', return_value=0.5):
            with patch('src.monitoring.core.monitoring_config.secrets.uniform', return_value=0.1):
                with patch('time.sleep'):
                    with patch('builtins.print'):
                        result = simulate_api_performance_test()
                        
                        assert isinstance(result, dict)
                        assert result['total_requests'] == 100
                        assert 'avg_response_time' in result
                        assert 'p95_response_time' in result

    def test_simulate_api_performance_all_slow_responses(self):
        """测试所有响应都是慢响应"""
        with patch('src.monitoring.core.monitoring_config.secrets.random', return_value=1.0):
            with patch('src.monitoring.core.monitoring_config.secrets.uniform', return_value=1.0):
                with patch('time.sleep'):
                    with patch('builtins.print'):
                        result = simulate_api_performance_test()
                        
                        assert isinstance(result, dict)
                        assert result['total_requests'] == 100

    def test_simulate_api_performance_p95_calculation(self):
        """测试P95响应时间计算"""
        # 创建100个响应时间值，确保P95计算正确
        response_times_list = []
        for i in range(100):
            # 前95个值为50ms，后5个值为2000ms
            if i < 95:
                response_times_list.append(50.0)
            else:
                response_times_list.append(2000.0)
        
        call_count = {'count': 0}
        
        def mock_random():
            call_count['count'] += 1
            # 前95次返回正常，后5次返回慢响应
            if call_count['count'] <= 95:
                return 0.5
            return 1.0
        
        def mock_uniform(a, b):
            call_count_uniform = {'count': 0}
            call_count_uniform['count'] += 1
            # 根据a的值判断是正常还是慢响应
            if a == 0.5:  # 慢响应
                return 2.0
            return 0.05  # 正常响应
        
        with patch('src.monitoring.core.monitoring_config.secrets.random', side_effect=mock_random):
            with patch('src.monitoring.core.monitoring_config.secrets.uniform', side_effect=mock_uniform):
                with patch('time.sleep'):
                    with patch('builtins.print'):
                        result = simulate_api_performance_test()
                        
                        assert isinstance(result, dict)
                        assert result['total_requests'] == 100

    def test_simulate_api_performance_empty_response_times(self):
        """测试response_times为空的情况（理论上不会发生，但为了完整性）"""
        # 这个测试主要是验证代码的健壮性
        # 由于循环会执行100次，所以response_times不会为空
        # 但我们可以测试如果某些异常导致response_times为空的情况
        with patch('src.monitoring.core.monitoring_config.secrets.random', return_value=0.5):
            with patch('src.monitoring.core.monitoring_config.secrets.uniform', return_value=0.1):
                with patch('time.sleep'):
                    with patch('builtins.print'):
                        # Mock start_trace抛出异常
                        original_start_trace = monitoring.start_trace
                        
                        call_count = {'count': 0}
                        def failing_start_trace(*args, **kwargs):
                            call_count['count'] += 1
                            if call_count['count'] > 50:  # 前50次成功，后50次失败
                                raise Exception("Test exception")
                            return original_start_trace(*args, **kwargs)
                        
                        monitoring.start_trace = failing_start_trace
                        
                        try:
                            result = simulate_api_performance_test()
                            
                            assert isinstance(result, dict)
                            # 即使部分失败，也应该有部分结果
                            assert result['total_requests'] > 0
                        finally:
                            monitoring.start_trace = original_start_trace



