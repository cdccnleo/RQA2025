#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig模拟API性能测试详细测试
补充simulate_api_performance_test函数的详细测试，包括返回值、统计计算等
"""

import pytest
from unittest.mock import patch, MagicMock
import time

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
    core_monitoring_config_module = importlib.import_module('src.monitoring.core.monitoring_config')
    simulate_api_performance_test = getattr(core_monitoring_config_module, 'simulate_api_performance_test', None)
    monitoring = getattr(core_monitoring_config_module, 'monitoring', None)
    if simulate_api_performance_test is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)

class TestSimulateApiPerformanceTest:
    """测试simulate_api_performance_test函数"""

    @pytest.fixture(autouse=True)
    def reset_monitoring(self):
        """每次测试前重置monitoring实例"""
        monitoring.metrics = {}
        monitoring.traces = []
        monitoring.alerts = []
        yield
        monitoring.metrics = {}
        monitoring.traces = []
        monitoring.alerts = []

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_simulate_api_performance_test_returns_dict(self, mock_monitoring, mock_sleep, mock_print):
        """测试函数返回字典"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        # Mock random and uniform
        with patch('random.random', return_value=0.5):
            with patch('random.uniform', return_value=0.1):
                result = simulate_api_performance_test()
        
        assert isinstance(result, dict)

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_simulate_api_performance_test_has_required_keys(self, mock_monitoring, mock_sleep, mock_print):
        """测试返回值包含必需的键"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.random', return_value=0.5):
            with patch('random.uniform', return_value=0.1):
                result = simulate_api_performance_test()
        
        assert 'avg_response_time' in result
        assert 'p95_response_time' in result
        assert 'total_requests' in result

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_simulate_api_performance_test_total_requests(self, mock_monitoring, mock_sleep, mock_print):
        """测试total_requests为100"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.random', return_value=0.5):
            with patch('random.uniform', return_value=0.1):
                result = simulate_api_performance_test()
        
        assert result['total_requests'] == 100

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_simulate_api_performance_test_calls_start_trace(self, mock_monitoring, mock_sleep, mock_print):
        """测试调用start_trace"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.random', return_value=0.5):
            with patch('random.uniform', return_value=0.1):
                simulate_api_performance_test()
        
        # 应该调用100次start_trace
        assert mock_monitoring.start_trace.call_count == 100

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_simulate_api_performance_test_calls_end_trace(self, mock_monitoring, mock_sleep, mock_print):
        """测试调用end_trace"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.random', return_value=0.5):
            with patch('random.uniform', return_value=0.1):
                simulate_api_performance_test()
        
        # 应该调用100次end_trace
        assert mock_monitoring.end_trace.call_count == 100

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_simulate_api_performance_test_calls_record_metric(self, mock_monitoring, mock_sleep, mock_print):
        """测试调用record_metric"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.random', return_value=0.5):
            with patch('random.uniform', return_value=0.1):
                simulate_api_performance_test()
        
        # 应该调用100次record_metric
        assert mock_monitoring.record_metric.call_count == 100

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_simulate_api_performance_test_avg_response_time_calculation(self, mock_monitoring, mock_sleep, mock_print):
        """测试平均响应时间计算"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        # 固定响应时间100ms
        with patch('random.random', return_value=0.5):
            with patch('random.uniform', return_value=0.1):
                result = simulate_api_performance_test()
        
        # 0.1秒 = 100ms
        assert result['avg_response_time'] == 100.0

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_simulate_api_performance_test_p95_calculation(self, mock_monitoring, mock_sleep, mock_print):
        """测试P95响应时间计算"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        # 固定响应时间100ms
        with patch('random.random', return_value=0.5):
            with patch('random.uniform', return_value=0.1):
                result = simulate_api_performance_test()
        
        # P95应该是排序后的第95个元素（索引94）
        assert result['p95_response_time'] == 100.0

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_simulate_api_performance_test_sleep_called(self, mock_monitoring, mock_sleep, mock_print):
        """测试sleep被调用"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.random', return_value=0.5):
            with patch('random.uniform', return_value=0.1):
                simulate_api_performance_test()
        
        # 应该调用100次sleep
        assert mock_sleep.call_count == 100

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_simulate_api_performance_test_trace_id_format(self, mock_monitoring, mock_sleep, mock_print):
        """测试trace_id格式"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.random', return_value=0.5):
            with patch('random.uniform', return_value=0.1):
                simulate_api_performance_test()
        
        # 验证start_trace被调用时的参数格式
        calls = mock_monitoring.start_trace.call_args_list
        assert calls[0][0][0] == 'api_test_0'
        assert calls[0][0][1] == 'api_call'
        assert calls[99][0][0] == 'api_test_99'

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_simulate_api_performance_test_endpoint_tag(self, mock_monitoring, mock_sleep, mock_print):
        """测试record_metric包含endpoint tag"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.random', return_value=0.5):
            with patch('random.uniform', return_value=0.1):
                simulate_api_performance_test()
        
        # 验证record_metric被调用时包含endpoint tag
        calls = mock_monitoring.record_metric.call_args_list
        assert calls[0][0][0] == 'api_response_time'
        # tags是第三个位置参数
        assert len(calls[0][0]) >= 3
        assert calls[0][0][2]['endpoint'] == '/api/test'

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_simulate_api_performance_test_response_time_tag(self, mock_monitoring, mock_sleep, mock_print):
        """测试end_trace包含response_time_ms tag"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.random', return_value=0.5):
            with patch('random.uniform', return_value=0.1):
                simulate_api_performance_test()
        
        # 验证end_trace被调用时包含response_time_ms tag
        calls = mock_monitoring.end_trace.call_args_list
        assert 'response_time_ms' in calls[0][0][1]

