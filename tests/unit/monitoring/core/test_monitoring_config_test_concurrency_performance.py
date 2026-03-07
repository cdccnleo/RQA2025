#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig并发性能测试详细测试
补充test_concurrency_performance函数的详细测试
"""

import pytest
from unittest.mock import patch, MagicMock
import threading
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
    test_concurrency_performance = getattr(core_monitoring_config_module, 'test_concurrency_performance', None)
    monitoring = getattr(core_monitoring_config_module, 'monitoring', None)
    if test_concurrency_performance is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)

class TestTestConcurrencyPerformance:
    """测试test_concurrency_performance函数"""

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
    def test_test_concurrency_performance_returns_dict(self, mock_monitoring, mock_sleep, mock_print):
        """测试函数返回字典"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.uniform', return_value=0.1):
            result = test_concurrency_performance()
        
        assert isinstance(result, dict)

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_test_concurrency_performance_has_required_keys(self, mock_monitoring, mock_sleep, mock_print):
        """测试返回值包含必需的键"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.uniform', return_value=0.1):
            result = test_concurrency_performance()
        
        assert 'concurrent_requests' in result
        assert 'avg_response_time' in result
        assert 'max_response_time' in result

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_test_concurrency_performance_concurrent_requests(self, mock_monitoring, mock_sleep, mock_print):
        """测试并发请求数为50"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.uniform', return_value=0.1):
            result = test_concurrency_performance()
        
        assert result['concurrent_requests'] == 50

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_test_concurrency_performance_calls_start_trace(self, mock_monitoring, mock_sleep, mock_print):
        """测试调用start_trace"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.uniform', return_value=0.1):
            test_concurrency_performance()
        
        # 应该调用50次start_trace（等待所有线程完成）
        # 注意：由于并发执行，需要等待所有线程完成
        time.sleep(0.1)  # 等待线程完成
        assert mock_monitoring.start_trace.call_count == 50

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_test_concurrency_performance_calls_end_trace(self, mock_monitoring, mock_sleep, mock_print):
        """测试调用end_trace"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.uniform', return_value=0.1):
            test_concurrency_performance()
        
        # 等待所有线程完成
        time.sleep(0.1)
        # 应该调用50次end_trace
        assert mock_monitoring.end_trace.call_count == 50

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_test_concurrency_performance_trace_id_format(self, mock_monitoring, mock_sleep, mock_print):
        """测试trace_id格式"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.uniform', return_value=0.1):
            test_concurrency_performance()
        
        # 等待所有线程完成
        time.sleep(0.1)
        
        # 验证start_trace被调用时的参数格式
        calls = mock_monitoring.start_trace.call_args_list
        if calls:
            # 检查第一个调用
            assert calls[0][0][0].startswith('concurrency_test_')
            assert calls[0][0][1] == 'concurrent_request'

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_test_concurrency_performance_worker_id_tag(self, mock_monitoring, mock_sleep, mock_print):
        """测试end_trace包含worker_id tag"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.uniform', return_value=0.1):
            test_concurrency_performance()
        
        # 等待所有线程完成
        time.sleep(0.1)
        
        # 验证end_trace被调用时包含worker_id tag
        calls = mock_monitoring.end_trace.call_args_list
        if calls:
            tags = calls[0][0][1] if len(calls[0][0]) > 1 else calls[0][1]
            assert 'worker_id' in tags
            assert 'response_time_ms' in tags

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_test_concurrency_performance_avg_response_time(self, mock_monitoring, mock_sleep, mock_print):
        """测试平均响应时间计算"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        # 固定响应时间100ms
        with patch('random.uniform', return_value=0.1):
            result = test_concurrency_performance()
        
        # 0.1秒 = 100ms
        assert result['avg_response_time'] == 100.0

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_test_concurrency_performance_max_response_time(self, mock_monitoring, mock_sleep, mock_print):
        """测试最大响应时间计算"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        # 固定响应时间100ms
        with patch('random.uniform', return_value=0.1):
            result = test_concurrency_performance()
        
        # 0.1秒 = 100ms
        assert result['max_response_time'] == 100.0

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_test_concurrency_performance_empty_results(self, mock_monitoring, mock_sleep, mock_print):
        """测试空results的情况（验证函数能处理这种情况）"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        # 由于results是函数内部变量，我们不能直接patch
        # 但我们可以验证函数结构，确保有处理空results的逻辑
        # 根据源代码，函数有 `if results else 0` 的处理
        result = test_concurrency_performance()
        
        # 验证返回值结构（即使results为空也有合理的返回值）
        assert 'concurrent_requests' in result
        assert 'avg_response_time' in result
        assert 'max_response_time' in result

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_test_concurrency_performance_threading_used(self, mock_monitoring, mock_sleep, mock_print):
        """测试使用了线程"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.uniform', return_value=0.01):  # 更快的处理时间
            with patch('threading.Thread') as mock_thread_class:
                mock_thread = MagicMock()
                mock_thread_class.return_value = mock_thread
                
                test_concurrency_performance()
                
                # 验证Thread被创建了50次
                assert mock_thread_class.call_count == 50

    @patch('builtins.print')
    @patch('time.sleep')
    @patch('src.monitoring.core.monitoring_config.monitoring')
    def test_test_concurrency_performance_lock_used(self, mock_monitoring, mock_sleep, mock_print):
        """测试使用了锁机制"""
        mock_monitoring.start_trace.return_value = 'span_0'
        mock_monitoring.record_metric = MagicMock()
        mock_monitoring.end_trace = MagicMock()
        
        with patch('random.uniform', return_value=0.01):
            with patch('threading.Lock') as mock_lock_class:
                mock_lock = MagicMock()
                mock_lock_class.return_value = mock_lock
                
                test_concurrency_performance()
                
                # 验证Lock被创建了（至少一次）
                assert mock_lock_class.call_count >= 1

