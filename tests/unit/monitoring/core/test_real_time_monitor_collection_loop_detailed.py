#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeMonitor MetricsCollector collection loop详细测试
补充_collection_loop、start_collection、stop_collection的详细测试
"""

import sys
import importlib
from pathlib import Path
import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta

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
    core_real_time_monitor_module = importlib.import_module('src.monitoring.core.real_time_monitor')
    MetricsCollector = getattr(core_real_time_monitor_module, 'MetricsCollector', None)
    MetricData = getattr(core_real_time_monitor_module, 'MetricData', None)
    if MetricsCollector is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestMetricsCollectorCollectionLoopDetailed:
    """测试MetricsCollector类collection loop的详细功能"""

    @pytest.fixture
    def collector(self):
        """创建MetricsCollector实例"""
        return MetricsCollector()

    def test_start_collection_creates_thread(self, collector):
        """测试启动收集创建线程"""
        collector._running = False
        
        collector.start_collection()
        
        assert collector._running == True
        assert collector._thread is not None
        assert collector._thread.is_alive()
        
        # 清理
        collector.stop_collection()

    def test_start_collection_thread_is_daemon(self, collector):
        """测试收集线程是守护线程"""
        collector._running = False
        
        collector.start_collection()
        
        assert collector._thread.daemon == True
        
        # 清理
        collector.stop_collection()

    def test_start_collection_idempotent(self, collector):
        """测试启动收集是幂等的（多次调用不会创建多个线程）"""
        collector._running = False
        
        collector.start_collection()
        first_thread = collector._thread
        
        collector.start_collection()
        second_thread = collector._thread
        
        # 应该是同一个线程
        assert first_thread == second_thread
        
        # 清理
        collector.stop_collection()

    def test_stop_collection_sets_running_false(self, collector):
        """测试停止收集设置running为False"""
        collector._running = True
        mock_thread = Mock()
        mock_thread.join = Mock()
        collector._thread = mock_thread
        
        collector.stop_collection()
        
        assert collector._running == False

    def test_stop_collection_joins_thread(self, collector):
        """测试停止收集会join线程"""
        collector._running = True
        mock_thread = Mock()
        mock_thread.join = Mock()
        collector._thread = mock_thread
        
        collector.stop_collection()
        
        mock_thread.join.assert_called_once_with(timeout=5)

    def test_stop_collection_no_thread_no_error(self, collector):
        """测试停止收集没有线程时不报错"""
        collector._running = True
        collector._thread = None
        
        # 不应该抛出异常
        collector.stop_collection()
        assert collector._running == False

    @patch('time.sleep')
    @patch.object(MetricsCollector, 'collect_all_metrics')
    def test_collection_loop_calls_collect_all_metrics(self, mock_collect, mock_sleep, collector):
        """测试收集循环调用collect_all_metrics"""
        collector.collection_interval = 0.1
        mock_collect.return_value = {'test_metric': MetricData(
            name='test_metric',
            value=42.0,
            timestamp=datetime.now(),
            tags={}
        )}
        
        # 设置运行标志并模拟循环一次
        collector._running = True
        call_count = 0
        
        def mock_sleep_side_effect(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                collector._running = False  # 停止循环
        
        mock_sleep.side_effect = mock_sleep_side_effect
        
        collector._collection_loop()
        
        # 至少调用了一次collect_all_metrics
        assert mock_collect.called

    @patch('time.sleep')
    @patch.object(MetricsCollector, 'collect_all_metrics')
    def test_collection_loop_updates_metrics(self, mock_collect, mock_sleep, collector):
        """测试收集循环更新metrics字典"""
        collector.collection_interval = 0.1
        test_metric = MetricData(
            name='test_metric',
            value=42.0,
            timestamp=datetime.now(),
            tags={}
        )
        mock_collect.return_value = {'test_metric': test_metric}
        
        collector._running = True
        call_count = 0
        
        def mock_sleep_side_effect(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                collector._running = False
        
        mock_sleep.side_effect = mock_sleep_side_effect
        
        collector._collection_loop()
        
        # 验证metrics被更新
        assert 'test_metric' in collector.metrics

    @patch('time.sleep')
    @patch('src.monitoring.core.real_time_monitor.datetime')
    def test_collection_loop_cleans_old_metrics(self, mock_datetime, mock_sleep, collector):
        """测试收集循环清理旧指标"""
        collector.collection_interval = 0.1
        
        # 创建一些旧的指标（超过1小时）
        old_time = datetime.now() - timedelta(hours=2)
        old_metric = MetricData(
            name='old_metric',
            value=1.0,
            timestamp=old_time,
            tags={}
        )
        
        # 创建一个新的指标
        new_time = datetime.now()
        new_metric = MetricData(
            name='new_metric',
            value=2.0,
            timestamp=new_time,
            tags={}
        )
        
        collector.metrics = {
            'old_metric': old_metric,
            'new_metric': new_metric
        }
        
        # 模拟collect_all_metrics返回一个新指标
        with patch.object(collector, 'collect_all_metrics', return_value={
            'latest_metric': MetricData(
                name='latest_metric',
                value=3.0,
                timestamp=new_time,
                tags={}
            )
        }):
            collector._running = True
            call_count = 0
            
            def mock_sleep_side_effect(seconds):
                nonlocal call_count
                call_count += 1
                if call_count >= 1:
                    collector._running = False
            
            mock_sleep.side_effect = mock_sleep_side_effect
            mock_datetime.now.return_value = new_time
            
            collector._collection_loop()
            
            # 旧指标应该被清理
            assert 'old_metric' not in collector.metrics
            # 新指标应该保留
            assert 'new_metric' in collector.metrics or 'latest_metric' in collector.metrics

    @patch('time.sleep')
    @patch.object(MetricsCollector, 'collect_all_metrics')
    def test_collection_loop_handles_exception(self, mock_collect, mock_sleep, collector):
        """测试收集循环处理异常"""
        collector.collection_interval = 0.1
        mock_collect.side_effect = Exception("Collection error")
        
        collector._running = True
        call_count = 0
        
        def mock_sleep_side_effect(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                collector._running = False
        
        mock_sleep.side_effect = mock_sleep_side_effect
        
        with patch('src.monitoring.core.real_time_monitor.logger') as mock_logger:
            # 不应该抛出异常，应该继续运行
            collector._collection_loop()
            
            # 验证错误被记录
            mock_logger.error.assert_called()
            error_call = mock_logger.error.call_args[0][0]
            assert 'Error in metrics collection loop' in error_call

    @patch('time.sleep')
    @patch.object(MetricsCollector, 'collect_all_metrics')
    def test_collection_loop_sleeps_with_interval(self, mock_collect, mock_sleep, collector):
        """测试收集循环按间隔sleep"""
        collector.collection_interval = 5.0
        mock_collect.return_value = {}
        
        collector._running = True
        call_count = 0
        
        def mock_sleep_side_effect(seconds):
            nonlocal call_count
            call_count += 1
            assert seconds == 5.0  # 验证sleep时间
            if call_count >= 1:
                collector._running = False
        
        mock_sleep.side_effect = mock_sleep_side_effect
        
        collector._collection_loop()
        
        # 验证sleep被调用
        assert mock_sleep.called

    @patch('time.sleep')
    @patch.object(MetricsCollector, 'collect_all_metrics')
    def test_collection_loop_stops_when_running_false(self, mock_collect, mock_sleep, collector):
        """测试收集循环在running为False时停止"""
        collector.collection_interval = 0.1
        mock_collect.return_value = {}
        
        collector._running = True
        call_count = 0
        max_calls = 3
        
        def mock_sleep_side_effect(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= max_calls:
                collector._running = False
        
        mock_sleep.side_effect = mock_sleep_side_effect
        
        collector._collection_loop()
        
        # 验证循环停止了
        assert collector._running == False
        # 验证调用了collect_all_metrics（至少一次）
        assert mock_collect.call_count >= 1

    def test_collection_lifecycle_full_cycle(self, collector):
        """测试完整的收集生命周期"""
        # 初始状态
        assert collector._running == False
        assert collector._thread is None
        
        # 启动收集
        collector.start_collection()
        assert collector._running == True
        assert collector._thread is not None
        
        # 等待一小段时间让线程运行
        time.sleep(0.2)
        
        # 停止收集
        collector.stop_collection()
        assert collector._running == False
        
        # 等待线程停止
        if collector._thread:
            collector._thread.join(timeout=1)
            assert not collector._thread.is_alive()

    def test_collection_loop_keeps_recent_metrics_only(self, collector):
        """测试收集循环只保留最近1小时的指标"""
        cutoff_time = datetime.now()
        
        # 创建不同时间的指标
        metrics = {
            'very_old': MetricData('very_old', 1.0, cutoff_time - timedelta(hours=2), {}),
            'old': MetricData('old', 2.0, cutoff_time - timedelta(minutes=61), {}),
            'recent': MetricData('recent', 3.0, cutoff_time - timedelta(minutes=30), {}),
            'very_recent': MetricData('very_recent', 4.0, cutoff_time - timedelta(minutes=1), {}),
        }
        
        collector.metrics = metrics.copy()
        
        with patch('src.monitoring.core.real_time_monitor.datetime') as mock_datetime:
            mock_datetime.now.return_value = cutoff_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            with patch.object(collector, 'collect_all_metrics', return_value={}):
                with patch('time.sleep') as mock_sleep:
                    collector._running = True
                    
                    def stop_after_one_iteration(seconds):
                        collector._running = False
                    
                    mock_sleep.side_effect = stop_after_one_iteration
                    
                    collector._collection_loop()
                    
                    # 只保留1小时内的指标
                    assert 'very_old' not in collector.metrics
                    assert 'old' not in collector.metrics
                    assert 'recent' in collector.metrics
                    assert 'very_recent' in collector.metrics

    def test_collection_loop_handles_empty_metrics_dict(self, collector):
        """测试收集循环处理空的metrics字典"""
        collector.metrics = {}
        
        with patch.object(collector, 'collect_all_metrics', return_value={}):
            with patch('time.sleep') as mock_sleep:
                collector._running = True
                
                def stop_immediately(seconds):
                    collector._running = False
                
                mock_sleep.side_effect = stop_immediately
                
                # 不应该抛出异常
                collector._collection_loop()
                assert isinstance(collector.metrics, dict)

