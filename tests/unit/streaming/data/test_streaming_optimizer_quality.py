#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流优化器质量测试
测试覆盖 StreamingOptimizer 的核心功能
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch

try:
    from src.streaming.data.streaming_optimizer import StreamingOptimizer
except ImportError:
    StreamingOptimizer = None


@pytest.fixture
def streaming_optimizer():
    """创建流优化器实例"""
    if StreamingOptimizer is None:
        pytest.skip("StreamingOptimizer不可用")
    return StreamingOptimizer(enable_auto_tuning=True)


class TestStreamingOptimizer:
    """StreamingOptimizer测试类"""

    def test_initialization(self, streaming_optimizer):
        """测试初始化"""
        assert streaming_optimizer.enable_auto_tuning is True
        assert streaming_optimizer.is_running is False
        assert streaming_optimizer.cpu_threshold == 80.0
        assert streaming_optimizer.memory_threshold == 85.0
        assert streaming_optimizer.latency_threshold == 100.0

    def test_start_optimization(self, streaming_optimizer):
        """测试启动优化"""
        result = streaming_optimizer.start_optimization()
        assert result is True
        assert streaming_optimizer.is_running is True
        
        # 清理
        streaming_optimizer.stop_optimization()

    def test_stop_optimization(self, streaming_optimizer):
        """测试停止优化"""
        streaming_optimizer.start_optimization()
        result = streaming_optimizer.stop_optimization()
        assert result is True
        assert streaming_optimizer.is_running is False

    def test_start_already_running(self, streaming_optimizer):
        """测试重复启动"""
        streaming_optimizer.start_optimization()
        result = streaming_optimizer.start_optimization()
        assert result is False
        
        streaming_optimizer.stop_optimization()

    def test_stop_not_running(self, streaming_optimizer):
        """测试停止未运行的优化器"""
        result = streaming_optimizer.stop_optimization()
        assert result is False

    def test_collect_performance_metrics(self, streaming_optimizer):
        """测试收集性能指标"""
        metrics = streaming_optimizer.collect_performance_metrics()
        assert isinstance(metrics, dict)
        assert 'timestamp' in metrics
        assert 'system' in metrics or 'process' in metrics

    def test_apply_optimizations(self, streaming_optimizer):
        """测试应用优化"""
        # apply_optimizations需要bottlenecks参数
        bottlenecks = ['CPU', 'memory']
        result = streaming_optimizer.apply_optimizations(bottlenecks)
        assert isinstance(result, list)

    def test_get_optimization_history(self, streaming_optimizer):
        """测试获取优化历史"""
        # 直接访问optimization_history属性
        history = streaming_optimizer.optimization_history
        assert isinstance(history, list)

    def test_add_optimization_rule(self, streaming_optimizer):
        """测试添加优化规则"""
        def custom_rule(metrics):
            return {}
        
        initial_count = len(streaming_optimizer.optimization_rules)
        streaming_optimizer.add_optimization_rule(custom_rule)
        assert len(streaming_optimizer.optimization_rules) == initial_count + 1

    def test_get_optimization_stats(self, streaming_optimizer):
        """测试获取优化统计"""
        stats = streaming_optimizer.get_optimization_stats()
        assert isinstance(stats, dict)
        assert 'is_running' in stats or 'last_optimization' in stats

    def test_optimization_loop(self, streaming_optimizer):
        """测试优化循环"""
        streaming_optimizer.start_optimization()
        
        # 等待优化循环运行
        time.sleep(0.5)
        
        # 验证优化线程在运行
        if streaming_optimizer.optimization_thread:
            assert streaming_optimizer.optimization_thread.is_alive()
        
        streaming_optimizer.stop_optimization()

    def test_add_optimization_rule_non_callable(self, streaming_optimizer):
        """测试添加非可调用优化规则"""
        initial_count = len(streaming_optimizer.optimization_rules)
        streaming_optimizer.add_optimization_rule("not_callable")
        # 应该不添加
        assert len(streaming_optimizer.optimization_rules) == initial_count

    def test_start_optimization_exception(self, streaming_optimizer):
        """测试启动优化异常处理"""
        with patch('threading.Thread', side_effect=Exception("Thread creation failed")):
            result = streaming_optimizer.start_optimization()
            assert result is False
            assert streaming_optimizer.is_running is False

    def test_stop_optimization_exception(self, streaming_optimizer):
        """测试停止优化异常处理"""
        streaming_optimizer.start_optimization()
        streaming_optimizer.optimization_thread = Mock()
        streaming_optimizer.optimization_thread.is_alive.return_value = True
        streaming_optimizer.optimization_thread.join.side_effect = Exception("Join failed")
        
        result = streaming_optimizer.stop_optimization()
        assert isinstance(result, bool)

    def test_collect_performance_metrics_process_exception(self, streaming_optimizer):
        """测试收集性能指标（进程指标异常）"""
        with patch('psutil.Process', side_effect=Exception("Process error")):
            metrics = streaming_optimizer.collect_performance_metrics()
            assert isinstance(metrics, dict)
            assert 'timestamp' in metrics

    def test_collect_performance_metrics_exception(self, streaming_optimizer):
        """测试收集性能指标异常处理"""
        with patch('psutil.cpu_percent', side_effect=Exception("Metrics error")):
            metrics = streaming_optimizer.collect_performance_metrics()
            assert metrics == {}

    def test_analyze_performance_bottlenecks_cpu(self, streaming_optimizer):
        """测试分析性能瓶颈（CPU）"""
        metrics = {
            'system': {
                'cpu_percent': 90.0,  # 超过阈值
                'memory_percent': 50.0
            }
        }
        bottlenecks = streaming_optimizer.analyze_performance_bottlenecks(metrics)
        assert len(bottlenecks) > 0
        assert any('CPU' in b for b in bottlenecks)

    def test_analyze_performance_bottlenecks_memory(self, streaming_optimizer):
        """测试分析性能瓶颈（内存）"""
        metrics = {
            'system': {
                'cpu_percent': 50.0,
                'memory_percent': 90.0  # 超过阈值
            }
        }
        bottlenecks = streaming_optimizer.analyze_performance_bottlenecks(metrics)
        assert len(bottlenecks) > 0
        assert any('memory' in b.lower() for b in bottlenecks)

    def test_analyze_performance_bottlenecks_latency(self, streaming_optimizer):
        """测试分析性能瓶颈（延迟）"""
        metrics = {
            'system': {
                'cpu_percent': 50.0,
                'memory_percent': 50.0
            },
            'streaming': {
                'processing_latency': 150.0  # 超过阈值
            }
        }
        bottlenecks = streaming_optimizer.analyze_performance_bottlenecks(metrics)
        assert len(bottlenecks) > 0
        assert any('latency' in b.lower() for b in bottlenecks)

    def test_analyze_performance_bottlenecks_exception(self, streaming_optimizer):
        """测试分析性能瓶颈异常处理"""
        metrics = None  # 无效指标
        bottlenecks = streaming_optimizer.analyze_performance_bottlenecks(metrics)
        assert isinstance(bottlenecks, list)

    def test_apply_optimizations_latency(self, streaming_optimizer):
        """测试应用优化（延迟）"""
        bottlenecks = ['High processing latency: 150.0ms']
        result = streaming_optimizer.apply_optimizations(bottlenecks)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_apply_optimizations_exception(self, streaming_optimizer):
        """测试应用优化异常处理"""
        with patch.object(streaming_optimizer, '_optimize_cpu_usage', side_effect=Exception("Optimize error")):
            bottlenecks = ['High CPU usage: 90.0%']
            result = streaming_optimizer.apply_optimizations(bottlenecks)
            assert isinstance(result, list)

    def test_apply_optimizations_history(self, streaming_optimizer):
        """测试应用优化记录历史"""
        initial_count = len(streaming_optimizer.optimization_history)
        bottlenecks = ['High CPU usage: 90.0%']
        streaming_optimizer.apply_optimizations(bottlenecks)
        assert len(streaming_optimizer.optimization_history) > initial_count

    def test_optimize_cpu_usage(self, streaming_optimizer):
        """测试优化CPU使用"""
        result = streaming_optimizer._optimize_cpu_usage()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_optimize_cpu_usage_exception(self, streaming_optimizer):
        """测试优化CPU使用异常处理"""
        with patch('gc.collect', side_effect=Exception("GC error")):
            result = streaming_optimizer._optimize_cpu_usage()
            assert isinstance(result, list)

    def test_optimize_memory_usage(self, streaming_optimizer):
        """测试优化内存使用"""
        result = streaming_optimizer._optimize_memory_usage()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_optimize_memory_usage_exception(self, streaming_optimizer):
        """测试优化内存使用异常处理"""
        with patch('logging.Logger.info', side_effect=Exception("Log error")):
            result = streaming_optimizer._optimize_memory_usage()
            assert isinstance(result, list)

    def test_optimize_latency(self, streaming_optimizer):
        """测试优化延迟"""
        result = streaming_optimizer._optimize_latency()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_optimize_latency_exception(self, streaming_optimizer):
        """测试优化延迟异常处理"""
        with patch('logging.Logger.info', side_effect=Exception("Log error")):
            result = streaming_optimizer._optimize_latency()
            assert isinstance(result, list)

    def test_optimization_loop_with_bottlenecks(self, streaming_optimizer):
        """测试优化循环（有瓶颈）"""
        streaming_optimizer.start_optimization()
        
        # Mock收集到高CPU指标
        with patch.object(streaming_optimizer, 'collect_performance_metrics', return_value={
            'system': {'cpu_percent': 90.0, 'memory_percent': 50.0}
        }):
            time.sleep(0.1)
        
        streaming_optimizer.stop_optimization()

    def test_optimization_loop_exception(self, streaming_optimizer):
        """测试优化循环异常处理"""
        streaming_optimizer.start_optimization()
        
        # Mock collect_performance_metrics抛出异常
        with patch.object(streaming_optimizer, 'collect_performance_metrics', side_effect=Exception("Metrics error")):
            time.sleep(0.1)
        
        streaming_optimizer.stop_optimization()

    def test_optimization_loop_logger_closed(self, streaming_optimizer):
        """测试优化循环中日志流关闭的情况"""
        import logging
        from unittest.mock import patch
        
        streaming_optimizer.start_optimization()
        
        # Mock logger在循环中关闭
        with patch('src.streaming.data.streaming_optimizer.logger.info', side_effect=ValueError("I/O operation on closed file")):
            with patch('src.streaming.data.streaming_optimizer.logger.error', side_effect=ValueError("I/O operation on closed file")):
                # Mock collect_performance_metrics抛出异常，触发error日志
                with patch.object(streaming_optimizer, 'collect_performance_metrics', side_effect=Exception("Metrics error")):
                    time.sleep(0.1)
        
        streaming_optimizer.stop_optimization()

    def test_stop_optimization_logger_closed(self, streaming_optimizer):
        """测试停止优化时日志流关闭的情况"""
        import logging
        from unittest.mock import patch
        
        streaming_optimizer.start_optimization()
        
        # Mock logger在stop时关闭
        with patch('src.streaming.data.streaming_optimizer.logger.warning', side_effect=ValueError("I/O operation on closed file")):
            with patch('src.streaming.data.streaming_optimizer.logger.info', side_effect=ValueError("I/O operation on closed file")):
                with patch('src.streaming.data.streaming_optimizer.logger.error', side_effect=ValueError("I/O operation on closed file")):
                    # 先停止，然后再次停止（触发warning）
                    streaming_optimizer.stop_optimization()
                    result = streaming_optimizer.stop_optimization()
                    assert result is False

    def test_set_thresholds(self, streaming_optimizer):
        """测试设置阈值"""
        streaming_optimizer.set_thresholds(cpu=85.0, memory=90.0, latency=150.0)
        assert streaming_optimizer.cpu_threshold == 85.0
        assert streaming_optimizer.memory_threshold == 90.0
        assert streaming_optimizer.latency_threshold == 150.0

    def test_set_thresholds_partial(self, streaming_optimizer):
        """测试设置部分阈值"""
        original_cpu = streaming_optimizer.cpu_threshold
        streaming_optimizer.set_thresholds(memory=90.0)
        assert streaming_optimizer.cpu_threshold == original_cpu
        assert streaming_optimizer.memory_threshold == 90.0
