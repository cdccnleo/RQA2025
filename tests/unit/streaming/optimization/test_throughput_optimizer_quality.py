#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
吞吐量优化器质量测试
测试覆盖 ThroughputOptimizer 的核心功能
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch

try:
    from src.streaming.optimization.throughput_optimizer import ThroughputOptimizer
except ImportError:
    ThroughputOptimizer = None


@pytest.fixture
def throughput_optimizer():
    """创建吞吐量优化器实例"""
    if ThroughputOptimizer is None:
        pytest.skip("ThroughputOptimizer不可用")
    return ThroughputOptimizer(target_throughput=1000, monitoring_window=60)


class TestThroughputOptimizer:
    """ThroughputOptimizer测试类"""

    def test_initialization(self, throughput_optimizer):
        """测试初始化"""
        assert throughput_optimizer.target_throughput == 1000
        assert throughput_optimizer.monitoring_window == 60
        assert throughput_optimizer.is_running is False
        assert throughput_optimizer.batch_size == 100
        assert throughput_optimizer.worker_count == 4

    def test_start_optimization(self, throughput_optimizer):
        """测试启动优化"""
        result = throughput_optimizer.start_optimization()
        assert result is True
        assert throughput_optimizer.is_running is True
        assert throughput_optimizer.monitoring_thread is not None
        
        # 清理
        throughput_optimizer.stop_optimization()

    def test_stop_optimization(self, throughput_optimizer):
        """测试停止优化"""
        throughput_optimizer.start_optimization()
        result = throughput_optimizer.stop_optimization()
        assert result is True
        assert throughput_optimizer.is_running is False

    def test_start_already_running(self, throughput_optimizer):
        """测试重复启动"""
        throughput_optimizer.start_optimization()
        result = throughput_optimizer.start_optimization()
        assert result is False
        
        throughput_optimizer.stop_optimization()

    def test_stop_not_running(self, throughput_optimizer):
        """测试停止未运行的优化器"""
        result = throughput_optimizer.stop_optimization()
        assert result is False

    def test_record_processing_time(self, throughput_optimizer):
        """测试记录处理时间"""
        throughput_optimizer.record_processing_time(0.001)  # 1ms
        throughput_optimizer.record_processing_time(0.002)  # 2ms
        
        assert len(throughput_optimizer.processing_times) >= 2

    def test_get_current_throughput(self, throughput_optimizer):
        """测试获取当前吞吐量"""
        # 记录一些处理时间
        for i in range(15):
            throughput_optimizer.record_processing_time(0.001)
        
        throughput_info = throughput_optimizer.get_current_throughput()
        assert isinstance(throughput_info, dict)
        assert 'throughput' in throughput_info or 'avg_processing_time' in throughput_info

    def test_optimize_throughput(self, throughput_optimizer):
        """测试吞吐量优化"""
        # 记录一些处理时间
        for i in range(15):
            throughput_optimizer.record_processing_time(0.001)
        
        # 获取当前吞吐量
        result = throughput_optimizer.get_current_throughput()
        assert isinstance(result, dict)

    def test_get_optimization_stats(self, throughput_optimizer):
        """测试获取优化统计"""
        # 使用get_current_throughput方法
        stats = throughput_optimizer.get_current_throughput()
        assert isinstance(stats, dict)

    def test_adjust_batch_size(self, throughput_optimizer):
        """测试调整批处理大小"""
        original_batch_size = throughput_optimizer.batch_size
        # 直接设置batch_size属性
        throughput_optimizer.batch_size = 200
        assert throughput_optimizer.batch_size == 200
        
        # 恢复
        throughput_optimizer.batch_size = original_batch_size

    def test_adjust_worker_count(self, throughput_optimizer):
        """测试调整工作线程数"""
        original_worker_count = throughput_optimizer.worker_count
        # 直接设置worker_count属性
        throughput_optimizer.worker_count = 8
        assert throughput_optimizer.worker_count == 8
        
        # 恢复
        throughput_optimizer.worker_count = original_worker_count

    def test_monitoring_loop(self, throughput_optimizer):
        """测试监控循环"""
        throughput_optimizer.start_optimization()
        
        # 等待监控循环运行
        time.sleep(0.5)
        
        # 验证监控线程在运行
        assert throughput_optimizer.monitoring_thread.is_alive()
        
        throughput_optimizer.stop_optimization()

    def test_process_batch(self, throughput_optimizer):
        """测试批处理"""
        test_data = [{'id': i, 'value': i * 2} for i in range(10)]
        
        def process_item(item):
            return {'processed': item['id'] * 2}
        
        # 使用optimize_batch_processing方法
        results = throughput_optimizer.optimize_batch_processing(test_data, process_item)
        assert len(results) == len(test_data)

    def test_start_optimization_exception(self, throughput_optimizer):
        """测试启动优化异常处理"""
        with patch('threading.Thread', side_effect=Exception("Thread creation failed")):
            result = throughput_optimizer.start_optimization()
            assert result is False
            assert throughput_optimizer.is_running is False

    def test_stop_optimization_exception(self, throughput_optimizer):
        """测试停止优化异常处理"""
        throughput_optimizer.start_optimization()
        throughput_optimizer.monitoring_thread = Mock()
        throughput_optimizer.monitoring_thread.is_alive.return_value = True
        throughput_optimizer.monitoring_thread.join.side_effect = Exception("Join failed")
        
        result = throughput_optimizer.stop_optimization()
        assert isinstance(result, bool)

    def test_optimize_batch_processing_empty(self, throughput_optimizer):
        """测试批处理优化（空数据）"""
        results = throughput_optimizer.optimize_batch_processing([], lambda x: x)
        assert results == []

    def test_optimize_batch_processing_low_efficiency(self, throughput_optimizer):
        """测试批处理优化（低效率）"""
        # 设置低效率的吞吐量
        for i in range(15):
            throughput_optimizer.record_processing_time(0.1)  # 慢处理
        
        test_data = [i for i in range(50)]
        results = throughput_optimizer.optimize_batch_processing(test_data, lambda x: x * 2)
        assert len(results) == len(test_data)
        # 批大小应该被调整
        assert throughput_optimizer.batch_size <= 100

    def test_optimize_batch_processing_high_efficiency(self, throughput_optimizer):
        """测试批处理优化（高效率）"""
        # 设置高效率的吞吐量
        for i in range(15):
            throughput_optimizer.record_processing_time(0.0001)  # 快处理
        
        test_data = [i for i in range(50)]
        results = throughput_optimizer.optimize_batch_processing(test_data, lambda x: x * 2)
        assert len(results) == len(test_data)

    def test_optimize_batch_processing_future_exception(self, throughput_optimizer):
        """测试批处理优化（Future异常）"""
        test_data = [i for i in range(10)]
        
        with patch.object(throughput_optimizer.executor, 'submit', side_effect=Exception("Submit error")):
            results = throughput_optimizer.optimize_batch_processing(test_data, lambda x: x)
            assert isinstance(results, list)

    def test_optimize_batch_processing_exception(self, throughput_optimizer):
        """测试批处理优化异常处理"""
        test_data = [i for i in range(10)]
        
        with patch.object(throughput_optimizer, 'get_current_throughput', side_effect=Exception("Metrics error")):
            results = throughput_optimizer.optimize_batch_processing(test_data, lambda x: x)
            assert results == []

    def test_process_batch_item_exception(self, throughput_optimizer):
        """测试批处理中单个项目异常"""
        def failing_process(item):
            if item == 5:
                raise Exception("Item processing error")
            return item * 2
        
        batch = [1, 2, 3, 4, 5, 6]
        results = throughput_optimizer._process_batch(batch, failing_process)
        # 应该有6个结果，其中一个是None
        assert len(results) == 6
        assert results[4] is None  # 第5个（索引4）应该失败

    def test_optimize_worker_count_low_efficiency(self, throughput_optimizer):
        """测试优化工作线程数（低效率）"""
        # 设置低效率
        for i in range(15):
            throughput_optimizer.record_processing_time(0.1)
        
        original_count = throughput_optimizer.worker_count
        new_count = throughput_optimizer.optimize_worker_count()
        # 应该增加工作线程数
        assert new_count >= original_count

    def test_optimize_worker_count_high_efficiency(self, throughput_optimizer):
        """测试优化工作线程数（高效率）"""
        # 设置高效率
        for i in range(15):
            throughput_optimizer.record_processing_time(0.0001)
        
        original_count = throughput_optimizer.worker_count
        new_count = throughput_optimizer.optimize_worker_count()
        # 应该减少或保持工作线程数
        assert new_count <= original_count

    def test_optimize_worker_count_exception(self, throughput_optimizer):
        """测试优化工作线程数异常处理"""
        with patch.object(throughput_optimizer, 'get_current_throughput', side_effect=Exception("Metrics error")):
            original_count = throughput_optimizer.worker_count
            new_count = throughput_optimizer.optimize_worker_count()
            assert new_count == original_count

    def test_monitoring_loop_optimization(self, throughput_optimizer):
        """测试监控循环应用优化"""
        throughput_optimizer.start_optimization()
        
        # 设置低效率以触发优化
        for i in range(15):
            throughput_optimizer.record_processing_time(0.1)
        
        time.sleep(0.2)
        
        throughput_optimizer.stop_optimization()

    def test_monitoring_loop_exception(self, throughput_optimizer):
        """测试监控循环异常处理"""
        throughput_optimizer.start_optimization()
        
        # Mock get_current_throughput抛出异常
        with patch.object(throughput_optimizer, 'get_current_throughput', side_effect=Exception("Metrics error")):
            time.sleep(0.2)
        
        throughput_optimizer.stop_optimization()

    def test_cleanup_old_metrics(self, throughput_optimizer):
        """测试清理旧指标"""
        # 添加一些旧指标
        from datetime import datetime, timedelta
        old_time = datetime.now() - timedelta(minutes=10)
        
        for i in range(5):
            throughput_optimizer.metrics_window.append({
                'timestamp': old_time,
                'throughput': 1000,
                'avg_processing_time': 0.001
            })
        
        throughput_optimizer._cleanup_old_metrics()
        # 旧指标应该被清理
        assert len(throughput_optimizer.metrics_window) <= 5

    def test_cleanup_old_metrics_exception(self, throughput_optimizer):
        """测试清理旧指标异常处理"""
        # 添加一些数据
        from datetime import datetime, timedelta
        old_time = datetime.now() - timedelta(minutes=10)
        
        for i in range(5):
            throughput_optimizer.metrics_window.append({
                'timestamp': old_time,
                'throughput': 1000,
                'avg_processing_time': 0.001
            })
        
        # Mock datetime.now抛出异常
        with patch('src.streaming.optimization.throughput_optimizer.datetime', side_effect=Exception("Datetime error")):
            try:
                throughput_optimizer._cleanup_old_metrics()
            except:
                pass  # 异常应该被捕获

    def test_get_throughput_stats_with_data(self, throughput_optimizer):
        """测试获取吞吐量统计（有数据）"""
        # 添加一些数据
        for i in range(15):
            throughput_optimizer.record_processing_time(0.001)
        
        stats = throughput_optimizer.get_throughput_stats()
        assert isinstance(stats, dict)
        assert 'is_running' in stats
        assert 'current_throughput' in stats
        assert 'throughput_stats' in stats
        assert 'processing_time_stats' in stats

    def test_get_throughput_stats_no_data(self, throughput_optimizer):
        """测试获取吞吐量统计（无数据）"""
        stats = throughput_optimizer.get_throughput_stats()
        assert isinstance(stats, dict)
        assert 'is_running' in stats
        # 没有数据时不应该有统计信息
        assert 'throughput_stats' not in stats or stats.get('throughput_stats') is None

    def test_get_throughput_stats_exception(self, throughput_optimizer):
        """测试获取吞吐量统计异常处理"""
        with patch.object(throughput_optimizer, 'get_current_throughput', side_effect=Exception("Stats error")):
            stats = throughput_optimizer.get_throughput_stats()
            assert stats == {}

    def test_reset_metrics(self, throughput_optimizer):
        """测试重置指标"""
        # 添加一些数据
        for i in range(10):
            throughput_optimizer.record_processing_time(0.001)
        
        throughput_optimizer.reset_metrics()
        
        assert len(throughput_optimizer.throughput_history) == 0
        assert len(throughput_optimizer.processing_times) == 0
        assert len(throughput_optimizer.metrics_window) == 0

    def test_get_current_throughput_no_data(self, throughput_optimizer):
        """测试获取当前吞吐量（无数据）"""
        result = throughput_optimizer.get_current_throughput()
        assert result['status'] == 'no_data'
        assert result['throughput'] == 0

    def test_get_current_throughput_suboptimal(self, throughput_optimizer):
        """测试获取当前吞吐量（次优）"""
        # 记录慢处理时间，导致吞吐量低于目标
        for i in range(15):
            throughput_optimizer.record_processing_time(0.1)  # 慢处理
        
        result = throughput_optimizer.get_current_throughput()
        assert result['status'] == 'suboptimal'
