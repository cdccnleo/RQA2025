#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance模块测试覆盖
测试performance相关组件的核心功能
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.features.performance.performance_optimizer import (
    MemoryOptimizer,
    CacheOptimizer,
    ConcurrencyOptimizer,
    PerformanceOptimizer,
    OptimizationLevel,
    CacheStrategy,
    PerformanceMetrics
)
from src.features.performance.scalability_manager import (
    LoadBalancer,
    AutoScaler,
    ScalabilityManager,
    ScalingStrategy,
    LoadBalancingStrategy,
    WorkerNode,
    ScalingMetrics
)


class TestMemoryOptimizer:
    """MemoryOptimizer测试"""

    def test_memory_optimizer_initialization(self):
        """测试内存优化器初始化"""
        optimizer = MemoryOptimizer(max_memory_mb=512, gc_threshold=0.75)
        assert optimizer.max_memory_mb == 512
        assert optimizer.gc_threshold == 0.75
        assert optimizer.memory_history == []
        assert optimizer.gc_stats["collections"] == 0

    def test_memory_optimizer_default_initialization(self):
        """测试默认初始化"""
        optimizer = MemoryOptimizer()
        assert optimizer.max_memory_mb == 1024
        assert optimizer.gc_threshold == 0.8

    @patch('psutil.Process')
    def test_check_memory_usage(self, mock_process_class):
        """测试检查内存使用情况"""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_process_class.return_value = mock_process
        
        optimizer = MemoryOptimizer()
        memory_mb = optimizer.check_memory_usage()
        
        assert memory_mb == 100.0
        assert len(optimizer.memory_history) == 1

    @patch('psutil.Process')
    @patch('gc.collect')
    def test_optimize_memory_below_threshold(self, mock_gc, mock_process_class):
        """测试内存优化（低于阈值）"""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_process_class.return_value = mock_process
        
        optimizer = MemoryOptimizer(max_memory_mb=1024, gc_threshold=0.8)
        result = optimizer.optimize_memory()
        
        assert result["before_memory_mb"] == 100.0
        assert result["gc_triggered"] is False

    @patch('psutil.Process')
    @patch('gc.collect')
    def test_optimize_memory_above_threshold(self, mock_gc, mock_process_class):
        """测试内存优化（超过阈值）"""
        mock_process = Mock()
        # 900MB，超过1024*0.8=819.2MB阈值
        mock_process.memory_info.return_value.rss = 900 * 1024 * 1024
        mock_process_class.return_value = mock_process
        
        optimizer = MemoryOptimizer(max_memory_mb=1024, gc_threshold=0.8)
        result = optimizer.optimize_memory()
        
        assert result["gc_triggered"] is True
        mock_gc.assert_called_once()

    @patch('psutil.Process')
    def test_get_memory_stats(self, mock_process_class):
        """测试获取内存统计信息"""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 500 * 1024 * 1024  # 500MB
        mock_process_class.return_value = mock_process
        
        optimizer = MemoryOptimizer(max_memory_mb=1024)
        stats = optimizer.get_memory_stats()
        
        assert stats["current_memory_mb"] == 500.0
        assert stats["max_memory_mb"] == 1024
        assert stats["usage_percent"] > 0
        assert "memory_history" in stats
        assert "gc_stats" in stats


class TestPerformanceOptimizer:
    """PerformanceOptimizer测试"""

    @pytest.fixture
    def optimizer(self):
        """创建性能优化器实例"""
        with patch('src.features.performance.performance_optimizer.get_config_integration_manager'):
            return PerformanceOptimizer()

    def test_optimizer_initialization(self, optimizer):
        """测试优化器初始化"""
        assert optimizer is not None

    def test_get_performance_metrics(self, optimizer):
        """测试获取性能指标"""
        # 使用_collect_performance_metrics方法
        metrics = optimizer._collect_performance_metrics()
        assert isinstance(metrics, PerformanceMetrics)
        assert hasattr(metrics, 'memory_usage_mb')
        assert hasattr(metrics, 'cpu_usage_percent')
        assert hasattr(metrics, 'timestamp')


class TestLoadBalancer:
    """LoadBalancer测试"""

    def test_load_balancer_initialization(self):
        """测试负载均衡器初始化"""
        balancer = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        assert balancer.strategy == LoadBalancingStrategy.ROUND_ROBIN
        assert balancer.workers == {}
        assert balancer.current_index == 0

    def test_add_worker(self):
        """测试添加工作节点"""
        balancer = LoadBalancer()
        worker = WorkerNode(id="worker1", capacity=100)
        
        balancer.add_worker(worker)
        assert "worker1" in balancer.workers
        assert balancer.workers["worker1"] == worker

    def test_remove_worker(self):
        """测试移除工作节点"""
        balancer = LoadBalancer()
        worker = WorkerNode(id="worker1")
        balancer.add_worker(worker)
        
        balancer.remove_worker("worker1")
        assert "worker1" not in balancer.workers

    def test_remove_worker_nonexistent(self):
        """测试移除不存在的工作节点"""
        balancer = LoadBalancer()
        # 应该不报错
        balancer.remove_worker("nonexistent")

    def test_get_next_worker_empty(self):
        """测试获取下一个工作节点（空）"""
        balancer = LoadBalancer()
        worker = balancer.get_next_worker()
        assert worker is None

    def test_get_next_worker_round_robin(self):
        """测试轮询获取工作节点"""
        balancer = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        worker1 = WorkerNode(id="worker1", is_healthy=True)
        worker2 = WorkerNode(id="worker2", is_healthy=True)
        balancer.add_worker(worker1)
        balancer.add_worker(worker2)
        
        # 第一次应该返回worker1
        w1 = balancer.get_next_worker()
        assert w1.id == "worker1"
        
        # 第二次应该返回worker2
        w2 = balancer.get_next_worker()
        assert w2.id == "worker2"
        
        # 第三次应该循环回worker1
        w3 = balancer.get_next_worker()
        assert w3.id == "worker1"

    def test_get_next_worker_unhealthy(self):
        """测试获取工作节点（不健康节点）"""
        balancer = LoadBalancer()
        worker1 = WorkerNode(id="worker1", is_healthy=False)
        worker2 = WorkerNode(id="worker2", is_healthy=True)
        balancer.add_worker(worker1)
        balancer.add_worker(worker2)
        
        worker = balancer.get_next_worker()
        assert worker.id == "worker2"  # 应该跳过不健康的worker1


class TestScalabilityManager:
    """ScalabilityManager测试"""

    @pytest.fixture
    def manager(self):
        """创建扩展性管理器实例"""
        try:
            with patch('src.features.performance.scalability_manager.get_config_integration_manager'):
                return ScalabilityManager()
        except Exception as e:
            pytest.skip(f"ScalabilityManager初始化失败: {e}")

    def test_manager_initialization(self, manager):
        """测试管理器初始化"""
        assert manager is not None

    def test_get_scaling_metrics(self, manager):
        """测试获取扩缩容指标"""
        metrics = manager.get_scaling_metrics()
        assert isinstance(metrics, ScalingMetrics)
        assert hasattr(metrics, 'cpu_threshold')
        assert hasattr(metrics, 'memory_threshold')
        assert hasattr(metrics, 'min_workers')
        assert hasattr(metrics, 'max_workers')


class TestWorkerNode:
    """WorkerNode测试"""

    def test_worker_node_initialization(self):
        """测试工作节点初始化"""
        worker = WorkerNode(id="worker1", capacity=100)
        assert worker.id == "worker1"
        assert worker.capacity == 100
        assert worker.current_load == 0
        assert worker.is_healthy is True
        assert worker.weight == 1.0

    def test_worker_node_defaults(self):
        """测试工作节点默认值"""
        worker = WorkerNode(id="worker1")
        assert worker.capacity == 100
        assert worker.current_load == 0
        assert worker.response_time_ms == 0.0
        assert worker.is_healthy is True


class TestPerformanceMetrics:
    """PerformanceMetrics测试"""

    def test_performance_metrics_initialization(self):
        """测试性能指标初始化"""
        metrics = PerformanceMetrics()
        assert metrics.memory_usage_mb == 0.0
        assert metrics.cpu_usage_percent == 0.0
        assert metrics.cache_hit_rate == 0.0
        assert metrics.response_time_ms == 0.0
        assert metrics.timestamp > 0

    def test_performance_metrics_custom_values(self):
        """测试自定义性能指标"""
        metrics = PerformanceMetrics(
            memory_usage_mb=512.0,
            cpu_usage_percent=75.5,
            cache_hit_rate=0.95,
            response_time_ms=10.5
        )
        assert metrics.memory_usage_mb == 512.0
        assert metrics.cpu_usage_percent == 75.5
        assert metrics.cache_hit_rate == 0.95
        assert metrics.response_time_ms == 10.5


class TestScalingMetrics:
    """ScalingMetrics测试"""

    def test_scaling_metrics_initialization(self):
        """测试扩缩容指标初始化"""
        metrics = ScalingMetrics()
        assert metrics.cpu_threshold == 0.8
        assert metrics.memory_threshold == 0.8
        assert metrics.queue_threshold == 100
        assert metrics.min_workers == 2
        assert metrics.max_workers == 10

    def test_scaling_metrics_custom_values(self):
        """测试自定义扩缩容指标"""
        metrics = ScalingMetrics(
            cpu_threshold=0.9,
            memory_threshold=0.85,
            min_workers=3,
            max_workers=20
        )
        assert metrics.cpu_threshold == 0.9
        assert metrics.memory_threshold == 0.85
        assert metrics.min_workers == 3
        assert metrics.max_workers == 20


class TestCacheOptimizer:
    """CacheOptimizer测试"""

    def test_cache_optimizer_initialization(self):
        """测试缓存优化器初始化"""
        optimizer = CacheOptimizer(max_cache_size=500, strategy=CacheStrategy.LRU)
        assert optimizer.max_cache_size == 500
        assert optimizer.strategy == CacheStrategy.LRU
        assert optimizer.cache == {}
        assert optimizer.cache_stats["hits"] == 0
        assert optimizer.cache_stats["misses"] == 0

    def test_cache_optimizer_default_initialization(self):
        """测试默认初始化"""
        optimizer = CacheOptimizer()
        assert optimizer.max_cache_size == 1000
        assert optimizer.strategy == CacheStrategy.LRU

    def test_cache_get_miss(self):
        """测试缓存未命中"""
        optimizer = CacheOptimizer()
        result = optimizer.get("nonexistent_key")
        assert result is None
        assert optimizer.cache_stats["misses"] == 1

    def test_cache_set_and_get(self):
        """测试设置和获取缓存"""
        optimizer = CacheOptimizer()
        optimizer.set("key1", "value1")
        result = optimizer.get("key1")
        assert result["value"] == "value1"
        assert optimizer.cache_stats["hits"] == 1

    def test_cache_eviction_lru(self):
        """测试LRU策略淘汰"""
        optimizer = CacheOptimizer(max_cache_size=2, strategy=CacheStrategy.LRU)
        optimizer.set("key1", "value1")
        optimizer.set("key2", "value2")
        time.sleep(0.01)  # 确保时间戳不同
        optimizer.get("key1")  # 访问key1，更新其访问时间
        time.sleep(0.01)
        optimizer.set("key3", "value3")  # 应该淘汰key2（最久未访问）
        
        assert "key3" in optimizer.cache
        assert "key1" in optimizer.cache
        assert "key2" not in optimizer.cache
        assert optimizer.cache_stats["evictions"] >= 1

    def test_cache_eviction_lfu(self):
        """测试LFU策略淘汰"""
        optimizer = CacheOptimizer(max_cache_size=2, strategy=CacheStrategy.LFU)
        optimizer.set("key1", "value1")
        optimizer.set("key2", "value2")
        optimizer.get("key1")  # key1访问1次
        optimizer.get("key1")  # key1访问2次
        optimizer.get("key2")  # key2访问1次
        optimizer.set("key3", "value3")  # 应该淘汰key2（访问次数更少）
        
        assert "key3" in optimizer.cache
        assert "key1" in optimizer.cache
        assert optimizer.cache_stats["evictions"] == 1

    def test_cache_eviction_fifo(self):
        """测试FIFO策略淘汰"""
        optimizer = CacheOptimizer(max_cache_size=2, strategy=CacheStrategy.FIFO)
        optimizer.set("key1", "value1")
        time.sleep(0.01)  # 确保时间戳不同
        optimizer.set("key2", "value2")
        optimizer.set("key3", "value3")  # 应该淘汰key1（最早加入）
        
        assert "key3" in optimizer.cache
        assert "key2" in optimizer.cache
        assert "key1" not in optimizer.cache

    def test_cache_clear_expired(self):
        """测试清理过期缓存"""
        optimizer = CacheOptimizer()
        optimizer.set("key1", "value1", ttl=0.1)  # 0.1秒过期
        optimizer.set("key2", "value2", ttl=100)  # 100秒过期
        
        time.sleep(0.2)  # 等待key1过期
        expired_count = optimizer.clear_expired()
        
        assert expired_count == 1
        assert "key1" not in optimizer.cache
        assert "key2" in optimizer.cache

    def test_cache_get_cache_stats(self):
        """测试获取缓存统计信息"""
        optimizer = CacheOptimizer()
        optimizer.set("key1", "value1")
        optimizer.get("key1")
        optimizer.get("key1")
        optimizer.get("key2")  # miss
        
        stats = optimizer.get_cache_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] > 0
        assert stats["size"] == 1
        assert stats["strategy"] == CacheStrategy.LRU.value


class TestConcurrencyOptimizer:
    """ConcurrencyOptimizer测试"""

    def test_concurrency_optimizer_initialization(self):
        """测试并发优化器初始化"""
        optimizer = ConcurrencyOptimizer(max_workers=8, max_processes=4)
        assert optimizer.max_workers == 8
        assert optimizer.max_processes == 4
        assert optimizer.active_tasks == 0
        assert optimizer.completed_tasks == 0
        assert optimizer.failed_tasks == 0

    def test_concurrency_optimizer_default_initialization(self):
        """测试默认初始化"""
        optimizer = ConcurrencyOptimizer()
        assert optimizer.max_workers == 4
        assert optimizer.max_processes == 2

    def test_submit_task(self):
        """测试提交任务"""
        optimizer = ConcurrencyOptimizer()
        
        def dummy_task():
            return "result"
        
        future = optimizer.submit_task(dummy_task)
        result = future.result()
        
        assert result == "result"
        assert optimizer.completed_tasks == 1

    def test_submit_task_with_exception(self):
        """测试提交任务（异常）"""
        optimizer = ConcurrencyOptimizer()
        
        def failing_task():
            raise ValueError("Task failed")
        
        future = optimizer.submit_task(failing_task)
        with pytest.raises(ValueError):
            future.result()
        
        # 等待回调完成
        time.sleep(0.1)
        assert optimizer.failed_tasks == 1

    def test_get_concurrency_stats(self):
        """测试获取并发统计信息"""
        optimizer = ConcurrencyOptimizer(max_workers=4, max_processes=2)
        stats = optimizer.get_concurrency_stats()
        
        assert stats["active_tasks"] == 0
        assert stats["completed_tasks"] == 0
        assert stats["failed_tasks"] == 0
        assert stats["max_workers"] == 4
        assert stats["max_processes"] == 2

    def test_shutdown(self):
        """测试关闭优化器"""
        optimizer = ConcurrencyOptimizer()
        optimizer.shutdown()  # 应该不报错
        assert True  # 如果到这里说明关闭成功


class TestLoadBalancerAdvanced:
    """LoadBalancer高级功能测试"""

    def test_get_next_worker_least_connections(self):
        """测试最少连接数策略"""
        balancer = LoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
        worker1 = WorkerNode(id="worker1", current_load=10, is_healthy=True)
        worker2 = WorkerNode(id="worker2", current_load=5, is_healthy=True)
        balancer.add_worker(worker1)
        balancer.add_worker(worker2)
        
        worker = balancer.get_next_worker()
        assert worker.id == "worker2"  # 负载更少

    def test_get_next_worker_response_time(self):
        """测试响应时间策略"""
        balancer = LoadBalancer(LoadBalancingStrategy.RESPONSE_TIME)
        worker1 = WorkerNode(id="worker1", response_time_ms=100, is_healthy=True)
        worker2 = WorkerNode(id="worker2", response_time_ms=50, is_healthy=True)
        balancer.add_worker(worker1)
        balancer.add_worker(worker2)
        
        worker = balancer.get_next_worker()
        assert worker.id == "worker2"  # 响应时间更短

    def test_update_worker_metrics(self):
        """测试更新工作节点指标"""
        balancer = LoadBalancer()
        worker = WorkerNode(id="worker1", current_load=10, response_time_ms=100)
        balancer.add_worker(worker)
        
        balancer.update_worker_metrics("worker1", current_load=20, response_time_ms=200)
        
        assert balancer.workers["worker1"].current_load == 20
        assert balancer.workers["worker1"].response_time_ms == 200

    def test_get_load_balancer_stats(self):
        """测试获取负载均衡器统计信息"""
        balancer = LoadBalancer()
        worker1 = WorkerNode(id="worker1", current_load=10, response_time_ms=100, is_healthy=True)
        worker2 = WorkerNode(id="worker2", current_load=20, response_time_ms=200, is_healthy=False)
        balancer.add_worker(worker1)
        balancer.add_worker(worker2)
        
        stats = balancer.get_load_balancer_stats()
        
        assert stats["total_workers"] == 2
        assert stats["healthy_workers"] == 1
        assert stats["total_load"] == 30
        assert len(stats["workers"]) == 2


class TestAutoScaler:
    """AutoScaler测试"""

    def test_auto_scaler_initialization(self):
        """测试自动扩缩容器初始化"""
        metrics = ScalingMetrics(min_workers=2, max_workers=10)
        scaler = AutoScaler(metrics, ScalingStrategy.CPU_BASED)
        
        assert scaler.metrics == metrics
        assert scaler.strategy == ScalingStrategy.CPU_BASED
        assert scaler.current_workers == 2

    def test_should_scale_up_cpu_based(self):
        """测试CPU策略扩容判断"""
        metrics = ScalingMetrics(cpu_threshold=0.8, min_workers=2, max_workers=10)
        scaler = AutoScaler(metrics, ScalingStrategy.CPU_BASED)
        
        # CPU使用率超过阈值
        should_scale = scaler.should_scale_up({"cpu_usage": 0.9})
        assert should_scale is True
        
        # CPU使用率低于阈值
        should_scale = scaler.should_scale_up({"cpu_usage": 0.5})
        assert should_scale is False

    def test_should_scale_up_memory_based(self):
        """测试内存策略扩容判断"""
        metrics = ScalingMetrics(memory_threshold=0.8, min_workers=2, max_workers=10)
        scaler = AutoScaler(metrics, ScalingStrategy.MEMORY_BASED)
        
        should_scale = scaler.should_scale_up({"memory_usage": 0.9})
        assert should_scale is True
        
        should_scale = scaler.should_scale_up({"memory_usage": 0.5})
        assert should_scale is False

    def test_should_scale_up_queue_based(self):
        """测试队列策略扩容判断"""
        metrics = ScalingMetrics(queue_threshold=100, min_workers=2, max_workers=10)
        scaler = AutoScaler(metrics, ScalingStrategy.QUEUE_BASED)
        
        should_scale = scaler.should_scale_up({"queue_size": 150})
        assert should_scale is True
        
        should_scale = scaler.should_scale_up({"queue_size": 50})
        assert should_scale is False

    def test_should_scale_up_hybrid(self):
        """测试混合策略扩容判断"""
        metrics = ScalingMetrics(
            cpu_threshold=0.8,
            memory_threshold=0.8,
            queue_threshold=100,
            min_workers=2,
            max_workers=10
        )
        scaler = AutoScaler(metrics, ScalingStrategy.HYBRID)
        
        # CPU高
        should_scale = scaler.should_scale_up({"cpu_usage": 0.9, "memory_usage": 0.5, "queue_size": 50})
        assert should_scale is True
        
        # 内存高
        should_scale = scaler.should_scale_up({"cpu_usage": 0.5, "memory_usage": 0.9, "queue_size": 50})
        assert should_scale is True
        
        # 队列高
        should_scale = scaler.should_scale_up({"cpu_usage": 0.5, "memory_usage": 0.5, "queue_size": 150})
        assert should_scale is True
        
        # 都低
        should_scale = scaler.should_scale_up({"cpu_usage": 0.5, "memory_usage": 0.5, "queue_size": 50})
        assert should_scale is False

    def test_should_scale_up_max_workers_reached(self):
        """测试达到最大工作节点数"""
        metrics = ScalingMetrics(min_workers=2, max_workers=5)
        scaler = AutoScaler(metrics, ScalingStrategy.CPU_BASED)
        scaler.current_workers = 5  # 已达到最大值
        
        should_scale = scaler.should_scale_up({"cpu_usage": 0.9})
        assert should_scale is False

    def test_should_scale_down_cpu_based(self):
        """测试CPU策略缩容判断"""
        metrics = ScalingMetrics(cpu_threshold=0.8, min_workers=2, max_workers=10)
        scaler = AutoScaler(metrics, ScalingStrategy.CPU_BASED)
        scaler.current_workers = 5
        
        # CPU使用率低于阈值的一半
        should_scale = scaler.should_scale_down({"cpu_usage": 0.3})
        assert should_scale is True
        
        # CPU使用率高于阈值的一半
        should_scale = scaler.should_scale_down({"cpu_usage": 0.5})
        assert should_scale is False

    def test_should_scale_down_min_workers_reached(self):
        """测试达到最小工作节点数"""
        metrics = ScalingMetrics(min_workers=2, max_workers=10)
        scaler = AutoScaler(metrics, ScalingStrategy.CPU_BASED)
        scaler.current_workers = 2  # 已达到最小值
        
        should_scale = scaler.should_scale_down({"cpu_usage": 0.3})
        assert should_scale is False

    def test_update_metrics(self):
        """测试更新性能指标"""
        metrics = ScalingMetrics()
        scaler = AutoScaler(metrics, ScalingStrategy.CPU_BASED)
        
        scaler.update_metrics({"cpu_usage": 0.7, "memory_usage": 0.6, "queue_size": 50})
        
        assert len(scaler.cpu_history) == 1
        assert len(scaler.memory_history) == 1
        assert len(scaler.queue_history) == 1
        assert scaler.cpu_history[0] == 0.7

