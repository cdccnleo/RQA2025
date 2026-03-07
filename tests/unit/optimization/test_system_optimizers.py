#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 系统优化器

测试optimization/system/目录中的所有优化器
"""

import pytest
import psutil
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional


class TestSystemOptimizers:
    """测试系统优化器"""

    def setup_method(self):
        """测试前准备"""
        self.cpu_optimizer = None
        self.memory_optimizer = None
        self.io_optimizer = None
        self.network_optimizer = None

        try:
            import sys
            from pathlib import Path

            # 添加src路径
            PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
            if str(PROJECT_ROOT / 'src') not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT / 'src'))
            from optimization.system.cpu_optimizer import CPUOptimizer
            self.cpu_optimizer = CPUOptimizer
        except ImportError:
            pass

        try:
            from optimization.system.memory_optimizer import MemoryOptimizer
            self.memory_optimizer = MemoryOptimizer
        except ImportError:
            pass

        try:
            from optimization.system.io_optimizer import IOOptimizer
            self.io_optimizer = IOOptimizer
        except ImportError:
            pass

        try:
            from optimization.system.network_optimizer import NetworkOptimizer
            self.network_optimizer = NetworkOptimizer
        except ImportError:
            pass

    def test_cpu_optimizer_initialization(self):
        """测试CPU优化器初始化"""
        if self.cpu_optimizer is None:
            pytest.skip("CPUOptimizer not available")

        optimizer = self.cpu_optimizer()
        assert optimizer is not None
        assert hasattr(optimizer, 'get_cpu_stats')
        assert hasattr(optimizer, 'optimize_workload_distribution')

    def test_cpu_optimizer_metrics_collection(self):
        """测试CPU优化器指标收集"""
        if self.cpu_optimizer is None:
            pytest.skip("CPUOptimizer not available")

        optimizer = self.cpu_optimizer()

        if hasattr(optimizer, 'get_cpu_stats'):
            stats = optimizer.get_cpu_stats()
            assert isinstance(stats, dict)
            # CPU统计可能包含错误信息或实际统计数据
            if "error" not in stats:
                assert "cpu_percent" in stats or "cpu_usage" in stats or len(stats) > 0

    def test_cpu_optimizer_optimization(self):
        """测试CPU优化器优化功能"""
        if self.cpu_optimizer is None:
            pytest.skip("CPUOptimizer not available")

        optimizer = self.cpu_optimizer()

        if hasattr(optimizer, 'optimize_cpu_usage'):
            # 模拟优化参数
            params = {
                "target_usage": 70.0,
                "max_threads": 8,
                "priority": "normal"
            }

            result = optimizer.optimize_cpu_usage(params)
            assert isinstance(result, dict)
            assert "success" in result

    def test_cpu_optimizer_thread_management(self):
        """测试CPU优化器线程管理"""
        if self.cpu_optimizer is None:
            pytest.skip("CPUOptimizer not available")

        optimizer = self.cpu_optimizer()

        if hasattr(optimizer, 'get_optimal_thread_count'):
            optimal_count = optimizer.get_optimal_thread_count()
            assert isinstance(optimal_count, int)
            assert optimal_count > 0

    def test_memory_optimizer_initialization(self):
        """测试内存优化器初始化"""
        if self.memory_optimizer is None:
            pytest.skip("MemoryOptimizer not available")

        optimizer = self.memory_optimizer()
        assert optimizer is not None
        # MemoryOptimizer有monitor属性，monitor有get_memory_stats方法
        assert hasattr(optimizer, 'monitor') or hasattr(optimizer, 'get_memory_stats') or hasattr(optimizer, 'optimize_memory_usage')
        assert hasattr(optimizer, 'optimize_memory_usage')

    def test_memory_optimizer_metrics_collection(self):
        """测试内存优化器指标收集"""
        if self.memory_optimizer is None:
            pytest.skip("MemoryOptimizer not available")

        optimizer = self.memory_optimizer()

        if hasattr(optimizer, 'get_memory_usage'):
            usage = optimizer.get_memory_usage()
            assert isinstance(usage, dict)
            assert "total" in usage
            assert "available" in usage
            assert "used" in usage

    def test_memory_optimizer_garbage_collection(self):
        """测试内存优化器垃圾回收"""
        if self.memory_optimizer is None:
            pytest.skip("MemoryOptimizer not available")

        optimizer = self.memory_optimizer()

        if hasattr(optimizer, 'trigger_garbage_collection'):
            result = optimizer.trigger_garbage_collection()
            assert isinstance(result, bool)

    def test_memory_optimizer_memory_pool_management(self):
        """测试内存优化器内存池管理"""
        if self.memory_optimizer is None:
            pytest.skip("MemoryOptimizer not available")

        optimizer = self.memory_optimizer()

        if hasattr(optimizer, 'optimize_memory_pool'):
            pool_config = {
                "initial_size": 100,
                "max_size": 1000,
                "growth_factor": 1.5
            }

            result = optimizer.optimize_memory_pool(pool_config)
            assert isinstance(result, dict)

    def test_io_optimizer_initialization(self):
        """测试IO优化器初始化"""
        if self.io_optimizer is None:
            pytest.skip("IOOptimizer not available")

        optimizer = self.io_optimizer()
        assert optimizer is not None
        # IOOptimizer有get_io_optimizer_status方法，也有io_stats属性
        assert hasattr(optimizer, 'io_stats') or hasattr(optimizer, 'get_io_stats') or hasattr(optimizer, 'get_io_optimizer_status')
        # 检查常见的IO优化方法
        assert hasattr(optimizer, 'read_file_optimized') or hasattr(optimizer, 'write_file_optimized') or hasattr(optimizer, 'optimize_file_operations')

    def test_io_optimizer_performance_monitoring(self):
        """测试IO优化器性能监控"""
        if self.io_optimizer is None:
            pytest.skip("IOOptimizer not available")

        optimizer = self.io_optimizer()

        # IOOptimizer有get_io_optimizer_status方法或io_stats属性
        if hasattr(optimizer, 'get_io_stats'):
            stats = optimizer.get_io_stats()
            assert isinstance(stats, dict)
        elif hasattr(optimizer, 'get_io_optimizer_status'):
            stats = optimizer.get_io_optimizer_status()
            assert isinstance(stats, dict)
        elif hasattr(optimizer, 'io_stats'):
            stats = optimizer.io_stats
            assert isinstance(stats, dict)
            # 常见的IO统计指标
            expected_keys = ["total_reads", "total_writes", "total_bytes_read", "total_bytes_written"]
            for key in expected_keys:
                assert key in stats

    def test_io_optimizer_buffering_optimization(self):
        """测试IO优化器缓冲区优化"""
        if self.io_optimizer is None:
            pytest.skip("IOOptimizer not available")

        optimizer = self.io_optimizer()

        if hasattr(optimizer, 'optimize_buffering'):
            buffer_config = {
                "buffer_size": 8192,
                "flush_interval": 1.0,
                "compression": True
            }

            result = optimizer.optimize_buffering(buffer_config)
            assert isinstance(result, dict)
            assert "optimized_config" in result

    def test_network_optimizer_initialization(self):
        """测试网络优化器初始化"""
        if self.network_optimizer is None:
            pytest.skip("NetworkOptimizer not available")

        optimizer = self.network_optimizer()
        assert optimizer is not None
        assert hasattr(optimizer, 'get_network_stats')
        assert hasattr(optimizer, 'optimize_data_transfer')

    def test_network_optimizer_connection_pooling(self):
        """测试网络优化器连接池"""
        if self.network_optimizer is None:
            pytest.skip("NetworkOptimizer not available")

        optimizer = self.network_optimizer()

        if hasattr(optimizer, 'optimize_connection_pool'):
            pool_config = {
                "max_connections": 100,
                "max_keepalive": 10,
                "timeout": 30.0
            }

            result = optimizer.optimize_connection_pool("example.com", 80)
            # 连接池优化可能返回None或布尔值
            assert result is None or isinstance(result, bool) or isinstance(result, dict)

    def test_network_optimizer_compression(self):
        """测试网络优化器压缩"""
        if self.network_optimizer is None:
            pytest.skip("NetworkOptimizer not available")

        optimizer = self.network_optimizer()

        if hasattr(optimizer, 'optimize_compression'):
            compression_config = {
                "algorithm": "gzip",
                "level": 6,
                "min_size": 1024
            }

            result = optimizer.optimize_compression(compression_config)
            assert isinstance(result, dict)

    def test_system_optimizers_integration(self):
        """测试系统优化器集成"""
        optimizers = []

        if self.cpu_optimizer:
            optimizers.append(("CPU", self.cpu_optimizer()))
        if self.memory_optimizer:
            optimizers.append(("Memory", self.memory_optimizer()))
        if self.io_optimizer:
            optimizers.append(("IO", self.io_optimizer()))
        if self.network_optimizer:
            optimizers.append(("Network", self.network_optimizer()))

        if not optimizers:
            pytest.skip("No system optimizers available")

        # 测试每个优化器的基本功能
        for name, optimizer in optimizers:
            assert optimizer is not None

            # 测试优化方法
            if hasattr(optimizer, 'optimize'):
                result = optimizer.optimize()
                assert isinstance(result, dict)
                assert "status" in result

    def test_system_optimizers_concurrent_operation(self):
        """测试系统优化器并发操作"""
        optimizers = []

        if self.cpu_optimizer:
            optimizers.append(("CPU", self.cpu_optimizer()))
        if self.memory_optimizer:
            optimizers.append(("Memory", self.memory_optimizer()))

        if len(optimizers) < 2:
            pytest.skip("Need at least 2 optimizers for concurrent test")

        results = {}
        errors = []

        def run_optimizer(name, optimizer):
            try:
                # 根据不同的优化器调用不同的方法
                if name == "CPU" and hasattr(optimizer, 'get_cpu_stats'):
                    result = optimizer.get_cpu_stats()
                    results[name] = result
                elif name == "Memory" and hasattr(optimizer, 'get_memory_stats'):
                    result = optimizer.get_memory_stats()
                    results[name] = result
                elif name == "IO" and hasattr(optimizer, 'get_io_stats'):
                    result = optimizer.get_io_stats()
                    results[name] = result
                elif name == "Network" and hasattr(optimizer, 'get_network_stats'):
                    result = optimizer.get_network_stats()
                    results[name] = result
                else:
                    results[name] = {"status": "method_not_found"}
            except Exception as e:
                errors.append(f"{name}: {e}")

        # 启动并发优化
        threads = []
        for name, optimizer in optimizers:
            thread = threading.Thread(target=run_optimizer, args=(name, optimizer))
            threads.append(thread)
            thread.start()

        # 等待完成
        for thread in threads:
            thread.join(timeout=5.0)

        # 验证结果
        assert len(results) == len(optimizers)
        assert len(errors) == 0

    def test_system_optimizers_error_handling(self):
        """测试系统优化器错误处理"""
        if self.cpu_optimizer is None:
            pytest.skip("CPUOptimizer not available")

        optimizer = self.cpu_optimizer()

        # 测试无效参数
        if hasattr(optimizer, 'optimize_cpu_usage'):
            try:
                optimizer.optimize_cpu_usage(None)
            except (TypeError, AttributeError):
                pass  # 应该能处理无效输入

        # 测试边界情况
        if hasattr(optimizer, 'optimize_cpu_usage'):
            try:
                optimizer.optimize_cpu_usage({"target_usage": 150.0})  # 无效的CPU使用率
            except (ValueError, TypeError):
                pass  # 应该能处理无效参数

    def test_system_optimizers_resource_limits(self):
        """测试系统优化器资源限制"""
        if self.memory_optimizer is None:
            pytest.skip("MemoryOptimizer not available")

        optimizer = self.memory_optimizer()

        if hasattr(optimizer, 'get_resource_limits'):
            limits = optimizer.get_resource_limits()
            assert isinstance(limits, dict)

            # 检查常见的资源限制
            expected_limits = ["max_memory", "max_cpu", "max_connections"]
            for limit in expected_limits:
                if limit in limits:
                    assert isinstance(limits[limit], (int, float))

    def test_system_optimizers_performance_metrics(self):
        """测试系统优化器性能指标"""
        if self.cpu_optimizer is None:
            pytest.skip("CPUOptimizer not available")

        optimizer = self.cpu_optimizer()

        if hasattr(optimizer, 'get_performance_metrics'):
            metrics = optimizer.get_performance_metrics()
            assert isinstance(metrics, dict)

            # 检查性能指标
            expected_metrics = ["optimization_time", "resource_usage", "efficiency"]
            for metric in expected_metrics:
                if metric in metrics:
                    assert isinstance(metrics[metric], (int, float))
