#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层性能基准框架组件测试

测试目标：提升utils/optimization/benchmark_framework.py的真实覆盖率
实际导入和使用src.infrastructure.utils.optimization.benchmark_framework模块
"""

import pytest
import time
from unittest.mock import Mock, patch


class TestBenchmarkConstants:
    """测试基准测试常量类"""
    
    def test_constants(self):
        """测试常量值"""
        from src.infrastructure.utils.optimization.benchmark_framework import BenchmarkConstants
        
        assert BenchmarkConstants.DEFAULT_ITERATIONS == 1
        assert BenchmarkConstants.DEFAULT_WARMUP_ITERATIONS == 3
        assert BenchmarkConstants.DEFAULT_MIN_ITERATIONS == 10
        assert BenchmarkConstants.DEFAULT_MAX_ITERATIONS == 1000
        assert BenchmarkConstants.DEFAULT_TARGET_DURATION == 1.0
        assert BenchmarkConstants.DEFAULT_THRESHOLD_PERCENTAGE == 10.0
        assert BenchmarkConstants.CONFIDENCE_LEVEL == 0.95


class TestBenchmarkResult:
    """测试基准测试结果类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.optimization.benchmark_framework import BenchmarkResult
        
        result = BenchmarkResult(
            test_name="test",
            test_category="category",
            execution_time=1.0,
            operations_per_second=100.0,
            memory_usage=1024.0,
            cpu_usage=50.0,
            timestamp=time.time()
        )
        
        assert result.test_name == "test"
        assert result.test_category == "category"
        assert result.execution_time == 1.0
        assert result.operations_per_second == 100.0
    
    def test_init_with_defaults(self):
        """测试使用默认值初始化"""
        from src.infrastructure.utils.optimization.benchmark_framework import BenchmarkResult
        
        result = BenchmarkResult(
            test_name="test",
            test_category="category",
            execution_time=1.0,
            operations_per_second=100.0,
            memory_usage=1024.0,
            cpu_usage=50.0,
            timestamp=time.time()
        )
        
        assert result.iterations == 1
        assert result.warmup_time == 0.0
        assert isinstance(result.metadata, dict)


class TestPerformanceBaseline:
    """测试性能基准线类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.optimization.benchmark_framework import PerformanceBaseline
        
        baseline = PerformanceBaseline(
            test_name="test",
            test_category="category",
            baseline_execution_time=1.0,
            baseline_operations_per_second=100.0,
            baseline_memory_usage=1024.0,
            baseline_cpu_usage=50.0
        )
        
        assert baseline.test_name == "test"
        assert baseline.baseline_execution_time == 1.0
        assert baseline.threshold_percentage == 10.0
    
    def test_is_within_threshold(self):
        """测试检查是否在阈值内"""
        from src.infrastructure.utils.optimization.benchmark_framework import PerformanceBaseline
        
        baseline = PerformanceBaseline(
            test_name="test",
            test_category="category",
            baseline_execution_time=1.0,
            baseline_operations_per_second=100.0,
            baseline_memory_usage=1024.0,
            baseline_cpu_usage=50.0,
            threshold_percentage=10.0
        )
        
        # 在阈值内（1.0 ± 10% = 0.9-1.1）
        assert baseline.is_within_threshold(1.05) is True
        
        # 超出阈值
        assert baseline.is_within_threshold(1.2) is False
    
    def test_is_within_threshold_zero_baseline(self):
        """测试零基准时间"""
        from src.infrastructure.utils.optimization.benchmark_framework import PerformanceBaseline
        
        baseline = PerformanceBaseline(
            test_name="test",
            test_category="category",
            baseline_execution_time=0.0,
            baseline_operations_per_second=100.0,
            baseline_memory_usage=1024.0,
            baseline_cpu_usage=50.0
        )
        
        assert baseline.is_within_threshold(1.0) is True
    
    def test_to_dict(self):
        """测试转换为字典"""
        from src.infrastructure.utils.optimization.benchmark_framework import PerformanceBaseline
        
        baseline = PerformanceBaseline(
            test_name="test",
            test_category="category",
            baseline_execution_time=1.0,
            baseline_operations_per_second=100.0,
            baseline_memory_usage=1024.0,
            baseline_cpu_usage=50.0
        )
        
        result = baseline.to_dict()
        
        assert isinstance(result, dict)
        assert result["test_name"] == "test"
        assert result["baseline_execution_time"] == 1.0


class TestBenchmarkRunner:
    """测试基准测试运行器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.optimization.benchmark_framework import BenchmarkRunner
        
        runner = BenchmarkRunner()
        
        assert runner.warmup_iterations == 3
        assert runner.min_iterations == 10
        assert runner.max_iterations == 1000
        assert runner.target_duration == 1.0
        assert isinstance(runner.results, list)
    
    def test_init_custom(self):
        """测试使用自定义参数初始化"""
        from src.infrastructure.utils.optimization.benchmark_framework import BenchmarkRunner
        
        runner = BenchmarkRunner(
            warmup_iterations=5,
            min_iterations=20,
            max_iterations=2000,
            target_duration=2.0
        )
        
        assert runner.warmup_iterations == 5
        assert runner.min_iterations == 20
        assert runner.max_iterations == 2000
        assert runner.target_duration == 2.0
    
    def test_run_benchmark(self):
        """测试运行基准测试"""
        from src.infrastructure.utils.optimization.benchmark_framework import BenchmarkRunner
        
        runner = BenchmarkRunner(
            warmup_iterations=1,
            min_iterations=2,
            max_iterations=10,
            target_duration=0.1
        )
        
        def test_func():
            time.sleep(0.01)
            return True
        
        result = runner.run_benchmark(test_func, "test_benchmark", "test_category")
        
        assert result.test_name == "test_benchmark"
        assert result.test_category == "test_category"
        assert result.execution_time > 0
        assert result.operations_per_second > 0
        assert len(runner.results) == 1
    
    def test_determine_iterations(self):
        """测试确定迭代次数"""
        from src.infrastructure.utils.optimization.benchmark_framework import BenchmarkRunner
        
        runner = BenchmarkRunner(
            min_iterations=5,
            max_iterations=100,
            target_duration=0.1
        )
        
        def fast_func():
            time.sleep(0.001)
        
        iterations = runner._determine_iterations(fast_func)
        
        assert iterations >= runner.min_iterations
        assert iterations <= runner.max_iterations
    
    def test_get_memory_usage(self):
        """测试获取内存使用量"""
        from src.infrastructure.utils.optimization.benchmark_framework import BenchmarkRunner
        
        runner = BenchmarkRunner()
        memory = runner._get_memory_usage()
        
        # 内存使用量可能是int或float
        assert isinstance(memory, (int, float))
        assert memory >= 0
    
    def test_get_cpu_usage(self):
        """测试获取CPU使用量"""
        from src.infrastructure.utils.optimization.benchmark_framework import BenchmarkRunner
        
        runner = BenchmarkRunner()
        cpu = runner._get_cpu_usage()
        
        assert isinstance(cpu, float)
        assert cpu >= 0
    
    def test_run_concurrent_benchmark(self):
        """测试运行并发基准测试"""
        from src.infrastructure.utils.optimization.benchmark_framework import BenchmarkRunner
        
        runner = BenchmarkRunner(
            warmup_iterations=1,
            min_iterations=2,
            max_iterations=10,
            target_duration=0.1
        )
        
        def test_func():
            time.sleep(0.01)
            return True
        
        results = runner.run_concurrent_benchmark(
            test_func,
            "concurrent_test",
            num_threads=2
        )
        
        assert len(results) == 2
        assert all(r.test_name.startswith("concurrent_test_worker_") for r in results)
    
    def test_run_stress_test(self):
        """测试运行压力测试"""
        from src.infrastructure.utils.optimization.benchmark_framework import BenchmarkRunner
        
        runner = BenchmarkRunner()
        
        def test_func():
            time.sleep(0.01)
            return True
        
        result = runner.run_stress_test(
            test_func,
            "stress_test",
            duration=0.1
        )
        
        assert result.test_name == "stress_test"
        assert result.test_category == "stress"
        assert result.iterations > 0

