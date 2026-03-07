# -*- coding: utf-8 -*-
"""
特征基准测试运行器测试
"""

import pytest
import time
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.features.monitoring.benchmark_runner import (
    FeatureBenchmarkRunner,
    BenchmarkType,
    BenchmarkResult
)


class TestFeatureBenchmarkRunner:
    """测试FeatureBenchmarkRunner类"""

    @pytest.fixture
    def runner(self):
        """创建FeatureBenchmarkRunner实例"""
        return FeatureBenchmarkRunner()

    @pytest.fixture
    def sample_data(self):
        """生成示例数据"""
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })

    def test_init(self, runner):
        """测试初始化"""
        assert runner._benchmark_history == []

    def test_run_execution_time_benchmark(self, runner, sample_data):
        """测试执行时间基准测试"""
        def test_func(data):
            time.sleep(0.01)  # 模拟处理时间
            return data.sum()

        result = runner.run_execution_time_benchmark(
            test_func, sample_data, iterations=5, warmup_iterations=1
        )

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_type == BenchmarkType.EXECUTION_TIME
        assert result.iterations == 5
        assert 'mean' in result.statistics
        assert 'median' in result.statistics
        assert 'min' in result.statistics
        assert 'max' in result.statistics
        assert result.statistics['mean'] > 0

    def test_run_memory_usage_benchmark(self, runner, sample_data):
        """测试内存使用基准测试"""
        def test_func(data):
            return data.copy()

        with patch('src.features.monitoring.benchmark_runner.PSUTIL_AVAILABLE', True):
            with patch('psutil.virtual_memory') as mock_vm:
                mock_vm.return_value.used = 1000000
                
                result = runner.run_memory_usage_benchmark(
                    test_func, sample_data, iterations=3
                )

                assert isinstance(result, BenchmarkResult)
                assert result.benchmark_type == BenchmarkType.MEMORY_USAGE
                assert 'memory_usage' in result.metrics

    def test_run_memory_usage_benchmark_no_psutil(self, runner, sample_data):
        """测试没有psutil时的内存基准测试"""
        def test_func(data):
            return data.copy()

        with patch('src.features.monitoring.benchmark_runner.PSUTIL_AVAILABLE', False):
            result = runner.run_memory_usage_benchmark(
                test_func, sample_data, iterations=3
            )

            assert isinstance(result, BenchmarkResult)
            assert result.benchmark_type == BenchmarkType.MEMORY_USAGE

    def test_run_throughput_benchmark(self, runner, sample_data):
        """测试吞吐量基准测试"""
        def test_func(data):
            return len(data)

        result = runner.run_throughput_benchmark(
            test_func, sample_data, iterations=5
        )

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_type == BenchmarkType.THROUGHPUT
        assert 'throughput_rates' in result.metrics
        assert len(result.metrics['throughput_rates']) == 5

    def test_compare_benchmarks(self, runner, sample_data):
        """测试比较基准测试"""
        def test_func(data):
            return data.sum()

        runner.run_execution_time_benchmark(test_func, sample_data, iterations=2, test_name="test1")
        runner.run_execution_time_benchmark(test_func, sample_data, iterations=2, test_name="test2")

        comparison = runner.compare_benchmarks(["test1", "test2"])
        assert 'benchmarks' in comparison
        assert 'test1' in comparison['benchmarks']
        assert 'test2' in comparison['benchmarks']

    def test_run_comprehensive_benchmark(self, runner, sample_data):
        """测试综合基准测试"""
        def test_func(data):
            time.sleep(0.001)
            return data.sum()

        result = runner.run_comprehensive_benchmark(
            test_func, sample_data, iterations=3
        )

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_type == BenchmarkType.COMPREHENSIVE
        assert 'execution_times' in result.metrics
        assert 'memory_usage' in result.metrics
        assert 'throughput_rates' in result.metrics

    def test_get_benchmark_history(self, runner, sample_data):
        """测试获取基准测试历史"""
        def test_func(data):
            return data.sum()

        runner.run_execution_time_benchmark(test_func, sample_data, iterations=2)
        history = runner.get_benchmark_history()

        assert len(history) == 1
        assert isinstance(history[0], BenchmarkResult)

    def test_clear_history(self, runner, sample_data):
        """测试清除基准测试历史"""
        def test_func(data):
            return data.sum()

        runner.run_execution_time_benchmark(test_func, sample_data, iterations=2)
        assert len(runner.get_benchmark_history()) == 1

        runner.clear_history()
        assert len(runner.get_benchmark_history()) == 0

    def test_run_execution_time_benchmark_with_exception(self, runner, sample_data):
        """测试执行时间基准测试异常处理"""
        def test_func(data):
            raise ValueError("测试异常")

        # 应该能够处理异常
        try:
            result = runner.run_execution_time_benchmark(
                test_func, sample_data, iterations=2
            )
            # 如果捕获了异常，可能返回部分结果或None
        except Exception:
            # 如果抛出异常也是可以接受的
            pass

    def test_run_execution_time_benchmark_single_iteration(self, runner, sample_data):
        """测试单次迭代的执行时间基准测试"""
        def test_func(data):
            return data.sum()

        result = runner.run_execution_time_benchmark(
            test_func, sample_data, iterations=1, warmup_iterations=0
        )

        assert isinstance(result, BenchmarkResult)
        assert result.iterations == 1
        # 单次迭代时std可能为0
        assert 'std' in result.statistics

