#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征基准测试运行器

from src.infrastructure.logging.core.unified_logger import get_unified_logger
提供特征处理性能的基准测试功能。
"""

import time
import statistics
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# 修复导入问题
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger('__name__')
except ImportError:
    # 如果无法导入，使用标准日志
    import logging
    logger = logging.getLogger(__name__)

# 尝试导入psutil，如果失败则提供fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil模块不可用，内存基准测试将使用模拟数据")


class BenchmarkType(Enum):

    """基准测试类型"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    COMPREHENSIVE = "comprehensive"


@dataclass
class BenchmarkResult:

    """基准测试结果"""
    benchmark_type: BenchmarkType
    test_name: str
    timestamp: datetime
    iterations: int
    metrics: Dict[str, float]
    statistics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureBenchmarkRunner:

    """特征基准测试运行器"""

    def __init__(self):
        """初始化基准测试运行器"""
        self._benchmark_history: List[BenchmarkResult] = []

    def run_execution_time_benchmark(self,


                                     test_func: Callable,
                                     test_data: Any,
                                     iterations: int = 10,
                                     warmup_iterations: int = 3,
                                     test_name: str = "execution_time_test") -> BenchmarkResult:
        """运行执行时间基准测试"""
        logger.info(f"开始执行时间基准测试: {test_name}")

        # 预热
        for _ in range(warmup_iterations):
            test_func(test_data)

        # 执行测试
        execution_times = []
        for i in range(iterations):
            start_time = time.time()
            result = test_func(test_data)
            end_time = time.time()
            execution_times.append(end_time - start_time)

            if (i + 1) % 5 == 0:
                logger.info(f"完成 {i + 1}/{iterations} 次测试")

        # 计算统计信息
        stats = {
            "mean": statistics.mean(execution_times),
            "median": statistics.median(execution_times),
            "min": min(execution_times),
            "max": max(execution_times),
            "std": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        }

        result = BenchmarkResult(
            benchmark_type=BenchmarkType.EXECUTION_TIME,
            test_name=test_name,
            timestamp=datetime.now(),
            iterations=iterations,
            metrics={"execution_times": execution_times},
            statistics=stats,
            metadata={"warmup_iterations": warmup_iterations}
        )

        self._benchmark_history.append(result)
        logger.info(f"执行时间基准测试完成: {test_name}, 平均时间: {stats['mean']:.4f}秒")

        return result

    def run_memory_usage_benchmark(self,


                                   test_func: Callable,
                                   test_data: Any,
                                   iterations: int = 5,
                                   test_name: str = "memory_usage_test") -> BenchmarkResult:
        """运行内存使用基准测试"""
        logger.info(f"开始内存使用基准测试: {test_name}")

        import gc

        memory_usage = []

        for i in range(iterations):
            # 强制垃圾回收
            gc.collect()

            # 记录初始内存
            initial_memory = psutil.virtual_memory().used if PSUTIL_AVAILABLE else 0

            # 执行测试
            result = test_func(test_data)

            # 记录最终内存
            final_memory = psutil.virtual_memory().used if PSUTIL_AVAILABLE else 0
            memory_delta = final_memory - initial_memory

            memory_usage.append(memory_delta)

            if (i + 1) % 2 == 0:
                logger.info(f"完成 {i + 1}/{iterations} 次测试")

        # 计算统计信息
        stats = {
            "mean": statistics.mean(memory_usage),
            "median": statistics.median(memory_usage),
            "min": min(memory_usage),
            "max": max(memory_usage),
            "std": statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0
        }

        result = BenchmarkResult(
            benchmark_type=BenchmarkType.MEMORY_USAGE,
            test_name=test_name,
            timestamp=datetime.now(),
            iterations=iterations,
            metrics={"memory_usage": memory_usage},
            statistics=stats
        )

        self._benchmark_history.append(result)
        logger.info(f"内存使用基准测试完成: {test_name}, 平均内存增量: {stats['mean']:.2f} bytes")

        return result

    def run_throughput_benchmark(self,


                                 test_func: Callable,
                                 test_data: Any,
                                 iterations: int = 10,
                                 test_name: str = "throughput_test") -> BenchmarkResult:
        """运行吞吐量基准测试"""
        logger.info(f"开始吞吐量基准测试: {test_name}")

        throughput_rates = []

        for i in range(iterations):
            start_time = time.time()
            result = test_func(test_data)
            end_time = time.time()

            # 计算吞吐量（操作 / 秒）
            execution_time = end_time - start_time
            throughput = 1.0 / execution_time if execution_time > 0 else 0
            throughput_rates.append(throughput)

            if (i + 1) % 5 == 0:
                logger.info(f"完成 {i + 1}/{iterations} 次测试")

        # 计算统计信息
        stats = {
            "mean": statistics.mean(throughput_rates),
            "median": statistics.median(throughput_rates),
            "min": min(throughput_rates),
            "max": max(throughput_rates),
            "std": statistics.stdev(throughput_rates) if len(throughput_rates) > 1 else 0
        }

        result = BenchmarkResult(
            benchmark_type=BenchmarkType.THROUGHPUT,
            test_name=test_name,
            timestamp=datetime.now(),
            iterations=iterations,
            metrics={"throughput_rates": throughput_rates},
            statistics=stats
        )

        self._benchmark_history.append(result)
        logger.info(f"吞吐量基准测试完成: {test_name}, 平均吞吐量: {stats['mean']:.2f} ops / sec")

        return result

    def run_comprehensive_benchmark(self,


                                    test_func: Callable,
                                    test_data: Any,
                                    iterations: int = 10,
                                    test_name: str = "comprehensive_test") -> BenchmarkResult:
        """运行综合基准测试"""
        logger.info(f"开始综合基准测试: {test_name}")

        # 运行各种基准测试
        execution_result = self.run_execution_time_benchmark(
            test_func, test_data, iterations, test_name=f"{test_name}_execution"
        )

        memory_result = self.run_memory_usage_benchmark(
            test_func, test_data, iterations // 2, test_name=f"{test_name}_memory"
        )

        throughput_result = self.run_throughput_benchmark(
            test_func, test_data, iterations, test_name=f"{test_name}_throughput"
        )

        # 合并结果
        combined_metrics = {
            "execution_times": execution_result.metrics["execution_times"],
            "memory_usage": memory_result.metrics["memory_usage"],
            "throughput_rates": throughput_result.metrics["throughput_rates"]
        }

        combined_stats = {
            "execution": execution_result.statistics,
            "memory": memory_result.statistics,
            "throughput": throughput_result.statistics
        }

        result = BenchmarkResult(
            benchmark_type=BenchmarkType.COMPREHENSIVE,
            test_name=test_name,
            timestamp=datetime.now(),
            iterations=iterations,
            metrics=combined_metrics,
            statistics=combined_stats
        )

        self._benchmark_history.append(result)
        logger.info(f"综合基准测试完成: {test_name}")

        return result

    def compare_benchmarks(self,


                           benchmark_names: List[str],
                           metric: str = "mean") -> Dict[str, Any]:
        """比较多个基准测试结果"""
        results = {}

        for name in benchmark_names:
            benchmark_results = [r for r in self._benchmark_history if r.test_name == name]
            if benchmark_results:
                latest_result = benchmark_results[-1]
                results[name] = {
                    "statistics": latest_result.statistics,
                    "timestamp": latest_result.timestamp,
                    "iterations": latest_result.iterations
                }

        # 生成比较报告
        comparison = {
            "benchmarks": results,
            "comparison_metric": metric,
            "summary": {}
        }

        if results:
            # 按指定指标排序
            sorted_results = sorted(
                results.items(),
                key=lambda x: x[1]["statistics"].get(metric, 0),
                reverse=True
            )

            comparison["summary"] = {
                "best_performer": sorted_results[0][0] if sorted_results else None,
                "worst_performer": sorted_results[-1][0] if sorted_results else None,
                "performance_ranking": [name for name, _ in sorted_results]
            }

        return comparison

    def generate_benchmark_report(self,


                                  benchmark_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """生成基准测试报告"""
        if benchmark_names:
            benchmarks = [r for r in self._benchmark_history if r.test_name in benchmark_names]
        else:
            benchmarks = self._benchmark_history.copy()

        report = {
            "timestamp": datetime.now(),
            "total_benchmarks": len(benchmarks),
            "benchmarks": {}
        }

        for benchmark in benchmarks:
            report["benchmarks"][benchmark.test_name] = {
                "type": benchmark.benchmark_type.value,
                "timestamp": benchmark.timestamp,
                "iterations": benchmark.iterations,
                "statistics": benchmark.statistics,
                "metadata": benchmark.metadata
            }

        return report

    def get_benchmark_history(self,


                              benchmark_type: Optional[BenchmarkType] = None,
                              test_name: Optional[str] = None,
                              limit: Optional[int] = None) -> List[BenchmarkResult]:
        """获取基准测试历史"""
        history = self._benchmark_history.copy()

        if benchmark_type:
            history = [r for r in history if r.benchmark_type == benchmark_type]

        if test_name:
            history = [r for r in history if r.test_name == test_name]

        if limit:
            history = history[-limit:]

        return history

    def clear_history(self) -> None:
        """清空基准测试历史"""
        self._benchmark_history.clear()


# 全局基准测试运行器实例
_benchmark_runner = FeatureBenchmarkRunner()


def get_benchmark_runner() -> FeatureBenchmarkRunner:
    """获取全局基准测试运行器"""
    return _benchmark_runner


def run_benchmark(benchmark_type: BenchmarkType,


                  test_func: Callable,
                  test_data: Any,
                  iterations: int = 10,
                  test_name: str = "benchmark_test") -> BenchmarkResult:
    """运行基准测试的便捷函数"""
    runner = get_benchmark_runner()

    if benchmark_type == BenchmarkType.EXECUTION_TIME:
        return runner.run_execution_time_benchmark(test_func, test_data, iterations, test_name=test_name)
    elif benchmark_type == BenchmarkType.MEMORY_USAGE:
        return runner.run_memory_usage_benchmark(test_func, test_data, iterations, test_name=test_name)
    elif benchmark_type == BenchmarkType.THROUGHPUT:
        return runner.run_throughput_benchmark(test_func, test_data, iterations, test_name=test_name)
    elif benchmark_type == BenchmarkType.COMPREHENSIVE:
        return runner.run_comprehensive_benchmark(test_func, test_data, iterations, test_name=test_name)
    else:
        raise ValueError(f"不支持的基准测试类型: {benchmark_type}")
