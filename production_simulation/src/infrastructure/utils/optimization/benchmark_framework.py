"""
benchmark_framework 模块

提供 benchmark_framework 相关功能和接口。
"""

import json
import logging

import psutil
# 使用线程池并行处理
import collections
import concurrent.futures
import numpy as np
import psutil
import asyncio
import csv
import multiprocessing
import sqlite3
import statistics
import threading
import time

# from src.infrastructure.config.unified_config_manager import (
from dataclasses import dataclass
from datetime import datetime
from src.infrastructure.cache.manager import MemoryCacheManager
# from src.infrastructure.monitoring.system_monitor import SystemMonitor  # TODO: Fix import
from src.infrastructure.utils.optimization.performance_baseline import PerformanceBaseline
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import asdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
"""
RQA2025 基础设施层工具系统 - 性能基准框架

本模块提供全面的性能基准测试框架，用于评估和比较系统组件的性能表现。
支持多种测试模式，包括单线程、多线程、分布式和并行测试。

主要特性:
- 自动化性能基准测试
- 多维度性能指标收集
- 统计分析和结果报告
- 内存使用和CPU消耗监控
- 并行和分布式测试支持

作者: RQA2025 Team
创建日期: 2025年9月13日
版本: 2.0.0
"""

# 修复导入路径
# try:
# except ImportError:
    # 如果performance_baseline不存在，创建一个简单的基类

@dataclass
class PerformanceBaseline:
        test_name: str
        baseline_time: float
        threshold_percentage: float
        sample_count: int
        created_at: datetime = None

        def __post_init__(self):
            if self.created_at is None:
                self.created_at = datetime.now()

        def is_within_threshold(self, current_time: float) -> bool:
            """检查是否在阈值内"""
            if self.baseline_time == 0:
                return True
            percentage_diff = abs(current_time - self.baseline_time) / self.baseline_time * 100
            return percentage_diff <= self.threshold_percentage

        def to_dict(self) -> Dict[str, Any]:
            """转换为字典"""
            return {
                'test_name': self.test_name,
                'baseline_time': self.baseline_time,
                'threshold_percentage': self.threshold_percentage,
                'sample_count': self.sample_count,
                'created_at': self.created_at.isoformat()
            }

try:
    pass
except ImportError:
    # 如果缓存管理器不存在，创建一个简单的Mock类
    class MemoryCacheManager:
        def __init__(self):
            self.cache = {}

        def get(self, key):
            return self.cache.get(key)

        def set(self, key, value):
            self.cache[key] = value

        def clear(self):
            self.cache.clear()

try:
    pass
except ImportError:
    # 如果系统监控不存在，创建一个简单的Mock类
    class SystemMonitor:
        def get_system_stats(self):
            return {
                'cpu_percent': 50.0,
                'memory_percent': 60.0,
                'disk_usage': 70.0
            }

#!/usr/bin/env python3
try:
    pass
except ImportError:
    psutil = None
# -*- coding: utf-8 -*-

"""
性能基准测试框架
实现自动化性能测试、基准比较和性能回归检测
"""

logger = logging.getLogger(__name__)

# 基准测试框架常量


class BenchmarkConstants:
    """基准测试框架相关常量"""

    # 默认迭代配置
    DEFAULT_ITERATIONS = 1
    DEFAULT_WARMUP_ITERATIONS = 3
    DEFAULT_MIN_ITERATIONS = 10
    DEFAULT_MAX_ITERATIONS = 1000

    # 时间配置 (秒)
    DEFAULT_TARGET_DURATION = 1.0
    DEFAULT_WARMUP_TIME = 0.0
    DEFAULT_CLEANUP_TIME = 0.0

    # 性能阈值配置
    DEFAULT_THRESHOLD_PERCENTAGE = 10.0  # 允许的性能变化百分比

    # 统计配置
    CONFIDENCE_LEVEL = 0.95
    OUTLIER_THRESHOLD = 2.0  # 标准差倍数

    # 数据库配置
    DEFAULT_BATCH_SIZE = 1000
    MAX_CONNECTIONS = 10

    # 文件输出配置
    DEFAULT_CSV_DELIMITER = ","
    DEFAULT_JSON_INDENT = 2

    # 进度报告配置
    PROGRESS_REPORT_INTERVAL = 10  # 每10次迭代报告一次

    # 并行化配置
    DEFAULT_PARALLEL_WORKERS = multiprocessing.cpu_count()
    MAX_PARALLEL_WORKERS = multiprocessing.cpu_count() * 2
    PARALLEL_BATCH_SIZE = 1000  # 并行批处理大小
    PARALLEL_CHUNK_SIZE = 10000  # 并行分块大小


@dataclass
class BenchmarkResult:
    """基准测试结果"""

    test_name: str
    test_category: str
    execution_time: float
    operations_per_second: float
    memory_usage: float
    cpu_usage: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    iterations: int = BenchmarkConstants.DEFAULT_ITERATIONS
    warmup_time: float = BenchmarkConstants.DEFAULT_WARMUP_TIME
    cleanup_time: float = BenchmarkConstants.DEFAULT_CLEANUP_TIME


@dataclass
class PerformanceBaseline:
    """性能基准线"""

    test_name: str
    test_category: str
    baseline_execution_time: float
    baseline_operations_per_second: float
    baseline_memory_usage: float
    baseline_cpu_usage: float
    threshold_percentage: float = BenchmarkConstants.DEFAULT_THRESHOLD_PERCENTAGE  # 允许的性能变化百分比
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def is_within_threshold(self, current_time: float) -> bool:
        """检查执行时间是否在阈值内"""
        if self.baseline_execution_time == 0:
            return True
        percentage_diff = abs(current_time - self.baseline_execution_time) / \
                              self.baseline_execution_time * 100
        return percentage_diff <= self.threshold_percentage

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class BenchmarkRunner:
    """基准测试运行器"""

    def __init__(
        self,
        warmup_iterations: int = BenchmarkConstants.DEFAULT_WARMUP_ITERATIONS,
        min_iterations: int = BenchmarkConstants.DEFAULT_MIN_ITERATIONS,
        max_iterations: int = BenchmarkConstants.DEFAULT_MAX_ITERATIONS,
        target_duration: float = BenchmarkConstants.DEFAULT_TARGET_DURATION,
    ):
        self.warmup_iterations = warmup_iterations
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.target_duration = target_duration
        self.results = []

    def run_benchmark(
        self,
        test_func: Callable,
        test_name: str,
        test_category: str = "general",
        **kwargs,
    ):
        """运行基准测试"""
        logger.info(f"开始运行基准测试: {test_name}")

        # 预热阶段
        warmup_time = self._run_warmup(test_func, **kwargs)

        # 确定迭代次数
        iterations = self._determine_iterations(test_func, **kwargs)

        # 执行测试阶段
        test_metrics = self._execute_test(test_func, iterations, **kwargs)

        # 创建和返回结果
        return self._create_result(test_name, test_category, iterations, warmup_time, test_metrics, kwargs)

    def _run_warmup(self, test_func: Callable, **kwargs) -> float:
        """运行预热阶段"""
        warmup_start = time.time()
        for _ in range(self.warmup_iterations):
            test_func(**kwargs)
        return time.time() - warmup_start

    def _execute_test(self, test_func: Callable, iterations: int, **kwargs) -> Dict[str, float]:
        """执行测试阶段"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()

        for _ in range(iterations):
            test_func(**kwargs)

        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_cpu = self._get_cpu_usage()

        return {
            "execution_time": end_time - start_time,
            "memory_usage": end_memory - start_memory,
            "cpu_usage": end_cpu - start_cpu,
        }

    def _create_result(
        self,
        test_name: str,
        test_category: str,
        iterations: int,
        warmup_time: float,
        test_metrics: Dict[str, float],
        kwargs: dict,
    ) -> BenchmarkResult:
        """创建测试结果"""
        execution_time = test_metrics["execution_time"]
        memory_usage = test_metrics["memory_usage"]
        cpu_usage = test_metrics["cpu_usage"]

        ops_per_second = iterations / execution_time if execution_time > 0 else float("inf")

        result = BenchmarkResult(
            test_name=test_name,
            test_category=test_category,
            execution_time=execution_time,
            operations_per_second=ops_per_second,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            timestamp=time.time(),
            iterations=iterations,
            warmup_time=warmup_time,
            metadata=kwargs,
        )

        self.results.append(result)
        logger.info(f"基准测试完成: {test_name}, 执行时间: {execution_time:.4f}s, OPS: {ops_per_second:.2f}")

        return result

    def _determine_iterations(self, test_func: Callable, **kwargs) -> int:
        """确定迭代次数"""
        # 先运行一次估算时间
        start_time = time.time()
        test_func(**kwargs)
        single_execution_time = time.time() - start_time

        if single_execution_time <= 0:
            return self.min_iterations

        # 计算达到目标持续时间所需的迭代次数
        target_iterations = int(self.target_duration / single_execution_time)

        # 确保在合理范围内
        target_iterations = max(self.min_iterations, min(self.max_iterations, target_iterations))

        return target_iterations

    def _get_memory_usage(self) -> float:
        """获取内存使用量"""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """获取CPU使用量"""
        try:
            if psutil is not None:
                process = psutil.Process()
                return process.cpu_percent()
            else:
                return 0.0
        except Exception:
            return 0.0

    def run_concurrent_benchmark(
        self,
        test_func: Callable,
        test_name: str,
        test_category: str = "concurrent",
        num_threads: int = 4,
        **kwargs,
    ):
        """运行并发基准测试"""
        logger.info(f"开始运行并发基准测试: {test_name}, 线程数: {num_threads}")

        results = []

        def worker(worker_id: int):

            return self.run_benchmark(test_func, f"{test_name}_worker_{worker_id}", test_category, **kwargs)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"并发测试失败: {e}")

        return results

    def run_stress_test(
        self,
        test_func: Callable,
        test_name: str,
        test_category: str = "stress",
        duration: float = 60.0,
        **kwargs,
    ):
        """运行压力测试"""
        logger.info(f"开始运行压力测试: {test_name}, 持续时间: {duration}s")

        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()

        iterations = 0
        while time.time() - start_time < duration:
            test_func(**kwargs)
            iterations += 1

        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_cpu = self._get_cpu_usage()

        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_usage = end_cpu - start_cpu
        ops_per_second = iterations / execution_time if execution_time > 0 else float("inf")

        result = BenchmarkResult(
            test_name=test_name,
            test_category=test_category,
            execution_time=execution_time,
            operations_per_second=ops_per_second,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            timestamp=time.time(),
            iterations=iterations,
            metadata=kwargs,
        )

        self.results.append(result)
        logger.info(f"压力测试完成: {test_name}, 迭代次数: {iterations}, OPS: {ops_per_second:.2f}")

        return result

    def run_parallel_benchmark(
        self,
        test_func: Callable,
        test_name: str,
        test_category: str = "parallel",
        num_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        **kwargs,
    ) -> BenchmarkResult:
        """
        运行并行基准测试 (性能优化版)

        Args:
            test_func: 测试函数
            test_name: 测试名称
            test_category: 测试类别
            num_workers: 并行工作线程数，默认使用CPU核心数
            chunk_size: 数据分块大小
            **kwargs: 测试函数参数

        Returns:
            BenchmarkResult: 基准测试结果
        """
        # 配置参数设置
        num_workers, chunk_size = self._setup_parallel_config(num_workers, chunk_size)

        logger.info(f"开始运行并行基准测试: {test_name} (工作线程: {num_workers})")

        # 预热阶段
        warmup_time = self._run_parallel_warmup(test_func, num_workers, **kwargs)

        # 确定迭代次数
        iterations = self._determine_iterations(test_func, **kwargs)
        parallel_iterations = max(iterations // num_workers, 1)

        # 执行并行测试
        test_metrics = self._execute_parallel_test(
            test_func, parallel_iterations, num_workers, **kwargs)

        # 创建和返回结果
        return self._create_parallel_result(
            test_name, test_category, num_workers, chunk_size, warmup_time, test_metrics, iterations, **kwargs
        )

    def _setup_parallel_config(self, num_workers: Optional[int], chunk_size: Optional[int]) -> Tuple[int, int]:
        """设置并行配置参数"""
        if num_workers is None:
            num_workers = BenchmarkConstants.DEFAULT_PARALLEL_WORKERS
        if chunk_size is None:
            chunk_size = BenchmarkConstants.PARALLEL_CHUNK_SIZE
        return num_workers, chunk_size

    def _run_parallel_warmup(self, test_func: Callable, num_workers: int, **kwargs) -> float:
        """运行并行预热阶段"""
        warmup_start = time.time()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            warmup_futures = [
                executor.submit(test_func, **kwargs) for _ in range(min(self.warmup_iterations, num_workers))
            ]
            for future in warmup_futures:
                future.result()  # 等待预热完成
        return time.time() - warmup_start

    def _execute_parallel_test(
        self, test_func: Callable, parallel_iterations: int, num_workers: int, **kwargs
    ) -> Dict[str, float]:
        """执行并行测试阶段"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for _ in range(parallel_iterations):
                for worker_id in range(num_workers):
                    future = executor.submit(test_func, **kwargs)
                    futures.append(future)

            completed_iterations = 0
            for future in futures:
                future.result()
                completed_iterations += 1

        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_cpu = self._get_cpu_usage()

        return {
            "execution_time": end_time - start_time,
            "memory_usage": end_memory - start_memory,
            "cpu_usage": end_cpu - start_cpu,
            "total_operations": completed_iterations,
        }

    def _create_parallel_result(
        self,
        test_name: str,
        test_category: str,
        num_workers: int,
        chunk_size: int,
        warmup_time: float,
        test_metrics: Dict[str, float],
        iterations: int,
        **kwargs,
    ) -> BenchmarkResult:
        """创建并行测试结果"""
        execution_time = test_metrics["execution_time"]
        memory_usage = test_metrics["memory_usage"]
        cpu_usage = test_metrics["cpu_usage"]
        total_operations = test_metrics["total_operations"]

        ops_per_second = total_operations / execution_time if execution_time > 0 else float("inf")

        result = BenchmarkResult(
            test_name=f"{test_name}_parallel_{num_workers}workers",
            test_category=test_category,
            execution_time=execution_time,
            operations_per_second=ops_per_second,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            timestamp=time.time(),
            iterations=total_operations,
            warmup_time=warmup_time,
            metadata={
                **kwargs,
                "parallel_workers": num_workers,
                "chunk_size": chunk_size,
                "parallel_efficiency": ops_per_second / (iterations / execution_time) if execution_time > 0 else 1.0,
            },
        )

        self.results.append(result)
        logger.info(
            f"并行基准测试完成: {test_name}, 工作线程: {num_workers}, "
            f"总操作数: {total_operations}, OPS: {ops_per_second:.2f}"
        )

        return result

    def run_distributed_benchmark(
        self,
        test_func: Callable,
        test_name: str,
        test_category: str = "distributed",
        num_processes: Optional[int] = None,
        **kwargs,
    ) -> BenchmarkResult:
        """
        运行分布式基准测试 (进程级并行)

        Args:
            test_func: 测试函数
            test_name: 测试名称
            test_category: 测试类别
            num_processes: 进程数，默认使用CPU核心数的一半
            **kwargs: 测试函数参数

        Returns:
            BenchmarkResult: 基准测试结果
        """
        # 配置参数设置
        num_processes = self._setup_distributed_config(num_processes)

        logger.info(f"开始运行分布式基准测试: {test_name} (进程数: {num_processes})")

        # 预热阶段
        warmup_time = self._run_distributed_warmup(test_func, num_processes, **kwargs)

        # 确定迭代次数
        iterations = self._determine_iterations(test_func, **kwargs)

        # 执行分布式测试
        test_metrics = self._execute_distributed_test(
            test_func, iterations, num_processes, **kwargs)

        # 创建和返回结果
        return self._create_distributed_result(
            test_name, test_category, num_processes, iterations, warmup_time, test_metrics, **kwargs
        )

    def _setup_distributed_config(self, num_processes: Optional[int]) -> int:
        """设置分布式配置参数"""
        if num_processes is None:
            num_processes = max(1, multiprocessing.cpu_count() // 2)
        return num_processes

    def _run_distributed_warmup(self, test_func: Callable, num_processes: int, **kwargs) -> float:
        """运行分布式预热阶段"""
        warmup_start = time.time()
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            warmup_futures = [
                executor.submit(test_func, **kwargs) for _ in range(min(self.warmup_iterations, num_processes))
            ]
            for future in warmup_futures:
                future.result()
        return time.time() - warmup_start

    def _execute_distributed_test(
        self, test_func: Callable, iterations: int, num_processes: int, **kwargs
    ) -> Dict[str, float]:
        """执行分布式测试阶段"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(test_func, **kwargs) for _ in range(iterations)]
            results = [future.result() for future in futures]

        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_cpu = self._get_cpu_usage()

        return {
            "execution_time": end_time - start_time,
            "memory_usage": end_memory - start_memory,
            "cpu_usage": end_cpu - start_cpu,
        }

    def _create_distributed_result(
        self,
        test_name: str,
        test_category: str,
        num_processes: int,
        iterations: int,
        warmup_time: float,
        test_metrics: Dict[str, float],
        **kwargs,
    ) -> BenchmarkResult:
        """创建分布式测试结果"""
        execution_time = test_metrics["execution_time"]
        memory_usage = test_metrics["memory_usage"]
        cpu_usage = test_metrics["cpu_usage"]

        ops_per_second = iterations / execution_time if execution_time > 0 else float("inf")

        result = BenchmarkResult(
            test_name=f"{test_name}_distributed_{num_processes}procs",
            test_category=test_category,
            execution_time=execution_time,
            operations_per_second=ops_per_second,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            timestamp=time.time(),
            iterations=iterations,
            warmup_time=warmup_time,
            metadata={
                **kwargs,
                "distributed_processes": num_processes,
                "process_efficiency": ops_per_second / (iterations / execution_time) if execution_time > 0 else 1.0,
            },
        )

        self.results.append(result)
        logger.info(
            f"分布式基准测试完成: {test_name}, 进程数: {num_processes}, "
            f"迭代次数: {iterations}, OPS: {ops_per_second:.2f}"
        )

        return result


class BaselineManager:
    """基准线管理器"""

    def __init__(self, baseline_file: str = "performance_baselines.json"):

        self.baseline_file = Path(baseline_file)
        self.baselines = {}
        self._load_baselines()

    def _load_baselines(self):
        """加载基准线"""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for baseline_data in data.values():
                    baseline = PerformanceBaseline(**baseline_data)
                    self.baselines[baseline.test_name] = baseline
            except Exception as e:
                logger.error(f"加载基准线失败: {e}")

    def _save_baselines(self):
        """保存基准线"""
        try:
            data = {name: asdict(baseline) for name, baseline in self.baselines.items()}
            with open(self.baseline_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"保存了 {len(self.baselines)} 个基准线")
        except Exception as e:
            logger.error(f"保存基准线失败: {e}")

    def update_baseline(self, result: BenchmarkResult, threshold_percentage: float = 10.0):
        """更新基准线"""
        baseline = PerformanceBaseline(
            test_name=result.test_name,
            test_category=result.test_category,
            baseline_execution_time=result.execution_time,
            baseline_operations_per_second=result.operations_per_second,
            baseline_memory_usage=result.memory_usage,
            baseline_cpu_usage=result.cpu_usage,
            threshold_percentage=threshold_percentage,
            updated_at=time.time(),
        )

        self.baselines[result.test_name] = baseline
        self._save_baselines()
        logger.info(f"更新基准线: {result.test_name}")

    def check_regression(self, result: BenchmarkResult) -> Dict[str, Any]:
        """检查性能回归"""
        if result.test_name not in self.baselines:
            return {"has_regression": False, "message": "无基准线数据"}

        baseline = self.baselines[result.test_name]

        # 计算性能变化百分比（避免除零错误）
        if baseline.baseline_execution_time > 0:
            time_change = (result.execution_time - baseline.baseline_execution_time) / \
                           baseline.baseline_execution_time
        else:
            time_change = 0.0  # 如果基准时间为0，设为0 % 变化

        if baseline.baseline_operations_per_second > 0:
            ops_change = (
                result.operations_per_second - baseline.baseline_operations_per_second
            ) / baseline.baseline_operations_per_second
        else:
            ops_change = 0.0  # 如果基准操作数为0，设为0 % 变化

        # 检查是否超过阈值
        has_regression = False
        regression_details = []

        if abs(time_change) > baseline.threshold_percentage:
            has_regression = True
            regression_details.append(f"执行时间变化: {time_change:+.2f}%")

        if abs(ops_change) > baseline.threshold_percentage:
            has_regression = True
            regression_details.append(f"操作性能变化: {ops_change:+.2f}%")

        return {
            "has_regression": has_regression,
            "time_change_percentage": time_change,
            "ops_change_percentage": ops_change,
            "regression_details": regression_details,
            "baseline": asdict(baseline),
        }


class PerformanceReporter:
    """性能报告生成器"""

    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_report(
        self,
        results: List[BenchmarkResult],
        baseline_manager: Optional[BaselineManager] = None,
        format: str = "html",
    ):
        """生成性能报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format.lower() == "html":
            return self._generate_html_report(results, baseline_manager, timestamp)
        elif format.lower() == "json":
            return self._generate_json_report(results, baseline_manager, timestamp)
        elif format.lower() == "csv":
            return self._generate_csv_report(results, baseline_manager, timestamp)
        else:
            raise ValueError(f"不支持的报告格式: {format}")

    def _generate_html_report(
        self,
        results: List[BenchmarkResult],
        baseline_manager: Optional[BaselineManager],
        timestamp: str,
    ):
        """生成HTML报告"""
        report_file = self.output_dir / f"performance_report_{timestamp}.html"

        # 构建HTML内容
        html_content = (
            self._get_html_header(timestamp, len(results)) +
            self._get_html_table_header() +
            self._generate_html_table_rows(results, baseline_manager) +
            self._get_html_footer()
        )

        # 写入文件
        return self._write_html_file(report_file, html_content)

    def _get_html_header(self, timestamp: str, test_count: int) -> str:
        """生成HTML头部"""
        return f"""
<!DOCTYPE html>
<html>
<head>
<title>性能基准测试报告 - {timestamp}</title>
<meta charset="utf-8">
<style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
    .summary {{ margin: 20px 0; }}
    .results {{ margin: 20px 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
    .regression {{ color: red; }}
    .improvement {{ color: green; }}
    .chart {{ margin: 20px 0; }}
</style>
</head>
<body>
<div class="header">
    <h1>性能基准测试报告</h1>
    <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p>测试数量: {test_count}</p>
</div>

<div class="summary">
    <h2>测试摘要</h2>
    <table>
"""

    def _get_html_table_header(self) -> str:
        """生成表格头部"""
        return """
        <tr>
            <th>测试名称</th>
            <th>类别</th>
            <th>执行时间(s)</th>
            <th>操作 / 秒</th>
            <th>内存使用(bytes)</th>
            <th>CPU使用(%)</th>
            <th>状态</th>
        </tr>
"""

    def _generate_html_table_rows(
        self,
        results: List[BenchmarkResult],
        baseline_manager: Optional[BaselineManager]
    ) -> str:
        """生成表格行内容"""
        rows_content = ""

        for result in results:
            regression_info = self._get_regression_info(result, baseline_manager)

            rows_content += f"""
        <tr>
        <td>{result.test_name}</td>
        <td>{result.test_category}</td>
        <td>{result.execution_time:.4f}</td>
        <td>{result.operations_per_second:.2f}</td>
        <td>{result.memory_usage:.0f}</td>
        <td>{result.cpu_usage:.2f}</td>
        <td>{regression_info}</td>
        </tr>
"""

        return rows_content

    def _get_regression_info(
        self,
        result: BenchmarkResult,
        baseline_manager: Optional[BaselineManager]
    ) -> str:
        """获取性能回归信息"""
        if not baseline_manager:
            return ""

        regression = baseline_manager.check_regression(result)
        if regression["has_regression"]:
            return '<span class="regression">⚠️ 性能回归</span>'
        else:
            return '<span class="improvement">✅ 正常</span>'

    def _get_html_footer(self) -> str:
        """生成HTML结尾"""
        return """
    </table>
</div>

<div class="chart">
    <h2>性能对比图</h2>
    <p>这里可以添加图表展示性能对比</p>
</div>
</body>
</html>
"""

    def _write_html_file(self, report_file: Path, html_content: str) -> str:
        """写入HTML文件"""
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML报告已生成: {report_file}")
        return str(report_file)

    def _generate_json_report(
        self,
        results: List[BenchmarkResult],
        baseline_manager: Optional[BaselineManager],
        timestamp: str,
    ):
        """生成JSON报告"""
        report_file = self.output_dir / f"performance_report_{timestamp}.json"

        # 向量化统计计算优化
        execution_times = np.array([r.execution_time for r in results])
        ops_per_second = np.array([r.operations_per_second for r in results])

        report_data = {
            "timestamp": timestamp,
            "summary": {
                "total_tests": len(results),
                "categories": {},
                "total_execution_time": np.sum(execution_times),
                "average_ops_per_second": np.mean(ops_per_second),
                "execution_time_std": np.std(execution_times),
                "ops_per_second_std": np.std(ops_per_second),
                "min_execution_time": np.min(execution_times),
                "max_execution_time": np.max(execution_times),
            },
            "results": [asdict(result) for result in results],
        }

        # 按类别统计 - 向量化优化
        categories = [result.test_category for result in results]
        report_data["summary"]["categories"] = dict(collections.Counter(categories))

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON报告已生成: {report_file}")
        return str(report_file)

    def _generate_csv_report(
        self,
        results: List[BenchmarkResult],
        baseline_manager: Optional[BaselineManager],
        timestamp: str,
    ):
        """生成CSV报告 - 并行优化版"""
        report_file = self.output_dir / f"performance_report_{timestamp}.csv"

        # 并行处理结果数据
        rows = self._parallel_process_results(results)

        # 写入CSV文件
        self._write_csv_file(report_file, rows)

        logger.info(f"CSV报告已生成: {report_file} (并行处理: {len(results)} 条记录)")
        return str(report_file)

    def _parallel_process_results(self, results: List[BenchmarkResult]) -> List[List]:
        """并行处理结果数据"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(results))) as executor:
            rows = list(executor.map(self._process_single_result, results))

        return rows

    def _process_single_result(self, result: BenchmarkResult) -> List:
        """处理单个结果数据"""
        return [
            result.test_name,
            result.test_category,
            result.execution_time,
            result.operations_per_second,
            result.memory_usage,
            result.cpu_usage,
            result.timestamp,
            result.iterations,
            result.warmup_time,
        ]

    def _write_csv_file(self, report_file: Path, rows: List[List]):
        """写入CSV文件"""
        with open(report_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # 写入表头
            headers = self._get_csv_headers()
            writer.writerow(headers)

            # 批量写入数据（性能优化）
            writer.writerows(rows)

    def _get_csv_headers(self) -> List[str]:
        """获取CSV表头"""
        return [
            "test_name",
            "test_category",
            "execution_time",
            "operations_per_second",
            "memory_usage",
            "cpu_usage",
            "timestamp",
            "iterations",
            "warmup_time",
        ]


class PerformanceBenchmarkFramework:
    """性能基准测试框架主类"""

    def __init__(
        self,
        baseline_file: str = "performance_baselines.json",
        output_dir: str = "performance_reports",
    ):
        self.runner = BenchmarkRunner()
        self.baseline_manager = BaselineManager(baseline_file)
        self.reporter = PerformanceReporter(output_dir)

    def run_full_benchmark_suite(self, test_suite: Dict[str, Callable], update_baselines: bool = False):
        """运行完整的基准测试套件"""
        logger.info("开始运行完整基准测试套件")

        all_results = []

        for test_name, test_func in test_suite.items():
            try:
                result = self.runner.run_benchmark(test_func, test_name, "automated")

                all_results.append(result)

                # 检查性能回归
                if not update_baselines:
                    regression = self.baseline_manager.check_regression(result)
                    if regression["has_regression"]:
                        logger.warning(f"检测到性能回归: {test_name}")
                        logger.warning(f"详情: {regression['regression_details']}")

                # 更新基准线
                if update_baselines:
                    self.baseline_manager.update_baseline(result)

            except Exception as e:
                logger.error(f"测试 {test_name} 失败: {e}")

        # 生成报告
        report_path = self.reporter.generate_report(all_results, self.baseline_manager)

        return {
            "results": all_results,
            "report_path": report_path,
            "total_tests": len(all_results),
            "successful_tests": len([r for r in all_results if r.execution_time > 0]),
        }

# 预定义的测试套件


def create_infrastructure_benchmark_suite() -> Dict[str, Callable]:
    """创建基础设施层基准测试套件"""
    try:
        # 跨层级导入：infrastructure层组件
        from src.config.core.config_manager_complete import UnifiedConfigManager
        pass

        # 跨层级导入：infrastructure层组件
        # 跨层级导入：infrastructure层组件

        def config_manager_benchmark():
            """配置管理器性能测试"""
            config_manager = UnifiedConfigManager()
            # 简化测试，避免长时间运行
            for i in range(10):  # 减少循环次数
                config_manager.set_config(f"test_key_{i}", f"test_value_{i}")
                config_manager.get_config(f"test_key_{i}")

        def cache_manager_benchmark():
            """缓存管理器性能测试"""
            cache_manager = MemoryCacheManager()
            # 简化测试，避免长时间运行
            for i in range(100):  # 减少循环次数
                cache_manager.set(f"test_key_{i}", f"test_value_{i}")
                cache_manager.get(f"test_key_{i}")

        def system_monitor_benchmark():
            """系统监控性能测试"""
            monitor = SystemMonitor()
            # 简化测试，避免长时间运行
            for i in range(5):  # 减少循环次数
                monitor.collect_system_metrics()

        return {
            "config_manager": config_manager_benchmark,
            "cache_manager": cache_manager_benchmark,
            "system_monitor": system_monitor_benchmark,
        }

    except ImportError as e:
        logger.warning(f"无法导入某些模块，使用模拟测试: {e}")
        # 返回模拟测试函数

        def mock_benchmark():
            """模拟性能测试"""
            time.sleep(0.01)  # 最小延迟

        return {"mock_test": mock_benchmark}

# 工厂函数

def create_benchmark_framework(
    baseline_file: str = "performance_baselines.json",
    output_dir: str = "performance_reports",
):
    """创建基准测试框架"""
    return PerformanceBenchmarkFramework(baseline_file, output_dir)

def get_default_benchmark_framework() -> PerformanceBenchmarkFramework:
    """获取默认基准测试框架"""
    if not hasattr(get_default_benchmark_framework, "_instance"):
        get_default_benchmark_framework._instance = create_benchmark_framework()
    return get_default_benchmark_framework._instance

if __name__ == "__main__":
    # 测试代码
    framework = create_benchmark_framework()

    # 运行基础设施层基准测试
    test_suite = create_infrastructure_benchmark_suite()
    results = framework.run_full_benchmark_suite(test_suite, update_baselines=True)

    print(f"基准测试完成，共运行 {results['total_tests']} 个测试")
    print(f"成功测试: {results['successful_tests']}")
    print(f"报告路径: {results['report_path']}")
