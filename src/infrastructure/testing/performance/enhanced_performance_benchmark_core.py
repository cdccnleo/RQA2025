#!/usr/bin/env python3
"""
RQA2025 增强版性能基准测试框架 - 核心模块
建立完整的性能测试基准，涵盖所有核心组件和业务场景
"""

import sys
import time
import logging
import threading
import multiprocessing
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
import psutil
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestCategory(Enum):
    """测试类别枚举"""
    CORE_SERVICE = "core_service"
    DATA_MANAGEMENT = "data_management"
    TRADING_SYSTEM = "trading_system"
    STRATEGY_SYSTEM = "strategy_system"
    RISK_MANAGEMENT = "risk_management"
    ML_SYSTEM = "ml_system"
    DISTRIBUTED_SYSTEM = "distributed_system"
    INFRASTRUCTURE = "infrastructure"


class PerformanceLevel(Enum):
    """性能水平枚举"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    test_name: str
    test_category: TestCategory
    timestamp: float

    # 延迟指标
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_p999: float

    # 吞吐量指标
    throughput_ops_per_sec: float
    requests_per_sec: float

    # 资源使用指标
    cpu_usage_percent: float
    memory_usage_mb: float
    disk_io_mb_per_sec: float
    network_io_mb_per_sec: float

    # 并发性能指标
    concurrent_users: int
    max_concurrent_supported: int

    # 稳定性指标
    error_rate_percent: float
    availability_percent: float

    # 扩展字段
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """性能阈值定义"""
    test_category: TestCategory

    # 延迟阈值 (毫秒)
    latency_p50_ms: Dict[PerformanceLevel, float]
    latency_p95_ms: Dict[PerformanceLevel, float]
    latency_p99_ms: Dict[PerformanceLevel, float]

    # 吞吐量阈值
    min_throughput_ops_per_sec: Dict[PerformanceLevel, float]

    # 资源使用阈值
    max_cpu_usage_percent: Dict[PerformanceLevel, float]
    max_memory_usage_mb: Dict[PerformanceLevel, float]

    # 错误率阈值
    max_error_rate_percent: Dict[PerformanceLevel, float]


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_suite_name: str
    execution_time: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    performance_level: PerformanceLevel
    metrics: List[PerformanceMetrics]
    summary: Dict[str, Any]
    recommendations: List[str]


class PerformanceCollector:
    """性能数据收集器"""

    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self._monitoring = False
        self._metrics_data = []
        self._monitor_thread = None

    def start_monitoring(self):
        """开始性能监控"""
        self._monitoring = True
        self._metrics_data = []
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, Any]:
        """停止性能监控并返回数据"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()

        if not self._metrics_data:
            return {}

        # 计算统计信息
        cpu_values = [d['cpu_percent'] for d in self._metrics_data]
        memory_values = [d['memory_mb'] for d in self._metrics_data]

        return {
            'duration': len(self._metrics_data) * self.sampling_interval,
            'cpu_usage': {
                'mean': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'p95': np.percentile(cpu_values, 95)
            },
            'memory_usage': {
                'mean': statistics.mean(memory_values),
                'max': max(memory_values),
                'p95': np.percentile(memory_values, 95)
            },
            'samples': len(self._metrics_data)
        }

    def _monitor_loop(self):
        """监控循环"""
        process = psutil.Process()

        while self._monitoring:
            try:
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)

                self._metrics_data.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'memory_vms_mb': memory_info.vms / (1024 * 1024)
                })

                time.sleep(self.sampling_interval)
            except Exception as e:
                logging.warning(f"监控数据收集失败: {e}")
                break


class LatencyMeasurer:
    """延迟测量器"""

    def __init__(self):
        self.measurements = []

    def measure(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """测量函数执行延迟"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        latency = end_time - start_time
        self.measurements.append(latency)

        return result, latency

    async def measure_async(self, coro):
        """测量异步函数执行延迟"""
        start_time = time.perf_counter()
        result = await coro
        end_time = time.perf_counter()

        latency = end_time - start_time
        self.measurements.append(latency)

        return result, latency

    def get_percentiles(self) -> Dict[str, float]:
        """获取延迟百分位数"""
        if not self.measurements:
            return {}

        measurements_ms = [m * 1000 for m in self.measurements]  # 转换为毫秒

        return {
            'p50': float(np.percentile(measurements_ms, 50)),
            'p95': float(np.percentile(measurements_ms, 95)),
            'p99': float(np.percentile(measurements_ms, 99)),
            'p999': float(np.percentile(measurements_ms, 99.9)),
            'mean': float(np.mean(measurements_ms)),
            'max': float(np.max(measurements_ms)),
            'min': float(np.min(measurements_ms))
        }

    def reset(self):
        """重置测量数据"""
        self.measurements = []


class ThroughputMeasurer:
    """吞吐量测量器"""

    def __init__(self):
        self.start_time = None
        self.operation_count = 0
        self.request_count = 0

    def start(self):
        """开始测量"""
        self.start_time = time.time()
        self.operation_count = 0
        self.request_count = 0

    def record_operation(self, count: int = 1):
        """记录操作数量"""
        self.operation_count += count

    def record_request(self, count: int = 1):
        """记录请求数量"""
        self.request_count += count

    def get_throughput(self) -> Dict[str, float]:
        """获取吞吐量指标"""
        if self.start_time is None:
            return {}

        duration = time.time() - self.start_time
        if duration <= 0:
            return {}

        return {
            'ops_per_sec': self.operation_count / duration,
            'requests_per_sec': self.request_count / duration,
            'duration': duration,
            'total_operations': self.operation_count,
            'total_requests': self.request_count
        }


class ConcurrencyTester:
    """并发测试器"""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()

    def test_concurrent_execution(self, func: Callable, args_list: List[Tuple],
                                  max_concurrent: Optional[int] = None) -> Dict[str, Any]:
        """测试并发执行性能"""
        max_concurrent = max_concurrent or self.max_workers

        start_time = time.time()
        results = []
        errors = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # 提交所有任务
            futures = [executor.submit(func, *args) for args in args_list]

            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append(str(e))

        end_time = time.time()
        duration = end_time - start_time

        return {
            'duration': duration,
            'total_tasks': len(args_list),
            'successful_tasks': len(results),
            'failed_tasks': len(errors),
            'error_rate': len(errors) / len(args_list) * 100,
            'throughput': len(results) / duration if duration > 0 else 0,
            'concurrent_workers': max_concurrent,
            'errors': errors[:10]  # 只保留前10个错误
        }

    def test_load_capacity(self, func: Callable, base_args: Tuple,
                           start_concurrent: int = 1, max_concurrent: Optional[int] = None,
                           step: int = 10, duration_per_step: float = 5.0) -> Dict[str, Any]:
        """测试负载容量"""
        max_concurrent = max_concurrent or self.max_workers * 4
        capacity_results = []

        for concurrent_users in range(start_concurrent, max_concurrent + 1, step):
            logging.info(f"测试并发用户数: {concurrent_users}")

            # 创建任务列表
            args_list = [base_args] * concurrent_users

            # 运行测试
            result = self.test_concurrent_execution(func, args_list, concurrent_users)
            result['concurrent_users'] = concurrent_users
            capacity_results.append(result)

            # 如果错误率过高，停止测试
            if result['error_rate'] > 50:
                logging.warning(f"并发用户数 {concurrent_users} 时错误率过高，停止测试")
                break

            # 休息一下避免系统过载
            time.sleep(1)

        return {
            'capacity_test_results': capacity_results,
            'max_stable_concurrent': self._find_max_stable_concurrent(capacity_results)
        }

    def _find_max_stable_concurrent(self, results: List[Dict[str, Any]]) -> int:
        """找到最大稳定并发数"""
        for result in reversed(results):
            if result['error_rate'] < 5 and result['throughput'] > 0:
                return result['concurrent_users']
        return 1
