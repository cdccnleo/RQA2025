#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
缓存管理系统性能测试运行器
提供高效的性能测试执行和基准比较
"""

import time
import statistics
import psutil
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    operation: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    p50_time: float
    p95_time: float
    p99_time: float
    ops_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float


class CachePerformanceTester:
    """缓存性能测试器"""

    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.process = psutil.Process(os.getpid())

    def measure_operation(self, operation_func, operation_name: str,
                        iterations: int = 1000, warmup: int = 100) -> PerformanceMetrics:
        """
        测量操作性能

        Args:
            operation_func: 要测量的操作函数
            operation_name: 操作名称
            iterations: 迭代次数
            warmup: 预热次数

        Returns:
            PerformanceMetrics: 性能指标
        """
        # 预热
        for _ in range(warmup):
            operation_func()

        # 收集执行时间
        execution_times = []

        # 测量内存和CPU使用率
        memory_usage_start = self.process.memory_info().rss / 1024 / 1024
        cpu_usage_start = self.process.cpu_percent(interval=None)

        start_time = time.perf_counter()

        for _ in range(iterations):
            op_start = time.perf_counter()
            operation_func()
            op_end = time.perf_counter()
            execution_times.append(op_end - op_start)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        memory_usage_end = self.process.memory_info().rss / 1024 / 1024
        cpu_usage_end = self.process.cpu_percent(interval=None)

        memory_usage = (memory_usage_start + memory_usage_end) / 2
        cpu_usage = max(cpu_usage_start, cpu_usage_end)  # 取峰值

        # 计算统计指标
        avg_time = statistics.mean(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        p50_time = statistics.median(execution_times)
        p95_time = statistics.quantiles(execution_times, n=20)[18]  # 95th percentile
        p99_time = statistics.quantiles(execution_times, n=100)[98]  # 99th percentile
        ops_per_second = iterations / total_time

        return PerformanceMetrics(
            operation=operation_name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            p50_time=p50_time,
            p95_time=p95_time,
            p99_time=p99_time,
            ops_per_second=ops_per_second,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage
        )

    def run_cache_operations_benchmark(self, iterations: int = 1000) -> List[PerformanceMetrics]:
        """
        运行缓存操作基准测试

        Args:
            iterations: 每次操作的迭代次数

        Returns:
            List[PerformanceMetrics]: 性能指标列表
        """
        results = []

        # Set操作基准测试
        def set_operation():
            key = f"bench_key_{time.time()}_{os.urandom(4).hex()}"
            value = f"bench_value_{os.urandom(8).hex()}"
            self.cache_manager.set(key, value, ttl=300)

        set_metrics = self.measure_operation(
            set_operation, "cache_set", iterations
        )
        results.append(set_metrics)

        # Get操作基准测试
        # 预先设置一些数据用于读取
        test_keys = []
        for i in range(min(iterations, 100)):  # 限制预设数据量
            key = f"bench_read_key_{i}"
            value = f"bench_read_value_{i}"
            self.cache_manager.set(key, value, ttl=300)
            test_keys.append(key)

        def get_operation():
            key = test_keys[hash(str(time.time())) % len(test_keys)]
            self.cache_manager.get(key)

        get_metrics = self.measure_operation(
            get_operation, "cache_get", iterations
        )
        results.append(get_metrics)

        # Delete操作基准测试
        def delete_operation():
            key = f"bench_del_key_{time.time()}_{os.urandom(4).hex()}"
            self.cache_manager.set(key, "temp_value", ttl=300)
            self.cache_manager.delete(key)

        delete_metrics = self.measure_operation(
            delete_operation, "cache_delete", iterations
        )
        results.append(delete_metrics)

        # Exists操作基准测试
        def exists_operation():
            key = test_keys[hash(str(time.time())) % len(test_keys)]
            self.cache_manager.exists(key)

        exists_metrics = self.measure_operation(
            exists_operation, "cache_exists", iterations
        )
        results.append(exists_metrics)

        return results

    def run_concurrent_performance_test(self, num_threads: int = 4,
                                        operations_per_thread: int = 500) -> Dict[str, Any]:
        """
        运行并发性能测试

        Args:
            num_threads: 线程数
            operations_per_thread: 每个线程的操作数

        Returns:
            Dict[str, Any]: 并发测试结果
        """
        results = {
            'num_threads': num_threads,
            'operations_per_thread': operations_per_thread,
            'total_operations': num_threads * operations_per_thread,
            'thread_results': [],
            'summary': {}
        }

        def thread_worker(thread_id: int) -> Dict[str, Any]:
            """线程工作函数"""
            thread_results = {
                'thread_id': thread_id,
                'operations': 0,
                'errors': 0,
                'duration': 0
            }

            start_time = time.time()

            try:
                for i in range(operations_per_thread):
                    key = f"concurrent_key_{thread_id}_{i}"
                    value = f"concurrent_value_{thread_id}_{i}"

                    # 执行混合操作
                    self.cache_manager.set(key, value, ttl=300)
                    retrieved = self.cache_manager.get(key)
                    exists = self.cache_manager.exists(key)

                    thread_results['operations'] += 3

                    # 验证数据一致性
                    if retrieved != value or not exists:
                        thread_results['errors'] += 1

            except Exception as e:
                thread_results['errors'] += 1

            thread_results['duration'] = time.time() - start_time
            return thread_results

        # 执行并发测试
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(thread_worker, i) for i in range(num_threads)]

            for future in as_completed(futures):
                result = future.result()
                results['thread_results'].append(result)

        # 计算汇总统计
        total_operations = sum(r['operations'] for r in results['thread_results'])
        total_errors = sum(r['errors'] for r in results['thread_results'])
        total_duration = max(r['duration'] for r in results['thread_results'])  # 最长执行时间

        results['summary'] = {
            'total_operations': total_operations,
            'total_errors': total_errors,
            'total_duration': total_duration,
            'ops_per_second': total_operations / total_duration if total_duration > 0 else 0,
            'error_rate': total_errors / total_operations if total_operations > 0 else 0,
            'success_rate': 1 - (total_errors / total_operations) if total_operations > 0 else 0
        }

        return results


def run_performance_benchmarks(cache_manager, output_file: Optional[str] = None):
    """
    运行完整的性能基准测试

    Args:
        cache_manager: 缓存管理器实例
        output_file: 输出文件路径
    """
    print("=== 缓存管理系统性能基准测试 ===")

    tester = CachePerformanceTester(cache_manager)

    # 1. 单线程性能测试
    print("\n1. 运行单线程性能测试...")
    single_thread_results = tester.run_cache_operations_benchmark(iterations=1000)

    print("单线程性能结果:")
    for metrics in single_thread_results:
        print(f"  {metrics.operation}:")
        print(f"    OPS: {metrics.ops_per_second:.0f}/s")
        print(f"    Avg: {metrics.avg_time*1000:.2f}ms")
        print(f"    P95: {metrics.p95_time*1000:.2f}ms")
        print(f"    Memory: {metrics.memory_usage_mb:.1f}MB")

    # 2. 并发性能测试
    print("\n2. 运行并发性能测试...")
    concurrent_results = tester.run_concurrent_performance_test(
        num_threads=4, operations_per_thread=250
    )

    print("并发性能结果:")
    print(f"  线程数: {concurrent_results['num_threads']}")
    print(f"  总操作数: {concurrent_results['total_operations']}")
    print(f"  OPS: {concurrent_results['summary']['ops_per_second']:.0f}/s")
    print(f"  成功率: {concurrent_results['summary']['success_rate']:.2f}")
    print(f"  错误率: {concurrent_results['summary']['error_rate']:.2f}")

    # 3. 保存结果
    if output_file:
        import json
        results = {
            'timestamp': time.time(),
            'single_thread': [vars(m) for m in single_thread_results],
            'concurrent': concurrent_results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 性能测试结果已保存到: {output_file}")

    # 4. 性能评估
    print("\n3. 性能评估:")

    # 单线程性能评估
    set_ops = next((m.ops_per_second for m in single_thread_results if m.operation == 'cache_set'), 0)
    get_ops = next((m.ops_per_second for m in single_thread_results if m.operation == 'cache_get'), 0)

    # 并发性能评估
    concurrent_ops = concurrent_results['summary']['ops_per_second']
    success_rate = concurrent_results['summary']['success_rate']

    # 评分标准
    single_thread_score = 0
    if set_ops >= 1000 and get_ops >= 2000:
        single_thread_score = 25  # 优秀
    elif set_ops >= 500 and get_ops >= 1000:
        single_thread_score = 20  # 良好
    elif set_ops >= 200 and get_ops >= 500:
        single_thread_score = 15  # 及格
    else:
        single_thread_score = 10  # 需改进

    concurrent_score = 0
    if concurrent_ops >= 1000 and success_rate >= 0.99:
        concurrent_score = 25  # 优秀
    elif concurrent_ops >= 500 and success_rate >= 0.95:
        concurrent_score = 20  # 良好
    elif concurrent_ops >= 200 and success_rate >= 0.90:
        concurrent_score = 15  # 及格
    else:
        concurrent_score = 10  # 需改进

    total_score = single_thread_score + concurrent_score

    if total_score >= 40:
        performance_grade = "优秀"
    elif total_score >= 30:
        performance_grade = "良好"
    elif total_score >= 20:
        performance_grade = "及格"
    else:
        performance_grade = "需改进"

    print(f"  单线程性能: {single_thread_score}/25")
    print(f"  并发性能: {concurrent_score}/25")
    print(f"  综合评分: {total_score}/50 ({performance_grade})")

    if total_score >= 30:
        print("🎉 缓存系统性能达标！")
    else:
        print("⚠️ 缓存系统性能需要优化")

    return {
        'single_thread_results': single_thread_results,
        'concurrent_results': concurrent_results,
        'performance_score': total_score,
        'performance_grade': performance_grade
    }


if __name__ == "__main__":
    # 示例用法
    import sys
    sys.path.insert(0, 'src')

    try:
        from infrastructure.cache.core.cache_manager import UnifiedCacheManager
        from infrastructure.cache.core.cache_configs import CacheConfig, CacheLevel

        # 创建缓存管理器
        config = CacheConfig(
            basic=type('Basic', (), {'max_size': 1000, 'ttl': 3600, 'strategy': 'LRU'})(),
            multi_level=type('MultiLevel', (), {
                'level': CacheLevel.MEMORY,
                'memory_max_size': 1000,
                'memory_ttl': 1800
            })(),
            smart=type('Smart', (), {'enable_monitoring': False, 'monitor_interval': 30})()
        )

        manager = UnifiedCacheManager(config)

        # 运行性能基准测试
        results = run_performance_benchmarks(manager, 'cache_performance_results.json')

        print(f"\n最终性能评分: {results['performance_score']}/50 ({results['performance_grade']})")

    except Exception as e:
        print(f"性能测试执行失败: {e}")
        import traceback
        traceback.print_exc()
