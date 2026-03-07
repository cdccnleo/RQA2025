#!/usr/bin/env python3
"""
RQA2025量化交易系统性能基准测试工具
用于测量系统在不同负载下的性能表现
"""
import asyncio
import time
import psutil
import tracemalloc
import statistics
from typing import Dict, List, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import json


class PerformanceBenchmark:
    """性能基准测试工具"""

    def __init__(self):
        self.results = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

    def benchmark_async_function(self, func: Callable, *args,
                                 iterations: int = 1000,
                                 concurrency: int = 10,
                                 warmup_iterations: int = 100) -> Dict[str, Any]:
        """
        异步函数性能基准测试

        Args:
            func: 要测试的异步函数
            args: 函数参数
            iterations: 测试迭代次数
            concurrency: 并发数
            warmup_iterations: 预热迭代次数

        Returns:
            性能测试结果字典
        """
        async def run_test():
            # 预热
            print(f"🔥 预热 {warmup_iterations} 次...")
            for _ in range(warmup_iterations):
                await func(*args)

            # 正式测试
            print(f"📊 开始性能测试: {iterations} 次迭代, {concurrency} 并发")

            tracemalloc.start()
            start_time = time.time()
            start_cpu = psutil.cpu_percent(interval=None)
            start_memory = psutil.virtual_memory().used

            # 创建并发任务
            tasks = []
            for i in range(concurrency):
                task_iterations = iterations // concurrency
                tasks.append(self._run_concurrent_test(func, args, task_iterations))

            # 等待所有任务完成
            latencies = await asyncio.gather(*tasks)
            latencies = [lat for sublist in latencies for lat in sublist]

            end_time = time.time()
            end_cpu = psutil.cpu_percent(interval=None)
            end_memory = psutil.virtual_memory().used

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # 计算统计信息
            total_time = end_time - start_time
            cpu_usage = max(0, end_cpu - start_cpu)  # CPU使用率变化
            memory_usage = end_memory - start_memory

            return {
                'total_time': total_time,
                'iterations': iterations,
                'concurrency': concurrency,
                'throughput': iterations / total_time,
                'avg_latency': statistics.mean(latencies) * 1000,  # 转换为毫秒
                'p50_latency': statistics.median(latencies) * 1000,
                'p95_latency': sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
                'p99_latency': sorted(latencies)[int(len(latencies) * 0.99)] * 1000,
                'min_latency': min(latencies) * 1000,
                'max_latency': max(latencies) * 1000,
                'cpu_usage': cpu_usage,
                'memory_delta': memory_usage,
                'memory_peak': peak,
                'memory_current': current
            }

        # 检查是否已经在事件循环中
        try:
            loop = asyncio.get_running_loop()
            # 如果已经在事件循环中，直接运行
            return loop.run_until_complete(run_test())
        except RuntimeError:
            # 不在事件循环中，使用asyncio.run
            return asyncio.run(run_test())

    async def _run_concurrent_test(self, func: Callable, args: tuple,
                                   iterations: int) -> List[float]:
        """运行并发测试"""
        latencies = []

        for _ in range(iterations):
            start = time.perf_counter()
            await func(*args)
            end = time.perf_counter()
            latencies.append(end - start)

        return latencies

    def benchmark_memory_usage(self, func: Callable, *args,
                               iterations: int = 1000) -> Dict[str, Any]:
        """内存使用基准测试"""

        tracemalloc.start()

        # 预热
        for _ in range(100):
            func(*args)

        # 测试
        snapshots = []
        for i in range(iterations):
            func(*args)
            if i % 100 == 0:  # 每100次记录一次快照
                snapshots.append(tracemalloc.take_snapshot())

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            'iterations': iterations,
            'memory_current': current,
            'memory_peak': peak,
            'avg_memory_per_operation': current / iterations,
            'snapshots': len(snapshots)
        }

    def benchmark_database_operations(self, db_func: Callable, *args,
                                      iterations: int = 1000) -> Dict[str, Any]:
        """数据库操作性能测试"""

        latencies = []
        start_memory = psutil.virtual_memory().used

        for _ in range(iterations):
            start = time.perf_counter()
            result = db_func(*args)
            end = time.perf_counter()
            latencies.append(end - start)

        end_memory = psutil.virtual_memory().used

        return {
            'iterations': iterations,
            'avg_latency': statistics.mean(latencies) * 1000,
            'p95_latency': sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
            'memory_delta': end_memory - start_memory,
            'operations_per_second': iterations / sum(latencies)
        }

    def save_results(self, results: Dict[str, Any], filename: str = "benchmark_results.json"):
        """保存测试结果"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def load_results(self, filename: str = "benchmark_results.json") -> Dict[str, Any]:
        """加载测试结果"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def compare_results(self, baseline_results: Dict[str, Any],
                        current_results: Dict[str, Any]) -> Dict[str, Any]:
        """比较性能结果"""

        comparison = {}

        for key in set(baseline_results.keys()) & set(current_results.keys()):
            baseline_val = baseline_results[key]
            current_val = current_results[key]

            if isinstance(baseline_val, (int, float)) and isinstance(current_val, (int, float)):
                if baseline_val != 0:
                    change_percent = ((current_val - baseline_val) / baseline_val) * 100
                    comparison[key] = {
                        'baseline': baseline_val,
                        'current': current_val,
                        'change_percent': change_percent,
                        'improvement': change_percent < 0 if key.endswith('_latency') or key.startswith('memory') else change_percent > 0
                    }

        return comparison

    def print_results(self, results: Dict[str, Any], title: str = "性能测试结果"):
        """打印测试结果"""
        print(f"\n{'='*60}")
        print(f"🎯 {title}")
        print(f"{'='*60}")

        for key, value in results.items():
            if isinstance(value, float):
                if 'latency' in key.lower():
                    print(f"{key}: {value:.4f}ms")
                elif 'throughput' in key.lower() or 'operations' in key.lower():
                    print(f"{key}: {value:.2f} ops/sec")
                elif 'memory' in key.lower():
                    print(f"{key}: {value:,} bytes")
                elif 'cpu' in key.lower():
                    print(f"{key}: {value:.2f}%")
                else:
                    print(f"{key}: {value:.4f}")
            elif isinstance(value, int):
                print(f"{key}: {value:,}")
            else:
                print(f"{key}: {value}")

        print(f"{'='*60}\n")


# 全局基准测试实例
benchmark = PerformanceBenchmark()


def benchmark_async(func: Callable) -> Callable:
    """
    异步函数性能测试装饰器

    使用示例:
    @benchmark_async
    async def my_async_function(param1, param2):
        # 函数实现
        pass

    # 运行测试
    results = my_async_function.benchmark(iterations=1000, concurrency=10)
    """
    async def benchmark_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    def benchmark_method(iterations: int = 1000, concurrency: int = 10,
                         warmup_iterations: int = 100) -> Dict[str, Any]:
        async def test_func():
            await func()

        return benchmark.benchmark_async_function(
            test_func,
            iterations=iterations,
            concurrency=concurrency,
            warmup_iterations=warmup_iterations
        )

    benchmark_wrapper.benchmark = benchmark_method
    return benchmark_wrapper


if __name__ == "__main__":
    # 示例使用
    async def sample_async_function():
        """示例异步函数"""
        await asyncio.sleep(0.001)  # 模拟一些异步操作
        return {"result": "success"}

    # 运行基准测试
    print("🚀 RQA2025性能基准测试工具")
    print("正在测试示例异步函数...")

    results = benchmark.benchmark_async_function(
        sample_async_function,
        iterations=1000,
        concurrency=10,
        warmup_iterations=100
    )

    benchmark.print_results(results, "示例异步函数性能测试")

    # 保存结果
    benchmark.save_results(results, "sample_benchmark.json")
    print("✅ 测试结果已保存到 sample_benchmark.json")
