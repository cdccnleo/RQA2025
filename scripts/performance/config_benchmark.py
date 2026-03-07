#!/usr/bin/env python3
"""
配置管理模块性能基准测试

本脚本用于测试配置管理模块在各种场景下的性能表现。
"""

import time
import statistics
from typing import List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.infrastructure.config import (
    UnifiedConfigManager,
    CachePolicy
)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    total_operations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    median_time: float
    operations_per_second: float
    cache_hit_rate: float
    memory_usage: float = 0.0


class ConfigBenchmark:
    """配置管理模块性能基准测试"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def run_benchmark(self, test_name: str, test_func, iterations: int = 1000) -> BenchmarkResult:
        """运行基准测试"""
        print(f"运行测试: {test_name}")

        # 预热
        test_func(10)

        # 执行测试
        times = []
        for i in range(iterations):
            start_time = time.time()
            test_func(1)
            end_time = time.time()
            times.append(end_time - start_time)

        # 计算统计信息
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        operations_per_second = iterations / total_time

        # 获取缓存命中率
        config_manager = UnifiedConfigManager()
        metrics = config_manager.get_performance_metrics()
        cache_hit_rate = metrics.get('cache_hit_rate', 0.0)

        result = BenchmarkResult(
            test_name=test_name,
            total_operations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            operations_per_second=operations_per_second,
            cache_hit_rate=cache_hit_rate
        )

        self.results.append(result)
        return result

    def print_result(self, result: BenchmarkResult):
        """打印测试结果"""
        print(f"\n📊 {result.test_name}")
        print(f"   总操作数: {result.total_operations:,}")
        print(f"   总耗时: {result.total_time:.3f}秒")
        print(f"   平均耗时: {result.avg_time:.6f}秒")
        print(f"   最小耗时: {result.min_time:.6f}秒")
        print(f"   最大耗时: {result.max_time:.6f}秒")
        print(f"   中位数耗时: {result.median_time:.6f}秒")
        print(f"   每秒操作数: {result.operations_per_second:,.0f}")
        print(f"   缓存命中率: {result.cache_hit_rate:.2%}")

    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("📈 性能基准测试总结")
        print("=" * 60)

        for result in self.results:
            self.print_result(result)

        # 找出最佳性能
        best_ops = max(self.results, key=lambda r: r.operations_per_second)
        print(f"\n🏆 最佳性能: {best_ops.test_name}")
        print(f"   每秒操作数: {best_ops.operations_per_second:,.0f}")


def basic_operations_benchmark():
    """基础操作性能测试"""
    benchmark = ConfigBenchmark()

    def get_operation(count):
        config_manager = UnifiedConfigManager()
        for i in range(count):
            config_manager.get(f"test.key_{i}", default=f"value_{i}")

    def set_operation(count):
        config_manager = UnifiedConfigManager()
        for i in range(count):
            config_manager.set(f"test.key_{i}", f"value_{i}")

    def get_set_mixed(count):
        config_manager = UnifiedConfigManager()
        for i in range(count):
            if i % 2 == 0:
                config_manager.set(f"test.key_{i}", f"value_{i}")
            else:
                config_manager.get(f"test.key_{i}", default=f"value_{i}")

    # 运行测试
    benchmark.run_benchmark("Get操作", get_operation, 10000)
    benchmark.run_benchmark("Set操作", set_operation, 10000)
    benchmark.run_benchmark("混合操作", get_set_mixed, 10000)

    benchmark.print_summary()
    return benchmark


def cache_performance_benchmark():
    """缓存性能测试"""
    benchmark = ConfigBenchmark()

    def lru_cache_test(count):
        config_manager = UnifiedConfigManager(
            cache_policy=CachePolicy.LRU,
            cache_size=100
        )
        # 预热缓存
        for i in range(100):
            config_manager.set(f"cache.key_{i}", f"value_{i}")
        # 测试缓存命中
        for i in range(count):
            config_manager.get(f"cache.key_{i % 100}")

    def ttl_cache_test(count):
        config_manager = UnifiedConfigManager(
            cache_policy=CachePolicy.TTL,
            cache_size=100
        )
        # 预热缓存
        for i in range(100):
            config_manager.set(f"cache.key_{i}", f"value_{i}")
        # 测试缓存命中
        for i in range(count):
            config_manager.get(f"cache.key_{i % 100}")

    def no_cache_test(count):
        config_manager = UnifiedConfigManager(
            cache_policy=CachePolicy.NO_CACHE
        )
        # 预热缓存
        for i in range(100):
            config_manager.set(f"cache.key_{i}", f"value_{i}")
        # 测试无缓存
        for i in range(count):
            config_manager.get(f"cache.key_{i % 100}")

    # 运行测试
    benchmark.run_benchmark("LRU缓存", lru_cache_test, 10000)
    benchmark.run_benchmark("TTL缓存", ttl_cache_test, 10000)
    benchmark.run_benchmark("无缓存", no_cache_test, 10000)

    benchmark.print_summary()
    return benchmark


def encryption_performance_benchmark():
    """加密性能测试"""
    benchmark = ConfigBenchmark()

    def encrypted_test(count):
        config_manager = UnifiedConfigManager(enable_encryption=True)
        for i in range(count):
            config_manager.set(f"encrypted.key_{i}", f"sensitive_value_{i}")
            config_manager.get(f"encrypted.key_{i}")

    def unencrypted_test(count):
        config_manager = UnifiedConfigManager(enable_encryption=False)
        for i in range(count):
            config_manager.set(f"plain.key_{i}", f"plain_value_{i}")
            config_manager.get(f"plain.key_{i}")

    # 运行测试
    benchmark.run_benchmark("加密操作", encrypted_test, 5000)
    benchmark.run_benchmark("非加密操作", unencrypted_test, 5000)

    benchmark.print_summary()
    return benchmark


def concurrent_performance_benchmark():
    """并发性能测试"""
    benchmark = ConfigBenchmark()

    def single_thread_test(count):
        config_manager = UnifiedConfigManager()
        for i in range(count):
            config_manager.set(f"single.key_{i}", f"value_{i}")
            config_manager.get(f"single.key_{i}")

    def multi_thread_test(count):
        config_manager = UnifiedConfigManager()

        def worker(thread_id):
            for i in range(count // 4):  # 4个线程平分工作
                key = f"multi.key_{thread_id}_{i}"
                config_manager.set(key, f"value_{i}")
                config_manager.get(key)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for future in as_completed(futures):
                future.result()

    # 运行测试
    benchmark.run_benchmark("单线程", single_thread_test, 5000)
    benchmark.run_benchmark("多线程", multi_thread_test, 5000)

    benchmark.print_summary()
    return benchmark


def validation_performance_benchmark():
    """验证性能测试"""
    benchmark = ConfigBenchmark()

    def valid_config_test(count):
        config_manager = UnifiedConfigManager()
        valid_config = {
            "database.port": 5432,
            "cache.max_size": 1000,
            "risk.max_drawdown": 0.1,
            "risk.stop_loss": 0.05,
            "database.url": "localhost"
        }
        for i in range(count):
            config_manager.validate(valid_config)

    def invalid_config_test(count):
        config_manager = UnifiedConfigManager()
        invalid_config = {
            "database.port": "invalid",
            "cache.max_size": -1,
            "risk.max_drawdown": 1.5,
            "risk.stop_loss": -0.1,
            "database.url": None
        }
        for i in range(count):
            config_manager.validate(invalid_config)

    # 运行测试
    benchmark.run_benchmark("有效配置验证", valid_config_test, 1000)
    benchmark.run_benchmark("无效配置验证", invalid_config_test, 1000)

    benchmark.print_summary()
    return benchmark


def memory_usage_benchmark():
    """内存使用测试"""
    import psutil
    import os

    benchmark = ConfigBenchmark()

    def memory_test(count):
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        config_manager = UnifiedConfigManager(cache_size=count)

        # 填充配置
        for i in range(count):
            config_manager.set(f"memory.key_{i}", f"value_{i}" * 100)  # 较大的值

        # 访问配置
        for i in range(count):
            config_manager.get(f"memory.key_{i}")

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory

        return memory_usage

    # 测试不同大小的配置
    sizes = [100, 500, 1000, 5000]

    for size in sizes:
        result = BenchmarkResult(
            test_name=f"内存测试 ({size}项)",
            total_operations=size,
            total_time=0.0,
            avg_time=0.0,
            min_time=0.0,
            max_time=0.0,
            median_time=0.0,
            operations_per_second=0.0,
            cache_hit_rate=0.0,
            memory_usage=memory_test(size)
        )
        benchmark.results.append(result)

    print("\n📊 内存使用测试")
    print("=" * 40)
    for result in benchmark.results:
        print(f"{result.test_name}: {result.memory_usage:.2f} MB")

    return benchmark


def load_test_benchmark():
    """负载测试"""
    benchmark = ConfigBenchmark()

    def high_load_test(count):
        config_manager = UnifiedConfigManager(
            cache_policy=CachePolicy.LRU,
            cache_size=1000
        )

        # 高负载操作
        for i in range(count):
            # 设置配置
            config_manager.set(f"load.key_{i}", f"value_{i}")

            # 获取配置
            config_manager.get(f"load.key_{i}")

            # 验证配置
            if i % 100 == 0:
                test_config = {
                    "database.port": 5432,
                    "cache.max_size": 1000,
                    "risk.max_drawdown": 0.1
                }
                config_manager.validate(test_config)

    # 运行高负载测试
    benchmark.run_benchmark("高负载测试", high_load_test, 10000)

    benchmark.print_summary()
    return benchmark


def stress_test_benchmark():
    """压力测试"""
    benchmark = ConfigBenchmark()

    def stress_test(count):
        config_manager = UnifiedConfigManager(
            cache_policy=CachePolicy.LRU,
            cache_size=100,
            enable_encryption=True
        )

        # 压力测试：频繁的读写操作
        for i in range(count):
            # 写入操作
            config_manager.set(f"stress.key_{i % 1000}", f"value_{i}")

            # 读取操作
            config_manager.get(f"stress.key_{i % 1000}")

            # 导出配置
            if i % 1000 == 0:
                config_manager.export_config()

            # 性能监控
            if i % 500 == 0:
                config_manager.get_performance_metrics()

    # 运行压力测试
    benchmark.run_benchmark("压力测试", stress_test, 20000)

    benchmark.print_summary()
    return benchmark


def main():
    """主函数"""
    print("配置管理模块性能基准测试")
    print("=" * 50)

    try:
        # 运行所有基准测试
        print("\n🔧 基础操作性能测试")
        basic_operations_benchmark()

        print("\n🚀 缓存性能测试")
        cache_performance_benchmark()

        print("\n🔐 加密性能测试")
        encryption_performance_benchmark()

        print("\n🔄 并发性能测试")
        concurrent_performance_benchmark()

        print("\n✅ 验证性能测试")
        validation_performance_benchmark()

        print("\n💾 内存使用测试")
        memory_usage_benchmark()

        print("\n⚡ 负载测试")
        load_test_benchmark()

        print("\n🔥 压力测试")
        stress_test_benchmark()

        print("\n" + "=" * 50)
        print("✅ 所有性能基准测试完成")

    except Exception as e:
        print(f"\n❌ 性能基准测试失败: {e}")
        raise


if __name__ == "__main__":
    main()
