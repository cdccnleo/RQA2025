#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU使用率优化脚本
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time
import psutil


def optimize_strategy_calculation_parallel(data_size=1000000, num_workers=None):
    """优化策略计算 - 并行处理"""
    if num_workers is None:
        num_workers = min(4, multiprocessing.cpu_count())

    print(f"使用 {num_workers} 个工作进程进行并行计算")

    # 分割数据
    chunk_size = data_size // num_workers
    chunks = [np.random.random(chunk_size) for _ in range(num_workers)]

    start_time = time.time()

    # 并行处理
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_data_chunk, chunk) for chunk in chunks]
        results = [future.result() for future in futures]

    end_time = time.time()

    return {
        "method": "parallel_processing",
        "data_size": data_size,
        "num_workers": num_workers,
        "execution_time": end_time - start_time,
        "results": results,
        "cpu_usage": psutil.cpu_percent(interval=1)
    }


def process_data_chunk(chunk):
    """处理数据块"""
    # 模拟复杂的数学计算
    result = np.sum(np.sqrt(np.abs(np.fft.fft(chunk))))
    return result


def optimize_with_vectorization(data_size=1000000):
    """向量化优化"""
    print("使用向量化处理优化计算")

    start_time = time.time()

    # 生成测试数据
    data = np.random.random(data_size)

    # 向量化计算
    result = np.sum(np.sqrt(np.abs(np.fft.fft(data))))

    end_time = time.time()

    return {
        "method": "vectorization",
        "data_size": data_size,
        "execution_time": end_time - start_time,
        "result": result,
        "cpu_usage": psutil.cpu_percent(interval=1)
    }


def optimize_with_caching(cache_size=1000):
    """缓存优化"""
    print("实施缓存优化策略")

    # 模拟缓存
    cache = {}
    cache_hits = 0
    cache_misses = 0

    start_time = time.time()

    for i in range(cache_size * 2):
        key = f"calculation_{i % cache_size}"

        if key in cache:
            cache_hits += 1
            result = cache[key]
        else:
            cache_misses += 1
            # 模拟计算
            result = np.sum(np.random.random(1000))
            cache[key] = result

    end_time = time.time()

    return {
        "method": "caching",
        "cache_size": cache_size,
        "total_requests": cache_size * 2,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "hit_rate": cache_hits / (cache_hits + cache_misses),
        "execution_time": end_time - start_time,
        "cpu_usage": psutil.cpu_percent(interval=1)
    }


def main():
    """主函数"""
    print("开始CPU使用率优化测试...")

    optimization_results = {
        "test_time": time.time(),
        "optimizations": []
    }

    # 测试并行处理优化
    print("\n1. 测试并行处理优化:")
    parallel_result = optimize_strategy_calculation_parallel()
    optimization_results["optimizations"].append(parallel_result)
    print(f"   执行时间: {parallel_result['execution_time']:.2f}秒")
    print(f"   CPU使用率: {parallel_result['cpu_usage']}%")

    # 测试向量化优化
    print("\n2. 测试向量化优化:")
    vectorization_result = optimize_with_vectorization()
    optimization_results["optimizations"].append(vectorization_result)
    print(f"   执行时间: {vectorization_result['execution_time']:.2f}秒")
    print(f"   CPU使用率: {vectorization_result['cpu_usage']}%")

    # 测试缓存优化
    print("\n3. 测试缓存优化:")
    caching_result = optimize_with_caching()
    optimization_results["optimizations"].append(caching_result)
    print(f"   缓存命中率: {caching_result['hit_rate']:.2%}")
    print(f"   CPU使用率: {caching_result['cpu_usage']}%")

    # 保存结果
    with open('cpu_optimization_results.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(optimization_results, f, indent=2, ensure_ascii=False)

    print("\nCPU优化测试完成，结果已保存到 cpu_optimization_results.json")

    return optimization_results


if __name__ == '__main__':
    main()
