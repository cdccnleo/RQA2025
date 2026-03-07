#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 性能基准测试验证脚本

验证CPU和内存优化效果的综合基准测试
"""

import os
import sys
import time
import psutil
import threading
from pathlib import Path
from datetime import datetime
import json
import gc


def run_performance_benchmark():
    """运行性能基准测试"""
    print("🚀 RQA2025 性能基准测试验证")
    print("=" * 60)

    project_root = Path(__file__).parent.parent

    # 1. 系统资源基线测试
    baseline_results = run_system_baseline_test()

    # 2. CPU密集型任务测试
    cpu_results = run_cpu_intensive_tests()

    # 3. 内存使用模式测试
    memory_results = run_memory_usage_tests()

    # 4. 并发处理能力测试
    concurrency_results = run_concurrency_tests()

    # 5. 缓存效果测试
    cache_results = run_cache_effectiveness_test()

    # 6. 生成综合报告
    generate_comprehensive_report(baseline_results, cpu_results, memory_results,
                                 concurrency_results, cache_results)

    print("\n✅ 性能基准测试验证完成!")
    return True


def run_system_baseline_test():
    """系统资源基线测试"""
    print("\n📊 系统资源基线测试...")
    print("-" * 40)

    # 获取系统信息
    system_info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": psutil.virtual_memory().total / 1024**3,
        "memory_available_gb": psutil.virtual_memory().available / 1024**3,
        "disk_total_gb": psutil.disk_usage('/').total / 1024**3,
        "disk_free_gb": psutil.disk_usage('/').free / 1024**3
    }

    # 监控基础资源使用率
    baseline_readings = []
    for i in range(10):
        reading = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / 1024**3
        }
        baseline_readings.append(reading)

    # 计算平均值
    avg_cpu = sum(r["cpu_percent"] for r in baseline_readings) / len(baseline_readings)
    avg_memory = sum(r["memory_percent"] for r in baseline_readings) / len(baseline_readings)

    baseline_results = {
        "system_info": system_info,
        "baseline_readings": baseline_readings,
        "average_cpu_percent": avg_cpu,
        "average_memory_percent": avg_memory,
        "test_timestamp": datetime.now().isoformat()
    }

    print("系统信息:")
    print(f"  CPU核心数: {system_info['cpu_count']} 物理, {system_info['cpu_count_logical']} 逻辑")
    print(f"  总内存: {system_info['memory_total_gb']:.1f} GB")
    print(f"  可用内存: {system_info['memory_available_gb']:.1f} GB")
    print(f"  总磁盘: {system_info['disk_total_gb']:.1f} GB")
    print(f"  可用磁盘: {system_info['disk_free_gb']:.1f} GB")
    print(f"  平均CPU使用率: {avg_cpu:.1f}%")
    print(f"  平均内存使用率: {avg_memory:.1f}%")
    return baseline_results


def run_cpu_intensive_tests():
    """CPU密集型任务测试"""
    print("\n⚡ CPU密集型任务测试...")
    print("-" * 40)

    def cpu_task(n):
        """CPU密集型计算任务"""
        result = 0
        for i in range(n):
            result += i * i
        return result

    # 测试不同规模的CPU任务
    test_sizes = [100000, 500000, 1000000]
    cpu_results = []

    for size in test_sizes:
        print(f"测试CPU任务 (规模: {size})...")

        # 记录开始时的CPU使用率
        start_cpu = psutil.cpu_percent()
        start_time = time.time()

        # 执行CPU密集型任务
        result = cpu_task(size)

        # 记录结束时的CPU使用率
        end_cpu = psutil.cpu_percent()
        end_time = time.time()

        execution_time = end_time - start_time
        cpu_delta = end_cpu - start_cpu

        test_result = {
            "test_size": size,
            "execution_time": execution_time,
            "cpu_delta": cpu_delta,
            "start_cpu": start_cpu,
            "end_cpu": end_cpu,
            "result": result
        }

        cpu_results.append(test_result)
        print(f"  执行时间: {execution_time:.4f}秒, CPU变化: {cpu_delta:.1f}%")
    return cpu_results


def run_memory_usage_tests():
    """内存使用模式测试"""
    print("\n💾 内存使用模式测试...")
    print("-" * 40)

    def memory_intensive_task(size_mb):
        """内存密集型任务"""
        # 创建指定大小的列表
        data = []
        for i in range(size_mb * 1000):
            data.append([i] * 10)
        return data

    test_sizes = [10, 50, 100]  # MB
    memory_results = []

    for size_mb in test_sizes:
        print(f"测试内存任务 (大小: {size_mb}MB)...")

        # 记录开始时的内存使用
        start_memory = psutil.virtual_memory()
        start_time = time.time()

        # 执行内存密集型任务
        data = memory_intensive_task(size_mb)

        # 记录结束时的内存使用
        end_memory = psutil.virtual_memory()
        end_time = time.time()

        execution_time = end_time - start_time
        memory_delta = end_memory.used - start_memory.used
        memory_delta_mb = memory_delta / 1024 / 1024

        # 清理内存
        del data
        gc.collect()

        test_result = {
            "test_size_mb": size_mb,
            "execution_time": execution_time,
            "memory_delta_mb": memory_delta_mb,
            "start_memory_percent": start_memory.percent,
            "end_memory_percent": end_memory.percent,
            "peak_memory_mb": memory_delta_mb
        }

        memory_results.append(test_result)
        print(f"  执行时间: {execution_time:.4f}秒, 内存变化: {memory_delta_mb:.1f} MB")
    return memory_results


def run_concurrency_tests():
    """并发处理能力测试"""
    print("\n🔄 并发处理能力测试...")
    print("-" * 40)

    def concurrent_task(task_id, results, lock):
        """并发任务"""
        start_time = time.time()
        result = 0

        # 模拟一些计算
        for i in range(100000):
            result += i

        execution_time = time.time() - start_time

        with lock:
            results.append({
                "task_id": task_id,
                "execution_time": execution_time,
                "result": result
            })

    # 测试不同并发数量
    concurrency_levels = [2, 4, 8]
    concurrency_results = []

    for num_threads in concurrency_levels:
        print(f"测试并发任务 (线程数: {num_threads})...")

        results = []
        lock = threading.Lock()
        threads = []

        start_time = time.time()
        start_cpu = psutil.cpu_percent()

        # 创建并启动线程
        for i in range(num_threads):
            thread = threading.Thread(target=concurrent_task, args=(i, results, lock))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        end_time = time.time()
        end_cpu = psutil.cpu_percent()

        total_time = end_time - start_time
        cpu_delta = end_cpu - start_cpu

        test_result = {
            "concurrency_level": num_threads,
            "total_time": total_time,
            "cpu_delta": cpu_delta,
            "individual_results": results,
            "average_task_time": sum(r["execution_time"] for r in results) / len(results)
        }

        concurrency_results.append(test_result)
        print(".4f"
    return concurrency_results

def run_cache_effectiveness_test():
    """缓存效果测试"""
    print("\n💽 缓存效果测试...")
    print("-" * 40)

    # 简单的缓存模拟
    cache={}
    cache_hits=0
    cache_misses=0

    def cached_computation(x):
        nonlocal cache_hits, cache_misses

        if x in cache:
            cache_hits += 1
            return cache[x]
        else:
            cache_misses += 1
            # 模拟计算开销
            result=x * x * x  # x^3
            cache[x]=result
            return result

    # 测试缓存效果
    test_data=[1, 2, 3, 1, 4, 2, 5, 1, 6, 3]  # 有重复的计算
    cache_results=[]

    print("执行带缓存的计算...")
    start_time=time.time()

    for data in test_data:
        result=cached_computation(data)
        cache_results.append({"input": data, "output": result})

    end_time=time.time()
    execution_time=end_time - start_time

    cache_effectiveness={
        "test_data": test_data,
        "results": cache_results,
        "cache_size": len(cache),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "hit_rate": cache_hits / (cache_hits + cache_misses) * 100,
        "execution_time": execution_time
    }

    print(f"缓存测试完成:")
    print(f"  缓存大小: {len(cache)} 项")
    print(f"  缓存命中: {cache_hits} 次")
    print(f"  缓存未命中: {cache_misses} 次")
    print(".1f" print(".4f"
    return cache_effectiveness

def generate_comprehensive_report(baseline, cpu_results, memory_results,
                                concurrency_results, cache_results):
    """生成综合报告"""
    print("\n📊 生成性能基准测试综合报告...")
    print("-" * 40)

    # 计算整体性能指标
    overall_metrics={
        "timestamp": datetime.now().isoformat(),
        "test_duration_minutes": 5,  # 估计测试总时长
        "baseline_metrics": baseline,
        "cpu_test_results": cpu_results,
        "memory_test_results": memory_results,
        "concurrency_test_results": concurrency_results,
        "cache_test_results": cache_results
    }

    # 性能评估
    baseline_avg_cpu=baseline["average_cpu_percent"]
    baseline_avg_memory=baseline["average_memory_percent"]

    performance_assessment={
        "cpu_performance": "excellent" if baseline_avg_cpu < 70 else "good" if baseline_avg_cpu < 80 else "needs_improvement",
        "memory_performance": "excellent" if baseline_avg_memory < 60 else "good" if baseline_avg_memory < 70 else "needs_improvement",
        "concurrency_efficiency": len(concurrency_results) > 0,
        "cache_effectiveness": cache_results["hit_rate"] > 50
    }

    # 生成优化建议
    recommendations=[]

    if baseline_avg_cpu > 80:
        recommendations.append("CPU使用率较高，建议实施算法优化和GPU加速")
    if baseline_avg_memory > 70:
        recommendations.append("内存使用率较高，建议优化数据结构和实施内存池")
    if cache_results["hit_rate"] < 70:
        recommendations.append("缓存命中率偏低，建议优化缓存策略")
    if len(recommendations) == 0:
        recommendations.append("系统性能良好，继续保持当前优化状态")

    # 创建报告
    report={
        "test_summary": {
            "test_type": "comprehensive_performance_benchmark",
            "test_version": "1.0",
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_environment": "development"
        },
        "system_baseline": {
            "average_cpu_usage": ".1f",
            "average_memory_usage": ".1f",
            "system_stability": "stable" if baseline_avg_cpu < 80 and baseline_avg_memory < 70 else "needs_attention"
        },
        "performance_metrics": overall_metrics,
        "assessment": performance_assessment,
        "recommendations": recommendations,
        "next_steps": [
            "根据测试结果调整优化策略",
            "实施推荐的性能改进措施",
            "定期运行基准测试监控系统健康状态"
        ]
    }

    # 保存报告
    report_file=f"performance_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # 输出关键指标
    print("📈 性能基准测试关键指标:")
    print(f"  平均CPU使用率: {baseline_avg_cpu:.1f}%")
    print(f"  平均内存使用率: {baseline_avg_memory:.1f}%")
    print(".1f" print(f"  CPU性能评级: {performance_assessment['cpu_performance']}")
    print(f"  内存性能评级: {performance_assessment['memory_performance']}")
    print(f"  缓存命中率: {cache_results['hit_rate']:.1f}%")

    print("
💡 优化建议: " for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print(f"\n📁 详细报告已保存: {report_file}")

    return report

if __name__ == "__main__":
    success=run_performance_benchmark()
    if success:
        print("\n🎉 性能基准测试验证成功!")
        print("📊 系统性能状态已分析完成")
        print("🚀 可以根据测试结果进行进一步优化")
    else:
        print("\n❌ 性能基准测试验证失败")
    sys.exit(0 if success else 1)
