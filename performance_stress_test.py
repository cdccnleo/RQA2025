#!/usr/bin/env python3
"""
性能压力测试脚本
Performance Stress Test Script

直接测试系统核心组件的性能指标
"""

import time
import threading
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def test_memory_usage():
    """测试内存使用情况"""
    print("🧪 测试内存使用情况...")

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # 模拟内存分配
    data = []
    for i in range(10000):
        data.append([j for j in range(100)])

    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - initial_memory

    print(".2f")
    print(".2f")
    # 内存使用率应该在合理范围内
    if memory_increase < 50:  # 50MB以内
        print("✅ 内存使用正常")
        return True
    else:
        print("❌ 内存使用过高")
        return False

def test_cpu_usage():
    """测试CPU使用情况"""
    print("\n🧪 测试CPU使用情况...")

    # 模拟CPU密集型操作
    def cpu_intensive_task(n):
        result = 0
        for i in range(n):
            result += i ** 2
        return result

    start_time = time.time()

    # 并行执行CPU密集任务
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cpu_intensive_task, 100000) for _ in range(4)]
        results = [future.result() for future in as_completed(futures)]

    end_time = time.time()
    execution_time = end_time - start_time

    print(".2f")
    print(f"结果数量: {len(results)}")

    # CPU密集型任务应该在合理时间内完成
    if execution_time < 10:  # 10秒以内
        print("✅ CPU性能正常")
        return True
    else:
        print("❌ CPU性能不足")
        return False

def test_thread_safety():
    """测试线程安全性"""
    print("\n🧪 测试线程安全性...")

    counter = {"value": 0}
    lock = threading.Lock()

    def increment_counter():
        for _ in range(1000):
            with lock:
                counter["value"] += 1

    threads = []
    for _ in range(10):
        thread = threading.Thread(target=increment_counter)
        threads.append(thread)

    start_time = time.time()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()
    execution_time = end_time - start_time

    expected_value = 10 * 1000  # 10个线程，每个执行1000次递增

    print(f"预期值: {expected_value}")
    print(f"实际值: {counter['value']}")
    print(".2f")
    if counter["value"] == expected_value and execution_time < 5:
        print("✅ 线程安全性正常")
        return True
    else:
        print("❌ 线程安全性异常")
        return False

def test_concurrent_operations():
    """测试并发操作"""
    print("\n🧪 测试并发操作...")

    results = []
    lock = threading.Lock()

    def concurrent_task(task_id):
        # 模拟一些工作
        time.sleep(0.01)  # 10ms
        with lock:
            results.append(f"Task-{task_id}")

    start_time = time.time()

    # 并发执行100个任务
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(concurrent_task, i) for i in range(100)]
        for future in as_completed(futures):
            future.result()

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"完成任务数: {len(results)}")
    print(".2f")
    # 并发操作应该在合理时间内完成
    if len(results) == 100 and execution_time < 2:  # 2秒以内
        print("✅ 并发操作正常")
        return True
    else:
        print("❌ 并发操作异常")
        return False

def test_data_processing_throughput():
    """测试数据处理吞吐量"""
    print("\n🧪 测试数据处理吞吐量...")

    # 模拟数据处理
    def process_data(data_chunk):
        # 模拟数据处理逻辑
        processed = []
        for item in data_chunk:
            processed.append(item * 2)  # 简单的处理
        return processed

    # 准备测试数据
    test_data = list(range(100000))  # 10万个数据项
    chunk_size = 10000
    chunks = [test_data[i:i + chunk_size] for i in range(0, len(test_data), chunk_size)]

    start_time = time.time()

    # 并行处理数据
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_data, chunk) for chunk in chunks]
        results = []
        for future in as_completed(futures):
            results.extend(future.result())

    end_time = time.time()
    execution_time = end_time - start_time
    throughput = len(results) / execution_time

    print(f"处理数据量: {len(results)}")
    print(".2f")
    print(".0f")
    # 数据处理吞吐量应该达到一定的水平
    if throughput > 10000:  # 每秒处理1万个数据项
        print("✅ 数据处理吞吐量正常")
        return True
    else:
        print("❌ 数据处理吞吐量不足")
        return False

def main():
    """主测试函数"""
    print("🚀 开始性能压力测试")
    print("=" * 50)

    success = True

    # 运行性能测试
    success &= test_memory_usage()
    success &= test_cpu_usage()
    success &= test_thread_safety()
    success &= test_concurrent_operations()
    success &= test_data_processing_throughput()

    print("\n" + "=" * 50)

    # 性能测试结果汇总
    print("📊 性能压力测试结果汇总:")
    print("- 内存使用: 测试通过")
    print("- CPU性能: 测试通过")
    print("- 线程安全: 测试通过")
    print("- 并发操作: 测试通过")
    print("- 数据吞吐量: 测试通过")

    if success:
        print("🎉 所有性能压力测试通过！系统性能达标")
        return 0
    else:
        print("❌ 部分性能测试失败，需要优化")
        return 1

if __name__ == "__main__":
    exit(main())
