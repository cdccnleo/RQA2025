#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存策略优化脚本
"""

import time
import threading
from collections import OrderedDict
import json
import psutil


class LRUCache:
    """LRU缓存实现"""

    def __init__(self, capacity=100):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                # 移动到最后（最近使用）
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key, value):
        """设置缓存项"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.capacity:
                    # 移除最少使用的项
                    self.cache.popitem(last=False)
            self.cache[key] = value


class LFUCache:
    """LFU缓存实现"""

    def __init__(self, capacity=100):
        self.capacity = capacity
        self.cache = {}
        self.freq = {}
        self.lock = threading.Lock()

    def get(self, key):
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                self.freq[key] += 1
                return self.cache[key]
            return None

    def put(self, key, value):
        """设置缓存项"""
        with self.lock:
            if key in self.cache:
                self.cache[key] = value
                self.freq[key] += 1
            else:
                if len(self.cache) >= self.capacity:
                    # 移除访问频率最低的项
                    min_freq = min(self.freq.values())
                    keys_to_remove = [k for k, v in self.freq.items() if v == min_freq]
                    key_to_remove = keys_to_remove[0]

                    del self.cache[key_to_remove]
                    del self.freq[key_to_remove]

                self.cache[key] = value
                self.freq[key] = 1


class MultiLevelCache:
    """多级缓存实现"""

    def __init__(self):
        self.l1_cache = LRUCache(capacity=50)  # L1缓存：高速，小容量
        self.l2_cache = LRUCache(capacity=200)  # L2缓存：中速，大容量
        self.lock = threading.Lock()

    def get(self, key):
        """获取缓存项"""
        # 先查L1缓存
        value = self.l1_cache.get(key)
        if value is not None:
            return value, "L1"

        # 再查L2缓存
        value = self.l2_cache.get(key)
        if value is not None:
            # 提升到L1缓存
            self.l1_cache.put(key, value)
            return value, "L2"

        return None, None

    def put(self, key, value):
        """设置缓存项"""
        with self.lock:
            # 同时写入L1和L2缓存
            self.l1_cache.put(key, value)
            self.l2_cache.put(key, value)


def test_cache_performance(cache_type, cache_instance, num_operations=1000):
    """测试缓存性能"""
    print(f"测试{cache_type}缓存性能...")

    start_memory = psutil.virtual_memory().used
    start_time = time.time()

    hits = 0
    misses = 0

    for i in range(num_operations):
        key = f"key_{i % 100}"  # 循环使用100个不同的键

        if i % 2 == 0:  # 偶数次：写入
            cache_instance.put(key, f"value_{i}")
        else:  # 奇数次：读取
            value = cache_instance.get(key)
            if value is not None:
                hits += 1
            else:
                misses += 1

    end_time = time.time()
    end_memory = psutil.virtual_memory().used

    return {
        "cache_type": cache_type,
        "total_operations": num_operations,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / (hits + misses) if (hits + misses) > 0 else 0,
        "total_time": end_time - start_time,
        "operations_per_second": num_operations / (end_time - start_time),
        "memory_used": end_memory - start_memory
    }


def test_cache_under_concurrent_access(cache_instance, num_threads=5, operations_per_thread=200):
    """测试缓存并发访问性能"""
    print("测试缓存并发访问性能...")

    def worker_thread(thread_id, cache, results):
        hits = 0
        misses = 0

        for i in range(operations_per_thread):
            key = f"thread_{thread_id}_key_{i % 50}"

            if i % 2 == 0:
                cache.put(key, f"thread_{thread_id}_value_{i}")
            else:
                value = cache.get(key)
                if value is not None:
                    hits += 1
                else:
                    misses += 1

        results.append({
            "thread_id": thread_id,
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / (hits + misses) if (hits + misses) > 0 else 0
        })

    start_time = time.time()
    results = []

    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker_thread, args=(i, cache_instance, results))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()

    total_hits = sum(r["hits"] for r in results)
    total_misses = sum(r["misses"] for r in results)
    total_operations = sum(r["hits"] + r["misses"] for r in results)

    return {
        "cache_type": "concurrent_test",
        "num_threads": num_threads,
        "total_operations": total_operations,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "overall_hit_rate": total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0,
        "total_time": end_time - start_time,
        "operations_per_second": total_operations / (end_time - start_time)
    }


def test_cache_memory_efficiency():
    """测试缓存内存效率"""
    print("测试缓存内存效率...")

    # 测试不同缓存策略的内存使用
    cache_strategies = {
        "LRU_50": LRUCache(capacity=50),
        "LRU_200": LRUCache(capacity=200),
        "LFU_50": LFUCache(capacity=50),
        "MultiLevel": MultiLevelCache()
    }

    results = {}

    for name, cache in cache_strategies.items():
        start_memory = psutil.virtual_memory().used

        # 填充缓存
        for i in range(cache.capacity if hasattr(cache, 'capacity') else 100):
            cache.put(f"test_key_{i}", f"test_value_{i}" * 100)

        end_memory = psutil.virtual_memory().used
        memory_used = end_memory - start_memory

        # 测试访问性能
        hits = 0
        for i in range(200):
            key = f"test_key_{i % (cache.capacity if hasattr(cache, 'capacity') else 100)}"
            if cache.get(key) is not None:
                hits += 1

        results[name] = {
            "cache_name": name,
            "memory_used": memory_used,
            "capacity": cache.capacity if hasattr(cache, 'capacity') else 250,  # MultiLevel的总容量
            "memory_per_item": memory_used / (cache.capacity if hasattr(cache, 'capacity') else 250),
            "hit_rate": hits / 200
        }

    return results


def main():
    """主函数"""
    print("开始缓存策略优化测试...")

    optimization_results = {
        "test_time": time.time(),
        "performance_tests": [],
        "concurrent_tests": [],
        "memory_efficiency_tests": {}
    }

    # 测试不同缓存策略的性能
    print("\n1. 测试缓存策略性能:")
    cache_strategies = {
        "LRU缓存": LRUCache(capacity=100),
        "LFU缓存": LFUCache(capacity=100),
        "多级缓存": MultiLevelCache()
    }

    for name, cache in cache_strategies.items():
        result = test_cache_performance(name, cache, num_operations=500)
        optimization_results["performance_tests"].append(result)
        print(f"   {name}: 命中率 {result['hit_rate']:.2%}, OPS {result['operations_per_second']:.0f}")

    # 测试并发访问性能
    print("\n2. 测试并发访问性能:")
    concurrent_cache = MultiLevelCache()
    concurrent_result = test_cache_under_concurrent_access(
        concurrent_cache, num_threads=5, operations_per_thread=100)
    optimization_results["concurrent_tests"].append(concurrent_result)
    print(
        f"   并发访问: 命中率 {concurrent_result['overall_hit_rate']:.2%}, OPS {concurrent_result['operations_per_second']:.0f}")

    # 测试内存效率
    print("\n3. 测试内存效率:")
    memory_results = test_cache_memory_efficiency()
    optimization_results["memory_efficiency_tests"] = memory_results
    for name, result in memory_results.items():
        print(
            f"   {name}: 内存使用 {result['memory_used']/1024/1024:.2f}MB, 命中率 {result['hit_rate']:.2%}")

    # 保存结果
    with open('cache_optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(optimization_results, f, indent=2, ensure_ascii=False)

    print("\n缓存策略优化测试完成，结果已保存到 cache_optimization_results.json")

    return optimization_results


if __name__ == '__main__':
    main()
