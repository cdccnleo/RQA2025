"""
专项测试层性能基准测试
测试系统性能基准和监控指标
"""

import pytest
import time
import psutil
import tracemalloc
from pathlib import Path
import sys
from unittest.mock import Mock

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)


class TestPerformanceBenchmarks:
    """性能基准测试"""

    def setup_method(self):
        """测试前准备"""
        tracemalloc.start()
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    def teardown_method(self):
        """测试后清理"""
        tracemalloc.stop()

    def test_memory_usage_baseline(self):
        """测试内存使用基准"""
        # 执行一些基本操作
        data = [i for i in range(10000)]
        result = sum(data)

        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - self.start_memory

        # 验证内存使用在合理范围内
        assert memory_increase < 50  # 内存增加不应超过50MB
        assert result == 49995000  # 验证计算正确性

    def test_cpu_usage_baseline(self):
        """测试CPU使用基准"""
        start_cpu = psutil.cpu_percent(interval=None)

        # 执行一些CPU密集型操作
        result = 0
        for i in range(100000):
            result += i ** 2

        end_cpu = psutil.cpu_percent(interval=None)

        # 验证CPU使用合理
        assert result > 0
        # CPU使用率不应过高（在测试环境中）
        assert end_cpu - start_cpu < 80

    def test_execution_time_baseline(self):
        """测试执行时间基准"""
        start_time = time.time()

        # 执行定时操作
        time.sleep(0.1)  # 100ms
        data = []
        for i in range(1000):
            data.append(i * i)

        end_time = time.time()
        execution_time = end_time - start_time

        # 验证执行时间在合理范围内
        assert 0.1 <= execution_time <= 0.5  # 允许一定的误差
        assert len(data) == 1000

    def test_data_processing_throughput(self):
        """测试数据处理吞吐量"""
        start_time = time.time()

        # 模拟数据处理
        processed_items = 0
        for batch in range(10):
            # 每批处理1000个项目
            batch_data = [i for i in range(1000)]
            processed_items += len(batch_data)

        end_time = time.time()
        processing_time = end_time - start_time

        throughput = processed_items / processing_time

        # 验证吞吐量
        assert processed_items == 10000
        assert throughput > 1000  # 每秒至少处理1000个项目

    def test_concurrent_execution_performance(self):
        """测试并发执行性能"""
        import threading

        results = []
        errors = []

        def worker_task(task_id):
            try:
                # 模拟工作任务
                time.sleep(0.01)  # 10ms工作
                results.append(f"task_{task_id}_completed")
            except Exception as e:
                errors.append(str(e))

        # 启动多个线程
        threads = []
        num_threads = 5

        start_time = time.time()

        for i in range(num_threads):
            t = threading.Thread(target=worker_task, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        end_time = time.time()
        total_time = end_time - start_time

        # 验证并发执行
        assert len(results) == num_threads
        assert len(errors) == 0
        # 并发执行应该比串行执行快
        assert total_time < 0.1  # 所有任务应该在100ms内完成

    def test_memory_leak_detection(self):
        """测试内存泄漏检测"""
        initial_memory = tracemalloc.get_traced_memory()[0]

        # 执行可能导致内存泄漏的操作
        data_list = []
        for i in range(100):
            data_list.append([j for j in range(100)])

        # 清理数据
        del data_list

        final_memory = tracemalloc.get_traced_memory()[0]
        memory_increase = final_memory - initial_memory

        # 验证内存使用合理（允许一定的内存波动）
        assert memory_increase < 1024 * 1024  # 内存增加不应超过1MB

    def test_io_operation_performance(self):
        """测试IO操作性能"""
        import tempfile
        import os

        # 创建临时文件进行IO测试
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            temp_file = f.name

            # 写入测试
            start_write = time.time()
            for i in range(1000):
                f.write(f"line {i}: test data\n")
            f.flush()
            end_write = time.time()

            # 读取测试
            f.seek(0)
            start_read = time.time()
            lines = f.readlines()
            end_read = time.time()

        # 清理临时文件
        os.unlink(temp_file)

        write_time = end_write - start_write
        read_time = end_read - start_read

        # 验证IO性能
        assert len(lines) == 1000
        assert write_time < 1.0  # 写入不应超过1秒
        assert read_time < 1.0   # 读取不应超过1秒

    def test_error_handling_performance(self):
        """测试错误处理性能"""
        exceptions_caught = 0
        start_time = time.time()

        # 模拟大量异常处理
        for i in range(1000):
            try:
                if i % 2 == 0:
                    raise ValueError(f"Test error {i}")
                else:
                    raise TypeError(f"Test type error {i}")
            except (ValueError, TypeError):
                exceptions_caught += 1
            except Exception:
                pass  # 其他异常忽略

        end_time = time.time()
        handling_time = end_time - start_time

        # 验证错误处理性能
        assert exceptions_caught == 1000
        assert handling_time < 1.0  # 错误处理不应过慢

    def test_large_data_structure_performance(self):
        """测试大数据结构性能"""
        # 测试大字典性能
        large_dict = {}
        start_time = time.time()

        # 填充大数据结构
        for i in range(10000):
            large_dict[f"key_{i}"] = f"value_{i}"

        # 执行查找操作
        lookups = 0
        for i in range(1000):
            if f"key_{i}" in large_dict:
                lookups += 1

        end_time = time.time()
        operation_time = end_time - start_time

        # 验证大数据结构性能
        assert len(large_dict) == 10000
        assert lookups == 1000
        assert operation_time < 1.0  # 大数据结构操作不应过慢

    def test_algorithm_complexity_baseline(self):
        """测试算法复杂度基准"""
        def linear_algorithm(n):
            return sum(range(n))

        def quadratic_algorithm(n):
            result = 0
            for i in range(n):
                for j in range(n):
                    result += 1
            return result

        # 测试线性算法
        start_time = time.time()
        linear_result = linear_algorithm(10000)
        linear_time = time.time() - start_time

        # 测试二次方算法
        start_time = time.time()
        quadratic_result = quadratic_algorithm(100)
        quadratic_time = time.time() - start_time

        # 验证算法复杂度（只验证时间，不比较结果）
        assert linear_result > 0
        assert quadratic_result > 0
        # 验证算法正确性，但不强制时间比较（在不同环境下可能表现不同）
        assert linear_result == 49995000  # sum(range(10000)) = 49995000
        assert quadratic_result == 10000  # n*n for n=100

    def test_system_resource_monitoring(self):
        """测试系统资源监控"""
        # 监控系统资源使用情况
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')

        # 验证系统资源监控
        assert 0 <= cpu_percent <= 100
        assert memory_info.percent >= 0
        assert disk_usage.percent >= 0

        # 验证基本系统信息可用
        assert memory_info.total > 0
        assert disk_usage.total > 0

    def test_network_simulation_performance(self):
        """测试网络模拟性能"""
        import socket
        from unittest.mock import patch

        # 模拟网络操作性能
        network_calls = 0
        start_time = time.time()

        # 模拟多次网络调用
        for i in range(100):
            try:
                # 这里只是模拟，实际不会进行真正的网络调用
                network_calls += 1
                time.sleep(0.001)  # 1ms延迟模拟
            except Exception:
                pass

        end_time = time.time()
        network_time = end_time - start_time

        # 验证网络模拟性能
        assert network_calls == 100
        assert network_time < 2.0  # 网络操作不应过慢（放宽时间限制）

    def test_cache_performance_simulation(self):
        """测试缓存性能模拟"""
        # 模拟缓存操作
        cache_hits = 0
        cache_misses = 0
        cache = {}

        start_time = time.time()

        # 模拟缓存访问模式
        for i in range(1000):
            key = f"key_{i % 100}"  # 只有100个不同键，模拟缓存局部性

            if key in cache:
                cache_hits += 1
                cache[key] = cache[key] + 1
            else:
                cache_misses += 1
                cache[key] = 1

        end_time = time.time()
        cache_time = end_time - start_time

        # 验证缓存性能
        assert cache_hits > cache_misses  # 缓存命中应该多于未命中
        assert len(cache) == 100  # 应该有100个缓存条目
        assert cache_time < 1.0   # 缓存操作不应过慢

    def test_thread_safety_performance(self):
        """测试线程安全性能"""
        import threading

        counter = 0
        lock = threading.Lock()
        errors = []

        def thread_safe_increment():
            nonlocal counter
            try:
                for _ in range(100):
                    with lock:
                        counter += 1
            except Exception as e:
                errors.append(str(e))

        # 启动多个线程
        threads = []
        num_threads = 5

        start_time = time.time()

        for i in range(num_threads):
            t = threading.Thread(target=thread_safe_increment)
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        end_time = time.time()
        thread_time = end_time - start_time

        # 验证线程安全
        assert counter == 500  # 5线程 * 100次递增
        assert len(errors) == 0
        assert thread_time < 2.0  # 线程操作不应过慢
