#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理并发测试

测试配置管理系统在并发环境下的表现
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock
import pytest

from src.infrastructure.config import UnifiedConfigManager


class TestConfigConcurrency(unittest.TestCase):
    """配置管理并发测试"""

    def setUp(self):
        """测试前准备"""
        self.manager = UnifiedConfigManager({
            "auto_reload": False,
            "validation_enabled": False
        })

    def tearDown(self):
        """测试后清理"""
        if hasattr(self.manager, '_data'):
            self.manager._data.clear()

    @pytest.mark.concurrent
    def test_concurrent_reads(self):
        """测试并发读取"""
        # 准备测试数据
        test_data = {"concurrent": {}}
        for i in range(100):
            test_data["concurrent"][f"section{i}"] = {f"key{j}": f"value{i}_{j}" for j in range(10)}

        self.manager._data = test_data

        results = []
        errors = []

        def read_worker(worker_id: int, num_reads: int):
            """读取工作线程"""
            worker_results = []
            try:
                for i in range(num_reads):
                    # 随机选择一个键进行读取
                    section_idx = random.randint(0, 99)
                    key_idx = random.randint(0, 9)
                    key = f"concurrent.section{section_idx}.key{key_idx}"
                    expected_value = f"value{section_idx}_{key_idx}"

                    value = self.manager.get(key)
                    if value == expected_value:
                        worker_results.append(True)
                    else:
                        worker_results.append(False)

                results.extend(worker_results)

            except Exception as e:
                errors.append(f"Read worker {worker_id}: {e}")

        # 启动多个读取线程
        num_threads = 10
        reads_per_thread = 100

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=read_worker, args=(i, reads_per_thread))
            threads.append(thread)

        start_time = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=30)

        end_time = time.time()

        # 验证结果
        total_reads = len(results)
        successful_reads = sum(results)
        success_rate = successful_reads / total_reads if total_reads > 0 else 0

        print(f"Concurrent reads: {total_reads} total, {successful_reads} successful")
        print(".1f")
        print(".3f")
        self.assertEqual(len(errors), 0, f"Read errors occurred: {errors}")
        self.assertGreater(success_rate, 0.99, "Read success rate too low")

    @pytest.mark.concurrent
    def test_concurrent_writes(self):
        """测试并发写入"""
        results = []
        errors = []

        def write_worker(worker_id: int, num_writes: int):
            """写入工作线程"""
            worker_results = []
            try:
                for i in range(num_writes):
                    key = f"write.thread{worker_id}.iteration{i}"
                    value = f"value{worker_id}_{i}"

                    success = self.manager.set(key, value)
                    worker_results.append(success)

                    # 立即验证写入
                    retrieved = self.manager.get(key)
                    if retrieved == value:
                        worker_results.append(True)
                    else:
                        worker_results.append(False)

                results.extend(worker_results)

            except Exception as e:
                errors.append(f"Write worker {worker_id}: {e}")

        # 启动多个写入线程
        num_threads = 5
        writes_per_thread = 50

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=write_worker, args=(i, writes_per_thread))
            threads.append(thread)

        start_time = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=30)

        end_time = time.time()

        # 验证结果
        total_operations = len(results)
        successful_operations = sum(results)
        success_rate = successful_operations / total_operations if total_operations > 0 else 0

        print(f"Concurrent writes: {total_operations} operations, {successful_operations} successful")
        print(".1f")        
        print(".3f")
        self.assertEqual(len(errors), 0, f"Write errors occurred: {errors}")
        self.assertGreater(success_rate, 0.95, "Write success rate too low")

    @pytest.mark.concurrent
    def test_mixed_read_write_operations(self):
        """测试混合读写操作"""
        # 初始化一些基础数据
        for i in range(50):
            self.manager.set(f"mixed.base{i}", f"initial_value{i}")

        results = []
        errors = []

        def mixed_worker(worker_id: int, num_operations: int):
            """混合操作工作线程"""
            worker_results = []
            try:
                for i in range(num_operations):
                    operation_type = random.choice(['read', 'write', 'update'])

                    if operation_type == 'read':
                        # 读取操作
                        key_idx = random.randint(0, 49)
                        key = f"mixed.base{key_idx}"
                        value = self.manager.get(key)
                        worker_results.append(value is not None)

                    elif operation_type == 'write':
                        # 写入操作
                        key = f"mixed.worker{worker_id}.write{i}"
                        value = f"write_value{worker_id}_{i}"
                        success = self.manager.set(key, value)
                        worker_results.append(success)

                    elif operation_type == 'update':
                        # 更新操作
                        key_idx = random.randint(0, 49)
                        key = f"mixed.base{key_idx}"
                        new_value = f"updated_value{worker_id}_{i}"
                        success = self.manager.set(key, new_value)
                        worker_results.append(success)

                results.extend(worker_results)

            except Exception as e:
                errors.append(f"Mixed worker {worker_id}: {e}")

        # 启动多个混合操作线程
        num_threads = 8
        operations_per_thread = 100

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=mixed_worker, args=(i, operations_per_thread))
            threads.append(thread)

        start_time = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=30)

        end_time = time.time()

        # 验证结果
        total_operations = len(results)
        successful_operations = sum(results)
        success_rate = successful_operations / total_operations if total_operations > 0 else 0

        print(f"Mixed operations: {total_operations} total, {successful_operations} successful")
        print(".1f")        
        print(".3f")
        self.assertEqual(len(errors), 0, f"Mixed operation errors: {errors}")
        self.assertGreater(success_rate, 0.90, "Mixed operation success rate too low")

    @pytest.mark.concurrent
    def test_thread_safety_with_events(self):
        """测试带事件监听的线程安全"""
        event_calls = []
        event_errors = []

        def event_listener(key, value):
            """事件监听器"""
            try:
                event_calls.append((key, value))
            except Exception as e:
                event_errors.append(str(e))

        # 注册事件监听器
        self.manager.watch("concurrent.*", event_listener)

        results = []
        errors = []

        def event_worker(worker_id: int, num_operations: int):
            """带事件的工作线程"""
            worker_results = []
            try:
                for i in range(num_operations):
                    key = f"concurrent.event{worker_id}.key{i}"
                    value = f"event_value{worker_id}_{i}"

                    success = self.manager.set(key, value)
                    worker_results.append(success)

                    # 小延迟以允许事件处理
                    time.sleep(0.001)

                results.extend(worker_results)

            except Exception as e:
                errors.append(f"Event worker {worker_id}: {e}")

        # 启动多个带事件的线程
        num_threads = 3
        operations_per_thread = 20

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=event_worker, args=(i, operations_per_thread))
            threads.append(thread)

        start_time = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=30)

        end_time = time.time()

        # 验证结果
        total_operations = len(results)
        successful_operations = sum(results)
        expected_events = total_operations  # 每个设置操作应该触发一个事件

        print(f"Event operations: {total_operations} operations, {len(event_calls)} events")
        print(f"Event errors: {len(event_errors)}")

        self.assertEqual(len(errors), 0, f"Worker errors: {errors}")
        self.assertEqual(len(event_errors), 0, f"Event errors: {event_errors}")
        self.assertGreaterEqual(len(event_calls), expected_events * 0.8, "Too few events received")

    @pytest.mark.concurrent
    def test_concurrent_file_operations(self):
        """测试并发文件操作"""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            results = []
            errors = []

            def file_worker(worker_id: int, num_operations: int):
                """文件操作工作线程"""
                from src.infrastructure.config.core.config_manager_complete import UnifiedConfigManager
                worker_manager = UnifiedConfigManager()  # 每个线程使用独立的配置管理器
                worker_results = []

                try:
                    for i in range(num_operations):
                        # 创建唯一的文件名
                        filename = f"config_worker{worker_id}_op{i}.json"
                        filepath = os.path.join(temp_dir, filename)

                        # 设置一些测试数据
                        test_data = {f"file.key{j}": f"file_value{worker_id}_{i}_{j}" for j in range(5)}
                        for key, value in test_data.items():
                            worker_manager.set(key, value)

                        # 保存到文件
                        success = worker_manager.save_to_file(filepath)
                        if success:
                            # 清空配置并重新加载
                            worker_manager.clear_all()
                            load_success = worker_manager.load_from_file(filepath)

                            if load_success:
                                # 验证数据完整性
                                data_integrity = True
                                for key, expected_value in test_data.items():
                                    actual_value = worker_manager.get(key)
                                    if actual_value != expected_value:
                                        data_integrity = False
                                        break

                                worker_results.append(data_integrity)
                            else:
                                worker_results.append(False)
                        else:
                            worker_results.append(False)

                    results.extend(worker_results)

                except Exception as e:
                    errors.append(f"File worker {worker_id}: {e}")

            # 启动多个文件操作线程
            num_threads = 3
            operations_per_thread = 5

            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=file_worker, args=(i, operations_per_thread))
                threads.append(thread)

            start_time = time.time()
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join(timeout=60)  # 文件操作可能需要更长时间

            end_time = time.time()

            # 验证结果
            total_operations = len(results)
            successful_operations = sum(results)
            success_rate = successful_operations / total_operations if total_operations > 0 else 0

            print(f"Concurrent file operations: {total_operations} total, {successful_operations} successful")
            print(".1f")            
            print(".3f")
            self.assertEqual(len(errors), 0, f"File operation errors: {errors}")
            self.assertGreater(success_rate, 0.85, "File operation success rate too low")

    @pytest.mark.concurrent
    def test_resource_contention(self):
        """测试资源竞争"""
        # 测试在高竞争情况下的表现
        contention_results = []
        contention_errors = []

        # 创建一个共享的配置键
        shared_key = "contention.shared_key"
        self.manager.set(shared_key, "initial_value")

        def contention_worker(worker_id: int, num_operations: int):
            """资源竞争工作线程"""
            worker_results = []
            try:
                for i in range(num_operations):
                    # 读取共享值
                    current_value = self.manager.get(shared_key)

                    # 修改值
                    new_value = f"worker{worker_id}_op{i}"
                    success = self.manager.set(shared_key, new_value)
                    worker_results.append(success)

                    # 短暂延迟以增加竞争
                    time.sleep(0.001)

                    # 再次读取以验证
                    final_value = self.manager.get(shared_key)
                    # 注意：由于并发，其他线程可能已经修改了值

                contention_results.append(len(worker_results))

            except Exception as e:
                contention_errors.append(f"Contention worker {worker_id}: {e}")

        # 启动多个竞争线程
        num_threads = 5
        operations_per_thread = 20

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=contention_worker, args=(i, operations_per_thread))
            threads.append(thread)

        start_time = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=30)

        end_time = time.time()

        # 验证结果
        total_results = sum(contention_results)
        expected_operations = num_threads * operations_per_thread

        print(f"Resource contention: {total_results}/{expected_operations} operations completed")
        print(f"Contention errors: {len(contention_errors)}")

        self.assertEqual(len(contention_errors), 0, f"Contention errors: {contention_errors}")
        self.assertGreaterEqual(total_results, expected_operations * 0.9, "Too many operations failed due to contention")

    @pytest.mark.concurrent
    def test_concurrent_validation(self):
        """测试并发验证"""
        # 准备验证规则
        self.manager.config["validation_rules"] = {
            "concurrent.*": {
                "type": "string",
                "pattern": r"^worker\d+_value\d+$"
            }
        }
        self.manager.config["validation_enabled"] = True

        validation_results = []
        validation_errors = []

        def validation_worker(worker_id: int, num_operations: int):
            """验证工作线程"""
            worker_results = []
            try:
                for i in range(num_operations):
                    key = f"concurrent.worker{worker_id}.key{i}"
                    value = f"worker{worker_id}_value{i}"  # 符合验证规则的格式

                    # 设置值（会触发验证）
                    success = self.manager.set(key, value)
                    worker_results.append(success)

                    # 验证配置
                    is_valid = self.manager.validate_config()
                    worker_results.append(is_valid)

                validation_results.extend(worker_results)

            except Exception as e:
                validation_errors.append(f"Validation worker {worker_id}: {e}")

        # 启动多个验证线程
        num_threads = 4
        operations_per_thread = 25

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=validation_worker, args=(i, operations_per_thread))
            threads.append(thread)

        start_time = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=30)

        end_time = time.time()

        # 验证结果
        total_results = len(validation_results)
        successful_results = sum(validation_results)
        success_rate = successful_results / total_results if total_results > 0 else 0

        print(f"Concurrent validation: {total_results} checks, {successful_results} passed")
        print(".1f")        
        print(".3f")
        self.assertEqual(len(validation_errors), 0, f"Validation errors: {validation_errors}")
        self.assertGreater(success_rate, 0.4, "Validation success rate too low")

    @pytest.mark.concurrent
    def test_deadlock_prevention(self):
        """测试死锁预防"""
        # 测试潜在的死锁场景
        deadlock_results = []
        deadlock_errors = []

        def deadlock_worker(worker_id: int):
            """死锁测试工作线程"""
            try:
                # 执行一系列可能导致死锁的操作
                for i in range(10):
                    # 设置多个相关键
                    keys = [f"deadlock.worker{worker_id}.key{j}" for j in range(5)]
                    for key in keys:
                        self.manager.set(key, f"value{worker_id}_{i}")

                    # 读取所有键
                    for key in keys:
                        self.manager.get(key)

                    # 执行验证
                    self.manager.validate_config()

                    # 小延迟以增加交织机会
                    time.sleep(0.001)

                deadlock_results.append(True)

            except Exception as e:
                deadlock_errors.append(f"Deadlock worker {worker_id}: {e}")
                deadlock_results.append(False)

        # 启动多个可能死锁的线程
        num_threads = 5

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=deadlock_worker, args=(i,))
            threads.append(thread)

        start_time = time.time()
        for thread in threads:
            thread.start()

        # 等待所有线程完成，设置超时
        for thread in threads:
            thread.join(timeout=60)  # 60秒超时，防止死锁

        end_time = time.time()

        # 检查是否有线程没有完成（可能死锁）
        active_threads = sum(1 for thread in threads if thread.is_alive())

        successful_workers = sum(deadlock_results)
        total_workers = len(deadlock_results)

        print(f"Deadlock test: {successful_workers}/{total_workers} workers completed successfully")
        print(f"Active threads after timeout: {active_threads}")
        print(".3f")
        self.assertEqual(active_threads, 0, "Possible deadlock: threads still active")
        self.assertEqual(len(deadlock_errors), 0, f"Deadlock test errors: {deadlock_errors}")
        self.assertEqual(successful_workers, total_workers, "Some workers failed to complete")


if __name__ == '__main__':
    unittest.main()
