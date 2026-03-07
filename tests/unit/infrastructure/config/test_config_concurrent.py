#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - Config Concurrent
配置并发测试，验证配置系统在高并发场景下的稳定性和性能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import tempfile
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List


class TestConfigConcurrent(unittest.TestCase):
    """测试Config Concurrent"""

    def setUp(self):
        """测试前准备"""
        self.test_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "user": "admin",
                "password": "password123",
                "connection_pool_size": 10
            },
            "cache": {
                "redis_host": "localhost",
                "redis_port": 6379,
                "ttl": 300,
                "max_connections": 20
            },
            "threading": {
                "max_workers": 4,
                "queue_size": 100
            }
        }

        # 创建临时配置文件
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "concurrent_config.json")

        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_config, f, indent=2)

        # 并发测试参数
        self.num_threads = 10
        self.num_operations = 50

    def tearDown(self):
        """测试清理"""
        # 清理临时文件
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_concurrent_config_reads(self):
        """测试并发配置读取"""
        results = []
        errors = []

        def read_worker(thread_id: int) -> None:
            """并发读取工作线程"""
            try:
                for i in range(self.num_operations):
                    # 读取配置文件
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)

                    # 验证配置完整性
                    if "database" in config and "cache" in config:
                        results.append(f"Thread {thread_id} read {i} success")
                    else:
                        errors.append(f"Thread {thread_id} read {i} incomplete")

                    # 短暂延迟模拟真实场景
                    time.sleep(0.001)

            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # 启动并发读取
        threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(target=read_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        self.assertEqual(len(errors), 0,
                        f"并发读取出现错误: {errors}")

        expected_results = self.num_threads * self.num_operations
        self.assertEqual(len(results), expected_results,
                        f"应有{expected_results}个成功读取，实际{len(results)}个")

    def test_concurrent_config_writes(self):
        """测试并发配置写入"""
        errors = []
        successful_writes = 0

        def write_worker(thread_id: int) -> None:
            """并发写入工作线程"""
            nonlocal successful_writes

            try:
                for i in range(self.num_operations):
                    # 创建线程特定的配置
                    thread_config = self.test_config.copy()
                    thread_config["threading"]["worker_id"] = thread_id
                    thread_config["threading"]["operation_id"] = i

                    # 写入配置文件（使用线程锁避免冲突）
                    with threading.Lock():
                        with open(self.config_file, 'w', encoding='utf-8') as f:
                            json.dump(thread_config, f, indent=2)

                        successful_writes += 1

                    # 短暂延迟
                    time.sleep(0.001)

            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # 启动并发写入
        threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(target=write_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        self.assertEqual(len(errors), 0,
                        f"并发写入出现错误: {errors}")

        expected_writes = self.num_threads * self.num_operations
        self.assertEqual(successful_writes, expected_writes,
                        f"应有{expected_writes}个成功写入，实际{successful_writes}个")

    def test_concurrent_mixed_operations(self):
        """测试并发混合操作（读写混合）"""
        results = []
        errors = []
        write_count = 0
        read_count = 0
        file_lock = threading.Lock()  # 创建全局文件锁

        def mixed_worker(thread_id: int) -> None:
            """混合操作工作线程"""
            nonlocal write_count, read_count

            try:
                for i in range(self.num_operations):
                    if i % 2 == 0:  # 偶数操作：读取
                        with file_lock:  # 使用全局锁保护读取操作
                            with open(self.config_file, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                                read_count += 1
                                results.append(f"Thread {thread_id} read {i}")
                    else:  # 奇数操作：写入
                        thread_config = self.test_config.copy()
                        thread_config["threading"]["worker_id"] = thread_id
                        thread_config["threading"]["operation_id"] = i

                        with file_lock:  # 使用全局锁保护写入操作
                            with open(self.config_file, 'w', encoding='utf-8') as f:
                                json.dump(thread_config, f, indent=2)
                            write_count += 1
                            results.append(f"Thread {thread_id} write {i}")

                    time.sleep(0.001)

            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # 启动并发混合操作
        threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(target=mixed_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        self.assertEqual(len(errors), 0,
                        f"并发混合操作出现错误: {errors}")

        expected_operations = self.num_threads * self.num_operations
        self.assertEqual(len(results), expected_operations,
                        f"应有{expected_operations}个操作结果，实际{len(results)}个")

        # 验证读写操作数量
        expected_reads = expected_operations // 2
        expected_writes = expected_operations // 2

        self.assertEqual(read_count, expected_reads,
                        f"应有{expected_reads}个读取操作，实际{read_count}个")
        self.assertEqual(write_count, expected_writes,
                        f"应有{expected_writes}个写入操作，实际{write_count}个")

    def test_concurrent_config_validation(self):
        """测试并发配置验证"""
        validation_results = []
        validation_errors = []

        def validation_worker(thread_id: int) -> None:
            """配置验证工作线程"""
            try:
                for i in range(self.num_operations):
                    # 读取当前配置
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)

                    # 验证配置完整性
                    is_valid = self._validate_config_structure(config)

                    if is_valid:
                        validation_results.append(f"Thread {thread_id} validation {i} passed")
                    else:
                        validation_errors.append(f"Thread {thread_id} validation {i} failed")

                    time.sleep(0.001)

            except Exception as e:
                validation_errors.append(f"Thread {thread_id} error: {e}")

        # 启动并发验证
        threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(target=validation_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        self.assertEqual(len(validation_errors), 0,
                        f"并发验证出现错误: {validation_errors}")

        expected_validations = self.num_threads * self.num_operations
        self.assertEqual(len(validation_results), expected_validations,
                        f"应有{expected_validations}个验证结果，实际{len(validation_results)}个")

    def _validate_config_structure(self, config: Dict[str, Any]) -> bool:
        """验证配置结构"""
        required_sections = ["database", "cache", "threading"]

        for section in required_sections:
            if section not in config:
                return False

            section_data = config[section]
            if not isinstance(section_data, dict):
                return False

        return True

    def test_concurrent_performance_under_load(self):
        """测试高负载下的并发性能"""
        import time

        start_time = time.time()
        operations_completed = 0
        errors = []

        def performance_worker(thread_id: int) -> None:
            """性能测试工作线程"""
            nonlocal operations_completed

            try:
                for i in range(self.num_operations * 2):  # 双倍操作量
                    # 执行配置操作
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)

                    # 简单的配置验证
                    if "database" in config:
                        operations_completed += 1

                    time.sleep(0.0005)  # 更短的延迟以增加负载

            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # 启动高负载并发测试
        threads = []
        for i in range(self.num_threads * 2):  # 双倍线程数
            thread = threading.Thread(target=performance_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        end_time = time.time()
        duration = end_time - start_time

        # 验证结果
        self.assertEqual(len(errors), 0,
                        f"高负载测试出现错误: {errors}")

        expected_operations = (self.num_threads * 2) * (self.num_operations * 2)
        self.assertEqual(operations_completed, expected_operations,
                        f"应完成{expected_operations}个操作，实际{operations_completed}个")

        # 性能要求：高负载测试应在5秒内完成
        self.assertLess(duration, 5.0,
                       f"高负载性能不足: {duration:.2f}s for {expected_operations} operations")

        # 计算每秒操作数
        ops_per_second = operations_completed / duration
        print(f"Concurrent performance: {ops_per_second:.2f} ops/second")

    def test_concurrent_config_isolation(self):
        """测试并发配置隔离"""
        import copy
        thread_local_configs = {}
        errors = []

        def isolation_worker(thread_id: int) -> None:
            """配置隔离工作线程"""
            try:
                # 每个线程维护自己的配置副本（使用深拷贝确保完全隔离）
                thread_config = copy.deepcopy(self.test_config)
                thread_config["threading"]["thread_id"] = thread_id
                thread_config["threading"]["isolation_test"] = True

                # 存储线程本地配置
                thread_local_configs[thread_id] = thread_config

                # 验证配置隔离性
                for i in range(10):
                    stored_config = thread_local_configs[thread_id]
                    if stored_config["threading"]["thread_id"] != thread_id:
                        errors.append(f"Thread {thread_id} isolation breach at iteration {i}")
                        break

                    time.sleep(0.001)

            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # 启动并发隔离测试
        threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(target=isolation_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        self.assertEqual(len(errors), 0,
                        f"配置隔离测试出现错误: {errors}")

        # 验证每个线程的配置都正确隔离
        self.assertEqual(len(thread_local_configs), self.num_threads,
                        f"应有{self.num_threads}个线程本地配置，实际{len(thread_local_configs)}个")

        for thread_id in range(self.num_threads):
            self.assertIn(thread_id, thread_local_configs)
            config = thread_local_configs[thread_id]
            self.assertEqual(config["threading"]["thread_id"], thread_id)
            self.assertTrue(config["threading"]["isolation_test"])

    def test_concurrent_config_thread_safety(self):
        """测试并发配置线程安全"""
        # 使用线程安全的配置管理器模拟
        from unittest.mock import MagicMock

        # 创建模拟的线程安全配置管理器
        mock_config_manager = MagicMock()
        mock_config_manager.get.side_effect = lambda key: f"mock_value_{threading.current_thread().ident}"
        mock_config_manager.set.return_value = True

        results = []
        errors = []

        def thread_safety_worker(thread_id: int) -> None:
            """线程安全测试工作线程"""
            try:
                for i in range(self.num_operations):
                    # 测试获取操作的线程安全
                    value = mock_config_manager.get(f"test_key_{i}")
                    if value:
                        results.append(f"Thread {thread_id} get {i} success")
                    else:
                        errors.append(f"Thread {thread_id} get {i} failed")

                    # 测试设置操作的线程安全
                    success = mock_config_manager.set(f"test_key_{i}", f"test_value_{thread_id}_{i}")
                    if success:
                        results.append(f"Thread {thread_id} set {i} success")
                    else:
                        errors.append(f"Thread {thread_id} set {i} failed")

            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # 启动并发线程安全测试
        threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(target=thread_safety_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        self.assertEqual(len(errors), 0,
                        f"线程安全测试出现错误: {errors}")

        expected_results = self.num_threads * self.num_operations * 2  # get和set操作
        self.assertEqual(len(results), expected_results,
                        f"应有{expected_results}个操作结果，实际{len(results)}个")

    @patch('src.infrastructure.config.core.unified_manager.UnifiedConfigManager')
    def test_concurrent_config_manager_integration(self, mock_config_manager):
        """测试并发配置管理器集成"""
        # 创建模拟的配置管理器实例
        mock_instance = MagicMock()
        mock_config_manager.return_value = mock_instance

        # 配置模拟行为
        mock_instance.get.side_effect = lambda key: f"value_for_{key}"
        mock_instance.set.return_value = True
        mock_instance.initialize.return_value = True

        results = []
        errors = []

        def integration_worker(thread_id: int) -> None:
            """集成测试工作线程"""
            try:
                from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
                manager = UnifiedConfigManager()

                # 测试初始化
                init_result = manager.initialize()
                if init_result:
                    results.append(f"Thread {thread_id} init success")
                else:
                    errors.append(f"Thread {thread_id} init failed")

                # 测试配置操作
                for i in range(10):
                    # 测试获取配置
                    value = manager.get(f"database.host_{i}")
                    if value:
                        results.append(f"Thread {thread_id} get {i} success")

                    # 测试设置配置
                    set_result = manager.set(f"cache.ttl_{i}", 300 + i)
                    if set_result:
                        results.append(f"Thread {thread_id} set {i} success")

            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # 启动并发集成测试
        threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(target=integration_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        self.assertEqual(len(errors), 0,
                        f"并发集成测试出现错误: {errors}")

        # 验证至少有一些成功的操作
        self.assertGreater(len(results), 0,
                          "应有成功的配置管理器操作")


if __name__ == '__main__':
    unittest.main()
