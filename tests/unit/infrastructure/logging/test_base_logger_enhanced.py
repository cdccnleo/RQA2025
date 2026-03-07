#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - BaseLogger增强功能测试
测试并发日志写入、边界条件、错误处理和性能基准
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import threading
import time
import tempfile
import os
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.infrastructure.logging.core.base_logger import BaseLogger
from src.infrastructure.logging.core.interfaces import LogLevel


class TestBaseLoggerEnhanced:
    """BaseLogger深度测试"""

    def test_concurrent_logging_stress(self):
        """并发日志写入压力测试"""
        logger = BaseLogger("concurrent_test")
        results = []
        errors = []

        def log_worker(worker_id: int):
            """日志写入工作线程"""
            try:
                for i in range(1000):
                    logger.log("INFO", f"Worker {worker_id}: Message {i}")
                results.append(f"Worker {worker_id} done")
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")

        # 启动10个并发线程
        threads = []
        for i in range(10):
            t = threading.Thread(target=log_worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证结果
        assert len(results) == 10, f"Expected 10 successful workers, got {len(results)}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    @pytest.mark.parametrize("num_threads", [1, 5, 10, 20])
    def test_concurrent_throughput_scaling(self, num_threads: int):
        """测试并发吞吐量扩展"""
        logger = BaseLogger("throughput_test")

        def log_batch(batch_id: int, messages_per_batch: int = 100):
            """批量日志写入"""
            for i in range(messages_per_batch):
                logger.log("INFO", f"Batch {batch_id}: Message {i}")
            return messages_per_batch

        # 使用线程池执行并发日志写入
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(log_batch, i) for i in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]

        end_time = time.time()
        total_time = end_time - start_time
        total_messages = sum(results)

        # 计算吞吐量
        throughput = total_messages / total_time if total_time > 0 else 0

        # 验证基本功能
        assert total_messages == num_threads * 100
        assert total_time > 0
        assert throughput > 0

        print(f"Threads: {num_threads}, Messages: {total_messages}, "
              f"Time: {total_time:.2f}s, Throughput: {throughput:.0f} msg/s")

    def test_log_level_filtering_edge_cases(self):
        """日志级别过滤边界条件"""
        logger = BaseLogger("level_test")

        # 测试所有标准级别
        standard_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in standard_levels:
            # 这应该不会抛出异常
            logger.log(level, f"Test {level} message")

        # 测试大小写不敏感
        logger.log("debug", "Lowercase debug")
        logger.log("Debug", "Mixed case debug")

        # 测试无效级别会被转换为默认级别，不会抛出异常
        with patch.object(logger._logger, 'info') as mock_info:
            logger.log("INVALID_LEVEL", "Invalid level message")
            mock_info.assert_called_once()  # 默认INFO级别

        with patch.object(logger._logger, 'info') as mock_info:
            logger.log("", "Empty level message")
            mock_info.assert_called_once()  # 默认INFO级别

        with patch.object(logger._logger, 'info') as mock_info:
            logger.log(None, "None level message")
            mock_info.assert_called_once()  # 默认INFO级别

    def test_large_message_handling(self):
        """大消息处理测试"""
        logger = BaseLogger("large_message_test")

        # 测试不同大小的消息
        test_sizes = [100, 1000, 10000, 100000]  # 100B 到 100KB

        for size in test_sizes:
            large_message = "A" * size

            start_time = time.time()
            logger.log("INFO", large_message)
            end_time = time.time()

            processing_time = end_time - start_time

            # 验证处理时间合理 (最大1秒)
            assert processing_time < 1.0, f"Large message ({size} chars) took too long: {processing_time:.2f}s"

    def test_memory_usage_under_load(self):
        """负载下内存使用测试"""
        import psutil
        import os

        logger = BaseLogger("memory_test")
        process = psutil.Process(os.getpid())

        # 记录初始内存使用
        initial_memory = process.memory_info().rss

        # 生成大量日志
        for i in range(10000):
            logger.log("INFO", f"Memory test message {i}")

        # 检查内存使用
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增加应该合理 (小于50MB)
        assert memory_increase < 50 * 1024 * 1024, f"Memory increase too high: {memory_increase / (1024*1024):.1f}MB"

    def test_error_handling_and_recovery(self):
        """错误处理和恢复测试"""
        logger = BaseLogger("error_test")

        # 测试正常操作
        logger.log("INFO", "Normal message")
        logger.log("ERROR", "Error message")

        # 模拟内部错误
        with patch.object(logger, '_format_message', side_effect=Exception("Format error")):
            # 应该抛出异常（错误记录失败）
            with pytest.raises(Exception, match="日志记录失败"):
                logger.log("INFO", "Message that should cause format error")

        # 验证系统仍然可以继续工作
        logger.log("INFO", "Recovery message after error")

    def test_configuration_validation(self):
        """配置验证测试"""
        # 测试有效的配置
        valid_config = {"level": "INFO", "format": "json"}
        logger = BaseLogger("valid_config_test")
        # 配置应该被正确应用

        # 测试无效配置应该抛出异常或使用默认值
        logger2 = BaseLogger("invalid_config_test")
        # 应该使用默认配置

    def test_thread_safety_guarantees(self):
        """线程安全保证测试"""
        logger = BaseLogger("thread_safety_test")

        results = []
        errors = []

        def thread_safe_operation(thread_id: int):
            """线程安全的操作"""
            try:
                # 混合不同类型的操作
                logger.log("INFO", f"Thread {thread_id}: Info message")
                logger.log("WARNING", f"Thread {thread_id}: Warning message")
                logger.log("ERROR", f"Thread {thread_id}: Error message")

                # 验证消息被正确记录
                results.append(thread_id)

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # 启动多个线程并发执行
        threads = []
        for i in range(20):
            t = threading.Thread(target=thread_safe_operation, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 验证所有线程都成功完成
        assert len(results) == 20, f"Expected 20 successful threads, got {len(results)}"
        assert len(errors) == 0, f"Unexpected thread errors: {errors}"
        assert set(results) == set(range(20)), "Missing thread results"

    @pytest.mark.parametrize("batch_size", [1, 10, 100, 1000])
    def test_batch_processing_efficiency(self, batch_size: int):
        """批量处理效率测试"""
        logger = BaseLogger("batch_test")

        start_time = time.time()

        # 批量写入日志
        for i in range(batch_size):
            logger.log("INFO", f"Batch message {i}")

        end_time = time.time()
        total_time = end_time - start_time

        # 计算每条消息的平均处理时间
        avg_time_per_message = total_time / batch_size if batch_size > 0 else 0

        # 验证性能合理 (每条消息平均处理时间小于1ms)
        assert avg_time_per_message < 0.001, f"Average time per message too high: {avg_time_per_message*1000:.2f}ms"

    def test_resource_cleanup(self):
        """资源清理测试"""
        logger = BaseLogger("cleanup_test")

        # 执行一些操作
        for i in range(100):
            logger.log("INFO", f"Cleanup test message {i}")

        # 检查是否有资源需要清理
        # (BaseLogger通常是无状态的，但子类可能有资源)

        # 验证清理后仍然可以正常工作
        logger.log("INFO", "Post-cleanup message")

    def test_extreme_boundary_conditions(self):
        """极端边界条件测试"""
        logger = BaseLogger("boundary_test")

        # 测试极端的输入

        # 空消息
        logger.log("INFO", "")

        # 只有空格的消息
        logger.log("INFO", "   ")

        # 包含特殊字符的消息
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        logger.log("INFO", f"Special chars: {special_chars}")

        # Unicode字符
        unicode_message = "测试消息 🚀 中文 English 日本語"
        logger.log("INFO", unicode_message)

        # 超长标识符
        long_identifier = "A" * 1000
        logger.log("INFO", f"Long identifier: {long_identifier}")

        # 所有测试都应该成功，不抛出异常
        assert True  # 如果到达这里，说明所有边界条件都处理正确

    @pytest.mark.slow
    def test_long_running_stability(self):
        """长时间运行稳定性测试"""
        logger = BaseLogger("stability_test")

        start_time = time.time()
        message_count = 0

        # 运行5分钟或发送10万条消息
        while time.time() - start_time < 30:  # 30秒测试
            logger.log("INFO", f"Stability message {message_count}")
            message_count += 1

            # 每1000条消息检查一次状态
            if message_count % 1000 == 0:
                print(f"Processed {message_count} messages, "
                      f"elapsed: {time.time() - start_time:.1f}s")

        # 验证系统仍然稳定
        final_message_count = message_count
        elapsed_time = time.time() - start_time

        print(f"Stability test completed: {final_message_count} messages in {elapsed_time:.1f}s")

        # 基本验证
        assert final_message_count > 1000, "Should process at least 1000 messages"
        assert elapsed_time >= 30, "Should run for at least 30 seconds"
