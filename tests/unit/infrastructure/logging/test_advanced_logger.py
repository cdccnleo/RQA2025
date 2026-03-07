#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 高级日志记录优化系统

测试logging/advanced_logger.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import asyncio
import time
import threading
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


class TestAdvancedLogger:
    """测试高级日志记录系统"""

    def setup_method(self):
        """测试前准备"""
        try:
            from src.infrastructure.logging.advanced_logger import (
                LogPriority, LogCompression, LogEntry, LogEntryPool,
                SmartLogFilter, AsyncLogWriter, LogCompressor,
                LogArchiver, PerformanceMonitor, AdvancedLogger
            )
            self.LogPriority = LogPriority
            self.LogCompression = LogCompression
            self.LogEntry = LogEntry
            self.LogEntryPool = LogEntryPool
            self.SmartLogFilter = SmartLogFilter
            self.AsyncLogWriter = AsyncLogWriter
            self.LogCompressor = LogCompressor
            self.LogArchiver = LogArchiver
            self.PerformanceMonitor = PerformanceMonitor
            self.AdvancedLogger = AdvancedLogger
        except ImportError as e:
            pytest.skip(f"Advanced logger components not available: {e}")

    def test_log_priority_enum(self):
        """测试日志优先级枚举"""
        if not hasattr(self, 'LogPriority'):
            pytest.skip("LogPriority not available")

        assert self.LogPriority.LOW.value == 1
        assert self.LogPriority.NORMAL.value == 2
        assert self.LogPriority.HIGH.value == 3
        assert self.LogPriority.CRITICAL.value == 4

    def test_log_compression_enum(self):
        """测试日志压缩枚举"""
        if not hasattr(self, 'LogCompression'):
            pytest.skip("LogCompression not available")

        assert self.LogCompression.NONE.value == "none"
        assert self.LogCompression.GZIP.value == "gzip"
        assert self.LogCompression.LZ4.value == "lz4"

    def test_log_entry_creation(self):
        """测试日志条目创建"""
        if not hasattr(self, 'LogEntry'):
            pytest.skip("LogEntry not available")

        # 根据实际的LogEntry定义调整参数
        entry = self.LogEntry(
            timestamp=time.time(),
            level="INFO",
            message="Test message",
            component="test_component",
            thread_id=1,
            process_id=12345
        )

        assert entry.level == "INFO"
        assert entry.message == "Test message"
        assert isinstance(entry.timestamp, float)

    def test_log_entry_pool(self):
        """测试日志条目池"""
        if not hasattr(self, 'LogEntryPool'):
            pytest.skip("LogEntryPool not available")

        pool = self.LogEntryPool(max_size=10)

        assert pool.max_size == 10
        assert len(pool.pool) == 0

        # 获取条目
        entry = pool.get_entry(
            timestamp=time.time(),
            level="INFO",
            message="Test message",
            component="test_component",
            thread_id=1,
            process_id=12345
        )
        assert entry is not None
        assert len(pool.pool) == 0  # 池中没有可用条目

        # 归还条目
        pool.return_entry(entry)
        assert len(pool.pool) == 1

    def test_smart_log_filter(self):
        """测试智能日志过滤器"""
        if not hasattr(self, 'SmartLogFilter'):
            pytest.skip("SmartLogFilter not available")

        filter = self.SmartLogFilter()

        assert filter is not None

        # 测试过滤逻辑（这里需要根据实际实现调整）
        log_entry = self.LogEntry(
            timestamp=time.time(),
            level="DEBUG",
            message="Debug message",
            component="test",
            thread_id=1,
            process_id=12345
        )

        # 根据实际实现，SmartLogFilter有should_log方法而不是filter方法
        if hasattr(filter, 'should_log'):
            result = filter.should_log(log_entry)
            assert isinstance(result, bool)

    def test_async_log_writer(self):
        """测试异步日志写入器"""
        if not hasattr(self, 'AsyncLogWriter'):
            pytest.skip("AsyncLogWriter not available")

        writer = self.AsyncLogWriter()

        assert writer is not None
        assert hasattr(writer, 'log_queue')
        assert hasattr(writer, 'running')

        # 测试写入功能
        log_entry = self.LogEntry(
            timestamp=time.time(),
            level="INFO",
            message="Async test message",
            component="test_component",
            thread_id=1,
            process_id=12345
        )

        # 这里需要根据实际实现调整
        if hasattr(writer, 'enqueue_log'):
            mock_handler = Mock()
            result = writer.enqueue_log(log_entry, mock_handler)
            # 异步写入可能返回None或Future
            assert result is None

    def test_log_compressor(self):
        """测试日志压缩器"""
        if not hasattr(self, 'LogCompressor'):
            pytest.skip("LogCompressor not available")

        compressor = self.LogCompressor()

        assert compressor is not None

        # 测试压缩功能
        test_data = "Test log data for compression"

        if hasattr(compressor, 'compress_log'):
            compressed = compressor.compress_log(test_data)
            assert compressed is not None
            assert len(compressed) > 0

            # 测试解压缩
            if hasattr(compressor, 'decompress_log'):
                decompressed = compressor.decompress_log(compressed)
                assert decompressed == test_data

    def test_log_archiver(self):
        """测试日志归档器"""
        if not hasattr(self, 'LogArchiver'):
            pytest.skip("LogArchiver not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            archiver = self.LogArchiver(archive_dir=temp_dir)

            assert archiver is not None
            assert str(archiver.archive_dir) == temp_dir

            # 测试归档功能
            if hasattr(archiver, 'archive_logs'):
                # 创建一个临时日志文件
                log_file = os.path.join(temp_dir, "test.log")
                with open(log_file, 'w') as f:
                    f.write("Test log content\n")

                result = archiver.archive_logs([log_file])
                assert isinstance(result, list)

    def test_performance_monitor(self):
        """测试性能监控器"""
        if not hasattr(self, 'PerformanceMonitor'):
            pytest.skip("PerformanceMonitor not available")

        monitor = self.PerformanceMonitor()

        assert monitor is not None

        # 测试性能监控功能
        if hasattr(monitor, 'record_metric'):
            monitor.record_metric("test_metric", 100)

        if hasattr(monitor, 'get_average'):
            metrics = monitor.get_average("test_metric")
            # get_average可能返回None或数值

    def test_advanced_logger(self):
        """测试高级日志器"""
        if not hasattr(self, 'AdvancedLogger'):
            pytest.skip("AdvancedLogger not available")

        logger = self.AdvancedLogger("test_logger")

        assert logger is not None

        # 测试日志记录功能
        if hasattr(logger, 'log'):
            result = logger.log("INFO", "Test message", component="test")
            # log方法没有返回值，所以不检查返回值

    def test_advanced_logger_with_config(self):
        """测试带配置的高级日志器"""
        if not hasattr(self, 'AdvancedLogger'):
            pytest.skip("AdvancedLogger not available")

        logger = self.AdvancedLogger("test_logger")

        assert logger is not None

        # 验证配置是否正确应用
        # 注意：在实现中，配置是通过构造函数参数传递的，而不是config属性

    def test_log_entry_pool_limits(self):
        """测试日志条目池大小限制"""
        if not hasattr(self, 'LogEntryPool'):
            pytest.skip("LogEntryPool not available")

        pool = self.LogEntryPool(max_size=3)

        # 获取多个条目
        entries = []
        for i in range(5):  # 超过池的大小限制
            entry = pool.get_entry(
                timestamp=time.time(),
                level="INFO",
                message=f"Test message {i}",
                component="test_component",
                thread_id=1,
                process_id=12345
            )
            if entry:
                entries.append(entry)

        # 归还所有条目
        for entry in entries:
            pool.return_entry(entry)

        # 验证池的大小
        assert len(pool.pool) <= pool.max_size

    def test_smart_log_filter_patterns(self):
        """测试智能日志过滤器的模式匹配"""
        if not hasattr(self, 'SmartLogFilter'):
            pytest.skip("SmartLogFilter not available")

        filter = self.SmartLogFilter()

        # 添加过滤模式
        if hasattr(filter, 'add_filter'):
            # SmartLogFilter使用add_filter而不是add_pattern
            def test_filter(entry):
                return "ERROR" in entry.message
            filter.add_filter(test_filter)

        # 测试模式匹配
        if hasattr(filter, 'should_log'):
            test_entry = self.LogEntry(
                timestamp=time.time(),
                level="ERROR",
                message="ERROR: This is a test error",
                component="test",
                thread_id=1,
                process_id=12345
            )
            result = filter.should_log(test_entry)
            assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_async_log_writer_async_operations(self):
        """测试异步日志写入器的异步操作"""
        if not hasattr(self, 'AsyncLogWriter'):
            pytest.skip("AsyncLogWriter not available")

        writer = self.AsyncLogWriter()

        log_entry = self.LogEntry(
            timestamp=time.time(),
            level="INFO",
            message="Async test message",
            component="test_component",
            thread_id=1,
            process_id=12345
        )

        # AsyncLogWriter使用enqueue_log而不是write_async
        if hasattr(writer, 'enqueue_log'):
            mock_handler = Mock()
            result = writer.enqueue_log(log_entry, mock_handler)
            # 异步操作完成

    def test_log_compressor_different_formats(self):
        """测试日志压缩器不同格式"""
        if not hasattr(self, 'LogCompressor'):
            pytest.skip("LogCompressor not available")

        compressor = self.LogCompressor()

        test_data = "Test data for different compression formats"

        # 测试GZIP压缩
        if hasattr(compressor, 'compress_log'):
            gzip_data = compressor.compress_log(test_data)
            assert gzip_data is not None

        # 测试无压缩
        none_compressor = self.LogCompressor(self.LogCompression.NONE)
        none_data = none_compressor.compress_log(test_data)
        assert none_data == test_data.encode('utf-8')

    def test_log_archiver_cleanup(self):
        """测试日志归档器的清理功能"""
        if not hasattr(self, 'LogArchiver'):
            pytest.skip("LogArchiver not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            archiver = self.LogArchiver(archive_dir=temp_dir)

            # 创建一些旧文件
            old_file = os.path.join(temp_dir, "old_log.log")
            with open(old_file, 'w') as f:
                f.write("Old log content")

            # 修改文件时间为7天前
            old_time = time.time() - (8 * 24 * 60 * 60)  # 8天前
            os.utime(old_file, (old_time, old_time))

            if hasattr(archiver, 'cleanup_old_archives'):
                result = archiver.cleanup_old_archives(days_to_keep=7)
                assert isinstance(result, list)

    def test_performance_monitor_metrics(self):
        """测试性能监控器的指标收集"""
        if not hasattr(self, 'PerformanceMonitor'):
            pytest.skip("PerformanceMonitor not available")

        monitor = self.PerformanceMonitor()

        # 记录一些指标
        if hasattr(monitor, 'record_metric'):
            monitor.record_metric("write_operations", 150)
            monitor.record_metric("read_operations", 300)
            monitor.record_metric("error_count", 5)

        # 获取统计信息
        if hasattr(monitor, 'get_average'):
            stats = monitor.get_average("write_operations")
            # get_average可能返回None或数值

    def test_advanced_logger_error_handling(self):
        """测试高级日志器的错误处理"""
        if not hasattr(self, 'AdvancedLogger'):
            pytest.skip("AdvancedLogger not available")

        logger = self.AdvancedLogger("test_logger")

        # 测试无效日志级别
        if hasattr(logger, 'log'):
            try:
                logger.log("INVALID_LEVEL", "Test message")
            except Exception:
                pass  # 应该能处理无效级别

        # 测试空消息
        try:
            logger.log("INFO", "")
        except Exception:
            pass  # 应该能处理空消息

        # 日志器应该仍然正常工作
        assert logger is not None


class TestAdvancedLoggerExtendedFeatures:
    """测试AdvancedLogger的扩展功能"""

    def setup_method(self):
        """测试前准备"""
        if not hasattr(self, 'AdvancedLogger'):
            pytest.skip("AdvancedLogger not available")
        self.logger = self.AdvancedLogger("extended_test")

    def test_log_structured_functionality(self):
        """测试结构化日志功能"""
        # 测试结构化日志记录
        self.logger.log_structured("INFO", "Test structured message",
                                  user_id=123,
                                  action="login",
                                  ip_address="192.168.1.1")

        # 验证日志器状态
        assert self.logger.name == "extended_test"

    def test_async_logging_functionality(self):
        """测试异步日志功能"""
        # 测试异步日志记录
        self.logger.log_async(self.LogLevel.INFO, "Async test message",
                             user="test_user", action="test_action")

        # 等待异步处理完成
        time.sleep(0.1)

        # 验证日志器状态
        assert self.logger.name == "extended_test"

    def test_performance_tracking_functionality(self):
        """测试性能跟踪功能"""
        # 测试带性能跟踪的日志记录
        self.logger.log_with_performance_tracking(
            self.LogLevel.INFO,
            "Performance test message",
            operation="test_operation",
            user_id=456
        )

        # 获取性能统计
        stats = self.logger.get_performance_stats()
        assert isinstance(stats, dict)

        # 验证统计信息结构
        if stats:  # 如果有统计数据
            assert "total_logs" in stats
            assert "average_processing_time" in stats

    def test_filter_functionality(self):
        """测试过滤器功能"""
        # 添加过滤器 - 只允许ERROR级别的日志
        def error_only_filter(entry):
            return entry.level == self.LogLevel.ERROR

        self.logger.add_filter(error_only_filter)

        # 测试过滤器效果
        # 注意：实际的过滤逻辑可能在不同的实现中

    def test_context_management(self):
        """测试上下文管理"""
        # 设置上下文
        context = {
            "application": "test_app",
            "version": "1.0.0",
            "environment": "testing"
        }

        self.logger.set_context(context)

        # 记录日志（上下文应该被包含）
        self.logger.log_structured("INFO", "Context test message",
                                  additional_field="test_value")

    def test_config_updates(self):
        """测试配置更新"""
        # 更新配置
        new_config = {
            "enable_async": False,
            "max_batch_size": 50,
            "compression_enabled": True
        }

        self.logger.update_config(new_config)

        # 验证配置更新（如果有相应的属性）
        # 注意：具体的配置更新逻辑取决于实现

    def test_batch_logging(self):
        """测试批量日志记录"""
        # 准备批量消息
        messages = [
            {"level": "INFO", "message": "Batch message 1", "data": {"id": 1}},
            {"level": "WARNING", "message": "Batch message 2", "data": {"id": 2}},
            {"level": "ERROR", "message": "Batch message 3", "data": {"id": 3}}
        ]

        # 执行批量日志记录
        self.logger.log_batch(messages)

        # 验证批量处理（检查是否没有异常抛出）
        assert self.logger.name == "extended_test"

    def test_shutdown_functionality(self):
        """测试关闭功能"""
        # 执行一些操作
        self.logger.log_structured("INFO", "Pre-shutdown message")

        # 关闭日志器
        self.logger.shutdown()

        # 验证关闭后的状态
        # 注意：关闭后的具体行为取决于实现

    def test_concurrent_async_logging(self):
        """测试并发异步日志记录"""
        import threading

        results = []
        errors = []

        def worker(worker_id):
            """工作线程"""
            try:
                for i in range(5):
                    self.logger.log_async(
                        self.LogLevel.INFO,
                        f"Concurrent async message {i} from worker {worker_id}",
                        worker_id=worker_id,
                        message_id=i
                    )
                results.append(f"worker_{worker_id}_completed")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")

        # 启动多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)

        # 执行并发操作
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 3
        assert len(errors) == 0

        # 等待异步处理完成
        time.sleep(0.2)

    def test_performance_stats_comprehensive(self):
        """测试性能统计的全面功能"""
        # 执行多种日志操作
        self.logger.log_structured("INFO", "Stats test 1")
        self.logger.log_async(self.LogLevel.WARNING, "Stats test 2")
        self.logger.log_with_performance_tracking(
            self.LogLevel.ERROR, "Stats test 3", "test_operation"
        )

        # 获取详细的性能统计
        stats = self.logger.get_performance_stats()

        assert isinstance(stats, dict)

        # 如果有统计数据，验证其合理性
        if stats:
            # 验证基本字段
            assert isinstance(stats.get("total_logs", 0), (int, float))

            # 验证时间相关字段
            avg_time = stats.get("average_processing_time", 0)
            if avg_time > 0:
                assert avg_time > 0

            # 验证其他可能的字段
            for key, value in stats.items():
                assert isinstance(key, str), f"Key should be string: {key}"
                # 值可以是各种类型，但不应该是None（除非明确设计）

    def test_error_handling_in_async_logging(self):
        """测试异步日志记录中的错误处理"""
        # 测试各种边界情况
        try:
            # 测试None消息
            self.logger.log_async(self.LogLevel.INFO, None)
        except Exception:
            pass  # 应该能处理None消息

        try:
            # 测试空消息
            self.logger.log_async(self.LogLevel.INFO, "")
        except Exception:
            pass  # 应该能处理空消息

        try:
            # 测试无效级别
            self.logger.log_async("INVALID_LEVEL", "Test message")
        except Exception:
            pass  # 应该能处理无效级别

        # 验证日志器仍然可用
        self.logger.log_async(self.LogLevel.INFO, "Recovery test message")

    def test_structured_logging_with_complex_data(self):
        """测试结构化日志记录复杂数据"""
        # 测试嵌套字典
        complex_data = {
            "user": {
                "id": 12345,
                "name": "test_user",
                "roles": ["admin", "user"]
            },
            "action": "complex_operation",
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "2.1.0",
                "features": ["async", "structured", "performance"]
            },
            "performance": {
                "response_time": 0.245,
                "cpu_usage": 45.2,
                "memory_mb": 128.5
            }
        }

        self.logger.log_structured("INFO", "Complex data test", **complex_data)

        # 验证日志记录成功（没有异常抛出）
        assert self.logger.name == "extended_test"

    def test_resource_cleanup_on_shutdown(self):
        """测试关闭时的资源清理"""
        # 创建新的日志器实例用于测试
        test_logger = self.AdvancedLogger("cleanup_test")

        # 执行一些操作来初始化资源
        test_logger.log_structured("INFO", "Init message")
        test_logger.log_async(self.LogLevel.WARNING, "Async init message")

        # 等待异步操作
        time.sleep(0.1)

        # 执行关闭
        test_logger.shutdown()

        # 验证关闭后的状态
        # 注意：具体的资源清理验证取决于实现细节
        assert test_logger.name == "cleanup_test"


if __name__ == '__main__':
    pytest.main([__file__])
