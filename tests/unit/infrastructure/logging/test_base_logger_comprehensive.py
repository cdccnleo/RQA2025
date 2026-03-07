#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - BaseLogger深度测试
测试BaseLogger的核心日志记录功能、边界条件和并发安全
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import logging
import time
import threading
from unittest.mock import patch, MagicMock
from io import StringIO

from infrastructure.logging.core.base_logger import BaseLogger
from infrastructure.logging.core.interfaces import LogLevel


class TestBaseLoggerInitialization:
    """BaseLogger初始化测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        logger = BaseLogger()
        assert logger.name == "BaseLogger"
        assert logger.level == LogLevel.INFO
        assert hasattr(logger, '_logger')
        assert hasattr(logger, '_lock')

    def test_initialization_custom(self):
        """测试自定义初始化"""
        logger = BaseLogger(name="CustomLogger", level=LogLevel.DEBUG)
        assert logger.name == "CustomLogger"
        assert logger.level == LogLevel.DEBUG

    def test_initialization_edge_cases(self):
        """测试初始化边界条件"""
        # 空名称
        logger1 = BaseLogger(name="")
        assert logger1.name == ""

        # None名称
        logger2 = BaseLogger(name=None)
        assert logger2.name is None

        # 特殊字符名称
        logger3 = BaseLogger(name="logger-with-dashes.and.dots_underscores")
        assert logger3.name == "logger-with-dashes.and.dots_underscores"


class TestBaseLoggerLogMethods:
    """BaseLogger日志方法测试"""

    @pytest.fixture
    def logger(self):
        """BaseLogger fixture"""
        return BaseLogger("TestLogger", LogLevel.DEBUG)

    def test_debug_method(self, logger):
        """测试debug方法"""
        with patch.object(logger._logger, 'debug') as mock_debug:
            logger.debug("Test debug message")
            mock_debug.assert_called_once()

    def test_info_method(self, logger):
        """测试info方法"""
        with patch.object(logger._logger, 'info') as mock_info:
            logger.info("Test info message")
            mock_info.assert_called_once()

    def test_warning_method(self, logger):
        """测试warning方法"""
        with patch.object(logger._logger, 'warning') as mock_warning:
            logger.warning("Test warning message")
            mock_warning.assert_called_once()

    def test_error_method(self, logger):
        """测试error方法"""
        with patch.object(logger._logger, 'error') as mock_error:
            logger.error("Test error message")
            mock_error.assert_called_once()

    def test_critical_method(self, logger):
        """测试critical方法"""
        with patch.object(logger._logger, 'critical') as mock_critical:
            logger.critical("Test critical message")
            mock_critical.assert_called_once()


class TestBaseLoggerLogLevelFiltering:
    """BaseLogger日志级别过滤测试"""

    def test_debug_level_logs_all(self):
        """测试DEBUG级别记录所有日志"""
        logger = BaseLogger("TestLogger", LogLevel.DEBUG)

        with patch.object(logger._logger, 'debug') as mock_debug, \
             patch.object(logger._logger, 'info') as mock_info, \
             patch.object(logger._logger, 'warning') as mock_warning, \
             patch.object(logger._logger, 'error') as mock_error, \
             patch.object(logger._logger, 'critical') as mock_critical:

            logger.debug("debug")
            logger.info("info")
            logger.warning("warning")
            logger.error("error")
            logger.critical("critical")

            # DEBUG级别应该记录所有日志
            mock_debug.assert_called_once()
            mock_info.assert_called_once()
            mock_warning.assert_called_once()
            mock_error.assert_called_once()
            mock_critical.assert_called_once()

    def test_info_level_filters_debug(self):
        """测试INFO级别过滤DEBUG日志"""
        logger = BaseLogger("TestLogger", LogLevel.INFO)

        with patch.object(logger._logger, 'debug') as mock_debug, \
             patch.object(logger._logger, 'info') as mock_info:

            logger.debug("debug message")
            logger.info("info message")

            # DEBUG日志应该被过滤
            mock_debug.assert_not_called()
            mock_info.assert_called_once()

    def test_warning_level_filters_lower(self):
        """测试WARNING级别过滤更低级别日志"""
        logger = BaseLogger("TestLogger", LogLevel.WARNING)

        with patch.object(logger._logger, 'debug') as mock_debug, \
             patch.object(logger._logger, 'info') as mock_info, \
             patch.object(logger._logger, 'warning') as mock_warning:

            logger.debug("debug")
            logger.info("info")
            logger.warning("warning")

            mock_debug.assert_not_called()
            mock_info.assert_not_called()
            mock_warning.assert_called_once()

    def test_error_level_filters_lower(self):
        """测试ERROR级别过滤更低级别日志"""
        logger = BaseLogger("TestLogger", LogLevel.ERROR)

        with patch.object(logger._logger, 'warning') as mock_warning, \
             patch.object(logger._logger, 'error') as mock_error:

            logger.warning("warning")
            logger.error("error")

            mock_warning.assert_not_called()
            mock_error.assert_called_once()

    def test_critical_level_only_critical(self):
        """测试CRITICAL级别只记录CRITICAL日志"""
        logger = BaseLogger("TestLogger", LogLevel.CRITICAL)

        with patch.object(logger._logger, 'error') as mock_error, \
             patch.object(logger._logger, 'critical') as mock_critical:

            logger.error("error")
            logger.critical("critical")

            mock_error.assert_not_called()
            mock_critical.assert_called_once()


class TestBaseLoggerContextHandling:
    """BaseLogger上下文信息处理测试"""

    @pytest.fixture
    def logger(self):
        """BaseLogger fixture"""
        return BaseLogger("TestLogger", LogLevel.DEBUG)

    def test_log_with_context_kwargs(self, logger):
        """测试带上下文信息的日志记录"""
        with patch.object(logger._logger, 'info') as mock_info:
            logger.info("Test message", user_id=123, action="login", ip="192.168.1.1")

            # 验证消息被正确格式化
            mock_info.assert_called_once()
            formatted_message = mock_info.call_args[0][0]
            assert "Test message" in formatted_message
            assert "user_id=123" in formatted_message
            assert "action=login" in formatted_message
            assert "ip=192.168.1.1" in formatted_message

    def test_log_with_empty_kwargs(self, logger):
        """测试空上下文信息的日志记录"""
        with patch.object(logger._logger, 'info') as mock_info:
            logger.info("Test message")

            mock_info.assert_called_once_with("Test message")

    def test_log_with_special_characters_in_context(self, logger):
        """测试上下文信息包含特殊字符"""
        with patch.object(logger._logger, 'info') as mock_info:
            logger.info("Test message", data="value with spaces", json='{"key": "value"}')

            formatted_message = mock_info.call_args[0][0]
            assert "data=value with spaces" in formatted_message
            assert 'json={"key": "value"}' in formatted_message

    def test_log_with_none_values_in_context(self, logger):
        """测试上下文信息包含None值"""
        with patch.object(logger._logger, 'info') as mock_info:
            logger.info("Test message", user=None, data="valid")

            formatted_message = mock_info.call_args[0][0]
            assert "user=None" in formatted_message
            assert "data=valid" in formatted_message


class TestBaseLoggerLargeMessages:
    """BaseLogger大消息处理测试"""

    @pytest.fixture
    def logger(self):
        """BaseLogger fixture"""
        return BaseLogger("TestLogger", LogLevel.DEBUG)

    def test_large_message_handling(self, logger):
        """测试大消息处理"""
        large_message = "x" * 10000  # 10KB消息

        with patch.object(logger._logger, 'info') as mock_info:
            logger.info(large_message)

            mock_info.assert_called_once()
            # 验证消息没有被截断
            logged_message = mock_info.call_args[0][0]
            assert len(logged_message) == len(large_message)
            assert logged_message == large_message

    def test_large_message_with_context(self, logger):
        """测试大消息带上下文"""
        large_message = "x" * 5000
        context_data = {"large_context": "y" * 2000}

        with patch.object(logger._logger, 'info') as mock_info:
            logger.info(large_message, **context_data)

            logged_message = mock_info.call_args[0][0]
            assert large_message in logged_message
            assert "large_context=" in logged_message

    def test_many_context_items(self, logger):
        """测试大量上下文项目"""
        context = {f"key_{i}": f"value_{i}" for i in range(50)}

        with patch.object(logger._logger, 'info') as mock_info:
            logger.info("Test message", **context)

            logged_message = mock_info.call_args[0][0]
            assert "Test message" in logged_message
            # 验证所有上下文都被包含
            for i in range(50):
                assert f"key_{i}=value_{i}" in logged_message


class TestBaseLoggerConcurrency:
    """BaseLogger并发安全测试"""

    @pytest.fixture
    def logger(self):
        """BaseLogger fixture"""
        return BaseLogger("ConcurrentLogger", LogLevel.DEBUG)

    def test_concurrent_logging(self, logger):
        """测试并发日志记录"""
        log_calls = []
        original_log = logger._logger.info

        def capture_log(*args, **kwargs):
            log_calls.append(args[0] if args else "")
            return original_log(*args, **kwargs)

        with patch.object(logger._logger, 'info', side_effect=capture_log):
            def worker(thread_id: int):
                for i in range(100):
                    logger.info(f"Thread {thread_id} message {i}")

            threads = []
            num_threads = 10

            # 启动多个线程
            for i in range(num_threads):
                t = threading.Thread(target=worker, args=(i,))
                threads.append(t)
                t.start()

            # 等待所有线程完成
            for t in threads:
                t.join()

            # 验证所有消息都被记录
            assert len(log_calls) == num_threads * 100

            # 验证消息内容正确
            for thread_id in range(num_threads):
                thread_messages = [msg for msg in log_calls if f"Thread {thread_id}" in msg]
                assert len(thread_messages) == 100

    def test_concurrent_level_changes(self, logger):
        """测试并发级别更改"""
        results = []

        def change_level_worker():
            try:
                # 快速更改日志级别
                for level in [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]:
                    logger.set_level(level)
                    time.sleep(0.001)  # 短暂延迟
                results.append("success")
            except Exception as e:
                results.append(f"error: {e}")

        threads = []
        num_threads = 5

        # 启动多个线程同时更改级别
        for i in range(num_threads):
            t = threading.Thread(target=change_level_worker)
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join()

        # 所有线程都应该成功完成
        assert len(results) == num_threads
        assert all(r == "success" for r in results)

    def test_concurrent_mixed_operations(self, logger):
        """测试并发混合操作"""
        operations_completed = []

        def mixed_operations_worker(thread_id: int):
            try:
                # 执行各种操作的混合
                logger.info(f"Thread {thread_id} starting")

                # 更改级别
                logger.set_level(LogLevel.DEBUG)
                logger.debug(f"Thread {thread_id} debug message")

                # 更改回INFO
                logger.set_level(LogLevel.INFO)

                # 记录不同级别的消息
                logger.info(f"Thread {thread_id} info message")
                logger.warning(f"Thread {thread_id} warning message")
                logger.error(f"Thread {thread_id} error message")

                operations_completed.append(thread_id)

            except Exception as e:
                operations_completed.append(f"error_{thread_id}: {e}")

        threads = []
        num_threads = 8

        # 启动并发操作
        for i in range(num_threads):
            t = threading.Thread(target=mixed_operations_worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join()

        # 验证所有线程都成功完成
        assert len(operations_completed) == num_threads
        for i in range(num_threads):
            assert i in operations_completed


class TestBaseLoggerErrorHandling:
    """BaseLogger错误处理测试"""

    @pytest.fixture
    def logger(self):
        """BaseLogger fixture"""
        return BaseLogger("ErrorTestLogger", LogLevel.DEBUG)

    def test_invalid_log_level_string(self, logger):
        """测试无效的日志级别字符串"""
        # 无效的日志级别会被转换为默认级别（INFO），不会抛出异常
        with patch.object(logger._logger, 'info') as mock_info:
            logger.log("INVALID_LEVEL", "Test message")
            # 验证使用默认级别INFO的info方法
            mock_info.assert_called_once()

    def test_invalid_log_level_enum(self, logger):
        """测试无效的日志级别枚举"""
        # 这个应该通过验证，但会在内部处理
        with patch.object(logger._logger, 'debug') as mock_debug:
            logger.log("DEBUG", "Test message")
            mock_debug.assert_called_once()

    def test_logger_exception_during_logging(self, logger):
        """测试日志记录过程中发生异常"""
        # Mock底层logger抛出异常
        with patch.object(logger._logger, 'info', side_effect=Exception("Logger error")):
            with pytest.raises(Exception, match="日志记录失败"):
                logger.info("Test message")

    def test_context_formatting_error(self, logger):
        """测试上下文格式化错误"""
        # 创建一个会导致格式化错误的上下文
        class BadRepr:
            def __repr__(self):
                raise ValueError("Bad repr")

        bad_context = BadRepr()

        with patch.object(logger._logger, 'info') as mock_info:
            # 应该仍然能够记录日志，尽管上下文格式化失败
            logger.info("Test message", bad_context=bad_context)

            # 验证日志仍然被记录（使用基本消息）
            mock_info.assert_called_once()
            logged_message = mock_info.call_args[0][0]
            assert "Test message" in logged_message

    def test_log_method_with_none_message(self, logger):
        """测试消息为None的情况"""
        with patch.object(logger._logger, 'info') as mock_info:
            logger.info(None)

            # 应该处理None消息
            mock_info.assert_called_once()

    def test_log_method_with_empty_message(self, logger):
        """测试空消息"""
        with patch.object(logger._logger, 'info') as mock_info:
            logger.info("")

            mock_info.assert_called_once_with("")


class TestBaseLoggerBoundaryConditions:
    """BaseLogger边界条件测试"""

    def test_log_level_boundary_values(self):
        """测试日志级别边界值"""
        # 测试所有有效级别
        for level in [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]:
            logger = BaseLogger("BoundaryTest", level)
            assert logger.level == level

    def test_set_level_method(self):
        """测试set_level方法"""
        logger = BaseLogger("SetLevelTest", LogLevel.INFO)

        # 更改级别
        logger.set_level(LogLevel.DEBUG)
        assert logger.level == LogLevel.DEBUG

        logger.set_level(LogLevel.ERROR)
        assert logger.level == LogLevel.ERROR

    def test_very_long_logger_name(self):
        """测试非常长的日志器名称"""
        long_name = "a" * 1000
        logger = BaseLogger(long_name, LogLevel.INFO)
        assert logger.name == long_name

    def test_unicode_in_messages_and_context(self):
        """测试消息和上下文中包含Unicode字符"""
        logger = BaseLogger("UnicodeTest", LogLevel.INFO)

        unicode_message = "测试消息 🚀 中文English"
        unicode_context = {
            "用户": "张三",
            "操作": "登录",
            "符号": "∑∆∞"
        }

        with patch.object(logger._logger, 'info') as mock_info:
            logger.info(unicode_message, **unicode_context)

            logged_message = mock_info.call_args[0][0]
            assert unicode_message in logged_message
            assert "用户=张三" in logged_message
            assert "操作=登录" in logged_message
            assert "符号=∑∆∞" in logged_message

    def test_log_method_case_insensitive_levels(self):
        """测试日志级别大小写不敏感"""
        logger = BaseLogger("CaseTest", LogLevel.DEBUG)

        with patch.object(logger._logger, 'debug') as mock_debug, \
             patch.object(logger._logger, 'info') as mock_info:

            # 测试小写
            logger.log("debug", "lowercase debug")
            logger.log("info", "lowercase info")

            # 测试大写
            logger.log("DEBUG", "uppercase debug")
            logger.log("INFO", "uppercase info")

            # 所有调用都应该成功
            assert mock_debug.call_count == 2
            assert mock_info.call_count == 2

    def test_multiple_loggers_isolation(self):
        """测试多个日志器实例的隔离"""
        logger1 = BaseLogger("Logger1", LogLevel.DEBUG)
        logger2 = BaseLogger("Logger2", LogLevel.ERROR)

        # 每个日志器应该有独立的级别
        assert logger1.level == LogLevel.DEBUG
        assert logger2.level == LogLevel.ERROR

        # 更改一个不影响另一个
        logger1.set_level(LogLevel.INFO)
        assert logger1.level == LogLevel.INFO
        assert logger2.level == LogLevel.ERROR

    def test_logger_name_uniqueness(self):
        """测试日志器名称唯一性"""
        logger1 = BaseLogger("UniqueName", LogLevel.INFO)
        logger2 = BaseLogger("UniqueName", LogLevel.DEBUG)
        logger3 = BaseLogger("DifferentName", LogLevel.INFO)

        # 名称可以相同，但实例不同
        assert logger1.name == logger2.name
        assert logger1 is not logger2
        assert logger1.level != logger2.level
        assert logger3.name != logger1.name


class TestBaseLoggerPerformance:
    """BaseLogger性能测试"""

    @pytest.fixture
    def logger(self):
        """BaseLogger fixture"""
        return BaseLogger("PerformanceTest", LogLevel.DEBUG)

    def test_high_frequency_logging_performance(self, logger):
        """测试高频日志记录性能"""
        num_messages = 10000

        with patch.object(logger._logger, 'debug') as mock_debug:
            start_time = time.time()

            for i in range(num_messages):
                logger.debug(f"Performance test message {i}")

            end_time = time.time()

            duration = end_time - start_time

            # 验证所有消息都被记录
            assert mock_debug.call_count == num_messages

            # 性能检查：10,000条消息应该在合理时间内完成
            # 允许比较宽松的时间限制，因为测试环境可能不同
            assert duration < 5.0, f"Performance too slow: {duration:.2f}s for {num_messages} messages"

            # 计算每秒消息数
            messages_per_second = num_messages / duration
            print(f"Logging performance: {messages_per_second:.0f} messages/second")

    def test_context_formatting_performance(self, logger):
        """测试上下文格式化性能"""
        num_messages = 1000
        large_context = {f"key_{i}": f"value_{i}" for i in range(20)}

        with patch.object(logger._logger, 'info') as mock_info:
            start_time = time.time()

            for i in range(num_messages):
                logger.info(f"Message {i}", **large_context)

            end_time = time.time()

            duration = end_time - start_time

            # 验证所有消息都被记录
            assert mock_info.call_count == num_messages

            # 检查每条消息都包含了上下文
            for call in mock_info.call_args_list:
                message = call[0][0]
                assert "Message" in message
                assert "key_0=value_0" in message

            # 性能检查
            assert duration < 3.0, f"Context formatting too slow: {duration:.2f}s for {num_messages} messages"
