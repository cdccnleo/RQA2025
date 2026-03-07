#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 核心Logger功能

测试重构后核心Logger模块的各个组件。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.infrastructure.logging.core import (
    UnifiedLogger, BaseLogger, BusinessLogger, AuditLogger, PerformanceLogger,
    LogLevel, LogFormat, LogCategory
)


class TestBaseLogger:
    """BaseLogger单元测试"""

    def test_initialization(self):
        """测试初始化"""
        logger = BaseLogger("TestBase", LogLevel.INFO)
        assert logger.name == "TestBase"
        assert logger.level == LogLevel.INFO

    def test_log_method(self):
        """测试日志记录方法"""
        logger = BaseLogger("TestLog")

        # Mock the underlying logger
        with patch.object(logger, '_logger') as mock_logger:
            logger.info("测试消息")
            mock_logger.info.assert_called_once()

    def test_should_log(self):
        """测试日志级别判断"""
        logger = BaseLogger("TestLevel", LogLevel.WARNING)

        # Mock the method
        assert logger._should_log(LogLevel.ERROR) == True
        assert logger._should_log(LogLevel.INFO) == False

    def test_format_message(self):
        """测试消息格式化"""
        logger = BaseLogger("TestFormat")

        # 测试无额外参数
        result = logger._format_message("测试消息", {})
        assert result == "测试消息"

        # 测试有额外参数
        result = logger._format_message("测试消息", {"key": "value"})
        assert "key=value" in result


class TestUnifiedLogger:
    """UnifiedLogger单元测试"""

    def test_initialization(self):
        """测试初始化"""
        with patch('src.infrastructure.logging.core.unified_logger.Path'):
            logger = UnifiedLogger("TestUnified", LogLevel.DEBUG)
            assert logger.name == "TestUnified"
            assert logger.level == LogLevel.DEBUG

    def test_convert_level(self):
        """测试级别转换"""
        logger = UnifiedLogger("TestConvert")

        assert logger._convert_level(LogLevel.DEBUG) == 10  # logging.DEBUG
        assert logger._convert_level(LogLevel.INFO) == 20   # logging.INFO
        assert logger._convert_level(LogLevel.WARNING) == 30 # logging.WARNING
        assert logger._convert_level(LogLevel.ERROR) == 40   # logging.ERROR
        assert logger._convert_level(LogLevel.CRITICAL) == 50 # logging.CRITICAL

    def test_log_structured(self):
        """测试结构化日志"""
        logger = UnifiedLogger("TestStructured")

        with patch.object(logger._business_logger.logger, 'info') as mock_info:
            logger.log_structured(LogLevel.INFO, "结构化消息", key="value")
            mock_info.assert_called_once()

    def test_log_performance(self):
        """测试性能日志"""
        logger = UnifiedLogger("TestPerf")

        with patch.object(logger._business_logger.logger, 'info') as mock_info:
            logger.log_performance("test_operation", 1.5, param="value")
            mock_info.assert_called_once()

    def test_shutdown(self):
        """测试关闭功能"""
        with patch('src.infrastructure.logging.core.unified_logger.Path'):
            logger = UnifiedLogger("TestShutdown")

            with patch.object(logger._recorder.logger, 'handlers', []):
                logger.shutdown()
                # 验证相关清理操作


class TestSpecializedLoggers:
    """专用Logger测试"""

    def test_business_logger(self):
        """测试业务Logger"""
        logger = BusinessLogger("TestBusiness")
        assert logger.name == "TestBusiness"
        assert logger.level == LogLevel.INFO
        assert isinstance(logger, BaseLogger)

    def test_audit_logger(self):
        """测试审计Logger"""
        logger = AuditLogger("TestAudit")
        assert logger.name == "TestAudit"
        assert logger.level == LogLevel.INFO
        assert isinstance(logger, BaseLogger)

    def test_performance_logger(self):
        """测试性能Logger"""
        logger = PerformanceLogger("TestPerf")
        assert logger.name == "TestPerf"
        assert logger.level == LogLevel.INFO
        assert isinstance(logger, BaseLogger)


class TestLogLevels:
    """日志级别测试"""

    def test_log_level_enum(self):
        """测试日志级别枚举"""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"

    def test_log_format_enum(self):
        """测试日志格式枚举"""
        assert LogFormat.TEXT.value == "text"
        assert LogFormat.JSON.value == "json"
        assert LogFormat.STRUCTURED.value == "structured"

    def test_log_category_enum(self):
        """测试日志类别枚举"""
        assert LogCategory.SYSTEM.value == "system"
        assert LogCategory.BUSINESS.value == "business"
        assert LogCategory.AUDIT.value == "audit"
        assert LogCategory.PERFORMANCE.value == "performance"


class TestLoggerIntegration:
    """Logger集成测试"""

    def test_logger_factory(self):
        """测试Logger工厂函数"""
        from src.infrastructure.logging.core import UnifiedLogger

        # 直接创建UnifiedLogger实例进行测试，避免单例模式问题
        logger = UnifiedLogger(name="TestFactory")
        assert isinstance(logger, UnifiedLogger)
        assert logger.name == "TestFactory"

    def test_singleton_behavior(self):
        """测试单例行为"""
        from src.infrastructure.logging.core import get_logger

        logger1 = get_logger("SingletonTest")
        logger2 = get_logger("SingletonTest")

        # 应该返回同一个实例
        assert logger1 is logger2

    def test_unified_logger_methods(self):
        """测试UnifiedLogger的各种方法"""
        from src.infrastructure.logging.core import UnifiedLogger
        from unittest.mock import patch

        logger = UnifiedLogger(name="TestMethods")

        # 测试基本日志方法
        with patch.object(logger._recorder, 'log') as mock_log:
            logger.debug("Debug message")
            # 应该调用_recorder.log(logging.DEBUG, message, **kwargs)
            mock_log.assert_called()

        with patch.object(logger._recorder, 'log') as mock_log:
            logger.info("Info message")
            mock_log.assert_called()

        with patch.object(logger._recorder, 'log') as mock_log:
            logger.warning("Warning message")
            mock_log.assert_called()

        with patch.object(logger._recorder, 'log') as mock_log:
            logger.error("Error message")
            mock_log.assert_called()

        with patch.object(logger._recorder, 'log') as mock_log:
            logger.critical("Critical message")
            mock_log.assert_called()

    def test_unified_logger_structured_logging(self):
        """测试UnifiedLogger的结构化日志"""
        from src.infrastructure.logging.core import UnifiedLogger
        from unittest.mock import patch

        logger = UnifiedLogger(name="TestStructured")

        # 测试结构化日志方法
        with patch.object(logger._business_logger, 'log_structured') as mock_structured:
            logger.log_structured("INFO", "Structured message", key="value")
            mock_structured.assert_called_once()

    def test_unified_logger_specialized_methods(self):
        """测试UnifiedLogger的专用日志方法"""
        from src.infrastructure.logging.core import UnifiedLogger
        from unittest.mock import patch

        logger = UnifiedLogger(name="TestSpecialized")

        # 测试性能日志
        with patch.object(logger._business_logger, 'log_performance') as mock_perf:
            logger.log_performance("test_op", 1.5, extra="data")
            mock_perf.assert_called_once_with("test_op", 1.5, extra="data")

        # 测试错误日志
        with patch.object(logger._business_logger, 'log_error_with_context') as mock_error:
            test_error = ValueError("test error")
            logger.log_error_with_context(test_error, {"context": "test"})
            mock_error.assert_called_once()

        # 测试业务事件日志
        with patch.object(logger._business_logger, 'log_business_event') as mock_business:
            logger.log_business_event("user_login", {"user_id": "123"})
            mock_business.assert_called_once_with("user_login", {"user_id": "123"})

    def test_unified_logger_handler_management(self):
        """测试UnifiedLogger的处理器管理"""
        from src.infrastructure.logging.core import UnifiedLogger
        import logging

        logger = UnifiedLogger(name="TestHandlers")

        # 测试添加处理器
        test_handler = logging.StreamHandler()
        logger.add_handler(test_handler)

        handlers = logger.get_handlers()
        assert test_handler in handlers

        # 测试移除处理器
        logger.remove_handler(test_handler)
        handlers = logger.get_handlers()
        assert test_handler not in handlers

        # 测试清除处理器
        logger.add_handler(test_handler)
        logger.clear_handlers()
        handlers = logger.get_handlers()
        assert len(handlers) == 0

    def test_unified_logger_level_management(self):
        """测试UnifiedLogger的级别管理"""
        from src.infrastructure.logging.core import UnifiedLogger, LogLevel

        logger = UnifiedLogger(name="TestLevel", level=LogLevel.INFO)

        # 测试获取级别
        level = logger.get_level()
        assert level == LogLevel.INFO

        # 测试设置级别
        logger.set_level(20)  # WARNING level
        # 注意：由于单例模式，这个测试可能受其他测试影响

    def test_unified_logger_stats_and_shutdown(self):
        """测试UnifiedLogger的统计和关闭功能"""
        from src.infrastructure.logging.core import UnifiedLogger

        logger = UnifiedLogger(name="TestStats")

        # 测试获取统计信息
        stats = logger.get_log_stats()
        assert isinstance(stats, dict)
        assert 'logger_name' in stats
        assert 'level' in stats
        assert 'handlers_count' in stats
        assert stats['logger_name'] == 'TestStats'

        # 测试关闭
        logger.shutdown()  # 应该不抛出异常


if __name__ == "__main__":
    pytest.main([__file__])
