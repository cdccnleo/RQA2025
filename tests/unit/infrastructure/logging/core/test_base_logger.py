"""
测试基础日志器

覆盖 base_logger.py 中的所有类和功能
"""

import pytest
import logging as python_logging
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.logging.core.base_logger import (
    BaseLogger,
    BusinessLogger,
    AuditLogger,
    PerformanceLogger
)
from src.infrastructure.logging.core.interfaces import LogLevel


class TestBaseLogger:
    """BaseLogger 类测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        logger = BaseLogger()

        assert logger.name == "BaseLogger"
        assert logger.level == LogLevel.INFO
        assert logger._buffer_size == 1000
        assert len(logger._buffer) == 0
        assert logger._stats == {
            "debug": 0, "info": 0, "warning": 0, "error": 0, "critical": 0
        }

    def test_initialization_custom(self):
        """测试自定义初始化"""
        logger = BaseLogger(name="CustomLogger", level=LogLevel.DEBUG)

        assert logger.name == "CustomLogger"
        assert logger.level == LogLevel.DEBUG

    def test_log_method_calls_python_logging(self):
        """测试log方法调用Python logging"""
        logger = BaseLogger()
        # 测试基本的log功能，验证统计计数器更新
        initial_stats = logger._stats.copy()

        logger.log(LogLevel.INFO, "Test message", key="value")

        # 验证统计计数器增加了
        assert logger._stats["info"] == initial_stats["info"] + 1
        assert logger._stats["debug"] == initial_stats["debug"]
        assert logger._stats["warning"] == initial_stats["warning"]
        assert logger._stats["error"] == initial_stats["error"]
        assert logger._stats["critical"] == initial_stats["critical"]

    def test_log_with_different_levels(self):
        """测试不同日志级别"""
        # 使用DEBUG级别以记录所有日志
        logger = BaseLogger(level=LogLevel.DEBUG)

        # 测试所有级别，验证统计计数器
        initial_stats = logger._stats.copy()

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        # 验证所有级别的计数器都增加了
        assert logger._stats["debug"] == initial_stats["debug"] + 1
        assert logger._stats["info"] == initial_stats["info"] + 1
        assert logger._stats["warning"] == initial_stats["warning"] + 1
        assert logger._stats["error"] == initial_stats["error"] + 1
        assert logger._stats["critical"] == initial_stats["critical"] + 1

    def test_should_log_method(self):
        """测试是否应该记录日志"""
        logger = BaseLogger(level=LogLevel.WARNING)

        # 高于当前级别的日志应该被记录
        assert logger._should_log(LogLevel.ERROR) == True
        assert logger._should_log(LogLevel.CRITICAL) == True

        # 低于当前级别的日志不应该被记录
        assert logger._should_log(LogLevel.INFO) == False
        assert logger._should_log(LogLevel.DEBUG) == False

        # 等于当前级别的日志应该被记录
        assert logger._should_log(LogLevel.WARNING) == True

    def test_format_message(self):
        """测试消息格式化"""
        logger = BaseLogger()

        # 无额外参数
        result = logger._format_message("Simple message", {})
        assert result == "Simple message"

        # 有额外参数
        result = logger._format_message("Message", {"user": "john", "action": "login"})
        assert "Message" in result
        assert "user=john" in result
        assert "action=login" in result

    def test_set_level(self):
        """测试设置日志级别"""
        logger = BaseLogger()

        logger.set_level(LogLevel.DEBUG)
        assert logger.level == LogLevel.DEBUG

        logger.set_level(LogLevel.ERROR)
        assert logger.level == LogLevel.ERROR

    def test_get_level(self):
        """测试获取日志级别"""
        logger = BaseLogger(level=LogLevel.WARNING)

        assert logger.get_level() == LogLevel.WARNING

    def test_get_buffered_records(self):
        """测试获取缓冲记录"""
        logger = BaseLogger()

        records = logger.get_buffered_records()

        assert isinstance(records, dict)
        assert "buffer" in records
        assert "size" in records
        assert records["size"] == 0
        assert records["buffer"] == []

    def test_get_stats(self):
        """测试获取统计信息"""
        logger = BaseLogger()

        stats = logger.get_stats()

        assert isinstance(stats, dict)
        assert "debug" in stats
        assert "info" in stats
        assert "warning" in stats
        assert "error" in stats
        assert "critical" in stats

        # 初始统计应该都是0
        for level in ["debug", "info", "warning", "error", "critical"]:
            assert stats[level] == 0

    def test_normalize_level(self):
        """测试级别标准化"""
        logger = BaseLogger()

        # LogLevel枚举
        level, value = logger._normalize_level(LogLevel.INFO)
        assert level == LogLevel.INFO
        assert value == 20

        # 字符串
        level, value = logger._normalize_level("DEBUG")
        assert level == LogLevel.DEBUG
        assert value == 10

        # 整数
        level, value = logger._normalize_level(30)
        assert level == LogLevel.WARNING
        assert value == 30

    def test_normalize_level_invalid(self):
        """测试无效级别标准化"""
        logger = BaseLogger()

        # 无效输入应该返回默认值
        level, value = logger._normalize_level("INVALID")
        assert level == LogLevel.INFO  # 默认级别
        assert value == 20

    def test_is_fast_path_active(self):
        """测试快速路径是否激活"""
        logger = BaseLogger()

        # 默认应该返回False（因为没有实现快速路径）
        assert logger._is_fast_path_active() == False


class TestBusinessLogger:
    """BusinessLogger 类测试"""

    def test_inheritance(self):
        """测试继承关系"""
        logger = BusinessLogger()

        assert isinstance(logger, BaseLogger)
        assert logger.name == "BusinessLogger"
        assert logger.level == LogLevel.INFO

    def test_initialization_custom_name(self):
        """测试自定义名称初始化"""
        logger = BusinessLogger(name="CustomBusinessLogger")

        assert logger.name == "CustomBusinessLogger"

    def test_business_logging_methods(self):
        """测试业务日志记录方法"""
        logger = BusinessLogger()

        # 测试业务相关的日志记录
        initial_stats = logger._stats.copy()
        logger.info("User login", user_id=123, action="login")
        logger.warning("Payment failed", amount=99.99, reason="insufficient_funds")

        # 验证统计计数器
        assert logger._stats["info"] == initial_stats["info"] + 1
        assert logger._stats["warning"] == initial_stats["warning"] + 1


class TestAuditLogger:
    """AuditLogger 类测试"""

    def test_inheritance(self):
        """测试继承关系"""
        logger = AuditLogger()

        assert isinstance(logger, BaseLogger)
        assert logger.name == "AuditLogger"
        assert logger.level == LogLevel.INFO

    def test_initialization_custom_name(self):
        """测试自定义名称初始化"""
        logger = AuditLogger(name="CustomAuditLogger")

        assert logger.name == "CustomAuditLogger"

    def test_audit_logging_methods(self):
        """测试审计日志记录方法"""
        logger = AuditLogger()

        # 测试审计相关的日志记录
        initial_stats = logger._stats.copy()
        logger.info("Security event", user="admin", action="login", ip="192.168.1.1")
        logger.warning("Policy violation", policy="password_policy", user="user123")

        # 验证统计计数器
        assert logger._stats["info"] == initial_stats["info"] + 1
        assert logger._stats["warning"] == initial_stats["warning"] + 1


class TestPerformanceLogger:
    """PerformanceLogger 类测试"""

    def test_inheritance(self):
        """测试继承关系"""
        logger = PerformanceLogger()

        assert isinstance(logger, BaseLogger)
        assert logger.name == "PerformanceLogger"
        assert logger.level == LogLevel.INFO

    def test_initialization_custom_name(self):
        """测试自定义名称初始化"""
        logger = PerformanceLogger(name="CustomPerformanceLogger")

        assert logger.name == "CustomPerformanceLogger"

    def test_performance_logging_methods(self):
        """测试性能日志记录方法"""
        logger = PerformanceLogger()

        # 测试性能相关的日志记录
        initial_stats = logger._stats.copy()
        logger.info("Query executed", query_time=0.125, rows_returned=100)
        logger.warning("Slow query detected", duration=5.2, query="SELECT * FROM large_table")

        # 验证统计计数器
        assert logger._stats["info"] == initial_stats["info"] + 1
        assert logger._stats["warning"] == initial_stats["warning"] + 1


class TestBaseLoggerIntegration:
    """BaseLogger 集成测试"""

    def test_full_logging_workflow(self):
        """测试完整日志工作流"""
        logger = BaseLogger(name="TestLogger", level=LogLevel.DEBUG)

        # 执行各种日志操作
        logger.debug("Debug message", component="test")
        logger.info("Info message", user="john")
        logger.warning("Warning message", error_code=500)
        logger.error("Error message", exception="ValueError")
        logger.critical("Critical message", system="down")

        # 验证统计信息
        stats = logger.get_stats()
        assert stats["debug"] == 1
        assert stats["info"] == 1
        assert stats["warning"] == 1
        assert stats["error"] == 1
        assert stats["critical"] == 1

        # 验证基本属性
        assert stats["name"] == "TestLogger"
        assert stats["level"] == "DEBUG"
        assert stats["type"] == "BaseLogger"

    def test_level_filtering_integration(self):
        """测试级别过滤集成"""
        logger = BaseLogger(level=LogLevel.WARNING)

        # 只应该记录WARNING及以上的级别
        logger.debug("Debug - should not log")
        logger.info("Info - should not log")
        logger.warning("Warning - should log")
        logger.error("Error - should log")
        logger.critical("Critical - should log")

        # 验证统计计数器，只有WARNING及以上的级别应该被计数
        stats = logger.get_stats()
        assert stats["debug"] == 0  # DEBUG级别被过滤
        assert stats["info"] == 0   # INFO级别被过滤
        assert stats["warning"] == 1
        assert stats["error"] == 1
        assert stats["critical"] == 1

    def test_multiple_loggers_isolation(self):
        """测试多个日志器隔离"""
        logger1 = BaseLogger(name="Logger1")
        logger2 = BaseLogger(name="Logger2")

        logger1.info("Message from logger1")
        logger2.info("Message from logger2")

        # 验证每个logger有自己的统计计数器
        stats1 = logger1.get_stats()
        stats2 = logger2.get_stats()

        assert stats1["name"] == "Logger1"
        assert stats2["name"] == "Logger2"
        assert stats1["info"] == 1
        assert stats2["info"] == 1

    def test_stats_and_buffer_integration(self):
        """测试统计和缓冲区集成"""
        logger = BaseLogger()

        # 初始状态
        stats = logger.get_stats()
        buffer = logger.get_buffered_records()

        # 验证统计计数器初始为0
        assert stats["debug"] == 0
        assert stats["info"] == 0
        assert stats["warning"] == 0
        assert stats["error"] == 0
        assert stats["critical"] == 0

        assert buffer["size"] == 0
        assert buffer["buffer"] == []

        # 验证返回的数据结构
        assert isinstance(stats, dict)
        assert isinstance(buffer, dict)
        assert "debug" in stats
        assert "size" in buffer
        assert "buffer" in buffer

    def test_different_logger_types_integration(self):
        """测试不同类型日志器集成"""
        # 创建不同类型的日志器
        business_logger = BusinessLogger()
        audit_logger = AuditLogger()
        performance_logger = PerformanceLogger()

        # 每个日志器都记录一条消息
        business_logger.info("Business event")
        audit_logger.info("Audit event")
        performance_logger.info("Performance event")

        # 验证每个日志器都有自己的名称和统计计数器
        assert business_logger.name == "BusinessLogger"
        assert audit_logger.name == "AuditLogger"
        assert performance_logger.name == "PerformanceLogger"

        assert business_logger._stats["info"] == 1
        assert audit_logger._stats["info"] == 1
        assert performance_logger._stats["info"] == 1

        assert audit_logger.name == "AuditLogger"
        assert performance_logger.name == "PerformanceLogger"
