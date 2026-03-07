"""
Unified Logger 单元测试

测试统一日志器模块的核心功能。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import logging
import tempfile
from pathlib import Path

from src.infrastructure.logging.core.unified_logger import (
    get_unified_logger,
    LogRecorder,
    UnifiedLogger,
    BusinessLoggerAdapter,
    TestableUnifiedLogger,
)
from src.infrastructure.logging.core.interfaces import LogLevel


class TestGetUnifiedLogger:
    """测试获取统一日志器函数"""

    def test_get_unified_logger(self):
        """测试获取统一日志器"""
        logger = get_unified_logger("test_unified")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_unified"

    def test_get_unified_logger_default_name(self):
        """测试获取默认名称的统一日志器"""
        logger = get_unified_logger()
        assert isinstance(logger, logging.Logger)
        assert "unified" in logger.name


class TestLogRecorder:
    """测试日志记录器"""

    @pytest.fixture
    def log_recorder(self):
        """创建日志记录器实例"""
        return LogRecorder("test_recorder")

    def test_init(self, log_recorder):
        """测试初始化"""
        assert isinstance(log_recorder.logger, logging.Logger)
        assert log_recorder.logger.name == "test_recorder"

    def test_log(self, log_recorder):
        """测试日志记录"""
        with patch.object(log_recorder.logger, 'log') as mock_log:
            log_recorder.log(logging.INFO, "Test message")

            mock_log.assert_called_once_with(logging.INFO, "Test message")


class TestUnifiedLogger:
    """测试统一日志器"""

    @pytest.fixture
    def unified_logger(self):
        """创建统一日志器实例"""
        return UnifiedLogger("test_unified", LogLevel.DEBUG, "test_category")

    def test_init(self, unified_logger):
        """测试初始化"""
        assert unified_logger.name == "test_unified"
        assert unified_logger.level == LogLevel.DEBUG
        assert unified_logger.category == "test_category"
        assert isinstance(unified_logger.logger, logging.Logger)

    def test_basic_operations(self, unified_logger):
        """测试基本操作"""
        # 测试设置和获取级别
        unified_logger.set_level(LogLevel.ERROR)
        assert unified_logger.get_level() == LogLevel.ERROR

        # 测试添加和移除处理器
        handler = logging.StreamHandler()
        unified_logger.add_handler(handler)
        assert handler in unified_logger.logger.handlers

        unified_logger.remove_handler(handler)
        assert handler not in unified_logger.logger.handlers


class TestUnifiedLoggerComprehensive:
    """统一日志器全面测试"""

    @pytest.fixture
    def unified_logger(self):
        """创建统一日志器实例"""
        return UnifiedLogger("test_unified", LogLevel.INFO, "test_category")

    def test_init_default(self):
        """测试默认初始化"""
        logger = UnifiedLogger()
        assert logger.name == "unified"
        assert logger.level == LogLevel.INFO
        assert logger.category == "general"
        assert isinstance(logger.logger, logging.Logger)

    def test_init_custom(self):
        """测试自定义初始化"""
        logger = UnifiedLogger("custom", LogLevel.DEBUG, "business")
        assert logger.name == "custom"
        assert logger.level == LogLevel.DEBUG
        assert logger.category == "business"

    def test_init_none_name(self):
        """测试空名称初始化"""
        logger = UnifiedLogger(None)
        assert logger.name == "unified"

    def test_normalize_level(self, unified_logger):
        """测试级别标准化"""
        # LogLevel枚举
        assert unified_logger._normalize_level(LogLevel.DEBUG) == LogLevel.DEBUG
        assert unified_logger._normalize_level(LogLevel.INFO) == LogLevel.INFO

        # 字符串
        assert unified_logger._normalize_level("DEBUG") == LogLevel.DEBUG
        assert unified_logger._normalize_level("info") == LogLevel.INFO

        # 数字
        assert unified_logger._normalize_level(10) == LogLevel.DEBUG
        assert unified_logger._normalize_level(20) == LogLevel.INFO

        # 无效值
        assert unified_logger._normalize_level("INVALID") == LogLevel.INFO

    def test_resolve_numeric_level(self, unified_logger):
        """测试数字级别解析"""
        assert unified_logger._resolve_numeric_level(LogLevel.DEBUG) == 10
        assert unified_logger._resolve_numeric_level(LogLevel.INFO) == 20
        assert unified_logger._resolve_numeric_level(LogLevel.WARNING) == 30
        assert unified_logger._resolve_numeric_level(LogLevel.ERROR) == 40
        assert unified_logger._resolve_numeric_level(LogLevel.CRITICAL) == 50

    def test_convert_level(self, unified_logger):
        """测试级别转换"""
        assert unified_logger._convert_level(LogLevel.DEBUG) == 10
        assert unified_logger._convert_level("DEBUG") == 10
        assert unified_logger._convert_level(10) == 10

    def test_log_structured(self, unified_logger):
        """测试结构化日志"""
        # 这个方法会通过_business_logger记录日志，验证它被调用了
        with patch.object(unified_logger._business_logger, 'log_structured') as mock_log:
            unified_logger.log_structured(LogLevel.INFO, "Test message", key="value")
            mock_log.assert_called_once_with("LogLevel.INFO", "Test message", key="value")

    def test_log_performance(self, unified_logger):
        """测试性能日志"""
        with patch.object(unified_logger._business_logger, 'log_performance') as mock_log:
            unified_logger.log_performance("operation", 1.5, success=True)
            mock_log.assert_called_once_with("operation", 1.5, success=True)

    def test_log_error_with_context(self, unified_logger):
        """测试带上下文的错误日志"""
        error = ValueError("Test error")
        context = {"user": "test", "action": "login"}

        with patch.object(unified_logger._business_logger, 'log_error_with_context') as mock_log:
            unified_logger.log_error_with_context(error, context)
            mock_log.assert_called_once_with(error, context)

    def test_log_business_event(self, unified_logger):
        """测试业务事件日志"""
        event_data = {"type": "user_action", "user_id": 123}

        with patch.object(unified_logger._business_logger, 'log_business_event') as mock_log:
            unified_logger.log_business_event("user_login", event_data)
            mock_log.assert_called_once_with("user_login", event_data)

    def test_add_handler(self, unified_logger):
        """测试添加处理器"""
        mock_handler = Mock()
        unified_logger.add_handler(mock_handler)
        assert mock_handler in unified_logger.logger.handlers

    def test_remove_handler(self, unified_logger):
        """测试移除处理器"""
        mock_handler = Mock()
        unified_logger.add_handler(mock_handler)
        unified_logger.remove_handler(mock_handler)
        assert mock_handler not in unified_logger.logger.handlers

    def test_clear_handlers(self, unified_logger):
        """测试清除处理器"""
        mock_handler1 = Mock()
        mock_handler2 = Mock()
        unified_logger.add_handler(mock_handler1)
        unified_logger.add_handler(mock_handler2)

        # 验证处理器已被添加
        assert mock_handler1 in unified_logger.logger.handlers
        assert mock_handler2 in unified_logger.logger.handlers

        unified_logger.clear_handlers()

        # 验证处理器已被清除
        assert mock_handler1 not in unified_logger.logger.handlers
        assert mock_handler2 not in unified_logger.logger.handlers
        assert len(unified_logger._custom_handlers) == 0

    def test_get_handlers(self, unified_logger):
        """测试获取处理器"""
        mock_handler = Mock()
        unified_logger.add_handler(mock_handler)
        handlers = unified_logger.get_handlers()
        assert mock_handler in handlers

    def test_add_filter(self, unified_logger):
        """测试添加过滤器"""
        mock_filter = Mock()
        unified_logger.add_filter(mock_filter)
        assert mock_filter in unified_logger.logger.filters

    def test_remove_filter(self, unified_logger):
        """测试移除过滤器"""
        mock_filter = Mock()
        unified_logger.add_filter(mock_filter)
        unified_logger.remove_filter(mock_filter)
        assert mock_filter not in unified_logger.logger.filters

    def test_set_level(self, unified_logger):
        """测试设置级别"""
        unified_logger.set_level(LogLevel.DEBUG)
        assert unified_logger.level == LogLevel.DEBUG
        assert unified_logger.logger.level == 10

    def test_get_level(self, unified_logger):
        """测试获取级别"""
        level = unified_logger.get_level()
        assert level == LogLevel.INFO

    def test_get_stats(self, unified_logger):
        """测试获取统计信息"""
        stats = unified_logger.get_stats()
        assert isinstance(stats, dict)
        assert "total" in stats
        assert "counts" in stats
        assert isinstance(stats["counts"], dict)

    def test_get_log_stats(self, unified_logger):
        """测试获取日志统计"""
        stats = unified_logger.get_log_stats()
        assert isinstance(stats, dict)

    def test_debug_method(self, unified_logger):
        """测试debug方法"""
        with patch.object(unified_logger, '_update_stats') as mock_update, \
             patch.object(unified_logger._recorder, 'log') as mock_recorder, \
             patch.object(unified_logger.logger, 'debug') as mock_logger_debug:
            unified_logger.debug("Debug message", extra="data")
            mock_update.assert_called_once_with("DEBUG", "Debug message", extra="data")
            mock_recorder.assert_called_once()
            mock_logger_debug.assert_called_once_with("Debug message", extra="data")

    def test_info_method(self, unified_logger):
        """测试info方法"""
        with patch.object(unified_logger, '_update_stats') as mock_update, \
             patch.object(unified_logger._recorder, 'log') as mock_recorder, \
             patch.object(unified_logger.logger, 'info') as mock_logger_info:
            unified_logger.info("Info message")
            mock_update.assert_called_once_with("INFO", "Info message")
            mock_recorder.assert_called_once()
            mock_logger_info.assert_called_once_with("Info message")

    def test_warning_method(self, unified_logger):
        """测试warning方法"""
        with patch.object(unified_logger, '_update_stats') as mock_update, \
             patch.object(unified_logger._recorder, 'log') as mock_recorder, \
             patch.object(unified_logger.logger, 'warning') as mock_logger_warning:
            unified_logger.warning("Warning message")
            mock_update.assert_called_once_with("WARNING", "Warning message")
            mock_recorder.assert_called_once()
            mock_logger_warning.assert_called_once_with("Warning message")

    def test_error_method(self, unified_logger):
        """测试error方法"""
        with patch.object(unified_logger, '_update_stats') as mock_update, \
             patch.object(unified_logger._recorder, 'log') as mock_recorder, \
             patch.object(unified_logger.logger, 'error') as mock_logger_error:
            unified_logger.error("Error message")
            mock_update.assert_called_once_with("ERROR", "Error message")
            mock_recorder.assert_called_once()
            mock_logger_error.assert_called_once_with("Error message")

    def test_critical_method(self, unified_logger):
        """测试critical方法"""
        with patch.object(unified_logger, '_update_stats') as mock_update, \
             patch.object(unified_logger._recorder, 'log') as mock_recorder, \
             patch.object(unified_logger.logger, 'critical') as mock_logger_critical:
            unified_logger.critical("Critical message")
            mock_update.assert_called_once_with("CRITICAL", "Critical message")
            mock_recorder.assert_called_once()
            mock_logger_critical.assert_called_once_with("Critical message")

    def test_log_method(self, unified_logger):
        """测试log方法"""
        with patch.object(unified_logger, '_update_stats') as mock_update, \
             patch.object(unified_logger._recorder, 'log') as mock_recorder, \
             patch.object(unified_logger.logger, 'log') as mock_logger_log:
            unified_logger.log("INFO", "Log message")
            mock_update.assert_called_once_with("INFO", "Log message")
            mock_recorder.assert_called_once()
            # logger.log使用数值level，而不是字符串
            mock_logger_log.assert_called_once_with(20, "Log message")

    def test_get_log_history(self, unified_logger):
        """测试获取日志历史"""
        history = unified_logger.get_log_history()
        assert isinstance(history, list)

    def test_get_log_history_with_limit(self, unified_logger):
        """测试获取有限日志历史"""
        history = unified_logger.get_log_history(limit=5)
        assert isinstance(history, list)
        assert len(history) <= 5

    def test_shutdown(self, unified_logger):
        """测试关闭日志器"""
        unified_logger.shutdown()
        # 验证日志器仍然可用
        assert unified_logger.logger is not None


class TestBusinessLoggerAdapter:
    """测试业务日志器适配器"""

    @pytest.fixture
    def adapter(self):
        """创建业务日志器适配器实例"""
        return BusinessLoggerAdapter("business_test", LogLevel.INFO, "test_category")

    def test_init(self, adapter):
        """测试初始化"""
        assert adapter.name == "business_test"
        assert adapter.level == LogLevel.INFO
        assert adapter.category == "test_category"

    def test_log_structured(self, adapter):
        """测试结构化日志"""
        with patch.object(adapter.logger, 'info') as mock_info:
            adapter.log_structured("INFO", "Business message", key="value")
            mock_info.assert_called_once()
            # 验证日志消息包含结构化数据
            call_args = mock_info.call_args[0][0]
            assert "Structured:" in call_args
            assert "Business message" in call_args

    def test_log_performance(self, adapter):
        """测试性能日志"""
        with patch.object(adapter.logger, 'info') as mock_info:
            adapter.log_performance("db_query", 0.5, rows=100)
            mock_info.assert_called_once()
            # 验证日志消息包含性能数据
            call_args = mock_info.call_args[0][0]
            assert "Performance:" in call_args
            assert "db_query" in call_args

    def test_log_error_with_context(self, adapter):
        """测试带上下文的错误日志"""
        with patch.object(adapter.logger, 'error') as mock_error:
            error = Exception("Test error")
            context = {"operation": "save"}
            adapter.log_error_with_context(error, context)
            mock_error.assert_called_once()
            # 验证日志消息包含错误和上下文
            call_args = mock_error.call_args[0][0]
            assert "Error with context:" in call_args
            assert "Test error" in call_args

    def test_log_business_event(self, adapter):
        """测试业务事件日志"""
        with patch.object(adapter.logger, 'info') as mock_info:
            event_data = {"event": "user_login", "user_id": 123}
            adapter.log_business_event("authentication", event_data)
            mock_info.assert_called_once()
            # 验证日志消息包含业务事件数据
            call_args = mock_info.call_args[0][0]
            assert "Business event:" in call_args
            assert "authentication" in call_args


class TestTestableUnifiedLogger:
    """测试可测试统一日志器"""

    @pytest.fixture
    def testable_logger(self):
        """创建可测试日志器实例"""
        return TestableUnifiedLogger("testable", LogLevel.DEBUG)

    def test_init(self, testable_logger):
        """测试初始化"""
        assert testable_logger.name == "testable"
        assert testable_logger.level == LogLevel.DEBUG
        assert hasattr(testable_logger, '_log_history')

    def test_log_methods(self, testable_logger):
        """测试各种日志方法"""
        testable_logger.debug("Debug message")
        testable_logger.info("Info message")
        testable_logger.warning("Warning message")
        testable_logger.error("Error message")
        testable_logger.critical("Critical message")

        history = testable_logger.get_log_history()
        assert len(history) == 5

    def test_exception_method(self, testable_logger):
        """测试exception方法"""
        try:
            raise ValueError("Test exception")
        except ValueError:
            testable_logger.exception("Exception occurred")

        history = testable_logger.get_log_history()
        assert len(history) == 1
        assert "error" in history[0]["level"].lower()

    def test_add_remove_handlers(self, testable_logger):
        """测试添加和移除处理器"""
        mock_handler = Mock()
        testable_logger.addHandler(mock_handler)
        assert mock_handler in testable_logger.handlers

        testable_logger.removeHandler(mock_handler)
        assert mock_handler not in testable_logger.handlers

    def test_add_remove_filters(self, testable_logger):
        """测试添加和移除过滤器"""
        mock_filter = Mock()
        testable_logger.addFilter(mock_filter)
        assert mock_filter in testable_logger.logger.filters

        testable_logger.removeFilter(mock_filter)
        assert mock_filter not in testable_logger.logger.filters

    def test_get_log_history_with_filters(self, testable_logger):
        """测试带过滤器的日志历史获取"""
        testable_logger.debug("Debug message")
        testable_logger.info("Info message")
        testable_logger.error("Error message")

        # 获取所有日志
        all_logs = testable_logger.get_log_history()
        assert len(all_logs) == 3

        # 获取特定级别的日志
        error_logs = testable_logger.get_log_history(level="ERROR")
        assert len(error_logs) == 1
        assert error_logs[0]["message"] == "Error message"

    def test_clear_history(self, testable_logger):
        """测试清除历史"""
        testable_logger.info("Test message")
        assert len(testable_logger.get_log_history()) == 1

        testable_logger.clear_history()
        assert len(testable_logger.get_log_history()) == 0

    def test_get_stats(self, testable_logger):
        """测试获取统计信息"""
        testable_logger.debug("Debug")
        testable_logger.info("Info")
        testable_logger.error("Error")

        stats = testable_logger.get_log_stats()
        assert isinstance(stats, dict)
        assert stats["performance"]["total_logs"] == 3
        assert stats["counts"]["DEBUG"] == 1
        assert stats["counts"]["INFO"] == 1
        assert stats["counts"]["ERROR"] == 1

    def test_get_logs(self, testable_logger):
        """测试获取日志"""
        testable_logger.info("Info log")
        testable_logger.error("Error log")

        all_logs = testable_logger.get_logs()
        assert len(all_logs) == 2

        error_logs = testable_logger.get_logs(level="ERROR")
        assert len(error_logs) == 1
        assert "Error log" in error_logs[0]["message"]

    def test_should_log(self, testable_logger):
        """测试是否应该记录日志"""
        testable_logger.setLevel(LogLevel.INFO)

        assert testable_logger._should_log("DEBUG") == False
        assert testable_logger._should_log("INFO") == True
        assert testable_logger._should_log("ERROR") == True

    def test_set_level_methods(self, testable_logger):
        """测试设置级别的方法"""
        # 测试setLevel方法
        testable_logger.setLevel(LogLevel.WARNING)
        assert testable_logger.level == LogLevel.WARNING

        # 测试set_level方法
        testable_logger.set_level(LogLevel.DEBUG)
        assert testable_logger.level == LogLevel.DEBUG



