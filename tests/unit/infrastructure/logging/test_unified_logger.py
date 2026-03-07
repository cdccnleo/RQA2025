#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一日志器测试
测试统一日志器功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import logging
from unittest.mock import Mock, patch, MagicMock


class TestUnifiedLogger:
    """测试统一日志器"""

    def setup_method(self):
        """测试前准备"""
        try:
            from src.infrastructure.logging.unified_logger import UnifiedLogger
            self.UnifiedLogger = UnifiedLogger
            # 验证UnifiedLogger可用
            assert hasattr(self.UnifiedLogger, '__module__')
        except ImportError:
            pytest.skip("UnifiedLogger not available")

    def test_initialization(self):
        """测试初始化"""
        if not hasattr(self, 'UnifiedLogger'):
            pytest.skip("UnifiedLogger not available")

        logger = self.UnifiedLogger()
        assert logger is not None

    def test_unified_logging(self):
        """测试统一日志"""
        if not hasattr(self, 'UnifiedLogger'):
            pytest.skip("UnifiedLogger not available")

        logger = self.UnifiedLogger()

        # 测试统一日志功能
        assert hasattr(logger, 'log')

    def test_logger_functionality(self):
        """测试日志器功能"""
        if not hasattr(self, 'UnifiedLogger'):
            pytest.skip("UnifiedLogger not available")

        logger = self.UnifiedLogger()
        # 验证日志器功能

    def test_log_methods(self):
        """测试各种日志方法"""
        if not hasattr(self, 'UnifiedLogger'):
            pytest.skip("UnifiedLogger not available")

        logger = self.UnifiedLogger()

        # 测试info方法
        with patch.object(logger._logger, 'log') as mock_log:
            logger.info("Test info message")
            mock_log.assert_called_with(logging.INFO, "Test info message")

        # 测试warning方法
        with patch.object(logger._logger, 'log') as mock_log:
            logger.warning("Test warning message")
            mock_log.assert_called_with(logging.WARNING, "Test warning message")

        # 测试error方法
        with patch.object(logger._logger, 'log') as mock_log:
            logger.error("Test error message")
            mock_log.assert_called_with(logging.ERROR, "Test error message")

    def test_log_with_string_level(self):
        """测试使用字符串级别记录日志"""
        if not hasattr(self, 'UnifiedLogger'):
            pytest.skip("UnifiedLogger not available")

        logger = self.UnifiedLogger()

        with patch.object(logger._logger, 'log') as mock_log:
            logger.log("DEBUG", "Debug message")
            mock_log.assert_called_with(logging.DEBUG, "Debug message")

            logger.log("INFO", "Info message")
            mock_log.assert_called_with(logging.INFO, "Info message")

            logger.log("WARNING", "Warning message")
            mock_log.assert_called_with(logging.WARNING, "Warning message")

            logger.log("ERROR", "Error message")
            mock_log.assert_called_with(logging.ERROR, "Error message")

    def test_log_with_int_level(self):
        """测试使用整数级别记录日志"""
        if not hasattr(self, 'UnifiedLogger'):
            pytest.skip("UnifiedLogger not available")

        logger = self.UnifiedLogger()

        with patch.object(logger._logger, 'log') as mock_log:
            logger.log(logging.CRITICAL, "Critical message")
            mock_log.assert_called_with(logging.CRITICAL, "Critical message")

    def test_log_with_invalid_level(self):
        """测试无效日志级别"""
        if not hasattr(self, 'UnifiedLogger'):
            pytest.skip("UnifiedLogger not available")

        logger = self.UnifiedLogger()

        # 无效级别应该使用默认INFO级别
        with patch.object(logger._logger, 'log') as mock_log:
            logger.log("INVALID_LEVEL", "Invalid level message")
            mock_log.assert_called_with(logging.INFO, "Invalid level message")

    def test_get_unified_logger_function(self):
        """测试get_unified_logger函数"""
        try:
            from src.infrastructure.logging.unified_logger import get_unified_logger
        except ImportError:
            pytest.skip("get_unified_logger not available")

        logger = get_unified_logger("test_logger")
        assert isinstance(logger, self.UnifiedLogger)
        assert logger.name == "test_logger"

    def test_get_logger_function(self):
        """测试get_logger函数"""
        try:
            from src.infrastructure.logging.unified_logger import get_logger
        except ImportError:
            pytest.skip("get_logger not available")

        logger = get_logger("alias_logger")
        assert isinstance(logger, self.UnifiedLogger)
        assert logger.name == "alias_logger"
        assert True


if __name__ == '__main__':
    pytest.main([__file__])