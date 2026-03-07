#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 统一日志管理接口

测试logging/core/unified_logging_interface.py中的所有类和方法
"""

import pytest
import json
from unittest.mock import Mock, MagicMock
from datetime import datetime


class TestUnifiedLoggingInterface:
    """测试统一日志管理接口"""

    def setup_method(self):
        """测试前准备"""
        try:
            from src.infrastructure.logging.core.unified_logging_interface import (
                LogLevel, LogFormat, LogCategory, ILogHandler, ILogFormatter,
                ILogger, ILogManager, ILogMonitor
            )
            self.LogLevel = LogLevel
            self.LogFormat = LogFormat
            self.LogCategory = LogCategory
            self.ILogHandler = ILogHandler
            self.ILogFormatter = ILogFormatter
            self.ILogger = ILogger
            self.ILogManager = ILogManager
            self.ILogMonitor = ILogMonitor
        except ImportError as e:
            pytest.skip(f"Unified logging interface components not available: {e}")

    def test_log_level_enum(self):
        """测试日志级别枚举"""
        assert self.LogLevel.DEBUG.value == 10
        assert self.LogLevel.INFO.value == 20
        assert self.LogLevel.WARNING.value == 30
        assert self.LogLevel.ERROR.value == 40
        assert self.LogLevel.CRITICAL.value == 50

        # 测试字符串表示
        assert str(self.LogLevel.DEBUG) == "LogLevel.DEBUG"
        assert self.LogLevel.INFO.name == "INFO"

    def test_log_format_enum(self):
        """测试日志格式枚举"""
        assert self.LogFormat.JSON.value == "json"
        assert self.LogFormat.TEXT.value == "text"
        assert self.LogFormat.STRUCTURED.value == "structured"
        assert self.LogFormat.XML.value == "xml"

    def test_log_category_enum(self):
        """测试日志分类枚举"""
        assert self.LogCategory.SYSTEM.value == "system"
        assert self.LogCategory.BUSINESS.value == "business"
        assert self.LogCategory.SECURITY.value == "security"
        assert self.LogCategory.PERFORMANCE.value == "performance"
        assert self.LogCategory.AUDIT.value == "audit"
        assert self.LogCategory.ERROR.value == "error"

    def test_log_enums_coverage(self):
        """测试日志枚举的完整覆盖"""
        # 测试所有LogLevel值
        for level in self.LogLevel:
            assert isinstance(level.value, int)
            assert level.value >= 10 and level.value <= 50

        # 测试所有LogFormat值
        for fmt in self.LogFormat:
            assert isinstance(fmt.value, str)
            assert fmt.value in ["json", "text", "structured", "xml"]

        # 测试所有LogCategory值
        for category in self.LogCategory:
            assert isinstance(category.value, str)
            assert len(category.value) > 0

    def test_interface_abstract_methods(self):
        """测试接口的抽象方法定义"""
        # ILogHandler
        assert hasattr(self.ILogHandler, 'emit')
        assert hasattr(self.ILogHandler, 'flush')
        assert hasattr(self.ILogHandler, 'close')

        # ILogFormatter
        assert hasattr(self.ILogFormatter, 'format')

        # ILogger - 检查实际的方法
        logger_methods = dir(self.ILogger)
        # ILogger主要定义了标准的日志方法

        # ILogManager
        assert hasattr(self.ILogManager, 'get_logger')
        assert hasattr(self.ILogManager, 'configure_logger')
        assert hasattr(self.ILogManager, 'remove_logger')
        assert hasattr(self.ILogManager, 'get_all_loggers')

        # ILogMonitor
        assert hasattr(self.ILogMonitor, 'on_log_record')
        assert hasattr(self.ILogMonitor, 'get_statistics')
        assert hasattr(self.ILogMonitor, 'get_recent_logs')

    def test_interface_instantiation_prevention(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            self.ILogHandler()

        with pytest.raises(TypeError):
            self.ILogFormatter()

        with pytest.raises(TypeError):
            self.ILogger()

        with pytest.raises(TypeError):
            self.ILogManager()

        with pytest.raises(TypeError):
            self.ILogMonitor()


if __name__ == '__main__':
    pytest.main([__file__])
