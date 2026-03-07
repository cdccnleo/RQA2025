#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - BaseLogger核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from src.infrastructure.logging.core.base_logger import BaseLogger
from src.infrastructure.logging.core.interfaces import LogLevel


class TestBaseLogger:
    """BaseLogger单元测试"""

    def test_initialization(self):
        """测试Logger初始化"""
        logger = BaseLogger(name="test_logger", level=LogLevel.INFO)
        assert logger.name == "test_logger"
        assert logger.level == LogLevel.INFO

    def test_dynamic_methods(self):
        """测试动态方法创建"""
        logger = BaseLogger(name="test_logger")

        # 验证动态方法存在
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'critical')

        # 测试方法调用不会抛出异常
        logger.info("Test message")
        logger.error("Test error")

    def test_set_get_level(self):
        """测试设置和获取日志级别"""
        logger = BaseLogger(name="test_logger", level=LogLevel.INFO)

        assert logger.get_level() == LogLevel.INFO

        logger.set_level(LogLevel.DEBUG)
        assert logger.get_level() == LogLevel.DEBUG

    def test_get_stats(self):
        """测试获取统计信息"""
        logger = BaseLogger(name="test_logger")
        stats = logger.get_stats()

        assert isinstance(stats, dict)
        assert 'name' in stats
        assert 'level' in stats
        assert stats['name'] == 'test_logger'

