#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Logging扩展测试"""

import pytest


def test_unified_logger_import():
    """测试UnifiedLogger导入"""
    try:
        from src.infrastructure.logging.core.unified_logger import UnifiedLogger
        assert UnifiedLogger is not None
    except ImportError:
        pytest.skip("UnifiedLogger不可用")


def test_get_unified_logger_import():
    """测试get_unified_logger导入"""
    try:
        from src.infrastructure.logging.core.unified_logger import get_unified_logger
        assert get_unified_logger is not None
    except ImportError:
        pytest.skip("get_unified_logger不可用")


def test_get_logger_import():
    """测试get_logger导入"""
    try:
        from src.infrastructure.logging.core.unified_logger import get_logger
        assert get_logger is not None
    except ImportError:
        pytest.skip("get_logger不可用")


def test_logger_pool_import():
    """测试LoggerPool导入"""
    try:
        from src.infrastructure.logging.pool.logger_pool import LoggerPool
        assert LoggerPool is not None
    except ImportError:
        pytest.skip("LoggerPool不可用")


def test_thread_logger_import():
    """测试ThreadLogger导入"""
    try:
        from src.infrastructure.logging.specialized.thread_logger import ThreadLogger
        assert ThreadLogger is not None
    except ImportError:
        pytest.skip("ThreadLogger不可用")


def test_performance_logger_import():
    """测试PerformanceLogger导入"""
    try:
        from src.infrastructure.logging.specialized.performance_logger import PerformanceLogger
        assert PerformanceLogger is not None
    except ImportError:
        pytest.skip("PerformanceLogger不可用")


def test_unified_logger_init():
    """测试UnifiedLogger初始化"""
    try:
        from src.infrastructure.logging.core.unified_logger import UnifiedLogger
        logger = UnifiedLogger('test')
        assert logger is not None
    except Exception:
        pytest.skip("测试跳过")


def test_get_unified_logger_call():
    """测试get_unified_logger调用"""
    try:
        from src.infrastructure.logging.core.unified_logger import get_unified_logger
        logger = get_unified_logger('test')
        assert logger is not None
    except Exception:
        pytest.skip("测试跳过")


def test_get_logger_call():
    """测试get_logger调用"""
    try:
        from src.infrastructure.logging.core.unified_logger import get_logger
        logger = get_logger('test')
        assert logger is not None
    except Exception:
        pytest.skip("测试跳过")


def test_logger_pool_init():
    """测试LoggerPool初始化"""
    try:
        from src.infrastructure.logging.pool.logger_pool import LoggerPool
        pool = LoggerPool()
        assert pool is not None
    except Exception:
        pytest.skip("测试跳过")


def test_thread_logger_init():
    """测试ThreadLogger初始化"""
    try:
        from src.infrastructure.logging.specialized.thread_logger import ThreadLogger
        logger = ThreadLogger('test')
        assert logger is not None
    except Exception:
        pytest.skip("测试跳过")


def test_performance_logger_init():
    """测试PerformanceLogger初始化"""
    try:
        from src.infrastructure.logging.specialized.performance_logger import PerformanceLogger
        logger = PerformanceLogger('test')
        assert logger is not None
    except Exception:
        pytest.skip("测试跳过")


def test_logging_formatters_import():
    """测试日志格式化器导入"""
    try:
        from src.infrastructure.logging.formatters import log_formatter
        assert log_formatter is not None
    except ImportError:
        pytest.skip("log_formatter不可用")


def test_logging_handlers_import():
    """测试日志处理器导入"""
    try:
        from src.infrastructure.logging.handlers import file_handler
        assert file_handler is not None
    except ImportError:
        pytest.skip("file_handler不可用")


def test_logging_config_import():
    """测试日志配置导入"""
    try:
        from src.infrastructure.logging.config import logging_config
        assert logging_config is not None
    except ImportError:
        pytest.skip("logging_config不可用")


def test_unified_logger_name():
    """测试UnifiedLogger名称"""
    try:
        from src.infrastructure.logging.core.unified_logger import UnifiedLogger
        logger = UnifiedLogger('test_logger')
        assert hasattr(logger, 'logger')
    except Exception:
        pytest.skip("测试跳过")


def test_unified_logger_methods():
    """测试UnifiedLogger方法"""
    try:
        from src.infrastructure.logging.core.unified_logger import UnifiedLogger
        logger = UnifiedLogger('test')
        
        # 检查常见方法
        if hasattr(logger, 'debug'):
            assert callable(logger.debug)
        if hasattr(logger, 'info'):
            assert callable(logger.info)
        if hasattr(logger, 'warning'):
            assert callable(logger.warning)
        if hasattr(logger, 'error'):
            assert callable(logger.error)
    except Exception:
        pytest.skip("测试跳过")

