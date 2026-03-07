#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层监控日志器组件测试

测试目标：提升utils/monitoring/logger.py的真实覆盖率
实际导入和使用src.infrastructure.utils.monitoring.logger模块
"""

import pytest
import logging


class TestLoggerModule:
    """测试日志器模块"""
    
    def test_get_logger_with_name(self):
        """测试使用名称获取日志器"""
        from src.infrastructure.utils.monitoring.logger import get_logger
        
        logger = get_logger("test_logger")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
    
    def test_get_logger_default_name(self):
        """测试使用默认名称获取日志器"""
        from src.infrastructure.utils.monitoring.logger import get_logger
        
        logger = get_logger()
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "rqa"
    
    def test_get_logger_different_names(self):
        """测试获取不同名称的日志器"""
        from src.infrastructure.utils.monitoring.logger import get_logger
        
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")
        
        assert logger1.name != logger2.name
        assert logger1.name == "logger1"
        assert logger2.name == "logger2"
    
    def test_logger_functionality(self):
        """测试日志器基本功能"""
        from src.infrastructure.utils.monitoring.logger import get_logger
        
        logger = get_logger("test")
        
        # 测试日志记录功能
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        assert True  # 如果没有抛出异常，说明功能正常

