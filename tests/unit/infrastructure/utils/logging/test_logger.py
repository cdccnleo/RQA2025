#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层日志器组件测试

测试目标：提升utils/logging/logger.py的真实覆盖率
实际导入和使用src.infrastructure.utils.logging.logger模块
"""

import pytest


class TestLoggerModule:
    """测试日志器模块"""
    
    def test_import_unified_logger(self):
        """测试导入UnifiedLogger"""
        from src.infrastructure.utils.logging.logger import UnifiedLogger
        
        logger = UnifiedLogger("test")
        assert logger is not None
    
    def test_import_get_unified_logger(self):
        """测试导入get_unified_logger函数"""
        from src.infrastructure.utils.logging.logger import get_unified_logger
        
        logger = get_unified_logger("test")
        assert logger is not None
    
    def test_import_get_logger(self):
        """测试导入get_logger函数"""
        from src.infrastructure.utils.logging.logger import get_logger
        
        logger = get_logger("test")
        assert logger is not None
    
    def test_get_unified_logger_with_name(self):
        """测试使用名称获取统一日志器"""
        from src.infrastructure.utils.logging.logger import get_unified_logger
        
        logger = get_unified_logger("test_logger")
        assert logger is not None
    
    def test_get_unified_logger_without_name(self):
        """测试不使用名称获取统一日志器"""
        from src.infrastructure.utils.logging.logger import get_unified_logger
        
        logger = get_unified_logger()
        assert logger is not None
    
    def test_get_logger_with_name(self):
        """测试使用名称获取日志器"""
        from src.infrastructure.utils.logging.logger import get_logger
        
        logger = get_logger("test_logger")
        assert logger is not None
    
    def test_unified_logger_init(self):
        """测试UnifiedLogger初始化"""
        from src.infrastructure.utils.logging.logger import UnifiedLogger
        
        logger = UnifiedLogger("test")
        assert hasattr(logger, 'logger')

