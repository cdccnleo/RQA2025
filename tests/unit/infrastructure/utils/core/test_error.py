#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层统一错误处理器测试

测试目标：提升utils/core/error.py的真实覆盖率
实际导入和使用src.infrastructure.utils.core.error模块
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


class TestUnifiedErrorHandler:
    """测试统一错误处理器"""
    
    def test_init_default_logger(self):
        """测试使用默认日志器初始化"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        assert handler.logger is not None
        assert handler.error_stats == {}
        assert handler.last_errors == []
    
    def test_init_custom_logger(self):
        """测试使用自定义日志器初始化"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler(logger_name="custom.logger")
        assert handler.logger.name == "custom.logger"
    
    def test_handle_error(self):
        """测试处理错误"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        error = ValueError("Test error")
        
        handler.handle(error)
        
        assert "ValueError" in handler.error_stats
        assert handler.error_stats["ValueError"] == 1
        assert len(handler.last_errors) == 1
    
    def test_handle_error_with_context(self):
        """测试带上下文处理错误"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        error = ValueError("Test error")
        
        handler.handle(error, context="test_context")
        
        assert handler.last_errors[0]["context"] == "test_context"
        assert handler.last_errors[0]["error_type"] == "ValueError"
    
    def test_handle_error_with_level(self):
        """测试使用不同日志级别处理错误"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        error = ValueError("Test error")
        
        # 测试不同级别
        handler.handle(error, level="debug")
        handler.handle(error, level="info")
        handler.handle(error, level="warning")
        handler.handle(error, level="error")
        handler.handle(error, level="critical")
        
        assert handler.error_stats["ValueError"] == 5
    
    def test_error_stats_accumulation(self):
        """测试错误统计累积"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        
        handler.handle(ValueError("Error 1"))
        handler.handle(ValueError("Error 2"))
        handler.handle(TypeError("Error 3"))
        
        assert handler.error_stats["ValueError"] == 2
        assert handler.error_stats["TypeError"] == 1
    
    def test_error_history_management(self):
        """测试错误历史管理"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        
        # 添加超过100个错误
        for i in range(105):
            handler.handle(ValueError(f"Error {i}"))
        
        # 应该只保留最后100个
        assert len(handler.last_errors) == 100
        
        # 第一个错误应该是最新的
        assert handler.last_errors[-1]["error_type"] == "ValueError"
    
    def test_get_error_stats(self):
        """测试获取错误统计"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        
        handler.handle(ValueError("Error 1"))
        handler.handle(ValueError("Error 2"))
        handler.handle(TypeError("Error 3"))
        
        stats = handler.get_error_stats()
        assert stats["total_errors"] == 3
        assert stats["error_types"]["ValueError"] == 2
        assert stats["error_types"]["TypeError"] == 1
        assert stats["recent_errors_count"] == 3
    
    def test_get_recent_errors(self):
        """测试获取最近错误"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        
        handler.handle(ValueError("Error 1"))
        handler.handle(TypeError("Error 2"))
        handler.handle(KeyError("Error 3"))
        
        recent = handler.get_recent_errors(limit=2)
        assert len(recent) == 2
        assert recent[0]["error_type"] == "KeyError"
        assert recent[1]["error_type"] == "TypeError"
    
    def test_get_error_stats_structure(self):
        """测试获取错误统计结构"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        
        handler.handle(ValueError("Error 1"))
        handler.handle(ValueError("Error 2"))
        handler.handle(TypeError("Error 3"))
        
        stats = handler.get_error_stats()
        
        assert stats["total_errors"] == 3
        assert stats["error_types"] == {"ValueError": 2, "TypeError": 1}
        assert stats["recent_errors_count"] == 3
    
    def test_clear_stats(self):
        """测试清空错误统计"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        
        handler.handle(ValueError("Error 1"))
        handler.handle(TypeError("Error 2"))
        
        assert len(handler.last_errors) == 2
        assert len(handler.error_stats) == 2
        
        handler.clear_stats()
        
        assert len(handler.last_errors) == 0
        assert len(handler.error_stats) == 0
    
    def test_handle_connection_error(self):
        """测试处理连接错误"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        error = ConnectionError("Connection failed")
        
        handler.handle_connection_error(error, host="localhost", port=5432)
        
        assert len(handler.last_errors) == 1
        assert "Connection failed to localhost:5432" in handler.last_errors[0]["context"]
    
    def test_handle_connection_error_without_host(self):
        """测试处理连接错误（无主机信息）"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        error = ConnectionError("Connection failed")
        
        handler.handle_connection_error(error)
        
        assert len(handler.last_errors) == 1
        assert handler.last_errors[0]["context"] == "Connection failed"
    
    def test_handle_timeout_error(self):
        """测试处理超时错误"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        error = TimeoutError("Operation timed out")
        
        handler.handle_timeout_error(error, operation="query", timeout=30.0)
        
        assert len(handler.last_errors) == 1
        assert "Operation 'query' timed out after 30.0s" in handler.last_errors[0]["context"]
        assert handler.last_errors[0]["level"] == "warning"
    
    def test_handle_timeout_error_without_operation(self):
        """测试处理超时错误（无操作信息）"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        error = TimeoutError("Operation timed out")
        
        handler.handle_timeout_error(error, timeout=30.0)
        
        assert len(handler.last_errors) == 1
        assert "Timeout after 30.0s" in handler.last_errors[0]["context"]
    
    def test_handle_validation_error(self):
        """测试处理验证错误"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        error = ValueError("Invalid value")
        
        handler.handle_validation_error(error, field="email", value="invalid@")
        
        assert len(handler.last_errors) == 1
        assert "Validation failed for field 'email'" in handler.last_errors[0]["context"]
        assert handler.last_errors[0]["level"] == "warning"
    
    def test_handle_validation_error_without_field(self):
        """测试处理验证错误（无字段信息）"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        error = ValueError("Invalid value")
        
        handler.handle_validation_error(error)
        
        assert len(handler.last_errors) == 1
        assert handler.last_errors[0]["context"] == "Validation failed"
    
    def test_get_error_handler(self):
        """测试获取默认错误处理器"""
        from src.infrastructure.utils.core.error import get_error_handler
        
        handler = get_error_handler()
        assert handler is not None
        assert hasattr(handler, "handle")
    
    def test_handle_error_logging_failure(self):
        """测试日志记录失败时的处理"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        
        # 模拟日志记录失败
        handler.logger.error = Mock(side_effect=Exception("Logging failed"))
        
        # 应该不会抛出异常
        error = ValueError("Test error")
        handler.handle(error)
        
        # 错误应该仍然被记录
        assert "ValueError" in handler.error_stats
    
    def test_handle_error_recording_failure(self):
        """测试错误记录失败时的处理"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        
        # 模拟_create_error_info失败
        original_create = handler._create_error_info
        handler._create_error_info = Mock(side_effect=Exception("Recording failed"))
        
        # 应该不会抛出异常
        error = ValueError("Test error")
        handler.handle(error)
        
        # 恢复原始方法
        handler._create_error_info = original_create
    
    def test_error_info_structure(self):
        """测试错误信息结构"""
        from src.infrastructure.utils.core.error import UnifiedErrorHandler
        
        handler = UnifiedErrorHandler()
        error = ValueError("Test error message")
        
        handler.handle(error, context="test_context", level="warning")
        
        error_info = handler.last_errors[0]
        
        assert error_info["error_type"] == "ValueError"
        assert error_info["message"] == "Test error message"
        assert error_info["context"] == "test_context"
        assert error_info["level"] == "warning"
        assert "timestamp" in error_info
        assert isinstance(error_info["timestamp"], str)

