#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层异步I/O优化器组件测试

测试目标：提升utils/optimization/async_io_optimizer.py的真实覆盖率
实际导入和使用src.infrastructure.utils.optimization.async_io_optimizer模块
"""

import pytest
from unittest.mock import MagicMock


class TestAsyncIOConstants:
    """测试异步I/O常量"""
    
    def test_constants(self):
        """测试常量值"""
        from src.infrastructure.utils.optimization.async_io_optimizer import AsyncIOConstants
        
        assert AsyncIOConstants.DEFAULT_MAX_CONCURRENT_FILES == 10
        assert AsyncIOConstants.DEFAULT_MAX_CONCURRENT_REQUESTS == 20
        assert AsyncIOConstants.DEFAULT_MAX_WORKERS == 10
        assert AsyncIOConstants.DEFAULT_TIMEOUT == 30.0
        assert AsyncIOConstants.DEFAULT_RETRY_DELAY == 1.0
        assert AsyncIOConstants.DEFAULT_MAX_RETRIES == 3
        assert AsyncIOConstants.PERCENTAGE_MULTIPLIER == 100
        assert AsyncIOConstants.HTTP_SUCCESS_STATUS == 200
        assert AsyncIOConstants.TEST_ITERATIONS == 10


class TestAsyncIOMetrics:
    """测试异步I/O性能指标类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.optimization.async_io_optimizer import AsyncIOMetrics
        
        metrics = AsyncIOMetrics()
        assert metrics.total_operations == 0
        assert metrics.successful_operations == 0
        assert metrics.failed_operations == 0
        assert metrics.total_response_time == 0.0
        assert metrics.average_response_time == 0.0
        assert metrics.min_response_time == float("inf")
        assert metrics.max_response_time == 0.0
    
    def test_record_operation_success(self):
        """测试记录成功操作"""
        from src.infrastructure.utils.optimization.async_io_optimizer import AsyncIOMetrics
        
        metrics = AsyncIOMetrics()
        metrics.record_operation(0.5, success=True)
        
        assert metrics.total_operations == 1
        assert metrics.successful_operations == 1
        assert metrics.failed_operations == 0
        assert metrics.average_response_time == 0.5
        assert metrics.min_response_time == 0.5
        assert metrics.max_response_time == 0.5
    
    def test_record_operation_failure(self):
        """测试记录失败操作"""
        from src.infrastructure.utils.optimization.async_io_optimizer import AsyncIOMetrics
        
        metrics = AsyncIOMetrics()
        metrics.record_operation(0.3, success=False)
        
        assert metrics.total_operations == 1
        assert metrics.successful_operations == 0
        assert metrics.failed_operations == 1
    
    def test_record_multiple_operations(self):
        """测试记录多个操作"""
        from src.infrastructure.utils.optimization.async_io_optimizer import AsyncIOMetrics
        
        metrics = AsyncIOMetrics()
        metrics.record_operation(0.1, success=True)
        metrics.record_operation(0.2, success=True)
        metrics.record_operation(0.3, success=False)
        
        assert metrics.total_operations == 3
        assert metrics.successful_operations == 2
        assert metrics.failed_operations == 1
        assert metrics.min_response_time == 0.1
        assert metrics.max_response_time == 0.3
    
    def test_to_dict(self):
        """测试转换为字典"""
        from src.infrastructure.utils.optimization.async_io_optimizer import AsyncIOMetrics
        
        metrics = AsyncIOMetrics()
        metrics.record_operation(0.5, success=True)
        
        result = metrics.to_dict()
        assert isinstance(result, dict)
        assert result["total_operations"] == 1
        assert result["successful_operations"] == 1
        assert "success_rate" in result
        assert result["average_response_time"] == 0.5
    
    def test_to_dict_with_failures(self):
        """测试包含失败的转换为字典"""
        from src.infrastructure.utils.optimization.async_io_optimizer import AsyncIOMetrics
        
        metrics = AsyncIOMetrics()
        metrics.record_operation(0.5, success=True)
        metrics.record_operation(0.3, success=False)
        
        result = metrics.to_dict()
        assert result["total_operations"] == 2
        assert result["successful_operations"] == 1
        assert result["failed_operations"] == 1
        assert result["success_rate"] == 50.0

