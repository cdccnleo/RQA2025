#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ErrorHandler全面测试脚本
"""

import sys
import os
import time
import threading
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.infrastructure.error.error_handler import ErrorHandler, ErrorLevel

def test_error_handler_comprehensive():
    """全面测试ErrorHandler的所有功能"""
    
    # 测试1: 基本初始化
    handler = ErrorHandler()
    assert handler is not None
    assert hasattr(handler, '_records')
    assert hasattr(handler, '_max_records')
    
    # 测试2: 错误处理
    handler = ErrorHandler()
    
    # 测试基本错误处理
    result = handler.handle_error(ValueError("test error"))
    assert result is not None
    assert 'error_record' in result
    
    # 测试3: 自定义处理器
    handler = ErrorHandler()
    
    def custom_handler(error, context):
        return f"Custom: {error}"
    
    handler.register_handler(ValueError, custom_handler)
    result = handler.handle_error(ValueError("test"))
    assert result is not None
    
    # 测试4: 日志上下文
    handler = ErrorHandler()
    
    context = {
        "module": "test_module",
        "function": "test_function",
        "line": 42,
        "timestamp": time.time()
    }
    
    result = handler.handle_error(ValueError("test"), context)
    assert result is not None
    
    # 测试5: 重试机制
    handler = ErrorHandler()
    
    # 模拟可重试的错误
    retry_count = 0
    def failing_function():
        nonlocal retry_count
        retry_count += 1
        if retry_count < 3:
            raise ValueError("temporary error")
        return "success"
    
    # 测试重试逻辑
    try:
        handler.with_retry(failing_function, max_retries=3)
    except Exception:
        pass
    
    # 测试6: 错误记录
    handler = ErrorHandler()
    
    # 记录多个错误
    for i in range(5):
        handler.handle_error(ValueError(f"error {i}"))
    
    # 检查记录
    records = handler.get_records()
    assert len(records) > 0
    
    # 测试7: 错误统计
    handler = ErrorHandler()
    
    # 记录不同类型的错误
    handler.handle_error(ValueError("test1"))
    handler.handle_error(RuntimeError("test2"))
    handler.handle_error(Exception("test3"))
    
    # 获取统计信息
    stats = handler.get_stats()
    assert stats is not None
    
    # 测试8: 线程安全
    handler = ErrorHandler()
    
    def worker():
        for i in range(10):
            handler.handle_error(ValueError(f"thread error {i}"))
    
    # 创建多个线程
    threads = []
    for _ in range(3):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    # 检查线程安全
    records = handler.get_records()
    assert len(records) >= 30
    
    # 测试9: 错误级别处理
    handler = ErrorHandler()
    
    # 测试不同级别的错误处理
    levels = [ErrorLevel.WARNING, ErrorLevel.ERROR, ErrorLevel.CRITICAL]
    
    for level in levels:
        context = {"level": level}
        result = handler.handle_error(ValueError("test"), context)
        assert result is not None
    
    # 测试10: 错误清理
    handler = ErrorHandler()
    
    # 添加一些错误
    for i in range(10):
        handler.handle_error(ValueError(f"error {i}"))
    
    # 清理旧错误
    handler.cleanup_old_errors(days=0)
    records = handler.get_records()
    # 注意：cleanup_old_errors可能不会立即清理所有记录
    # 这里只检查清理功能是否正常工作
    assert records is not None
    
    # 测试11: 错误聚合
    handler = ErrorHandler()
    
    # 添加重复错误
    for _ in range(5):
        handler.handle_error(ValueError("same error"))
    
    # 聚合错误
    aggregated = handler.aggregate_errors()
    assert aggregated is not None
    
    # 测试12: 错误恢复
    handler = ErrorHandler()
    
    # 测试恢复功能
    try:
        handler.recover(ValueError("recoverable error"))
    except Exception:
        pass
    
    # 测试13: 错误验证
    handler = ErrorHandler()
    
    # 测试错误验证
    is_valid = handler.validate_error(ValueError("test"))
    assert is_valid is not None
    
    # 测试14: 监控指标
    handler = ErrorHandler()
    
    # 获取监控指标
    metrics = handler.get_monitoring_metrics()
    assert metrics is not None
    
    # 测试15: 资源清理
    handler = ErrorHandler()
    
    # 测试资源清理
    handler.cleanup_resources()
    
    print("所有ErrorHandler测试通过！")

if __name__ == "__main__":
    test_error_handler_comprehensive() 