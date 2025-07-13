#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CircuitBreaker测试用例
"""

import pytest
import sys
import os
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.infrastructure.error.circuit_breaker import CircuitBreaker, CircuitBreakerError

class TestCircuitBreaker:
    """测试CircuitBreaker类"""
    
    def test_import(self):
        """测试模块导入"""
        assert CircuitBreaker is not None
        assert CircuitBreakerError is not None
    
    def test_circuit_breaker_initialization(self):
        """测试熔断器初始化"""
        cb = CircuitBreaker()
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30.0
        assert cb.state == "closed"
        assert cb.failure_count == 0
    
    def test_circuit_breaker_custom_parameters(self):
        """测试自定义参数初始化"""
        cb = CircuitBreaker(
            failure_threshold=5, 
            recovery_timeout=60.0,
            expected_exceptions=(ValueError,)
        )
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 60.0
        assert ValueError in cb.expected_exceptions
    
    def test_is_closed(self):
        """测试关闭状态检查"""
        cb = CircuitBreaker()
        assert cb.is_closed() is True
        assert cb.is_open() is False
        assert cb.is_half_open() is False
    
    def test_record_failure(self):
        """测试记录失败"""
        cb = CircuitBreaker(failure_threshold=2)
        
        # 记录一次失败
        cb.record_failure()
        assert cb.failure_count == 1
        assert cb.state == "closed"
        
        # 记录第二次失败，触发熔断
        cb.record_failure()
        assert cb.failure_count == 2
        assert cb.state == "open"
        assert cb.is_open() is True
    
    def test_reset(self):
        """测试重置熔断器"""
        cb = CircuitBreaker(failure_threshold=1)
        
        # 触发熔断
        cb.record_failure()
        assert cb.state == "open"
        
        # 重置
        cb.reset()
        assert cb.state == "closed"
        assert cb.failure_count == 0
        assert cb.is_closed() is True
    
    def test_call_decorator(self):
        """测试装饰器功能"""
        cb = CircuitBreaker(failure_threshold=1)
        
        @cb
        def test_func():
            raise ValueError("test error")
        
        # 第一次调用应该抛出原始异常
        with pytest.raises(ValueError):
            test_func()
        
        # 第二次调用应该抛出熔断器异常
        with pytest.raises(CircuitBreakerError):
            test_func()
    
    def test_call_method(self):
        """测试call方法"""
        cb = CircuitBreaker(failure_threshold=1)
        
        def test_func():
            raise ValueError("test error")
        
        # 第一次调用应该抛出原始异常
        with pytest.raises(ValueError):
            cb.call(test_func)
        
        # 第二次调用应该抛出熔断器异常
        with pytest.raises(CircuitBreakerError):
            cb.call(test_func)
    
    def test_execute_method(self):
        """测试execute方法"""
        cb = CircuitBreaker(failure_threshold=1)
        
        def test_func():
            return "success"
        
        # 成功执行
        result = cb.execute(test_func)
        assert result == "success"
        
        def failing_func():
            raise ValueError("test error")
        
        # 第一次失败
        with pytest.raises(ValueError):
            cb.execute(failing_func)
        
        # 第二次应该触发熔断
        with pytest.raises(CircuitBreakerError):
            cb.execute(failing_func)
    
    def test_half_open_state(self):
        """测试半开状态"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        # 触发熔断
        cb.record_failure()
        assert cb.state == "open"
        
        # 等待恢复时间
        time.sleep(0.2)
        
        # 应该转换到半开状态
        def test_func():
            return "success"
        
        result = cb.call(test_func)
        assert result == "success"
        assert cb.state == "closed"
    
    def test_multiple_instances(self):
        """测试多个实例不会冲突"""
        cb1 = CircuitBreaker(failure_threshold=1)
        cb2 = CircuitBreaker(failure_threshold=1)
        
        assert cb1.state == "closed"
        assert cb2.state == "closed"
        
        # 一个实例熔断不影响另一个
        cb1.record_failure()
        assert cb1.state == "open"
        assert cb2.state == "closed"
