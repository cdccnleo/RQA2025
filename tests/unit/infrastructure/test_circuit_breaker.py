#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CircuitBreaker测试用例
"""

import pytest
import sys
import os
from prometheus_client import CollectorRegistry

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.infrastructure.circuit_breaker import CircuitBreaker, CircuitState

class TestCircuitBreaker:
    """测试CircuitBreaker类"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        # 为每个测试创建独立的registry
        self.registry = CollectorRegistry()
    
    def test_import(self):
        """测试模块导入"""
        assert CircuitBreaker is not None
        assert CircuitState is not None
    
    def test_circuit_breaker_initialization(self):
        """测试熔断器初始化"""
        cb = CircuitBreaker("test_service", registry=self.registry)
        assert cb.name == "test_service"
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 60
        assert cb.state == CircuitState.CLOSED
        assert cb.get_failure_count() == 0
    
    def test_circuit_breaker_custom_parameters(self):
        """测试自定义参数初始化"""
        cb = CircuitBreaker(
            "custom_service", 
            failure_threshold=3, 
            recovery_timeout=30,
            registry=self.registry
        )
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30
    
    def test_can_execute_when_closed(self):
        """测试关闭状态时可以执行"""
        cb = CircuitBreaker("test_service", registry=self.registry)
        assert cb.can_execute() is True
    
    def test_record_failure(self):
        """测试记录失败"""
        cb = CircuitBreaker("test_service", failure_threshold=2, registry=self.registry)
        
        # 记录一次失败
        cb.record_failure()
        assert cb.get_failure_count() == 1
        assert cb.state == CircuitState.CLOSED
        
        # 记录第二次失败，触发熔断
        cb.record_failure()
        assert cb.get_failure_count() == 2
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False
    
    def test_trip_manual(self):
        """测试手动触发熔断"""
        cb = CircuitBreaker("test_service", registry=self.registry)
        assert cb.state == CircuitState.CLOSED
        
        cb.trip("manual_test")
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False
    
    def test_reset(self):
        """测试重置熔断器"""
        cb = CircuitBreaker("test_service", failure_threshold=1, registry=self.registry)
        
        # 触发熔断
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        
        # 重置
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.get_failure_count() == 0
        assert cb.can_execute() is True
    
    def test_state_transitions(self):
        """测试状态转换"""
        cb = CircuitBreaker("test_service", registry=self.registry)
        
        # 初始状态
        assert cb.get_state_name() == "CLOSED"
        
        # 触发熔断
        cb.trip("test")
        assert cb.get_state_name() == "OPEN"
        
        # 重置
        cb.reset()
        assert cb.get_state_name() == "CLOSED"
    
    def test_multiple_instances(self):
        """测试多个实例不会冲突"""
        cb1 = CircuitBreaker("service1", registry=self.registry)
        cb2 = CircuitBreaker("service2", registry=self.registry)
        
        assert cb1.name == "service1"
        assert cb2.name == "service2"
        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED
        
        # 一个实例熔断不影响另一个
        cb1.trip("test")
        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.CLOSED
