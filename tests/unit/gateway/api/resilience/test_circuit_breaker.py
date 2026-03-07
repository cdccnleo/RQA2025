#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
熔断器测试

测试目标：提升circuit_breaker.py的覆盖率
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, MagicMock
import time
from datetime import datetime, timedelta

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入熔断器模块
try:
    circuit_breaker_module = importlib.import_module('src.gateway.api.resilience.circuit_breaker')
    CircuitBreaker = getattr(circuit_breaker_module, 'CircuitBreaker', None)
    if CircuitBreaker is None:
        pytest.skip("CircuitBreaker不可用", allow_module_level=True)
except ImportError:
    pytest.skip("熔断器模块导入失败", allow_module_level=True)


class TestCircuitBreaker:
    """测试熔断器"""
    
    @pytest.fixture
    def circuit_breaker(self):
        """熔断器fixture"""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10,
            success_threshold=2
        )
    
    def test_circuit_breaker_init(self):
        """测试熔断器初始化"""
        cb = CircuitBreaker()
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 60
        assert cb.success_threshold == 3
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.state == "CLOSED"
    
    def test_circuit_breaker_init_custom(self):
        """测试熔断器初始化 - 自定义参数"""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10,
            success_threshold=2
        )
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 10
        assert cb.success_threshold == 2
    
    def test_call_success(self, circuit_breaker):
        """测试调用成功"""
        def success_func():
            return "success"
        
        result = circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.state == "CLOSED"
    
    def test_call_failure(self, circuit_breaker):
        """测试调用失败"""
        def failure_func():
            raise Exception("Test error")
        
        with pytest.raises(Exception):
            circuit_breaker.call(failure_func)
        
        assert circuit_breaker.failure_count == 1
    
    def test_call_multiple_failures_triggers_open(self, circuit_breaker):
        """测试多次失败触发熔断"""
        def failure_func():
            raise Exception("Test error")
        
        # 触发熔断
        for _ in range(3):
            try:
                circuit_breaker.call(failure_func)
            except Exception:
                pass
        
        assert circuit_breaker.state == "OPEN"
        assert circuit_breaker.failure_count >= 3
    
    def test_call_open_state_rejects(self, circuit_breaker):
        """测试OPEN状态拒绝请求"""
        circuit_breaker.state = "OPEN"
        circuit_breaker.last_failure_time = datetime.now() - timedelta(seconds=5)  # 5秒前，未超过recovery_timeout
        
        def func():
            return "should not execute"
        
        with pytest.raises(Exception) as exc_info:
            circuit_breaker.call(func)
        assert "Circuit breaker is OPEN" in str(exc_info.value)
    
    def test_call_half_open_success(self, circuit_breaker):
        """测试HALF_OPEN状态成功"""
        circuit_breaker.state = "HALF_OPEN"
        circuit_breaker.success_count = 1
        
        def success_func():
            return "success"
        
        result = circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.success_count == 2
    
    def test_call_half_open_failure(self, circuit_breaker):
        """测试HALF_OPEN状态失败"""
        circuit_breaker.state = "HALF_OPEN"
        circuit_breaker.failure_count = 2  # 接近阈值
        
        def failure_func():
            raise Exception("Test error")
        
        with pytest.raises(Exception):
            circuit_breaker.call(failure_func)
        
        # 失败后应该增加failure_count，如果达到阈值则变为OPEN
        assert circuit_breaker.failure_count >= 3
        if circuit_breaker.failure_count >= circuit_breaker.failure_threshold:
            assert circuit_breaker.state == "OPEN"
    
    def test_get_state(self, circuit_breaker):
        """测试获取状态"""
        state = circuit_breaker.get_state()
        assert state == "CLOSED"
    
    def test_record_failure(self, circuit_breaker):
        """测试记录失败"""
        circuit_breaker.record_failure()
        assert circuit_breaker.failure_count == 1
    
    def test_record_success(self, circuit_breaker):
        """测试记录成功"""
        circuit_breaker.state = "HALF_OPEN"
        circuit_breaker.record_success()
        assert circuit_breaker.success_count == 1
    
    def test_can_attempt(self, circuit_breaker):
        """测试是否可以尝试"""
        assert circuit_breaker.can_attempt() is True
        
        circuit_breaker.state = "OPEN"
        from datetime import datetime, timedelta
        circuit_breaker.last_failure_time = datetime.now() - timedelta(seconds=5)  # 未超过recovery_timeout
        assert circuit_breaker.can_attempt() is False
        
        circuit_breaker.last_failure_time = datetime.now() - timedelta(seconds=15)  # 超过recovery_timeout
        assert circuit_breaker.can_attempt() is True
    
    def test_get_stats(self, circuit_breaker):
        """测试获取统计信息"""
        stats = circuit_breaker.get_stats()
        assert isinstance(stats, dict)
        assert "state" in stats
        assert "failure_count" in stats
        assert "success_count" in stats

