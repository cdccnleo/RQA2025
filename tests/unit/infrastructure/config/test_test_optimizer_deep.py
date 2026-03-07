#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TestOptimizer深度测试"""

import pytest


def test_test_mode_enum():
    """测试TestMode枚举"""
    from src.infrastructure.config.tools.test_optimizer import TestMode
    
    assert TestMode.UNIT.value == "unit"
    assert TestMode.INTEGRATION.value == "integration"
    assert TestMode.PERFORMANCE.value == "performance"
    assert TestMode.STRESS.value == "stress"


def test_test_optimization_config():
    """测试TestOptimizationConfig"""
    from src.infrastructure.config.tools.test_optimizer import TestOptimizationConfig
    
    config = TestOptimizationConfig()
    assert config.thread_pool_size == 4
    assert config.enable_caching is True
    assert config.async_execution is True


def test_thread_manager():
    """测试ThreadManager"""
    from src.infrastructure.config.tools.test_optimizer import ThreadManager
    
    manager = ThreadManager()
    assert manager.active_threads == 0
    assert manager.get_active_threads_count() == 0


def test_test_optimizer_init():
    """测试TestOptimizer初始化"""
    from src.infrastructure.config.tools.test_optimizer import TestOptimizer
    
    optimizer = TestOptimizer()
    assert optimizer is not None
    assert optimizer.thread_manager is not None
    assert optimizer.optimization_config is not None
    assert optimizer._optimizations_applied is False


def test_test_optimizer_apply():
    """测试应用优化"""
    from src.infrastructure.config.tools.test_optimizer import TestOptimizer
    
    optimizer = TestOptimizer()
    assert optimizer._optimizations_applied is False
    
    optimizer.apply_optimizations()
    assert optimizer._optimizations_applied is True


def test_test_optimizer_restore():
    """测试恢复优化"""
    from src.infrastructure.config.tools.test_optimizer import TestOptimizer
    
    optimizer = TestOptimizer()
    optimizer.apply_optimizations()
    assert optimizer._optimizations_applied is True
    
    optimizer.restore_optimizations()
    assert optimizer._optimizations_applied is False


def test_test_optimizer_get_status():
    """测试获取优化状态"""
    from src.infrastructure.config.tools.test_optimizer import TestOptimizer
    
    optimizer = TestOptimizer()
    status = optimizer.get_optimization_status()
    
    assert isinstance(status, dict)
    assert 'optimizations_applied' in status
    assert 'thread_pool_size' in status
    assert 'caching_enabled' in status
    assert 'async_execution' in status
    
    assert status['optimizations_applied'] is False
    assert status['thread_pool_size'] == 4
    assert status['caching_enabled'] is True
    assert status['async_execution'] is True


def test_test_optimizer_status_after_apply():
    """测试应用优化后的状态"""
    from src.infrastructure.config.tools.test_optimizer import TestOptimizer
    
    optimizer = TestOptimizer()
    optimizer.apply_optimizations()
    
    status = optimizer.get_optimization_status()
    assert status['optimizations_applied'] is True


def test_get_test_optimizer_singleton():
    """测试get_test_optimizer全局实例"""
    from src.infrastructure.config.tools.test_optimizer import get_test_optimizer
    
    optimizer1 = get_test_optimizer()
    optimizer2 = get_test_optimizer()
    
    assert optimizer1 is not None
    assert optimizer2 is not None
    assert optimizer1 is optimizer2  # 必须是同一个实例


def test_get_test_optimizer_functionality():
    """测试全局实例的功能"""
    from src.infrastructure.config.tools.test_optimizer import get_test_optimizer
    
    optimizer = get_test_optimizer()
    
    # 测试功能
    initial_status = optimizer._optimizations_applied
    optimizer.apply_optimizations()
    assert optimizer._optimizations_applied is True
    
    # 恢复状态
    if not initial_status:
        optimizer.restore_optimizations()


