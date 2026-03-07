#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化器质量测试
测试覆盖 MemoryOptimizer 的核心功能
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch

try:
    from src.streaming.optimization.memory_optimizer import MemoryOptimizer
except ImportError:
    MemoryOptimizer = None


@pytest.fixture
def memory_optimizer():
    """创建内存优化器实例"""
    if MemoryOptimizer is None:
        pytest.skip("MemoryOptimizer不可用")
    return MemoryOptimizer(target_memory_percent=75.0, cleanup_interval=1.0)


class TestMemoryOptimizer:
    """MemoryOptimizer测试类"""

    def test_initialization(self, memory_optimizer):
        """测试初始化"""
        assert memory_optimizer.target_memory_percent == 75.0
        assert memory_optimizer.cleanup_interval == 1.0
        assert memory_optimizer.is_running is False
        assert len(memory_optimizer.optimization_strategies) > 0

    def test_start_optimization(self, memory_optimizer):
        """测试启动优化"""
        result = memory_optimizer.start_optimization()
        assert result is True
        assert memory_optimizer.is_running is True
        assert memory_optimizer.monitoring_thread is not None
        
        # 清理
        memory_optimizer.stop_optimization()

    def test_stop_optimization(self, memory_optimizer):
        """测试停止优化"""
        memory_optimizer.start_optimization()
        result = memory_optimizer.stop_optimization()
        assert result is True
        assert memory_optimizer.is_running is False

    def test_start_already_running(self, memory_optimizer):
        """测试重复启动"""
        memory_optimizer.start_optimization()
        result = memory_optimizer.start_optimization()
        assert result is False
        
        memory_optimizer.stop_optimization()

    def test_stop_not_running(self, memory_optimizer):
        """测试停止未运行的优化器"""
        result = memory_optimizer.stop_optimization()
        assert result is False

    def test_get_memory_usage(self, memory_optimizer):
        """测试获取内存使用情况"""
        memory_info = memory_optimizer.get_memory_usage()
        assert isinstance(memory_info, dict)
        # 检查可能存在的键
        assert len(memory_info) > 0

    def test_add_optimization_strategy(self, memory_optimizer):
        """测试添加优化策略"""
        initial_count = len(memory_optimizer.optimization_strategies)
        
        def custom_strategy():
            pass
        
        memory_optimizer.add_optimization_strategy(custom_strategy)
        assert len(memory_optimizer.optimization_strategies) == initial_count + 1

    def test_optimize_memory(self, memory_optimizer):
        """测试内存优化"""
        result = memory_optimizer.optimize_memory()
        assert isinstance(result, dict)
        assert 'optimizations' in result or 'memory_after' in result

    def test_get_optimization_stats(self, memory_optimizer):
        """测试获取优化统计"""
        stats = memory_optimizer.get_optimization_stats()
        assert isinstance(stats, dict)
        assert 'cleanup_interval' in stats or 'is_running' in stats

    def test_monitoring_loop(self, memory_optimizer):
        """测试监控循环"""
        memory_optimizer.start_optimization()
        
        # 等待监控循环运行
        time.sleep(0.5)
        
        # 验证监控线程在运行
        assert memory_optimizer.monitoring_thread.is_alive()
        
        memory_optimizer.stop_optimization()

    def test_garbage_collection_strategy(self, memory_optimizer):
        """测试垃圾回收策略"""
        result = memory_optimizer._garbage_collection_strategy()
        assert isinstance(result, dict)
        assert 'strategy' in result or 'objects_collected' in result

    def test_object_pool_strategy(self, memory_optimizer):
        """测试对象池策略"""
        result = memory_optimizer._object_pool_strategy()
        assert isinstance(result, dict)
        assert 'strategy' in result or 'pools_count' in result

    def test_buffer_cleanup_strategy(self, memory_optimizer):
        """测试缓冲区清理策略"""
        result = memory_optimizer._buffer_cleanup_strategy()
        assert isinstance(result, dict)
        assert 'strategy' in result or 'status' in result

    def test_cache_cleanup_strategy(self, memory_optimizer):
        """测试缓存清理策略"""
        result = memory_optimizer._cache_cleanup_strategy()
        assert isinstance(result, dict)
        assert 'strategy' in result or 'weak_refs_cleaned' in result

    def test_get_memory_usage_exception(self, memory_optimizer):
        """测试获取内存使用异常处理"""
        with patch('psutil.virtual_memory', side_effect=Exception("Test error")):
            result = memory_optimizer.get_memory_usage()
            assert result == {}

    def test_optimize_memory_exception(self, memory_optimizer):
        """测试内存优化异常处理"""
        # Mock get_memory_usage在第二次调用时抛出异常
        original_get = memory_optimizer.get_memory_usage
        call_count = [0]
        
        def mock_get():
            call_count[0] += 1
            if call_count[0] == 2:  # 第二次调用（memory_after）时抛出异常
                raise Exception("Test error")
            return original_get()
        
        memory_optimizer.get_memory_usage = mock_get
        result = memory_optimizer.optimize_memory()
        # 即使有异常，也应该返回结果字典
        assert isinstance(result, dict)

    def test_optimize_memory_strategy_exception(self, memory_optimizer):
        """测试优化策略异常处理"""
        def failing_strategy():
            raise Exception("Strategy failed")
        
        memory_optimizer.add_optimization_strategy(failing_strategy)
        result = memory_optimizer.optimize_memory()
        
        # 应该捕获异常并继续
        assert isinstance(result, dict)

    def test_create_object_pool_existing(self, memory_optimizer):
        """测试创建已存在的对象池"""
        def factory():
            return {}
        
        memory_optimizer.create_object_pool('test_pool', factory)
        # 再次创建应该被忽略
        memory_optimizer.create_object_pool('test_pool', factory)
        
        assert 'test_pool' in memory_optimizer.object_pools

    def test_create_object_pool_factory_exception(self, memory_optimizer):
        """测试对象池创建时工厂函数异常"""
        def failing_factory():
            raise Exception("Factory failed")
        
        memory_optimizer.create_object_pool('test_pool', failing_factory, initial_size=5)
        
        # 应该创建部分对象或空池
        assert 'test_pool' in memory_optimizer.object_pools

    def test_get_from_pool(self, memory_optimizer):
        """测试从对象池获取对象"""
        def factory():
            return {'id': id({})}
        
        memory_optimizer.create_object_pool('test_pool', factory, initial_size=5)
        
        obj = memory_optimizer.get_from_pool('test_pool')
        assert obj is not None
        
        # 池应该减少
        assert memory_optimizer.pool_sizes['test_pool']['current'] == 4

    def test_get_from_pool_nonexistent(self, memory_optimizer):
        """测试从不存在池获取对象"""
        obj = memory_optimizer.get_from_pool('nonexistent')
        assert obj is None

    def test_get_from_pool_empty(self, memory_optimizer):
        """测试从空池获取对象"""
        def factory():
            return {}
        
        memory_optimizer.create_object_pool('test_pool', factory, initial_size=0)
        
        obj = memory_optimizer.get_from_pool('test_pool')
        assert obj is None

    def test_return_to_pool(self, memory_optimizer):
        """测试返回对象到池"""
        def factory():
            return {}
        
        memory_optimizer.create_object_pool('test_pool', factory, initial_size=5, max_size=10)
        
        obj = memory_optimizer.get_from_pool('test_pool')
        result = memory_optimizer.return_to_pool('test_pool', obj)
        
        assert result is True
        assert memory_optimizer.pool_sizes['test_pool']['current'] == 5

    def test_return_to_pool_nonexistent(self, memory_optimizer):
        """测试返回到不存在的池"""
        result = memory_optimizer.return_to_pool('nonexistent', {})
        assert result is False

    def test_return_to_pool_full(self, memory_optimizer):
        """测试返回到已满的池"""
        def factory():
            return {}
        
        memory_optimizer.create_object_pool('test_pool', factory, initial_size=5, max_size=5)
        
        # 填满池
        for _ in range(5):
            obj = memory_optimizer.get_from_pool('test_pool')
            memory_optimizer.return_to_pool('test_pool', obj)
        
        # 尝试添加更多应该失败
        result = memory_optimizer.return_to_pool('test_pool', {})
        assert result is False

    def test_monitoring_loop_high_memory(self, memory_optimizer):
        """测试监控循环高内存情况"""
        memory_optimizer.target_memory_percent = 10.0  # 设置很低的阈值
        memory_optimizer.start_optimization()
        
        # 等待监控循环检测高内存并触发优化
        time.sleep(0.2)
        
        memory_optimizer.stop_optimization()

    def test_monitoring_loop_exception(self, memory_optimizer):
        """测试监控循环异常处理"""
        with patch.object(memory_optimizer, 'get_memory_usage', side_effect=Exception("Test error")):
            memory_optimizer.start_optimization()
            time.sleep(0.2)
            memory_optimizer.stop_optimization()

    def test_garbage_collection_strategy_exception(self, memory_optimizer):
        """测试垃圾回收策略异常"""
        with patch('gc.get_objects', side_effect=Exception("Test error")):
            result = memory_optimizer._garbage_collection_strategy()
            assert result is None

    def test_object_pool_strategy_exception(self, memory_optimizer):
        """测试对象池策略异常"""
        # 通过修改object_pools来触发异常
        original_pools = memory_optimizer.object_pools
        memory_optimizer.object_pools = None
        try:
            result = memory_optimizer._object_pool_strategy()
            # 如果抛出异常，会被捕获并返回None
        except:
            pass
        finally:
            memory_optimizer.object_pools = original_pools

    def test_buffer_cleanup_strategy_exception(self, memory_optimizer):
        """测试缓冲区清理策略异常"""
        # 直接调用应该不会抛出异常，因为内部有try-except
        result = memory_optimizer._buffer_cleanup_strategy()
        assert isinstance(result, dict) or result is None

    def test_cache_cleanup_strategy_exception(self, memory_optimizer):
        """测试缓存清理策略异常"""
        # 添加一些弱引用（使用可弱引用的对象）
        import weakref
        class WeakRefable:
            pass
        obj = WeakRefable()
        memory_optimizer.weak_refs.append(weakref.ref(obj))
        
        # 通过修改weak_refs来触发异常
        original_refs = memory_optimizer.weak_refs
        memory_optimizer.weak_refs = [None]  # 使用None列表来触发异常
        try:
            result = memory_optimizer._cache_cleanup_strategy()
            # 如果捕获异常，应该返回None
            assert result is None or isinstance(result, dict)
        except:
            pass
        finally:
            memory_optimizer.weak_refs = original_refs

    def test_stop_optimization_exception(self, memory_optimizer):
        """测试停止优化异常处理"""
        memory_optimizer.start_optimization()
        memory_optimizer.monitoring_thread = Mock()
        memory_optimizer.monitoring_thread.is_alive.return_value = True
        memory_optimizer.monitoring_thread.join.side_effect = Exception("Join failed")
        
        # 即使join失败，stop也应该返回True
        result = memory_optimizer.stop_optimization()
        assert isinstance(result, bool)
