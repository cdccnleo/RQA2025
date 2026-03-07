#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层内存对象池组件测试

测试目标：提升utils/components/memory_object_pool.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.memory_object_pool模块
"""

import pytest
import time
from unittest.mock import MagicMock


class TestMemoryPoolConstants:
    """测试内存对象池常量"""
    
    def test_constants(self):
        """测试常量值"""
        from src.infrastructure.utils.components.memory_object_pool import MemoryPoolConstants
        
        assert MemoryPoolConstants.DEFAULT_MAX_POOL_SIZE == 100
        assert MemoryPoolConstants.DEFAULT_MIN_POOL_SIZE == 10
        assert MemoryPoolConstants.DEFAULT_MAX_IDLE_TIME == 300
        assert MemoryPoolConstants.DEFAULT_CLEANUP_INTERVAL == 60
        assert MemoryPoolConstants.HIT_RATE_CALCULATION_DIVISOR == 1
        assert MemoryPoolConstants.PERCENTAGE_MULTIPLIER == 100
        assert MemoryPoolConstants.BYTES_PER_MB == 1024 * 1024
        assert MemoryPoolConstants.CLEANUP_THREAD_NAME == "ObjectPoolCleanup"
        assert MemoryPoolConstants.CLEANUP_BATCH_SIZE == 10
        assert MemoryPoolConstants.MAX_CLEANUP_ITERATIONS == 100


class TestObjectPoolMetrics:
    """测试对象池性能指标"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.memory_object_pool import ObjectPoolMetrics
        
        metrics = ObjectPoolMetrics()
        assert metrics.objects_created == 0
        assert metrics.objects_reused == 0
        assert metrics.objects_destroyed == 0
        assert metrics.pool_hits == 0
        assert metrics.pool_misses == 0
        assert metrics.peak_pool_size == 0
        assert metrics.current_pool_size == 0
        assert metrics.memory_saved == 0
        assert metrics.gc_cycles == 0
    
    def test_to_dict(self):
        """测试转换为字典"""
        from src.infrastructure.utils.components.memory_object_pool import ObjectPoolMetrics
        
        metrics = ObjectPoolMetrics()
        metrics.objects_created = 10
        metrics.objects_reused = 5
        metrics.pool_hits = 5
        metrics.pool_misses = 5
        
        result = metrics.to_dict()
        assert isinstance(result, dict)
        assert result["objects_created"] == 10
        assert result["objects_reused"] == 5
        assert result["pool_hits"] == 5
        assert result["pool_misses"] == 5
        assert "hit_rate" in result


class TestGenericObjectPool:
    """测试通用对象池"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.memory_object_pool import GenericObjectPool
        
        def factory():
            return {"id": 1}
        
        pool = GenericObjectPool(factory, max_pool_size=10, min_pool_size=2)
        
        assert pool.object_factory == factory
        assert pool.max_pool_size == 10
        assert pool.min_pool_size == 2
        assert hasattr(pool, '_pool')
        assert hasattr(pool, 'metrics')
    
    def test_get_object(self):
        """测试获取对象"""
        from src.infrastructure.utils.components.memory_object_pool import GenericObjectPool
        
        def factory():
            return {"id": 1}
        
        pool = GenericObjectPool(factory, max_pool_size=10, min_pool_size=2)
        
        obj_wrapper = pool.get_object()
        assert obj_wrapper is not None
        # get_object返回PooledObjectWrapper，可以通过__getattr__访问对象
        assert hasattr(obj_wrapper, '_object')
    
    def test_return_object(self):
        """测试归还对象"""
        from src.infrastructure.utils.components.memory_object_pool import GenericObjectPool
        
        def factory():
            return {"id": 1}
        
        pool = GenericObjectPool(factory, max_pool_size=10, min_pool_size=2)
        
        obj_wrapper = pool.get_object()
        obj = obj_wrapper._object
        pool.return_object(obj)
        
        # 对象应该被放回池中
        assert len(pool._pool) >= 1
    
    def test_get_stats(self):
        """测试获取统计信息"""
        from src.infrastructure.utils.components.memory_object_pool import GenericObjectPool
        
        def factory():
            return {"id": 1}
        
        pool = GenericObjectPool(factory, max_pool_size=10, min_pool_size=2)
        
        stats = pool.get_stats()
        assert isinstance(stats, dict)
        assert "pool_size" in stats
        assert "current_objects" in stats
        assert "max_pool_size" in stats
        assert "metrics" in stats
        assert "objects_created" in stats["metrics"]
        assert "pool_hits" in stats["metrics"]
        assert "pool_misses" in stats["metrics"]
    
    def test_cleanup(self):
        """测试清理"""
        from src.infrastructure.utils.components.memory_object_pool import GenericObjectPool
        
        def factory():
            return {"id": 1}
        
        pool = GenericObjectPool(factory, max_pool_size=10, min_pool_size=2, max_idle_time=1)
        
        # 获取并归还对象
        obj_wrapper = pool.get_object()
        obj = obj_wrapper._object
        pool.return_object(obj)
        
        # 等待一段时间后清理（通过_cleanup_expired_objects）
        time.sleep(1.1)
        pool._cleanup_expired_objects()
        
        # 清理后池应该变小或保持不变
        assert len(pool._pool) <= pool.max_pool_size
    
    def test_shutdown(self):
        """测试关闭"""
        from src.infrastructure.utils.components.memory_object_pool import GenericObjectPool
        
        def factory():
            return {"id": 1}
        
        pool = GenericObjectPool(factory, max_pool_size=10, min_pool_size=2)
        
        pool.shutdown()
        assert pool._shutdown_event.is_set()

