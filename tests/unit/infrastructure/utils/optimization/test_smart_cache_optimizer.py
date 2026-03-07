#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层智能缓存优化器组件测试

测试目标：提升utils/optimization/smart_cache_optimizer.py的真实覆盖率
实际导入和使用src.infrastructure.utils.optimization.smart_cache_optimizer模块
"""

import pytest
import time
from unittest.mock import MagicMock


class TestCacheConstants:
    """测试缓存常量"""
    
    def test_constants(self):
        """测试常量值"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import CacheConstants
        
        assert CacheConstants.DEFAULT_MAX_SIZE == 1000
        assert CacheConstants.DEFAULT_TTL == 300
        assert CacheConstants.MIN_HIT_RATE == 0.5
        assert CacheConstants.MAX_EVICTION_RATE == 0.3
        assert CacheConstants.BYTES_PER_KB == 1024
        assert CacheConstants.BYTES_PER_MB == 1024 * 1024


class TestCacheConfig:
    """测试缓存配置数据类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import CacheConfig
        
        config = CacheConfig()
        assert config.max_size == 1000
        assert config.ttl == 300
        assert config.eviction_policy == "lru"
        assert config.enabled is True
    
    def test_init_with_params(self):
        """测试使用参数初始化"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import CacheConfig
        
        config = CacheConfig(max_size=2000, ttl=600, eviction_policy="fifo", enabled=False)
        assert config.max_size == 2000
        assert config.ttl == 600
        assert config.eviction_policy == "fifo"
        assert config.enabled is False


class TestCacheMetrics:
    """测试缓存指标数据类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import CacheMetrics
        
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.sets == 0
        assert metrics.size == 0
    
    def test_hit_rate(self):
        """测试命中率计算"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import CacheMetrics
        
        metrics = CacheMetrics()
        assert metrics.hit_rate == 0.0
        
        metrics.hits = 5
        metrics.misses = 5
        assert metrics.hit_rate == 0.5
        
        metrics.hits = 10
        metrics.misses = 0
        assert metrics.hit_rate == 1.0
    
    def test_record_hit(self):
        """测试记录命中"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import CacheMetrics
        
        metrics = CacheMetrics()
        metrics.record_hit()
        
        assert metrics.hits == 1
    
    def test_record_miss(self):
        """测试记录未命中"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import CacheMetrics
        
        metrics = CacheMetrics()
        metrics.record_miss()
        
        assert metrics.misses == 1
    
    def test_record_set(self):
        """测试记录设置"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import CacheMetrics
        
        metrics = CacheMetrics()
        metrics.record_set()
        
        assert metrics.sets == 1
    
    def test_record_eviction(self):
        """测试记录驱逐"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import CacheMetrics
        
        metrics = CacheMetrics()
        metrics.record_eviction()
        
        assert metrics.evictions == 1


class TestCacheEntry:
    """测试缓存条目数据类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import CacheEntry
        
        entry = CacheEntry(key="test_key", value="test_value")
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.timestamp == 0.0
        assert entry.ttl == 300
        assert entry.access_count == 0


class TestSmartCache:
    """测试智能缓存类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import SmartCache
        
        cache = SmartCache()
        assert cache.max_size == 1000
        assert len(cache._cache) == 0
        assert cache._metrics.hits == 0
    
    def test_init_with_max_size(self):
        """测试使用最大大小初始化"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import SmartCache
        
        cache = SmartCache(max_size=500)
        assert cache.max_size == 500
    
    def test_get_miss(self):
        """测试获取未命中"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import SmartCache
        
        cache = SmartCache()
        result = cache.get("nonexistent")
        
        assert result is None
        assert cache._metrics.misses == 1
    
    def test_set_and_get(self):
        """测试设置和获取"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import SmartCache
        
        cache = SmartCache()
        cache.set("test_key", "test_value")
        
        result = cache.get("test_key")
        assert result == "test_value"
        assert cache._metrics.hits == 1
    
    def test_set_with_ttl(self):
        """测试使用TTL设置"""
        from src.infrastructure.utils.optimization.smart_cache_optimizer import SmartCache
        
        cache = SmartCache()
        cache.set("test_key", "test_value", ttl=60)
        
        entry = cache._cache["test_key"]
        assert entry.ttl == 60

