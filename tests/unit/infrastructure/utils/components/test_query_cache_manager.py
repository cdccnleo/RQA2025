#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层查询缓存管理器组件测试

测试目标：提升utils/components/query_cache_manager.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.query_cache_manager模块
"""

import pytest
import time
from unittest.mock import MagicMock


class TestQueryCacheManager:
    """测试查询缓存管理器"""
    
    def test_init_default(self):
        """测试默认初始化"""
        from src.infrastructure.utils.components.query_cache_manager import QueryCacheManager
        
        manager = QueryCacheManager()
        assert manager.cache_enabled is True
        assert manager.cache_ttl == 300
        assert len(manager.query_cache) == 0
        assert manager.cache_hits == 0
        assert manager.cache_misses == 0
    
    def test_init_with_config(self):
        """测试使用配置初始化"""
        from src.infrastructure.utils.components.query_cache_manager import QueryCacheManager
        
        config = {"enabled": False, "ttl": 600}
        manager = QueryCacheManager(config=config)
        assert manager.cache_enabled is False
        assert manager.cache_ttl == 600
    
    def test_init_with_params(self):
        """测试使用参数初始化"""
        from src.infrastructure.utils.components.query_cache_manager import QueryCacheManager
        
        manager = QueryCacheManager(cache_enabled=False, cache_ttl=600)
        assert manager.cache_enabled is False
        assert manager.cache_ttl == 600
    
    def test_set_and_get(self):
        """测试设置和获取缓存"""
        from src.infrastructure.utils.components.query_cache_manager import QueryCacheManager
        
        manager = QueryCacheManager()
        
        manager.set("test_key", "test_value")
        result = manager.get("test_key")
        
        assert result == "test_value"
        assert manager.cache_hits == 1
    
    def test_get_nonexistent_key(self):
        """测试获取不存在的键"""
        from src.infrastructure.utils.components.query_cache_manager import QueryCacheManager
        
        manager = QueryCacheManager()
        
        result = manager.get("nonexistent_key")
        assert result is None
        assert manager.cache_misses == 1
    
    def test_get_expired_cache(self):
        """测试获取过期缓存"""
        from src.infrastructure.utils.components.query_cache_manager import QueryCacheManager
        
        manager = QueryCacheManager(cache_ttl=1)
        
        manager.set("test_key", "test_value")
        time.sleep(1.1)
        
        result = manager.get("test_key")
        assert result is None
        assert manager.cache_misses == 1
    
    def test_clear(self):
        """测试清空缓存"""
        from src.infrastructure.utils.components.query_cache_manager import QueryCacheManager
        
        manager = QueryCacheManager()
        
        manager.set("key1", "value1")
        manager.set("key2", "value2")
        
        assert len(manager.query_cache) == 2
        
        manager.clear()
        assert len(manager.query_cache) == 0
    
    def test_get_cache_statistics(self):
        """测试获取缓存统计"""
        from src.infrastructure.utils.components.query_cache_manager import QueryCacheManager
        
        manager = QueryCacheManager()
        
        manager.set("key1", "value1")
        manager.get("key1")
        manager.get("nonexistent")
        
        stats = manager.get_cache_statistics()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["cache_size"] == 1
        assert "cache_hit_rate" in stats
        assert "cache_enabled" in stats
    
    def test_get_cache_hit_rate(self):
        """测试获取缓存命中率"""
        from src.infrastructure.utils.components.query_cache_manager import QueryCacheManager
        
        manager = QueryCacheManager()
        
        manager.set("key1", "value1")
        manager.get("key1")
        manager.get("key1")
        manager.get("nonexistent")
        
        hit_rate = manager.get_cache_hit_rate()
        assert hit_rate > 0
        assert hit_rate <= 1.0
    
    def test_cleanup_expired_cache(self):
        """测试清理过期缓存"""
        from src.infrastructure.utils.components.query_cache_manager import QueryCacheManager
        
        manager = QueryCacheManager(cache_ttl=1)
        
        manager.set("key1", "value1")
        manager.set("key2", "value2")
        
        time.sleep(1.1)
        
        cleaned = manager.cleanup_expired_cache()
        assert cleaned == 2
        assert len(manager.query_cache) == 0
    
    def test_clear_cache(self):
        """测试清空缓存（包括统计）"""
        from src.infrastructure.utils.components.query_cache_manager import QueryCacheManager
        
        manager = QueryCacheManager()
        
        manager.set("key1", "value1")
        manager.get("key1")
        
        manager.clear_cache()
        
        assert len(manager.query_cache) == 0
        assert manager.cache_hits == 0
        assert manager.cache_misses == 0
    
    def test_cache_disabled(self):
        """测试缓存禁用"""
        from src.infrastructure.utils.components.query_cache_manager import QueryCacheManager
        
        manager = QueryCacheManager(cache_enabled=False)
        
        manager.set("key", "value")
        result = manager.get("key")
        
        assert result is None
        assert len(manager.query_cache) == 0

