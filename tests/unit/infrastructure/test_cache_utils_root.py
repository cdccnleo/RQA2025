#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层根目录缓存工具组件测试

测试目标：提升cache_utils.py的真实覆盖率
实际导入和使用src.infrastructure.cache_utils模块
"""

import pytest


class TestCacheUtils:
    """测试缓存工具类"""
    
    def test_generate_cache_key(self):
        """测试生成缓存键"""
        from src.infrastructure.cache_utils import CacheUtils
        
        key = CacheUtils.generate_cache_key("prefix", "suffix", 123)
        
        assert isinstance(key, str)
        assert "prefix" in key
        assert "suffix" in key
        assert "123" in key
    
    def test_generate_cache_key_single_arg(self):
        """测试单个参数生成缓存键"""
        from src.infrastructure.cache_utils import CacheUtils
        
        key = CacheUtils.generate_cache_key("single")
        
        assert key == "single"
    
    def test_generate_cache_key_empty(self):
        """测试空参数生成缓存键"""
        from src.infrastructure.cache_utils import CacheUtils
        
        key = CacheUtils.generate_cache_key()
        
        assert key == ""
    
    def test_is_cacheable(self):
        """测试检查值是否可以缓存"""
        from src.infrastructure.cache_utils import CacheUtils
        
        assert CacheUtils.is_cacheable("string") is True
        assert CacheUtils.is_cacheable(123) is True
        assert CacheUtils.is_cacheable([1, 2, 3]) is True
        assert CacheUtils.is_cacheable({"key": "value"}) is True
        assert CacheUtils.is_cacheable(None) is True
    
    def test_calculate_hash(self):
        """测试计算数据哈希"""
        from src.infrastructure.cache_utils import CacheUtils
        
        hash1 = CacheUtils.calculate_hash("test_data")
        hash2 = CacheUtils.calculate_hash("test_data")
        
        assert isinstance(hash1, str)
        assert hash1 == hash2
    
    def test_calculate_hash_different_data(self):
        """测试不同数据计算不同哈希"""
        from src.infrastructure.cache_utils import CacheUtils
        
        hash1 = CacheUtils.calculate_hash("data1")
        hash2 = CacheUtils.calculate_hash("data2")
        
        assert hash1 != hash2
    
    def test_calculate_hash_complex_data(self):
        """测试复杂数据计算哈希"""
        from src.infrastructure.cache_utils import CacheUtils
        
        complex_data = {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}}
        hash_value = CacheUtils.calculate_hash(complex_data)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) > 0

