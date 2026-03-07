#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLearningPredictor缓存管理器测试
补充ModelCacheManager的所有方法测试
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, MagicMock

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    dl_predictor_core_module = importlib.import_module('src.monitoring.ai.dl_predictor_core')
    ModelCacheManager = getattr(dl_predictor_core_module, 'ModelCacheManager', None)
    if ModelCacheManager is None:
        pytest.skip("ModelCacheManager不可用", allow_module_level=True)
except ImportError:
    pytest.skip("深度学习预测器核心模块导入失败", allow_module_level=True)


class TestModelCacheManager:
    """测试ModelCacheManager"""

    def test_init_default(self):
        """测试默认初始化"""
        cache = ModelCacheManager()
        assert cache.max_cache_size == 10
        assert len(cache.cache) == 0
        assert len(cache.access_count) == 0

    def test_init_with_max_size(self):
        """测试指定最大缓存大小"""
        cache = ModelCacheManager(max_cache_size=5)
        assert cache.max_cache_size == 5

    def test_get_nonexistent_key(self):
        """测试获取不存在的键"""
        cache = ModelCacheManager()
        result = cache.get("nonexistent")
        assert result is None
        assert len(cache.access_count) == 0

    def test_get_existing_key(self):
        """测试获取存在的键"""
        cache = ModelCacheManager()
        test_model = {"model": "test"}
        
        cache.set("key1", test_model)
        result = cache.get("key1")
        
        assert result == test_model
        assert cache.access_count["key1"] == 1

    def test_get_increments_access_count(self):
        """测试获取时访问计数增加"""
        cache = ModelCacheManager()
        cache.set("key1", "model1")
        
        cache.get("key1")
        assert cache.access_count["key1"] == 1
        
        cache.get("key1")
        assert cache.access_count["key1"] == 2

    def test_set_basic(self):
        """测试基本设置"""
        cache = ModelCacheManager()
        test_model = {"model": "test"}
        
        cache.set("key1", test_model)
        
        assert "key1" in cache.cache
        assert cache.cache["key1"] == test_model
        assert cache.access_count["key1"] == 0

    def test_set_multiple_keys(self):
        """测试设置多个键"""
        cache = ModelCacheManager()
        
        cache.set("key1", "model1")
        cache.set("key2", "model2")
        cache.set("key3", "model3")
        
        assert len(cache.cache) == 3
        assert "key1" in cache.cache
        assert "key2" in cache.cache
        assert "key3" in cache.cache

    def test_set_cache_full_lru_eviction(self):
        """测试缓存满时的LRU淘汰"""
        cache = ModelCacheManager(max_cache_size=2)
        
        # 添加2个模型
        cache.set("key1", "model1")
        cache.set("key2", "model2")
        
        # 访问key1使其访问计数增加
        cache.get("key1")
        cache.get("key1")
        # key2访问1次
        cache.get("key2")
        
        # key1访问计数=2, key2访问计数=1
        # 添加第3个时应该淘汰key2（访问次数最少）
        cache.set("key3", "model3")
        
        assert len(cache.cache) == 2
        assert "key1" in cache.cache  # 访问次数多，保留
        assert "key2" not in cache.cache  # 访问次数少，被淘汰
        assert "key3" in cache.cache

    def test_set_cache_full_same_access_count(self):
        """测试缓存满时访问次数相同的情况"""
        cache = ModelCacheManager(max_cache_size=2)
        
        cache.set("key1", "model1")
        cache.set("key2", "model2")
        
        # 都访问一次
        cache.get("key1")
        cache.get("key2")
        
        # 添加第3个，应该淘汰第一个（min会选择第一个）
        cache.set("key3", "model3")
        
        assert len(cache.cache) == 2
        assert "key3" in cache.cache

    def test_set_resets_access_count(self):
        """测试设置时重置访问计数"""
        cache = ModelCacheManager()
        
        cache.set("key1", "model1")
        cache.get("key1")
        cache.get("key1")
        assert cache.access_count["key1"] == 2
        
        # 重新设置应该重置访问计数
        cache.set("key1", "model1_updated")
        assert cache.access_count["key1"] == 0

    def test_clear(self):
        """测试清空缓存"""
        cache = ModelCacheManager()
        
        cache.set("key1", "model1")
        cache.set("key2", "model2")
        cache.get("key1")  # 增加访问计数
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert len(cache.access_count) == 0

    def test_clear_empty_cache(self):
        """测试清空空缓存"""
        cache = ModelCacheManager()
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert len(cache.access_count) == 0

