#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一接口测试"""

import pytest
from unittest.mock import Mock, patch
from src.infrastructure.config.interfaces.unified_interface import CachePolicy


class TestCachePolicy:
    """测试缓存策略枚举"""

    def test_cache_policy_exists(self):
        """测试CachePolicy枚举存在"""
        assert CachePolicy is not None

    def test_cache_policy_has_values(self):
        """测试CachePolicy有值"""
        attrs = [attr for attr in dir(CachePolicy) if not attr.startswith('_')]
        assert len(attrs) > 0

    def test_cache_policy_values(self):
        """测试CachePolicy的具体值"""
        assert CachePolicy.LRU.value == "lru"
        assert CachePolicy.LFU.value == "lfu"
        assert CachePolicy.FIFO.value == "fifo"
        assert CachePolicy.RANDOM.value == "random"
