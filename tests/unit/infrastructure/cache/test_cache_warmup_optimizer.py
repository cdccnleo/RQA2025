#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""缓存预热优化器测试"""

import pytest
from unittest.mock import Mock, patch
from src.infrastructure.cache.cache_warmup_optimizer import (
    WarmupStrategy,
    FailoverMode
)


class TestWarmupStrategy:
    """测试预热策略枚举"""

    def test_warmup_strategy_exists(self):
        """测试WarmupStrategy枚举存在"""
        assert WarmupStrategy is not None

    def test_warmup_strategy_has_values(self):
        """测试WarmupStrategy有值"""
        attrs = [attr for attr in dir(WarmupStrategy) if not attr.startswith('_')]
        assert len(attrs) > 0


class TestFailoverMode:
    """测试故障转移模式枚举"""

    def test_failover_mode_exists(self):
        """测试FailoverMode枚举存在"""
        assert FailoverMode is not None

    def test_failover_mode_has_values(self):
        """测试FailoverMode有值"""
        attrs = [attr for attr in dir(FailoverMode) if not attr.startswith('_')]
        assert len(attrs) > 0
