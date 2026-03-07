#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存优化器测试

测试缓存系统的优化功能，包括大小优化、淘汰策略建议、性能监控等。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import patch, MagicMock
from src.infrastructure.cache.core.cache_optimizer import (
    CacheOptimizer, CachePolicy, handle_cache_exception,
    CACHE_HIT_RATE_WARNING, CACHE_HIT_RATE_CRITICAL,
    DEFAULT_CACHE_SIZE, MAX_CACHE_SIZE, MIN_CACHE_SIZE
)


class TestCacheOptimizer:
    """缓存优化器测试类"""

    def setup_method(self):
        """测试前准备"""
        self.optimizer = CacheOptimizer()

    def test_initialization(self):
        """测试初始化"""
        assert self.optimizer._optimization_history == []
        assert self.optimizer._recommendation_cache == {}

    def test_optimize_cache_size_normal_conditions(self):
        """测试缓存大小优化 - 正常情况"""
        # 测试正常情况：保持当前大小
        result = self.optimizer.optimize_cache_size(1000, 0.8, 0.7)
        assert result == 1000

    def test_optimize_cache_size_low_hit_rate_high_memory(self):
        """测试缓存大小优化 - 命中率低且内存使用率高"""
        result = self.optimizer.optimize_cache_size(1000, 0.3, 0.9)
        assert result == 500  # 减半

    def test_optimize_cache_size_high_hit_rate_low_memory(self):
        """测试缓存大小优化 - 命中率高且内存使用率低"""
        result = self.optimizer.optimize_cache_size(1000, 0.9, 0.4)
        assert result == 2000  # 加倍

    def test_optimize_cache_size_extreme_cases(self):
        """测试缓存大小优化 - 极端情况"""
        # 命中率0%，内存使用率100%
        result = self.optimizer.optimize_cache_size(1000, 0.0, 1.0)
        assert result == 100

        # 命中率100%，内存使用率0%
        result = self.optimizer.optimize_cache_size(1000, 1.0, 0.0)
        assert result == 10000

    def test_optimize_cache_size_minimum_size(self):
        """测试缓存大小优化 - 最小大小限制"""
        # 确保不会低于最小值
        result = self.optimizer.optimize_cache_size(50, 0.3, 0.9)
        assert result == 100  # 最小值

    def test_optimize_cache_size_maximum_size(self):
        """测试缓存大小优化 - 最大大小限制"""
        # 确保不会超过最大值
        result = self.optimizer.optimize_cache_size(5000, 0.9, 0.4)
        assert result == 10000  # 最大值

    def test_suggest_eviction_policy_empty_pattern(self):
        """测试淘汰策略建议 - 空访问模式"""
        result = self.optimizer.suggest_eviction_policy({})
        assert result == CachePolicy.LRU

    def test_suggest_eviction_policy_zero_accesses(self):
        """测试淘汰策略建议 - 零访问次数"""
        result = self.optimizer.suggest_eviction_policy({"key1": 0, "key2": 0})
        assert result == CachePolicy.LRU

    def test_suggest_eviction_policy_high_frequency_variance(self):
        """测试淘汰策略建议 - 访问频率差异大"""
        # 模拟高频率差异的情况
        access_pattern = {"key1": 100, "key2": 1, "key3": 2}
        result = self.optimizer.suggest_eviction_policy(access_pattern)
        assert result == CachePolicy.LFU

    def test_suggest_eviction_policy_low_frequency_variance(self):
        """测试淘汰策略建议 - 访问频率差异小"""
        # 模拟低频率差异的情况
        access_pattern = {"key1": 10, "key2": 8, "key3": 12}
        result = self.optimizer.suggest_eviction_policy(access_pattern)
        assert result == CachePolicy.LRU

    def test_get_cache_recommendations_basic(self):
        """测试获取缓存建议 - 基础功能"""
        cache_stats = {
            "hit_rate": 0.8,
            "memory_usage": 0.6,
            "size": 1000,
            "eviction_count": 10
        }
        result = self.optimizer.get_cache_recommendations(cache_stats)

        assert isinstance(result, dict)
        assert "size_optimization" in result
        assert "policy_recommendation" in result
        assert "performance_improvements" in result
        assert "warnings" in result

    def test_get_cache_recommendations_low_performance(self):
        """测试获取缓存建议 - 性能较差"""
        cache_stats = {
            "hit_rate": 0.2,  # 低于0.3的阈值
            "memory_usage": 0.95,  # 高于0.9的阈值
            "size": 1000,
            "eviction_count": 100
        }
        result = self.optimizer.get_cache_recommendations(cache_stats)

        # 性能较差时应该有警告
        assert len(result["warnings"]) > 0

    def test_get_cache_recommendations_high_performance(self):
        """测试获取缓存建议 - 性能良好"""
        cache_stats = {
            "hit_rate": 0.95,
            "memory_usage": 0.3,
            "size": 1000,
            "eviction_count": 5
        }
        result = self.optimizer.get_cache_recommendations(cache_stats)

        # 性能良好时应该有改进建议
        assert len(result["performance_improvements"]) > 0

    def test_get_optimization_history_empty(self):
        """测试获取优化历史 - 空历史"""
        result = self.optimizer.get_optimization_history()
        assert result == []

    def test_get_optimization_history_with_data(self):
        """测试获取优化历史 - 有数据"""
        # 执行一次优化
        self.optimizer.optimize_cache_size(1000, 0.8, 0.7)

        result = self.optimizer.get_optimization_history()
        assert len(result) == 1
        assert "current_size" in result[0]
        assert "target_size" in result[0]
        assert "hit_rate" in result[0]
        assert "timestamp" in result[0]

    def test_clear_optimization_history(self):
        """测试清除优化历史"""
        # 先添加一些历史
        self.optimizer.optimize_cache_size(1000, 0.8, 0.7)

        # 清除历史
        self.optimizer.clear_optimization_history()

        # 验证已清除
        result = self.optimizer.get_optimization_history()
        assert result == []

    def test_analyze_access_patterns_empty(self):
        """测试访问模式分析 - 空模式"""
        result = self.optimizer.analyze_access_patterns({})
        assert isinstance(result, dict)
        assert "total_accesses" in result
        assert result["total_accesses"] == 0

    def test_analyze_access_patterns_with_data(self):
        """测试访问模式分析 - 有数据"""
        access_pattern = {"key1": 10, "key2": 20, "key3": 5}
        result = self.optimizer.analyze_access_patterns(access_pattern)

        assert result["total_accesses"] == 35
        assert result["unique_keys"] == 3
        assert result["avg_access_per_key"] == 35 / 3

    def test_get_optimization_recommendations_empty_history(self):
        """测试获取优化建议 - 空历史"""
        result = self.optimizer.get_optimization_recommendations()
        assert isinstance(result, dict)
        assert "recommendations" in result

    def test_get_optimization_recommendations_with_history(self):
        """测试获取优化建议 - 有历史"""
        # 添加一些优化历史
        for i in range(5):
            self.optimizer.optimize_cache_size(1000 + i * 100, 0.8 - i * 0.1, 0.7)

        result = self.optimizer.get_optimization_recommendations()
        assert isinstance(result, dict)
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0

    def test_get_performance_metrics(self):
        """测试获取性能指标"""
        result = self.optimizer.get_performance_metrics()
        assert isinstance(result, dict)
        assert "total_optimizations" in result
        assert "avg_hit_rate" in result
        assert "avg_memory_usage" in result

    def test_monitor_cache_performance(self):
        """测试监控缓存性能"""
        performance_metrics = {
            "hit_rate": 0.85,
            "memory_usage": 0.75,
            "response_time": 0.05,
            "throughput": 1000
        }

        # 不应该抛出异常 - 这个方法只是记录日志，不修改历史
        self.optimizer.monitor_cache_performance(performance_metrics)

        # 验证没有抛出异常即可
        assert True

    def test_optimize_eviction_policy(self):
        """测试优化淘汰策略"""
        cache_stats = {
            "hit_rate": 0.8,
            "memory_usage": 0.6,
            "access_pattern": {"key1": 10, "key2": 5},
            "eviction_count": 10,
            "total_requests": 100
        }

        result = self.optimizer.optimize_eviction_policy(cache_stats)
        assert isinstance(result, str)
        assert result in ["LRU", "LFU", "FIFO", "RANDOM"]

    def test_reset_optimization_history(self):
        """测试重置优化历史"""
        # 先添加一些历史
        self.optimizer.optimize_cache_size(1000, 0.8, 0.7)

        # 重置
        self.optimizer.reset_optimization_history()

        # 验证已重置
        history = self.optimizer.get_optimization_history()
        assert history == []
        assert self.optimizer._recommendation_cache == {}

    def test_handle_cache_exception_decorator_success(self):
        """测试缓存异常处理装饰器 - 成功情况"""
        @handle_cache_exception("test_operation")
        def test_function():
            return {"success": True}

        result = test_function()
        assert result == {"success": True}

    def test_handle_cache_exception_decorator_failure(self):
        """测试缓存异常处理装饰器 - 异常情况"""
        @handle_cache_exception("test_operation")
        def test_function():
            raise Exception("Test exception")

        with patch('src.infrastructure.cache.core.cache_optimizer.logger') as mock_logger:
            result = test_function()
            assert result == {}
            mock_logger.error.assert_called_once()

    def test_constants_values(self):
        """测试常量值"""
        assert CACHE_HIT_RATE_WARNING == 0.7
        assert CACHE_HIT_RATE_CRITICAL == 0.5
        assert DEFAULT_CACHE_SIZE == 1000
        assert MAX_CACHE_SIZE == 10000
        assert MIN_CACHE_SIZE == 100

    def test_cache_policy_enum_values(self):
        """测试缓存策略枚举值"""
        assert CachePolicy.LRU.value == "lru"
        assert CachePolicy.LFU.value == "lfu"
        assert CachePolicy.FIFO.value == "fifo"
        assert CachePolicy.RANDOM.value == "random"
