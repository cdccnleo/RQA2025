#!/usr/bin/env python3
"""
缓存优化器增强测试套件

全面覆盖 cache_optimizer.py 的所有功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import unittest.mock as mock
from unittest.mock import Mock, MagicMock, patch
import tempfile
import threading
import time
from typing import Dict, List, Any, Optional

# 导入缓存优化器组件
from src.infrastructure.cache.core.cache_optimizer import (
    CacheOptimizer,
    CachePolicy
)


class TestCacheOptimizer:
    """缓存优化器测试"""

    @pytest.fixture
    def cache_optimizer(self):
        """缓存优化器实例"""
        optimizer = CacheOptimizer()
        yield optimizer

    def test_initialization(self):
        """测试初始化"""
        optimizer = CacheOptimizer()

        assert optimizer._optimization_history == []
        assert optimizer._recommendation_cache == {}

    def test_optimize_cache_size_low_hit_rate_high_memory(self, cache_optimizer):
        """测试低命中率高内存使用情况下的缓存大小优化"""
        # 低命中率(0.3)，高内存使用(0.9)
        result = cache_optimizer.optimize_cache_size(1000, 0.3, 0.9)

        # 应该减少缓存大小
        assert result < 1000
        assert result >= 100  # 最小值

        # 验证优化历史记录
        assert len(cache_optimizer._optimization_history) == 1
        history = cache_optimizer._optimization_history[0]
        assert history["current_size"] == 1000
        assert history["target_size"] == result
        assert history["hit_rate"] == 0.3
        assert history["memory_usage"] == 0.9

    def test_optimize_cache_size_high_hit_rate_low_memory(self, cache_optimizer):
        """测试高命中率低内存使用情况下的缓存大小优化"""
        # 高命中率(0.9)，低内存使用(0.4)
        result = cache_optimizer.optimize_cache_size(1000, 0.9, 0.4)

        # 应该增加缓存大小
        assert result > 1000
        assert result <= 10000  # 最大值

    def test_optimize_cache_size_extreme_cases(self, cache_optimizer):
        """测试极端情况的缓存大小优化"""
        # 命中率0%，内存使用100%
        result = cache_optimizer.optimize_cache_size(1000, 0.0, 1.0)
        assert result == 100  # 强制最小值

        # 命中率100%，内存使用0%
        result = cache_optimizer.optimize_cache_size(1000, 1.0, 0.0)
        assert result == 10000  # 强制最大值

    def test_optimize_cache_size_normal_case(self, cache_optimizer):
        """测试正常情况的缓存大小优化"""
        # 中等命中率，中等内存使用
        result = cache_optimizer.optimize_cache_size(1000, 0.7, 0.7)

        # 应该保持不变或小幅调整
        assert abs(result - 1000) <= 1000  # 变化不大

    def test_suggest_eviction_policy_empty_pattern(self, cache_optimizer):
        """测试空访问模式下的淘汰策略建议"""
        result = cache_optimizer.suggest_eviction_policy({})

        assert result == CachePolicy.LRU

    def test_suggest_eviction_policy_sequential_access(self, cache_optimizer):
        """测试顺序访问模式的淘汰策略建议"""
        # 模拟顺序访问模式
        access_pattern = {
            "sequential_access": 80,
            "random_access": 20
        }

        result = cache_optimizer.suggest_eviction_policy(access_pattern)

        # 顺序访问应该建议LRU
        assert result == CachePolicy.LRU

    def test_suggest_eviction_policy_random_access(self, cache_optimizer):
        """测试随机访问模式的淘汰策略建议"""
        # 模拟随机访问模式
        access_pattern = {
            "random_access": 70,
            "sequential_access": 30
        }

        result = cache_optimizer.suggest_eviction_policy(access_pattern)

        # 随机访问应该建议LFU
        assert result == CachePolicy.LFU

    def test_suggest_eviction_policy_frequent_access(self, cache_optimizer):
        """测试频繁访问模式的淘汰策略建议"""
        # 模拟频繁访问模式
        access_pattern = {
            "frequent_access": 60,
            "normal_access": 40
        }

        result = cache_optimizer.suggest_eviction_policy(access_pattern)

        # 频繁访问应该建议LFU
        assert result == CachePolicy.LFU

    def test_get_cache_recommendations(self, cache_optimizer):
        """测试获取缓存建议"""
        cache_stats = {
            "size": 1000,
            "hit_rate": 0.3,  # 低命中率
            "memory_usage": 0.95,  # 高内存使用
            "policy": "lru",
            "access_pattern": {"random_access": 60},
            "eviction_rate": 0.15,  # 高淘汰率
            "load_time": 2.0  # 高加载时间
        }

        recommendations = cache_optimizer.get_cache_recommendations(cache_stats)

        assert isinstance(recommendations, dict)
        assert "size_optimization" in recommendations
        assert "policy_recommendation" in recommendations
        assert "performance_improvements" in recommendations
        assert "warnings" in recommendations

        # 应该有警告信息
        assert len(recommendations["warnings"]) > 0
        assert len(recommendations["performance_improvements"]) > 0

    def test_get_cache_recommendations_good_performance(self, cache_optimizer):
        """测试良好性能情况下的缓存建议"""
        cache_stats = {
            "size": 1000,
            "hit_rate": 0.95,  # 高命中率
            "memory_usage": 0.3,  # 低内存使用
            "policy": "lru",
            "access_pattern": {"sequential_access": 70},
            "eviction_rate": 0.02,  # 低淘汰率
            "load_time": 0.5  # 低加载时间
        }

        recommendations = cache_optimizer.get_cache_recommendations(cache_stats)

        assert isinstance(recommendations, dict)

        # 应该有性能改进建议
        assert len(recommendations["performance_improvements"]) > 0

    def test_get_optimization_history(self, cache_optimizer):
        """测试获取优化历史"""
        # 先进行一些优化操作
        cache_optimizer.optimize_cache_size(1000, 0.5, 0.8)
        cache_optimizer.optimize_cache_size(800, 0.6, 0.7)

        history = cache_optimizer.get_optimization_history()

        assert isinstance(history, list)
        assert len(history) == 2

        # 验证历史记录结构
        for record in history:
            assert "current_size" in record
            assert "target_size" in record
            assert "hit_rate" in record
            assert "memory_usage" in record
            assert "reason" in record
            assert "timestamp" in record

    def test_clear_optimization_history(self, cache_optimizer):
        """测试清空优化历史"""
        # 先添加一些历史记录
        cache_optimizer.optimize_cache_size(1000, 0.5, 0.8)

        # 验证有历史记录
        assert len(cache_optimizer.get_optimization_history()) > 0

        # 清空历史记录
        cache_optimizer.clear_optimization_history()

        # 验证已清空
        assert len(cache_optimizer.get_optimization_history()) == 0

    def test_analyze_access_patterns(self, cache_optimizer):
        """测试访问模式分析"""
        access_pattern = {
            "read_operations": 100,
            "write_operations": 20,
            "cache_hits": 80,
            "cache_misses": 20,
            "sequential_access": 60,
            "random_access": 40
        }

        analysis = cache_optimizer.analyze_access_patterns(access_pattern)

        assert isinstance(analysis, dict)
        assert "read_write_ratio" in analysis
        assert "hit_rate" in analysis
        assert "access_pattern_type" in analysis
        assert "recommendations" in analysis

    def test_get_optimization_recommendations(self, cache_optimizer):
        """测试获取优化建议"""
        # 先添加一些历史数据
        cache_optimizer.optimize_cache_size(1000, 0.5, 0.8)
        cache_optimizer.optimize_cache_size(800, 0.6, 0.7)
        cache_optimizer.optimize_cache_size(900, 0.7, 0.6)

        recommendations = cache_optimizer.get_optimization_recommendations()

        assert isinstance(recommendations, dict)
        assert "size_trend_analysis" in recommendations
        assert "hit_rate_trend_analysis" in recommendations
        assert "overall_recommendations" in recommendations

    def test_get_performance_metrics_no_history(self, cache_optimizer):
        """测试无历史记录时的性能指标"""
        metrics = cache_optimizer.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert metrics["total_optimizations"] == 0
        assert metrics["optimization_count"] == 0
        assert metrics["avg_hit_rate"] == 0.0
        assert metrics["avg_memory_usage"] == 0.0
        assert metrics["optimization_success_rate"] == 0.0
        assert metrics["performance_score"] == 0.0
        assert metrics["avg_improvement"] == 0.0
        assert metrics["last_optimization"] is None

    def test_get_performance_metrics_with_history(self, cache_optimizer):
        """测试有历史记录时的性能指标"""
        # 添加一些历史记录
        cache_optimizer.optimize_cache_size(1000, 0.5, 0.8)
        cache_optimizer.optimize_cache_size(800, 0.6, 0.7)
        cache_optimizer.optimize_cache_size(900, 0.7, 0.6)

        metrics = cache_optimizer.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert metrics["total_optimizations"] == 3
        assert metrics["optimization_count"] == 3
        assert metrics["avg_hit_rate"] > 0
        assert metrics["avg_memory_usage"] > 0
        assert "optimization_success_rate" in metrics
        assert "performance_score" in metrics
        assert "last_optimization" in metrics

    def test_monitor_cache_performance(self, cache_optimizer):
        """测试缓存性能监控"""
        performance_metrics = {
            "hit_rate": 0.85,
            "memory_usage": 0.7,
            "response_time": 0.05,
            "throughput": 1000,
            "error_rate": 0.01
        }

        # 应该不会抛出异常
        cache_optimizer.monitor_cache_performance(performance_metrics)

    def test_optimize_eviction_policy(self, cache_optimizer):
        """测试淘汰策略优化"""
        cache_stats = {
            "hit_rate": 0.6,
            "memory_usage": 0.8,
            "access_pattern": {"random_access": 70},
            "current_policy": "lru"
        }

        result = cache_optimizer.optimize_eviction_policy(cache_stats)

        assert isinstance(result, str)
        # 基于随机访问模式，应该建议LFU
        assert "lfu" in result.lower()

    def test_reset_optimization_history(self, cache_optimizer):
        """测试重置优化历史"""
        # 添加一些历史记录
        cache_optimizer.optimize_cache_size(1000, 0.5, 0.8)

        # 重置历史
        cache_optimizer.reset_optimization_history()

        # 验证已重置
        assert len(cache_optimizer._optimization_history) == 0

    def test_private_methods(self, cache_optimizer):
        """测试私有方法"""
        # 测试优化原因获取
        reason = cache_optimizer._get_optimization_reason(0.3, 0.9)
        assert isinstance(reason, str)

        # 测试策略推荐原因获取
        access_pattern = {"random_access": 60}
        reason = cache_optimizer._get_policy_recommendation_reason(access_pattern)
        assert isinstance(reason, str)

    def test_generate_recommendations(self, cache_optimizer):
        """测试生成建议"""
        # 先添加一些历史记录
        cache_optimizer.optimize_cache_size(1000, 0.5, 0.8)
        cache_optimizer.optimize_cache_size(800, 0.6, 0.7)

        recommendations = cache_optimizer._generate_recommendations()

        assert isinstance(recommendations, list)

    def test_analyze_trends(self, cache_optimizer):
        """测试趋势分析"""
        # 添加历史记录用于趋势分析
        recommendations = []
        recent_optimizations = [
            {"timestamp": time.time() - 300, "target_size": 900, "current_size": 1000, "hit_rate": 0.5, "memory_usage": 0.8},
            {"timestamp": time.time() - 200, "target_size": 800, "current_size": 900, "hit_rate": 0.6, "memory_usage": 0.7},
            {"timestamp": time.time() - 100, "target_size": 700, "current_size": 800, "hit_rate": 0.65, "memory_usage": 0.6}
        ]

        # 测试大小趋势分析
        cache_optimizer._analyze_size_trend(recommendations, recent_optimizations)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0  # 应该添加了趋势分析建议

        # 清空recommendations重新测试命中率趋势分析
        recommendations.clear()
        cache_optimizer._analyze_hit_rate_trend(recommendations, recent_optimizations)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0  # 应该添加了趋势分析建议

    def test_analyze_cache_size_optimization(self, cache_optimizer):
        """测试缓存大小优化分析"""
        analysis = cache_optimizer._analyze_cache_size_optimization()

        assert isinstance(analysis, dict)

    def test_analyze_performance_trends(self, cache_optimizer):
        """测试性能趋势分析"""
        trends = cache_optimizer._analyze_performance_trends()

        assert isinstance(trends, dict)


class TestCachePolicy:
    """缓存策略枚举测试"""

    def test_cache_policy_values(self):
        """测试缓存策略枚举值"""
        assert CachePolicy.LRU.value == "lru"
        assert CachePolicy.LFU.value == "lfu"
        assert CachePolicy.FIFO.value == "fifo"
        assert CachePolicy.RANDOM.value == "random"

    def test_cache_policy_members(self):
        """测试缓存策略枚举成员"""
        policies = list(CachePolicy)
        assert len(policies) == 4
        assert CachePolicy.LRU in policies
        assert CachePolicy.LFU in policies
        assert CachePolicy.FIFO in policies
        assert CachePolicy.RANDOM in policies


class TestIntegrationScenarios:
    """集成场景测试"""

    def test_complete_optimization_workflow(self):
        """测试完整的优化工作流程"""
        optimizer = CacheOptimizer()

        # 1. 初始缓存配置
        initial_stats = {
            "size": 1000,
            "hit_rate": 0.4,
            "memory_usage": 0.9,
            "policy": "lru",
            "access_pattern": {"random_access": 80}
        }

        # 2. 获取建议
        recommendations = optimizer.get_cache_recommendations(initial_stats)

        # 3. 应用大小优化
        optimal_size = optimizer.optimize_cache_size(
            initial_stats["size"],
            initial_stats["hit_rate"],
            initial_stats["memory_usage"]
        )

        # 4. 建议淘汰策略
        recommended_policy = optimizer.suggest_eviction_policy(
            initial_stats["access_pattern"]
        )

        # 5. 监控性能
        performance_metrics = {
            "hit_rate": initial_stats["hit_rate"],
            "memory_usage": initial_stats["memory_usage"],
            "response_time": 0.1,
            "throughput": 500
        }
        optimizer.monitor_cache_performance(performance_metrics)

        # 6. 获取最终性能指标
        final_metrics = optimizer.get_performance_metrics()

        # 验证整个流程
        assert isinstance(recommendations, dict)
        assert isinstance(optimal_size, int)
        assert isinstance(recommended_policy, CachePolicy)
        assert isinstance(final_metrics, dict)

        # 验证优化历史
        history = optimizer.get_optimization_history()
        assert len(history) >= 1

    def test_adaptive_optimization_based_on_patterns(self):
        """测试基于访问模式的自适应优化"""
        optimizer = CacheOptimizer()

        # 测试不同访问模式的优化
        test_cases = [
            ({"sequential_access": 80}, CachePolicy.LRU),
            ({"random_access": 70}, CachePolicy.LFU),
            ({"frequent_access": 60}, CachePolicy.LFU),
            ({}, CachePolicy.LRU)  # 默认情况
        ]

        for access_pattern, expected_policy in test_cases:
            recommended = optimizer.suggest_eviction_policy(access_pattern)
            assert recommended == expected_policy

    def test_performance_monitoring_and_reporting(self):
        """测试性能监控和报告"""
        optimizer = CacheOptimizer()

        # 模拟一系列优化操作
        operations = [
            (1000, 0.4, 0.9),
            (800, 0.5, 0.8),
            (900, 0.6, 0.7),
            (1000, 0.7, 0.6)
        ]

        for size, hit_rate, memory_usage in operations:
            optimizer.optimize_cache_size(size, hit_rate, memory_usage)

        # 获取性能指标
        metrics = optimizer.get_performance_metrics()

        # 验证指标计算
        assert metrics["total_optimizations"] == 4
        assert metrics["avg_hit_rate"] > 0
        assert metrics["avg_memory_usage"] > 0
        assert metrics["performance_score"] >= 0
        assert metrics["performance_score"] <= 100

        # 获取优化建议
        recommendations = optimizer.get_optimization_recommendations()
        assert isinstance(recommendations, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

