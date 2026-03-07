#!/usr/bin/env python3
"""
测试趋势分析器模块

测试覆盖：
- trend_analyzer.py中的TrendAnalyzer类
- 数据点添加和窗口管理
- 趋势分析算法（线性回归、R²计算）
- 边界条件和异常处理
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import patch, MagicMock
import math

# 尝试导入模块
try:
    from src.infrastructure.config.monitoring.trend_analyzer import TrendAnalyzer
except ImportError:
    TrendAnalyzer = None


class TestTrendAnalyzer:
    """测试趋势分析器"""
    
    def setup_method(self):
        """测试前准备"""
        if TrendAnalyzer is None:
            pytest.skip("TrendAnalyzer导入失败，跳过测试")
    
    def test_initialization(self):
        """测试初始化"""
        analyzer = TrendAnalyzer()
        
        assert analyzer.window_size == 50
        assert isinstance(analyzer._data_series, dict)
        assert len(analyzer._data_series) == 0
    
    def test_initialization_custom_window_size(self):
        """测试自定义窗口大小初始化"""
        analyzer = TrendAnalyzer(window_size=100)
        
        assert analyzer.window_size == 100
    
    def test_add_data_point_new_metric(self):
        """测试添加新指标的数据点"""
        analyzer = TrendAnalyzer(window_size=10)
        
        analyzer.add_data_point("cpu_usage", 45.5)
        
        assert "cpu_usage" in analyzer._data_series
        assert analyzer._data_series["cpu_usage"] == [45.5]
    
    def test_add_data_point_existing_metric(self):
        """测试添加现有指标的数据点"""
        analyzer = TrendAnalyzer(window_size=10)
        
        analyzer.add_data_point("memory_usage", 30.0)
        analyzer.add_data_point("memory_usage", 35.0)
        analyzer.add_data_point("memory_usage", 40.0)
        
        assert analyzer._data_series["memory_usage"] == [30.0, 35.0, 40.0]
    
    def test_window_size_limiting(self):
        """测试窗口大小限制"""
        analyzer = TrendAnalyzer(window_size=3)
        
        # 添加超过窗口大小的数据点
        for i in range(5):
            analyzer.add_data_point("test_metric", float(i))
        
        # 应该只保留最后3个数据点
        expected_data = [2.0, 3.0, 4.0]
        assert analyzer._data_series["test_metric"] == expected_data
        assert len(analyzer._data_series["test_metric"]) == 3
    
    def test_analyze_trend_no_data(self):
        """测试分析没有数据的指标"""
        analyzer = TrendAnalyzer()
        
        result = analyzer.analyze_trend("nonexistent_metric")
        
        expected = {
            "trend": "insufficient_data",
            "slope": 0.0,
            "confidence": 0.0
        }
        assert result == expected
    
    def test_analyze_trend_insufficient_data(self):
        """测试分析数据不足的指标"""
        analyzer = TrendAnalyzer()
        
        # 添加少于10个数据点
        for i in range(5):
            analyzer.add_data_point("insufficient_metric", float(i))
        
        result = analyzer.analyze_trend("insufficient_metric")
        
        expected = {
            "trend": "insufficient_data",
            "slope": 0.0,
            "confidence": 0.0
        }
        assert result == expected
    
    def test_analyze_trend_increasing(self):
        """测试分析上升趋势"""
        analyzer = TrendAnalyzer()
        
        # 添加明显的上升趋势数据
        for i in range(15):
            analyzer.add_data_point("increasing_metric", float(i * 2 + 10))
        
        result = analyzer.analyze_trend("increasing_metric")
        
        assert result["trend"] == "increasing"
        assert result["slope"] > 0
        assert 0 <= result["confidence"] <= 1
        assert "intercept" in result
        assert "data_points" in result
        assert result["data_points"] == 15
    
    def test_analyze_trend_decreasing(self):
        """测试分析下降趋势"""
        analyzer = TrendAnalyzer()
        
        # 添加明显的下降趋势数据
        for i in range(15):
            analyzer.add_data_point("decreasing_metric", float(50 - i * 2))
        
        result = analyzer.analyze_trend("decreasing_metric")
        
        assert result["trend"] == "decreasing"
        assert result["slope"] < 0
        assert 0 <= result["confidence"] <= 1
    
    def test_analyze_trend_stable(self):
        """测试分析稳定趋势"""
        analyzer = TrendAnalyzer()
        
        # 添加稳定的数据（变化很小）
        base_value = 25.0
        for i in range(15):
            # 添加小的随机变化
            value = base_value + (0.1 if i % 2 == 0 else -0.1)
            analyzer.add_data_point("stable_metric", value)
        
        result = analyzer.analyze_trend("stable_metric")
        
        assert result["trend"] == "stable"
        assert abs(result["slope"]) < 0.001
        assert 0 <= result["confidence"] <= 1
    
    def test_analyze_trend_perfect_correlation(self):
        """测试完美相关性"""
        analyzer = TrendAnalyzer()
        
        # 添加完美的线性关系
        for i in range(20):
            analyzer.add_data_point("perfect_metric", float(i * 3 + 5))
        
        result = analyzer.analyze_trend("perfect_metric")
        
        # 完美线性关系的R²应该接近1
        assert result["confidence"] > 0.99
        assert abs(result["slope"] - 3.0) < 0.01  # 斜率应该接近3
    
    def test_analyze_trend_zero_variance(self):
        """测试零方差数据（所有值相同）"""
        analyzer = TrendAnalyzer()
        
        # 添加所有相同的值
        for i in range(15):
            analyzer.add_data_point("constant_metric", 42.0)
        
        result = analyzer.analyze_trend("constant_metric")
        
        assert result["slope"] == 0.0
        # 当ss_tot为0时，r_squared应该为0
        assert result["confidence"] == 0.0
        assert result["trend"] == "stable"
    
    def test_multiple_metrics_independence(self):
        """测试多个指标的独立性"""
        analyzer = TrendAnalyzer()
        
        # 为不同指标添加不同的数据
        for i in range(15):
            analyzer.add_data_point("metric1", float(i))
            analyzer.add_data_point("metric2", float(-i + 10))
        
        result1 = analyzer.analyze_trend("metric1")
        result2 = analyzer.analyze_trend("metric2")
        
        # 两个指标应该有相反的趋势
        assert result1["trend"] == "increasing"
        assert result2["trend"] == "decreasing"
        assert result1["slope"] > 0
        assert result2["slope"] < 0
    
    def test_large_window_size(self):
        """测试大窗口大小"""
        analyzer = TrendAnalyzer(window_size=1000)
        
        # 添加大量数据点
        for i in range(1200):
            analyzer.add_data_point("large_window_metric", float(i))
        
        # 应该被限制在1000个数据点
        assert len(analyzer._data_series["large_window_metric"]) == 1000
        
        result = analyzer.analyze_trend("large_window_metric")
        assert result["data_points"] == 1000
    
    def test_edge_case_minimum_data_points(self):
        """测试最小数据点的边界情况"""
        analyzer = TrendAnalyzer()
        
        # 添加恰好10个数据点（最小要求）
        for i in range(10):
            analyzer.add_data_point("min_data_metric", float(i))
        
        result = analyzer.analyze_trend("min_data_metric")
        
        # 应该能正常分析
        assert result["trend"] in ["increasing", "decreasing", "stable"]
        assert "slope" in result
        assert "confidence" in result
    
    def test_floating_point_precision(self):
        """测试浮点数精度"""
        analyzer = TrendAnalyzer()
        
        # 使用有精度问题的浮点数
        values = [0.1 + 0.2, 0.2 + 0.3, 0.3 + 0.4, 0.4 + 0.5, 0.5 + 0.6,
                 0.6 + 0.7, 0.7 + 0.8, 0.8 + 0.9, 0.9 + 1.0, 1.0 + 1.1,
                 1.1 + 1.2, 1.2 + 1.3, 1.3 + 1.4, 1.4 + 1.5, 1.5 + 1.6]
        
        for value in values:
            analyzer.add_data_point("precision_metric", value)
        
        result = analyzer.analyze_trend("precision_metric")
        
        # 结果应该是合理的
        assert isinstance(result["slope"], float)
        assert isinstance(result["confidence"], float)
        assert not math.isnan(result["slope"])
        assert not math.isinf(result["slope"])
    
    def test_negative_values(self):
        """测试负数值"""
        analyzer = TrendAnalyzer()
        
        # 添加负数和正数混合的数据
        values = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        for value in values:
            analyzer.add_data_point("negative_metric", float(value))
        
        result = analyzer.analyze_trend("negative_metric")
        
        assert result["trend"] == "increasing"
        assert result["slope"] > 0
        assert 0 <= result["confidence"] <= 1
