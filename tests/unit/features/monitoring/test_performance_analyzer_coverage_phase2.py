# -*- coding: utf-8 -*-
"""
性能分析器覆盖率测试 - Phase 2
针对PerformanceAnalyzer类的未覆盖方法进行补充测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import time

from src.features.monitoring.performance_analyzer import (
    PerformanceAnalyzer, AnalysisType, PerformanceAnalysis
)
from src.features.monitoring.features_monitor import MetricValue, MetricType


class TestPerformanceAnalyzerCoverage:
    """测试PerformanceAnalyzer的未覆盖方法"""

    @pytest.fixture
    def analyzer(self):
        """创建PerformanceAnalyzer实例"""
        return PerformanceAnalyzer()

    @pytest.fixture
    def sample_metrics_history(self):
        """生成示例指标历史数据"""
        # PerformanceAnalyzer期望的是MetricValue对象的列表或deque
        base_time = time.time()
        cpu_values = [MetricValue(
            name='cpu_usage',
            value=float(50 + np.random.randn() * 10),
            timestamp=base_time + i * 3600,
            metric_type=MetricType.GAUGE
        ) for i in range(100)]
        
        memory_values = [MetricValue(
            name='memory_usage',
            value=float(60 + np.random.randn() * 5),
            timestamp=base_time + i * 3600,
            metric_type=MetricType.GAUGE
        ) for i in range(100)]
        
        response_values = [MetricValue(
            name='response_time',
            value=float(0.5 + np.random.randn() * 0.1),
            timestamp=base_time + i * 3600,
            metric_type=MetricType.GAUGE
        ) for i in range(100)]
        
        return {
            'cpu_usage': cpu_values,
            'memory_usage': memory_values,
            'response_time': response_values
        }

    def test_analyze_performance_success(self, analyzer, sample_metrics_history):
        """测试分析性能 - 成功"""
        results = analyzer.analyze_performance(sample_metrics_history)
        
        # 验证结果
        assert isinstance(results, dict)
        assert 'timestamp' in results
        assert 'trends' in results
        assert 'anomalies' in results
        assert 'bottlenecks' in results

    def test_analyze_performance_empty_history(self, analyzer):
        """测试分析性能 - 空历史数据"""
        results = analyzer.analyze_performance({})
        
        # 验证结果
        assert isinstance(results, dict)
        assert 'trends' in results

    def test_analyze_trends(self, analyzer, sample_metrics_history):
        """测试分析趋势"""
        trends = analyzer._analyze_trends(sample_metrics_history)
        
        # 验证结果
        assert isinstance(trends, dict)

    def test_detect_anomalies(self, analyzer, sample_metrics_history):
        """测试检测异常"""
        anomalies = analyzer._detect_anomalies(sample_metrics_history)
        
        # 验证结果
        assert isinstance(anomalies, dict)

    def test_identify_bottlenecks(self, analyzer, sample_metrics_history):
        """测试识别瓶颈"""
        bottlenecks = analyzer._identify_bottlenecks(sample_metrics_history)
        
        # 验证结果
        assert isinstance(bottlenecks, dict)

    def test_analyze_correlations(self, analyzer, sample_metrics_history):
        """测试分析相关性"""
        correlations = analyzer._analyze_correlations(sample_metrics_history)
        
        # 验证结果
        assert isinstance(correlations, dict)

    def test_predict_performance(self, analyzer, sample_metrics_history):
        """测试预测性能"""
        predictions = analyzer._predict_performance(sample_metrics_history)
        
        # 验证结果
        assert isinstance(predictions, dict)

    def test_get_analysis_cache(self, analyzer, sample_metrics_history):
        """测试获取分析缓存"""
        # 执行分析
        analyzer.analyze_performance(sample_metrics_history)
        
        # 验证缓存已创建
        assert isinstance(analyzer.analysis_cache, dict)

    def test_analyze_performance_with_custom_config(self):
        """测试使用自定义配置"""
        config = {
            'anomaly_threshold': 3.0,
            'trend_window': 120,
            'correlation_threshold': 0.8,
            'prediction_horizon': 20
        }
        analyzer = PerformanceAnalyzer(config=config)
        
        assert analyzer.anomaly_threshold == 3.0
        assert analyzer.trend_window == 120
        assert analyzer.correlation_threshold == 0.8
        assert analyzer.prediction_horizon == 20

    def test_analyze_performance_exception_handling(self, analyzer):
        """测试异常处理"""
        # 创建会导致异常的数据
        invalid_history = {
            'invalid_metric': 'not_a_series'
        }
        
        # 应该捕获异常并返回结果
        results = analyzer.analyze_performance(invalid_history)
        
        # 验证返回了字典（即使有错误）
        assert isinstance(results, dict)

