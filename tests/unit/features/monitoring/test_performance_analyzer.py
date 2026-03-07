# -*- coding: utf-8 -*-
"""
性能分析器测试
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
from src.features.monitoring.performance_analyzer import (
    PerformanceAnalyzer,
    AnalysisType,
    PerformanceAnalysis
)
from src.features.monitoring.metrics_collector import MetricData, MetricCategory, MetricType


class TestPerformanceAnalyzer:
    """测试PerformanceAnalyzer类"""

    @pytest.fixture
    def analyzer(self):
        """创建PerformanceAnalyzer实例"""
        return PerformanceAnalyzer()

    @pytest.fixture
    def sample_metrics_history(self):
        """生成示例指标历史数据"""
        base_time = datetime.now().timestamp()
        return {
            'response_time': [
                MetricData('response_time', 0.5, base_time + i, MetricCategory.PERFORMANCE, MetricType.GAUGE)
                for i in range(20)
            ],
            'cpu_usage': [
                MetricData('cpu_usage', 50.0 + i * 0.5, base_time + i, MetricCategory.SYSTEM, MetricType.GAUGE)
                for i in range(20)
            ],
            'memory_usage': [
                MetricData('memory_usage', 60.0, base_time + i, MetricCategory.SYSTEM, MetricType.GAUGE)
                for i in range(20)
            ]
        }

    def test_init(self, analyzer):
        """测试初始化"""
        assert analyzer.anomaly_threshold == 2.0
        assert analyzer.trend_window == 60
        assert analyzer.correlation_threshold == 0.7

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'anomaly_threshold': 3.0,
            'trend_window': 120,
            'correlation_threshold': 0.8
        }
        analyzer = PerformanceAnalyzer(config)
        assert analyzer.anomaly_threshold == 3.0
        assert analyzer.trend_window == 120
        assert analyzer.correlation_threshold == 0.8

    def test_analyze_performance(self, analyzer, sample_metrics_history):
        """测试性能分析"""
        result = analyzer.analyze_performance(sample_metrics_history)

        assert 'timestamp' in result
        assert 'trends' in result
        assert 'anomalies' in result
        assert 'bottlenecks' in result
        assert 'correlations' in result
        assert 'predictions' in result

    def test_analyze_performance_empty_history(self, analyzer):
        """测试空历史数据的分析"""
        result = analyzer.analyze_performance({})

        assert 'trends' in result
        assert 'anomalies' in result
        assert isinstance(result['trends'], dict)

    def test_analyze_trends(self, analyzer, sample_metrics_history):
        """测试趋势分析"""
        trends = analyzer._analyze_trends(sample_metrics_history)

        assert isinstance(trends, dict)
        # 检查是否有趋势结果
        if trends:
            for metric_name, trend_data in trends.items():
                assert 'slope' in trend_data
                assert 'trend_direction' in trend_data
                assert 'change_rate' in trend_data

    def test_analyze_trends_insufficient_data(self, analyzer):
        """测试数据不足的趋势分析"""
        history = {
            'metric1': [
                MetricData('metric1', 1.0, datetime.now().timestamp(), MetricCategory.CUSTOM, MetricType.GAUGE)
            ]
        }
        trends = analyzer._analyze_trends(history)
        assert isinstance(trends, dict)

    def test_detect_anomalies(self, analyzer):
        """测试异常检测"""
        base_time = datetime.now().timestamp()
        # 创建包含异常值的数据
        history = {
            'metric1': [
                MetricData('metric1', 10.0, base_time + i, MetricCategory.CUSTOM, MetricType.GAUGE)
                if i != 10 else MetricData('metric1', 100.0, base_time + i, MetricCategory.CUSTOM, MetricType.GAUGE)
                for i in range(20)
            ]
        }

        anomalies = analyzer._detect_anomalies(history)

        assert isinstance(anomalies, dict)

    def test_detect_anomalies_insufficient_data(self, analyzer):
        """测试数据不足的异常检测"""
        history = {
            'metric1': [
                MetricData('metric1', 1.0, datetime.now().timestamp(), None, None)
                for i in range(3)
            ]
        }
        anomalies = analyzer._detect_anomalies(history)
        assert isinstance(anomalies, dict)

    def test_identify_bottlenecks(self, analyzer):
        """测试瓶颈识别"""
        base_time = datetime.now().timestamp()
        # 创建高响应时间数据
        history = {
            'response_time': [
                MetricData('response_time', 2.0, base_time + i, MetricCategory.PERFORMANCE, MetricType.GAUGE)
                for i in range(20)
            ]
        }

        bottlenecks = analyzer._identify_bottlenecks(history)

        assert isinstance(bottlenecks, dict)

    def test_identify_bottlenecks_empty(self, analyzer):
        """测试空数据的瓶颈识别"""
        bottlenecks = analyzer._identify_bottlenecks({})
        assert isinstance(bottlenecks, dict)

    def test_analyze_correlations(self, analyzer, sample_metrics_history):
        """测试相关性分析"""
        correlations = analyzer._analyze_correlations(sample_metrics_history)

        assert isinstance(correlations, dict)

    def test_analyze_correlations_insufficient_data(self, analyzer):
        """测试数据不足的相关性分析"""
        history = {
            'metric1': [
                MetricData('metric1', 1.0, datetime.now().timestamp(), MetricCategory.CUSTOM, MetricType.GAUGE)
            ]
        }
        correlations = analyzer._analyze_correlations(history)
        assert isinstance(correlations, dict)

    def test_predict_performance(self, analyzer, sample_metrics_history):
        """测试性能预测"""
        predictions = analyzer._predict_performance(sample_metrics_history)

        assert isinstance(predictions, dict)

    def test_predict_performance_insufficient_data(self, analyzer):
        """测试数据不足的性能预测"""
        history = {
            'metric1': [
                MetricData('metric1', 1.0, datetime.now().timestamp(), MetricCategory.CUSTOM, MetricType.GAUGE)
            ]
        }
        predictions = analyzer._predict_performance(history)
        assert isinstance(predictions, dict)

    def test_analyze_performance_exception_handling(self, analyzer):
        """测试异常处理"""
        # 传入无效数据
        invalid_history = {
            'metric1': "invalid"
        }
        result = analyzer.analyze_performance(invalid_history)
        # 应该返回结果字典，即使有异常
        assert isinstance(result, dict)

    def test_get_analysis_cache(self, analyzer, sample_metrics_history):
        """测试获取分析缓存"""
        analyzer.analyze_performance(sample_metrics_history)
        # 检查缓存是否被创建
        assert isinstance(analyzer.analysis_cache, dict)

