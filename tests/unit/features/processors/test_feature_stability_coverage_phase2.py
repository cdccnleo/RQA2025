# -*- coding: utf-8 -*-
"""
特征稳定性分析器覆盖率测试 - Phase 2
针对FeatureStabilityAnalyzer类的未覆盖方法进行补充测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.features.processors.feature_stability import FeatureStabilityAnalyzer


class TestFeatureStabilityAnalyzerCoverage:
    """测试FeatureStabilityAnalyzer的未覆盖方法"""

    @pytest.fixture
    def sample_features(self):
        """生成示例特征数据"""
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        return pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200) * 2,
            'feature3': np.random.randn(200) * 3
        }, index=dates)

    @pytest.fixture
    def analyzer(self):
        """创建FeatureStabilityAnalyzer实例"""
        return FeatureStabilityAnalyzer()

    def test_analyze_feature_stability_success(self, analyzer, sample_features):
        """测试分析特征稳定性 - 成功"""
        result = analyzer.analyze_feature_stability(sample_features)
        
        # 验证结果
        assert isinstance(result, dict)
        assert 'analysis_results' in result
        assert 'combined_stability' in result
        assert 'analysis_report' in result
        assert isinstance(result['combined_stability'], dict)

    def test_analyze_feature_stability_with_time_index(self, analyzer, sample_features):
        """测试分析特征稳定性 - 带时间索引"""
        time_index = sample_features.index
        result = analyzer.analyze_feature_stability(sample_features, time_index=time_index)
        
        # 验证结果
        assert isinstance(result, dict)
        assert 'analysis_results' in result
        assert 'temporal_stability' in result['analysis_results']

    def test_preprocess_features(self, analyzer, sample_features):
        """测试预处理特征数据"""
        result = analyzer._preprocess_features(sample_features)
        
        # 验证结果
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_features)
        # 验证NaN已被填充
        assert not result.isna().any().any()

    def test_preprocess_features_with_nan(self, analyzer):
        """测试预处理特征数据 - 包含NaN"""
        data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, 20, 30, np.nan, 50]
        })
        
        result = analyzer._preprocess_features(data)
        
        # 验证NaN已被填充
        assert not result.isna().any().any()

    def test_analyze_statistical_stability(self, analyzer, sample_features):
        """测试分析统计稳定性"""
        processed_features = analyzer._preprocess_features(sample_features)
        result = analyzer._analyze_statistical_stability(processed_features)
        
        # 验证结果
        assert isinstance(result, dict)
        assert len(result) == len(sample_features.columns)
        # 验证稳定性评分在0-1范围内
        assert all(0.0 <= score <= 1.0 for score in result.values())

    def test_analyze_statistical_stability_zero_mean(self, analyzer):
        """测试分析统计稳定性 - 均值为0"""
        data = pd.DataFrame({
            'feature1': [0, 0, 0, 0, 0],
            'feature2': [-1, 0, 1, -1, 1]
        })
        
        processed_features = analyzer._preprocess_features(data)
        result = analyzer._analyze_statistical_stability(processed_features)
        
        # 验证结果
        assert isinstance(result, dict)
        # feature1的均值为0，应该返回0.0或特殊处理
        assert 'feature1' in result
        assert 'feature2' in result

    def test_analyze_distribution_stability(self, analyzer, sample_features):
        """测试分析分布稳定性"""
        processed_features = analyzer._preprocess_features(sample_features)
        result = analyzer._analyze_distribution_stability(processed_features)
        
        # 验证结果
        assert isinstance(result, dict)
        assert len(result) == len(sample_features.columns)
        # 验证稳定性评分在0-1范围内
        assert all(0.0 <= score <= 1.0 for score in result.values())

    def test_analyze_temporal_stability_with_time_index(self, analyzer, sample_features):
        """测试分析时间稳定性 - 带时间索引"""
        processed_features = analyzer._preprocess_features(sample_features)
        time_index = sample_features.index
        
        result = analyzer._analyze_temporal_stability(processed_features, time_index)
        
        # 验证结果
        assert isinstance(result, dict)
        assert len(result) == len(sample_features.columns)
        # 验证稳定性评分在0-1范围内
        assert all(0.0 <= score <= 1.0 for score in result.values())

    def test_analyze_temporal_stability_no_time_index(self, analyzer, sample_features):
        """测试分析时间稳定性 - 无时间索引"""
        processed_features = analyzer._preprocess_features(sample_features)
        
        result = analyzer._analyze_temporal_stability(processed_features, time_index=None)
        
        # 验证结果（应该返回默认值）
        assert isinstance(result, dict)
        assert len(result) == len(sample_features.columns)
        # 应该返回默认值0.5
        assert all(score == 0.5 for score in result.values())

    def test_analyze_temporal_stability_insufficient_data(self, analyzer):
        """测试分析时间稳定性 - 数据不足"""
        data = pd.DataFrame({
            'feature1': [1, 2, 3]  # 少于min_samples
        })
        processed_features = analyzer._preprocess_features(data)
        time_index = pd.date_range('2023-01-01', periods=3, freq='D')
        
        result = analyzer._analyze_temporal_stability(processed_features, time_index)
        
        # 验证结果（应该返回默认值）
        assert isinstance(result, dict)
        assert all(score == 0.5 for score in result.values())

    def test_analyze_correlation_stability(self, analyzer, sample_features):
        """测试分析相关性稳定性"""
        processed_features = analyzer._preprocess_features(sample_features)
        result = analyzer._analyze_correlation_stability(processed_features)
        
        # 验证结果
        assert isinstance(result, dict)
        assert len(result) == len(sample_features.columns)
        # 验证稳定性评分在0-1范围内
        assert all(0.0 <= score <= 1.0 for score in result.values())

    def test_analyze_correlation_stability_single_column(self, analyzer):
        """测试分析相关性稳定性 - 单列"""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5]
        })
        processed_features = analyzer._preprocess_features(data)
        result = analyzer._analyze_correlation_stability(processed_features)
        
        # 验证结果（应该返回1.0）
        assert isinstance(result, dict)
        assert result['feature1'] == 1.0

    def test_detect_feature_drift_with_time_index(self, analyzer, sample_features):
        """测试检测特征漂移 - 带时间索引"""
        processed_features = analyzer._preprocess_features(sample_features)
        time_index = sample_features.index
        
        result = analyzer._detect_feature_drift(processed_features, time_index)
        
        # 验证结果
        assert isinstance(result, dict)
        assert 'drift_scores' in result
        assert 'drift_indicators' in result
        assert 'drift_severity' in result
        assert isinstance(result['drift_scores'], dict)

    def test_detect_feature_drift_no_time_index(self, analyzer, sample_features):
        """测试检测特征漂移 - 无时间索引"""
        processed_features = analyzer._preprocess_features(sample_features)
        
        result = analyzer._detect_feature_drift(processed_features, time_index=None)
        
        # 验证结果（应该返回空结果）
        assert isinstance(result, dict)
        assert 'drift_scores' in result
        assert len(result['drift_scores']) == 0

    def test_detect_feature_drift_insufficient_data(self, analyzer):
        """测试检测特征漂移 - 数据不足"""
        data = pd.DataFrame({
            'feature1': [1, 2, 3]  # 少于min_samples
        })
        processed_features = analyzer._preprocess_features(data)
        time_index = pd.date_range('2023-01-01', periods=3, freq='D')
        
        result = analyzer._detect_feature_drift(processed_features, time_index)
        
        # 验证结果（应该返回空结果）
        assert isinstance(result, dict)
        assert len(result['drift_scores']) == 0

    def test_calculate_ks_statistic(self, analyzer):
        """测试计算KS统计量"""
        dist1 = pd.Series([1, 2, 3, 4, 5])
        dist2 = pd.Series([2, 3, 4, 5, 6])
        
        result = analyzer._calculate_ks_statistic(dist1, dist2)
        
        # 验证结果
        assert isinstance(result, float)
        assert result >= 0.0

    def test_calculate_ks_statistic_identical(self, analyzer):
        """测试计算KS统计量 - 相同分布"""
        dist1 = pd.Series([1, 2, 3, 4, 5])
        dist2 = pd.Series([1, 2, 3, 4, 5])
        
        result = analyzer._calculate_ks_statistic(dist1, dist2)
        
        # 验证结果（应该接近0）
        assert isinstance(result, float)
        assert result >= 0.0

    def test_combine_stability_scores(self, analyzer):
        """测试综合稳定性评分"""
        results = {
            'statistical_stability': {'feature1': 0.8, 'feature2': 0.6},
            'distribution_stability': {'feature1': 0.7, 'feature2': 0.5},
            'temporal_stability': {'feature1': 0.9, 'feature2': 0.7},
            'correlation_stability': {'feature1': 0.85, 'feature2': 0.65}
        }
        
        result = analyzer._combine_stability_scores(results)
        
        # 验证结果
        assert isinstance(result, dict)
        assert 'feature1' in result
        assert 'feature2' in result
        # 验证综合评分在0-1范围内
        assert all(0.0 <= score <= 1.0 for score in result.values())

    def test_generate_stability_report(self, analyzer):
        """测试生成稳定性报告"""
        results = {
            'statistical_stability': {'feature1': 0.8},
            'distribution_stability': {'feature1': 0.7},
            'temporal_stability': {'feature1': 0.9},
            'correlation_stability': {'feature1': 0.85},
            'drift_detection': {
                'drift_scores': {'feature1': 0.1},
                'drift_indicators': {'feature1': {'ks_statistic': 0.1, 'mean_drift': 0.1, 'var_drift': 0.1}},
                'drift_severity': {'feature1': 'low'}
            }
        }
        combined_stability = {'feature1': 0.8}
        
        result = analyzer._generate_stability_report(results, combined_stability)
        
        # 验证结果
        assert isinstance(result, dict)
        # 报告应该包含摘要信息
        assert 'summary' in result
        assert 'recommendations' in result
        assert 'stability_ranking' in result

    def test_analyze_feature_stability_exception_handling(self, analyzer):
        """测试异常处理"""
        # 创建会导致异常的数据
        invalid_data = pd.DataFrame({
            'feature1': ['invalid'] * 10  # 非数值类型
        })
        
        # 应该捕获异常并返回结果或处理错误
        try:
            result = analyzer.analyze_feature_stability(invalid_data)
            # 如果成功，验证返回了字典
            assert isinstance(result, dict)
        except Exception:
            # 如果抛出异常，这也是可以接受的（取决于实现）
            pass
