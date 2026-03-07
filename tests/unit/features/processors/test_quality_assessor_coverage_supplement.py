#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quality_assessor补充测试覆盖
针对未覆盖的代码分支编写测试
"""

import pytest
import pandas as pd
import numpy as np
from src.features.processors.quality_assessor import (
    FeatureQualityAssessor,
    AssessmentConfig,
    QualityMetrics
)


class TestQualityAssessorCoverageSupplement:
    """quality_assessor补充测试"""

    def test_preprocess_features_empty(self):
        """测试_preprocess_features（空DataFrame）"""
        assessor = FeatureQualityAssessor()
        result = assessor._preprocess_features(pd.DataFrame())
        assert result.empty

    def test_calculate_importance_empty_features(self):
        """测试_calculate_importance（空features）"""
        assessor = FeatureQualityAssessor()
        features = pd.DataFrame()
        target = pd.Series([1, 2, 3])
        
        result = assessor._calculate_importance(features, target)
        assert result == {}

    def test_calculate_importance_empty_target(self):
        """测试_calculate_importance（空target）"""
        assessor = FeatureQualityAssessor()
        features = pd.DataFrame({'f1': [1, 2, 3]})
        target = pd.Series([], dtype=float)
        
        result = assessor._calculate_importance(features, target)
        assert result == {}

    def test_calculate_importance_length_mismatch(self):
        """测试_calculate_importance（长度不匹配）"""
        assessor = FeatureQualityAssessor()
        features = pd.DataFrame({'f1': [1, 2, 3, 4, 5]})
        target = pd.Series([1, 2, 3])  # 长度不匹配
        
        result = assessor._calculate_importance(features, target)
        # 应该自动截断到最小长度
        assert isinstance(result, dict)

    def test_calculate_importance_classification(self):
        """测试_calculate_importance（分类问题）"""
        config = AssessmentConfig(
            use_random_forest=True,
            use_mutual_info=False,
            n_estimators=10,
            random_state=42
        )
        assessor = FeatureQualityAssessor(config)
        
        # 创建分类数据（目标变量为类别）
        features = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100)
        })
        target = pd.Series(['A', 'B'] * 50)  # 分类目标
        
        result = assessor._calculate_importance(features, target)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_calculate_stability_exception_handling(self):
        """测试_calculate_stability异常处理"""
        assessor = FeatureQualityAssessor()
        # 创建会导致异常的数据
        features = pd.DataFrame({
            'f1': [1, 2, 3],
            'f2': [np.nan, np.nan, np.nan]  # 全NaN列
        })
        
        result = assessor._calculate_stability(features)
        assert isinstance(result, dict)

    def test_calculate_stability_zero_std(self):
        """测试_calculate_stability（标准差为0）"""
        assessor = FeatureQualityAssessor()
        features = pd.DataFrame({
            'f1': [1, 1, 1],  # 常量列
            'f2': [2, 2, 2]
        })
        
        result = assessor._calculate_stability(features)
        assert isinstance(result, dict)

    def test_calculate_information_content_empty_values(self):
        """测试_calculate_information_content（空值）"""
        assessor = FeatureQualityAssessor()
        features = pd.DataFrame({
            'f1': [np.nan, np.nan, np.nan]  # 全NaN
        })
        target = pd.Series([1, 2, 3])
        
        result = assessor._calculate_information_content(features, target)
        assert isinstance(result, dict)
        assert result.get('f1', 0) == 0.0

    def test_calculate_information_content_exception(self):
        """测试_calculate_information_content异常处理"""
        assessor = FeatureQualityAssessor()
        # 方法内部有异常处理，不会抛出异常，而是返回结果
        features = pd.DataFrame({
            'f1': [1, 2, 3]
        })
        target = pd.Series([1, 2, 3])
        
        result = assessor._calculate_information_content(features, target)
        assert isinstance(result, dict)

    def test_calculate_redundancy_single_feature(self):
        """测试_calculate_redundancy（单个特征）"""
        assessor = FeatureQualityAssessor()
        features = pd.DataFrame({
            'f1': [1, 2, 3]
        })
        
        result = assessor._calculate_redundancy(features)
        assert isinstance(result, dict)
        # 单个特征时，其他特征为空，应该返回0.0
        assert result.get('f1', 0) == 0.0

    def test_calculate_redundancy_exception(self):
        """测试_calculate_redundancy异常处理"""
        assessor = FeatureQualityAssessor()
        features = pd.DataFrame({
            'f1': [1, 2, 3],
            'f2': [4, 5, 6]
        })
        
        result = assessor._calculate_redundancy(features)
        assert isinstance(result, dict)

    def test_identify_redundant_features_empty(self):
        """测试_identify_redundant_features（空DataFrame）"""
        assessor = FeatureQualityAssessor()
        result = assessor._identify_redundant_features(pd.DataFrame())
        assert result == []

    def test_identify_redundant_features_single_column(self):
        """测试_identify_redundant_features（单列）"""
        assessor = FeatureQualityAssessor()
        features = pd.DataFrame({'f1': [1, 2, 3]})
        result = assessor._identify_redundant_features(features)
        assert result == []

    def test_get_quality_report_empty_metrics(self):
        """测试get_quality_report（空metrics）"""
        assessor = FeatureQualityAssessor()
        result = assessor.get_quality_report()
        assert result == {}

    def test_generate_recommendations_empty_metrics_source(self):
        """测试_generate_recommendations（空metrics_source）"""
        assessor = FeatureQualityAssessor()
        result = assessor._generate_recommendations()
        assert result == []

    def test_generate_recommendations_with_dict_metrics(self):
        """测试_generate_recommendations（字典类型metrics）"""
        assessor = FeatureQualityAssessor()
        quality_report = {
            'feature_scores': {
                'f1': {
                    'importance_score': 0.1,
                    'correlation_score': 0.2,
                    'stability_score': 0.3,
                    'information_score': 0.4,
                    'redundancy_score': 0.5,
                    'overall_score': 0.3
                }
            }
        }
        result = assessor._generate_recommendations(quality_report)
        assert isinstance(result, list)

    def test_generate_recommendations_with_quality_metrics(self):
        """测试_generate_recommendations（QualityMetrics实例）"""
        assessor = FeatureQualityAssessor()
        assessor.quality_metrics = {
            'f1': QualityMetrics(
                importance_score=0.1,
                correlation_score=0.2,
                stability_score=0.3,
                information_score=0.4,
                redundancy_score=0.5,
                overall_score=0.3
            )
        }
        result = assessor._generate_recommendations()
        assert isinstance(result, list)

    def test_calculate_correlation_matrix_empty(self):
        """测试_calculate_correlation_matrix（空DataFrame）"""
        assessor = FeatureQualityAssessor()
        result = assessor._calculate_correlation_matrix(pd.DataFrame())
        assert result.empty

    def test_downsample_for_model_large_dataset(self):
        """测试_downsample_for_model（大数据集）"""
        config = AssessmentConfig(max_sample_size=100, random_state=42)
        assessor = FeatureQualityAssessor(config)
        
        # 创建大于max_sample_size的数据集
        features = pd.DataFrame({
            'f1': np.random.randn(500),
            'f2': np.random.randn(500)
        })
        target = pd.Series(np.random.randn(500))
        
        result_features, result_target = assessor._downsample_for_model(features, target)
        assert len(result_features) == 100
        assert len(result_target) == 100

    def test_downsample_for_model_small_dataset(self):
        """测试_downsample_for_model（小数据集）"""
        config = AssessmentConfig(max_sample_size=1000, random_state=42)
        assessor = FeatureQualityAssessor(config)
        
        # 创建小于max_sample_size的数据集
        features = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50)
        })
        target = pd.Series(np.random.randn(50))
        
        result_features, result_target = assessor._downsample_for_model(features, target)
        assert len(result_features) == 50
        assert len(result_target) == 50

    def test_calculate_importance_scores_wrapper(self):
        """测试_calculate_importance_scores（测试辅助方法）"""
        assessor = FeatureQualityAssessor()
        features = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6]})
        target = pd.Series([1, 2, 3])
        
        result = assessor._calculate_importance_scores(features, target)
        assert isinstance(result, dict)

    def test_calculate_stability_scores_wrapper(self):
        """测试_calculate_stability_scores（测试辅助方法）"""
        assessor = FeatureQualityAssessor()
        features = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6]})
        target = pd.Series([1, 2, 3])
        
        result = assessor._calculate_stability_scores(features, target)
        assert isinstance(result, dict)

    def test_calculate_information_scores_wrapper(self):
        """测试_calculate_information_scores（测试辅助方法）"""
        assessor = FeatureQualityAssessor()
        features = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6]})
        target = pd.Series([1, 2, 3])
        
        result = assessor._calculate_information_scores(features, target)
        assert isinstance(result, dict)

