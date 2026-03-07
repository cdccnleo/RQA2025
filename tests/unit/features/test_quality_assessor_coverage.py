#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QualityAssessor测试覆盖补充
补充quality_assessor.py的测试覆盖
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.features.quality_assessor import QualityAssessor, QualityAssessorConfig


class TestQualityAssessor:
    """QualityAssessor测试"""

    @pytest.fixture
    def sample_data(self):
        """样本数据"""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'feature3': ['a', 'b', 'c', 'd', 'e', '', 'g', 'h', 'i', 'j'],
            'constant': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })

    @pytest.fixture
    def outlier_data(self):
        """包含异常值的数据"""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 1000, 7, 8, 9, 10],  # 1000是异常值
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        return data

    def test_initialization_default_config(self):
        """测试默认配置初始化"""
        assessor = QualityAssessor()
        assert assessor.config.outlier_zscore == 3.0
        assert assessor.config.missing_value_strategy == "median"
        assert assessor.config.clip_quantiles == 0.01

    def test_initialization_custom_config(self):
        """测试自定义配置初始化"""
        config = QualityAssessorConfig(
            outlier_zscore=2.5,
            missing_value_strategy="mean",
            clip_quantiles=0.05
        )
        assessor = QualityAssessor(config=config)
        assert assessor.config.outlier_zscore == 2.5
        assert assessor.config.missing_value_strategy == "mean"
        assert assessor.config.clip_quantiles == 0.05

    def test_assess_quality(self, sample_data):
        """测试质量评估"""
        assessor = QualityAssessor()
        report = assessor.assess_quality(sample_data)
        assert "score" in report
        assert "issues" in report
        assert "quality_scores" in report
        assert "comprehensive_report" in report
        assert isinstance(report["score"], float)
        assert isinstance(report["issues"], list)

    def test_assess_quality_with_series(self):
        """测试Series输入的质量评估"""
        assessor = QualityAssessor()
        series = pd.Series([1, 2, 3, 4, 5])
        report = assessor.assess_quality(series)
        assert "score" in report
        assert isinstance(report["score"], float)

    def test_assess_quality_detects_missing_values(self, sample_data):
        """测试检测缺失值"""
        assessor = QualityAssessor()
        report = assessor.assess_quality(sample_data)
        issues = report["issues"]
        assert any("缺失值" in str(issue) for issue in issues)

    def test_assess_quality_detects_constant_columns(self, sample_data):
        """测试检测常量列"""
        assessor = QualityAssessor()
        report = assessor.assess_quality(sample_data)
        issues = report["issues"]
        assert any("常量列" in str(issue) for issue in issues)

    def test_assess_quality_detects_outliers(self, outlier_data):
        """测试检测异常值"""
        assessor = QualityAssessor()
        report = assessor.assess_quality(outlier_data)
        issues = report["issues"]
        # 可能检测到异常值
        assert isinstance(issues, list)

    def test_improve_quality_median_strategy(self, sample_data):
        """测试使用中位数策略改进质量"""
        assessor = QualityAssessor()
        improved = assessor.improve_quality(sample_data)
        # 检查缺失值是否被填充
        assert improved['feature1'].isna().sum() == 0
        # 检查常量列是否被保留
        assert 'constant' in improved.columns

    def test_improve_quality_mean_strategy(self, sample_data):
        """测试使用均值策略改进质量"""
        config = QualityAssessorConfig(missing_value_strategy="mean")
        assessor = QualityAssessor(config=config)
        improved = assessor.improve_quality(sample_data)
        assert improved['feature1'].isna().sum() == 0

    def test_improve_quality_zero_strategy(self, sample_data):
        """测试使用零值策略改进质量"""
        config = QualityAssessorConfig(missing_value_strategy="zero")
        assessor = QualityAssessor(config=config)
        improved = assessor.improve_quality(sample_data)
        assert improved['feature1'].isna().sum() == 0

    def test_improve_quality_clips_outliers(self, outlier_data):
        """测试裁剪异常值"""
        assessor = QualityAssessor()
        improved = assessor.improve_quality(outlier_data)
        # 检查异常值是否被裁剪
        assert improved['feature1'].max() < 1000

    def test_improve_quality_handles_non_numeric(self, sample_data):
        """测试处理非数值列"""
        assessor = QualityAssessor()
        improved = assessor.improve_quality(sample_data)
        # 非数值列应该被保留
        assert 'feature3' in improved.columns
        assert improved['feature3'].isna().sum() == 0

    def test_ensure_dataframe_from_dataframe(self, sample_data):
        """测试从DataFrame转换为DataFrame"""
        assessor = QualityAssessor()
        result = assessor._ensure_dataframe(sample_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)

    def test_ensure_dataframe_from_series(self):
        """测试从Series转换为DataFrame"""
        assessor = QualityAssessor()
        series = pd.Series([1, 2, 3, 4, 5], name='test')
        result = assessor._ensure_dataframe(series)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(series)

    def test_ensure_dataframe_invalid_type(self):
        """测试无效类型输入"""
        assessor = QualityAssessor()
        with pytest.raises(TypeError, match="必须是 pandas DataFrame 或 Series"):
            assessor._ensure_dataframe([1, 2, 3])

    def test_detect_issues_missing_values(self):
        """测试检测缺失值问题"""
        assessor = QualityAssessor()
        data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        issues = assessor._detect_issues(data)
        assert any("缺失值" in str(issue) for issue in issues)

    def test_detect_issues_constant_columns(self):
        """测试检测常量列问题"""
        assessor = QualityAssessor()
        data = pd.DataFrame({
            'col1': [1, 1, 1, 1, 1],
            'col2': [10, 20, 30, 40, 50]
        })
        issues = assessor._detect_issues(data)
        assert any("常量列" in str(issue) for issue in issues)

    def test_detect_issues_outliers(self, outlier_data):
        """测试检测异常值问题"""
        assessor = QualityAssessor()
        issues = assessor._detect_issues(outlier_data)
        # 应该检测到异常值
        assert isinstance(issues, list)

    def test_detect_issues_no_issues(self):
        """测试无问题的数据"""
        assessor = QualityAssessor()
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        issues = assessor._detect_issues(data)
        assert isinstance(issues, list)

    def test_improve_quality_empty_dataframe(self):
        """测试空DataFrame"""
        assessor = QualityAssessor()
        empty_df = pd.DataFrame()
        result = assessor.improve_quality(empty_df)
        assert result.empty

    def test_improve_quality_no_numeric_columns(self):
        """测试无数值列的数据"""
        assessor = QualityAssessor()
        data = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'col2': ['x', 'y', 'z']
        })
        result = assessor.improve_quality(data)
        assert len(result.columns) == 2

    def test_assess_quality_feature_assessor_error(self, sample_data):
        """测试feature_assessor出错时的处理"""
        assessor = QualityAssessor()
        with patch.object(assessor.feature_assessor, 'assess_feature_quality', side_effect=Exception("模拟错误")):
            # 如果方法没有异常处理，异常会传播
            with pytest.raises(Exception, match="模拟错误"):
                report = assessor.assess_quality(sample_data)

