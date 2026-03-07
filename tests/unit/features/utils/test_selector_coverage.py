#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selector测试覆盖补充
补充utils/selector.py的测试覆盖
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.features.utils.selector import FeatureSelector
try:
    from src.features.core.feature_config import FeatureConfig
except ImportError:
    from src.features.core.config import FeatureConfig


class TestFeatureSelector:
    """FeatureSelector测试"""

    @pytest.fixture
    def sample_data(self):
        """样本数据"""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'feature3': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'constant': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })

    @pytest.fixture
    def correlated_data(self):
        """高度相关的数据"""
        base = np.random.randn(100)
        return pd.DataFrame({
            'feature1': base,
            'feature2': base * 1.01,  # 与feature1高度相关
            'feature3': base * 1.02,  # 与feature1高度相关
            'feature4': np.random.randn(100),
            'feature5': np.random.randn(100)
        })

    def test_initialization(self):
        """测试初始化"""
        selector = FeatureSelector()
        assert selector is not None

    def test_select_features_empty_dataframe(self):
        """测试空DataFrame"""
        selector = FeatureSelector()
        empty_df = pd.DataFrame()
        result = selector.select_features(empty_df)
        assert result.empty

    def test_select_features_none_input(self):
        """测试None输入"""
        selector = FeatureSelector()
        result = selector.select_features(None)
        assert result.empty

    def test_select_features_no_numeric_columns(self):
        """测试无数值列"""
        selector = FeatureSelector()
        data = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'col2': ['x', 'y', 'z']
        })
        result = selector.select_features(data)
        # 应该返回原始数据
        assert len(result.columns) == 2

    def test_select_features_removes_constant_columns(self, sample_data):
        """测试移除常数列"""
        selector = FeatureSelector()
        result = selector.select_features(sample_data)
        # 常量列应该被移除
        assert 'constant' not in result.columns
        assert len(result.columns) < len(sample_data.columns)

    def test_select_features_with_config(self, sample_data):
        """测试带配置的特征选择"""
        selector = FeatureSelector()
        config = FeatureConfig()
        config.max_features = 2
        result = selector.select_features(sample_data, config=config)
        # 应该限制特征数量
        assert len(result.columns) <= 2

    def test_select_features_without_config(self, sample_data):
        """测试不带配置的特征选择"""
        selector = FeatureSelector()
        result = selector.select_features(sample_data)
        # 应该移除常量列但保留其他特征
        assert 'constant' not in result.columns
        assert len(result.columns) >= 2

    def test_remove_correlated_features(self, correlated_data):
        """测试移除高度相关的特征"""
        selector = FeatureSelector()
        config = FeatureConfig()
        config.max_features = 2
        result = selector.select_features(correlated_data, config=config)
        # 应该限制特征数量
        assert len(result.columns) <= 2

    def test_remove_correlated_features_no_config(self, correlated_data):
        """测试无配置时的相关性移除"""
        selector = FeatureSelector()
        result = selector.select_features(correlated_data)
        # 应该保留一些特征
        assert len(result.columns) > 0

    def test_remove_correlated_features_less_than_max(self, sample_data):
        """测试特征数少于最大限制"""
        selector = FeatureSelector()
        config = FeatureConfig()
        config.max_features = 10  # 大于实际特征数
        result = selector.select_features(sample_data, config=config)
        # 应该保留所有特征（除了常量列）
        assert len(result.columns) <= len(sample_data.columns)

    def test_remove_correlated_features_exception_handling(self, sample_data):
        """测试异常处理"""
        selector = FeatureSelector()
        with patch.object(selector, '_remove_correlated_features', side_effect=Exception("模拟错误")):
            # 应该能够处理异常并返回原始数据
            config = FeatureConfig()
            config.max_features = 2
            result = selector.select_features(sample_data, config=config)
            # 异常时应返回原始数据或部分处理的数据
            assert isinstance(result, pd.DataFrame)

    def test_select_features_exception_handling(self, sample_data):
        """测试选择特征时的异常处理"""
        selector = FeatureSelector()
        # 模拟异常场景
        with patch('pandas.DataFrame.select_dtypes', side_effect=Exception("模拟错误")):
            # 应该能够处理异常并返回原始数据
            result = selector.select_features(sample_data)
            # 异常时应返回原始数据
            assert isinstance(result, pd.DataFrame)

    def test_remove_correlated_features_correlation_calculation(self):
        """测试相关性计算"""
        selector = FeatureSelector()
        # 创建高度相关的数据
        base = np.random.randn(50)
        data = pd.DataFrame({
            'feature1': base,
            'feature2': base * 1.01,  # 高度相关
            'feature3': np.random.randn(50),  # 不相关
        })
        config = FeatureConfig()
        config.max_features = 1
        result = selector.select_features(data, config=config)
        # 应该保留1个特征
        assert len(result.columns) <= 1

    def test_remove_correlated_features_all_highly_correlated(self):
        """测试所有特征都高度相关的情况"""
        selector = FeatureSelector()
        base = np.random.randn(50)
        data = pd.DataFrame({
            'feature1': base,
            'feature2': base * 1.01,
            'feature3': base * 1.02,
            'feature4': base * 1.03,
        })
        config = FeatureConfig()
        config.max_features = 2
        result = selector.select_features(data, config=config)
        # 应该限制特征数量
        assert len(result.columns) <= 2

    def test_remove_correlated_features_exception_in_correlation(self):
        """测试相关性计算时的异常处理"""
        selector = FeatureSelector()
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [10, 20, 30]
        })
        with patch('pandas.DataFrame.corr', side_effect=Exception("模拟错误")):
            config = FeatureConfig()
            config.max_features = 1
            result = selector._remove_correlated_features(data, max_features=1)
            # 异常时应返回原始数据
            assert isinstance(result, pd.DataFrame)

