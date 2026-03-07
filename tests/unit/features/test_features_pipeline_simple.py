#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程管道简化测试
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from datetime import datetime, timedelta

class TestFeaturesPipelineSimple:
    """特征工程管道简化测试"""

    def test_technical_indicators_calculation(self):
        """测试技术指标计算"""
        # 创建价格数据
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

        # 计算简单移动平均
        def sma(prices, period):
            return np.convolve(prices, np.ones(period), 'valid') / period

        sma_5 = sma(prices, 5)
        expected_sma = np.array([102, 103, 104, 105, 106, 107])

        np.testing.assert_array_almost_equal(sma_5, expected_sma, decimal=1)

        # 计算收益率
        returns = np.diff(prices) / prices[:-1]
        assert len(returns) == len(prices) - 1
        assert returns[0] == 0.01  # 100 -> 101

        print("✅ 技术指标计算测试通过")

    def test_feature_scaling_normalization(self):
        """测试特征缩放和归一化"""
        # 创建测试特征
        features = np.array([[1, 100], [2, 200], [3, 300], [4, 400]])

        # 手动实现标准化
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        standardized = (features - mean) / std

        # 验证标准化结果
        assert np.abs(np.mean(standardized, axis=0)).max() < 0.01, "标准化后均值不为0"
        assert np.abs(np.std(standardized, axis=0) - 1).max() < 0.01, "标准化后标准差不为1"

        # Min-Max归一化
        min_vals = np.min(features, axis=0)
        max_vals = np.max(features, axis=0)
        normalized = (features - min_vals) / (max_vals - min_vals)

        # 验证归一化结果
        assert np.min(normalized, axis=0).max() == 0, "归一化后最小值不为0"
        assert np.max(normalized, axis=0).min() == 1, "归一化后最大值不为1"

        print("✅ 特征缩放和归一化测试通过")

    def test_feature_selection_effectiveness(self):
        """测试特征选择有效性"""
        np.random.seed(42)

        # 创建相关特征
        n_samples = 100
        X = np.random.randn(n_samples, 10)

        # 添加目标变量（与前3个特征相关）
        weights = np.array([0.8, 0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        noise = np.random.randn(n_samples) * 0.1
        y = X.dot(weights) + noise

        # 计算特征与目标的相关性
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr))

        # 验证相关特征的识别
        top_features = np.argsort(correlations)[-3:]  # 相关性最高的3个特征
        expected_top = [0, 1, 2]  # 我们知道前3个特征是最相关的

        # 检查是否正确识别了最重要的特征
        overlap = len(set(top_features) & set(expected_top))
        assert overlap >= 2, f"特征选择准确性不足: {overlap}/3"

        print("✅ 特征选择有效性测试通过")

    def test_feature_quality_assessment(self):
        """测试特征质量评估"""
        # 创建包含质量问题的特征数据
        features = pd.DataFrame({
            'good_feature': [1, 2, 3, 4, 5],
            'constant_feature': [1, 1, 1, 1, 1],  # 常量特征
            'missing_feature': [1, 2, np.nan, 4, 5],  # 包含缺失值
            'outlier_feature': [1, 2, 3, 4, 100]  # 包含异常值
        })

        # 评估特征质量
        quality_scores = {}

        for col in features.columns:
            data = features[col].dropna()

            if len(data) == 0:
                quality_scores[col] = 0.0  # 全缺失
            elif data.std() == 0:
                quality_scores[col] = 0.1  # 常量特征
            else:
                # 计算异常值比例
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()
                outlier_rate = outlier_count / len(data)

                # 计算质量分数
                quality_scores[col] = 1.0 - outlier_rate - (features[col].isna().mean() * 0.5)

        # 验证质量评估结果
        assert quality_scores['good_feature'] > 0.8, "优质特征质量分数过低"
        assert quality_scores['constant_feature'] < 0.2, "常量特征质量分数过高"
        assert quality_scores['missing_feature'] < quality_scores['good_feature'], "缺失特征质量不应高于优质特征"
        assert quality_scores['outlier_feature'] < quality_scores['good_feature'], "异常特征质量不应高于优质特征"

        print("✅ 特征质量评估测试通过")
