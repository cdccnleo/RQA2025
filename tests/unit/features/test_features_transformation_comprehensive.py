#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Features层 - 特征变换综合测试

测试标准化、归一化、离散化、组合特征、特征编码
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TestFeatureStandardization:
    """测试特征标准化"""
    
    @pytest.fixture
    def raw_features(self):
        """创建原始特征数据"""
        return np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    
    def test_z_score_standardization(self, raw_features):
        """测试Z-score标准化"""
        scaler = StandardScaler()
        standardized = scaler.fit_transform(raw_features)
        
        # 验证均值为0，标准差为1
        assert np.allclose(standardized.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(standardized.std(axis=0), 1, atol=1e-10)
    
    def test_min_max_normalization(self, raw_features):
        """测试Min-Max归一化"""
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(raw_features)
        
        # 验证范围在[0, 1]
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_robust_scaling(self, raw_features):
        """测试鲁棒缩放"""
        # 使用中位数和四分位数进行缩放
        median = np.median(raw_features, axis=0)
        q75 = np.percentile(raw_features, 75, axis=0)
        q25 = np.percentile(raw_features, 25, axis=0)
        iqr = q75 - q25
        
        scaled = (raw_features - median) / iqr
        
        assert scaled.shape == raw_features.shape


class TestFeatureDiscretization:
    """测试特征离散化"""
    
    def test_equal_width_binning(self):
        """测试等宽分箱"""
        data = np.array([1, 5, 10, 15, 20, 25, 30])
        n_bins = 3
        
        bins = np.linspace(data.min(), data.max(), n_bins + 1)
        digitized = np.digitize(data, bins[1:-1])
        
        assert digitized.max() <= n_bins - 1
        assert digitized.min() >= 0
    
    def test_equal_frequency_binning(self):
        """测试等频分箱"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        n_bins = 3
        
        binned = pd.qcut(data, q=n_bins, labels=False)
        
        assert binned.nunique() == n_bins
    
    def test_custom_threshold_binning(self):
        """测试自定义阈值分箱"""
        data = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
        thresholds = [30, 60, 90]
        
        binned = np.digitize(data, thresholds)
        
        assert binned.min() >= 0
        assert binned.max() <= len(thresholds)


class TestFeatureCombination:
    """测试组合特征"""
    
    def test_create_interaction_features(self):
        """测试创建交互特征"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        # 创建交互特征
        df['interaction'] = df['feature1'] * df['feature2']
        
        assert 'interaction' in df.columns
        assert (df['interaction'] == [4, 10, 18]).all()
    
    def test_create_polynomial_features(self):
        """测试创建多项式特征"""
        data = pd.Series([1, 2, 3, 4, 5])
        
        # 创建平方特征
        data_squared = data ** 2
        
        assert len(data_squared) == len(data)
        assert (data_squared == [1, 4, 9, 16, 25]).all()
    
    def test_create_ratio_features(self):
        """测试创建比率特征"""
        df = pd.DataFrame({
            'numerator': [10, 20, 30],
            'denominator': [2, 4, 5]
        })
        
        df['ratio'] = df['numerator'] / df['denominator']
        
        assert (df['ratio'] == [5.0, 5.0, 6.0]).all()
    
    def test_create_aggregation_features(self):
        """测试创建聚合特征"""
        df = pd.DataFrame({
            'f1': [1, 2, 3],
            'f2': [4, 5, 6],
            'f3': [7, 8, 9]
        })
        
        # 创建聚合特征
        df['sum'] = df[['f1', 'f2', 'f3']].sum(axis=1)
        df['mean'] = df[['f1', 'f2', 'f3']].mean(axis=1)
        
        assert (df['sum'] == [12, 15, 18]).all()
        assert (df['mean'] == [4.0, 5.0, 6.0]).all()


class TestFeatureEncoding:
    """测试特征编码"""
    
    def test_one_hot_encoding(self):
        """测试One-Hot编码"""
        categories = pd.Series(['A', 'B', 'C', 'A', 'B'])
        
        # One-Hot编码
        encoded = pd.get_dummies(categories, prefix='cat')
        
        assert encoded.shape == (5, 3)  # 3个类别
        assert encoded.sum(axis=1).eq(1).all()  # 每行只有一个1
    
    def test_label_encoding(self):
        """测试标签编码"""
        categories = pd.Series(['low', 'medium', 'high', 'low', 'high'])
        
        # 创建标签映射
        label_map = {'low': 0, 'medium': 1, 'high': 2}
        encoded = categories.map(label_map)
        
        assert encoded.min() == 0
        assert encoded.max() == 2
    
    def test_ordinal_encoding(self):
        """测试序数编码"""
        # 有序类别
        sizes = pd.Series(['S', 'M', 'L', 'XL', 'M', 'S'])
        size_order = {'S': 1, 'M': 2, 'L': 3, 'XL': 4}
        
        encoded = sizes.map(size_order)
        
        assert (encoded.values == [1, 2, 3, 4, 2, 1]).all()
    
    def test_target_encoding(self):
        """测试目标编码"""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B', 'A'],
            'target': [1, 0, 1, 1, 0, 0]
        })
        
        # 计算每个类别的目标均值
        target_means = df.groupby('category')['target'].mean()
        df['category_encoded'] = df['category'].map(target_means)
        
        assert 'category_encoded' in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

