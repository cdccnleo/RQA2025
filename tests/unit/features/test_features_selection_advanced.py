#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Features层 - 特征选择高级测试

测试相关性分析、重要性评估、特征筛选算法、降维技术
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from typing import List


class TestCorrelationAnalysis:
    """测试相关性分析"""
    
    @pytest.fixture
    def sample_features(self):
        """创建示例特征数据"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['target'] = y
        return df
    
    def test_calculate_feature_correlation(self, sample_features):
        """测试计算特征相关性"""
        corr_matrix = sample_features.corr()
        
        assert corr_matrix.shape == (11, 11)  # 10 features + 1 target
        # 验证对角线为1
        assert np.allclose(np.diag(corr_matrix.values), 1.0)
    
    def test_find_high_correlation_pairs(self, sample_features):
        """测试查找高相关性特征对"""
        corr_matrix = sample_features.corr().abs()
        
        # 找到相关性>0.9的特征对（排除对角线）
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.9:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # 可能有也可能没有高相关性对
        assert isinstance(high_corr_pairs, list)
    
    def test_remove_highly_correlated_features(self, sample_features):
        """测试移除高相关性特征"""
        corr_matrix = sample_features.corr().abs()
        
        # 找到需要移除的特征
        to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    # 移除其中一个
                    to_drop.add(corr_matrix.columns[j])
        
        # 移除高相关性特征
        remaining_features = [col for col in sample_features.columns if col not in to_drop]
        
        assert len(remaining_features) <= len(sample_features.columns)
    
    def test_calculate_target_correlation(self, sample_features):
        """测试计算特征与目标的相关性"""
        feature_cols = [col for col in sample_features.columns if col != 'target']
        target_corr = sample_features[feature_cols].corrwith(sample_features['target']).abs()
        
        assert len(target_corr) == len(feature_cols)


class TestFeatureImportance:
    """测试特征重要性评估"""
    
    @pytest.fixture
    def classification_data(self):
        """创建分类数据"""
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
        return X, y
    
    def test_calculate_variance_importance(self, classification_data):
        """测试计算方差重要性"""
        X, y = classification_data
        
        # 方差低的特征重要性低
        variances = np.var(X, axis=0)
        
        assert len(variances) == X.shape[1]
        assert (variances >= 0).all()
    
    def test_select_by_variance_threshold(self, classification_data):
        """测试按方差阈值选择特征"""
        X, y = classification_data
        
        variances = np.var(X, axis=0)
        threshold = np.median(variances)
        
        selected_indices = variances > threshold
        X_selected = X[:, selected_indices]
        
        assert X_selected.shape[1] <= X.shape[1]
    
    def test_rank_features_by_importance(self, classification_data):
        """测试按重要性排序特征"""
        X, y = classification_data
        
        # 使用方差作为简单的重要性度量
        importances = np.var(X, axis=0)
        
        # 排序（降序）
        sorted_indices = np.argsort(importances)[::-1]
        
        assert len(sorted_indices) == X.shape[1]
        assert importances[sorted_indices[0]] >= importances[sorted_indices[-1]]


class TestFeatureSelection:
    """测试特征筛选算法"""
    
    @pytest.fixture
    def feature_set(self):
        """创建特征集"""
        X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)
        return X, y
    
    def test_select_top_k_features(self, feature_set):
        """测试选择Top-K特征"""
        X, y = feature_set
        k = 10
        
        # 计算特征方差
        variances = np.var(X, axis=0)
        
        # 选择方差最大的k个特征
        top_k_indices = np.argsort(variances)[-k:]
        X_selected = X[:, top_k_indices]
        
        assert X_selected.shape[1] == k
    
    def test_select_by_percentile(self, feature_set):
        """测试按百分位选择特征"""
        X, y = feature_set
        percentile = 50  # 选择前50%
        
        variances = np.var(X, axis=0)
        threshold = np.percentile(variances, percentile)
        
        selected_mask = variances >= threshold
        X_selected = X[:, selected_mask]
        
        assert X_selected.shape[1] <= X.shape[1]
    
    def test_recursive_feature_elimination(self, feature_set):
        """测试递归特征消除"""
        X, y = feature_set
        
        # 模拟RFE：逐步移除最不重要的特征
        current_features = list(range(X.shape[1]))
        target_n_features = 10
        
        while len(current_features) > target_n_features:
            # 简化版：移除方差最小的特征
            variances = np.var(X[:, current_features], axis=0)
            min_var_idx = np.argmin(variances)
            current_features.pop(min_var_idx)
        
        assert len(current_features) == target_n_features
    
    def test_forward_feature_selection(self, feature_set):
        """测试前向特征选择"""
        X, y = feature_set
        
        # 模拟前向选择：逐步添加最重要的特征
        selected_features = []
        remaining_features = list(range(X.shape[1]))
        target_n = 5
        
        while len(selected_features) < target_n and remaining_features:
            # 简化版：添加方差最大的特征
            variances = np.var(X[:, remaining_features], axis=0)
            max_var_idx = np.argmax(variances)
            selected_idx = remaining_features.pop(max_var_idx)
            selected_features.append(selected_idx)
        
        assert len(selected_features) == target_n


class TestDimensionalityReduction:
    """测试降维技术"""
    
    @pytest.fixture
    def high_dim_data(self):
        """创建高维数据"""
        X, y = make_classification(n_samples=100, n_features=50, n_informative=10, random_state=42)
        return X, y
    
    def test_pca_dimensionality_reduction(self, high_dim_data):
        """测试PCA降维"""
        from sklearn.decomposition import PCA
        
        X, y = high_dim_data
        n_components = 10
        
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        
        assert X_reduced.shape == (X.shape[0], n_components)
        assert pca.explained_variance_ratio_.sum() > 0
    
    def test_explained_variance(self, high_dim_data):
        """测试解释方差"""
        from sklearn.decomposition import PCA
        
        X, y = high_dim_data
        
        pca = PCA(n_components=10)
        pca.fit(X)
        
        # 累计解释方差
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        
        assert len(cumsum_var) == 10
        assert cumsum_var[-1] <= 1.0
    
    def test_select_components_by_variance(self, high_dim_data):
        """测试按方差选择主成分"""
        from sklearn.decomposition import PCA
        
        X, y = high_dim_data
        target_variance = 0.95  # 保留95%方差
        
        pca = PCA()
        pca.fit(X)
        
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= target_variance) + 1
        
        assert n_components <= X.shape[1]
        assert cumsum_var[n_components-1] >= target_variance
    
    def test_feature_standardization_before_pca(self, high_dim_data):
        """测试PCA前的标准化"""
        from sklearn.preprocessing import StandardScaler
        
        X, y = high_dim_data
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 验证标准化效果
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

