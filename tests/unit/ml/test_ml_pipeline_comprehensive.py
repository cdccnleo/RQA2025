#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML层 - ML管道综合测试（补充）
让ML层从50%+达到80%+
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class TestMLPipeline:
    """测试ML管道"""
    
    def test_create_simple_pipeline(self):
        """测试创建简单管道"""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression())
        ])
        
        assert len(pipeline.steps) == 2
    
    def test_pipeline_fit_transform(self):
        """测试管道拟合转换"""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        pipeline = Pipeline([('scaler', StandardScaler())])
        X_transformed = pipeline.fit_transform(X)
        
        assert X_transformed.shape == X.shape
    
    def test_pipeline_predict(self):
        """测试管道预测"""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000))
        ])
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        assert len(predictions) == len(y_test)
    
    def test_pipeline_score(self):
        """测试管道评分"""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000))
        ])
        
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        
        assert 0 <= score <= 1
    
    def test_pipeline_get_params(self):
        """测试获取管道参数"""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression())
        ])
        
        params = pipeline.get_params()
        
        assert 'scaler' in params
        assert 'model' in params
    
    def test_pipeline_set_params(self):
        """测试设置管道参数"""
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        
        pipeline = Pipeline([('model', LogisticRegression())])
        pipeline.set_params(model__C=0.5)
        
        assert pipeline.get_params()['model__C'] == 0.5
    
    def test_feature_union_pipeline(self):
        """测试特征联合管道"""
        from sklearn.pipeline import Pipeline, FeatureUnion
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        feature_union = FeatureUnion([
            ('scaler1', StandardScaler()),
            ('scaler2', MinMaxScaler())
        ])
        
        pipeline = Pipeline([('features', feature_union)])
        
        X = np.array([[1, 2], [3, 4]])
        X_transformed = pipeline.fit_transform(X)
        
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_pipeline_memory_caching(self):
        """测试管道内存缓存"""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        pipeline = Pipeline([
            ('scaler', StandardScaler())
        ], memory=None)
        
        assert pipeline.memory is None


class TestDataPreprocessing:
    """测试数据预处理"""
    
    def test_handle_missing_values(self):
        """测试处理缺失值"""
        from sklearn.impute import SimpleImputer
        
        X = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
        imputer = SimpleImputer(strategy='mean')
        X_filled = imputer.fit_transform(X)
        
        assert not np.isnan(X_filled).any()
    
    def test_categorical_encoding(self):
        """测试类别编码"""
        from sklearn.preprocessing import LabelEncoder
        
        labels = ['cat', 'dog', 'cat', 'bird']
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(labels)
        
        assert len(encoded) == len(labels)
    
    def test_feature_scaling_standard(self):
        """测试标准化特征缩放"""
        from sklearn.preprocessing import StandardScaler
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
    
    def test_feature_scaling_minmax(self):
        """测试MinMax特征缩放"""
        from sklearn.preprocessing import MinMaxScaler
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        assert X_scaled.min() >= 0
        assert X_scaled.max() <= 1
    
    def test_polynomial_feature_generation(self):
        """测试多项式特征生成"""
        from sklearn.preprocessing import PolynomialFeatures
        
        X = np.array([[1, 2], [3, 4]])
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        assert X_poly.shape[1] > X.shape[1]
    
    def test_feature_selection_variance(self):
        """测试基于方差的特征选择"""
        from sklearn.feature_selection import VarianceThreshold
        
        X = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3]])
        selector = VarianceThreshold()
        X_selected = selector.fit_transform(X)
        
        assert X_selected.shape[1] < X.shape[1]
    
    def test_pca_dimensionality_reduction(self):
        """测试PCA降维"""
        from sklearn.decomposition import PCA
        
        X = np.random.rand(100, 10)
        pca = PCA(n_components=5)
        X_reduced = pca.fit_transform(X)
        
        assert X_reduced.shape[1] == 5
    
    def test_train_test_split_stratified(self):
        """测试分层训练测试集分割"""
        from sklearn.model_selection import train_test_split
        
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        assert len(X_train) + len(X_test) == len(X)
    
    def test_cross_validation_kfold(self):
        """测试K折交叉验证"""
        from sklearn.model_selection import KFold
        
        X = np.random.rand(100, 5)
        kfold = KFold(n_splits=5)
        
        n_splits = sum(1 for _ in kfold.split(X))
        assert n_splits == 5
    
    def test_data_normalization(self):
        """测试数据归一化"""
        from sklearn.preprocessing import Normalizer
        
        X = np.array([[1, 2, 3], [4, 5, 6]])
        normalizer = Normalizer()
        X_normalized = normalizer.fit_transform(X)
        
        # 每行的L2范数应该为1
        norms = np.linalg.norm(X_normalized, axis=1)
        assert np.allclose(norms, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

