#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_importance测试覆盖
测试processors/feature_importance.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sklearn.base import BaseEstimator

from src.features.processors.feature_importance import FeatureImportanceAnalyzer


class MockModel(BaseEstimator):
    """Mock模型用于测试"""
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        return np.random.rand(len(X))
    
    def score(self, X, y=None):
        return 0.5


class TestFeatureImportanceAnalyzer:
    """FeatureImportanceAnalyzer测试"""

    @pytest.fixture
    def sample_model(self):
        """样本模型"""
        return MockModel()

    @pytest.fixture
    def sample_data(self):
        """样本数据"""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        return X, y, feature_names

    def test_initialization(self, sample_model):
        """测试初始化"""
        analyzer = FeatureImportanceAnalyzer(sample_model)
        assert analyzer.model == sample_model
        assert analyzer.importance_scores is None
        assert analyzer.importance_std is None

    def test_calculate_permutation_importance(self, sample_model, sample_data):
        """测试计算排列重要性"""
        X, y, feature_names = sample_data
        analyzer = FeatureImportanceAnalyzer(sample_model)
        # 先训练模型
        analyzer.model.fit(X, y)
        
        # 对于回归问题，使用r2评分
        result = analyzer.calculate_permutation_importance(X, y, feature_names, scoring="r2")
        assert isinstance(result, dict)
        assert len(result) == len(feature_names)
        assert all(name in result for name in feature_names)
        assert analyzer.importance_scores is not None
        assert analyzer.importance_std is not None

    def test_calculate_permutation_importance_mismatch(self, sample_model):
        """测试特征数量不匹配"""
        analyzer = FeatureImportanceAnalyzer(sample_model)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        feature_names = ['feature1', 'feature2', 'feature3']  # 数量不匹配
        
        with pytest.raises(ValueError, match="特征数量与名称数量不匹配"):
            analyzer.calculate_permutation_importance(X, y, feature_names)

    def test_calculate_permutation_importance_with_params(self, sample_model, sample_data):
        """测试计算排列重要性（带参数）"""
        X, y, feature_names = sample_data
        analyzer = FeatureImportanceAnalyzer(sample_model)
        analyzer.model.fit(X, y)
        
        result = analyzer.calculate_permutation_importance(
            X, y, feature_names,
            n_repeats=5,
            scoring="r2",
            random_state=42
        )
        assert isinstance(result, dict)
        assert len(result) == len(feature_names)

    def test_get_top_features(self, sample_model, sample_data):
        """测试获取top特征"""
        X, y, feature_names = sample_data
        analyzer = FeatureImportanceAnalyzer(sample_model)
        analyzer.model.fit(X, y)
        
        # 先计算重要性（使用r2评分）
        analyzer.calculate_permutation_importance(X, y, feature_names, scoring="r2")
        
        top_features = analyzer.get_top_features(top_n=3)
        assert isinstance(top_features, list)
        assert len(top_features) == 3
        assert all(f in feature_names for f in top_features)

    def test_get_top_features_before_calculation(self, sample_model):
        """测试在计算重要性前获取top特征"""
        analyzer = FeatureImportanceAnalyzer(sample_model)
        with pytest.raises(RuntimeError, match="请先计算特征重要性"):
            analyzer.get_top_features()

    def test_get_top_features_default_n(self, sample_model, sample_data):
        """测试获取top特征（默认n=10）"""
        X, y, feature_names = sample_data
        analyzer = FeatureImportanceAnalyzer(sample_model)
        analyzer.model.fit(X, y)
        analyzer.calculate_permutation_importance(X, y, feature_names, scoring="r2")
        
        top_features = analyzer.get_top_features()
        assert isinstance(top_features, list)
        assert len(top_features) <= 10
        assert len(top_features) <= len(feature_names)

    def test_plot_importance(self, sample_model, sample_data):
        """测试绘制重要性图"""
        X, y, feature_names = sample_data
        analyzer = FeatureImportanceAnalyzer(sample_model)
        analyzer.model.fit(X, y)
        analyzer.calculate_permutation_importance(X, y, feature_names, scoring="r2")
        
        fig = analyzer.plot_importance(title="Test Importance")
        assert fig is not None
        # 关闭图形避免显示
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_importance_before_calculation(self, sample_model):
        """测试在计算重要性前绘图"""
        analyzer = FeatureImportanceAnalyzer(sample_model)
        with pytest.raises(RuntimeError, match="请先计算特征重要性"):
            analyzer.plot_importance()

    def test_plot_importance_with_top_n(self, sample_model, sample_data):
        """测试绘制重要性图（指定top_n）"""
        X, y, feature_names = sample_data
        analyzer = FeatureImportanceAnalyzer(sample_model)
        analyzer.model.fit(X, y)
        analyzer.calculate_permutation_importance(X, y, feature_names, scoring="r2")
        
        fig = analyzer.plot_importance(top_n=3, title="Top 3 Features")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_importance_with_figsize(self, sample_model, sample_data):
        """测试绘制重要性图（指定figsize）"""
        X, y, feature_names = sample_data
        analyzer = FeatureImportanceAnalyzer(sample_model)
        analyzer.model.fit(X, y)
        analyzer.calculate_permutation_importance(X, y, feature_names, scoring="r2")
        
        fig = analyzer.plot_importance(figsize=(12, 8))
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.skipif(True, reason="需要安装shap包")
    def test_calculate_shap_values(self, sample_model, sample_data):
        """测试计算SHAP值"""
        X, y, feature_names = sample_data
        analyzer = FeatureImportanceAnalyzer(sample_model)
        analyzer.model.fit(X, y)
        
        try:
            result = analyzer.calculate_shap_values(X, feature_names)
            assert isinstance(result, dict)
            assert len(result) == len(feature_names)
        except ImportError:
            pytest.skip("shap包未安装")

    def test_calculate_shap_values_import_error(self, sample_model, sample_data):
        """测试计算SHAP值（shap未安装）"""
        X, y, feature_names = sample_data
        analyzer = FeatureImportanceAnalyzer(sample_model)
        analyzer.model.fit(X, y)
        
        # 模拟import shap失败
        import sys
        original_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == 'shap':
                raise ImportError("No module named 'shap'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(ImportError, match="请先安装shap包"):
                analyzer.calculate_shap_values(X, feature_names)

    def test_calculate_shap_values_mismatch(self, sample_model):
        """测试计算SHAP值（特征数量不匹配）"""
        analyzer = FeatureImportanceAnalyzer(sample_model)
        X = np.random.randn(100, 5)
        feature_names = ['feature1', 'feature2', 'feature3']
        
        # 先检查特征数量不匹配的错误（在导入shap之前就会检查）
        try:
            with pytest.raises(ValueError, match="特征数量与名称数量不匹配"):
                analyzer.calculate_shap_values(X, feature_names)
        except ImportError:
            # 如果shap不可用，跳过测试
            pytest.skip("需要安装shap库: pip install shap")

    @pytest.mark.skipif(True, reason="需要安装shap包")
    def test_plot_shap_summary(self, sample_model, sample_data):
        """测试绘制SHAP摘要图"""
        X, y, feature_names = sample_data
        analyzer = FeatureImportanceAnalyzer(sample_model)
        analyzer.model.fit(X, y)
        
        try:
            fig = analyzer.plot_shap_summary(X, feature_names)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("shap包未安装")

    def test_plot_shap_summary_import_error(self, sample_model, sample_data):
        """测试绘制SHAP摘要图（shap未安装）"""
        X, y, feature_names = sample_data
        analyzer = FeatureImportanceAnalyzer(sample_model)
        analyzer.model.fit(X, y)
        
        # 模拟import shap失败
        import sys
        original_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == 'shap':
                raise ImportError("No module named 'shap'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(ImportError, match="请先安装shap包"):
                analyzer.plot_shap_summary(X, feature_names)

    def test_permutation_importance_fallback(self, sample_model, sample_data):
        """测试排列重要性降级处理"""
        X, y, feature_names = sample_data
        analyzer = FeatureImportanceAnalyzer(sample_model)
        analyzer.model.fit(X, y)
        
        # 模拟permutation_importance抛出ImportError
        with patch('src.features.processors.feature_importance.permutation_importance', side_effect=ImportError("模拟导入错误")):
            result = analyzer.calculate_permutation_importance(X, y, feature_names, scoring="r2")
            # 降级处理应该返回全零
            assert isinstance(result, dict)
            assert len(result) == len(feature_names)
            # 降级时应该都是0
            assert all(score == 0.0 for score in result.values())

    def test_get_top_features_all_features(self, sample_model, sample_data):
        """测试获取所有特征（top_n大于特征数）"""
        X, y, feature_names = sample_data
        analyzer = FeatureImportanceAnalyzer(sample_model)
        analyzer.model.fit(X, y)
        analyzer.calculate_permutation_importance(X, y, feature_names, scoring="r2")
        
        top_features = analyzer.get_top_features(top_n=100)
        assert isinstance(top_features, list)
        assert len(top_features) == len(feature_names)

