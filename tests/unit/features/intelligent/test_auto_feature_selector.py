# -*- coding: utf-8 -*-
"""
自动特征选择器测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from src.features.intelligent.auto_feature_selector import AutoFeatureSelector
from src.features.core.config_integration import ConfigScope


class TestAutoFeatureSelector:
    """测试AutoFeatureSelector类"""

    @pytest.fixture
    def sample_data(self):
        """生成示例数据"""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100) * 2,
            'feature3': np.random.randn(100) * 3,
            'feature4': np.random.randn(100) * 0.5,
            'feature5': np.random.randn(100) * 1.5
        })
        y = pd.Series(np.random.randint(0, 2, 100))  # 分类任务
        return X, y

    @pytest.fixture
    def sample_data_regression(self):
        """生成回归任务示例数据"""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100) * 2,
            'feature3': np.random.randn(100) * 3
        })
        y = pd.Series(np.random.randn(100))  # 回归任务
        return X, y

    @pytest.fixture
    def selector(self):
        """创建AutoFeatureSelector实例"""
        with patch('src.features.intelligent.auto_feature_selector.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.get_config.return_value = {}
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            return AutoFeatureSelector(
                strategy="auto",
                task_type="classification",
                max_features=3,
                min_features=2
            )

    def test_init_default(self):
        """测试默认初始化"""
        with patch('src.features.intelligent.auto_feature_selector.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            selector = AutoFeatureSelector()
            assert selector.strategy == "auto"
            assert selector.task_type == "classification"
            assert selector.max_features is None
            assert selector.min_features == 3

    def test_init_with_config(self):
        """测试带配置初始化"""
        with patch('src.features.intelligent.auto_feature_selector.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            selector = AutoFeatureSelector(
                strategy="statistical",
                task_type="regression",
                max_features=5,
                min_features=2,
                cv_folds=3
            )
            assert selector.strategy == "statistical"
            assert selector.task_type == "regression"
            assert selector.max_features == 5
            assert selector.min_features == 2
            assert selector.cv_folds == 3

    def test_select_features_auto_strategy(self, selector, sample_data):
        """测试自动策略特征选择"""
        X, y = sample_data
        X_selected, features, info = selector.select_features(X, y, target_features=3)
        
        assert isinstance(X_selected, pd.DataFrame)
        assert isinstance(features, list)
        assert len(features) <= 3
        assert 'method' in info

    def test_select_features_statistical_strategy(self, sample_data):
        """测试统计策略特征选择"""
        with patch('src.features.intelligent.auto_feature_selector.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            selector = AutoFeatureSelector(strategy="statistical", task_type="classification")
            X, y = sample_data
            X_selected, features, info = selector.select_features(X, y, target_features=3)
            
            assert isinstance(X_selected, pd.DataFrame)
            assert isinstance(features, list)
            assert info['method'] == 'statistical'

    def test_select_features_model_based_strategy(self, sample_data):
        """测试基于模型的策略特征选择"""
        with patch('src.features.intelligent.auto_feature_selector.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            selector = AutoFeatureSelector(strategy="model_based", task_type="classification")
            X, y = sample_data
            X_selected, features, info = selector.select_features(X, y, target_features=3)
            
            assert isinstance(X_selected, pd.DataFrame)
            assert isinstance(features, list)
            assert info['method'] == 'model_based'

    def test_select_features_wrapper_strategy(self, sample_data):
        """测试包装策略特征选择"""
        with patch('src.features.intelligent.auto_feature_selector.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            selector = AutoFeatureSelector(strategy="wrapper", task_type="classification", cv_folds=3)
            X, y = sample_data
            X_selected, features, info = selector.select_features(X, y, target_features=3)
            
            assert isinstance(X_selected, pd.DataFrame)
            assert isinstance(features, list)
            assert info['method'] == 'wrapper'

    def test_select_features_ensemble_strategy(self, sample_data):
        """测试集成策略特征选择"""
        with patch('src.features.intelligent.auto_feature_selector.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            selector = AutoFeatureSelector(strategy="ensemble", task_type="classification")
            X, y = sample_data
            X_selected, features, info = selector.select_features(X, y, target_features=3)
            
            assert isinstance(X_selected, pd.DataFrame)
            assert isinstance(features, list)
            assert info['method'] == 'ensemble'

    def test_select_features_empty_data(self, selector):
        """测试空数据"""
        X = pd.DataFrame()
        y = pd.Series()
        
        with pytest.raises(ValueError, match="输入数据不能为空"):
            selector.select_features(X, y)

    def test_select_features_invalid_strategy(self, sample_data):
        """测试无效策略"""
        with patch('src.features.intelligent.auto_feature_selector.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            selector = AutoFeatureSelector(strategy="invalid")
            X, y = sample_data
            
            with pytest.raises(ValueError, match="不支持的选择策略"):
                selector.select_features(X, y)

    def test_determine_optimal_feature_count(self, selector, sample_data):
        """测试确定最优特征数量"""
        X, y = sample_data
        optimal = selector._determine_optimal_feature_count(X, y)
        
        assert isinstance(optimal, int)
        assert optimal >= selector.min_features
        assert optimal <= X.shape[1]
        if selector.max_features:
            assert optimal <= selector.max_features

    def test_fit(self, selector, sample_data):
        """测试拟合"""
        X, y = sample_data
        selector.fit(X, y)
        
        assert selector.is_fitted is True
        assert len(selector.selected_features) > 0

    def test_transform(self, selector, sample_data):
        """测试转换"""
        X, y = sample_data
        selector.fit(X, y)
        
        X_transformed = selector.transform(X)
        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed.columns) == len(selector.selected_features)

    def test_transform_not_fitted(self, selector, sample_data):
        """测试未拟合时转换"""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="特征选择器尚未拟合"):
            selector.transform(X)

    def test_fit_transform(self, selector, sample_data):
        """测试拟合并转换"""
        X, y = sample_data
        X_transformed = selector.fit_transform(X, y)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert selector.is_fitted is True

    def test_get_feature_importance(self, selector, sample_data):
        """测试获取特征重要性"""
        X, y = sample_data
        selector.fit(X, y)
        
        importance = selector.get_feature_importance()
        assert isinstance(importance, dict)

    def test_save(self, selector, sample_data):
        """测试保存"""
        X, y = sample_data
        selector.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "selector.pkl"
            selector.save(filepath)
            assert filepath.exists()
            
            # 验证文件已创建
            assert filepath.is_file()

    def test_on_config_change(self, selector):
        """测试配置变更处理"""
        selector._on_config_change(ConfigScope.PROCESSING, "max_features", 5)
        assert selector.max_features == 5
        
        selector._on_config_change(ConfigScope.PROCESSING, "min_features", 2)
        assert selector.min_features == 2

    def test_regression_task(self, sample_data_regression):
        """测试回归任务"""
        with patch('src.features.intelligent.auto_feature_selector.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            selector = AutoFeatureSelector(task_type="regression")
            X, y = sample_data_regression
            X_selected, features, info = selector.select_features(X, y, target_features=2)
            
            assert isinstance(X_selected, pd.DataFrame)
            assert isinstance(features, list)

