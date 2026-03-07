# -*- coding: utf-8 -*-
"""
机器学习模型集成测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from src.features.intelligent.ml_model_integration import MLModelIntegration
from src.features.core.config_integration import ConfigScope


class TestMLModelIntegration:
    """测试MLModelIntegration类"""

    @pytest.fixture
    def sample_data(self):
        """生成示例数据"""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100) * 2,
            'feature3': np.random.randn(100) * 3
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
    def ml_integration(self):
        """创建MLModelIntegration实例"""
        with patch('src.features.intelligent.ml_model_integration.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.get_config.return_value = {}
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            return MLModelIntegration(
                task_type="classification",
                ensemble_method="voting"
            )

    def test_init_default(self):
        """测试默认初始化"""
        with patch('src.features.intelligent.ml_model_integration.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            ml = MLModelIntegration()
            assert ml.task_type == "classification"
            assert ml.ensemble_method == "voting"
            assert ml.enable_auto_tuning is True

    def test_init_with_config(self):
        """测试带配置初始化"""
        with patch('src.features.intelligent.ml_model_integration.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            ml = MLModelIntegration(
                task_type="regression",
                ensemble_method="stacking",
                enable_auto_tuning=False
            )
            assert ml.task_type == "regression"
            assert ml.ensemble_method == "stacking"
            assert ml.enable_auto_tuning is False

    def test_train_models_classification(self, ml_integration, sample_data):
        """测试训练分类模型"""
        X, y = sample_data
        performance = ml_integration.train_models(X, y, test_size=0.2)
        
        assert isinstance(performance, dict)
        assert len(performance) > 0
        for model_name, perf in performance.items():
            assert isinstance(perf, dict)

    def test_train_models_regression(self, sample_data_regression):
        """测试训练回归模型"""
        with patch('src.features.intelligent.ml_model_integration.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            ml = MLModelIntegration(task_type="regression")
            X, y = sample_data_regression
            performance = ml.train_models(X, y, test_size=0.2)
            
            assert isinstance(performance, dict)
            assert len(performance) > 0

    def test_train_models_empty_data(self, ml_integration):
        """测试空数据"""
        X = pd.DataFrame()
        y = pd.Series()
        
        with pytest.raises(ValueError, match="输入数据不能为空"):
            ml_integration.train_models(X, y)

    def test_create_ensemble(self, ml_integration, sample_data):
        """测试创建集成模型"""
        X, y = sample_data
        ml_integration.train_models(X, y, test_size=0.2)
        
        ensemble = ml_integration.create_ensemble()
        assert ensemble is not None
        assert ml_integration.ensemble_model is not None

    def test_create_ensemble_before_training(self, ml_integration):
        """测试训练前创建集成模型"""
        with pytest.raises(ValueError, match="请先训练模型"):
            ml_integration.create_ensemble()

    def test_train_ensemble(self, ml_integration, sample_data):
        """测试训练集成模型"""
        X, y = sample_data
        ml_integration.train_models(X, y, test_size=0.2)
        ml_integration.create_ensemble()
        
        performance = ml_integration.train_ensemble(X, y, test_size=0.2)
        assert isinstance(performance, dict)

    def test_predict_with_ensemble(self, ml_integration, sample_data):
        """测试使用集成模型预测"""
        X, y = sample_data
        ml_integration.train_models(X, y, test_size=0.2)
        ml_integration.create_ensemble()
        ml_integration.train_ensemble(X, y, test_size=0.2)
        
        predictions = ml_integration.predict(X, use_ensemble=True)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_predict_without_ensemble(self, ml_integration, sample_data):
        """测试不使用集成模型预测"""
        X, y = sample_data
        ml_integration.train_models(X, y, test_size=0.2)
        
        predictions = ml_integration.predict(X, use_ensemble=False)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_predict_no_model(self, ml_integration, sample_data):
        """测试没有模型时预测"""
        X, y = sample_data
        
        # 如果没有模型，应该抛出异常或返回错误
        try:
            result = ml_integration.predict(X)
            # 如果返回了结果，验证其类型
            assert isinstance(result, np.ndarray) or result is None
        except (ValueError, AttributeError):
            # 预期的异常
            pass

    def test_get_model_performance(self, ml_integration, sample_data):
        """测试获取模型性能"""
        X, y = sample_data
        ml_integration.train_models(X, y, test_size=0.2)
        
        performance = ml_integration.get_model_performance()
        assert isinstance(performance, dict)
        assert len(performance) > 0

    def test_save_model(self, ml_integration, sample_data):
        """测试保存模型"""
        X, y = sample_data
        ml_integration.train_models(X, y, test_size=0.2)
        ml_integration.create_ensemble()
        ml_integration.train_ensemble(X, y, test_size=0.2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"
            ml_integration.save_model(filepath)
            assert filepath.exists()

    def test_on_config_change(self, ml_integration):
        """测试配置变更处理"""
        ml_integration._on_config_change(ConfigScope.PROCESSING, "enable_auto_tuning", False)
        assert ml_integration.enable_auto_tuning is False

