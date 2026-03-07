"""
测试AutoML引擎
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from src.ml.deep_learning.automl_engine import (
    AutoMLConfig,
    ModelCandidate,
    AutoMLResult,
    AutoMLEngine,
    ModelSelector,
    HyperparameterOptimizer,
    create_automl_config,
    run_automl
)


class TestAutoMLConfig:
    """测试AutoML配置"""

    def test_automl_config_default_values(self):
        """测试AutoML配置默认值"""
        config = AutoMLConfig()
        assert config.task_type == "classification"
        assert config.time_limit == 3600
        assert config.max_models == 10
        assert config.cv_folds == 5
        assert config.random_state == 42
        assert config.metric == "accuracy"
        assert config.enable_feature_selection == True
        assert config.enable_hyperparameter_tuning == True
        assert config.ensemble_methods == ["voting", "stacking"]

    def test_automl_config_custom_values(self):
        """测试AutoML配置自定义值"""
        config = AutoMLConfig(
            task_type="regression",
            time_limit=1800,
            max_models=5,
            metric="rmse"
        )
        assert config.task_type == "regression"
        assert config.time_limit == 1800
        assert config.max_models == 5
        assert config.metric == "rmse"


class TestModelCandidate:
    """测试模型候选类"""

    def test_model_candidate_creation(self):
        """测试模型候选创建"""
        candidate = ModelCandidate(
            name="test_model",
            model_class=Mock,
            param_space={"param1": [1, 2, 3]}
        )
        assert candidate.name == "test_model"
        assert candidate.model_class == Mock
        assert candidate.param_space == {"param1": [1, 2, 3]}
        assert candidate.preprocessing_steps == []


class TestAutoMLResult:
    """测试AutoML结果"""

    def test_automl_result_creation(self):
        """测试AutoML结果创建"""
        result = AutoMLResult(
            best_model={"name": "best_model"},
            model_candidates=[{"name": "model1"}],
            feature_importance={"feature1": 0.8},
            performance_metrics={"accuracy": 0.95},
            training_time=120.5
        )
        assert result.best_model["name"] == "best_model"
        assert len(result.model_candidates) == 1
        assert result.feature_importance["feature1"] == 0.8
        assert result.performance_metrics["accuracy"] == 0.95
        assert result.training_time == 120.5
        assert isinstance(result.timestamp, datetime)


class TestAutoMLEngine:
    """测试AutoML引擎"""

    def setup_method(self):
        """测试前准备"""
        self.config = {"pipeline": "test", "evaluator": "test", "enable_optuna": True}
        self.engine = AutoMLEngine(self.config)

    @patch('src.ml.deep_learning.automl_engine.get_models_adapter')
    @patch('src.ml.deep_learning.automl_engine.get_automl_pipeline')
    @patch('src.ml.deep_learning.automl_engine.get_evaluator')
    def test_automl_engine_init(self, mock_get_evaluator, mock_get_pipeline, mock_get_adapter):
        """测试AutoML引擎初始化"""
        mock_adapter = Mock()
        mock_logger = Mock()
        mock_adapter.get_models_logger.return_value = mock_logger
        mock_get_adapter.return_value = mock_adapter

        mock_pipeline = Mock()
        mock_get_pipeline.return_value = mock_pipeline

        mock_evaluator = Mock()
        mock_get_evaluator.return_value = mock_evaluator

        engine = AutoMLEngine(self.config)

        assert engine.config == self.config
        assert engine.enable_optuna == True
        assert engine.health == {"runs": 0, "failures": 0}

    @patch('src.ml.deep_learning.automl_engine.get_models_adapter')
    @patch('src.ml.deep_learning.automl_engine.get_automl_pipeline')
    @patch('src.ml.deep_learning.automl_engine.get_evaluator')
    def test_automl_engine_train_success(self, mock_get_evaluator, mock_get_pipeline, mock_get_adapter):
        """测试AutoML引擎训练成功"""
        mock_adapter = Mock()
        mock_logger = Mock()
        mock_adapter.get_models_logger.return_value = mock_logger
        mock_get_adapter.return_value = mock_adapter

        mock_pipeline = Mock()
        mock_pipeline.fit.return_value = {"accuracy": 0.9}
        mock_get_pipeline.return_value = mock_pipeline

        mock_evaluator = Mock()
        mock_get_evaluator.return_value = mock_evaluator

        engine = AutoMLEngine(self.config)
        data = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
        result = engine.train(data, "target")

        assert result == {"accuracy": 0.9}
        assert engine.health["runs"] == 1
        assert engine.health["failures"] == 0

    @patch('src.ml.deep_learning.automl_engine.get_models_adapter')
    @patch('src.ml.deep_learning.automl_engine.get_automl_pipeline')
    @patch('src.ml.deep_learning.automl_engine.get_evaluator')
    def test_automl_engine_train_failure(self, mock_get_evaluator, mock_get_pipeline, mock_get_adapter):
        """测试AutoML引擎训练失败"""
        mock_adapter = Mock()
        mock_logger = Mock()
        mock_adapter.get_models_logger.return_value = mock_logger
        mock_get_adapter.return_value = mock_adapter

        mock_pipeline = Mock()
        mock_pipeline.fit.side_effect = Exception("Training failed")
        mock_get_pipeline.return_value = mock_pipeline

        mock_evaluator = Mock()
        mock_get_evaluator.return_value = mock_evaluator

        engine = AutoMLEngine(self.config)
        data = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})

        with pytest.raises(Exception, match="Training failed"):
            engine.train(data, "target")

        assert engine.health["runs"] == 0
        assert engine.health["failures"] == 1

    @patch('src.ml.deep_learning.automl_engine.get_models_adapter')
    @patch('src.ml.deep_learning.automl_engine.get_automl_pipeline')
    @patch('src.ml.deep_learning.automl_engine.get_evaluator')
    def test_automl_engine_evaluate(self, mock_get_evaluator, mock_get_pipeline, mock_get_adapter):
        """测试AutoML引擎评估"""
        mock_adapter = Mock()
        mock_logger = Mock()
        mock_adapter.get_models_logger.return_value = mock_logger
        mock_get_adapter.return_value = mock_adapter

        mock_pipeline = Mock()
        mock_get_pipeline.return_value = mock_pipeline

        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {"accuracy": 0.85}
        mock_get_evaluator.return_value = mock_evaluator

        engine = AutoMLEngine(self.config)
        engine.health = {"runs": 5, "failures": 1}

        predictions = [0, 1, 0, 1]
        actual = [0, 1, 1, 1]
        result = engine.evaluate(predictions, actual)

        assert result["accuracy"] == 0.85
        assert result["runs"] == 5
        assert result["failures"] == 1


class TestModelSelector:
    """测试模型选择器"""

    def test_model_selector_init(self):
        """测试模型选择器初始化"""
        config = {"method": "accuracy"}
        selector = ModelSelector(config)
        assert selector.config == config

    def test_model_selector_select_best_model(self):
        """测试选择最佳模型"""
        selector = ModelSelector()
        candidates = [
            ModelCandidate(name="model1", model_class=Mock, param_space={}),
            ModelCandidate(name="model2", model_class=Mock, param_space={})
        ]

        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        result = selector.select_best_model(candidates, X, y)
        assert result == candidates[0]

    def test_model_selector_select_best_model_empty(self):
        """测试选择最佳模型（空候选列表）"""
        selector = ModelSelector()
        result = selector.select_best_model([], None, None)
        assert result is None

    def test_model_selector_rank_models(self):
        """测试模型排序"""
        selector = ModelSelector()
        candidates = [
            ModelCandidate(name="model1", model_class=Mock, param_space={}),
            ModelCandidate(name="model2", model_class=Mock, param_space={})
        ]

        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        result = selector.rank_models(candidates, X, y)
        assert result == candidates


class TestHyperparameterOptimizer:
    """测试超参数优化器"""

    def test_hyperparameter_optimizer_init(self):
        """测试超参数优化器初始化"""
        config = {"method": "grid"}
        optimizer = HyperparameterOptimizer(config)
        assert optimizer.config == config

    def test_hyperparameter_optimizer_optimize(self):
        """测试超参数优化"""
        optimizer = HyperparameterOptimizer()
        model_class = Mock
        param_space = {"param1": [1, 2, 3]}

        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        result = optimizer.optimize(model_class, param_space, X, y)
        assert result == {}


class TestCreateAutoMLConfig:
    """测试创建AutoML配置函数"""

    def test_create_automl_config_default(self):
        """测试创建默认AutoML配置"""
        config = create_automl_config()
        assert isinstance(config, AutoMLConfig)
        assert config.task_type == "classification"
        assert config.time_limit == 3600
        assert config.max_models == 10

    def test_create_automl_config_custom(self):
        """测试创建自定义AutoML配置"""
        config = create_automl_config(
            task_type="regression",
            time_limit=1800,
            max_models=5
        )
        assert config.task_type == "regression"
        assert config.time_limit == 1800
        assert config.max_models == 5


class TestRunAutoML:
    """测试运行AutoML函数"""

    def test_run_automl_default_config(self):
        """测试运行AutoML（默认配置）"""
        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})
        y = pd.Series([0, 1, 0])

        result = run_automl(X, y)
        assert isinstance(result, AutoMLResult)
        assert result.best_model["name"] == "dummy_model"
        assert len(result.model_candidates) == 1
        assert result.performance_metrics["accuracy"] == 0.5

    def test_run_automl_custom_config(self):
        """测试运行AutoML（自定义配置）"""
        X = pd.DataFrame({"feature1": [1, 2], "feature2": [0.1, 0.2]})
        y = pd.Series([0, 1])

        config = AutoMLConfig(task_type="regression", time_limit=7200)
        result = run_automl(X, y, config=config, task_type="regression")

        assert isinstance(result, AutoMLResult)
        assert result.training_time >= 0
