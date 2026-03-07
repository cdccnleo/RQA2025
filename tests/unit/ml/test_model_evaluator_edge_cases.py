"""
ML模型评估器边界条件和性能测试
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

try:
    IMPORTS_AVAILABLE = True
    from src.ml.models.model_evaluator import ModelEvaluator
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="ML model imports not available")
class TestModelEvaluatorEdgeCases:
    """测试模型评估器的边界条件"""

    def setup_method(self):
        """测试前准备"""
        self.evaluator = ModelEvaluator()

    def test_get_best_model_empty_models(self):
        """测试get_best_model方法处理空模型字典"""
        result = self.evaluator.get_best_model({}, pd.DataFrame(), pd.Series())
        assert result is None

    def test_get_best_model_invalid_metric(self):
        """测试get_best_model方法处理无效指标"""
        models = {'model1': Mock(is_trained=True)}
        X_test = pd.DataFrame([[1, 2], [3, 4]])
        y_test = pd.Series([0, 1])

        result = self.evaluator.get_best_model(models, X_test, y_test, metric="invalid_metric")
        assert result is None

    def test_get_best_model_normal_case(self):
        """测试get_best_model方法正常情况"""
        models = {
            'model1': Mock(is_trained=True, predict=Mock(return_value=np.array([0.1, 0.9]))),
            'model2': Mock(is_trained=True, predict=Mock(return_value=np.array([0.2, 0.8])))
        }
        X_test = pd.DataFrame([[1, 2], [3, 4]])
        y_test = pd.Series([0, 1])

        result = self.evaluator.get_best_model(models, X_test, y_test, metric="accuracy")
        assert result in ['model1', 'model2']

    def test_performance_monitoring(self):
        """测试性能监控功能"""
        import time

        model = Mock()
        model.is_trained = True
        model.predict.return_value = np.array([0.8, 0.6, 0.7, 0.9])

        X_test = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test = pd.Series([1, 0, 1, 1])

        start_time = time.time()
        metrics = self.evaluator.evaluate_model(model, X_test, y_test, model_name="perf_test")
        execution_time = time.time() - start_time

        # 验证性能
        assert execution_time < 1.0, f"执行时间过长: {execution_time}s"
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        # 验证结果存储
        assert "perf_test" in self.evaluator.evaluation_results
        assert self.evaluator.evaluation_results["perf_test"] == metrics

    def test_model_comparison_edge_cases(self):
        """测试模型比较的边界情况"""
        # 单个模型比较
        models = {'model1': Mock(is_trained=True, predict=Mock(return_value=np.array([0.8, 0.6])))}
        X_test = pd.DataFrame([[1, 2], [3, 4]])
        y_test = pd.Series([1, 0])

        comparison = self.evaluator.compare_models(models, X_test, y_test)
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 1
        assert 'model' in comparison.columns

    def test_evaluate_model_with_different_data_types(self):
        """测试评估不同数据类型的模型"""
        # 测试分类模型
        classification_model = Mock()
        classification_model.is_trained = True
        classification_model.predict.return_value = np.array([0, 1, 1, 0])  # 分类预测

        X_test_cls = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test_cls = pd.Series([0, 1, 1, 0])  # 二分类标签

        metrics_cls = self.evaluator.evaluate_model(classification_model, X_test_cls, y_test_cls, "classification_test")
        assert 'accuracy' in metrics_cls
        assert 'precision' in metrics_cls
        assert 'recall' in metrics_cls
        assert 'f1' in metrics_cls

        # 测试回归模型
        regression_model = Mock()
        regression_model.is_trained = True
        regression_model.predict.return_value = np.array([1.1, 2.2, 2.8, 3.9])  # 回归预测

        X_test_reg = pd.DataFrame([[1], [2], [3], [4]])
        y_test_reg = pd.Series([1.0, 2.0, 3.0, 4.0])  # 回归目标

        metrics_reg = self.evaluator.evaluate_model(regression_model, X_test_reg, y_test_reg, "regression_test")
        assert 'mse' in metrics_reg
        assert 'mae' in metrics_reg
        assert 'r2' in metrics_reg
        assert 'rmse' in metrics_reg

    def test_evaluation_results_persistence(self):
        """测试评估结果的持久性"""
        # 清理之前的评估结果
        self.evaluator.evaluation_results.clear()

        # 执行多次评估
        for i in range(3):
            model = Mock()
            model.is_trained = True
            model.predict.return_value = np.array([0.8, 0.6])

            X_test = pd.DataFrame([[1, 2], [3, 4]])
            y_test = pd.Series([1, 0])

            self.evaluator.evaluate_model(model, X_test, y_test, f"model_{i}")

        # 验证所有结果都被保存
        assert len(self.evaluator.evaluation_results) == 3
        for i in range(3):
            assert f"model_{i}" in self.evaluator.evaluation_results
            assert isinstance(self.evaluator.evaluation_results[f"model_{i}"], dict)
