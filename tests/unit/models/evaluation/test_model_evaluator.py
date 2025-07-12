import pytest
from unittest.mock import Mock
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.models.evaluation.model_evaluator import ModelEvaluator

class TestModelEvaluator:
    @pytest.fixture
    def classification_data(self):
        X, y = make_classification(
            n_samples=100,
            n_classes=3,
            n_features=5,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def regression_data(self):
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def mock_classification_model(self):
        model = Mock()
        model.predict.return_value = np.array([0, 1, 0, 1, 2] * 20)
        return model

    @pytest.fixture
    def mock_regression_model(self):
        model = Mock()
        model.predict.return_value = np.linspace(0, 10, 100)
        return model

    def test_classification_evaluation(self, mock_classification_model, classification_data):
        X, y = classification_data
        y = np.array([0, 1, 0, 1, 2] * 20)  # 保持与mock预测一致

        evaluator = ModelEvaluator(mock_classification_model, 'classification')
        metrics = evaluator.evaluate(X, y)

        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

    def test_regression_evaluation(self, mock_regression_model, regression_data):
        X, y = regression_data
        evaluator = ModelEvaluator(mock_regression_model, 'regression')
        metrics = evaluator.evaluate(X, y)

        assert 'mse' in metrics
        assert metrics['mse'] >= 0
        assert 'rmse' in metrics
        assert metrics['rmse'] >= 0

    def test_invalid_task_type(self):
        with pytest.raises(ValueError):
            ModelEvaluator(Mock(), 'invalid_type')

    def test_get_metrics_before_evaluation(self):
        evaluator = ModelEvaluator(Mock())
        with pytest.raises(RuntimeError):
            evaluator.get_metrics()

    def test_plot_confusion_matrix_on_regression(self, mock_regression_model):
        evaluator = ModelEvaluator(mock_regression_model, 'regression')
        with pytest.raises(RuntimeError):
            evaluator.plot_confusion_matrix(np.array([0,1]), np.array([0,1]))

    def test_report_generation(self, mock_classification_model, classification_data):
        X, y = classification_data
        evaluator = ModelEvaluator(mock_classification_model, 'classification')
        evaluator.evaluate(X, y)
        report = evaluator.generate_report()

        assert "Model Evaluation Report" in report
        assert "Task Type: classification" in report
