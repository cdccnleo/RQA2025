import pytest
from unittest.mock import Mock
import numpy as np
from sklearn.datasets import make_classification
from src.models.evaluation.cross_validator import CrossValidator

class TestCrossValidator:
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
    def mock_model(self):
        model = Mock()
        model.predict.return_value = np.array([0, 1, 0, 1, 2] * 20)
        return model

    def test_k_fold_validation(self, mock_model, classification_data):
        X, y = classification_data
        validator = CrossValidator(mock_model, n_splits=5, random_state=42)
        mean_scores = validator.k_fold_validate(X, y)

        assert isinstance(mean_scores, dict)
        assert 'accuracy' in mean_scores
        assert 0 <= mean_scores['accuracy'] <= 1
        assert len(validator.get_full_results()) == 5

    def test_stratified_validation(self, mock_model, classification_data):
        X, y = classification_data
        validator = CrossValidator(mock_model, n_splits=5, random_state=42)
        mean_scores = validator.stratified_validate(X, y)

        assert isinstance(mean_scores, dict)
        assert 'accuracy' in mean_scores
        assert len(validator.get_full_results()) == 5

    def test_validation_with_custom_metrics(self, mock_model, classification_data):
        X, y = classification_data
        validator = CrossValidator(mock_model)
        mean_scores = validator.k_fold_validate(X, y, metrics=['accuracy', 'precision'])

        assert 'accuracy' in mean_scores
        assert 'precision' in mean_scores
        assert 'recall' not in mean_scores

    def test_get_mean_scores(self, mock_model, classification_data):
        X, y = classification_data
        validator = CrossValidator(mock_model)
        validator.k_fold_validate(X, y)

        mean_scores = validator.get_mean_scores()
        assert isinstance(mean_scores, dict)
        assert all(isinstance(v, float) for v in mean_scores.values())

    def test_get_std_scores(self, mock_model, classification_data):
        X, y = classification_data
        validator = CrossValidator(mock_model)
        validator.k_fold_validate(X, y)

        std_scores = validator.get_std_scores()
        assert isinstance(std_scores, dict)
        assert all(isinstance(v, float) for v in std_scores.values())

    def test_get_results_before_validation(self, mock_model):
        validator = CrossValidator(mock_model)
        with pytest.raises(RuntimeError):
            validator.get_mean_scores()

    def test_invalid_n_splits(self, mock_model):
        with pytest.raises(ValueError):
            CrossValidator(mock_model, n_splits=0)

        with pytest.raises(ValueError):
            CrossValidator(mock_model, n_splits=100)

    def test_random_state_effect(self, mock_model, classification_data):
        X, y = classification_data

        validator1 = CrossValidator(mock_model, random_state=42)
        mean1 = validator1.k_fold_validate(X, y)

        validator2 = CrossValidator(mock_model, random_state=42)
        mean2 = validator2.k_fold_validate(X, y)

        assert mean1 == mean2
