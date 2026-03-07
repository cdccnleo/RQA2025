import numpy as np
import pandas as pd
import pytest

from src.ml.ensemble.model_ensemble import (
    AverageEnsemble,
    EnsembleMethod,
    EnsembleMonitor,
    EnsembleResult,
    ModelEnsemble,
    WeightUpdateRule,
    WeightedEnsemble,
)


class DummyModel:
    def __init__(self, prediction):
        self.prediction = np.asarray(prediction, dtype=np.float64)

    def predict(self, X):
        return np.tile(self.prediction, (len(X), 1)) if self.prediction.ndim > 1 else np.copy(self.prediction)


def test_model_ensemble_base_warns(monkeypatch):
    warnings = []
    monkeypatch.setattr("warnings.warn", lambda msg: warnings.append(msg))
    ensemble = ModelEnsemble.__new__(ModelEnsemble)
    ensemble.models = {"m1": DummyModel([0.2])}
    result = ModelEnsemble.predict(ensemble, np.array([[0.0]]))
    assert result is None
    assert warnings


def test_model_ensemble_init_requires_models():
    with pytest.raises(ValueError):
        ModelEnsemble({})


def test_average_ensemble_prediction():
    models = {
        "m1": DummyModel(np.array([0.2, 0.8])),
        "m2": DummyModel(np.array([0.6, 0.4])),
    }
    ensemble = AverageEnsemble(models)
    X = np.array([[0], [1]])
    result = ensemble.predict(X)

    assert isinstance(result, EnsembleResult)
    np.testing.assert_allclose(result.prediction, np.array([0.4, 0.6]))
    for weight in result.model_weights.values():
        assert weight == pytest.approx(0.5)


def test_weighted_ensemble_equal_weights():
    models = {
        "m1": DummyModel(np.array([0.2, 0.8])),
        "m2": DummyModel(np.array([0.6, 0.4])),
    }
    ensemble = WeightedEnsemble(models, update_rule=WeightUpdateRule.EQUAL)
    result = ensemble.predict(np.array([[0], [1]]))

    np.testing.assert_allclose(result.prediction, np.array([0.4, 0.6]))
    assert result.uncertainty is not None


def test_weighted_ensemble_performance_weights():
    models = {
        "good": DummyModel(np.array([0.1, 0.1])),
        "bad": DummyModel(np.array([0.9, 0.9])),
    }
    ensemble = WeightedEnsemble(models, update_rule=WeightUpdateRule.PERFORMANCE)
    y = np.array([0, 0])
    result = ensemble.predict(np.array([[0], [1]]), y=y)

    assert result.model_weights["good"] > result.model_weights["bad"]
    assert result.performance_metrics["accuracy"] == pytest.approx(1.0)


def test_weighted_ensemble_multiclass_accuracy():
    class MatrixModel:
        def __init__(self, matrix):
            self.matrix = np.asarray(matrix, dtype=np.float64)

        def predict(self, X):
            return self.matrix

    models = {
        "m1": MatrixModel([[0.9, 0.1], [0.2, 0.8]]),
        "m2": MatrixModel([[0.8, 0.2], [0.1, 0.9]]),
    }
    ensemble = WeightedEnsemble(models, update_rule=WeightUpdateRule.EQUAL)
    X = np.array([[0], [1]])
    y = np.array([0, 1])

    result = ensemble.predict(X, y=y)

    assert result.performance_metrics["accuracy"] == pytest.approx(1.0)
    for weight in result.model_weights.values():
        assert weight == pytest.approx(0.5)


def test_ensemble_monitor_update_and_summary():
    predictions = {
        "m1": np.array([0.1, 0.2]),
        "m2": np.array([0.3, 0.4]),
    }
    y = np.array([0.0, 0.0])
    ensemble_pred = np.array([0.2, 0.3])

    monitor = EnsembleMonitor(["m1", "m2"])
    monitor.update(predictions, y, ensemble_pred)
    summary = monitor.get_summary()

    assert "model_performance" in summary
    assert summary["ensemble_performance"]["mean"] >= 0
    assert summary["correlation_matrix"].shape == (2, 2)

