from __future__ import annotations

import builtins
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.ml.core.ml_core import MLCore
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


@pytest.fixture
def ml_core_factory(monkeypatch):
    def failing_adapter():
        raise RuntimeError("adapter unavailable")

    monkeypatch.setattr("src.ml.core.ml_core._get_models_adapter", failing_adapter)

    def factory(config: Dict = None) -> MLCore:
        return MLCore(config or {"random_state": 0})

    return factory


def make_dataframe():
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, -0.2, 0.5]})
    y = pd.Series([0, 1, 0])
    return X, y


def test_predict_requires_existing_model(ml_core_factory):
    from src.ml.core.exceptions import ModelNotFoundError

    core = ml_core_factory()
    with pytest.raises(ModelNotFoundError):
        core.predict("missing", np.zeros((2, 2)))


def test_predict_logs_and_raises_on_failure(ml_core_factory):
    core = ml_core_factory()

    class BrokenModel:
        def predict(self, _):
            raise RuntimeError("boom")

    core.models["broken"] = {"model": BrokenModel(), "feature_names": ["f1"]}

    with pytest.raises(RuntimeError):
        core.predict("broken", pd.DataFrame({"f1": [1.0]}))


def test_evaluate_model_returns_metrics(ml_core_factory):
    core = ml_core_factory({"random_state": 0})
    X, y = make_dataframe()
    model_id = core.train_model(X, y, model_type="linear")
    metrics = core.evaluate_model(model_id, X, y)
    assert set(metrics.keys()) == {"mse", "rmse", "mae", "r2"}


def test_create_model_random_forest_merges_defaults(ml_core_factory):
    core = ml_core_factory({"random_state": 123})
    model = core._create_model("rf", {"n_estimators": 10})
    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 10
    assert model.random_state == 123


def test_create_model_xgb_import_error_fallback(monkeypatch, ml_core_factory):
    core = ml_core_factory()
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("simulated missing xgboost")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    model = core._create_model("xgb", {"n_estimators": 5})
    assert isinstance(model, RandomForestRegressor)


def test_create_model_lstm_uses_simple_network(monkeypatch, ml_core_factory):
    core = ml_core_factory()
    sentinel = object()
    monkeypatch.setattr(core, "_create_simple_neural_network", lambda params: sentinel)
    result = core._create_model("lstm", {"hidden_layer_sizes": (8,)})
    assert result is sentinel


def test_create_simple_neural_network_defaults(ml_core_factory):
    core = ml_core_factory({"epochs": 5, "random_state": 7})
    model = core._create_simple_neural_network({"hidden_layer_sizes": (4,)})
    assert model.max_iter == 5
    assert model.random_state == 7


def test_preprocess_features_missing_columns_returns_original_values(ml_core_factory):
    core = ml_core_factory()
    X, _ = make_dataframe()
    data = core._preprocess_features(X, ["f1", "missing"])
    np.testing.assert_array_equal(data, X.values)


def test_create_feature_processor_invalid_type(ml_core_factory):
    core = ml_core_factory()
    with pytest.raises(ValueError):
        core.create_feature_processor("unknown")


def test_fit_and_transform_missing_processor_raise(ml_core_factory):
    core = ml_core_factory()
    with pytest.raises(ValueError):
        core.fit_feature_processor("does_not_exist", np.zeros((1, 1)))
    with pytest.raises(ValueError):
        core.transform_features("does_not_exist", np.zeros((1, 1)))


def test_get_feature_importance_variants(ml_core_factory):
    core = ml_core_factory()
    assert core.get_feature_importance("missing") is None

    class LinearLike:
        coef_ = np.array([0.5, -0.1])

    core.models["coef_model"] = {"model": LinearLike()}
    importance = core.get_feature_importance("coef_model")
    assert importance == {"feature_0": 0.5, "feature_1": 0.1}


def test_cross_validate_with_numpy_inputs(ml_core_factory):
    core = ml_core_factory({"cross_validation_folds": 2})
    X, y = make_dataframe()
    results = core.cross_validate(X.values, y.values, model_type="linear")
    assert "scores" in results
    assert results["folds"] == 2

