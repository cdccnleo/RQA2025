import builtins
import sys
import types

import numpy as np
import pandas as pd
import pytest

from src.ml.core.ml_core import MLCore
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


@pytest.fixture
def ml_core(monkeypatch):
    def failing_adapter():
        raise RuntimeError("no infra")

    monkeypatch.setattr("src.ml.core.ml_core._get_models_adapter", failing_adapter)
    core = MLCore({"epochs": 5})
    yield core


def test_prepare_features_with_dataframe(ml_core):
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, np.nan],
            "f2": [4.0, 5.0, 6.0],
        }
    )

    processed, feature_names = ml_core._prepare_features(df, None)

    assert feature_names == ["f1", "f2"]
    assert isinstance(processed, np.ndarray)
    # 缺失值应被均值填充
    np.testing.assert_allclose(processed[-1, 0], df["f1"].mean())


def test_prepare_target_returns_numpy_array(ml_core):
    series = pd.Series([1, 2, 3])
    result = ml_core._prepare_target(series)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_save_list_delete_model_workflow(ml_core):
    dummy_model = object()
    model_id = ml_core._save_trained_model(
        dummy_model,
        model_type="linear",
        feature_names=["f1"],
        model_params={"alpha": 0.1},
    )

    assert model_id in ml_core.list_models()
    info = ml_core.get_model_info(model_id)
    assert info["model"] is dummy_model

    assert ml_core.delete_model(model_id) is True
    assert model_id not in ml_core.list_models()
    assert ml_core.get_model_info(model_id) is None


def test_calculate_metrics_returns_expected_keys(ml_core):
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    metrics = ml_core._calculate_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"mse", "rmse", "mae", "r2"}
    assert metrics["mse"] >= 0
    assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))


def test_create_feature_processor_registers_processor(ml_core):
    processor_id = ml_core.create_feature_processor("standard")
    assert processor_id in ml_core.feature_processors
    info = ml_core.feature_processors[processor_id]
    assert info["type"] == "standard"


def test_create_model_random_forest_merges_defaults(monkeypatch, ml_core):
    ml_core.config["random_state"] = 42

    captured = {}

    class DummyRF:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("sklearn.ensemble.RandomForestRegressor", DummyRF)

    model = ml_core._create_model("rf", {"n_estimators": 10})

    assert isinstance(model, DummyRF)
    assert captured["n_estimators"] == 10
    assert captured["random_state"] == 42


def test_create_model_xgb_importerror_fallbacks(monkeypatch, ml_core):
    ml_core.config["random_state"] = 7
    captured = {}

    class DummyRF:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "xgboost":
            raise ImportError("missing xgboost")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("sklearn.ensemble.RandomForestRegressor", DummyRF)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    model = ml_core._create_model("xgb", {"n_estimators": 50})

    assert isinstance(model, DummyRF)
    assert captured["n_estimators"] == 50


def test_create_simple_neural_network_importerror_fallback(monkeypatch, ml_core):
    class DummyLR:
        def __init__(self, *args, **kwargs):
            self.called = True

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sklearn.neural_network":
            raise ImportError("no neural network")
        if name == "sklearn.linear_model" and "LinearRegression" in fromlist:
            module = types.SimpleNamespace(LinearRegression=DummyLR)
            return module
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    model = ml_core._create_simple_neural_network({})
    assert isinstance(model, DummyLR)
    assert getattr(model, "called", False) is True


def test_preprocess_features_handles_exception(monkeypatch, ml_core):
    def boom_fillna(self, *args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(pd.DataFrame, "fillna", boom_fillna, raising=False)
    df = pd.DataFrame({"f1": [1.0, 2.0]})

    result = ml_core._preprocess_features(df, ["f1"])
    np.testing.assert_array_equal(result, df.values)


def test_calculate_metrics_failure_returns_empty_dict(monkeypatch, ml_core):
    def boom(*args, **kwargs):
        raise ValueError("metrics fail")

    monkeypatch.setattr("sklearn.metrics.mean_squared_error", boom)

    result = ml_core._calculate_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert result == {}


def test_save_model_failure_returns_false(monkeypatch, ml_core, tmp_path):
    fake_joblib = types.SimpleNamespace(
        dump=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("dump fail"))
    )
    monkeypatch.setitem(sys.modules, "joblib", fake_joblib)

    model_id = ml_core._save_trained_model(object(), "linear", ["f1"], {})
    assert ml_core.save_model(model_id, str(tmp_path / "model.pkl")) is False


def test_load_model_failure_returns_none(monkeypatch, ml_core):
    fake_joblib = types.SimpleNamespace(
        load=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("load fail"))
    )
    monkeypatch.setitem(sys.modules, "joblib", fake_joblib)

    assert ml_core.load_model("missing.pkl") is None


