import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.ml.core.ml_core import MLCore


@pytest.fixture
def ml_core_factory(monkeypatch):
    def failing_adapter():
        raise RuntimeError("adapter unavailable")

    monkeypatch.setattr("src.ml.core.ml_core._get_models_adapter", failing_adapter, raising=False)

    def factory(config=None):
        cfg = {"random_state": 0}
        if config:
            cfg.update(config)
        return MLCore(cfg)

    return factory


def test_module_import_adapter_failure_uses_defaults(monkeypatch):
    import src.ml.core.ml_core as ml_core_module

    original_integration = sys.modules.get("src.core.integration")
    fake_module = types.ModuleType("src.core.integration")

    def failing_get_models_adapter():
        raise RuntimeError("boom")

    fake_module.get_models_adapter = failing_get_models_adapter
    sys.modules["src.core.integration"] = fake_module

    reloaded = importlib.reload(ml_core_module)
    assert reloaded.cache_manager is None
    assert reloaded.logger.name == reloaded.__name__

    if original_integration is not None:
        sys.modules["src.core.integration"] = original_integration
    else:
        sys.modules.pop("src.core.integration", None)
    importlib.reload(ml_core_module)


def test_prepare_helpers_numpy_paths(ml_core_factory):
    core = ml_core_factory()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    processed, names = core._prepare_features(arr, None)
    assert names is None
    assert processed is arr

    target = core._prepare_target(arr[:, 0])
    np.testing.assert_array_equal(target, arr[:, 0])


def test_predict_with_numpy_input_path(monkeypatch, ml_core_factory):
    core = ml_core_factory()

    captured = {}

    class DummyModel:
        def predict(self, data):
            captured["payload"] = data
            return np.sum(data, axis=1)

    core.models["dummy"] = {"model": DummyModel(), "feature_names": None}
    result = core.predict("dummy", np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert captured["payload"].shape == (2, 2)
    np.testing.assert_array_equal(result, np.array([3.0, 7.0]))


def test_evaluate_model_propagates_errors(monkeypatch, ml_core_factory):
    core = ml_core_factory()
    core.models["dummy"] = {"model": object(), "feature_names": None}

    def failing_predict(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(core, "predict", failing_predict)
    with pytest.raises(RuntimeError):
        core.evaluate_model("dummy", np.array([[1.0]]), np.array([1.0]))


def test_create_model_xgb_success(monkeypatch, ml_core_factory):
    fake_xgb = types.ModuleType("xgboost")

    class DummyRegressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_xgb.XGBRegressor = DummyRegressor
    original_xgb = sys.modules.get("xgboost")
    sys.modules["xgboost"] = fake_xgb

    core = ml_core_factory({"random_state": 5})
    model = core._create_model("xgb", {"n_estimators": 10})
    assert isinstance(model, DummyRegressor)
    assert model.kwargs["learning_rate"] == 0.1

    if original_xgb is not None:
        sys.modules["xgboost"] = original_xgb
    else:
        sys.modules.pop("xgboost", None)


def test_delete_model_exception_returns_false(ml_core_factory):
    core = ml_core_factory()

    class BadDict(dict):
        def __delitem__(self, key):
            raise RuntimeError("boom")

    core.models = BadDict({"foo": {"model": object()}})
    assert core.delete_model("foo") is False


def test_save_model_missing_id_returns_false(tmp_path: Path, ml_core_factory):
    core = ml_core_factory()
    path = tmp_path / "model.pkl"
    assert core.save_model("missing", path) is False


def test_create_feature_processor_variants(ml_core_factory):
    core = ml_core_factory()
    robust_id = core.create_feature_processor("robust")
    minmax_id = core.create_feature_processor("minmax")
    assert robust_id in core.feature_processors
    assert minmax_id in core.feature_processors


def test_get_feature_importance_missing_attributes_returns_none(ml_core_factory):
    core = ml_core_factory()
    core.models["plain"] = {"model": object()}
    assert core.get_feature_importance("plain") is None


def test_get_feature_importance_exception_returns_none(ml_core_factory):
    core = ml_core_factory()

    class FaultyModel:
        @property
        def feature_importances_(self):
            raise RuntimeError("bad importance")

    core.models["faulty"] = {"model": FaultyModel()}
    assert core.get_feature_importance("faulty") is None


def test_cross_validate_exception_reraises(monkeypatch, ml_core_factory):
    core = ml_core_factory()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(core, "_create_model", boom)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.0, 1.0])
    with pytest.raises(RuntimeError):
        core.cross_validate(X, y)

