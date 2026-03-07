import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.ml.core import ml_core
from src.ml.core.ml_core import MLCore, ModelTrainingError
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class FakeAdapter:
    def get_models_cache_manager(self):
        return SimpleNamespace(store={})

    def get_models_config_manager(self):
        return SimpleNamespace(config={})

    def get_models_logger(self):
        import logging

        return logging.getLogger("ml_core.fake")


def test_ml_core_fallback_applies_default_config(monkeypatch):
    monkeypatch.setenv("ML_CORE_FORCE_FALLBACK", "1")
    import importlib

    import src.ml.core.ml_core as ml_core_module

    importlib.reload(ml_core_module)
    core = ml_core_module.MLCore()
    assert core.cache_manager is None
    assert core.config["model_cache_enabled"] is True

    monkeypatch.delenv("ML_CORE_FORCE_FALLBACK", raising=False)
    importlib.reload(ml_core_module)


def test_train_model_handles_validation_error(monkeypatch):
    core = MLCore()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 2])  # mismatched length to trigger validation error

    with pytest.raises(DataValidationError := ml_core.DataValidationError):
        core.train_model(X, y)


def test_train_model_wraps_exceptions(monkeypatch):
    core = MLCore()
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([1, 2, 3])

    class BrokenModel(LinearRegression):
        def fit(self, *args, **kwargs):
            raise RuntimeError("fit boom")

    monkeypatch.setattr(core, "_create_model", lambda *_, **__: BrokenModel())

    with pytest.raises(ModelTrainingError) as exc:
        core.train_model(X, y)
    assert "fit boom" in str(exc.value)

