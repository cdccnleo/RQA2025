import pytest
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.ml.core import exceptions


def test_model_training_error_includes_model_type():
    err = exceptions.ModelTrainingError("loss diverged", model_type="xgb")
    assert "xgb" in str(err)
    assert err.model_type == "xgb"


def test_handle_ml_exception_wraps_generic():
    @exceptions.handle_ml_exception
    def boom():
        raise ValueError("generic failure")

    with pytest.raises(exceptions.MLException) as exc:
        boom()
    assert "意外错误" in str(exc.value)


def test_handle_ml_exception_passthrough():
    @exceptions.handle_ml_exception
    def boom():
        raise exceptions.ModelPredictionError("fail", model_id="m1")

    with pytest.raises(exceptions.ModelPredictionError):
        boom()


def test_validate_data_shape_length_mismatch():
    with pytest.raises(exceptions.DataValidationError):
        exceptions.validate_data_shape([[1, 2]], [1, 2])


def test_validate_model_exists_missing():
    with pytest.raises(exceptions.ModelNotFoundError):
        exceptions.validate_model_exists({}, "missing")


def test_configuration_error_includes_key():
    err = exceptions.ConfigurationError("invalid", config_key="lr")
    assert "lr" in str(err)
    assert err.config_key == "lr"


def test_resource_exhaustion_error_records_resource():
    err = exceptions.ResourceExhaustionError("oom", resource_type="gpu")
    assert "gpu" in str(err)
    assert err.resource_type == "gpu"


def test_validate_data_shape_expected_features():
    import numpy as np

    X = np.array([[1, 2, 3]])
    with pytest.raises(exceptions.DataValidationError):
        exceptions.validate_data_shape(X, expected_features=2)

