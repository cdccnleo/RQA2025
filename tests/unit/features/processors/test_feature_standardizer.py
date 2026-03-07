import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.features.processors.feature_standardizer import FeatureStandardizer


@pytest.fixture
def numeric_frame():
    index = pd.date_range("2024-03-01", periods=8, freq="1h")
    data = {
        "open": np.linspace(100, 120, num=8),
        "high": np.linspace(101, 121, num=8),
        "low": np.linspace(99, 119, num=8),
        "volume": np.linspace(1_000, 2_000, num=8),
    }
    return pd.DataFrame(data, index=index)


def test_fit_transform_persists_scaler(tmp_path, numeric_frame):
    standardizer = FeatureStandardizer(tmp_path)
    metadata = SimpleNamespace()

    result = standardizer.fit_transform(numeric_frame, metadata=metadata)

    assert standardizer.is_fitted is True
    assert metadata.scaler_path.exists()
    assert set(result.columns) == set(numeric_frame.columns)
    assert pytest.approx(float(result.mean().abs().max()), abs=1e-6) == 0.0


def test_transform_after_fit(tmp_path, numeric_frame):
    standardizer = FeatureStandardizer(tmp_path)
    standardizer.fit_transform(numeric_frame)

    same_data = standardizer.transform(numeric_frame)
    assert pytest.approx(float(same_data.mean().abs().max()), abs=1e-6) == 0.0

    shifted = numeric_frame + 5
    shifted_result = standardizer.transform(shifted)
    assert shifted_result.shape == shifted.shape


def test_inverse_transform_recovers_original(tmp_path, numeric_frame):
    standardizer = FeatureStandardizer(tmp_path)
    standardized = standardizer.fit_transform(numeric_frame)

    recovered = standardizer.inverse_transform(standardized)

    pd.testing.assert_frame_equal(recovered, numeric_frame)


def test_fit_transform_inference_mode(tmp_path, numeric_frame):
    standardizer = FeatureStandardizer(tmp_path)
    standardizer.fit_transform(numeric_frame)

    scaled = standardizer.fit_transform(numeric_frame * 1.5, is_training=False)

    assert scaled.shape == numeric_frame.shape
    assert standardizer.scaler_path.exists()


def test_non_numeric_input_raises(tmp_path):
    standardizer = FeatureStandardizer(tmp_path)
    frame = pd.DataFrame({"text": ["a", "b", "c"]})

    with pytest.raises(ValueError):
        standardizer.fit_transform(frame)


def test_load_scaler_sets_state(tmp_path, numeric_frame):
    standardizer = FeatureStandardizer(tmp_path)
    standardizer.fit_transform(numeric_frame)

    new_standardizer = FeatureStandardizer(tmp_path)
    new_standardizer.load_scaler(new_standardizer.scaler_path)

    assert new_standardizer.is_fitted is True
    transformed = new_standardizer.transform(numeric_frame)
    assert transformed.shape == numeric_frame.shape


def test_partial_fit_updates_scaler(tmp_path, numeric_frame):
    standardizer = FeatureStandardizer(tmp_path)
    standardizer.partial_fit(numeric_frame)

    assert standardizer.is_fitted is True
    transformed = standardizer.transform(numeric_frame)
    assert transformed.shape == numeric_frame.shape


def test_fit_transform_empty_raises(tmp_path):
    standardizer = FeatureStandardizer(tmp_path)
    with pytest.raises(ValueError):
        standardizer.fit_transform(pd.DataFrame())


def test_fit_transform_inference_missing_file(tmp_path, numeric_frame, caplog):
    standardizer = FeatureStandardizer(tmp_path)
    with caplog.at_level("WARNING"):
        result = standardizer.fit_transform(numeric_frame, is_training=False)
    assert result.equals(numeric_frame)
    assert any("标准化器文件未找到" in msg for msg in caplog.messages)


def test_inverse_transform_unfitted_raises(tmp_path):
    standardizer = FeatureStandardizer(tmp_path)
    with pytest.raises(RuntimeError):
        standardizer.inverse_transform(pd.DataFrame({"a": [1, 2, 3]}))


def test_transform_unfitted_raises(tmp_path, numeric_frame):
    standardizer = FeatureStandardizer(tmp_path)
    with pytest.raises(RuntimeError):
        standardizer.transform(numeric_frame)


def test_partial_fit_unsupported_scaler(tmp_path, numeric_frame, monkeypatch, caplog):
    standardizer = FeatureStandardizer(tmp_path, method="standard")

    class NoPartial:
        def partial_fit(self, *args, **kwargs):
            raise AttributeError("no partial")

    monkeypatch.setattr(standardizer, "scaler", NoPartial())

    with pytest.raises(AttributeError):
        standardizer.partial_fit(numeric_frame)


def test_unknown_method_raises(tmp_path):
    with pytest.raises(ValueError):
        FeatureStandardizer(tmp_path, method="unknown")

