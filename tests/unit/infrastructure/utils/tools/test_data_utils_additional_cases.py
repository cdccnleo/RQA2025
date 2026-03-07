import numpy as np
import pandas as pd
import pytest

from src.infrastructure.utils.tools import data_utils


def test_normalize_data_dataframe_with_non_numeric_columns():
    df = pd.DataFrame(
        {
            "value": [1.0, 3.0, 5.0],
            "category": ["a", "b", "c"],
        }
    )

    normalized, params = data_utils.normalize_data(df, method="standard")

    assert list(normalized["category"]) == ["a", "b", "c"]
    assert "means" in params and "stds" in params
    assert params["means"].shape[0] == 1


def test_normalize_list_minmax_handles_constant_values():
    normalized, params = data_utils.normalize_data([5, 5, 5], method="minmax")

    assert all(np.isnan(normalized))
    assert params["method"] == "minmax"
    np.testing.assert_array_equal(params["min_vals"], np.array([5]))
    np.testing.assert_array_equal(params["max_vals"], np.array([5]))


def test_normalize_data_invalid_input_type():
    with pytest.raises(TypeError):
        data_utils.normalize_data(42, method="standard")


def test_denormalize_data_standard_mismatched_params():
    normalized = np.array([[0.0, 1.0], [1.0, 0.0]])
    params = {"means": np.array([1.0, 2.0, 3.0]), "stds": np.array([0.5, 1.5, 2.0])}

    with pytest.raises(ValueError):
        data_utils.denormalize_data(normalized, params, method="standard")


def test_denormalize_data_mixed_dataframe():
    normalized_df = pd.DataFrame({"a": [0.0, 1.0], "b": [0.5, 0.25]})
    params = {
        "a": {"means": 1.0, "stds": 2.0},
        "b": {"min_vals": 0.0, "max_vals": 4.0},
    }

    restored = data_utils.denormalize_data(normalized_df, params, method="mixed")

    np.testing.assert_allclose(restored["a"].to_numpy(), normalized_df["a"] * 2.0 + 1.0)
    np.testing.assert_allclose(
        restored["b"].to_numpy(), normalized_df["b"] * (4.0 - 0.0) + 0.0
    )


def test_validate_normalize_input_errors():
    with pytest.raises(ValueError):
        data_utils._validate_normalize_input(None)
    with pytest.raises(TypeError):
        data_utils._validate_normalize_input({"unsupported": "type"})


def test_internal_normalize_dataframe_robust():
    df = pd.DataFrame({"metric": [2.0, 4.0, 4.0], "tag": ["x", "y", "z"]})

    normalized, params = data_utils._normalize_dataframe(df, method="robust")

    assert "metric" in params and "tag" in params
    assert "iqrs" in params["metric"]
    assert list(normalized.columns) == ["metric", "tag"]


def test_normalize_dataframe_data_without_numeric_columns():
    df = pd.DataFrame({"category": ["x", "y", "z"]})

    with pytest.raises(ValueError, match="没有数值列"):
        data_utils._normalize_dataframe_data(df, method="standard")


def test_normalize_dataframe_data_minmax_constant_column():
    df = pd.DataFrame({"metric": [3.0, 3.0, 3.0], "label": ["a", "b", "c"]})

    normalized, params = data_utils._normalize_dataframe_data(df, method="minmax")

    assert normalized["metric"].tolist() == [0, 0, 0]
    np.testing.assert_array_equal(params["min_vals"], np.array([3.0]))
    np.testing.assert_array_equal(params["max_vals"], np.array([3.0]))


def test_denormalize_minmax_dimension_mismatch_error():
    normalized = np.array([[0.1, 0.2], [0.3, 0.4]])
    params = {"min_vals": np.array([0.0, 1.0, 2.0]), "max_vals": np.array([1.0, 2.0, 3.0])}

    with pytest.raises(ValueError, match="参数维度不匹配"):
        data_utils.denormalize_data(normalized, params, method="minmax")


def test_normalize_array_standard_empty_returns_empty():
    empty = np.array([], dtype=float)
    normalized, params = data_utils._normalize_array(empty, method="standard")

    assert normalized.size == 0
    np.testing.assert_array_equal(params["means"], np.array([]))

