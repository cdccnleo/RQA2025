import builtins
import os
import pickle
import time
from pathlib import Path

import pytest

from src.features.processors.feature_metadata import FeatureMetadata


def test_init_rejects_invalid_params():
    with pytest.raises(TypeError):
        FeatureMetadata(feature_params=["not", "a", "dict"])  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        FeatureMetadata(feature_list=["dup", "dup"])

    with pytest.raises(ValueError):
        FeatureMetadata(feature_list=["", "valid"])


def test_init_loads_existing_metadata(tmp_path):
    original = FeatureMetadata(
        feature_params={"sma": {"window": 5}},
        feature_list=["sma"],
        scaler_path="scaler.pkl",
        selector_path="selector.pkl",
    )
    persist_path = tmp_path / "meta.pkl"
    original.save_metadata(str(persist_path))

    loaded = FeatureMetadata(metadata_path=str(persist_path))
    assert loaded.feature_params == original.feature_params
    assert loaded.feature_list == original.feature_list
    assert loaded.scaler_path == original.scaler_path
    assert loaded.selector_path == original.selector_path


def test_update_feature_columns_updates_timestamp():
    metadata = FeatureMetadata(feature_list=["old"])
    previous = metadata.last_updated
    time.sleep(0.01)
    metadata.update_feature_columns(["new"])
    assert metadata.feature_list == ["new"]
    assert metadata.last_updated >= previous


def test_update_feature_params_merges_and_updates_timestamp():
    metadata = FeatureMetadata(feature_params={"a": 1})
    previous = metadata.last_updated
    time.sleep(0.01)
    metadata.update_feature_params({"b": 2})
    assert metadata.feature_params == {"a": 1, "b": 2}
    assert metadata.last_updated >= previous


def test_validate_compatibility_checks_feature_lists():
    base = FeatureMetadata(feature_list=["a", "b"])
    same = FeatureMetadata(feature_list=["b", "a"])
    different = FeatureMetadata(feature_list=["c"])

    assert base.validate_compatibility(same) is True
    assert base.validate_compatibility(different) is False
    assert base.validate_compatibility("not metadata") is False  # type: ignore[arg-type]


def test_add_and_remove_feature_updates_params():
    metadata = FeatureMetadata()
    metadata.add_feature("alpha", {"window": 5})
    assert "alpha" in metadata.feature_list
    assert metadata.feature_params["alpha"]["window"] == 5

    metadata.remove_feature("alpha")
    assert "alpha" not in metadata.feature_list
    assert "alpha" not in metadata.feature_params


def test_save_metadata_failure(monkeypatch, tmp_path):
    metadata = FeatureMetadata()

    def fake_open(*_args, **_kwargs):
        raise IOError("disk full")

    monkeypatch.setattr(builtins, "open", fake_open)
    with pytest.raises(IOError):
        metadata.save_metadata(str(tmp_path / "blocked" / "meta.pkl"))


def test_load_metadata_failure(tmp_path):
    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"not a pickle")

    metadata = FeatureMetadata()
    with pytest.raises(Exception):
        metadata.load_metadata(str(corrupt))


def test_get_feature_info_structure():
    metadata = FeatureMetadata(
        feature_params={"sma": {"window": 5}},
        feature_list=["sma"],
        version="2.0",
    )
    info = metadata.get_feature_info()
    assert info["feature_count"] == 1
    assert info["feature_list"] == ["sma"]
    assert info["parameters"]["sma"]["window"] == 5
    assert info["version"] == "2.0"


