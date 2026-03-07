from pathlib import Path

import pandas as pd
import pytest

from src.features.core.feature_saver import FeatureSaver


@pytest.fixture
def sample_df():
    return pd.DataFrame({"price": [100.0, 101.5], "volume": [200, 210]})


def test_save_and_load_parquet(tmp_path, sample_df):
    saver = FeatureSaver(base_path=tmp_path / "outputs")
    output = tmp_path / "features.parquet"

    assert saver.save_features(sample_df, output, format="parquet", metadata={"source": "test"})
    reloaded = saver.load_features(output, format="parquet")
    pd.testing.assert_frame_equal(sample_df, reloaded)

    meta = saver.get_last_metadata()
    assert meta["path"] == str(output)
    assert meta["metadata"]["source"] == "test"


def test_save_versions_csv(tmp_path, sample_df):
    saver = FeatureSaver(base_path=tmp_path / "csv_outputs")
    output = tmp_path / "features.csv"

    assert saver.save_features(sample_df, output, format="csv")
    reloaded = saver.load_features(output, format="csv")
    pd.testing.assert_frame_equal(sample_df, reloaded)


def test_save_pickle_and_metadata(tmp_path, sample_df):
    saver = FeatureSaver(base_path=tmp_path / "pickle_outputs")
    output = tmp_path / "features.pkl"

    assert saver.save_features(sample_df, output, format="pickle")
    reloaded = saver.load_features(output, format="pickle")
    pd.testing.assert_frame_equal(sample_df, reloaded)

    meta_file = saver.metadata_path
    assert meta_file.exists()
    contents = meta_file.read_text(encoding="utf-8")
    assert output.name in contents


def test_save_features_invalid_format(tmp_path, sample_df):
    saver = FeatureSaver(base_path=tmp_path / "invalid")
    assert saver.save_features(sample_df, tmp_path / "feature.txt", format="txt") is False


def test_load_missing_file(tmp_path):
    saver = FeatureSaver(base_path=tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        saver.load_features(tmp_path / "not_exists.parquet")


def test_get_last_metadata_without_save(tmp_path):
    saver = FeatureSaver(base_path=tmp_path / "empty")
    assert saver.get_last_metadata() is None

