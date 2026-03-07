import pandas as pd
import pytest
from types import SimpleNamespace

from src.features.processors.general_processor import FeatureProcessor, ProcessorConfig


def _sample_frame():
    return pd.DataFrame(
        {
            "num": [1, None, 3, 1],
            "cat": ["a", "b", None, "a"],
        }
    )


def test_process_features_returns_empty_on_none():
    processor = FeatureProcessor()
    result = processor.process_features(None)
    assert result.empty


def test_process_features_removes_duplicates_and_handles_missing():
    processor = FeatureProcessor()
    config = SimpleNamespace(handle_missing_values=True)
    frame = _sample_frame()
    result = processor.process_features(frame, config=config)
    assert len(result) == 3  # duplicates removed
    assert result["num"].isna().sum() == 0
    assert result["cat"].isna().sum() == 0


def test_process_features_skips_missing_when_disabled():
    processor = FeatureProcessor()
    config = SimpleNamespace(handle_missing_values=False)
    frame = _sample_frame()
    result = processor.process_features(frame, config=config)
    assert result["num"].isna().sum() == 1


def test_handle_missing_values_fallback_on_error(monkeypatch):
    processor = FeatureProcessor()
    frame = _sample_frame()

    def bad_median(self):
        raise ValueError("boom")

    monkeypatch.setattr(pd.Series, "median", bad_median, raising=False)
    result = processor._handle_missing_values(frame)
    # error should return original
    pd.testing.assert_frame_equal(result, frame)


def test_compute_feature_returns_existing_column():
    processor = FeatureProcessor()
    frame = _sample_frame()
    series = processor._compute_feature(frame, "num", {})
    pd.testing.assert_series_equal(series, frame["num"])
    empty = processor._compute_feature(frame, "unknown", {})
    assert empty.dtype == float and empty.isna().all()


def test_get_feature_metadata():
    processor = FeatureProcessor()
    meta = processor._get_feature_metadata("num")
    assert meta["name"] == "num"
    assert meta["type"] == "general_feature"


def test_get_available_features_returns_empty():
    processor = FeatureProcessor()
    assert processor._get_available_features() == []

