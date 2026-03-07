#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FeatureProcessor unit tests covering core processing paths."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.processors.feature_processor import FeatureProcessor

pytestmark = pytest.mark.features


@pytest.fixture()
def price_frame():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    return pd.DataFrame(
        {
            "open": pd.Series(range(40), index=idx).astype(float) + 100,
            "high": pd.Series(range(40), index=idx).astype(float) + 101,
            "low": pd.Series(range(40), index=idx).astype(float) + 99,
            "close": pd.Series(range(40), index=idx).astype(float) + 100.5,
            "volume": pd.Series(range(1, 41), index=idx).astype(float) * 100,
        },
        index=idx,
    )


def test_process_generates_requested_features(price_frame: pd.DataFrame):
    processor = FeatureProcessor()
    result = processor.process(price_frame, features=["sma", "ema", "macd"])
    assert "feature_sma" in result.columns
    assert "feature_ema" in result.columns
    assert "feature_macd" in result.columns
    assert result["feature_sma"].notna().sum() > 0


def test_process_defaults_all_features(price_frame: pd.DataFrame):
    processor = FeatureProcessor()
    result = processor.process(price_frame)
    expected = {f"feature_{name}" for name in processor._available_features}
    assert expected.issubset(set(result.columns))


def test_process_with_unknown_feature_raises(price_frame: pd.DataFrame):
    processor = FeatureProcessor()
    with pytest.raises(ValueError):
        processor.process(price_frame, features=["unknown_indicator"])


def test_compute_feature_handles_missing_required_columns():
    processor = FeatureProcessor()
    data = pd.DataFrame({"volume": [100, 110]}, index=pd.date_range("2024-01-01", periods=2))
    values = processor._compute_feature(data, "sma", params={"period": 5})
    assert values.isna().all()


def test_update_config_merges_feature_params(price_frame: pd.DataFrame):
    processor = FeatureProcessor()
    processor.update_config({"period": 10})
    assert processor.config.feature_params["period"] == 10
    result = processor.process(price_frame, features=["price_change"])
    assert "feature_price_change" in result.columns


def test_get_feature_metadata_contains_defaults():
    processor = FeatureProcessor()
    metadata = processor._get_feature_metadata("macd")
    assert metadata["name"] == "macd"
    assert metadata["parameters"]["fast"] == processor.config.feature_params.get("macd_fast", 12)


def test_process_raises_on_empty_dataframe():
    processor = FeatureProcessor()
    with pytest.raises(ValueError):
        processor.process(pd.DataFrame())


def test_moving_average_and_indicator_helpers(price_frame: pd.DataFrame):
    processor = FeatureProcessor()

    ma = processor._calculate_moving_averages(price_frame)
    assert "SMA_5" in ma.columns and "EMA_5" in ma.columns

    rsi = processor._calculate_rsi(price_frame, period=5)
    assert "RSI" in rsi.columns

    macd = processor._calculate_macd(price_frame)
    assert {"MACD", "MACD_Signal", "MACD_Histogram"}.issubset(macd.columns)

    bb = processor._calculate_bollinger_bands(price_frame)
    assert {"BB_Middle", "BB_Upper", "BB_Lower", "BB_Width"}.issubset(bb.columns)


def test_process_data_and_validation(price_frame: pd.DataFrame):
    processor = FeatureProcessor()
    processed = processor.process_data(price_frame, features=["sma", "volatility"])
    assert "sma" in processed.columns
    assert "volatility" in processed.columns

    is_valid, errors = processor.validate_features(["sma", "unknown"])
    assert is_valid is False
    assert errors and "unknown" in errors[0]

    feature_list = processor.list_features()
    assert any(item["name"] == "sma" for item in feature_list)

    summary = processor.get_feature_summary()
    assert summary["total_features"] == len(processor._available_features)


def test_compute_feature_variants(price_frame: pd.DataFrame):
    processor = FeatureProcessor()
    feature_specs = [
        ("ema", {"period": 10}),
        ("rsi", {"period": 7}),
        ("macd", {"fast": 5, "slow": 12, "signal": 3}),
        ("bollinger_bands", {"period": 10, "std_dev": 2}),
        ("price_change", {"period": 2}),
        ("volume_ratio", {"period": 3}),
        ("volatility", {"period": 5}),
    ]
    for name, params in feature_specs:
        values = processor._compute_feature(price_frame, name, params)
        assert len(values) == len(price_frame)

    fallback = processor._compute_feature(price_frame, "unsupported", {})
    assert fallback.isna().all()


def test_process_data_defaults(price_frame: pd.DataFrame):
    processor = FeatureProcessor()
    result = processor.process_data(price_frame)
    for feature_name in processor._available_features:
        assert feature_name in result.columns

