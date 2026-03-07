import pandas as pd
import pytest

from src.features.processors.base_processor import ProcessorConfig
from src.features.processors.feature_processor import FeatureProcessor


def test_process_appends_default_features(sample_price_frame):
    processor = FeatureProcessor()
    result = processor.process(sample_price_frame)

    for feature_name in processor._get_available_features():
        assert f"feature_{feature_name}" in result.columns
    assert result.shape[0] == sample_price_frame.shape[0]


def test_process_with_invalid_feature_raises(sample_price_frame):
    processor = FeatureProcessor()
    with pytest.raises(ValueError):
        processor.process(sample_price_frame, ["unknown_feature"])


def test_process_with_empty_data_raises():
    processor = FeatureProcessor()
    empty_df = pd.DataFrame(columns=["close", "volume"])
    with pytest.raises(ValueError):
        processor.process(empty_df)


def test_update_config_injects_new_params(sample_price_frame):
    processor = FeatureProcessor()
    processor.update_config({"custom_metric": {"period": 5}})
    assert "custom_metric" in processor.config.feature_params
    assert processor.config.feature_params["custom_metric"]["period"] == 5


def test_compute_feature_missing_column_returns_empty_series():
    processor = FeatureProcessor()
    data = pd.DataFrame({"volume": [100, 200, 300]})
    series = processor._compute_feature(data, "sma", {})
    assert series.isna().all()


def test_process_data_adds_columns(sample_price_frame):
    processor = FeatureProcessor()
    subset = ["sma", "price_change"]
    result = processor.process_data(sample_price_frame, subset)
    for name in subset:
        assert name in result.columns


def test_get_feature_summary_and_list_features():
    processor = FeatureProcessor()
    summary = processor.get_feature_summary()
    features = processor.list_features()

    assert summary["total_features"] == len(processor._get_available_features())
    assert isinstance(features, list)
    assert features[0]["name"] in processor._get_available_features()


def test_validate_features(sample_price_frame):
    processor = FeatureProcessor()
    valid, errors = processor.validate_features(["sma", "rsi"])
    assert valid is True
    assert errors == []

    valid, errors = processor.validate_features(["sma", "unknown"])
    assert valid is False
    assert any("unknown" in message for message in errors)


def test_compute_feature_unknown_logs_warning(sample_price_frame, caplog):
    processor = FeatureProcessor()
    with caplog.at_level("WARNING"):
        series = processor._compute_feature(sample_price_frame, "unknown_feature", {})
    assert series.isna().all()
    assert any("Unknown feature" in message for message in caplog.messages)


def test_compute_feature_error_path_returns_empty(sample_price_frame, caplog, monkeypatch):
    processor = FeatureProcessor()

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(processor, "_compute_sma", boom)

    with caplog.at_level("ERROR"):
        series = processor._compute_feature(sample_price_frame, "sma", {})
    assert series.isna().all()
    assert any("Error computing feature sma" in message for message in caplog.messages)


def test_calculate_moving_averages_missing_close_logs_warning(caplog):
    processor = FeatureProcessor()
    data = pd.DataFrame({"open": [1, 2, 3]})
    with caplog.at_level("WARNING"):
        result = processor._calculate_moving_averages(data)
    assert result.equals(data)
    assert any("No 'close' column found" in message for message in caplog.messages)


def test_calculate_macd_insufficient_data_warns(sample_price_frame, caplog):
    processor = FeatureProcessor()
    short_frame = sample_price_frame.head(5)
    with caplog.at_level("WARNING"):
        result = processor._calculate_macd(short_frame)
    assert result.equals(short_frame)
    assert any("Insufficient data for MACD calculation" in message for message in caplog.messages)


def test_calculate_bollinger_bands_insufficient_data_warns(sample_price_frame, caplog):
    processor = FeatureProcessor()
    short_frame = sample_price_frame.head(10)
    with caplog.at_level("WARNING"):
        result = processor._calculate_bollinger_bands(short_frame)
    assert result.equals(short_frame)
    assert any("Insufficient data for Bollinger Bands calculation" in message for message in caplog.messages)


def test_process_with_unknown_feature_logs_and_skips(sample_price_frame, caplog):
    processor = FeatureProcessor()
    with caplog.at_level("WARNING"):
        valid, errors = processor.validate_features(["unknown_feature"])
    assert valid is False
    assert any("unknown" in message for message in errors)


def test_update_config_creates_feature_params_when_missing():
    empty_config_processor = FeatureProcessor(
        ProcessorConfig(processor_type="general", feature_params=None)
    )
    empty_config_processor.config.feature_params = None

    empty_config_processor.update_config({"new_feature": {"period": 7}})
    assert empty_config_processor.config.feature_params["new_feature"]["period"] == 7


def test_calculate_moving_averages_success(sample_price_frame):
    processor = FeatureProcessor()
    result = processor._calculate_moving_averages(sample_price_frame)
    for period in processor.config.feature_params.get("moving_averages", []):
        sma_col = f"SMA_{period}"
        ema_col = f"EMA_{period}"
        if len(sample_price_frame) >= period:
            assert sma_col in result.columns
            assert ema_col in result.columns


def test_calculate_rsi_success(sample_price_frame):
    processor = FeatureProcessor()
    result = processor._calculate_rsi(sample_price_frame.copy(), period=14)
    assert "RSI" in result.columns
    assert result["RSI"].notna().any()


def test_calculate_macd_success(sample_price_frame):
    processor = FeatureProcessor()
    result = processor._calculate_macd(sample_price_frame.copy())
    for col in ["MACD", "MACD_Signal", "MACD_Histogram"]:
        assert col in result.columns
        assert result[col].notna().any()


def test_calculate_bollinger_bands_success(sample_price_frame):
    processor = FeatureProcessor()
    result = processor._calculate_bollinger_bands(sample_price_frame.copy())
    for col in ["BB_Middle", "BB_Std", "BB_Upper", "BB_Lower", "BB_Width"]:
        assert col in result.columns
        assert result[col].notna().any()


def test_process_data_empty_returns_empty():
    processor = FeatureProcessor()
    result = processor.process_data(pd.DataFrame())
    assert result.empty


def test_get_feature_metadata_default_branch(sample_price_frame):
    processor = FeatureProcessor()
    metadata = processor._get_feature_metadata("price_change")
    assert metadata["name"] == "price_change"
    assert metadata["type"] == "technical"


def test_get_feature_description_and_parameters_fallback():
    processor = FeatureProcessor()
    assert processor._get_feature_description("custom_feature") == "custom_feature指标"
    assert processor._get_feature_parameters("custom_feature") == {}

