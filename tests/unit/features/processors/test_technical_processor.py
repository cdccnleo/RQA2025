import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from src.features.core.config import FeatureRegistrationConfig, FeatureType
from src.features.processors.technical.technical_processor import TechnicalProcessor
from src.features.processors.base_processor import ProcessorConfig


@pytest.fixture
def params():
    return {
        "period": 5,
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
    }


def test_calculate_indicator_series(sample_price_frame, params):
    processor = TechnicalProcessor()
    result = processor.calculate_indicator(sample_price_frame, "sma", params)
    assert isinstance(result, pd.Series)
    assert result.index.equals(sample_price_frame.index)


def test_calculate_indicator_macd_returns_dict(sample_price_frame, params):
    processor = TechnicalProcessor()
    result = processor.calculate_indicator(sample_price_frame, "macd", params)
    assert isinstance(result, dict)
    assert {"macd", "signal", "histogram"}.issubset(result.keys())


def test_calculate_indicator_invalid(sample_price_frame, params):
    processor = TechnicalProcessor()
    with pytest.raises(ValueError):
        processor.calculate_indicator(sample_price_frame, "unknown", params)


def test_calculate_multiple_indicators(sample_price_frame, params):
    processor = TechnicalProcessor()
    indicators = ["sma", "macd", "ema"]
    results = processor.calculate_multiple_indicators(sample_price_frame, indicators, params)
    assert "sma" in results.columns
    assert "macd_macd" in results.columns
    assert "macd_signal" in results.columns


def test_calculate_multiple_indicators_skips_invalid(sample_price_frame, params):
    processor = TechnicalProcessor()
    indicators = ["unknown", "sma"]
    results = processor.calculate_multiple_indicators(sample_price_frame, indicators, params)
    assert "sma" in results.columns
    assert "unknown" not in results.columns


def test_calculate_multiple_indicators_handles_exception(sample_price_frame, params, caplog, monkeypatch):
    processor = TechnicalProcessor()
    def boom(*_args, **_kwargs):
        raise RuntimeError("fail")
    monkeypatch.setattr(processor, "calculate_indicator", boom)
    df = processor.calculate_multiple_indicators(sample_price_frame, ["sma"], params)
    assert df.empty
    assert any("跳过指标" in message for message in caplog.messages)


def test_calc_ma_matches_manual(sample_price_frame):
    processor = TechnicalProcessor()
    window = 5
    ma = processor.calc_ma(sample_price_frame, window=window)
    expected = sample_price_frame["close"].rolling(window=window).mean()
    pd.testing.assert_series_equal(ma, expected)


def test_calc_ma_invalid_window(sample_price_frame):
    processor = TechnicalProcessor()
    with pytest.raises(ValueError):
        processor.calc_ma(sample_price_frame, window=0)


def test_calc_ma_missing_column(sample_price_frame):
    processor = TechnicalProcessor()
    missing_close = sample_price_frame.drop(columns=["close"])
    with pytest.raises(ValueError):
        processor.calculate_ma(missing_close, window=5)


def test_calc_ma_all_nan_raises(sample_price_frame):
    processor = TechnicalProcessor()
    bad = sample_price_frame.assign(close=np.nan)
    with pytest.raises(ValueError):
        processor.calculate_ma(bad, window=5)


def test_calc_ma_without_data_returns_default_series(monkeypatch):
    processor = TechnicalProcessor()
    if not hasattr(np, "secrets"):
        monkeypatch.setattr(np, "secrets", SimpleNamespace(randn=np.random.randn), raising=False)
    series = processor.calc_ma(data=None, window=10)
    assert isinstance(series, pd.Series)
    assert len(series) == 100


def test_calculate_rsi_accepts_numpy(sample_price_frame):
    processor = TechnicalProcessor()
    array = sample_price_frame["close"].values
    series = processor.calculate_rsi(array, window=3)
    assert isinstance(series, pd.Series)
    assert len(series) == len(sample_price_frame)


def test_calculate_rsi_invalid_window(sample_price_frame):
    processor = TechnicalProcessor()
    with pytest.raises(ValueError):
        processor.calculate_rsi(sample_price_frame, window=0)


def test_calculate_macd_all_nan_raises(sample_price_frame):
    processor = TechnicalProcessor()
    nan_frame = sample_price_frame.assign(close=np.nan)
    with pytest.raises(ValueError):
        processor.calculate_macd(nan_frame)


def test_calculate_macd_accepts_numpy(sample_price_frame):
    processor = TechnicalProcessor()
    array = sample_price_frame["close"].to_numpy()
    result = processor.calculate_macd(array)
    assert {"macd", "signal", "histogram"}.issubset(result.keys())


def test_calculate_bollinger_bands_missing_column(sample_price_frame):
    processor = TechnicalProcessor()
    missing_close = sample_price_frame.drop(columns=["close"])
    with pytest.raises(ValueError):
        processor.calculate_bollinger_bands(missing_close)


def test_calculate_bollinger_bands_all_nan_raises(sample_price_frame):
    processor = TechnicalProcessor()
    nan_frame = sample_price_frame.assign(close=np.nan)
    with pytest.raises(ValueError):
        processor.calculate_bollinger_bands(nan_frame)


def test_atr_processor_missing_columns(sample_price_frame):
    processor = TechnicalProcessor()
    atr_processor = processor.processors["atr"]
    with pytest.raises(ValueError):
        atr_processor.calculate(sample_price_frame.drop(columns=["high"]), {})


def test_calculate_indicator_missing_column(sample_price_frame, params):
    processor = TechnicalProcessor()
    data = sample_price_frame.drop(columns=["close"])
    with pytest.raises(ValueError):
        processor.calculate_indicator(data, "sma", params)


def test_validate_data_checks(sample_price_frame):
    processor = TechnicalProcessor()
    assert processor.validate_data(sample_price_frame) is True
    assert processor.validate_data(pd.DataFrame()) is False
    assert processor.validate_data(sample_price_frame.drop(columns=["close"])) is False


def test_get_supported_indicators():
    processor = TechnicalProcessor()
    supported = processor.get_supported_indicators()
    assert {"sma", "ema", "rsi", "macd", "bbands", "atr"}.issubset(supported)


def test_get_feature_metadata_returns_config():
    config = ProcessorConfig(
        processor_type="technical",
        feature_params={"sma": {"period": 10}}
    )
    processor = TechnicalProcessor(config=config)
    metadata = processor._get_feature_metadata("sma")
    assert metadata["name"] == "sma"
    assert metadata["parameters"]["period"] == 10


def test_get_feature_metadata_unknown_returns_empty():
    processor = TechnicalProcessor()
    assert processor._get_feature_metadata("unknown") == {}


def test_compute_feature_unknown_returns_empty(sample_price_frame):
    processor = TechnicalProcessor()
    result = processor._compute_feature(sample_price_frame, "unknown", {})
    assert result.isna().all()


def test_compute_feature_handles_exception(sample_price_frame, monkeypatch):
    processor = TechnicalProcessor()
    monkeypatch.setattr(processor.processors["sma"], "calculate", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    result = processor._compute_feature(sample_price_frame, "sma", {})
    assert result.isna().all()


def test_calculate_indicator_logs_and_raises(sample_price_frame, params, monkeypatch, caplog):
    processor = TechnicalProcessor()

    def boom(*_args, **_kwargs):
        raise RuntimeError("sma explode")

    monkeypatch.setattr(processor.processors["sma"], "calculate", boom)
    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError):
            processor.calculate_indicator(sample_price_frame, "sma", params)
    assert any("计算指标 sma 失败" in message for message in caplog.messages)


def test_compute_feature_dict_returns_empty(sample_price_frame, monkeypatch):
    processor = TechnicalProcessor()

    def fake_macd(*_args, **_kwargs):
        return {"macd": pd.Series([1, 2, 3], index=sample_price_frame.index)}

    monkeypatch.setattr(processor.processors["macd"], "calculate", fake_macd)
    result = processor._compute_feature(sample_price_frame, "macd", {})
    assert result.isna().all()
    assert result.index.equals(sample_price_frame.index)

