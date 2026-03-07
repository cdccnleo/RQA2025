import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.features.processors.technical_indicator_processor import (
    IndicatorType,
    TechnicalIndicatorProcessor,
)


class _StubCalculator:
    def __init__(self, prefix: str, params):
        self.prefix = prefix
        self.params = params

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.assign(
            **{f"{self.prefix}_value": np.arange(len(data), dtype=float)}
        )


@pytest.fixture(autouse=True)
def stub_external_calculators(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.technical_indicator_processor.logger",
        logging.getLogger(__name__),
    )

    monkeypatch.setattr(
        "src.features.processors.technical_indicator_processor.IchimokuCalculator",
        lambda params: _StubCalculator("ichimoku", params),
    )
    monkeypatch.setattr(
        "src.features.processors.technical_indicator_processor.FibonacciCalculator",
        lambda params: _StubCalculator("fib", params),
    )
    monkeypatch.setattr(
        "src.features.processors.technical_indicator_processor.MomentumCalculator",
        lambda params: _StubCalculator("mom", params),
    )
    monkeypatch.setattr(
        "src.features.processors.technical_indicator_processor.VolatilityCalculator",
        lambda params: _StubCalculator("vol", params),
    )


@pytest.fixture
def processor():
    config = {"disabled_indicators": ["ichimoku", "momentum", "volatility"], "custom_indicators": {}}
    return TechnicalIndicatorProcessor(config=config)


@pytest.fixture
def price_frame():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    return pd.DataFrame(
        {
            "close": np.linspace(100, 140, num=40),
            "high": np.linspace(101, 141, num=40),
            "low": np.linspace(99, 139, num=40),
            "volume": np.linspace(1000, 2000, num=40),
        },
        index=idx,
    )


def test_calculate_selected_indicators(processor, price_frame):
    result = processor.calculate_indicators(price_frame, ["sma", "macd", "fibonacci"])
    expected_columns = {
        "SMA_20",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "fib_value",
    }
    assert expected_columns <= set(result.columns)


def test_calculate_indicators_missing_columns(processor, price_frame):
    incomplete = price_frame.drop(columns=["high"])
    result = processor.calculate_indicators(incomplete, ["sma"])
    pd.testing.assert_frame_equal(result, incomplete)


def test_disable_enable_indicator(processor, price_frame):
    processor.disable_indicator("sma")
    result_disabled = processor.calculate_indicators(price_frame, ["sma"])
    assert "SMA_20" not in result_disabled.columns

    processor.enable_indicator("sma")
    result_enabled = processor.calculate_indicators(price_frame, ["sma"])
    assert "SMA_20" in result_enabled.columns


def test_get_indicator_types(processor):
    grouped = processor.get_indicator_types()
    assert IndicatorType.TREND in grouped
    assert "sma" in grouped[IndicatorType.TREND]
    assert "ichimoku" not in grouped.get(IndicatorType.TREND, [])


def test_add_custom_indicator(processor, price_frame, monkeypatch):
    custom_config = {
        "name": "Custom",
        "type": IndicatorType.TREND,
        "parameters": {},
        "description": "custom",
        "enabled": True,
    }
    processor.add_custom_indicator("custom_indicator", custom_config)

    class _CustomCalc:
        def __init__(self, params):
            self.params = params

        def calculate(self, data):
            return data.assign(custom_indicator=np.arange(len(data)))

    original = processor._calculate_single_indicator

    def fake_single(data, name, cfg):
        if name == "custom_indicator":
            return _CustomCalc(cfg.parameters).calculate(data)
        return original(data, name, cfg)

    monkeypatch.setattr(processor, "_calculate_single_indicator", fake_single)

    result = processor.calculate_indicators(price_frame, ["custom_indicator"])
    assert "custom_indicator" in result.columns


def test_calculate_indicators_empty_input():
    tip = TechnicalIndicatorProcessor()
    empty = pd.DataFrame()
    assert tip.calculate_indicators(empty).empty


def test_calculate_indicators_unknown_indicator(processor, price_frame):
    before_columns = set(price_frame.columns)
    result = processor.calculate_indicators(price_frame, ["unknown"])
    assert set(result.columns) == before_columns


def test_calculate_obv_missing_volume(processor, price_frame):
    frame_no_volume = price_frame.drop(columns=["volume"])
    output = processor.calculate_indicators(frame_no_volume, ["obv"])
    assert "OBV" not in output.columns


def test_calculate_all_indicators_hits_every_branch(price_frame):
    tip = TechnicalIndicatorProcessor()
    all_indicators = [
        "sma",
        "ema",
        "rsi",
        "macd",
        "bollinger_bands",
        "stochastic",
        "williams_r",
        "cci",
        "atr",
        "obv",
        "ichimoku",
        "fibonacci",
        "momentum",
        "volatility",
    ]
    result = tip.calculate_indicators(price_frame, all_indicators)
    expected_columns = {
        "SMA_20",
        "EMA_12",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "BB_Upper_20",
        "BB_Middle_20",
        "BB_Lower_20",
        "STOCH_K_14",
        "STOCH_D_14_3",
        "WILLR_14",
        "CCI_20",
        "ATR_14",
        "OBV",
        "ichimoku_value",
        "fib_value",
        "mom_value",
        "vol_value",
    }
    assert expected_columns <= set(result.columns)

