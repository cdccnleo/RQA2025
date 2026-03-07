#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TechnicalIndicatorProcessor 关键路径测试，覆盖核心分支。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.processors.technical_indicator_processor import (
    IndicatorConfig,
    IndicatorType,
    TechnicalIndicatorProcessor,
)

pytestmark = pytest.mark.features


@pytest.fixture()
def price_frame():
    periods = 80
    base = np.linspace(100, 120, periods)
    return pd.DataFrame(
        {
            "open": base + 0.2,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base,
            "volume": np.linspace(1_000, 2_000, periods),
        }
    )


@pytest.fixture(autouse=True)
def stub_external_calculators(monkeypatch):
    class DummyFibonacci:
        def __init__(self, params=None):
            self.params = params

        def calculate(self, data):
            return pd.DataFrame(
                {
                    "FIB_SUPPORT": data["close"].rolling(5).mean(),
                    "FIB_RESISTANCE": data["close"].rolling(5).mean() + 1,
                }
            )

    class DummyMomentum:
        def __init__(self, params=None):
            self.params = params

        def calculate(self, data):
            return pd.DataFrame({"MOMENTUM_VALUE": data["close"].diff()})

    class DummyIchimoku:
        def __init__(self, params=None):
            self.params = params

        def calculate(self, data):
            return pd.DataFrame({"ICHIMOKU_SPAN": data["close"].rolling(9).mean()})

    class DummyVolatility:
        def __init__(self, params=None):
            self.params = params

        def calculate(self, data):
            return pd.DataFrame({"volatility_atr": data["close"].rolling(5).std()})

    monkeypatch.setattr(
        "src.features.processors.technical_indicator_processor.FibonacciCalculator",
        DummyFibonacci,
    )
    monkeypatch.setattr(
        "src.features.processors.technical_indicator_processor.MomentumCalculator",
        DummyMomentum,
    )
    monkeypatch.setattr(
        "src.features.processors.technical_indicator_processor.IchimokuCalculator",
        DummyIchimoku,
    )
    monkeypatch.setattr(
        "src.features.processors.technical_indicator_processor.VolatilityCalculator",
        DummyVolatility,
    )


def test_calculate_single_indicator_branches(price_frame):
    processor = TechnicalIndicatorProcessor()
    branch_expectations = {
        "sma": "SMA_20",
        "ema": "EMA_12",
        "rsi": "RSI_14",
        "macd": "MACD",
        "bollinger_bands": "BB_Upper_20",
        "stochastic": "STOCH_K_14",
        "williams_r": "WILLR_14",
        "cci": "CCI_20",
        "atr": "ATR_14",
        "obv": "OBV",
        "ichimoku": "ICHIMOKU_SPAN",
        "fibonacci": "FIB_SUPPORT",
        "momentum": "MOMENTUM_VALUE",
        "volatility": "volatility_atr",
    }

    for name, expected_column in branch_expectations.items():
        config = processor.get_indicator_config(name)
        result = processor._calculate_single_indicator(price_frame.copy(), name, config)
        assert expected_column in result.columns, f"{name} missing {expected_column}"


def test_parallel_calculation_generates_columns(price_frame):
    processor = TechnicalIndicatorProcessor()
    subset = ["sma", "ema", "rsi"]
    result = processor.calculate_indicators_parallel(
        price_frame, indicators=subset, max_workers=2
    )
    for expected in ["SMA_20", "EMA_12", "RSI_14"]:
        assert expected in result.columns


def test_custom_indicator_support(price_frame):
    custom = {
        "use_defaults": False,
        "custom_indicators": {
            "SMA_5": {
                "name": "SMA_5",
                "type": IndicatorType.TREND,
                "parameters": {"period": 5},
                "description": "短周期SMA",
                "enabled": True,
            }
        },
    }
    processor = TechnicalIndicatorProcessor(custom)
    result = processor.calculate_indicators(price_frame, ["SMA_5"])
    assert "SMA_5" in result.columns


def test_indicator_management_helpers(price_frame):
    processor = TechnicalIndicatorProcessor()
    processor.disable_indicator("sma")
    assert not processor.is_indicator_enabled("sma")

    processor.enable_indicator("sma")
    assert processor.is_indicator_enabled("sma")

    processor.add_custom_indicator(
        "custom",
        {
            "name": "custom",
            "type": IndicatorType.TREND,
            "parameters": {"period": 3},
            "description": "自定义",
            "enabled": True,
        },
    )
    assert "custom" in processor.get_available_indicators()

    grouped = processor.get_indicator_types()
    assert IndicatorType.TREND in grouped

    info = processor.get_indicator_info("custom")
    assert isinstance(info, IndicatorConfig)

    assert processor.update_indicator_config(
        "custom",
        IndicatorConfig(name="custom", type=IndicatorType.TREND, parameters={"period": 10}),
    )


def test_resolve_indicator_config_with_dynamic_names(price_frame):
    processor = TechnicalIndicatorProcessor()
    dynamic_config = processor._resolve_indicator_config("SMA_15", None)
    assert dynamic_config.parameters["period"] == 15

    result = processor.calculate_indicators(price_frame, ["SMA_15"])
    assert "SMA_15" in result.columns

