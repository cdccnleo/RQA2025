#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VolatilityCalculator 精准单元测试。"""

import numpy as np
import pandas as pd
import pytest

from src.features.indicators.volatility_calculator import VolatilityCalculator

pytestmark = pytest.mark.features


@pytest.fixture()
def price_data():
    periods = 80
    base = np.linspace(100, 120, periods)
    return pd.DataFrame(
        {
            "open": base + 0.1,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base,
        }
    )


def test_calculate_adds_expected_columns(price_data):
    calculator = VolatilityCalculator(
        {
            "bb_period": 5,
            "kc_period": 5,
            "atr_period": 5,
            "hv_period": 5,
            "rv_period": 5,
        }
    )
    result = calculator.calculate(price_data)

    expected_columns = {
        "BB_Upper",
        "BB_Lower",
        "KC_Upper",
        "KC_Lower",
        "ATR",
        "HV_5",
        "PV_10",
        "GK_10",
        "YZ_10",
        "RV_5",
        "volatility_atr",
        "volatility_bb_width",
        "volatility_kc_width",
        "volatility_donchian",
        "volatility_high",
        "volatility_low",
        "volatility_breakout",
    }
    missing = expected_columns - set(result.columns)
    assert not missing
    assert result["ATR"].notna().sum() > 0
    assert result["volatility_bb_width"].notna().sum() > 0


def test_calculate_handles_missing_required_columns(price_data):
    calculator = VolatilityCalculator()
    incomplete = price_data[["close"]].copy()
    result = calculator.calculate(incomplete)
    pd.testing.assert_frame_equal(result, incomplete)


def test_calculate_with_none_returns_empty():
    calculator = VolatilityCalculator()
    result = calculator.calculate(None)
    assert result.empty

