import numpy as np
import pandas as pd
import pytest

from src.features.indicators.volatility_calculator import VolatilityCalculator


def _build_sample_ohlc(rows: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    base = np.linspace(100, 110, rows)
    data = pd.DataFrame(
        {
            "open": base + np.sin(np.linspace(0, 3, rows)),
            "high": base + 1.5,
            "low": base - 1.5,
            "close": base + np.cos(np.linspace(0, 3, rows)),
        },
        index=idx,
    )
    return data


def test_calculate_returns_empty_for_empty_input():
    calc = VolatilityCalculator()
    result = calc.calculate(pd.DataFrame())
    assert result.empty


def test_calculate_handles_missing_required_columns():
    data = _build_sample_ohlc()[["open", "close"]]  # 缺少high/low
    calc = VolatilityCalculator()
    result = calc.calculate(data)
    # 返回原始数据，不新增衍生列
    assert list(result.columns) == ["open", "close"]


def test_calculate_generates_core_indicators_and_signals():
    data = _build_sample_ohlc()
    # 使用较短窗口以便小样本也能产生数值
    calc = VolatilityCalculator(
        {
            "bb_period": 5,
            "kc_period": 5,
            "atr_period": 5,
            "hv_period": 5,
            "rv_period": 5,
        }
    )
    result = calc.calculate(data)

    required_columns = {
        "BB_Upper",
        "BB_Middle",
        "BB_Lower",
        "KC_Upper",
        "KC_Middle",
        "KC_Lower",
        "ATR",
        "volatility_atr",
        "volatility_bb_width",
        "volatility_kc_width",
        "volatility_donchian",
        "volatility_high",
        "volatility_low",
        "volatility_breakout",
    }
    assert required_columns.issubset(result.columns)

    # 末尾至少有一个布尔信号为布尔类型
    last_row = result.iloc[-1]
    assert isinstance(last_row["volatility_high"], (bool, np.bool_))
    assert isinstance(last_row["volatility_low"], (bool, np.bool_))
    assert isinstance(last_row["volatility_breakout"], (bool, np.bool_))

    # 指标应产生非空数值
    assert pd.notna(last_row["BB_Upper"])
    assert pd.notna(last_row["KC_Upper"])
    assert pd.notna(last_row["ATR"])

