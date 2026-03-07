import pandas as pd
import numpy as np
import pytest

from decimal import Decimal
from datetime import datetime

from src.infrastructure.utils.tools.convert import DataConverter


def build_sample_df():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    df = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.5, 10.5, 11.5],
            "close": [10.5, 11.5, 12.5],
            "volume": [1000.0, 1100.0, 1200.0],
        },
        index=idx,
    )
    return df


def test_calculate_limit_prices_normal_and_st():
    result = DataConverter.calculate_limit_prices(10.0)
    assert result == {"upper_limit": 11.0, "lower_limit": 9.0}

    st_result = DataConverter.calculate_limit_prices(5, is_st=True)
    assert st_result == {"upper_limit": 5.25, "lower_limit": 4.75}


@pytest.mark.parametrize("value", ["10", object(), None])
def test_calculate_limit_prices_invalid_value_raises(value):
    with pytest.raises(ValueError):
        DataConverter.calculate_limit_prices(value)


def test_apply_adjustment_factor_vectorized_behavior():
    df = build_sample_df()
    original = df.copy()
    factors = {
        datetime(2024, 1, 3): 1.1,
        datetime(2024, 1, 4): 0.9,
    }

    adjusted = DataConverter.apply_adjustment_factor(df, factors, inplace=False)

    # Ensure original untouched
    pd.testing.assert_frame_equal(df, original)

    # First row should be scaled by cumulative factor 1.0
    assert adjusted.loc["2024-01-02", "close"] == pytest.approx(10.5)
    # Second row factor 1.1 applied
    assert adjusted.loc["2024-01-03", "close"] == pytest.approx(11.5 * 1.1)
    # Third row factor 1.1 * 0.9 applied
    expected_factor = 1.1 * 0.9
    assert adjusted.loc["2024-01-04", "close"] == pytest.approx(12.5 * expected_factor)
    # Volume adjusted by inverse factors
    assert adjusted.loc["2024-01-04", "volume"] == pytest.approx(1200.0 / expected_factor)


def test_parse_margin_data_success_and_error():
    raw = {
        "symbol": "000001",
        "name": "平安银行",
        "margin_balance": "1000",
        "short_balance": "200",
        "margin_buy": "300",
        "short_sell": "100",
        "repayment": "50",
    }
    df = DataConverter.parse_margin_data(raw)
    assert df.loc[0, "net_margin"] == 250
    assert df["margin_balance"].dtype.kind in {"f", "i"}

    missing = raw.copy()
    missing.pop("margin_buy")
    with pytest.raises(ValueError):
        DataConverter.parse_margin_data(missing)


def test_normalize_dragon_board_cleans_strings_and_flags():
    raw = [
        {
            "branch_name": " 上海 营业部 ",
            "direction": "买入",
            "net_amount": "1,200.5万",
            "buy_amount": "500.0万",
        }
    ]
    df = DataConverter.normalize_dragon_board(raw)
    assert df.loc[0, "branch_name"] == "上海营业部"
    assert bool(df.loc[0, "is_buy"])
    assert df.loc[0, "net_amount"] == pytest.approx(1200.5)
    assert df.loc[0, "buy_amount"] == pytest.approx(500.0)


def test_convert_frequency_success_and_errors():
    df = build_sample_df()
    result = DataConverter.convert_frequency(df, "2D")
    assert len(result) == 2
    assert result.iloc[0]["open"] == 10.0
    assert result.iloc[0]["volume"] == 2100.0

    df_no_datetime = df.reset_index()
    with pytest.raises(ValueError):
        DataConverter.convert_frequency(df_no_datetime, "1D")

    df_missing_cols = df[["close"]]
    with pytest.raises(ValueError):
        DataConverter.convert_frequency(df_missing_cols, "1D", {"not_exist": "sum"})

    with pytest.raises(ValueError):
        DataConverter.convert_frequency(df, "BAD_FREQ")

