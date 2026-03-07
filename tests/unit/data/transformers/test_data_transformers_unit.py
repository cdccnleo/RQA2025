import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pandas as pd
import numpy as np
import pytest

from src.data.transformers.data_transformer import (
    DataFrameTransformer,
    TimeSeriesTransformer,
    FeatureTransformer,
    NormalizationTransformer,
    MissingValueTransformer,
    DateColumnTransformer,
)


def test_dataframe_transformer_drop_keep_fill():
    df = pd.DataFrame({"x": [1, None, 3], "y": [4, 5, None], "z": [7, 8, 9]})
    t = DataFrameTransformer({"columns_to_drop": ["z"], "columns_to_keep": ["x", "y"], "fill_method": "interpolate"})
    out = t.transform(df)
    assert list(out.columns) == ["x", "y"]
    assert out.isna().sum().sum() == 0


def test_time_series_transformer_resample_and_fill():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0, 1, 2],
            "close": [1.5, 2.5, 3.5],
            "volume": [10, 20, 30],
        }
    )
    t = TimeSeriesTransformer({"resample_freq": "2D", "fill_method": "ffill"})
    out = t.transform(df)
    assert "close" in out.columns and "volume" in out.columns


def test_feature_transformer_normalize_and_scale():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    t = FeatureTransformer({"normalize": True, "scale_features": ["b"]})
    out = t.transform(df)
    assert "a" in out.columns and "b" in out.columns
    assert out["b"].min() == 0.0 and out["b"].max() == 1.0


def test_normalization_transformer_methods():
    df = pd.DataFrame({"v": [1.0, 2.0, 4.0, 8.0]})
    t1 = NormalizationTransformer({"method": "zscore"})
    z = t1.transform(df)
    assert np.isclose(z["v"].mean(), 0.0, atol=1e-6)
    t2 = NormalizationTransformer({"method": "minmax"})
    mm = t2.transform(df)
    assert mm["v"].min() == 0.0 and mm["v"].max() == 1.0


def test_missing_value_transformer_strategies():
    df = pd.DataFrame({"v": [1.0, np.nan, 3.0]})
    t = MissingValueTransformer({"strategy": "mean"})
    out = t.transform(df)
    assert out.isna().sum().sum() == 0
    t2 = MissingValueTransformer({"strategy": "drop"})
    out2 = t2.transform(df)
    assert len(out2) < len(df)


def test_date_column_transformer_parse_and_features():
    df = pd.DataFrame({"d": ["2024-01-01", "2024-01-02", "invalid"]})
    t = DateColumnTransformer({"date_columns": ["d"], "extract_features": ["year", "month", "day"]})
    out = t.transform(df)
    assert "d_year" in out.columns and "d_month" in out.columns and "d_day" in out.columns


