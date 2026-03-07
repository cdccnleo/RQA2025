# -*- coding: utf-8 -*-
"""
数据转换器真实实现测试
测试 DataTransformer 及其子类的核心功能
"""

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


import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.transformers.data_transformer import (
    DataTransformer,
    DataFrameTransformer,
    TimeSeriesTransformer,
    FeatureTransformer,
    NormalizationTransformer
)


@pytest.fixture
def sample_dataframe():
    """创建示例DataFrame"""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'open': [100 + i for i in range(10)],
        'high': [105 + i for i in range(10)],
        'low': [95 + i for i in range(10)],
        'close': [102 + i for i in range(10)],
        'volume': [1000 + i * 100 for i in range(10)],
        'extra_col': [i * 2 for i in range(10)]
    })


@pytest.fixture
def dataframe_with_nulls(sample_dataframe):
    """创建包含空值的DataFrame"""
    data = sample_dataframe.copy()
    data.loc[2, 'close'] = np.nan
    data.loc[5, 'volume'] = np.nan
    return data


def test_dataframe_transformer_initialization():
    """测试DataFrame转换器初始化"""
    transformer = DataFrameTransformer()
    
    assert transformer.config == {}
    assert transformer.columns_to_drop == []
    assert transformer.columns_to_keep is None


def test_dataframe_transformer_drop_columns(sample_dataframe):
    """测试删除指定列"""
    config = {'columns_to_drop': ['extra_col']}
    transformer = DataFrameTransformer(config)
    
    result = transformer.transform(sample_dataframe)
    
    assert 'extra_col' not in result.columns
    assert 'close' in result.columns


def test_dataframe_transformer_keep_columns(sample_dataframe):
    """测试保留指定列"""
    config = {'columns_to_keep': ['date', 'close', 'volume']}
    transformer = DataFrameTransformer(config)
    
    result = transformer.transform(sample_dataframe)
    
    assert set(result.columns) == {'date', 'close', 'volume'}


def test_dataframe_transformer_ffill(sample_dataframe, dataframe_with_nulls):
    """测试前向填充"""
    config = {'fill_method': 'ffill'}
    transformer = DataFrameTransformer(config)
    
    result = transformer.transform(dataframe_with_nulls)
    
    assert not result['close'].isnull().any()
    assert not result['volume'].isnull().any()


def test_dataframe_transformer_bfill(sample_dataframe, dataframe_with_nulls):
    """测试后向填充"""
    config = {'fill_method': 'bfill'}
    transformer = DataFrameTransformer(config)
    
    result = transformer.transform(dataframe_with_nulls)
    
    assert not result['close'].isnull().any()


def test_dataframe_transformer_interpolate(sample_dataframe, dataframe_with_nulls):
    """测试插值填充"""
    config = {'fill_method': 'interpolate'}
    transformer = DataFrameTransformer(config)
    
    result = transformer.transform(dataframe_with_nulls)
    
    assert not result['close'].isnull().any()


def test_dataframe_transformer_invalid_input():
    """测试无效输入"""
    transformer = DataFrameTransformer()
    
    with pytest.raises(ValueError, match="输入数据必须是pandas DataFrame"):
        transformer.transform("not a dataframe")


def test_timeseries_transformer_initialization():
    """测试时间序列转换器初始化"""
    transformer = TimeSeriesTransformer()
    
    assert transformer.config == {}
    assert transformer.resample_freq is None


def test_timeseries_transformer_sets_date_index(sample_dataframe):
    """测试设置日期索引"""
    transformer = TimeSeriesTransformer()
    
    result = transformer.transform(sample_dataframe)
    
    assert isinstance(result.index, pd.DatetimeIndex)


def test_timeseries_transformer_resample(sample_dataframe):
    """测试重采样"""
    config = {'resample_freq': '2D'}
    transformer = TimeSeriesTransformer(config)
    
    result = transformer.transform(sample_dataframe)
    
    assert len(result) <= len(sample_dataframe)
    assert isinstance(result.index, pd.DatetimeIndex)


def test_timeseries_transformer_handles_missing_date_column(sample_dataframe):
    """测试处理缺少date列的情况"""
    data = sample_dataframe.drop(columns=['date'])
    transformer = TimeSeriesTransformer()
    
    # 应该不会报错，只是不设置日期索引
    result = transformer.transform(data)
    
    assert isinstance(result, pd.DataFrame)


def test_feature_transformer_initialization():
    """测试特征转换器初始化"""
    transformer = FeatureTransformer()
    
    assert transformer.config == {}
    assert transformer.normalize is False


def test_feature_transformer_normalize(sample_dataframe):
    """测试特征标准化"""
    config = {'normalize': True}
    transformer = FeatureTransformer(config)
    
    result = transformer.transform(sample_dataframe)
    
    # 检查数值列是否被标准化（均值接近0，标准差接近1）
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in result.columns and result[col].std() > 0:
            assert abs(result[col].mean()) < 0.1
            assert abs(result[col].std() - 1.0) < 0.1


def test_feature_transformer_scale_features(sample_dataframe):
    """测试特征缩放"""
    config = {'scale_features': ['close', 'volume']}
    transformer = FeatureTransformer(config)
    
    result = transformer.transform(sample_dataframe)
    
    # 检查指定特征是否被缩放到[0, 1]
    for col in ['close', 'volume']:
        if col in result.columns and result[col].std() > 0:
            assert result[col].min() >= 0
            assert result[col].max() <= 1


def test_normalization_transformer_initialization():
    """测试标准化转换器初始化"""
    transformer = NormalizationTransformer()
    
    assert transformer.config == {}
    assert transformer.method == 'zscore'


def test_normalization_transformer_zscore(sample_dataframe):
    """测试Z-score标准化"""
    config = {'method': 'zscore'}
    transformer = NormalizationTransformer(config)
    
    result = transformer.transform(sample_dataframe)
    
    # 检查数值列的均值是否接近0
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in result.columns and result[col].std() > 0:
            assert abs(result[col].mean()) < 0.1


def test_normalization_transformer_minmax(sample_dataframe):
    """测试MinMax标准化"""
    config = {'method': 'minmax'}
    transformer = NormalizationTransformer(config)
    
    result = transformer.transform(sample_dataframe)
    
    # 检查数值列是否在[0, 1]范围内
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in result.columns and len(result) > 1:
            assert result[col].min() >= 0
            assert result[col].max() <= 1


def test_transformer_fit_transform(sample_dataframe):
    """测试fit_transform方法"""
    transformer = FeatureTransformer({'normalize': True})
    
    result = transformer.fit_transform(sample_dataframe)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_dataframe)


def test_missing_value_transformer_strategies():
    import pandas as pd
    import numpy as np
    from src.data.transformers.data_transformer import MissingValueTransformer

    df = pd.DataFrame({
        'a': [1.0, np.nan, 3.0, np.nan],
        'b': [np.nan, 2.0, np.nan, 4.0],
        'c': ['x', None, 'z', None],
    })

    # ffill
    t_ffill = MissingValueTransformer({'strategy': 'ffill'})
    r_ffill = t_ffill.transform(df)
    assert r_ffill.isna().sum().sum() < df.isna().sum().sum()

    # bfill
    t_bfill = MissingValueTransformer({'strategy': 'bfill'})
    r_bfill = t_bfill.transform(df)
    assert r_bfill.isna().sum().sum() < df.isna().sum().sum()

    # interpolate（仅对数值列有效）
    t_interp = MissingValueTransformer({'strategy': 'interpolate'})
    r_interp = t_interp.transform(df)
    assert r_interp['a'].isna().sum() <= 1

    # drop
    t_drop = MissingValueTransformer({'strategy': 'drop'})
    r_drop = t_drop.transform(df)
    assert len(r_drop) <= len(df)

    # fill 常量
    t_fill = MissingValueTransformer({'strategy': 'fill', 'fill_value': -1})
    r_fill = t_fill.transform(df)
    assert (r_fill.select_dtypes(include=[float, int]).values == -1).any()

    # mean/median（数值列）
    t_mean = MissingValueTransformer({'strategy': 'mean'})
    r_mean = t_mean.transform(df)
    assert r_mean['a'].isna().sum() == 0

    t_median = MissingValueTransformer({'strategy': 'median'})
    r_median = t_median.transform(df)
    assert r_median['b'].isna().sum() == 0

    # constant（别名）
    t_const = MissingValueTransformer({'strategy': 'constant', 'fill_value': 0})
    r_const = t_const.transform(df)
    assert r_const.isna().sum().sum() == 0

    # default fallback
    t_unknown = MissingValueTransformer({'strategy': 'unknown'})
    r_unknown = t_unknown.transform(df)
    assert r_unknown.isna().sum().sum() < df.isna().sum().sum()


def test_date_column_transformer_parse_and_features():
    import pandas as pd
    from src.data.transformers.data_transformer import DateColumnTransformer

    df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', 'invalid'],
        'value': [1, 2, 3]
    })

    # 基础解析 + 特征抽取
    t = DateColumnTransformer({'extract_features': ['year', 'month', 'day', 'weekday', 'hour', 'minute', 'second']})
    r = t.transform(df)
    assert 'date_year' in r.columns
    assert 'date_month' in r.columns
    assert 'date_day' in r.columns
    assert 'date_weekday' in r.columns
    assert 'date_hour' in r.columns
    assert 'date_minute' in r.columns
    assert 'date_second' in r.columns

    # 指定多列与不同格式
    df2 = pd.DataFrame({
        'trade_date': ['01/03/2024', '01/04/2024'],
        'settle_date': ['2024-01-05', '2024-01-06'],
        'x': [10, 20]
    })
    t2 = DateColumnTransformer({
        'date_columns': ['trade_date', 'settle_date'],
        'date_formats': {'trade_date': '%m/%d/%Y', 'settle_date': '%Y-%m-%d'},
        'extract_features': ['year']
    })
    r2 = t2.transform(df2)
    assert 'trade_date_year' in r2.columns
    assert 'settle_date_year' in r2.columns

    # 时区处理（包含 NaT 时不抛错）
    df3 = pd.DataFrame({'date': ['2024-01-01T12:00:00', None]})
    t3 = DateColumnTransformer({'timezone': 'UTC'})
    r3 = t3.transform(df3)
    assert 'date' in r3.columns
