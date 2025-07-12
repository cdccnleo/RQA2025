"""
数据处理模块测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import shutil
from unittest.mock import Mock, patch

from src.data.processing.data_processor import DataProcessor, FillMethod
from src.infrastructure.utils.exceptions import DataProcessingError


@pytest.fixture
def data_processor():
    """数据处理器fixture"""
    return DataProcessor()


@pytest.fixture
def sample_df():
    """样本数据框fixture"""
    # 创建测试数据
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    df = pd.DataFrame({
        'close': [100.0, 101.0, 102.0, np.nan, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        'volume': [1000, 1100, 1200, 1300, np.nan, 1500, 1600, 1700, 1800, 1900],
        'category': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A']
    }, index=dates)
    return df


@pytest.fixture
def multi_dfs():
    """多个数据框fixture"""
    # 创建多个测试数据框
    dates1 = pd.date_range(start='2023-01-01', end='2023-01-10')
    df1 = pd.DataFrame({
        'close': np.random.randn(len(dates1)) + 100,
        'volume': np.random.randint(1000, 10000, len(dates1))
    }, index=dates1)
    
    dates2 = pd.date_range(start='2023-01-03', end='2023-01-12')
    df2 = pd.DataFrame({
        'open': np.random.randn(len(dates2)) + 100,
        'high': np.random.randn(len(dates2)) + 105
    }, index=dates2)
    
    dates3 = pd.date_range(start='2023-01-05', end='2023-01-15')
    df3 = pd.DataFrame({
        'sentiment': np.random.randn(len(dates3)),
        'news_count': np.random.randint(0, 100, len(dates3))
    }, index=dates3)
    
    return {
        'price': df1,
        'ohlc': df2,
        'sentiment': df3
    }


def test_align_data_outer(data_processor, multi_dfs):
    """测试外部对齐"""
    # 对齐数据
    aligned = data_processor.align_data(
        multi_dfs,
        freq='D',
        method='outer',
        fill_method='ffill'
    )
    
    # 验证结果
    assert len(aligned) == len(multi_dfs)
    
    # 验证所有数据框有相同的索引
    expected_index = pd.date_range(
        start=min(df.index.min() for df in multi_dfs.values()),
        end=max(df.index.max() for df in multi_dfs.values()),
        freq='D'
    )
    
    for name, df in aligned.items():
        assert df.index.equals(expected_index)
        assert not df.isnull().all().all()  # 确保不是全部为NaN


def test_align_data_inner(data_processor, multi_dfs):
    """测试内部对齐"""
    # 对齐数据
    aligned = data_processor.align_data(
        multi_dfs,
        freq='D',
        method='inner',
        fill_method=None
    )
    
    # 验证结果
    assert len(aligned) == len(multi_dfs)
    
    # 验证所有数据框有相同的索引
    expected_start = max(df.index.min() for df in multi_dfs.values())
    expected_end = min(df.index.max() for df in multi_dfs.values())
    expected_index = pd.date_range(start=expected_start, end=expected_end, freq='D')
    
    for name, df in aligned.items():
        assert df.index.equals(expected_index)


@pytest.mark.parametrize("method,expected_nulls", [
    ('outer', 0),  # 外部对齐+填充应该没有空值
    ('inner', 0)   # 内部对齐+填充应该没有空值
])
def test_align_with_fill(data_processor, multi_dfs, method, expected_nulls):
    """测试对齐并填充"""
    # 对齐数据并填充
    aligned = data_processor.align_data(
        multi_dfs,
        freq='D',
        method=method,
        fill_method='ffill'
    )
    
    # 验证结果
    for name, df in aligned.items():
        assert df.isnull().sum().sum() == expected_nulls


def test_align_with_different_fill_methods(data_processor, multi_dfs):
    """测试使用不同的填充方法"""
    # 为每个数据源指定不同的填充方法
    fill_methods = {
        'price': 'ffill',
        'ohlc': 'bfill',
        'sentiment': 'mean'
    }
    
    # 对齐数据
    aligned = data_processor.align_data(
        multi_dfs,
        freq='D',
        method='outer',
        fill_method=fill_methods
    )
    
    # 验证结果
    assert len(aligned) == len(multi_dfs)
    
    # 验证填充方法的效果
    # 这里只是简单检查是否有NaN值，实际上应该更详细地验证每种填充方法的效果
    for name, df in aligned.items():
        assert not df.isnull().all().all()


def test_align_invalid_method(data_processor, multi_dfs):
    """测试无效的对齐方法"""
    with pytest.raises(DataProcessingError, match="不支持的对齐方法"):
        data_processor.align_data(
            multi_dfs,
            freq='D',
            method='invalid',
            fill_method=None
        )


def test_align_non_datetime_index(data_processor):
    """测试非日期时间索引"""
    # 创建带有非日期时间索引的数据框
    df1 = pd.DataFrame({
        'close': [100, 101, 102],
        'volume': [1000, 1100, 1200]
    }, index=[1, 2, 3])
    
    df2 = pd.DataFrame({
        'open': [99, 100, 101],
        'high': [102, 103, 104]
    }, index=[2, 3, 4])
    
    # 尝试对齐
    with pytest.raises(DataProcessingError, match="无法转换为DatetimeIndex"):
        data_processor.align_data(
            {'df1': df1, 'df2': df2},
            freq='D',
            method='outer'
        )


@pytest.mark.parametrize("method", [
    FillMethod.FORWARD,
    FillMethod.BACKWARD,
    FillMethod.MEAN,
    FillMethod.MEDIAN,
    FillMethod.ZERO,
    FillMethod.INTERPOLATE,
    'ffill',  # 字符串形式
    'bfill',
    'mean',
    'median',
    'zero',
    'interpolate'
])
def test_fill_missing(data_processor, sample_df, method):
    """测试缺失值填充"""
    # 填充缺失值
    filled_df = data_processor.fill_missing(sample_df, method=method)
    
    # 验证结果
    assert filled_df is not None
    assert filled_df.shape == sample_df.shape
    
    # 对于数值列，验证是否有NaN值
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 某些方法可能无法填充所有NaN值（如前向填充第一个值）
        if method not in [FillMethod.FORWARD, 'ffill']:
            assert filled_df[col].isnull().sum() == 0


def test_fill_missing_custom(data_processor, sample_df):
    """测试自定义填充方法"""
    # 定义自定义填充函数
    def custom_fill(series):
        return series.fillna(series.mean() * 1.1)
    
    # 填充缺失值
    filled_df = data_processor.fill_missing(
        sample_df,
        method=FillMethod.CUSTOM,
        custom_func=custom_fill
    )
    
    # 验证结果
    assert filled_df is not None
    assert filled_df.shape == sample_df.shape
    
    # 验证数值列是否有NaN值
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert filled_df[col].isnull().sum() == 0


def test_fill_missing_by_column(data_processor, sample_df):
    """测试按列填充缺失值"""
    # 为每列指定不同的填充方法
    methods = {
        'close': 'mean',
        'volume': 'ffill',
        'category': 'bfill'
    }
    
    # 填充缺失值
    filled_df = data_processor.fill_missing(sample_df, method=methods)
    
    # 验证结果
    assert filled_df is not None
    assert filled_df.shape == sample_df.shape
    
    # 验证数值列是否有NaN值
    assert filled_df['close'].isnull().sum() == 0
    assert filled_df['volume'].isnull().sum() == 0


def test_fill_missing_invalid_method(data_processor, sample_df):
    """测试无效的填充方法"""
    with pytest.raises(DataProcessingError, match="不支持的填充方法"):
        data_processor.fill_missing(sample_df, method='invalid')


def test_fill_missing_custom_without_func(data_processor, sample_df):
    """测试自定义填充方法但未提供函数"""
    with pytest.raises(DataProcessingError, match="使用CUSTOM方法时必须提供custom_func"):
        data_processor.fill_missing(sample_df, method=FillMethod.CUSTOM)


@pytest.mark.parametrize("freq,method", [
    ('D', 'mean'),
    ('W', 'sum'),
    ('M', 'last'),
    ('H', 'first')
])
def test_resample_data(data_processor, sample_df, freq, method):
    """测试数据重采样"""
    # 重采样数据
    resampled_df = data_processor.resample_data(
        sample_df,
        freq=freq,
        method=method,
        fill_method='ffill'
    )
    
    # 验证结果
    assert resampled_df is not None
    assert isinstance(resampled_df.index, pd.DatetimeIndex)
    assert resampled_df.index.freq == pd.tseries.frequencies.to_offset(freq)


def test_resample_data_invalid_method(data_processor, sample_df):
    """测试无效的重采样方法"""
    with pytest.raises(DataProcessingError, match="不支持的重采样方法"):
        data_processor.resample_data(
            sample_df,
            freq='D',
            method='invalid'
        )


@pytest.mark.parametrize("method,threshold,handle_method", [
    ('zscore', 2.0, 'mark'),
    ('zscore', 3.0, 'remove'),
    ('zscore', 2.5, 'fill'),
    ('iqr', 1.5, 'mark'),
    ('iqr', 1.5, 'remove'),
    ('mad', 3.0, 'mark'),
    ('mad', 3.0, 'fill')
])
def test_detect_outliers(data_processor, sample_df, method, threshold, handle_method):
    """测试异常值检测"""
    # 添加一些异常值
    df = sample_df.copy()
    df.loc[df.index[2], 'close'] = 200  # 明显的异常值
    df.loc[df.index[7], 'volume'] = 10000  # 明显的异常值
    
    # 检测异常值
    result = data_processor.detect_outliers(
        df,
        method=method,
        threshold=threshold,
        handle_method=handle_method
    )
    
    # 验证结果
    if handle_method == 'mark':
        processed_df, outlier_mask = result
        assert isinstance(outlier_mask, pd.DataFrame)
        assert outlier_mask.shape == df.shape
        # 验证是否检测到异常值
        assert outlier_mask.any().any()
    else:
        processed_df = result
        # 如果是'remove'或'fill'，验证异常值是否被处理
        if handle_method == 'remove':
            # 异常值应该被设为NaN
            assert processed_df.isnull().any().any()
        elif handle_method == 'fill':
            # 异常值应该被填充
            assert not processed_df.isnull().any().any()


def test_detect_outliers_invalid_method(data_processor, sample_df):
    """测试无效的异常值检测方法"""
    with pytest.raises(DataProcessingError, match="不支持的异常值检测方法"):
        data_processor.detect_outliers(
            sample_df,
            method='invalid',
            threshold=3.0,
            handle_method='mark'
        )


def test_detect_outliers_invalid_handle_method(data_processor, sample_df):
    """测试无效的异常值处理方法"""
    with pytest.raises(DataProcessingError, match="不支持的异常值处理方法"):
        data_processor.detect_outliers(
            sample_df,
            method='zscore',
            threshold=3.0,
            handle_method='invalid'
        )


@pytest.mark.parametrize("method,columns", [
    ('zscore', None),
    ('minmax', ['close']),
    ('robust', ['close', 'volume']),
    ('zscore', ['volume'])
])
def test_normalize_data(data_processor, sample_df, method, columns):
    """测试数据标准化"""
    # 标准化数据
    normalized_df = data_processor.normalize_data(
        sample_df,
        method=method,
        columns=columns
    )
    
    # 验证结果
    assert normalized_df is not None
    assert normalized_df.shape == sample_df.shape
    
    # 确定要验证的列
    if columns is None:
        columns = sample_df.select_dtypes(include=[np.number]).columns
    
    # 验证标准化结果
    for col in columns:
        series = normalized_df[col].dropna()
        if method == 'zscore':
            # Z-score标准化后的均值应接近0，标准差应接近1
            assert abs(series.mean()) < 0.1
            assert abs(series.std() - 1) < 0.1
        elif method == 'minmax':
            # Min-Max标准化后的值应在0-1之间
            assert series.min() >= -0.0001  # 允许一点数值误差
            assert series.max() <= 1.0001
        elif method == 'robust':
            # 稳健标准化后的中位数应接近0
            assert abs(series.median()) < 0.1


def test_normalize_data_invalid_method(data_processor, sample_df):
    """测试无效的标准化方法"""
    with pytest.raises(DataProcessingError, match="不支持的标准化方法"):
        data_processor.normalize_data(
            sample_df,
            method='invalid'
        )


def test_normalize_data_invalid_columns(data_processor, sample_df):
    """测试无效的列名"""
    with pytest.raises(DataProcessingError, match="以下列不存在"):
        data_processor.normalize_data(
            sample_df,
            method='zscore',
            columns=['invalid_column']
        )


def test_normalize_data_non_numeric(data_processor, sample_df):
    """测试非数值列标准化"""
    with pytest.raises(DataProcessingError, match="以下列不是数值类型"):
        data_processor.normalize_data(
            sample_df,
            method='zscore',
            columns=['category']
        )


def test_merge_data_on_index(data_processor, multi_dfs):
    """测试按索引合并数据"""
    # 按索引合并
    merged_df = data_processor.merge_data(
        multi_dfs,
        merge_on='index',
        how='outer'
    )
    
    # 验证结果
    assert merged_df is not None
    # 验证列数（应该是所有数据框的列数之和）
    expected_cols = sum(df.shape[1] for df in multi_dfs.values())
    assert merged_df.shape[1] == expected_cols


@pytest.mark.parametrize("how", ['outer', 'inner', 'left', 'right'])
def test_merge_data_different_joins(data_processor, multi_dfs, how):
    """测试不同的合并方式"""
    # 合并数据
    merged_df = data_processor.merge_data(
        multi_dfs,
        merge_on='index',
        how=how
    )
    
    # 验证结果
    assert merged_df is not None
    if how == 'outer':
        # 外连接应该包含所有时间点
        expected_len = len(
            pd.date_range(
                start=min(df.index.min() for df in multi_dfs.values()),
                end=max(df.index.max() for df in multi_dfs.values())
            )
        )
        assert len(merged_df) == expected_len
    elif how == 'inner':
        # 内连接应该只包含共同的时间点
        expected_len = len(
            pd.date_range(
                start=max(df.index.min() for df in multi_dfs.values()),
                end=min(df.index.max() for df in multi_dfs.values())
            )
        )
        assert len(merged_df) == expected_len


def test_merge_data_on_column(data_processor):
    """测试按列合并数据"""
    # 创建测试数据
    df1 = pd.DataFrame({
        'id': [1, 2, 3],
        'value1': [10, 20, 30]
    })
    df2 = pd.DataFrame({
        'id': [2, 3, 4],
        'value2': [200, 300, 400]
    })
    
    # 按列合并
    merged_df = data_processor.merge_data(
        {'df1': df1, 'df2': df2},
        merge_on='id',
        how='outer'
    )
    
    # 验证结果
    assert merged_df is not None
    assert 'id' in merged_df.columns
    assert 'value1' in merged_df.columns
    assert 'value2' in merged_df.columns
    assert len(merged_df) == 4  # 应该有4个唯一的id


def test_merge_data_invalid_column(data_processor, multi_dfs):
    """测试无效的合并列"""
    with pytest.raises(DataProcessingError, match="列 'invalid_column' 不在"):
        data_processor.merge_data(
            multi_dfs,
            merge_on='invalid_column'
        )


def test_filter_data_single_condition(data_processor, sample_df):
    """测试单一条件筛选"""
    # 使用单一条件筛选
    filtered_df = data_processor.filter_data(
        sample_df,
        conditions={'category': 'A'}
    )
    
    # 验证结果
    assert filtered_df is not None
    assert all(filtered_df['category'] == 'A')


def test_filter_data_multiple_conditions(data_processor, sample_df):
    """测试多条件筛选"""
    # 使用多个条件筛选
    filtered_df = data_processor.filter_data(
        sample_df,
        conditions={
            'category': 'A',
            'close': ('>', 105)
        }
    )
    
    # 验证结果
    assert filtered_df is not None
    assert all(filtered_df['category'] == 'A')
    assert all(filtered_df['close'] > 105)


@pytest.mark.parametrize("operator", ['and', 'or'])
def test_filter_data_operators(data_processor, sample_df, operator):
    """测试不同的条件组合操作符"""
    # 使用不同操作符筛选
    filtered_df = data_processor.filter_data(
        sample_df,
        conditions={
            'category': 'A',
            'close': ('>', 105)
        },
        operator=operator
    )
    
    # 验证结果
    assert filtered_df is not None
    if operator == 'and':
        assert all(filtered_df['category'] == 'A')
        assert all(filtered_df['close'] > 105)
    else:  # 'or'
        assert all((filtered_df['category'] == 'A') | (filtered_df['close'] > 105))


@pytest.mark.parametrize("op,value,expected_len", [
    ('==', 'A', 4),
    ('!=', 'A', 6),
    ('>', 105, 4),
    ('>=', 105, 5),
    ('<', 105, 4),
    ('<=', 105, 5),
    ('in', ['A', 'B'], 7),
    ('not in', ['A', 'B'], 3),
    ('contains', 'A', 4),
    ('between', [105, 107], 3)
])
def test_filter_data_operators(data_processor, sample_df, op, value, expected_len):
    """测试不同的筛选操作符"""
    # 根据操作符选择列
    col = 'category' if op in ['==', '!=', 'in', 'not in', 'contains'] else 'close'
    
    # 使用不同操作符筛选
    filtered_df = data_processor.filter_data(
        sample_df,
        conditions={col: (op, value)}
    )
    
    # 验证结果
    assert filtered_df is not None
    assert len(filtered_df) == expected_len


def test_filter_data_invalid_column(data_processor, sample_df):
    """测试无效的筛选列"""
    with pytest.raises(DataProcessingError, match="列 'invalid_column' 不存在"):
        data_processor.filter_data(
            sample_df,
            conditions={'invalid_column': 'value'}
        )


def test_filter_data_invalid_operator(data_processor, sample_df):
    """测试无效的筛选操作符"""
    with pytest.raises(DataProcessingError, match="不支持的操作符"):
        data_processor.filter_data(
            sample_df,
            conditions={'close': ('invalid_op', 100)}
        )


def test_filter_data_invalid_value_type(data_processor, sample_df):
    """测试无效的值类型"""
    with pytest.raises(DataProcessingError, match="需要可迭代的值"):
        data_processor.filter_data(
            sample_df,
            conditions={'category': ('in', 'A')}  # 'in'操作符需要可迭代对象
        )