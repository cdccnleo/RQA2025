"""
数据对齐模块测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

from src.data.alignment.data_aligner import DataAligner, AlignmentMethod, FrequencyType
from src.data.processing.data_processor import DataProcessor, FillMethod
from src.infrastructure.utils.exceptions import DataProcessingError


@pytest.fixture
def data_aligner():
    """数据对齐器fixture"""
    return DataAligner()


@pytest.fixture
def sample_dfs():
    """样本数据框字典fixture"""
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


@pytest.fixture
def multi_freq_dfs():
    """多频率数据框字典fixture"""
    # 日频数据
    dates_daily = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    df_daily = pd.DataFrame({
        'close': np.random.randn(len(dates_daily)) + 100,
        'volume': np.random.randint(1000, 10000, len(dates_daily))
    }, index=dates_daily)
    
    # 小时频数据
    dates_hourly = pd.date_range(start='2023-01-01', end='2023-01-03', freq='H')
    df_hourly = pd.DataFrame({
        'price': np.random.randn(len(dates_hourly)) + 100,
        'trades': np.random.randint(10, 1000, len(dates_hourly))
    }, index=dates_hourly)
    
    # 分钟频数据
    dates_minute = pd.date_range(start='2023-01-01', end='2023-01-01 12:00:00', freq='T')
    df_minute = pd.DataFrame({
        'bid': np.random.randn(len(dates_minute)) + 100,
        'ask': np.random.randn(len(dates_minute)) + 101
    }, index=dates_minute)
    
    return {
        'daily': df_daily,
        'hourly': df_hourly,
        'minute': df_minute
    }


def test_align_time_series_outer(data_aligner, sample_dfs):
    """测试外部对齐时间序列"""
    # 对齐数据
    aligned = data_aligner.align_time_series(
        sample_dfs,
        freq=FrequencyType.DAILY,
        method=AlignmentMethod.OUTER,
        fill_method='ffill'
    )
    
    # 验证结果
    assert len(aligned) == len(sample_dfs)
    
    # 验证所有数据框有相同的索引
    expected_index = pd.date_range(
        start=min(df.index.min() for df in sample_dfs.values()),
        end=max(df.index.max() for df in sample_dfs.values()),
        freq='D'
    )
    
    for name, df in aligned.items():
        assert df.index.equals(expected_index)
        assert not df.isnull().all().all()  # 确保不是全部为NaN


def test_align_time_series_inner(data_aligner, sample_dfs):
    """测试内部对齐时间序列"""
    # 对齐数据
    aligned = data_aligner.align_time_series(
        sample_dfs,
        freq='D',
        method='inner',
        fill_method=None
    )
    
    # 验证结果
    assert len(aligned) == len(sample_dfs)
    
    # 验证所有数据框有相同的索引
    expected_start = max(df.index.min() for df in sample_dfs.values())
    expected_end = min(df.index.max() for df in sample_dfs.values())
    expected_index = pd.date_range(start=expected_start, end=expected_end, freq='D')
    
    for name, df in aligned.items():
        assert df.index.equals(expected_index)


@pytest.mark.parametrize("method,expected_start,expected_end", [
    (AlignmentMethod.OUTER, '2023-01-01', '2023-01-15'),
    (AlignmentMethod.INNER, '2023-01-05', '2023-01-10'),
    (AlignmentMethod.LEFT, '2023-01-01', '2023-01-10'),
    (AlignmentMethod.RIGHT, '2023-01-05', '2023-01-15')
])
def test_align_time_series_methods(data_aligner, sample_dfs, method, expected_start, expected_end):
    """测试不同的对齐方法"""
    # 对齐数据
    aligned = data_aligner.align_time_series(
        sample_dfs,
        freq='D',
        method=method,
        fill_method=None
    )
    
    # 验证结果
    assert len(aligned) == len(sample_dfs)
    
    # 验证索引范围
    expected_index = pd.date_range(start=expected_start, end=expected_end, freq='D')
    
    for name, df in aligned.items():
        assert df.index.equals(expected_index)


def test_align_time_series_with_custom_dates(data_aligner, sample_dfs):
    """测试使用自定义日期范围对齐"""
    # 自定义日期范围
    start_date = '2023-01-04'
    end_date = '2023-01-11'
    
    # 对齐数据
    aligned = data_aligner.align_time_series(
        sample_dfs,
        freq='D',
        method='outer',
        fill_method=None,
        start_date=start_date,
        end_date=end_date
    )
    
    # 验证结果
    assert len(aligned) == len(sample_dfs)
    
    # 验证索引范围
    expected_index = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for name, df in aligned.items():
        assert df.index.equals(expected_index)


def test_align_time_series_with_different_fill_methods(data_aligner, sample_dfs):
    """测试使用不同的填充方法"""
    # 为每个数据源指定不同的填充方法
    fill_methods = {
        'price': 'ffill',
        'ohlc': 'bfill',
        'sentiment': 'mean'
    }
    
    # 对齐数据
    aligned = data_aligner.align_time_series(
        sample_dfs,
        freq='D',
        method='outer',
        fill_method=fill_methods
    )
    
    # 验证结果
    assert len(aligned) == len(sample_dfs)
    
    # 验证填充方法的效果
    # 这里只是简单检查是否有NaN值，实际上应该更详细地验证每种填充方法的效果
    for name, df in aligned.items():
        assert not df.isnull().all().all()


def test_align_time_series_invalid_method(data_aligner, sample_dfs):
    """测试无效的对齐方法"""
    with pytest.raises(DataProcessingError, match="不支持的对齐方法"):
        data_aligner.align_time_series(
            sample_dfs,
            freq='D',
            method='invalid',
            fill_method=None
        )


def test_align_and_merge(data_aligner, sample_dfs):
    """测试对齐并合并"""
    # 对齐并合并
    merged = data_aligner.align_and_merge(
        sample_dfs,
        freq='D',
        method='outer',
        fill_method='ffill'
    )
    
    # 验证结果
    assert isinstance(merged, pd.DataFrame)
    
    # 验证列数（应该是所有数据框的列数之和）
    expected_cols = sum(df.shape[1] for df in sample_dfs.values())
    assert merged.shape[1] == expected_cols
    
    # 验证索引范围
    expected_index = pd.date_range(
        start=min(df.index.min() for df in sample_dfs.values()),
        end=max(df.index.max() for df in sample_dfs.values()),
        freq='D'
    )
    assert merged.index.equals(expected_index)


@pytest.mark.parametrize("method,how", [
    (AlignmentMethod.OUTER, 'outer'),
    (AlignmentMethod.INNER, 'inner'),
    (AlignmentMethod.LEFT, 'inner'),
    (AlignmentMethod.RIGHT, 'inner')
])
def test_align_and_merge_methods(data_aligner, sample_dfs, method, how):
    """测试不同方法的对齐并合并"""
    # 对齐并合并
    merged = data_aligner.align_and_merge(
        sample_dfs,
        freq='D',
        method=method,
        fill_method='ffill'
    )
    
    # 验证结果
    assert isinstance(merged, pd.DataFrame)
    
    # 验证列数（应该是所有数据框的列数之和）
    expected_cols = sum(df.shape[1] for df in sample_dfs.values())
    assert merged.shape[1] == expected_cols


def test_align_to_reference(data_aligner, sample_dfs):
    """测试对齐到参考数据框"""
    # 使用第一个数据框作为参考
    reference_df = sample_dfs['price']
    target_dfs = {k: v for k, v in sample_dfs.items() if k != 'price'}
    
    # 对齐到参考数据框
    aligned = data_aligner.align_to_reference(
        reference_df,
        target_dfs,
        fill_method='ffill'
    )
    
    # 验证结果
    assert len(aligned) == len(target_dfs) + 1  # +1 是因为包含了参考数据框
    assert 'reference' in aligned
    
    # 验证所有数据框有相同的索引
    for name, df in aligned.items():
        assert df.index.equals(reference_df.index)


def test_align_to_reference_with_freq(data_aligner, sample_dfs):
    """测试使用指定频率对齐到参考数据框"""
    # 使用第一个数据框作为参考
    reference_df = sample_dfs['price']
    target_dfs = {k: v for k, v in sample_dfs.items() if k != 'price'}
    
    # 对齐到参考数据框，使用指定频率
    aligned = data_aligner.align_to_reference(
        reference_df,
        target_dfs,
        freq='D',
        fill_method='ffill'
    )
    
    # 验证结果
    assert len(aligned) == len(target_dfs) + 1
    
    # 验证所有数据框有相同的索引
    for name, df in aligned.items():
        assert df.index.equals(reference_df.index)


def test_align_multi_frequency(data_aligner, multi_freq_dfs):
    """测试多频率数据对齐"""
    # 对齐到日频
    aligned = data_aligner.align_multi_frequency(
        multi_freq_dfs,
        target_freq='D',
        resample_methods={
            'daily': 'last',
            'hourly': 'mean',
            'minute': 'max'
        },
        fill_method='ffill'
    )
    
    # 验证结果
    assert len(aligned) == len(multi_freq_dfs)
    
    # 验证所有数据框有相同的频率
    for name, df in aligned.items():
        assert pd.infer_freq(df.index) == 'D'
    
    # 验证所有数据框有相同的索引
    expected_index = pd.date_range(
        start=min(df.index.min() for df in aligned.values()),
        end=max(df.index.max() for df in aligned.values()),
        freq='D'
    )
    
    for name, df in aligned.items():
        assert df.index.equals(expected_index)


@pytest.mark.parametrize("target_freq", [
    FrequencyType.DAILY,
    FrequencyType.HOURLY,
    FrequencyType.WEEKLY
])
def test_align_multi_frequency_different_targets(data_aligner, multi_freq_dfs, target_freq):
    """测试不同目标频率的多频率数据对齐"""
    # 对齐到指定频率
    aligned = data_aligner.align_multi_frequency(
        multi_freq_dfs,
        target_freq=target_freq,
        fill_method='ffill'
    )
    
    # 验证结果
    assert len(aligned) == len(multi_freq_dfs)
    
    # 验证所有数据框有相同的频率
    expected_freq = target_freq.value
    for name, df in aligned.items():
        # 注意：对于某些频率，pd.infer_freq可能返回None，特别是当数据点不足时
        if len(df) > 1:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq is not None:
                assert inferred_freq == expected_freq
    
    # 验证所有数据框有相同的索引范围
    min_date = min(df.index.min() for df in aligned.values())
    max_date = max(df.index.max() for df in aligned.values())
    
    for name, df in aligned.items():
        assert df.index.min() == min_date
        assert df.index.max() == max_date


def test_alignment_history(data_aligner, sample_dfs):
    """测试对齐历史记录"""
    # 执行对齐操作
    data_aligner.align_time_series(
        sample_dfs,
        freq='D',
        method='outer',
        fill_method='ffill'
    )
    
    # 获取历史记录
    history = data_aligner.get_alignment_history()
    
    # 验证结果
    assert len(history) == 1
    assert 'timestamp' in history[0]
    assert 'input_sources' in history[0]
    assert 'output_sources' in history[0]
    assert 'freq' in history[0]
    assert 'method' in history[0]
    assert 'fill_method' in history[0]
    assert 'start_date' in history[0]
    assert 'end_date' in history[0]
    assert 'input_shapes' in history[0]
    assert 'output_shapes' in history[0]


def test_alignment_history_limit(data_aligner, sample_dfs):
    """测试对齐历史记录限制"""
    # 执行多次对齐操作
    for i in range(3):
        data_aligner.align_time_series(
            sample_dfs,
            freq='D',
            method='outer',
            fill_method='ffill'
        )
    
    # 获取限制数量的历史记录
    history = data_aligner.get_alignment_history(limit=2)
    
    # 验证结果
    assert len(history) == 2


def test_save_load_alignment_history(data_aligner, sample_dfs, tmpdir):
    """测试保存和加载对齐历史记录"""
    # 执行对齐操作
    data_aligner.align_time_series(
        sample_dfs,
        freq='D',
        method='outer',
        fill_method='ffill'
    )
    
    # 保存历史记录
    history_file = tmpdir.join("alignment_history.json")
    data_aligner.save_alignment_history(history_file)
    
    # 验证文件存在
    assert os.path.exists(history_file)
    
    # 创建新的对齐器
    new_aligner = DataAligner()
    
    # 加载历史记录
    new_aligner.load_alignment_history(history_file)
    
    # 验证历史记录已加载
    assert len(new_aligner.get_alignment_history()) > 0
    assert new_aligner.get_alignment_history() == data_aligner.get_alignment_history()


@patch('src.data.processing.data_processor.DataProcessor.fill_missing')
def test_mock_processor(mock_fill_missing, data_aligner, sample_dfs):
    """测试使用Mock对象"""
    # 设置Mock对象的行为
    mock_fill_missing.side_effect = lambda df, method, **kwargs: df  # 简单地返回输入数据框
    
    # 执行对齐操作
    aligned = data_aligner.align_time_series(
        sample_dfs,
        freq='D',
        method='outer',
        fill_method='ffill'
    )
    
    # 验证Mock对象被调用
    assert mock_fill_missing.called
    
    # 验证结果
    assert len(aligned) == len(sample_dfs)


def test_align_time_series_non_datetime_index(data_aligner):
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
        data_aligner.align_time_series(
            {'df1': df1, 'df2': df2},
            freq='D',
            method='outer'
        )


def test_align_to_reference_non_datetime_index(data_aligner, sample_dfs):
    """测试参考数据框非日期时间索引"""
    # 创建带有非日期时间索引的参考数据框
    reference_df = pd.DataFrame({
        'close': [100, 101, 102],
        'volume': [1000, 1100, 1200]
    }, index=[1, 2, 3])
    
    # 尝试对齐
    with pytest.raises(DataProcessingError, match="无法转换为DatetimeIndex"):
        data_aligner.align_to_reference(
            reference_df,
            {k: v for k, v in sample_dfs.items() if k != 'price'},
            fill_method='ffill'
        )


def test_align_multi_frequency_non_datetime_index(data_aligner):
    """测试多频率对齐非日期时间索引"""
    # 创建带有非日期时间索引的数据框
    df1 = pd.DataFrame({
        'close': [100, 101, 102],
        'volume': [1000, 1100, 1200]
    }, index=[1, 2, 3])
    
    # 尝试对齐
    with pytest.raises(DataProcessingError, match="无法转换为DatetimeIndex"):
        data_aligner.align_multi_frequency(
            {'df1': df1},
            target_freq='D',
            fill_method='ffill'
        )


def test_align_and_merge_error(data_aligner, sample_dfs):
    """测试对齐并合并错误"""
    # 模拟错误
    with patch.object(data_aligner, 'align_time_series', side_effect=DataProcessingError("模拟错误")):
        with pytest.raises(DataProcessingError, match="对齐并合并失败"):
            data_aligner.align_and_merge(
                sample_dfs,
                freq='D',
                method='outer',
                fill_method='ffill'
            )


def test_save_alignment_history_error(data_aligner, sample_dfs):
    """测试保存历史记录错误"""
    # 执行对齐操作
    data_aligner.align_time_series(
        sample_dfs,
        freq='D',
        method='outer',
        fill_method='ffill'
    )
    
    # 尝试保存到无效路径
    invalid_path = "/invalid/path/history.json"
    data_aligner.save_alignment_history(invalid_path)
    # 由于我们在save_alignment_history中捕获了异常并记录日志，这里不会抛出异常
    # 我们只需要验证历史记录仍然存在
    assert len(data_aligner.get_alignment_history()) > 0


def test_load_alignment_history_error(data_aligner):
    """测试加载历史记录错误"""
    # 尝试从不存在的文件加载
    non_existent_file = "non_existent.json"
    data_aligner.load_alignment_history(non_existent_file)
    # 由于我们在load_alignment_history中捕获了异常并记录日志，这里不会抛出异常
    # 我们只需要验证历史记录为空
    assert len(data_aligner.get_alignment_history()) == 0