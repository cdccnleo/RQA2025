"""
数据转换器模块的边界测试
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
from unittest.mock import Mock, MagicMock

from src.data.transformers.data_transformer import (
    DataTransformer,
    DataFrameTransformer,
    TimeSeriesTransformer,
    FeatureTransformer,
    NormalizationTransformer,
    MissingValueTransformer,
    DateColumnTransformer,
)


class TestDataTransformer:
    """测试 DataTransformer 抽象基类"""

    def test_data_transformer_is_abstract(self):
        """测试 DataTransformer 是抽象类"""
        with pytest.raises(TypeError):
            DataTransformer()

    def test_data_transformer_init_with_config(self):
        """测试带配置初始化"""
        class ConcreteTransformer(DataTransformer):
            def transform(self, data):
                return data
        
        transformer = ConcreteTransformer(config={"key": "value"})
        assert transformer.config == {"key": "value"}

    def test_data_transformer_init_without_config(self):
        """测试不带配置初始化"""
        class ConcreteTransformer(DataTransformer):
            def transform(self, data):
                return data
        
        transformer = ConcreteTransformer()
        assert transformer.config == {}

    def test_data_transformer_fit(self):
        """测试 fit 方法"""
        class ConcreteTransformer(DataTransformer):
            def transform(self, data):
                return data
        
        transformer = ConcreteTransformer()
        # fit 方法应该不抛出异常
        transformer.fit([1, 2, 3])

    def test_data_transformer_fit_transform(self):
        """测试 fit_transform 方法"""
        class ConcreteTransformer(DataTransformer):
            def transform(self, data):
                return [x * 2 for x in data]
        
        transformer = ConcreteTransformer()
        result = transformer.fit_transform([1, 2, 3])
        assert result == [2, 4, 6]


class TestDataFrameTransformer:
    """测试 DataFrameTransformer 类"""

    def test_dataframe_transformer_init_default(self):
        """测试默认初始化"""
        transformer = DataFrameTransformer()
        assert transformer.columns_to_drop == []
        assert transformer.columns_to_keep is None
        assert transformer.fill_method == 'ffill'

    def test_dataframe_transformer_init_custom(self):
        """测试自定义配置初始化"""
        config = {
            'columns_to_drop': ['col1', 'col2'],
            'columns_to_keep': ['col3', 'col4'],
            'fill_method': 'bfill'
        }
        transformer = DataFrameTransformer(config=config)
        assert transformer.columns_to_drop == ['col1', 'col2']
        assert transformer.columns_to_keep == ['col3', 'col4']
        assert transformer.fill_method == 'bfill'

    def test_dataframe_transformer_transform_invalid_input(self):
        """测试无效输入"""
        transformer = DataFrameTransformer()
        with pytest.raises(ValueError, match="输入数据必须是pandas DataFrame"):
            transformer.transform([1, 2, 3])

    def test_dataframe_transformer_transform_empty_dataframe(self):
        """测试空DataFrame"""
        transformer = DataFrameTransformer()
        df = pd.DataFrame()
        result = transformer.transform(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_dataframe_transformer_transform_drop_columns(self):
        """测试删除列"""
        transformer = DataFrameTransformer(config={'columns_to_drop': ['col1']})
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        result = transformer.transform(df)
        assert 'col1' not in result.columns
        assert 'col2' in result.columns

    def test_dataframe_transformer_transform_keep_columns(self):
        """测试保留列"""
        transformer = DataFrameTransformer(config={'columns_to_keep': ['col1']})
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        result = transformer.transform(df)
        assert 'col1' in result.columns
        assert 'col2' not in result.columns

    def test_dataframe_transformer_transform_fill_method_ffill(self):
        """测试前向填充"""
        transformer = DataFrameTransformer(config={'fill_method': 'ffill'})
        df = pd.DataFrame({'col1': [1, None, 3]})
        result = transformer.transform(df)
        assert result['col1'].isna().sum() == 0

    def test_dataframe_transformer_transform_fill_method_bfill(self):
        """测试后向填充"""
        transformer = DataFrameTransformer(config={'fill_method': 'bfill'})
        df = pd.DataFrame({'col1': [1, None, 3]})
        result = transformer.transform(df)
        assert result['col1'].isna().sum() == 0

    def test_dataframe_transformer_transform_fill_method_interpolate(self):
        """测试插值填充"""
        transformer = DataFrameTransformer(config={'fill_method': 'interpolate'})
        df = pd.DataFrame({'col1': [1.0, None, 3.0]})
        result = transformer.transform(df)
        assert result['col1'].isna().sum() == 0

    def test_dataframe_transformer_transform_fill_method_default(self):
        """测试默认填充（填充0）"""
        transformer = DataFrameTransformer(config={'fill_method': 'unknown'})
        df = pd.DataFrame({'col1': [1, None, 3]})
        result = transformer.transform(df)
        assert result['col1'].isna().sum() == 0


class TestTimeSeriesTransformer:
    """测试 TimeSeriesTransformer 类"""

    def test_timeseries_transformer_init_default(self):
        """测试默认初始化"""
        transformer = TimeSeriesTransformer()
        assert transformer.resample_freq is None
        assert transformer.fill_method == 'ffill'

    def test_timeseries_transformer_transform_invalid_input(self):
        """测试无效输入"""
        transformer = TimeSeriesTransformer()
        with pytest.raises(ValueError, match="输入数据必须是pandas DataFrame"):
            transformer.transform([1, 2, 3])

    def test_timeseries_transformer_transform_with_date_column(self):
        """测试带日期列转换"""
        transformer = TimeSeriesTransformer()
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'value': [1, 2]
        })
        result = transformer.transform(df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_timeseries_transformer_transform_resample(self):
        """测试重采样"""
        transformer = TimeSeriesTransformer(config={'resample_freq': '1D'})
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='H'),
            'open': range(10),
            'high': range(10, 20),
            'low': range(20, 30),
            'close': range(30, 40),
            'volume': range(40, 50)
        })
        result = transformer.transform(df)
        assert len(result) <= len(df)


class TestFeatureTransformer:
    """测试 FeatureTransformer 类"""

    def test_feature_transformer_init_default(self):
        """测试默认初始化"""
        transformer = FeatureTransformer()
        assert transformer.normalize is False
        assert transformer.scale_features == []

    def test_feature_transformer_transform_invalid_input(self):
        """测试无效输入"""
        transformer = FeatureTransformer()
        with pytest.raises(ValueError, match="输入数据必须是pandas DataFrame"):
            transformer.transform([1, 2, 3])

    def test_feature_transformer_transform_normalize(self):
        """测试标准化"""
        transformer = FeatureTransformer(config={'normalize': True})
        df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        result = transformer.transform(df)
        # 标准化后均值应该接近0，标准差应该接近1
        assert abs(result['col1'].mean()) < 0.01
        assert abs(result['col1'].std() - 1.0) < 0.01

    def test_feature_transformer_transform_scale_features(self):
        """测试特征缩放"""
        transformer = FeatureTransformer(config={'scale_features': ['col1']})
        df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        result = transformer.transform(df)
        # 缩放后最小值应该接近0，最大值应该接近1
        assert abs(result['col1'].min()) < 0.01
        assert abs(result['col1'].max() - 1.0) < 0.01

    def test_feature_transformer_transform_zero_std(self):
        """测试零标准差（不进行标准化）"""
        transformer = FeatureTransformer(config={'normalize': True})
        df = pd.DataFrame({'col1': [1, 1, 1, 1, 1]})
        result = transformer.transform(df)
        # 标准差为0时不应该改变数据
        assert (result['col1'] == df['col1']).all()


class TestNormalizationTransformer:
    """测试 NormalizationTransformer 类"""

    def test_normalization_transformer_init_default(self):
        """测试默认初始化"""
        transformer = NormalizationTransformer()
        assert transformer.method == 'zscore'

    def test_normalization_transformer_transform_invalid_input(self):
        """测试无效输入"""
        transformer = NormalizationTransformer()
        with pytest.raises(ValueError, match="输入数据必须是pandas DataFrame"):
            transformer.transform([1, 2, 3])

    def test_normalization_transformer_transform_zscore(self):
        """测试Z-score标准化"""
        transformer = NormalizationTransformer(config={'method': 'zscore'})
        df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        result = transformer.transform(df)
        assert abs(result['col1'].mean()) < 0.01
        assert abs(result['col1'].std() - 1.0) < 0.01

    def test_normalization_transformer_transform_minmax(self):
        """测试MinMax标准化"""
        transformer = NormalizationTransformer(config={'method': 'minmax'})
        df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        result = transformer.transform(df)
        assert abs(result['col1'].min()) < 0.01
        assert abs(result['col1'].max() - 1.0) < 0.01

    def test_normalization_transformer_transform_robust(self):
        """测试Robust标准化"""
        transformer = NormalizationTransformer(config={'method': 'robust'})
        df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        result = transformer.transform(df)
        # Robust标准化后中位数应该接近0
        assert abs(result['col1'].median()) < 0.01


class TestMissingValueTransformer:
    """测试 MissingValueTransformer 类"""

    def test_missing_value_transformer_init_default(self):
        """测试默认初始化"""
        transformer = MissingValueTransformer()
        assert transformer.method == 'ffill'

    def test_missing_value_transformer_transform_invalid_input(self):
        """测试无效输入"""
        transformer = MissingValueTransformer()
        with pytest.raises(ValueError, match="输入数据必须是pandas DataFrame"):
            transformer.transform([1, 2, 3])

    def test_missing_value_transformer_transform_ffill(self):
        """测试前向填充"""
        transformer = MissingValueTransformer(config={'method': 'ffill'})
        df = pd.DataFrame({'col1': [1, None, 3]})
        result = transformer.transform(df)
        assert result['col1'].isna().sum() == 0

    def test_missing_value_transformer_transform_bfill(self):
        """测试后向填充"""
        transformer = MissingValueTransformer(config={'method': 'bfill'})
        df = pd.DataFrame({'col1': [1, None, 3]})
        result = transformer.transform(df)
        assert result['col1'].isna().sum() == 0

    def test_missing_value_transformer_transform_drop(self):
        """测试删除缺失值"""
        transformer = MissingValueTransformer(config={'method': 'drop'})
        df = pd.DataFrame({'col1': [1, None, 3]})
        result = transformer.transform(df)
        assert result['col1'].isna().sum() == 0
        assert len(result) < len(df)

    def test_missing_value_transformer_transform_fill(self):
        """测试填充指定值"""
        transformer = MissingValueTransformer(config={'method': 'fill', 'fill_value': 999})
        df = pd.DataFrame({'col1': [1, None, 3]})
        result = transformer.transform(df)
        assert result['col1'].isna().sum() == 0
        assert 999 in result['col1'].values

    def test_missing_value_transformer_transform_mean(self):
        """测试均值填充"""
        transformer = MissingValueTransformer(config={'method': 'mean'})
        df = pd.DataFrame({'col1': [1.0, None, 3.0]})
        result = transformer.transform(df)
        assert result['col1'].isna().sum() == 0

    def test_missing_value_transformer_transform_median(self):
        """测试中位数填充"""
        transformer = MissingValueTransformer(config={'method': 'median'})
        df = pd.DataFrame({'col1': [1.0, None, 3.0]})
        result = transformer.transform(df)
        assert result['col1'].isna().sum() == 0

    def test_missing_value_transformer_transform_constant(self):
        """测试常量填充"""
        transformer = MissingValueTransformer(config={'method': 'constant', 'fill_value': 0})
        df = pd.DataFrame({'col1': [1, None, 3]})
        result = transformer.transform(df)
        assert result['col1'].isna().sum() == 0


class TestDateColumnTransformer:
    """测试 DateColumnTransformer 类"""

    def test_date_column_transformer_init_default(self):
        """测试默认初始化"""
        transformer = DateColumnTransformer()
        assert transformer.date_column == 'date'
        assert transformer.format is None

    def test_date_column_transformer_transform_invalid_input(self):
        """测试无效输入"""
        transformer = DateColumnTransformer()
        with pytest.raises(ValueError, match="输入数据必须是pandas DataFrame"):
            transformer.transform([1, 2, 3])

    def test_date_column_transformer_transform_single_column(self):
        """测试单列转换"""
        transformer = DateColumnTransformer(config={'date_column': 'date'})
        df = pd.DataFrame({'date': ['2024-01-01', '2024-01-02']})
        result = transformer.transform(df)
        assert pd.api.types.is_datetime64_any_dtype(result['date'])

    def test_date_column_transformer_transform_multiple_columns(self):
        """测试多列转换"""
        transformer = DateColumnTransformer(config={'date_columns': ['date1', 'date2']})
        df = pd.DataFrame({
            'date1': ['2024-01-01', '2024-01-02'],
            'date2': ['2024-02-01', '2024-02-02']
        })
        result = transformer.transform(df)
        assert pd.api.types.is_datetime64_any_dtype(result['date1'])
        assert pd.api.types.is_datetime64_any_dtype(result['date2'])

    def test_date_column_transformer_transform_extract_features(self):
        """测试提取特征"""
        transformer = DateColumnTransformer(config={
            'date_column': 'date',
            'extract_features': ['year', 'month', 'day']
        })
        df = pd.DataFrame({'date': ['2024-01-15', '2024-02-20']})
        result = transformer.transform(df)
        assert 'date_year' in result.columns
        assert 'date_month' in result.columns
        assert 'date_day' in result.columns

    def test_date_column_transformer_transform_nonexistent_column(self):
        """测试不存在的列"""
        transformer = DateColumnTransformer(config={'date_column': 'nonexistent'})
        df = pd.DataFrame({'other_col': [1, 2]})
        result = transformer.transform(df)
        # 应该不抛出异常，只是跳过不存在的列
        assert 'nonexistent' not in result.columns


class TestEdgeCases:
    """测试边界情况"""

    def test_dataframe_transformer_all_none_values(self):
        """测试全部为None值"""
        transformer = DataFrameTransformer()
        df = pd.DataFrame({'col1': [None, None, None]})
        result = transformer.transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_feature_transformer_empty_dataframe(self):
        """测试空DataFrame"""
        transformer = FeatureTransformer()
        df = pd.DataFrame()
        result = transformer.transform(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_normalization_transformer_single_value(self):
        """测试单个值"""
        transformer = NormalizationTransformer()
        df = pd.DataFrame({'col1': [5]})
        result = transformer.transform(df)
        # 单个值无法标准化，应该保持原值或处理异常
        assert isinstance(result, pd.DataFrame)

    def test_missing_value_transformer_all_missing(self):
        """测试全部缺失值"""
        transformer = MissingValueTransformer(config={'method': 'fill', 'fill_value': 0})
        df = pd.DataFrame({'col1': [None, None, None]})
        result = transformer.transform(df)
        assert result['col1'].isna().sum() == 0

    def test_date_column_transformer_invalid_date_format(self):
        """测试无效日期格式"""
        transformer = DateColumnTransformer(config={'date_column': 'date'})
        df = pd.DataFrame({'date': ['invalid', '2024-01-01']})
        result = transformer.transform(df)
        # 应该能够处理无效日期（转换为NaT）
        assert isinstance(result, pd.DataFrame)


def test_timeseries_transformer_fill_method_bfill():
    """测试 TimeSeriesTransformer（后向填充）"""
    transformer = TimeSeriesTransformer(config={'fill_method': 'bfill'})
    df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'open': [None, None, 102],  # 前两个为None，bfill会从后面填充
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [102, 103, 104],
        'volume': [1000, 2000, 3000]
    })
    result = transformer.transform(df)
    # 检查缺失值被填充（bfill从后向前填充）
    assert result['open'].isna().sum() == 0


def test_timeseries_transformer_fill_method_interpolate():
    """测试 TimeSeriesTransformer（插值填充）"""
    transformer = TimeSeriesTransformer(config={'fill_method': 'interpolate'})
    df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'open': [100.0, None, 102.0],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [102, 103, 104],
        'volume': [1000, 2000, 3000]
    })
    result = transformer.transform(df)
    # 检查缺失值被填充
    assert result.isna().sum().sum() == 0


def test_timeseries_transformer_fill_method_default():
    """测试 TimeSeriesTransformer（默认填充）"""
    transformer = TimeSeriesTransformer(config={'fill_method': 'unknown'})
    df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'open': [100, None],
        'high': [105, 106],
        'low': [95, 96],
        'close': [102, 103],
        'volume': [1000, 2000]
    })
    result = transformer.transform(df)
    # 检查缺失值被填充（默认填充0）
    assert result.isna().sum().sum() == 0


def test_missing_value_transformer_interpolate():
    """测试 MissingValueTransformer（插值策略）"""
    transformer = MissingValueTransformer(config={'method': 'interpolate'})
    df = pd.DataFrame({'col1': [1.0, None, 3.0]})
    result = transformer.transform(df)
    assert result['col1'].isna().sum() == 0


def test_missing_value_transformer_default_fallback():
    """测试 MissingValueTransformer（默认回退）"""
    transformer = MissingValueTransformer(config={'method': 'unknown_strategy'})
    df = pd.DataFrame({'col1': [1, None, 3]})
    result = transformer.transform(df)
    # 应该回退到 ffill
    assert result['col1'].isna().sum() == 0


def test_date_column_transformer_date_columns_string():
    """测试 DateColumnTransformer（date_columns 为字符串）"""
    transformer = DateColumnTransformer(config={'date_columns': 'date1'})
    df = pd.DataFrame({'date1': ['2024-01-01', '2024-01-02']})
    result = transformer.transform(df)
    assert pd.api.types.is_datetime64_any_dtype(result['date1'])


def test_date_column_transformer_date_formats_dict():
    """测试 DateColumnTransformer（date_formats 为字典）"""
    transformer = DateColumnTransformer(config={
        'date_columns': ['date1', 'date2'],
        'date_formats': {'date1': '%Y-%m-%d', 'date2': None}
    })
    df = pd.DataFrame({
        'date1': ['2024-01-01', '2024-01-02'],
        'date2': ['2024-02-01', '2024-02-02']
    })
    result = transformer.transform(df)
    assert pd.api.types.is_datetime64_any_dtype(result['date1'])
    assert pd.api.types.is_datetime64_any_dtype(result['date2'])


def test_date_column_transformer_date_formats_list():
    """测试 DateColumnTransformer（date_formats 为列表）"""
    transformer = DateColumnTransformer(config={
        'date_columns': ['date1', 'date2'],
        'date_formats': ['%Y-%m-%d', None]
    })
    df = pd.DataFrame({
        'date1': ['2024-01-01', '2024-01-02'],
        'date2': ['2024-02-01', '2024-02-02']
    })
    result = transformer.transform(df)
    assert pd.api.types.is_datetime64_any_dtype(result['date1'])
    assert pd.api.types.is_datetime64_any_dtype(result['date2'])


def test_date_column_transformer_date_formats_tuple():
    """测试 DateColumnTransformer（date_formats 为元组）"""
    transformer = DateColumnTransformer(config={
        'date_columns': ['date1', 'date2'],
        'date_formats': ('%Y-%m-%d', None)
    })
    df = pd.DataFrame({
        'date1': ['2024-01-01', '2024-01-02'],
        'date2': ['2024-02-01', '2024-02-02']
    })
    result = transformer.transform(df)
    assert pd.api.types.is_datetime64_any_dtype(result['date1'])
    assert pd.api.types.is_datetime64_any_dtype(result['date2'])


def test_date_column_transformer_date_formats_list_short():
    """测试 DateColumnTransformer（date_formats 列表长度不足）"""
    transformer = DateColumnTransformer(config={
        'date_columns': ['date1', 'date2'],
        'date_formats': ['%Y-%m-%d']  # 只有一个格式，第二个应该使用 self.format
    })
    df = pd.DataFrame({
        'date1': ['2024-01-01', '2024-01-02'],
        'date2': ['2024-02-01', '2024-02-02']
    })
    result = transformer.transform(df)
    assert pd.api.types.is_datetime64_any_dtype(result['date1'])
    assert pd.api.types.is_datetime64_any_dtype(result['date2'])


def test_date_column_transformer_date_conversion_exception():
    """测试 DateColumnTransformer（日期转换异常处理）"""
    transformer = DateColumnTransformer(config={
        'date_column': 'date',
        'format': 'invalid_format_string'
    })
    df = pd.DataFrame({'date': ['2024-01-01', '2024-01-02']})
    # 应该捕获异常并使用 errors='coerce'
    result = transformer.transform(df)
    assert isinstance(result, pd.DataFrame)


def test_date_column_transformer_timezone_localize():
    """测试 DateColumnTransformer（时区本地化）"""
    transformer = DateColumnTransformer(config={
        'date_column': 'date',
        'timezone': 'UTC'
    })
    df = pd.DataFrame({'date': ['2024-01-01', '2024-01-02']})
    result = transformer.transform(df)
    # 应该有时区信息
    assert isinstance(result, pd.DataFrame)
    # 检查时区信息（如果转换成功）
    if not result['date'].isna().all():
        # 如果日期列有值，检查时区
        assert hasattr(result['date'].dtype, 'tz') or result['date'].dt.tz is not None or result['date'].isna().any()


def test_date_column_transformer_timezone_convert():
    """测试 DateColumnTransformer（时区转换）"""
    # 创建带时区的日期
    dates = pd.to_datetime(['2024-01-01', '2024-01-02']).tz_localize('UTC')
    transformer = DateColumnTransformer(config={
        'date_column': 'date',
        'timezone': 'Asia/Shanghai'
    })
    df = pd.DataFrame({'date': dates})
    result = transformer.transform(df)
    assert isinstance(result, pd.DataFrame)


def test_date_column_transformer_timezone_typeerror():
    """测试 DateColumnTransformer（时区处理 TypeError）"""
    transformer = DateColumnTransformer(config={
        'date_column': 'date',
        'timezone': 'UTC'
    })
    # 创建包含 NaT 的日期列（可能导致 tz_localize 失败）
    df = pd.DataFrame({'date': ['2024-01-01', 'invalid', '2024-01-03']})
    result = transformer.transform(df)
    # 应该捕获 TypeError 并继续处理
    assert isinstance(result, pd.DataFrame)


def test_date_column_transformer_extract_weekday():
    """测试 DateColumnTransformer（提取 weekday 特征）"""
    transformer = DateColumnTransformer(config={
        'date_column': 'date',
        'extract_features': ['weekday']
    })
    df = pd.DataFrame({'date': ['2024-01-01', '2024-01-02']})
    result = transformer.transform(df)
    assert 'date_weekday' in result.columns


def test_date_column_transformer_extract_hour():
    """测试 DateColumnTransformer（提取 hour 特征）"""
    transformer = DateColumnTransformer(config={
        'date_column': 'date',
        'extract_features': ['hour']
    })
    df = pd.DataFrame({'date': ['2024-01-01 10:00:00', '2024-01-02 15:00:00']})
    result = transformer.transform(df)
    assert 'date_hour' in result.columns


def test_date_column_transformer_extract_minute():
    """测试 DateColumnTransformer（提取 minute 特征）"""
    transformer = DateColumnTransformer(config={
        'date_column': 'date',
        'extract_features': ['minute']
    })
    df = pd.DataFrame({'date': ['2024-01-01 10:30:00', '2024-01-02 15:45:00']})
    result = transformer.transform(df)
    assert 'date_minute' in result.columns


def test_date_column_transformer_extract_second():
    """测试 DateColumnTransformer（提取 second 特征）"""
    transformer = DateColumnTransformer(config={
        'date_column': 'date',
        'extract_features': ['second']
    })
    df = pd.DataFrame({'date': ['2024-01-01 10:30:45', '2024-01-02 15:45:30']})
    result = transformer.transform(df)
    assert 'date_second' in result.columns


def test_date_column_transformer_date_conversion_exception_with_invalid_format(monkeypatch):
    """测试 DateColumnTransformer（日期转换异常，使用无效格式）"""
    transformer = DateColumnTransformer(config={
        'date_column': 'date',
        'format': '%Y-%m-%d'
    })
    df = pd.DataFrame({'date': ['2024-01-01', '2024-01-02']})
    
    # Mock pd.to_datetime 来触发异常（第一次调用）
    call_count = [0]
    original_to_datetime = pd.to_datetime
    
    def _bad_to_datetime(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1 and 'format' in kwargs:
            # 第一次调用（带format）抛出异常
            raise ValueError("Invalid format")
        # 后续调用使用原始函数
        return original_to_datetime(*args, **kwargs)
    
    monkeypatch.setattr(pd, "to_datetime", _bad_to_datetime)
    
    # 应该捕获异常并使用 errors='coerce' 重新调用
    result = transformer.transform(df)
    assert isinstance(result, pd.DataFrame)


def test_date_column_transformer_timezone_typeerror_with_nat(monkeypatch):
    """测试 DateColumnTransformer（时区处理 TypeError，包含 NaT）"""
    transformer = DateColumnTransformer(config={
        'date_column': 'date',
        'timezone': 'UTC'
    })
    # 创建包含 NaT 的日期列
    dates = pd.to_datetime(['2024-01-01', 'invalid', '2024-01-03'], errors='coerce')
    df = pd.DataFrame({'date': dates})
    
    # Mock dt accessor 来触发 TypeError
    original_dt = pd.Series.dt
    
    class MockDT:
        def __init__(self, series):
            self._series = series
        
        @property
        def tz(self):
            return None
        
        def tz_localize(self, tz):
            # 如果包含 NaT，tz_localize 可能失败
            if self._series.isna().any():
                raise TypeError("Cannot localize tz-aware timestamps")
            return self._series.dt.tz_localize(tz)
        
        def tz_convert(self, tz):
            return self._series.dt.tz_convert(tz)
    
    # Mock Series 的 dt 属性
    def _get_dt(self):
        return MockDT(self)
    
    # 为 DataFrame 的 date 列 mock dt 属性
    original_getattr = pd.Series.__getattribute__
    
    def _mock_getattr(self, name):
        if name == 'dt':
            return MockDT(self)
        return original_getattr(self, name)
    
    monkeypatch.setattr(pd.Series, "__getattribute__", _mock_getattr)
    
    # 直接测试时区处理逻辑
    result = transformer.transform(df)
    # 应该捕获 TypeError 并继续处理
    assert isinstance(result, pd.DataFrame)
