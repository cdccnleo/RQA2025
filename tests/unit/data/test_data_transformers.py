# -*- coding: utf-8 -*-
"""
数据转换器测试
测试各种数据转换功能
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
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.data.transformers.data_transformer import (
    DataTransformer,
    DataFrameTransformer,
    TimeSeriesTransformer,
    FeatureTransformer,
    NormalizationTransformer,
    MissingValueTransformer,
    DateColumnTransformer
)


class TestDataTransformerBase:
    """测试数据转换器基类"""

    def test_abstract_methods(self):
        """测试抽象方法"""
        # DataTransformer是抽象类，不能直接实例化
        with pytest.raises(TypeError):
            DataTransformer()

    def test_base_initialization(self):
        """测试基类初始化"""
        # 使用具体子类进行测试
        transformer = DataFrameTransformer()
        assert transformer.config == {}

    def test_custom_config(self):
        """测试自定义配置"""
        config = {'param1': 'value1', 'param2': 42}
        transformer = DataFrameTransformer(config)
        assert transformer.config == config


class TestDataFrameTransformer:
    """测试DataFrame转换器"""

    def setup_method(self):
        """设置测试方法"""
        self.transformer = DataFrameTransformer()

    def test_transform_basic_dataframe(self):
        """测试基础DataFrame转换"""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['x', 'y', 'z']
        })

        result = self.transformer.transform(df)
        pd.testing.assert_frame_equal(result, df)  # 默认实现应该返回原数据

    def test_transform_empty_dataframe(self):
        """测试空DataFrame转换"""
        df = pd.DataFrame()
        result = self.transformer.transform(df)
        assert result.empty

    def test_fit_and_transform(self):
        """测试拟合和转换"""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

        # 拟合
        self.transformer.fit(df)

        # 转换
        result = self.transformer.transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_fit_transform_combined(self):
        """测试拟合转换组合方法"""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

        result = self.transformer.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)


class TestTimeSeriesTransformer:
    """测试时间序列转换器"""

    def setup_method(self):
        """设置测试方法"""
        self.transformer = TimeSeriesTransformer()

    def test_transform_time_series_data(self):
        """测试时间序列数据转换"""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'price': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=dates)

        result = self.transformer.transform(df)

        # 验证索引保持
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 5

    def test_resample_data(self):
        """测试数据重采样"""
        dates = pd.date_range('2023-01-01', periods=10, freq='h')
        # 使用OHLCV格式的数据
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=dates)

        # 创建配置了重采样的转换器
        transformer = TimeSeriesTransformer({'resample_freq': '2h'})

        result = transformer.transform(df)

        # 验证重采样结果
        assert len(result) == 5  # 10小时数据重采样为2小时间隔应该有5个点


class TestFeatureTransformer:
    """测试特征转换器"""

    def setup_method(self):
        """设置测试方法"""
        self.transformer = FeatureTransformer()

    def test_normalize_features(self):
        """测试特征标准化"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

        self.transformer.config = {'normalize': True}

        result = self.transformer.transform(df)

        # 验证标准化效果（Z-score标准化）
        # 第一个特征的标准差应该接近1
        assert abs(result['feature1'].std() - 1.0) < 0.1
        assert abs(result['feature1'].mean()) < 0.1  # 均值接近0

    def test_scale_features(self):
        """测试特征缩放"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

        self.transformer.config = {'scale_features': ['feature1']}

        result = self.transformer.transform(df)

        # 验证缩放效果（Min-Max标准化）
        assert result['feature1'].min() == 0.0
        assert result['feature1'].max() == 1.0
        # feature2 应该保持不变
        assert result['feature2'].equals(df['feature2'])


class TestNormalizationTransformer:
    """测试标准化转换器"""

    def setup_method(self):
        """设置测试方法"""
        self.transformer = NormalizationTransformer()

    def test_minmax_normalization(self):
        """测试最小最大标准化"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

        self.transformer.config = {'method': 'minmax'}

        result = self.transformer.transform(df)

        # 验证标准化结果
        assert result['feature1'].min() == 0.0
        assert result['feature1'].max() == 1.0
        assert result['feature2'].min() == 0.0
        assert result['feature2'].max() == 1.0

    def test_zscore_normalization(self):
        """测试Z-score标准化"""
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5]
        })

        self.transformer.config = {'method': 'zscore'}

        result = self.transformer.transform(df)

        # 验证Z-score标准化
        assert abs(result['feature'].mean()) < 1e-10  # 均值接近0
        assert abs(result['feature'].std() - 1.0) < 1e-10  # 标准差为1

    def test_robust_normalization(self):
        """测试鲁棒标准化"""
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 100]  # 包含异常值
        })

        self.transformer.config = {'method': 'robust'}

        result = self.transformer.transform(df)

        # 验证鲁棒标准化（基于中位数和四分位距）
        # 中位数应该是3，IQR应该是4-2=2
        expected_median = 3.0
        expected_iqr = 2.0

        # 验证中位数为0
        assert abs(result['feature'].median()) < 1e-10

    def test_fit_transform_workflow(self):
        """测试拟合转换工作流"""
        train_df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5]
        })

        test_df = pd.DataFrame({
            'feature': [2.5, 3.5]
        })

        # 拟合训练数据
        self.transformer.fit(train_df)

        # 转换测试数据
        result = self.transformer.transform(test_df)

        # 验证使用训练数据的参数进行转换
        assert len(result) == 2


class TestMissingValueTransformer:
    """测试缺失值转换器"""

    def setup_method(self):
        """设置测试方法"""
        self.transformer = MissingValueTransformer()

    def test_fill_with_mean(self):
        """测试均值填充"""
        df = pd.DataFrame({
            'feature': [1, 2, np.nan, 4, 5]
        })

        self.transformer.config = {'strategy': 'mean'}

        result = self.transformer.transform(df)

        # 验证缺失值被填充
        assert not result['feature'].isnull().any()
        assert result['feature'].iloc[2] == 3.0  # (1+2+4+5)/4 = 3.0

    def test_fill_with_median(self):
        """测试中位数填充"""
        df = pd.DataFrame({
            'feature': [1, 2, np.nan, 4, 5]
        })

        self.transformer.config = {'strategy': 'median'}

        result = self.transformer.transform(df)

        # 验证缺失值被填充为中位数
        assert not result['feature'].isnull().any()
        assert result['feature'].iloc[2] == 3.0  # 中位数是3

    def test_fill_with_constant(self):
        """测试常量填充"""
        df = pd.DataFrame({
            'feature': [1, 2, np.nan, 4, 5]
        })

        self.transformer.config = {'strategy': 'constant', 'fill_value': 999}

        result = self.transformer.transform(df)

        # 验证缺失值被填充为指定常量
        assert not result['feature'].isnull().any()
        assert result['feature'].iloc[2] == 999

    def test_drop_missing(self):
        """测试删除缺失值"""
        df = pd.DataFrame({
            'feature': [1, 2, np.nan, 4, 5]
        })

        self.transformer.config = {'strategy': 'drop'}

        result = self.transformer.transform(df)

        # 验证缺失值行被删除
        assert not result['feature'].isnull().any()
        assert len(result) == 4  # 从5行变为4行

    def test_multiple_columns(self):
        """测试多列缺失值处理"""
        df = pd.DataFrame({
            'feature1': [1, np.nan, 3],
            'feature2': [np.nan, 2, 3],
            'feature3': [1, 2, np.nan]
        })

        self.transformer.config = {'strategy': 'mean'}

        result = self.transformer.transform(df)

        # 验证所有列的缺失值都被填充
        assert not result.isnull().any().any()


class TestDateColumnTransformer:
    """测试日期列转换器"""

    def setup_method(self):
        """设置测试方法"""
        self.transformer = DateColumnTransformer()

    def test_parse_date_strings(self):
        """测试解析日期字符串"""
        df = pd.DataFrame({
            'date_str': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'value': [1, 2, 3]
        })

        self.transformer.config = {'date_columns': ['date_str']}

        result = self.transformer.transform(df)

        # 验证日期列被转换为datetime类型
        assert pd.api.types.is_datetime64_any_dtype(result['date_str'])

    def test_extract_date_features(self):
        """测试提取日期特征"""
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': [1, 2, 3]
        })

        self.transformer.config = {
            'date_columns': ['date'],
            'extract_features': ['year', 'month', 'day', 'weekday']
        }

        result = self.transformer.transform(df)

        # 验证日期特征被提取
        assert 'date_year' in result.columns
        assert 'date_month' in result.columns
        assert 'date_day' in result.columns
        assert 'date_weekday' in result.columns

        # 验证特征值正确
        assert result['date_year'].iloc[0] == 2023
        assert result['date_month'].iloc[0] == 1
        assert result['date_day'].iloc[0] == 1

    def test_multiple_date_formats(self):
        """测试多种日期格式"""
        df = pd.DataFrame({
            'date1': ['2023/01/01', '2023/01/02'],
            'date2': ['01-Jan-2023', '02-Jan-2023'],
            'value': [1, 2]
        })

        self.transformer.config = {
            'date_columns': ['date1', 'date2'],
            'date_formats': ['%Y/%m/%d', '%d-%b-%Y']
        }

        result = self.transformer.transform(df)

        # 验证所有日期列都被正确解析
        assert pd.api.types.is_datetime64_any_dtype(result['date1'])
        assert pd.api.types.is_datetime64_any_dtype(result['date2'])

    def test_timezone_handling(self):
        """测试时区处理"""
        df = pd.DataFrame({
            'datetime_str': ['2023-01-01 10:00:00+00:00', '2023-01-02 11:00:00+00:00'],
            'value': [1, 2]
        })

        self.transformer.config = {
            'date_columns': ['datetime_str'],
            'timezone': 'UTC'
        }

        result = self.transformer.transform(df)

        # 验证时区信息被保留
        assert result['datetime_str'].dt.tz is not None


class TestTransformerIntegration:
    """测试转换器集成"""

    def test_chained_transformers(self):
        """测试转换器链式调用"""
        df = pd.DataFrame({
            'price': [100, np.nan, 102, 103, 104],
            'volume': [1000, 1100, np.nan, 1300, 1400]
        })

        # 创建转换器链
        missing_transformer = MissingValueTransformer({'strategy': 'mean'})
        normalization_transformer = NormalizationTransformer({'method': 'minmax'})

        # 依次应用转换器
        result1 = missing_transformer.transform(df)
        result2 = normalization_transformer.transform(result1)

        # 验证最终结果
        assert not result2.isnull().any().any()  # 没有缺失值
        assert result2['price'].min() >= 0.0  # 标准化到[0,1]范围
        assert result2['price'].max() <= 1.0
        assert result2['volume'].min() >= 0.0
        assert result2['volume'].max() <= 1.0

    def test_transformer_pipeline(self):
        """测试转换器管道"""
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'price': [100, np.nan, 104],
            'volume': [1000, 1100, 1200]
        })

        # 创建转换管道
        transformers = [
            DateColumnTransformer({'date_columns': ['date']}),
            MissingValueTransformer({'strategy': 'mean'}),
            NormalizationTransformer({'method': 'minmax'})
        ]

        result = df.copy()
        for transformer in transformers:
            result = transformer.transform(result)

        # 验证管道结果
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
        assert not result[['price', 'volume']].isnull().any().any()
        assert result['price'].min() >= 0.0
        assert result['price'].max() <= 1.0
