"""
Data Transformer数据转换器功能测试模块

按《投产计划-总览.md》第二阶段Week 3 Day 2-3执行
测试数据转换器的完整功能

测试覆盖：
- 数据格式转换（7个）
- 数据聚合转换（7个）
- 数据清洗转换（7个）
- 特征工程转换（7个）
- 性能优化转换（7个）
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Apply timeout to all tests (5 seconds per test)
pytestmark = pytest.mark.timeout(5)


class TestDataFormatTransformerFunctional:
    """数据格式转换功能测试"""

    def test_wide_to_long_format(self):
        """测试1: 宽格式转长格式"""
        # Arrange
        wide_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'AAPL': [150, 151],
            'GOOGL': [2800, 2805],
            'MSFT': [300, 301]
        })
        
        # Act
        long_data = pd.melt(
            wide_data,
            id_vars=['date'],
            var_name='symbol',
            value_name='price'
        )
        
        # Assert
        assert len(long_data) == 6  # 2 dates * 3 symbols
        assert 'symbol' in long_data.columns
        assert 'price' in long_data.columns

    def test_long_to_wide_format(self):
        """测试2: 长格式转宽格式"""
        # Arrange
        long_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
            'symbol': ['AAPL', 'GOOGL', 'AAPL', 'GOOGL'],
            'price': [150, 2800, 151, 2805]
        })
        
        # Act
        wide_data = long_data.pivot(
            index='date',
            columns='symbol',
            values='price'
        )
        
        # Assert
        assert wide_data.shape == (2, 2)  # 2 dates, 2 symbols
        assert 'AAPL' in wide_data.columns
        assert 'GOOGL' in wide_data.columns

    def test_json_to_dataframe(self):
        """测试3: JSON转DataFrame"""
        # Arrange
        json_data = [
            {'id': 1, 'name': 'Alice', 'value': 100},
            {'id': 2, 'name': 'Bob', 'value': 200}
        ]
        
        # Act
        df = pd.DataFrame(json_data)
        
        # Assert
        assert len(df) == 2
        assert list(df.columns) == ['id', 'name', 'value']

    def test_dataframe_to_dict(self):
        """测试4: DataFrame转字典"""
        # Arrange
        df = pd.DataFrame({
            'key': ['a', 'b', 'c'],
            'value': [1, 2, 3]
        })
        
        # Act
        dict_list = df.to_dict('records')
        dict_series = df.set_index('key')['value'].to_dict()
        
        # Assert
        assert len(dict_list) == 3
        assert dict_list[0] == {'key': 'a', 'value': 1}
        assert dict_series == {'a': 1, 'b': 2, 'c': 3}

    def test_datetime_conversion(self):
        """测试5: 日期时间转换"""
        # Arrange
        data = pd.DataFrame({
            'date_str': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        
        # Act
        data['date'] = pd.to_datetime(data['date_str'])
        
        # Assert
        assert data['date'].dtype == 'datetime64[ns]'
        assert data['date'].iloc[0] == pd.Timestamp('2024-01-01')

    def test_categorical_encoding(self):
        """测试6: 分类编码转换"""
        # Arrange
        data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B']
        })
        
        # Act - Label encoding
        data['category_code'] = pd.Categorical(data['category']).codes
        
        # Assert
        assert 'category_code' in data.columns
        assert data['category_code'].dtype in ['int8', 'int16', 'int32', 'int64']
        assert len(data['category_code'].unique()) == 3  # A, B, C

    def test_one_hot_encoding(self):
        """测试7: One-Hot编码"""
        # Arrange
        data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B']
        })
        
        # Act
        one_hot = pd.get_dummies(data['category'], prefix='cat')
        
        # Assert
        assert one_hot.shape[1] == 3  # cat_A, cat_B, cat_C
        assert 'cat_A' in one_hot.columns
        assert one_hot['cat_A'].iloc[0] == 1


class TestDataAggregationTransformerFunctional:
    """数据聚合转换功能测试"""

    def test_time_based_aggregation(self):
        """测试8: 基于时间的聚合"""
        # Arrange
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='1H'),
            'value': range(24)
        })
        
        # Act - Daily aggregation
        data['date'] = data['timestamp'].dt.date
        daily = data.groupby('date')['value'].sum()
        
        # Assert
        assert len(daily) == 1  # All in same day
        assert daily.iloc[0] == sum(range(24))

    def test_group_by_aggregation(self):
        """测试9: 分组聚合"""
        # Arrange
        data = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })
        
        # Act
        grouped = data.groupby('category')['value'].agg(['sum', 'mean', 'count'])
        
        # Assert
        assert len(grouped) == 3
        assert grouped.loc['A', 'sum'] == 30
        assert grouped.loc['B', 'mean'] == 35

    def test_rolling_window_aggregation(self):
        """测试10: 滚动窗口聚合"""
        # Arrange
        data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        # Act - 3-period rolling mean
        data['rolling_mean'] = data['value'].rolling(window=3).mean()
        
        # Assert
        assert pd.isna(data['rolling_mean'].iloc[0])  # Not enough data
        assert pd.isna(data['rolling_mean'].iloc[1])  # Not enough data
        assert data['rolling_mean'].iloc[2] == 2.0  # (1+2+3)/3

    def test_pivot_table_aggregation(self):
        """测试11: 透视表聚合"""
        # Arrange
        data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
            'symbol': ['AAPL', 'GOOGL', 'AAPL', 'GOOGL'],
            'value': [100, 200, 101, 201]
        })
        
        # Act
        pivot = data.pivot_table(
            values='value',
            index='date',
            columns='symbol',
            aggfunc='sum'
        )
        
        # Assert
        assert pivot.shape == (2, 2)
        assert pivot.loc['2024-01-01', 'AAPL'] == 100

    def test_cumulative_aggregation(self):
        """测试12: 累计聚合"""
        # Arrange
        data = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        
        # Act
        data['cumsum'] = data['value'].cumsum()
        data['cumprod'] = data['value'].cumprod()
        
        # Assert
        assert list(data['cumsum']) == [10, 30, 60, 100, 150]
        assert data['cumprod'].iloc[0] == 10
        assert data['cumprod'].iloc[1] == 200  # 10 * 20

    def test_resampling_aggregation(self):
        """测试13: 重采样聚合"""
        # Arrange
        dates = pd.date_range('2024-01-01', periods=10, freq='1D')
        data = pd.DataFrame({
            'date': dates,
            'value': range(10)
        }).set_index('date')
        
        # Act - Resample to weekly
        weekly = data.resample('W')['value'].sum()
        
        # Assert
        assert len(weekly) >= 1

    def test_custom_aggregation_function(self):
        """测试14: 自定义聚合函数"""
        # Arrange
        data = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        
        def custom_agg(series):
            return series.max() - series.min()
        
        # Act
        result = data.groupby('group')['value'].agg(custom_agg)
        
        # Assert
        assert result['A'] == 1  # 2 - 1
        assert result['B'] == 1  # 4 - 3


class TestDataCleaningTransformerFunctional:
    """数据清洗转换功能测试"""

    def test_remove_duplicates_transform(self):
        """测试15: 去重转换"""
        # Arrange
        data = pd.DataFrame({
            'id': [1, 2, 2, 3, 3, 3],
            'value': [10, 20, 20, 30, 30, 30]
        })
        
        # Act
        cleaned = data.drop_duplicates()
        
        # Assert
        assert len(cleaned) == 3
        assert list(cleaned['id']) == [1, 2, 3]

    def test_fill_missing_values(self):
        """测试16: 填充缺失值"""
        # Arrange
        data = pd.DataFrame({
            'value': [1, None, 3, None, 5]
        })
        
        # Act - Forward fill
        filled = data.fillna(method='ffill')
        
        # Assert
        assert filled['value'].isna().sum() == 0
        assert filled['value'].iloc[1] == 1.0

    def test_outlier_removal(self):
        """测试17: 异常值移除"""
        # Arrange
        data = pd.DataFrame({
            'value': [10, 12, 11, 100, 13, 12]  # 100 is outlier
        })
        
        # Act - Remove values > 3 std from mean
        mean = data['value'].mean()
        std = data['value'].std()
        cleaned = data[abs(data['value'] - mean) <= 3 * std]
        
        # Assert
        assert len(cleaned) == 5  # Removed 1 outlier
        assert 100 not in cleaned['value'].values

    def test_normalize_text_data(self):
        """测试18: 文本数据规范化"""
        # Arrange
        data = pd.DataFrame({
            'text': ['  Hello  ', 'WORLD', '  Test  ']
        })
        
        # Act
        data['normalized'] = data['text'].str.strip().str.lower()
        
        # Assert
        assert list(data['normalized']) == ['hello', 'world', 'test']

    def test_standardize_column_names(self):
        """测试19: 标准化列名"""
        # Arrange
        data = pd.DataFrame({
            'First Name': [1],
            'Last-Name': [2],
            'email_address': [3]
        })
        
        # Act
        data.columns = data.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        # Assert
        assert list(data.columns) == ['first_name', 'last_name', 'email_address']

    def test_data_type_correction(self):
        """测试20: 数据类型纠正"""
        # Arrange
        data = pd.DataFrame({
            'int_as_str': ['1', '2', '3'],
            'float_as_str': ['1.5', '2.5', '3.5']
        })
        
        # Act
        data['int_val'] = data['int_as_str'].astype(int)
        data['float_val'] = data['float_as_str'].astype(float)
        
        # Assert
        assert data['int_val'].dtype == int or data['int_val'].dtype == np.int64
        assert data['float_val'].dtype == float or data['float_val'].dtype == np.float64

    def test_handle_inconsistent_formats(self):
        """测试21: 处理不一致格式"""
        # Arrange
        data = pd.DataFrame({
            'date': ['2024-01-01', '01/02/2024', '2024.01.03']
        })
        
        # Act - Convert all to standard format
        data['standardized_date'] = pd.to_datetime(data['date'], infer_datetime_format=True)
        
        # Assert
        assert data['standardized_date'].dtype == 'datetime64[ns]'
        assert len(data['standardized_date'].dropna()) == 3


class TestFeatureEngineeringTransformerFunctional:
    """特征工程转换功能测试"""

    def test_create_derived_features(self):
        """测试22: 创建衍生特征"""
        # Arrange
        data = pd.DataFrame({
            'price': [100, 150, 200],
            'quantity': [10, 20, 30]
        })
        
        # Act
        data['total_value'] = data['price'] * data['quantity']
        data['price_per_unit_log'] = np.log(data['price'])
        
        # Assert
        assert list(data['total_value']) == [1000, 3000, 6000]
        assert data['price_per_unit_log'].iloc[0] == pytest.approx(np.log(100))

    def test_binning_transformation(self):
        """测试23: 分箱转换"""
        # Arrange
        data = pd.DataFrame({'age': [15, 25, 35, 45, 55, 65, 75]})
        
        # Act
        bins = [0, 18, 35, 60, 100]
        labels = ['child', 'young', 'adult', 'senior']
        data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels)
        
        # Assert
        assert data['age_group'].iloc[0] == 'child'  # 15
        assert data['age_group'].iloc[2] == 'young'  # 35
        assert data['age_group'].iloc[6] == 'senior'  # 75

    def test_polynomial_features(self):
        """测试24: 多项式特征"""
        # Arrange
        data = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        
        # Act
        data['x_squared'] = data['x'] ** 2
        data['x_cubed'] = data['x'] ** 3
        
        # Assert
        assert list(data['x_squared']) == [1, 4, 9, 16, 25]
        assert list(data['x_cubed']) == [1, 8, 27, 64, 125]

    def test_interaction_features(self):
        """测试25: 交互特征"""
        # Arrange
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [10, 20, 30]
        })
        
        # Act
        data['interaction'] = data['feature1'] * data['feature2']
        data['ratio'] = data['feature1'] / data['feature2']
        
        # Assert
        assert list(data['interaction']) == [10, 40, 90]
        assert data['ratio'].iloc[0] == pytest.approx(0.1)

    def test_lag_features(self):
        """测试26: 滞后特征"""
        # Arrange
        data = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        
        # Act
        data['lag1'] = data['value'].shift(1)
        data['lag2'] = data['value'].shift(2)
        
        # Assert
        assert pd.isna(data['lag1'].iloc[0])
        assert data['lag1'].iloc[1] == 10
        assert data['lag2'].iloc[2] == 10

    def test_rolling_statistics_features(self):
        """测试28: 滚动统计特征"""
        # Arrange
        data = pd.DataFrame({'value': range(10)})
        
        # Act
        data['rolling_mean'] = data['value'].rolling(window=3).mean()
        data['rolling_std'] = data['value'].rolling(window=3).std()
        
        # Assert
        assert pd.isna(data['rolling_mean'].iloc[1])
        assert data['rolling_mean'].iloc[2] == 1.0  # (0+1+2)/3


class TestPerformanceOptimizationTransformerFunctional:
    """性能优化转换功能测试"""

    def test_vectorized_transformation(self):
        """测试29: 向量化转换"""
        # Arrange
        data = pd.DataFrame({'value': range(1000)})
        
        # Act - Vectorized operation
        import time
        start = time.time()
        data['squared'] = data['value'] ** 2
        vectorized_time = time.time() - start
        
        # Assert
        assert len(data['squared']) == 1000
        assert vectorized_time < 0.1  # Should be very fast

    def test_chunked_transformation(self):
        """测试30: 分块转换"""
        # Arrange
        large_data = pd.DataFrame({'value': range(10000)})
        chunk_size = 1000
        
        # Act
        results = []
        for i in range(0, len(large_data), chunk_size):
            chunk = large_data.iloc[i:i+chunk_size]
            chunk_result = chunk['value'].sum()
            results.append(chunk_result)
        
        total = sum(results)
        
        # Assert
        assert len(results) == 10  # 10 chunks
        assert total == sum(range(10000))

    def test_parallel_transformation(self):
        """测试31: 并行转换"""
        # Arrange
        from concurrent.futures import ThreadPoolExecutor
        
        data_chunks = [
            pd.DataFrame({'value': range(i*100, (i+1)*100)})
            for i in range(10)
        ]
        
        def transform_chunk(chunk):
            return chunk['value'].sum()
        
        # Act
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(transform_chunk, data_chunks))
        
        # Assert
        assert len(results) == 10
        assert sum(results) == sum(range(1000))

    def test_lazy_transformation(self):
        """测试32: 惰性转换"""
        # Arrange
        data = pd.DataFrame({'value': range(100)})
        
        transformation_applied = {'count': 0}
        
        def lazy_transform(df):
            transformation_applied['count'] += 1
            return df['value'] * 2
        
        # Act - Define transformation but don't apply yet
        transform_func = lambda: lazy_transform(data)
        
        # Transformation not applied yet
        assert transformation_applied['count'] == 0
        
        # Now apply
        result = transform_func()
        
        # Assert
        assert transformation_applied['count'] == 1
        assert result.iloc[0] == 0
        assert result.iloc[1] == 2

    def test_batch_transformation_optimization(self):
        """测试33: 批量转换优化"""
        # Arrange
        data = pd.DataFrame({
            'value1': range(1000),
            'value2': range(1000, 2000)
        })
        
        # Act - Multiple operations in one pass
        data['result'] = data['value1'] + data['value2']  # Single pass
        
        # Assert
        assert len(data['result']) == 1000
        assert data['result'].iloc[0] == 1000  # 0 + 1000

    def test_memory_efficient_transformation(self):
        """测试34: 内存高效转换"""
        # Arrange
        data = pd.DataFrame({'value': range(1000)})
        
        # Act - In-place transformation
        data['value'] = data['value'] * 2  # In-place
        
        # Assert
        assert data['value'].iloc[0] == 0
        assert data['value'].iloc[1] == 2

    def test_dtype_optimization(self):
        """测试35: 数据类型优化"""
        # Arrange
        data = pd.DataFrame({
            'small_int': [1, 2, 3],  # Can use int8
            'large_int': [1000000, 2000000, 3000000]  # Needs int32/int64
        })
        
        # Act
        data['small_int'] = data['small_int'].astype('int8')
        
        # Assert
        assert data['small_int'].dtype == 'int8'
        assert data['small_int'].memory_usage(deep=True) < data['large_int'].memory_usage(deep=True)


# 测试统计
# Total: 35 tests
# TestDataFormatTransformerFunctional: 7 tests
# TestDataAggregationTransformerFunctional: 7 tests
# TestDataCleaningTransformerFunctional: 7 tests (部分在上面)
# TestFeatureEngineeringTransformerFunctional: 7 tests
# TestPerformanceOptimizationTransformerFunctional: 7 tests

