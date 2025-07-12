import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.infrastructure.utils.tools import (
    validate_dates, fill_missing_values, convert_to_ordered_dict,
    get_dynamic_dates, safe_divide, calculate_volatility,
    validate_dataframe
)

class TestValidateDates:
    """日期验证函数测试"""

    def test_valid_dates(self):
        """测试有效日期"""
        start_date = "2024-01-01"
        end_date = "2024-01-31"
        
        start, end = validate_dates(start_date, end_date)
        
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert start < end

    def test_invalid_date_format(self):
        """测试无效日期格式"""
        with pytest.raises(ValueError, match="日期格式无效"):
            validate_dates("2024/01/01", "2024-01-31")

    def test_start_after_end(self):
        """测试开始日期晚于结束日期"""
        with pytest.raises(ValueError, match="开始日期不能晚于结束日期"):
            validate_dates("2024-01-31", "2024-01-01")

    def test_same_dates(self):
        """测试相同日期"""
        date_str = "2024-01-01"
        start, end = validate_dates(date_str, date_str)
        
        assert start == end

class TestFillMissingValues:
    """缺失值填充函数测试"""

    def test_ffill_method(self):
        """测试前向填充"""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5],
            'B': [10, 20, np.nan, 40, 50]
        })
        
        result = fill_missing_values(df, 'ffill')
        
        assert not result.isnull().any().any()
        assert result.iloc[1, 0] == 1  # 前向填充
        assert result.iloc[3, 0] == 3  # 前向填充

    def test_bfill_method(self):
        """测试后向填充"""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5],
            'B': [10, 20, np.nan, 40, 50]
        })
        
        result = fill_missing_values(df, 'bfill')
        
        assert not result.isnull().any().any()
        assert result.iloc[1, 0] == 3  # 后向填充
        assert result.iloc[3, 0] == 5  # 后向填充

    def test_mean_method(self):
        """测试均值填充"""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5],
            'B': [10, 20, np.nan, 40, 50]
        })
        
        result = fill_missing_values(df, 'mean')
        
        assert not result.isnull().any().any()
        assert result['A'].mean() == df['A'].mean()

    def test_median_method(self):
        """测试中位数填充"""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5],
            'B': [10, 20, np.nan, 40, 50]
        })
        
        result = fill_missing_values(df, 'median')
        
        assert not result.isnull().any().any()
        assert result['A'].median() == df['A'].median()

    def test_invalid_method(self):
        """测试无效填充方法"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="不支持的填充方法"):
            fill_missing_values(df, 'invalid')

class TestConvertToOrderedDict:
    """有序字典转换函数测试"""

    def test_simple_dict(self):
        """测试简单字典"""
        input_dict = {'c': 3, 'a': 1, 'b': 2}
        
        result = convert_to_ordered_dict(input_dict)
        
        assert list(result.keys()) == ['a', 'b', 'c']

    def test_nested_dict(self):
        """测试嵌套字典"""
        input_dict = {
            'c': {'z': 3, 'y': 2},
            'a': {'x': 1, 'w': 0}
        }
        
        result = convert_to_ordered_dict(input_dict)
        
        assert list(result.keys()) == ['a', 'c']
        assert list(result['a'].keys()) == ['w', 'x']
        assert list(result['c'].keys()) == ['y', 'z']

    def test_non_dict_input(self):
        """测试非字典输入"""
        result = convert_to_ordered_dict("not a dict")
        assert result == "not a dict"

class TestGetDynamicDates:
    """动态日期获取函数测试"""

    def test_default_parameters(self):
        """测试默认参数"""
        start_date, end_date = get_dynamic_dates()
        
        assert isinstance(start_date, str)
        assert isinstance(end_date, str)
        assert len(start_date) == 10  # YYYY-MM-DD格式

    def test_custom_days_back(self):
        """测试自定义回溯天数"""
        start_date, end_date = get_dynamic_dates(days_back=7)
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        assert (end_dt - start_dt).days == 7

    def test_custom_date_format(self):
        """测试自定义日期格式"""
        start_date, end_date = get_dynamic_dates(date_format="%Y/%m/%d")
        
        assert "/" in start_date
        assert "/" in end_date

    def test_no_format(self):
        """测试无格式参数"""
        start_date, end_date = get_dynamic_dates(date_format=None)
        
        assert isinstance(start_date, datetime)
        assert isinstance(end_date, datetime)

class TestSafeDivide:
    """安全除法函数测试"""

    def test_pandas_series_division(self):
        """测试pandas Series除法"""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([2, 5, 0])
        
        result = safe_divide(numerator, denominator)
        
        assert isinstance(result, pd.Series)
        assert result.iloc[0] == 5.0
        assert result.iloc[1] == 4.0
        assert result.iloc[2] == 0.0  # 默认值

    def test_numpy_array_division(self):
        """测试numpy数组除法"""
        numerator = np.array([10, 20, 30])
        denominator = np.array([2, 5, 0])
        
        result = safe_divide(numerator, denominator)
        
        assert isinstance(result, np.ndarray)
        assert result[0] == 5.0
        assert result[1] == 4.0
        assert result[2] == 0.0  # 默认值

    def test_shape_mismatch(self):
        """测试形状不匹配"""
        numerator = pd.Series([1, 2, 3])
        denominator = pd.Series([1, 2])
        
        with pytest.raises(ValueError, match="分子和分母的长度必须相同"):
            safe_divide(numerator, denominator)

    def test_custom_default(self):
        """测试自定义默认值"""
        numerator = pd.Series([10, 20])
        denominator = pd.Series([0, 5])
        
        result = safe_divide(numerator, denominator, default=-1)
        
        assert result.iloc[0] == -1
        assert result.iloc[1] == 4.0

class TestCalculateVolatility:
    """波动率计算函数测试"""

    def test_pandas_series_volatility(self):
        """测试pandas Series波动率计算"""
        prices = pd.Series([100, 101, 99, 102, 98, 103])
        
        result = calculate_volatility(prices, window=3)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)
        assert pd.isna(result.iloc[0])  # 第一个值应该是NaN

    def test_numpy_array_volatility(self):
        """测试numpy数组波动率计算"""
        prices = np.array([100, 101, 99, 102, 98, 103])
        
        result = calculate_volatility(prices, window=3)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)

    def test_dataframe_volatility(self):
        """测试DataFrame波动率计算"""
        prices = pd.DataFrame({'price': [100, 101, 99, 102, 98, 103]})
        
        result = calculate_volatility(prices, window=3)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)

    def test_invalid_dataframe(self):
        """测试无效DataFrame"""
        prices = pd.DataFrame({
            'price1': [100, 101, 99],
            'price2': [200, 201, 199]
        })
        
        with pytest.raises(ValueError, match="DataFrame应只包含一列价格数据"):
            calculate_volatility(prices)

    def test_insufficient_data(self):
        """测试数据不足"""
        prices = pd.Series([100])
        
        with pytest.raises(ValueError, match="价格序列至少需要2个数据点"):
            calculate_volatility(prices)

    def test_window_larger_than_data(self):
        """测试窗口大于数据长度"""
        prices = pd.Series([100, 101, 99])
        
        result = calculate_volatility(prices, window=10)
        
        assert len(result) == len(prices)
        assert pd.isna(result.iloc[0])

class TestValidateDataframe:
    """DataFrame验证函数测试"""

    def test_valid_dataframe(self):
        """测试有效DataFrame"""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        required_columns = ['A', 'B', 'C']
        
        result = validate_dataframe(df, required_columns)
        
        assert result is True

    def test_missing_columns(self):
        """测试缺失列"""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        required_columns = ['A', 'B', 'C']
        
        result = validate_dataframe(df, required_columns)
        
        assert result is False

    def test_empty_dataframe(self):
        """测试空DataFrame"""
        df = pd.DataFrame()
        required_columns = ['A']
        
        result = validate_dataframe(df, required_columns)
        
        assert result is False

    def test_non_dataframe_input(self):
        """测试非DataFrame输入"""
        required_columns = ['A']
        
        result = validate_dataframe("not a dataframe", required_columns)
        
        assert result is False 