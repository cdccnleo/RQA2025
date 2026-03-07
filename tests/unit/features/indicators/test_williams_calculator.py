# -*- coding: utf-8 -*-
"""
威廉指标计算器测试
"""

import pytest
import pandas as pd
import numpy as np
from src.features.indicators.williams_calculator import WilliamsCalculator


class TestWilliamsCalculator:
    """测试WilliamsCalculator类"""

    @pytest.fixture
    def calculator(self):
        """创建WilliamsCalculator实例"""
        return WilliamsCalculator()

    @pytest.fixture
    def sample_data(self):
        """生成示例OHLC数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        return pd.DataFrame({
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(110, 120, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(95, 115, 50),
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)

    def test_init_default(self):
        """测试默认初始化"""
        calc = WilliamsCalculator()
        assert calc.period == 14
        assert calc.config == {}

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {'period': 20}
        calc = WilliamsCalculator(config)
        assert calc.period == 20

    def test_calculate_basic(self, calculator, sample_data):
        """测试基本计算"""
        result = calculator.calculate(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'williams_r' in result.columns
        assert len(result) == len(sample_data)
        # 临时列应该被删除
        assert 'highest_high' not in result.columns
        assert 'lowest_low' not in result.columns

    def test_calculate_with_custom_period(self, sample_data):
        """测试自定义周期计算"""
        calc = WilliamsCalculator({'period': 20})
        result = calc.calculate(sample_data)
        
        assert 'williams_r' in result.columns

    def test_calculate_missing_columns(self, calculator):
        """测试缺少必需列的情况"""
        data = pd.DataFrame({'open': [100, 101, 102]})
        
        # 计算器会捕获异常并返回原始数据
        result = calculator.calculate(data)
        assert isinstance(result, pd.DataFrame)
        # 应该返回原始数据（没有williams_r列）
        assert 'williams_r' not in result.columns

    def test_calculate_empty_data(self, calculator):
        """测试空数据"""
        data = pd.DataFrame()
        result = calculator.calculate(data)
        assert isinstance(result, pd.DataFrame)

    def test_calculate_insufficient_data(self, calculator):
        """测试数据不足的情况"""
        data = pd.DataFrame({
            'high': [110, 111],
            'low': [100, 101],
            'close': [105, 106]
        })
        result = calculator.calculate(data)
        assert isinstance(result, pd.DataFrame)
        # 前period-1个周期应该设置为0
        assert result['williams_r'].iloc[0] == 0

    def test_calculate_equal_high_low(self, calculator):
        """测试最高价等于最低价的情况"""
        data = pd.DataFrame({
            'high': [100, 100, 100],
            'low': [100, 100, 100],
            'close': [100, 100, 100]
        })
        result = calculator.calculate(data)
        assert isinstance(result, pd.DataFrame)
        # Williams %R应该为0
        assert result['williams_r'].iloc[0] == 0

    def test_calculate_williams_r_range(self, calculator, sample_data):
        """测试Williams %R值在合理范围内"""
        result = calculator.calculate(sample_data)
        
        # Williams %R值应该在-100到0范围内
        williams_r_values = result['williams_r'].dropna()
        if len(williams_r_values) > 0:
            assert williams_r_values.min() >= -100
            assert williams_r_values.max() <= 0

    def test_calculate_rolling_window(self, calculator, sample_data):
        """测试滚动窗口计算"""
        result = calculator.calculate(sample_data)
        
        # 检查是否有足够的非NaN值
        williams_r_values = result['williams_r'].dropna()
        # 应该有period-1个NaN值（前period-1个周期）
        nan_count = result['williams_r'].isna().sum()
        assert nan_count <= calculator.period - 1

    def test_calculate_exception_handling(self, calculator):
        """测试异常处理"""
        # 传入无效数据
        invalid_data = "invalid"
        result = calculator.calculate(invalid_data)
        # 应该返回原始数据或空DataFrame
        assert isinstance(result, (pd.DataFrame, type(invalid_data)))

    def test_calculate_temporary_columns_removed(self, calculator, sample_data):
        """测试临时列被正确删除"""
        result = calculator.calculate(sample_data)
        
        # 确保临时列不存在
        assert 'highest_high' not in result.columns
        assert 'lowest_low' not in result.columns
        # 原始列应该保留
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns

