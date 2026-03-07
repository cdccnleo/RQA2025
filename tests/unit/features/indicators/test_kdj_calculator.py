# -*- coding: utf-8 -*-
"""
KDJ指标计算器测试
"""

import pytest
import pandas as pd
import numpy as np
from src.features.indicators.kdj_calculator import KDJCalculator


class TestKDJCalculator:
    """测试KDJCalculator类"""

    @pytest.fixture
    def calculator(self):
        """创建KDJCalculator实例"""
        return KDJCalculator()

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
        calc = KDJCalculator()
        assert calc.period == 9
        assert calc.config == {}

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {'period': 14}
        calc = KDJCalculator(config)
        assert calc.period == 14

    def test_calculate_basic(self, calculator, sample_data):
        """测试基本计算"""
        result = calculator.calculate(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'kdj_k' in result.columns
        assert 'kdj_d' in result.columns
        assert 'kdj_j' in result.columns
        assert len(result) == len(sample_data)

    def test_calculate_with_custom_period(self, sample_data):
        """测试自定义周期计算"""
        calc = KDJCalculator({'period': 14})
        result = calc.calculate(sample_data)
        
        assert 'kdj_k' in result.columns
        assert 'kdj_d' in result.columns
        assert 'kdj_j' in result.columns

    def test_calculate_missing_columns(self, calculator):
        """测试缺少必需列的情况"""
        data = pd.DataFrame({'open': [100, 101, 102]})
        
        # 计算器会捕获异常并返回原始数据
        result = calculator.calculate(data)
        assert isinstance(result, pd.DataFrame)
        # 应该返回原始数据（没有KDJ列）
        assert 'kdj_k' not in result.columns

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
        # 前period-1个周期应该设置为50
        assert result['rsv'].iloc[0] == 50

    def test_calculate_equal_high_low(self, calculator):
        """测试最高价等于最低价的情况"""
        data = pd.DataFrame({
            'high': [100, 100, 100],
            'low': [100, 100, 100],
            'close': [100, 100, 100]
        })
        result = calculator.calculate(data)
        assert isinstance(result, pd.DataFrame)
        # RSV应该为50
        assert result['rsv'].iloc[0] == 50

    def test_calculate_kdj_values_range(self, calculator, sample_data):
        """测试KDJ值在合理范围内"""
        result = calculator.calculate(sample_data)
        
        # K、D、J值应该在合理范围内（通常0-100）
        k_values = result['kdj_k'].dropna()
        d_values = result['kdj_d'].dropna()
        j_values = result['kdj_j'].dropna()
        
        if len(k_values) > 0:
            assert k_values.min() >= 0
            assert d_values.min() >= 0

    def test_calculate_rsv_column(self, calculator, sample_data):
        """测试RSV列的计算"""
        result = calculator.calculate(sample_data)
        
        # 检查RSV列是否存在
        assert 'rsv' in result.columns
        
        # RSV值应该在0-100范围内
        rsv_values = result['rsv'].dropna()
        if len(rsv_values) > 0:
            assert rsv_values.min() >= 0
            assert rsv_values.max() <= 100

    def test_calculate_exception_handling(self, calculator):
        """测试异常处理"""
        # 传入无效数据
        invalid_data = "invalid"
        result = calculator.calculate(invalid_data)
        # 应该返回原始数据或空DataFrame
        assert isinstance(result, (pd.DataFrame, type(invalid_data)))

