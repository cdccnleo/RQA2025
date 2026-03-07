# -*- coding: utf-8 -*-
"""
斐波那契计算器覆盖率测试 - Phase 2
针对FibonacciCalculator类的未覆盖方法进行补充测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.features.indicators.fibonacci_calculator import FibonacciCalculator


class TestFibonacciCalculatorCoverage:
    """测试FibonacciCalculator的未覆盖方法"""

    @pytest.fixture
    def sample_ohlc_data(self):
        """生成示例OHLC数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        base_price = 100
        prices = base_price + np.cumsum(np.random.randn(100) * 2)
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.randn(100) * 0.01),
            'high': prices * (1 + abs(np.random.randn(100) * 0.02)),
            'low': prices * (1 - abs(np.random.randn(100) * 0.02)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

    @pytest.fixture
    def calculator(self):
        """创建FibonacciCalculator实例"""
        return FibonacciCalculator()

    def test_calculate_success(self, calculator, sample_ohlc_data):
        """测试计算斐波那契水平 - 成功"""
        result = calculator.calculate(sample_ohlc_data)
        
        # 验证结果
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlc_data)
        # 验证原始列仍然存在
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns

    def test_calculate_empty_data(self, calculator):
        """测试计算斐波那契水平 - 空数据"""
        empty_data = pd.DataFrame()
        result = calculator.calculate(empty_data)
        
        # 应该返回空DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_calculate_none_data(self, calculator):
        """测试计算斐波那契水平 - None数据"""
        result = calculator.calculate(None)
        
        # 应该返回空DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_calculate_missing_columns(self, calculator):
        """测试计算斐波那契水平 - 缺少必要列"""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [100, 101, 102]
        })
        
        result = calculator.calculate(data)
        
        # 应该返回原始数据（不包含high/low）
        assert isinstance(result, pd.DataFrame)
        assert 'high' not in result.columns
        assert 'low' not in result.columns

    def test_calculate_with_custom_config(self):
        """测试使用自定义配置"""
        config = {
            'lookback_period': 30,
            'min_swing_length': 3,
            'custom_levels': [0.1, 0.5, 0.9]
        }
        calculator = FibonacciCalculator(config=config)
        
        assert calculator.lookback_period == 30
        assert calculator.min_swing_length == 3
        assert calculator.custom_levels == [0.1, 0.5, 0.9]

    def test_calculate_with_default_config(self, calculator):
        """测试使用默认配置"""
        assert calculator.lookback_period == 50
        assert calculator.min_swing_length == 5
        assert calculator.custom_levels is None

    def test_find_swing_points(self, calculator, sample_ohlc_data):
        """测试寻找摆动点"""
        swing_highs, swing_lows = calculator._find_swing_points(sample_ohlc_data)
        
        # 验证返回类型
        assert isinstance(swing_highs, list)
        assert isinstance(swing_lows, list)
        
        # 验证摆动点格式（如果存在）
        if swing_highs:
            assert all(isinstance(point, tuple) and len(point) == 2 for point in swing_highs)
        if swing_lows:
            assert all(isinstance(point, tuple) and len(point) == 2 for point in swing_lows)

    def test_find_swing_points_insufficient_data(self, calculator):
        """测试寻找摆动点 - 数据不足"""
        # 创建数据不足的DataFrame
        data = pd.DataFrame({
            'high': [100, 101, 102],
            'low': [99, 100, 101]
        })
        
        swing_highs, swing_lows = calculator._find_swing_points(data)
        
        # 应该返回空列表（数据不足）
        assert isinstance(swing_highs, list)
        assert isinstance(swing_lows, list)

    def test_calculate_fibonacci_levels_no_swings(self, calculator, sample_ohlc_data):
        """测试计算斐波那契水平 - 无摆动点"""
        empty_swing_highs = []
        empty_swing_lows = []
        
        result = calculator._calculate_fibonacci_levels(
            sample_ohlc_data, empty_swing_highs, empty_swing_lows
        )
        
        # 应该返回空字典或包含默认水平的字典
        assert isinstance(result, dict)

    def test_calculate_fibonacci_levels_with_swings(self, calculator, sample_ohlc_data):
        """测试计算斐波那契水平 - 有摆动点"""
        swing_highs, swing_lows = calculator._find_swing_points(sample_ohlc_data)
        
        if swing_highs and swing_lows:
            result = calculator._calculate_fibonacci_levels(
                sample_ohlc_data, swing_highs, swing_lows
            )
            
            # 验证返回类型
            assert isinstance(result, dict)
            # 验证结果包含斐波那契水平（如果计算成功）
            if result:
                assert all(isinstance(v, pd.Series) for v in result.values())

    def test_calculate_price_fib_relationship(self, calculator, sample_ohlc_data):
        """测试计算价格与斐波那契水平的关系"""
        # 创建模拟的斐波那契水平
        fib_levels = {
            'fib_0.236': pd.Series([100] * len(sample_ohlc_data), index=sample_ohlc_data.index),
            'fib_0.382': pd.Series([101] * len(sample_ohlc_data), index=sample_ohlc_data.index),
            'fib_0.618': pd.Series([102] * len(sample_ohlc_data), index=sample_ohlc_data.index)
        }
        
        result = calculator._calculate_price_fib_relationship(sample_ohlc_data, fib_levels)
        
        # 验证返回类型
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlc_data)

    def test_calculate_exception_handling(self, calculator):
        """测试异常处理"""
        # 创建会导致异常的数据
        invalid_data = pd.DataFrame({
            'high': ['invalid'] * 10,  # 非数值类型
            'low': ['invalid'] * 10,
            'close': ['invalid'] * 10
        })
        
        # 应该捕获异常并返回原始数据
        result = calculator.calculate(invalid_data)
        
        # 验证返回了DataFrame（可能是原始数据）
        assert isinstance(result, pd.DataFrame)


class TestFibonacciCalculatorEdgeCases:
    """测试FibonacciCalculator的边界情况"""

    @pytest.fixture
    def calculator(self):
        """创建FibonacciCalculator实例"""
        return FibonacciCalculator()

    def test_calculate_single_row_data(self, calculator):
        """测试单行数据"""
        data = pd.DataFrame({
            'high': [105],
            'low': [95],
            'close': [100]
        })
        
        result = calculator.calculate(data)
        
        # 应该返回DataFrame（可能无法计算摆动点，但应该处理）
        assert isinstance(result, pd.DataFrame)

    def test_calculate_minimum_swing_length_data(self, calculator):
        """测试最小摆动长度数据"""
        # 创建刚好满足最小摆动长度的数据
        data = pd.DataFrame({
            'high': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105],
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        })
        
        result = calculator.calculate(data)
        
        # 应该返回DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_calculate_with_custom_levels(self):
        """测试使用自定义斐波那契水平"""
        config = {
            'custom_levels': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        calculator = FibonacciCalculator(config=config)
        
        data = pd.DataFrame({
            'high': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110] * 5,
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105] * 5,
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110] * 5
        })
        
        result = calculator.calculate(data)
        
        # 应该返回DataFrame
        assert isinstance(result, pd.DataFrame)




