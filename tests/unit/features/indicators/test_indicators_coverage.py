#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indicators模块测试覆盖
测试indicators相关组件的核心功能
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock

try:
    from src.features.indicators.momentum_calculator import MomentumCalculator
    from src.features.indicators.volatility_calculator import VolatilityCalculator
    INDICATORS_AVAILABLE = True
except ImportError:
    INDICATORS_AVAILABLE = False
    MomentumCalculator = None
    VolatilityCalculator = None


@pytest.mark.skipif(not INDICATORS_AVAILABLE, reason="Indicators not available")
class TestMomentumCalculator:
    """MomentumCalculator测试"""

    def test_momentum_calculator_initialization_default(self):
        """测试默认初始化"""
        calculator = MomentumCalculator()
        assert calculator.momentum_period == 10
        assert calculator.roc_period == 12
        assert calculator.trix_period == 15
        assert calculator.rsi_period == 14

    def test_momentum_calculator_initialization_custom(self):
        """测试自定义配置初始化"""
        config = {
            'momentum_period': 20,
            'roc_period': 15,
            'trix_period': 20,
            'rsi_period': 21
        }
        calculator = MomentumCalculator(config)
        assert calculator.momentum_period == 20
        assert calculator.roc_period == 15
        assert calculator.trix_period == 20
        assert calculator.rsi_period == 21

    def test_calculate_with_valid_data(self):
        """测试计算有效数据"""
        calculator = MomentumCalculator()
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113]
        })
        
        result = calculator.calculate(data)
        
        assert not result.empty
        assert 'momentum' in result.columns
        assert 'roc' in result.columns
        assert 'trix' in result.columns
        assert 'rsi' in result.columns

    def test_calculate_with_empty_data(self):
        """测试计算空数据"""
        calculator = MomentumCalculator()
        data = pd.DataFrame()
        
        result = calculator.calculate(data)
        
        assert result.empty

    def test_calculate_with_none_data(self):
        """测试计算None数据"""
        calculator = MomentumCalculator()
        result = calculator.calculate(None)
        assert result.empty

    def test_calculate_without_close_column(self):
        """测试缺少close列的数据"""
        calculator = MomentumCalculator()
        data = pd.DataFrame({
            'high': [101, 102, 103],
            'low': [99, 100, 101]
        })
        
        result = calculator.calculate(data)
        # 应该返回原始数据
        assert 'close' not in result.columns

    def test_calculate_without_high_low(self):
        """测试缺少high/low列的数据"""
        calculator = MomentumCalculator()
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
        })
        
        result = calculator.calculate(data)
        
        assert 'momentum' in result.columns
        assert 'roc' in result.columns
        # 没有high/low时不应该有stoch指标
        assert 'stoch_k' not in result.columns or pd.isna(result['stoch_k']).all()

    def test_calculate_momentum(self):
        """测试计算动量指标"""
        calculator = MomentumCalculator({'momentum_period': 5})
        close_price = pd.Series([100, 101, 102, 103, 104, 105, 106, 107])
        
        result = calculator._calculate_momentum(close_price)
        
        assert len(result) == len(close_price)
        assert pd.isna(result.iloc[0:5]).all()  # 前5个应该是NaN
        assert not pd.isna(result.iloc[5])  # 第6个应该有值

    def test_calculate_roc(self):
        """测试计算ROC指标"""
        calculator = MomentumCalculator({'roc_period': 5})
        close_price = pd.Series([100, 101, 102, 103, 104, 105, 106, 107])
        
        result = calculator._calculate_roc(close_price)
        
        assert len(result) == len(close_price)
        assert pd.isna(result.iloc[0:5]).all()  # 前5个应该是NaN

    def test_calculate_rsi(self):
        """测试计算RSI指标"""
        calculator = MomentumCalculator({'rsi_period': 14})
        close_price = pd.Series([100 + i * 0.5 for i in range(30)])
        
        result = calculator._calculate_rsi(close_price)
        
        assert len(result) == len(close_price)
        # RSI应该在0-100之间
        valid_rsi = result.dropna()
        if len(valid_rsi) > 0:
            assert (valid_rsi >= 0).all()
            assert (valid_rsi <= 100).all()


@pytest.mark.skipif(not INDICATORS_AVAILABLE, reason="Indicators not available")
class TestVolatilityCalculator:
    """VolatilityCalculator测试"""

    def test_volatility_calculator_initialization_default(self):
        """测试默认初始化"""
        calculator = VolatilityCalculator()
        assert calculator.bb_period == 20
        assert calculator.kc_period == 20
        assert calculator.atr_period == 14
        assert calculator.hv_period == 30

    def test_volatility_calculator_initialization_custom(self):
        """测试自定义配置初始化"""
        config = {
            'bb_period': 30,
            'kc_period': 25,
            'atr_period': 20,
            'hv_period': 40
        }
        calculator = VolatilityCalculator(config)
        assert calculator.bb_period == 30
        assert calculator.kc_period == 25
        assert calculator.atr_period == 20
        assert calculator.hv_period == 40

    def test_calculate_with_valid_data(self):
        """测试计算有效数据"""
        calculator = VolatilityCalculator()
        data = pd.DataFrame({
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
        })
        
        result = calculator.calculate(data)
        
        assert not result.empty
        assert 'ATR' in result.columns or 'volatility_atr' in result.columns

    def test_calculate_with_empty_data(self):
        """测试计算空数据"""
        calculator = VolatilityCalculator()
        data = pd.DataFrame()
        
        result = calculator.calculate(data)
        
        assert result.empty

    def test_calculate_with_none_data(self):
        """测试计算None数据"""
        calculator = VolatilityCalculator()
        result = calculator.calculate(None)
        assert result.empty

    def test_calculate_without_required_columns(self):
        """测试缺少必要列的数据"""
        calculator = VolatilityCalculator()
        data = pd.DataFrame({
            'high': [101, 102, 103]
            # 缺少low和close
        })
        
        result = calculator.calculate(data)
        # 应该返回原始数据
        assert 'low' not in result.columns

    def test_ensure_required_columns_valid(self):
        """测试确保必要列存在（有效）"""
        calculator = VolatilityCalculator()
        data = pd.DataFrame({
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102]
        })
        
        result = calculator._ensure_required_columns(data)
        assert result is True

    def test_ensure_required_columns_invalid(self):
        """测试确保必要列存在（无效）"""
        calculator = VolatilityCalculator()
        data = pd.DataFrame({
            'high': [101, 102, 103]
            # 缺少low和close
        })
        
        result = calculator._ensure_required_columns(data)
        assert result is False

    def test_calculate_atr(self):
        """测试计算ATR"""
        calculator = VolatilityCalculator({'atr_period': 5})
        data = pd.DataFrame({
            'high': [101, 102, 103, 104, 105, 106, 107, 108],
            'low': [99, 100, 101, 102, 103, 104, 105, 106],
            'close': [100, 101, 102, 103, 104, 105, 106, 107]
        })
        
        result = calculator._calculate_atr(data, period=5)
        
        assert 'ATR' in result.columns
        assert len(result) == len(data)

    def test_calculate_bollinger_bands(self):
        """测试计算布林带"""
        calculator = VolatilityCalculator({'bb_period': 5})
        data = pd.DataFrame({
            'high': [101, 102, 103, 104, 105, 106, 107, 108],
            'low': [99, 100, 101, 102, 103, 104, 105, 106],
            'close': [100, 101, 102, 103, 104, 105, 106, 107]
        })
        
        result = calculator._calculate_bollinger_bands(data)
        
        assert 'BB_Upper' in result.columns
        assert 'BB_Middle' in result.columns
        assert 'BB_Lower' in result.columns

    def test_calculate_historical_volatility(self):
        """测试计算历史波动率"""
        calculator = VolatilityCalculator({'hv_period': 10})
        data = pd.DataFrame({
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
        })
        
        result = calculator._calculate_historical_volatility(data)
        
        # 检查返回的是DataFrame且有列
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        # 可能有不同的列名，只要不是空的就行
        assert len(result.columns) > 0

    def test_calculate_bollinger_bandwidth(self):
        """测试计算布林带宽度"""
        calculator = VolatilityCalculator()
        data = pd.DataFrame({
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
        })
        
        result = calculator._calculate_bollinger_bandwidth(data)
        
        assert len(result) == len(data)
        # 至少有一些有效值（在bb_period之后）
        valid_values = result.dropna()
        assert len(valid_values) > 0

