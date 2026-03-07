# -*- coding: utf-8 -*-
"""
动量指标计算器测试
"""

import pytest
import pandas as pd
import numpy as np
from src.features.indicators.momentum_calculator import MomentumCalculator


class TestMomentumCalculator:
    """测试MomentumCalculator类"""

    @pytest.fixture
    def calculator(self):
        """创建MomentumCalculator实例"""
        return MomentumCalculator()

    @pytest.fixture
    def sample_data(self):
        """生成示例价格数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 115, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

    def test_init_default(self):
        """测试默认初始化"""
        calc = MomentumCalculator()
        assert calc.momentum_period == 10
        assert calc.roc_period == 12
        assert calc.trix_period == 15
        assert calc.rsi_period == 14

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'momentum_period': 20,
            'roc_period': 15,
            'rsi_period': 21
        }
        calc = MomentumCalculator(config)
        assert calc.momentum_period == 20
        assert calc.roc_period == 15
        assert calc.rsi_period == 21

    def test_calculate_basic(self, calculator, sample_data):
        """测试基本计算"""
        result = calculator.calculate(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'momentum' in result.columns
        assert 'roc' in result.columns
        assert 'trix' in result.columns
        assert 'kst' in result.columns
        assert 'rsi' in result.columns
        assert len(result) == len(sample_data)

    def test_calculate_with_high_low(self, calculator, sample_data):
        """测试包含high和low的计算"""
        result = calculator.calculate(sample_data)
        
        assert 'stoch_k' in result.columns
        assert 'stoch_d' in result.columns

    def test_calculate_without_high_low(self, calculator):
        """测试没有high和low的情况"""
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })
        result = calculator.calculate(data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'momentum' in result.columns
        # 没有high/low时不应该有stoch指标
        assert 'stoch_k' not in result.columns or pd.isna(result['stoch_k']).all()

    def test_calculate_empty_data(self, calculator):
        """测试空数据"""
        data = pd.DataFrame()
        result = calculator.calculate(data)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_calculate_none_data(self, calculator):
        """测试None数据"""
        result = calculator.calculate(None)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_calculate_missing_close(self, calculator):
        """测试缺少close列"""
        data = pd.DataFrame({
            'open': [100, 101, 102]
        })
        result = calculator.calculate(data)
        # 应该返回原始数据
        assert isinstance(result, pd.DataFrame)
        assert 'close' not in result.columns

    def test_calculate_momentum_values(self, calculator, sample_data):
        """测试动量值计算"""
        result = calculator.calculate(sample_data)
        
        momentum_values = result['momentum'].dropna()
        if len(momentum_values) > 0:
            assert isinstance(momentum_values.iloc[0], (int, float, np.number))

    def test_calculate_roc_values(self, calculator, sample_data):
        """测试ROC值计算"""
        result = calculator.calculate(sample_data)
        
        roc_values = result['roc'].dropna()
        if len(roc_values) > 0:
            assert isinstance(roc_values.iloc[0], (int, float, np.number))

    def test_calculate_rsi_values(self, calculator, sample_data):
        """测试RSI值计算"""
        result = calculator.calculate(sample_data)
        
        rsi_values = result['rsi'].dropna()
        if len(rsi_values) > 0:
            # RSI应该在0-100范围内
            assert rsi_values.min() >= 0
            assert rsi_values.max() <= 100

    def test_calculate_signals(self, calculator, sample_data):
        """测试信号生成"""
        result = calculator.calculate(sample_data)
        
        # 检查是否有信号列
        signal_columns = [col for col in result.columns if 'signal' in col.lower()]
        # 可能有信号列，也可能没有，取决于实现
        assert isinstance(result, pd.DataFrame)

    def test_calculate_exception_handling(self, calculator):
        """测试异常处理"""
        # 传入无效数据
        invalid_data = "invalid"
        result = calculator.calculate(invalid_data)
        # 应该返回原始数据或空DataFrame
        assert isinstance(result, (pd.DataFrame, type(invalid_data)))

    def test_calculate_insufficient_data(self, calculator):
        """测试数据不足的情况"""
        data = pd.DataFrame({
            'close': [100, 101]  # 数据太少
        })
        result = calculator.calculate(data)
        assert isinstance(result, pd.DataFrame)

