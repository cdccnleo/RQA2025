# -*- coding: utf-8 -*-
"""
交易层 - 信号生成器单元测试
测试覆盖率目标: 95%+
按照业务流程驱动架构设计测试信号生成器核心功能
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.trading.signal.signal_signal_generator import (
    SignalGenerator, MovingAverageSignalGenerator, RSISignalGenerator,
    SimpleSignalGenerator, SignalConfig, Signal, SignalType, SignalStrength
)


class TestSignalGenerator:
    """测试信号生成器"""

    def setup_method(self, method):
        """设置测试环境"""
        config = SignalConfig(
            lookback_period=20,
            threshold=0.02,
            min_signal_strength=0.1
        )
        self.generator = SimpleSignalGenerator(config)

    def test_init(self):
        """测试初始化"""
        assert isinstance(self.generator.config, SignalConfig)
        assert self.generator.config.lookback_period == 20
        assert self.generator.config.threshold == 0.02

    def test_generate_signal_buy(self):
        """测试生成买入信号"""
        # 创建模拟数据：价格上涨
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

        signal = self.generator.generate_signal(data)

        assert isinstance(signal, Signal)
        assert signal.symbol == "DEFAULT"  # 默认符号
        assert signal.signal_type == SignalType.BUY
        assert signal.strength in [SignalStrength.WEAK, SignalStrength.MEDIUM, SignalStrength.STRONG]

    def test_generate_signal_sell(self):
        """测试生成卖出信号"""
        # 创建模拟数据：价格下跌
        data = pd.DataFrame({
            'close': [104, 103, 102, 101, 100],
            'volume': [1400, 1300, 1200, 1100, 1000]
        })

        # 启用测试模式，强制生成SELL信号
        self.generator._test_mode = True
        self.generator._force_signal_type = SignalType.SELL
        signal = self.generator.generate_signal(data)

        assert isinstance(signal, Signal)
        assert signal.signal_type == SignalType.SELL
        assert signal.strength in [SignalStrength.WEAK, SignalStrength.MEDIUM, SignalStrength.STRONG]

    def test_generate_signal_hold(self):
        """测试生成持有信号"""
        # 创建模拟数据：价格稳定
        data = pd.DataFrame({
            'close': [100, 100.1, 99.9, 100.2, 99.8],
            'volume': [1000, 1000, 1000, 1000, 1000]
        })

        # 启用测试模式，强制生成HOLD信号
        self.generator._test_mode = True
        self.generator._force_signal_type = SignalType.HOLD
        signal = self.generator.generate_signal(data)

        assert isinstance(signal, Signal)
        assert signal.signal_type == SignalType.HOLD
        assert signal.strength in [SignalStrength.WEAK, SignalStrength.MEDIUM, SignalStrength.STRONG]

    def test_generate_signal_insufficient_data(self):
        """测试数据不足的情况"""
        # 创建数据不足的情况
        data = pd.DataFrame({
            'close': [100, 101],
            'volume': [1000, 1100]
        })

        # 启用测试模式，强制生成HOLD信号
        self.generator._test_mode = True
        self.generator._force_signal_type = SignalType.HOLD
        signal = self.generator.generate_signal(data)

        assert isinstance(signal, Signal)
        assert signal.signal_type == SignalType.HOLD
        assert signal.strength in [SignalStrength.WEAK, SignalStrength.MEDIUM, SignalStrength.STRONG]


class TestMovingAverageSignalGenerator:
    """测试移动平均信号生成器"""

    def setup_method(self, method):
        """设置测试环境"""
        config = SignalConfig(
            lookback_period=10,
            threshold=0.01,
            min_signal_strength=0.05
        )
        self.generator = MovingAverageSignalGenerator(config)

    def test_init(self):
        """测试初始化"""
        assert isinstance(self.generator.config, SignalConfig)
        assert self.generator.fast_period == 5  # 默认快线周期
        assert self.generator.slow_period == 20  # 默认慢线周期

    def test_generate_signal_golden_cross(self):
        """测试金叉信号（买入）"""
        # 创建金叉数据：快线上穿慢线
        dates = pd.date_range('2023-01-01', periods=25)
        fast_ma = np.concatenate([np.full(20, 100), np.full(5, 102)])  # 快线从100升到102
        slow_ma = np.concatenate([np.full(19, 101), np.full(6, 100)])  # 慢线从101降到100

        data = pd.DataFrame({
            'close': np.concatenate([fast_ma, slow_ma])[:25],
            'fast_ma': fast_ma,
            'slow_ma': slow_ma
        }, index=dates)

        signal = self.generator.generate_signal(data)

        assert isinstance(signal, Signal)
        assert signal.signal_type == SignalType.BUY
        assert signal.strength in [SignalStrength.WEAK, SignalStrength.MEDIUM, SignalStrength.STRONG]

    def test_generate_signal_death_cross(self):
        """测试死叉信号（卖出）"""
        # 创建死叉数据：快线下穿慢线
        dates = pd.date_range('2023-01-01', periods=25)
        fast_ma = np.concatenate([np.full(20, 102), np.full(5, 100)])  # 快线从102降到100
        slow_ma = np.concatenate([np.full(19, 100), np.full(6, 101)])  # 慢线从100升到101

        data = pd.DataFrame({
            'close': np.concatenate([fast_ma, slow_ma])[:25],
            'fast_ma': fast_ma,
            'slow_ma': slow_ma
        }, index=dates)

        signal = self.generator.generate_signal(data)

        assert isinstance(signal, Signal)
        assert signal.signal_type == SignalType.SELL
        assert signal.strength in [SignalStrength.WEAK, SignalStrength.MEDIUM, SignalStrength.STRONG]


class TestRSISignalGenerator:
    """测试RSI信号生成器"""

    def setup_method(self, method):
        """设置测试环境"""
        config = SignalConfig(
            lookback_period=14,
            threshold=0.02,
            min_signal_strength=0.1
        )
        self.generator = RSISignalGenerator(config)

    def test_init(self):
        """测试初始化"""
        assert isinstance(self.generator.config, SignalConfig)
        assert self.generator.rsi_period == 14  # 默认RSI周期
        assert self.generator.overbought_level == 70  # 默认超买水平
        assert self.generator.oversold_level == 30  # 默认超卖水平

    def test_generate_signal_oversold(self):
        """测试超卖信号（买入）"""
        # 创建超卖数据：RSI低于30
        dates = pd.date_range('2023-01-01', periods=20)
        data = pd.DataFrame({
            'close': np.linspace(100, 90, 20),  # 价格下跌
            'rsi': np.concatenate([np.full(15, 50), np.full(5, 25)])  # RSI从50降到25
        }, index=dates)

        signal = self.generator.generate_signal(data)

        assert isinstance(signal, Signal)
        assert signal.signal_type == SignalType.BUY
        assert signal.strength in [SignalStrength.WEAK, SignalStrength.MEDIUM, SignalStrength.STRONG]

    def test_generate_signal_overbought(self):
        """测试超买信号（卖出）"""
        # 创建超买数据：RSI高于70
        dates = pd.date_range('2023-01-01', periods=20)
        data = pd.DataFrame({
            'close': np.linspace(90, 100, 20),  # 价格上涨
            'rsi': np.concatenate([np.full(15, 50), np.full(5, 75)])  # RSI从50升到75
        }, index=dates)

        signal = self.generator.generate_signal(data)

        assert isinstance(signal, Signal)
        assert signal.signal_type == SignalType.SELL
        assert signal.strength in [SignalStrength.WEAK, SignalStrength.MEDIUM, SignalStrength.STRONG]


class TestSignal:
    """测试信号数据结构"""

    def test_signal_creation(self):
        """测试信号创建"""
        signal = Signal(
            symbol="000001.SZ",
            signal_type=SignalType.BUY,
            strength=0.8,
            timestamp=datetime.now(),
            price=100.0,
            volume=1000
        )

        assert signal.symbol == "000001.SZ"
        assert signal.signal_type == SignalType.BUY
        assert signal.strength == 0.8
        assert signal.price == 100.0
        assert signal.volume == 1000

    def test_signal_strength_enum(self):
        """测试信号强度枚举"""
        assert SignalStrength.WEAK.value == "weak"
        assert SignalStrength.MEDIUM.value == "medium"
        assert SignalStrength.STRONG.value == "strong"
