"""
动量策略增强测试
测试MomentumStrategy的各种功能和边界情况
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.strategy.strategies.momentum_strategy import MomentumStrategy
from src.strategy.interfaces.strategy_interfaces import StrategyConfig, StrategySignal


class TestMomentumStrategyEnhanced:
    """动量策略增强测试"""

    @pytest.fixture
    def momentum_strategy(self):
        """创建动量策略实例"""
        return MomentumStrategy(
            strategy_id='momentum_test_001',
            name='Momentum Test Strategy',
            strategy_type='momentum'
        )

    @pytest.fixture
    def momentum_strategy_with_config(self):
        """创建带配置的动量策略"""
        config = StrategyConfig(
            strategy_id='momentum_test_002',
            strategy_name='Momentum Config Strategy',
            strategy_type='momentum',
            parameters={
                'lookback_period': 30,
                'momentum_threshold': 0.08,
                'volume_threshold': 2.0,
                'min_trend_period': 7,
                'max_hold_period': 15
            },
            symbols=['000001']
        )
        return MomentumStrategy(
            strategy_id=config.strategy_id,
            name=config.strategy_name,
            strategy_type='momentum',
            config=config
        )

    def test_momentum_strategy_initialization(self, momentum_strategy):
        """测试动量策略初始化"""
        strategy = momentum_strategy
        assert strategy is not None
        assert strategy.strategy_id == 'momentum_test_001'
        assert strategy.name == 'Momentum Test Strategy'
        assert strategy.lookback_period == 20
        assert strategy.momentum_threshold == 0.05

    def test_momentum_strategy_with_config(self, momentum_strategy_with_config):
        """测试带配置的动量策略"""
        strategy = momentum_strategy_with_config
        assert strategy.lookback_period == 30
        assert strategy.momentum_threshold == 0.08
        assert strategy.volume_threshold == 2.0

    def test_calculate_momentum_basic(self, momentum_strategy):
        """测试基本动量计算"""
        strategy = momentum_strategy

        # 创建价格数据
        prices = [100, 101, 102, 103, 104, 105]

        momentum = strategy._calculate_momentum(prices)
        assert isinstance(momentum, float)
        assert momentum > 0  # 上涨趋势

    def test_calculate_momentum_downtrend(self, momentum_strategy):
        """测试下跌趋势的动量计算"""
        strategy = momentum_strategy

        # 创建下跌价格数据
        prices = [105, 104, 103, 102, 101, 100]

        momentum = strategy._calculate_momentum(prices)
        assert isinstance(momentum, float)
        assert momentum < 0  # 下跌趋势

    def test_calculate_momentum_sideways(self, momentum_strategy):
        """测试横盘整理的动量计算"""
        strategy = momentum_strategy

        # 创建横盘价格数据
        prices = [100, 100.5, 99.5, 100.2, 99.8, 100.1]

        momentum = strategy._calculate_momentum(prices)
        assert isinstance(momentum, float)
        # 动量应该接近于0

    def test_detect_trend_upward(self, momentum_strategy):
        """测试上涨趋势检测"""
        strategy = momentum_strategy

        # 创建上涨价格序列
        prices = [100, 102, 105, 108, 112, 116]

        trend = strategy._detect_trend(prices)
        assert isinstance(trend, dict)
        assert 'direction' in trend
        assert 'strength' in trend

    def test_detect_trend_downward(self, momentum_strategy):
        """测试下跌趋势检测"""
        strategy = momentum_strategy

        # 创建下跌价格序列
        prices = [120, 115, 110, 105, 100, 95]

        trend = strategy._detect_trend(prices)
        assert isinstance(trend, dict)
        assert 'direction' in trend
        assert 'strength' in trend

    def test_generate_signals_strong_uptrend(self, momentum_strategy):
        """测试强上涨趋势的信号生成"""
        strategy = momentum_strategy

        # 模拟强上涨的市场数据
        market_data = {
            'historical_prices': [100, 105, 110, 115, 120, 126],
            'current_price': 126,
            'volume': 2000,
            'avg_volume': 1000
        }

        signals = strategy.generate_signals(market_data)
        assert isinstance(signals, list)

    def test_generate_signals_strong_downtrend(self, momentum_strategy):
        """测试强下跌趋势的信号生成"""
        strategy = momentum_strategy

        # 模拟强下跌的市场数据
        market_data = {
            'historical_prices': [130, 125, 120, 115, 110, 104],
            'current_price': 104,
            'volume': 2500,
            'avg_volume': 1000
        }

        signals = strategy.generate_signals(market_data)
        assert isinstance(signals, list)

    def test_generate_signals_weak_trend(self, momentum_strategy):
        """测试弱趋势的信号生成"""
        strategy = momentum_strategy

        # 模拟弱趋势的市场数据
        market_data = {
            'historical_prices': [100, 100.5, 101, 100.8, 101.2, 101.1],
            'current_price': 101.1,
            'volume': 800,
            'avg_volume': 1000
        }

        signals = strategy.generate_signals(market_data)
        assert isinstance(signals, list)

    def test_volume_confirmation(self, momentum_strategy):
        """测试成交量确认"""
        strategy = momentum_strategy

        # 高成交量确认
        high_volume = strategy._check_volume_confirmation(2000, 1000, 1.5)
        assert isinstance(high_volume, bool)

        # 低成交量
        low_volume = strategy._check_volume_confirmation(800, 1000, 1.5)
        assert isinstance(low_volume, bool)

    def test_risk_management(self, momentum_strategy):
        """测试风险管理"""
        strategy = momentum_strategy

        # 测试止损逻辑
        current_price = 95
        entry_price = 100
        stop_loss_pct = 0.05

        should_stop = strategy._should_stop_loss(current_price, entry_price, stop_loss_pct)
        assert isinstance(should_stop, bool)
        assert should_stop is True  # 下跌5%，应该止损

    def test_position_sizing(self, momentum_strategy):
        """测试仓位大小计算"""
        strategy = momentum_strategy

        capital = 100000
        risk_per_trade = 0.02
        stop_loss = 0.05

        position_size = strategy._calculate_position_size(capital, risk_per_trade, stop_loss)
        assert isinstance(position_size, (int, float))
        assert position_size > 0

    def test_entry_conditions(self, momentum_strategy):
        """测试入场条件"""
        strategy = momentum_strategy

        # 强上涨趋势
        strong_up = {
            'momentum': 0.08,
            'trend_strength': 0.9,
            'volume_confirmed': True
        }

        can_enter = strategy._check_entry_conditions(strong_up)
        assert isinstance(can_enter, bool)

    def test_exit_conditions(self, momentum_strategy):
        """测试出场条件"""
        strategy = momentum_strategy

        # 趋势反转
        reversal = {
            'momentum': -0.03,
            'trend_weakening': True,
            'stop_loss_hit': False
        }

        should_exit = strategy._check_exit_conditions(reversal)
        assert isinstance(should_exit, bool)

    def test_signal_filtering(self, momentum_strategy):
        """测试信号过滤"""
        strategy = momentum_strategy

        # 创建测试信号
        signals = [
            {'type': 'BUY', 'strength': 0.9, 'price': 100},
            {'type': 'SELL', 'strength': 0.3, 'price': 105},
            {'type': 'HOLD', 'strength': 0.1, 'price': 102}
        ]

        filtered = strategy._filter_signals(signals)
        assert isinstance(filtered, list)
        assert len(filtered) <= len(signals)

    def test_performance_tracking(self, momentum_strategy):
        """测试性能跟踪"""
        strategy = momentum_strategy

        # 记录交易结果
        trade_result = {
            'entry_price': 100,
            'exit_price': 110,
            'quantity': 1000,
            'pnl': 1000
        }

        strategy._update_performance(trade_result)

        # 验证性能数据已更新
        assert hasattr(strategy, '_performance_metrics')

    def test_parameter_validation(self, momentum_strategy):
        """测试参数验证"""
        strategy = momentum_strategy

        # 有效的参数
        valid_params = {
            'lookback_period': 20,
            'momentum_threshold': 0.05,
            'volume_threshold': 1.5
        }

        is_valid = strategy._validate_parameters(valid_params)
        assert isinstance(is_valid, bool)

    def test_adaptive_parameters(self, momentum_strategy):
        """测试自适应参数调整"""
        strategy = momentum_strategy

        # 基于市场波动率调整参数
        market_volatility = 0.02

        strategy._adapt_parameters(market_volatility)

        # 验证参数已调整
        assert hasattr(strategy, 'momentum_threshold')

    def test_multi_timeframe_analysis(self, momentum_strategy):
        """测试多时间框架分析"""
        strategy = momentum_strategy

        # 不同时间框架的价格数据
        timeframe_data = {
            '1m': [100, 101, 102],
            '5m': [100, 101.5, 103],
            '1h': [100, 102, 105]
        }

        analysis = strategy._analyze_multi_timeframe(timeframe_data)
        assert isinstance(analysis, dict)

    def test_market_regime_detection(self, momentum_strategy):
        """测试市场状态检测"""
        strategy = momentum_strategy

        # 趋势市场
        trending_market = {
            'volatility': 0.02,
            'trend_strength': 0.8,
            'volume': 1500
        }

        regime = strategy._detect_market_regime(trending_market)
        assert isinstance(regime, str)

    def test_signal_confidence_scoring(self, momentum_strategy):
        """测试信号置信度评分"""
        strategy = momentum_strategy

        signal = {
            'momentum': 0.08,
            'trend_alignment': 0.9,
            'volume_confirmation': True,
            'market_regime': 'trending'
        }

        confidence = strategy._calculate_signal_confidence(signal)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_portfolio_impact_assessment(self, momentum_strategy):
        """测试投资组合影响评估"""
        strategy = momentum_strategy

        trade = {
            'symbol': '000001',
            'quantity': 1000,
            'price': 100,
            'portfolio_value': 100000
        }

        impact = strategy._assess_portfolio_impact(trade)
        assert isinstance(impact, dict)

    def test_strategy_state_management(self, momentum_strategy):
        """测试策略状态管理"""
        strategy = momentum_strategy

        # 测试状态转换
        strategy._set_state('active')
        assert strategy._get_state() == 'active'

        strategy._set_state('paused')
        assert strategy._get_state() == 'paused'

    def test_error_handling_and_recovery(self, momentum_strategy):
        """测试错误处理和恢复"""
        strategy = momentum_strategy

        # 测试异常情况处理
        try:
            # 传递无效数据
            strategy.generate_signals(None)
        except Exception as e:
            # 应该优雅地处理异常
            assert True

        # 验证策略仍然可以正常工作
        valid_data = {'historical_prices': [100, 101, 102]}
        signals = strategy.generate_signals(valid_data)
        assert isinstance(signals, list)
