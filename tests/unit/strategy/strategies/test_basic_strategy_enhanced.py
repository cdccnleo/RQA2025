"""
基础策略增强测试
测试BasicStrategy的各种功能和边界情况
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime

from src.strategy.strategies.basic_strategy import BasicStrategy
from src.strategy.interfaces.strategy_interfaces import StrategyConfig, StrategyType, StrategySignal


class TestBasicStrategyEnhanced:
    """基础策略增强测试"""

    @pytest.fixture
    def basic_strategy_dict_config(self):
        """创建使用字典配置的基础策略"""
        config = {
            'strategy_id': 'basic_test_001',
            'name': 'Basic Test Strategy',
            'strategy_type': 'momentum',
            'parameters': {
                'threshold': 0.02,
                'window': 20,
                'stop_loss': 0.05
            }
        }
        return BasicStrategy(config)

    @pytest.fixture
    def basic_strategy_config_object(self):
        """创建使用StrategyConfig对象的基础策略"""
        config = StrategyConfig(
            strategy_id='basic_test_002',
            strategy_name='Basic Test Strategy 2',
            strategy_type=StrategyType.MOMENTUM,
            parameters={'threshold': 0.02, 'window': 20},
            symbols=['000001']
        )
        return BasicStrategy(config)

    def test_initialization_with_dict_config(self, basic_strategy_dict_config):
        """测试使用字典配置初始化"""
        strategy = basic_strategy_dict_config
        assert strategy is not None
        assert strategy.strategy_id == 'basic_test_001'
        assert strategy.name == 'Basic Test Strategy'
        assert hasattr(strategy, 'config')

    def test_initialization_with_config_object(self, basic_strategy_config_object):
        """测试使用StrategyConfig对象初始化"""
        strategy = basic_strategy_config_object
        assert strategy is not None
        assert strategy.strategy_id == 'basic_test_002'
        assert strategy.name == 'Basic Test Strategy 2'

    def test_generate_signals_basic(self, basic_strategy_dict_config):
        """测试基本信号生成"""
        strategy = basic_strategy_dict_config

        market_data = {
            'price': 100.0,
            'volume': 1000,
            'timestamp': '2024-01-01T10:00:00Z'
        }

        signals = strategy._generate_signals_impl(market_data)
        assert isinstance(signals, list)

    def test_generate_signals_with_price_movement(self, basic_strategy_dict_config):
        """测试价格变动时的信号生成"""
        strategy = basic_strategy_dict_config

        # 测试上涨趋势
        market_data_up = {
            'price': 105.0,  # 上涨5%
            'volume': 1000,
            'timestamp': '2024-01-01T10:00:00Z'
        }

        signals_up = strategy._generate_signals_impl(market_data_up)
        assert isinstance(signals_up, list)

        # 测试下跌趋势
        market_data_down = {
            'price': 95.0,  # 下跌5%
            'volume': 1000,
            'timestamp': '2024-01-01T10:00:00Z'
        }

        signals_down = strategy._generate_signals_impl(market_data_down)
        assert isinstance(signals_down, list)

    def test_generate_signals_with_volume(self, basic_strategy_dict_config):
        """测试成交量相关的信号生成"""
        strategy = basic_strategy_dict_config

        # 高成交量数据
        market_data_high_volume = {
            'price': 100.0,
            'volume': 5000,  # 高成交量
            'timestamp': '2024-01-01T10:00:00Z'
        }

        signals = strategy._generate_signals_impl(market_data_high_volume)
        assert isinstance(signals, list)

    def test_signal_structure(self, basic_strategy_dict_config):
        """测试信号结构"""
        strategy = basic_strategy_dict_config

        market_data = {
            'price': 102.0,
            'volume': 1000,
            'timestamp': '2024-01-01T10:00:00Z'
        }

        signals = strategy._generate_signals_impl(market_data)

        if signals:  # 如果有信号生成
            signal = signals[0]
            # 验证信号的基本属性
            assert hasattr(signal, 'symbol') or 'symbol' in signal
            assert hasattr(signal, 'action') or 'action' in signal
            assert hasattr(signal, 'price') or 'price' in signal

    def test_strategy_with_dataframe_input(self, basic_strategy_dict_config):
        """测试DataFrame输入的信号生成"""
        strategy = basic_strategy_dict_config

        # 创建DataFrame格式的市场数据
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='D')
        })

        # 转换为字典格式进行测试
        market_data = {
            'price': df['close'].iloc[-1],
            'volume': df['volume'].iloc[-1],
            'dataframe': df
        }

        signals = strategy._generate_signals_impl(market_data)
        assert isinstance(signals, list)

    def test_empty_market_data_handling(self, basic_strategy_dict_config):
        """测试空市场数据的处理"""
        strategy = basic_strategy_dict_config

        # 空数据
        signals = strategy._generate_signals_impl({})
        assert isinstance(signals, list)
        assert len(signals) == 0

    def test_invalid_market_data_handling(self, basic_strategy_dict_config):
        """测试无效市场数据的处理"""
        strategy = basic_strategy_dict_config

        # 缺少价格的数据
        market_data = {
            'volume': 1000,
            'timestamp': '2024-01-01T10:00:00Z'
            # 缺少price
        }

        signals = strategy._generate_signals_impl(market_data)
        assert isinstance(signals, list)
        # 应该没有生成信号或生成默认信号

    def test_strategy_parameter_access(self, basic_strategy_dict_config):
        """测试策略参数访问"""
        strategy = basic_strategy_dict_config

        # 验证配置参数可以访问
        assert 'threshold' in strategy.config['parameters']
        assert 'window' in strategy.config['parameters']
        assert strategy.config['parameters']['threshold'] == 0.02
        assert strategy.config['parameters']['window'] == 20

    def test_strategy_type_setting(self, basic_strategy_dict_config):
        """测试策略类型设置"""
        strategy = basic_strategy_dict_config

        # 验证策略类型已正确设置
        assert hasattr(strategy, 'strategy_type')
        # 可能是StrategyType枚举或字符串

    def test_multiple_signals_generation(self, basic_strategy_dict_config):
        """测试多个信号的生成"""
        strategy = basic_strategy_dict_config

        # 模拟可能生成多个信号的情况
        market_data = {
            'price': 103.0,  # 明显上涨
            'volume': 2000,  # 高成交量
            'timestamp': '2024-01-01T10:00:00Z'
        }

        signals = strategy._generate_signals_impl(market_data)
        assert isinstance(signals, list)
        # 不限制信号数量，但应该是合理的

    def test_signal_timestamp_handling(self, basic_strategy_dict_config):
        """测试信号时间戳处理"""
        strategy = basic_strategy_dict_config

        market_data = {
            'price': 101.0,
            'volume': 1000,
            'timestamp': '2024-01-01T10:30:00Z'
        }

        signals = strategy._generate_signals_impl(market_data)

        if signals:
            signal = signals[0]
            # 如果信号包含时间戳，应该与输入一致或合理
            if hasattr(signal, 'timestamp') and signal.timestamp:
                assert isinstance(signal.timestamp, str)

    def test_strategy_config_validation(self):
        """测试策略配置验证"""
        # 测试有效的配置
        valid_config = {
            'strategy_id': 'test_001',
            'name': 'Test Strategy',
            'strategy_type': 'momentum',
            'parameters': {'threshold': 0.02}
        }

        strategy = BasicStrategy(valid_config)
        assert strategy is not None

        # 测试缺少必要参数的配置
        invalid_config = {
            'name': 'Invalid Strategy'
            # 缺少strategy_id等
        }

        # 应该能够处理不完整的配置
        strategy = BasicStrategy(invalid_config)
        assert strategy is not None

    def test_strategy_with_different_parameters(self):
        """测试不同参数的策略"""
        # 高阈值策略
        high_threshold_config = {
            'strategy_id': 'high_threshold_001',
            'name': 'High Threshold Strategy',
            'strategy_type': 'momentum',
            'parameters': {
                'threshold': 0.10,  # 高阈值
                'window': 20
            }
        }

        high_strategy = BasicStrategy(high_threshold_config)

        # 低阈值策略
        low_threshold_config = {
            'strategy_id': 'low_threshold_001',
            'name': 'Low Threshold Strategy',
            'strategy_type': 'momentum',
            'parameters': {
                'threshold': 0.005,  # 低阈值
                'window': 20
            }
        }

        low_strategy = BasicStrategy(low_threshold_config)

        # 验证两个策略都创建成功
        assert high_strategy is not None
        assert low_strategy is not None

        # 测试相同的市场数据，不同策略可能产生不同信号
        market_data = {
            'price': 101.0,  # 小幅上涨
            'volume': 1000,
            'timestamp': '2024-01-01T10:00:00Z'
        }

        high_signals = high_strategy._generate_signals_impl(market_data)
        low_signals = low_strategy._generate_signals_impl(market_data)

        assert isinstance(high_signals, list)
        assert isinstance(low_signals, list)

    def test_strategy_error_handling(self, basic_strategy_dict_config):
        """测试策略错误处理"""
        strategy = basic_strategy_dict_config

        # 测试None输入
        signals = strategy._generate_signals_impl(None)
        assert isinstance(signals, list)

        # 测试异常数据类型
        signals = strategy._generate_signals_impl("invalid_data")
        assert isinstance(signals, list)

    def test_strategy_state_consistency(self, basic_strategy_dict_config):
        """测试策略状态一致性"""
        strategy = basic_strategy_dict_config

        # 多次调用应该保持一致性
        market_data = {
            'price': 100.5,
            'volume': 1000,
            'timestamp': '2024-01-01T10:00:00Z'
        }

        signals1 = strategy._generate_signals_impl(market_data)
        signals2 = strategy._generate_signals_impl(market_data)

        # 相同输入应该产生相同类型的输出
        assert isinstance(signals1, type(signals2))
