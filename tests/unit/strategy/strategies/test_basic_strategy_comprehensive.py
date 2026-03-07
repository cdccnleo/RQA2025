"""
测试基础策略 - 综合测试
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, time


class TestBasicStrategyComprehensive:
    """测试基础策略 - 综合测试"""

    def test_basic_strategy_initialization_with_dict(self):
        """测试用字典初始化基础策略"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            config = {
                "strategy_id": "test_basic_001",
                "name": "Test Basic Strategy",
                "strategy_type": "basic",
                "parameters": {
                    "threshold": 0.05,
                    "period": 20
                }
            }

            strategy = BasicStrategy(config)
            assert strategy is not None
            assert strategy.strategy_id == "test_basic_001"
            assert strategy.strategy_name == "Test Basic Strategy"

        except ImportError:
            pytest.skip("BasicStrategy not available")

    def test_basic_strategy_initialization_with_config_object(self):
        """测试用配置对象初始化基础策略"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy
            from src.strategy.interfaces.strategy_interfaces import StrategyConfig, StrategyType

            config = StrategyConfig(
                strategy_id="test_basic_002",
                name="Test Basic Strategy 2",
                strategy_type=StrategyType.MOMENTUM,
                parameters={"momentum_period": 10}
            )

            strategy = BasicStrategy(config)
            assert strategy is not None
            assert strategy.strategy_id == "test_basic_002"
            assert strategy.strategy_name == "Test Basic Strategy 2"

        except ImportError:
            pytest.skip("BasicStrategy or interfaces not available")

    def test_generate_signals_basic(self):
        """测试基本信号生成"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            strategy = BasicStrategy({
                "strategy_id": "signal_test",
                "name": "Signal Test Strategy"
            })

            # 测试买入信号
            market_data_buy = {
                'price': 105.0,
                'volume': 1000,
                'prev_price': 100.0,
                'trend': 'up'
            }

            signals = strategy._generate_signals_impl(market_data_buy)
            assert isinstance(signals, list)

            # 测试卖出信号
            market_data_sell = {
                'price': 95.0,
                'volume': 1200,
                'prev_price': 100.0,
                'trend': 'down'
            }

            signals = strategy._generate_signals_impl(market_data_sell)
            assert isinstance(signals, list)

        except ImportError:
            pytest.skip("BasicStrategy not available")

    def test_generate_signals_with_price_movements(self):
        """测试价格变动信号生成"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            strategy = BasicStrategy({
                "strategy_id": "price_test",
                "name": "Price Movement Test"
            })

            # 测试大幅上涨 - 应该生成买入信号
            market_data_up = {
                'price': 110.0,
                'volume': 2000,
                'prev_price': 100.0,
                'change_percent': 10.0
            }

            signals = strategy._generate_signals_impl(market_data_up)
            assert isinstance(signals, list)
            # 检查是否有买入信号
            buy_signals = [s for s in signals if hasattr(s, 'signal_type') and 'BUY' in str(s.signal_type).upper()]
            assert len(buy_signals) > 0

            # 测试大幅下跌 - 应该生成卖出信号
            market_data_down = {
                'price': 90.0,
                'volume': 1800,
                'prev_price': 100.0,
                'change_percent': -10.0
            }

            signals = strategy._generate_signals_impl(market_data_down)
            assert isinstance(signals, list)
            # 检查是否有卖出信号
            sell_signals = [s for s in signals if hasattr(s, 'signal_type') and 'SELL' in str(s.signal_type).upper()]
            assert len(sell_signals) > 0

        except ImportError:
            pytest.skip("BasicStrategy not available")

    def test_validate_market_data(self):
        """测试市场数据验证"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            strategy = BasicStrategy({
                "strategy_id": "validation_test",
                "name": "Validation Test"
            })

            # 测试有效数据
            valid_data = {
                'price': 100.0,
                'volume': 1000,
                'timestamp': datetime.now()
            }

            # 不应该抛出异常
            strategy._validate_market_data(valid_data)

            # 测试无效数据 - 缺少价格
            invalid_data = {
                'volume': 1000,
                'timestamp': datetime.now()
            }

            with pytest.raises((ValueError, KeyError)):
                strategy._validate_market_data(invalid_data)

        except ImportError:
            pytest.skip("BasicStrategy not available")

    def test_get_parameters(self):
        """测试获取参数"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            config = {
                "strategy_id": "param_test",
                "name": "Parameter Test",
                "parameters": {
                    "threshold": 0.05,
                    "period": 20,
                    "stop_loss": 0.02
                }
            }

            strategy = BasicStrategy(config)
            params = strategy.get_parameters()

            assert isinstance(params, dict)
            assert 'threshold' in params
            assert 'period' in params
            assert params['threshold'] == 0.05
            assert params['period'] == 20

        except ImportError:
            pytest.skip("BasicStrategy not available")

    def test_is_trading_time(self):
        """测试交易时间检查"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            strategy = BasicStrategy({
                "strategy_id": "time_test",
                "name": "Trading Time Test"
            })

            # 测试正常交易时间
            trading_time = pd.Timestamp("2023-01-01 14:30:00")
            is_trading = strategy.is_trading_time(trading_time)
            assert isinstance(is_trading, bool)

            # 测试非交易时间
            non_trading_time = pd.Timestamp("2023-01-01 18:30:00")  # 下午6:30
            is_trading = strategy.is_trading_time(non_trading_time)
            assert isinstance(is_trading, bool)

        except ImportError:
            pytest.skip("BasicStrategy not available")

    def test_check_price_limit(self):
        """测试价格限制检查"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            strategy = BasicStrategy({
                "strategy_id": "limit_test",
                "name": "Price Limit Test"
            })

            # 测试正常价格变动
            result = strategy.check_price_limit(105.0, 100.0)
            assert isinstance(result, dict)
            assert 'within_limit' in result

            # 测试大幅价格变动
            result = strategy.check_price_limit(150.0, 100.0)  # 50%上涨
            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("BasicStrategy not available")

    def test_calculate_position_size(self):
        """测试仓位大小计算"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            strategy = BasicStrategy({
                "strategy_id": "position_test",
                "name": "Position Size Test"
            })

            available_capital = 100000.0
            price = 50.0
            risk_percent = 0.02
            stop_loss_percent = 0.05

            position_size = strategy.calculate_position_size(
                available_capital, price, risk_percent, stop_loss_percent
            )

            assert isinstance(position_size, (int, float))
            assert position_size > 0
            assert position_size <= available_capital / price  # 不能超过可用资金能买的数量

        except ImportError:
            pytest.skip("BasicStrategy not available")

    def test_validate_order(self):
        """测试订单验证"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            strategy = BasicStrategy({
                "strategy_id": "order_test",
                "name": "Order Validation Test"
            })

            # 测试有效订单
            valid_order = {
                'symbol': 'AAPL',
                'quantity': 100,
                'price': 150.0,
                'order_type': 'market'
            }

            is_valid = strategy.validate_order(valid_order)
            assert isinstance(is_valid, bool)

            # 测试无效订单 - 缺少必要字段
            invalid_order = {
                'quantity': 100,
                'price': 150.0
            }

            is_valid = strategy.validate_order(invalid_order)
            assert isinstance(is_valid, bool)

        except ImportError:
            pytest.skip("BasicStrategy not available")

    def test_calculate_commission(self):
        """测试佣金计算"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            strategy = BasicStrategy({
                "strategy_id": "commission_test",
                "name": "Commission Test"
            })

            order_value = 10000.0
            commission_rate = 0.0003

            commission = strategy.calculate_commission(order_value, commission_rate)
            expected_commission = order_value * commission_rate

            assert isinstance(commission, float)
            assert commission == expected_commission

        except ImportError:
            pytest.skip("BasicStrategy not available")

    def test_calculate_slippage(self):
        """测试滑点计算"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            strategy = BasicStrategy({
                "strategy_id": "slippage_test",
                "name": "Slippage Test"
            })

            order_quantity = 1000
            market_impact = 0.0001

            slippage = strategy.calculate_slippage(order_quantity, market_impact)
            expected_slippage = order_quantity * market_impact

            assert isinstance(slippage, float)
            assert slippage == expected_slippage

        except ImportError:
            pytest.skip("BasicStrategy not available")

    def test_get_market_status(self):
        """测试获取市场状态"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            strategy = BasicStrategy({
                "strategy_id": "market_test",
                "name": "Market Status Test"
            })

            symbol = "AAPL"
            market_status = strategy.get_market_status(symbol)

            assert isinstance(market_status, dict)
            assert 'symbol' in market_status or 'status' in market_status

        except ImportError:
            pytest.skip("BasicStrategy not available")

    def test_preprocess_data(self):
        """测试数据预处理"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            strategy = BasicStrategy({
                "strategy_id": "preprocess_test",
                "name": "Data Preprocessing Test"
            })

            # 创建测试数据
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10),
                'open': np.random.uniform(100, 110, 10),
                'high': np.random.uniform(105, 115, 10),
                'low': np.random.uniform(95, 105, 10),
                'close': np.random.uniform(100, 110, 10),
                'volume': np.random.randint(1000, 10000, 10)
            })

            processed_data = strategy.preprocess_data(test_data)

            assert isinstance(processed_data, pd.DataFrame)
            assert len(processed_data) == len(test_data)
            assert 'timestamp' in processed_data.columns

        except ImportError:
            pytest.skip("BasicStrategy not available")

    def test_basic_strategy_edge_cases(self):
        """测试基础策略边界情况"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            strategy = BasicStrategy({
                "strategy_id": "edge_test",
                "name": "Edge Cases Test"
            })

            # 测试空市场数据
            empty_data = {}
            signals = strategy._generate_signals_impl(empty_data)
            assert isinstance(signals, list)

            # 测试极端价格
            extreme_data = {
                'price': 1000000.0,  # 极端高价
                'volume': 0,  # 零成交量
                'prev_price': 100.0
            }
            signals = strategy._generate_signals_impl(extreme_data)
            assert isinstance(signals, list)

            # 测试负价格（异常情况）
            negative_data = {
                'price': -100.0,
                'volume': 1000,
                'prev_price': 50.0
            }
            signals = strategy._generate_signals_impl(negative_data)
            assert isinstance(signals, list)

        except ImportError:
            pytest.skip("BasicStrategy not available")

    def test_basic_strategy_error_handling(self):
        """测试基础策略错误处理"""
        try:
            from src.strategy.strategies.basic_strategy import BasicStrategy

            strategy = BasicStrategy({
                "strategy_id": "error_test",
                "name": "Error Handling Test"
            })

            # 测试无效参数
            invalid_params = strategy.get_parameters()
            assert isinstance(invalid_params, dict)  # 即使参数无效也应该返回dict

            # 测试无效订单验证
            invalid_order = None
            result = strategy.validate_order(invalid_order)
            assert isinstance(result, bool)  # 应该返回布尔值

        except ImportError:
            pytest.skip("BasicStrategy not available")
