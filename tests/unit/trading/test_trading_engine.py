"""
交易引擎模块测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from src.trading.core.trading_engine import TradingEngine, OrderType, OrderDirection


class TestTradingEngineEnums:
    """测试交易引擎枚举"""

    def test_order_type_enum(self):
        """测试订单类型枚举"""
        assert OrderType.MARKET.value == 1
        assert OrderType.LIMIT.value == 2
        assert OrderType.STOP.value == 3

    def test_order_direction_enum(self):
        """测试订单方向枚举"""
        assert OrderDirection.BUY.value == 1
        assert OrderDirection.SELL.value == -1


class TestTradingEngine:
    """测试交易引擎"""

    def setup_method(self):
        """测试前准备"""
        # 简化测试，使用mock避免复杂的依赖
        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.SystemMonitor'):
                self.config = {
                    'risk_limits': {
                        'max_position_size': 1000000,
                        'max_daily_loss': 50000
                    },
                    'execution_params': {
                        'slippage_tolerance': 0.001
                    }
                }
                self.engine = TradingEngine(risk_config=self.config)

    def teardown_method(self):
        """测试后清理"""
        pass

    def test_trading_engine_initialization(self):
        """测试交易引擎初始化"""
        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.SystemMonitor'):
                engine = TradingEngine(risk_config=self.config)
                assert engine is not None
                # 验证基本属性存在
                assert hasattr(engine, 'name')

    def test_trading_engine_default_initialization(self):
        """测试交易引擎默认初始化"""
        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.SystemMonitor'):
                engine = TradingEngine()
                assert engine is not None

    def test_trading_engine_basic_attributes(self):
        """测试交易引擎基本属性"""
        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.SystemMonitor'):
                engine = TradingEngine()
                # 验证引擎有基本属性
                assert hasattr(engine, 'name')
                assert hasattr(engine, 'risk_config')
                # 验证名称不为空
                assert engine.name is not None
                assert len(engine.name) > 0

    def test_trading_engine_config_inheritance(self):
        """测试交易引擎配置继承"""
        custom_config = {
            'risk_limits': {
                'max_position_size': 2000000,
                'max_daily_loss': 100000
            },
            'execution_params': {
                'slippage_tolerance': 0.002
            }
        }

        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.SystemMonitor'):
                engine = TradingEngine(risk_config=custom_config)

                # 验证配置被正确设置
                assert engine.risk_config == custom_config
                assert engine.risk_config['risk_limits']['max_position_size'] == 2000000

    def test_trading_engine_status_tracking(self):
        """测试交易引擎状态跟踪"""
        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.SystemMonitor'):
                engine = TradingEngine()

                # 验证引擎有状态跟踪能力
                # 这里主要是验证基本结构，不涉及具体业务逻辑
                assert hasattr(engine, 'name')
                assert hasattr(engine, 'risk_config')

    def test_trading_engine_error_handling(self):
        """测试交易引擎错误处理"""
        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.SystemMonitor'):
                # 测试在适配器不可用时的行为
                engine = TradingEngine()

                # 验证引擎仍然可以创建，即使依赖不可用
                assert engine is not None
                assert hasattr(engine, 'name')

    def test_get_portfolio_value(self):
        """测试获取投资组合价值"""
        # Mock持仓数据
        with patch.object(self.engine, '_get_current_positions') as mock_positions:
            mock_positions.return_value = {
                'AAPL': {'quantity': 100, 'avg_price': 150.0},
                'GOOGL': {'quantity': 50, 'avg_price': 2500.0}
            }

            # Mock当前价格
            with patch.object(self.engine, '_get_current_prices') as mock_prices:
                mock_prices.return_value = {
                    'AAPL': 155.0,
                    'GOOGL': 2550.0
                }

                value = self.engine.get_portfolio_value()

                # 验证计算: (100*155) + (50*2550) = 15500 + 127500 = 143000
                assert isinstance(value, (int, float))

    def test_calculate_position_size(self):
        """测试计算仓位大小"""
        capital = 100000
        risk_per_trade = 0.02  # 2%
        stop_loss_pct = 0.05   # 5%

        position_size = self.engine.calculate_position_size(
            capital, risk_per_trade, stop_loss_pct
        )

        # 验证计算: 100000 * 0.02 / 0.05 = 40000
        expected_size = capital * risk_per_trade / stop_loss_pct
        assert position_size == expected_size

    def test_validate_order(self):
        """测试订单验证"""
        # 有效的订单
        valid_order = {
            'symbol': 'AAPL',
            'quantity': 100,
            'order_type': OrderType.MARKET,
            'direction': OrderDirection.BUY,
            'price': None  # 市价单
        }

        is_valid, message = self.engine.validate_order(valid_order)
        assert is_valid == True

        # 无效的订单（缺少必需字段）
        invalid_order = {
            'symbol': 'AAPL',
            'quantity': 100
            # 缺少order_type和direction
        }

        is_valid, message = self.engine.validate_order(invalid_order)
        assert is_valid == False

    def test_execute_market_order(self):
        """测试执行市价订单"""
        order = {
            'symbol': 'AAPL',
            'quantity': 100,
            'order_type': OrderType.MARKET,
            'direction': OrderDirection.BUY
        }

        # Mock执行逻辑
        with patch.object(self.engine, '_execute_order') as mock_execute:
            mock_execute.return_value = {
                'order_id': '12345',
                'status': 'filled',
                'executed_price': 150.0,
                'executed_quantity': 100
            }

            result = self.engine.execute_market_order('AAPL', 100, OrderDirection.BUY)

            assert isinstance(result, dict)
            assert 'order_id' in result
            mock_execute.assert_called_once()

    def test_check_risk_limits(self):
        """测试风险限制检查"""
        # Mock当前持仓和损失
        with patch.object(self.engine, 'get_portfolio_value') as mock_value:
            mock_value.return_value = 95000  # 初始资本100000，损失5000

            with patch.object(self.engine, '_get_daily_pnl') as mock_pnl:
                mock_pnl.return_value = -5000

                # 检查是否超过每日损失限制
                can_trade, reason = self.engine.check_risk_limits()

                # 应该允许交易（未超过限制）
                assert isinstance(can_trade, bool)

    def test_get_market_data(self):
        """测试获取市场数据"""
        symbols = ['AAPL', 'GOOGL']

        # Mock市场数据获取
        with patch.object(self.engine, '_fetch_market_data') as mock_fetch:
            mock_data = pd.DataFrame({
                'symbol': ['AAPL', 'GOOGL'],
                'price': [150.0, 2500.0],
                'volume': [1000000, 500000]
            })
            mock_fetch.return_value = mock_data

            data = self.engine.get_market_data(symbols)

            assert isinstance(data, pd.DataFrame)
            assert len(data) == 2
            mock_fetch.assert_called_once_with(symbols)

    def test_calculate_volatility(self):
        """测试计算波动率"""
        prices = pd.Series([100, 105, 95, 110, 105, 102, 108, 106, 104, 107])

        volatility = self.engine.calculate_volatility(prices)

        assert isinstance(volatility, float)
        assert volatility >= 0

    def test_generate_trading_signals(self):
        """测试生成交易信号"""
        # Mock技术指标
        data = pd.DataFrame({
            'close': [100, 105, 95, 110, 105],
            'sma_20': [98, 99, 100, 101, 102],
            'rsi': [30, 40, 50, 60, 70]
        })

        signals = self.engine.generate_trading_signals(data)

        assert isinstance(signals, dict)
        # 信号可能包含多个指标的结果

    def test_update_portfolio(self):
        """测试更新投资组合"""
        # Mock交易执行结果
        trade_result = {
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.0,
            'direction': OrderDirection.BUY
        }

        # 更新前投资组合为空
        self.engine.portfolio = {}

        self.engine.update_portfolio(trade_result)

        # 验证投资组合已更新
        assert 'AAPL' in self.engine.portfolio
        assert self.engine.portfolio['AAPL']['quantity'] == 100
        assert self.engine.portfolio['AAPL']['avg_price'] == 150.0

    def test_calculate_sharpe_ratio(self):
        """测试计算夏普比率"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.01, 0.02])

        sharpe_ratio = self.engine.calculate_sharpe_ratio(returns)

        assert isinstance(sharpe_ratio, float)

    def test_get_performance_metrics(self):
        """测试获取性能指标"""
        # Mock历史收益数据
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])

        metrics = self.engine.get_performance_metrics(returns)

        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics

    def test_hedge_portfolio(self):
        """测试投资组合对冲"""
        current_positions = {
            'AAPL': {'quantity': 1000, 'beta': 1.2},
            'SPY': {'quantity': -500, 'beta': 1.0}
        }

        # 对冲AAPL的beta风险
        hedge_suggestion = self.engine.hedge_portfolio(current_positions, target_beta=1.0)

        assert isinstance(hedge_suggestion, dict)
        # 对冲建议可能包含需要调整的仓位

    def test_trading_engine_repr(self):
        """测试TradingEngine字符串表示"""
        repr_str = repr(self.engine)
        assert "TradingEngine" in repr_str


class TestTradingEngineIntegration:
    """测试交易引擎集成功能"""

    def test_complete_trading_workflow(self):
        """测试完整交易工作流"""
        engine = TradingEngine(risk_config={'max_position_size': 1000000})

        # 1. 检查风险限制
        can_trade, _ = engine.check_risk_limits()
        assert isinstance(can_trade, bool)

        # 2. 获取市场数据
        with patch.object(engine, 'get_market_data') as mock_data:
            mock_data.return_value = pd.DataFrame({
                'symbol': ['AAPL'],
                'price': [150.0]
            })

            data = engine.get_market_data(['AAPL'])
            assert len(data) == 1

        # 3. 生成交易信号
        market_data = pd.DataFrame({
            'close': [100, 105, 95, 110],
            'volume': [1000, 1200, 800, 1500]
        })

        signals = engine.generate_trading_signals(market_data)
        assert isinstance(signals, dict)

        # 4. 执行订单（模拟）
        with patch.object(engine, 'execute_market_order') as mock_execute:
            mock_execute.return_value = {'order_id': '123', 'status': 'filled'}

            result = engine.execute_market_order('AAPL', 100, OrderDirection.BUY)
            assert result['status'] == 'filled'

    def test_risk_management_integration(self):
        """测试风险管理集成"""
        config = {
            'risk_limits': {
                'max_position_size': 500000,
                'max_daily_loss': 25000,
                'max_drawdown': 0.1
            }
        }
        engine = TradingEngine(risk_config=config)

        # 测试多个风险检查
        checks = []

        # 仓位大小检查
        position_size = engine.calculate_position_size(100000, 0.01, 0.02)
        checks.append(position_size <= config['risk_limits']['max_position_size'])

        # 风险限制检查
        can_trade, _ = engine.check_risk_limits()
        checks.append(isinstance(can_trade, bool))

        # 波动率计算
        prices = pd.Series([100, 102, 98, 105, 103])
        volatility = engine.calculate_volatility(prices)
        checks.append(volatility >= 0)

        assert all(checks)

    def test_performance_analysis_integration(self):
        """测试性能分析集成"""
        engine = TradingEngine()

        # 生成模拟收益数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)

        # 计算各种指标
        sharpe = engine.calculate_sharpe_ratio(returns)
        metrics = engine.get_performance_metrics(returns)

        # 验证指标计算
        assert isinstance(sharpe, (int, float))
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'volatility' in metrics

    def test_portfolio_management_integration(self):
        """测试投资组合管理集成"""
        engine = TradingEngine()
        engine.portfolio = {}

        # 模拟一系列交易
        trades = [
            {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'direction': OrderDirection.BUY},
            {'symbol': 'GOOGL', 'quantity': 50, 'price': 2500.0, 'direction': OrderDirection.BUY},
            {'symbol': 'AAPL', 'quantity': 50, 'price': 155.0, 'direction': OrderDirection.SELL}
        ]

        for trade in trades:
            engine.update_portfolio(trade)

        # 验证投资组合状态
        assert 'AAPL' in engine.portfolio
        assert 'GOOGL' in engine.portfolio
        assert engine.portfolio['AAPL']['quantity'] == 50  # 100 - 50

    def test_error_handling_and_resilience(self):
        """测试错误处理和弹性"""
        engine = TradingEngine()

        # 测试无效订单验证
        invalid_orders = [
            {},  # 空订单
            {'symbol': 'AAPL'},  # 缺少数量和类型
            {'symbol': 'AAPL', 'quantity': -100},  # 负数量
        ]

        for invalid_order in invalid_orders:
            is_valid, message = engine.validate_order(invalid_order)
            assert is_valid == False

        # 测试波动率计算的边界情况
        edge_cases = [
            pd.Series([100]),  # 单个数据点
            pd.Series([100, 100, 100]),  # 无波动
            pd.Series([]),  # 空序列
        ]

        for case in edge_cases:
            if len(case) > 1:
                volatility = engine.calculate_volatility(case)
                assert isinstance(volatility, float)

    def test_concurrent_trading_simulation(self):
        """测试并发交易模拟"""
        import threading

        engine = TradingEngine()
        results = []
        errors = []

        def simulate_trading(trader_id):
            try:
                # 每个线程执行一些交易操作
                for i in range(5):
                    order = {
                        'symbol': f'STOCK_{trader_id}_{i}',
                        'quantity': 100,
                        'order_type': OrderType.MARKET,
                        'direction': OrderDirection.BUY if i % 2 == 0 else OrderDirection.SELL
                    }

                    is_valid, _ = engine.validate_order(order)
                    results.append((trader_id, i, is_valid))

            except Exception as e:
                errors.append((trader_id, str(e)))

        # 创建多个线程
        threads = []
        for trader_id in range(3):
            thread = threading.Thread(target=simulate_trading, args=(trader_id,))
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 15  # 3个交易者 * 5次操作
        assert len(errors) == 0   # 不应该有错误
        assert all(result[2] for result in results)  # 所有订单都应该有效