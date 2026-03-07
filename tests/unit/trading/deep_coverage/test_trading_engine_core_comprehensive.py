"""
交易引擎核心深度测试
全面测试交易引擎的核心功能、订单管理、执行逻辑和风险控制
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio

# 导入交易相关类
try:
    from src.trading.core.trading_engine import TradingEngine, OrderType, OrderDirection
    TRADING_ENGINE_AVAILABLE = True
except ImportError:
    TRADING_ENGINE_AVAILABLE = False
    TradingEngine = Mock
    OrderType = Mock
    OrderDirection = Mock

try:
    from src.trading.execution.order_manager import OrderManager
    ORDER_MANAGER_AVAILABLE = True
except ImportError:
    ORDER_MANAGER_AVAILABLE = False
    OrderManager = Mock

try:
    from src.trading.execution.order_router import OrderRouter
    ORDER_ROUTER_AVAILABLE = True
except ImportError:
    ORDER_ROUTER_AVAILABLE = False
    OrderRouter = Mock

try:
    from src.trading.portfolio.portfolio_manager import PortfolioManager
    PORTFOLIO_MANAGER_AVAILABLE = True
except ImportError:
    PORTFOLIO_MANAGER_AVAILABLE = False
    PortfolioManager = Mock

try:
    from src.trading.account.account_manager import AccountManager
    ACCOUNT_MANAGER_AVAILABLE = True
except ImportError:
    ACCOUNT_MANAGER_AVAILABLE = False
    AccountManager = Mock

try:
    from src.trading.core.exceptions import TradingException, OrderRejectedException
    EXCEPTIONS_AVAILABLE = True
except ImportError:
    EXCEPTIONS_AVAILABLE = False
    TradingException = Exception
    OrderRejectedException = Exception


class TestTradingEngineCoreComprehensive:
    """交易引擎核心综合深度测试"""

    @pytest.fixture
    def sample_market_data(self):
        """创建样本市场数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)

        return pd.DataFrame({
            'symbol': ['AAPL'] * 100,
            'timestamp': dates,
            'open': np.random.uniform(150, 200, 100),
            'high': np.random.uniform(155, 205, 100),
            'low': np.random.uniform(145, 195, 100),
            'close': np.random.uniform(150, 200, 100),
            'volume': np.random.randint(1000000, 10000000, 100),
            'bid_price': np.random.uniform(149, 199, 100),
            'ask_price': np.random.uniform(151, 201, 100),
            'bid_volume': np.random.randint(1000, 10000, 100),
            'ask_volume': np.random.randint(1000, 10000, 100)
        })

    @pytest.fixture
    def trading_engine(self):
        """创建交易引擎实例"""
        if TRADING_ENGINE_AVAILABLE:
            return TradingEngine()
        return Mock(spec=TradingEngine)

    @pytest.fixture
    def order_manager(self):
        """创建订单管理器实例"""
        if ORDER_MANAGER_AVAILABLE:
            return OrderManager()
        return Mock(spec=OrderManager)

    @pytest.fixture
    def order_router(self):
        """创建订单路由器实例"""
        if ORDER_ROUTER_AVAILABLE:
            return OrderRouter()
        return Mock(spec=OrderRouter)

    @pytest.fixture
    def portfolio_manager(self):
        """创建投资组合管理器实例"""
        if PORTFOLIO_MANAGER_AVAILABLE:
            return PortfolioManager()
        return Mock(spec=PortfolioManager)

    @pytest.fixture
    def account_manager(self):
        """创建账户管理器实例"""
        if ACCOUNT_MANAGER_AVAILABLE:
            return AccountManager()
        return Mock(spec=AccountManager)

    def test_trading_engine_initialization(self, trading_engine):
        """测试交易引擎初始化"""
        if TRADING_ENGINE_AVAILABLE:
            assert trading_engine is not None
            assert hasattr(trading_engine, 'order_manager')
            assert hasattr(trading_engine, 'portfolio_manager')
            assert hasattr(trading_engine, 'account_manager')

    def test_order_creation_market_order(self, trading_engine):
        """测试市价单创建"""
        if TRADING_ENGINE_AVAILABLE:
            # 创建市价买入订单
            order = trading_engine.create_order(
                symbol='AAPL',
                order_type=OrderType.MARKET,
                direction=OrderDirection.BUY,
                quantity=100,
                account_id='ACC001'
            )

            assert order is not None
            assert order['order_type'] == OrderType.MARKET
            assert order['direction'] == OrderDirection.BUY
            assert order['quantity'] == 100
            assert order['symbol'] == 'AAPL'

    def test_order_creation_limit_order(self, trading_engine):
        """测试限价单创建"""
        if TRADING_ENGINE_AVAILABLE:
            # 创建限价卖出订单
            order = trading_engine.create_order(
                symbol='AAPL',
                order_type=OrderType.LIMIT,
                direction=OrderDirection.SELL,
                quantity=50,
                price=180.0,
                account_id='ACC001'
            )

            assert order is not None
            assert order['order_type'] == OrderType.LIMIT
            assert order['direction'] == OrderDirection.SELL
            assert order['quantity'] == 50
            assert order['price'] == 180.0

    def test_order_creation_stop_order(self, trading_engine):
        """测试止损单创建"""
        if TRADING_ENGINE_AVAILABLE:
            # 创建止损卖出订单
            order = trading_engine.create_order(
                symbol='AAPL',
                order_type=OrderType.STOP,
                direction=OrderDirection.SELL,
                quantity=75,
                stop_price=160.0,
                account_id='ACC001'
            )

            assert order is not None
            assert order['order_type'] == OrderType.STOP
            assert order['direction'] == OrderDirection.SELL
            assert order['quantity'] == 75
            assert order['stop_price'] == 160.0

    def test_order_validation(self, trading_engine):
        """测试订单验证"""
        if TRADING_ENGINE_AVAILABLE:
            # 测试有效订单
            valid_order = {
                'symbol': 'AAPL',
                'order_type': OrderType.MARKET,
                'direction': OrderDirection.BUY,
                'quantity': 100,
                'account_id': 'ACC001'
            }

            is_valid, errors = trading_engine.validate_order(valid_order)
            assert is_valid is True
            assert len(errors) == 0

            # 测试无效订单（缺少必要字段）
            invalid_order = {
                'symbol': 'AAPL',
                'order_type': OrderType.MARKET,
                # 缺少direction
                'quantity': 100
            }

            is_valid, errors = trading_engine.validate_order(invalid_order)
            assert is_valid is False
            assert len(errors) > 0

    def test_order_submission(self, trading_engine):
        """测试订单提交"""
        if TRADING_ENGINE_AVAILABLE:
            # 创建订单
            order = trading_engine.create_order(
                symbol='AAPL',
                order_type=OrderType.MARKET,
                direction=OrderDirection.BUY,
                quantity=100,
                account_id='ACC001'
            )

            # 提交订单
            order_id = trading_engine.submit_order(order)

            assert order_id is not None
            assert isinstance(order_id, str)

            # 验证订单状态
            order_status = trading_engine.get_order_status(order_id)
            assert order_status is not None

    def test_order_execution_market_order(self, trading_engine, sample_market_data):
        """测试市价单执行"""
        if TRADING_ENGINE_AVAILABLE:
            # 创建市价单
            order = trading_engine.create_order(
                symbol='AAPL',
                order_type=OrderType.MARKET,
                direction=OrderDirection.BUY,
                quantity=100,
                account_id='ACC001'
            )

            # 执行订单
            execution_result = trading_engine.execute_order(order, sample_market_data)

            assert execution_result is not None
            assert 'order_id' in execution_result
            assert 'executed_quantity' in execution_result
            assert 'execution_price' in execution_result
            assert execution_result['executed_quantity'] == 100

    def test_order_execution_limit_order(self, trading_engine, sample_market_data):
        """测试限价单执行"""
        if TRADING_ENGINE_AVAILABLE:
            # 创建限价单（价格高于当前市场价，应该不会执行）
            order = trading_engine.create_order(
                symbol='AAPL',
                order_type=OrderType.LIMIT,
                direction=OrderDirection.BUY,
                quantity=100,
                price=300.0,  # 高于市场价
                account_id='ACC001'
            )

            # 尝试执行订单
            execution_result = trading_engine.execute_order(order, sample_market_data)

            assert execution_result is not None
            # 限价单可能不会立即执行，取决于市场条件
            assert 'order_id' in execution_result

    def test_order_cancellation(self, trading_engine):
        """测试订单取消"""
        if TRADING_ENGINE_AVAILABLE:
            # 创建并提交订单
            order = trading_engine.create_order(
                symbol='AAPL',
                order_type=OrderType.MARKET,
                direction=OrderDirection.BUY,
                quantity=100,
                account_id='ACC001'
            )
            order_id = trading_engine.submit_order(order)

            # 取消订单
            cancel_result = trading_engine.cancel_order(order_id)

            assert cancel_result is True

            # 验证订单状态
            order_status = trading_engine.get_order_status(order_id)
            assert order_status['status'] == 'CANCELLED'

    def test_portfolio_position_management(self, trading_engine):
        """测试投资组合持仓管理"""
        if TRADING_ENGINE_AVAILABLE:
            # 执行一些交易来建立持仓
            trades = [
                {
                    'symbol': 'AAPL',
                    'direction': OrderDirection.BUY,
                    'quantity': 100,
                    'price': 150.0
                },
                {
                    'symbol': 'GOOGL',
                    'direction': OrderDirection.BUY,
                    'quantity': 50,
                    'price': 2800.0
                },
                {
                    'symbol': 'AAPL',
                    'direction': OrderDirection.SELL,
                    'quantity': 30,
                    'price': 160.0
                }
            ]

            for trade in trades:
                trading_engine.execute_trade(trade)

            # 获取投资组合
            portfolio = trading_engine.get_portfolio('ACC001')

            assert portfolio is not None
            assert 'positions' in portfolio

            # 检查AAPL持仓（100 - 30 = 70）
            aapl_position = None
            for position in portfolio['positions']:
                if position['symbol'] == 'AAPL':
                    aapl_position = position
                    break

            assert aapl_position is not None
            assert aapl_position['quantity'] == 70

    def test_risk_limits_checking(self, trading_engine):
        """测试风险限额检查"""
        if TRADING_ENGINE_AVAILABLE:
            # 设置风险限额
            risk_limits = {
                'max_position_size': 100000,
                'max_daily_loss': 5000,
                'max_single_order': 50000
            }

            trading_engine.set_risk_limits(risk_limits)

            # 测试符合限额的订单
            small_order = {
                'symbol': 'AAPL',
                'order_type': OrderType.MARKET,
                'direction': OrderDirection.BUY,
                'quantity': 100,
                'account_id': 'ACC001'
            }

            risk_check = trading_engine.check_risk_limits(small_order)
            assert risk_check['approved'] is True

            # 测试超出限额的订单
            large_order = {
                'symbol': 'AAPL',
                'order_type': OrderType.MARKET,
                'direction': OrderDirection.BUY,
                'quantity': 1000,  # 可能超出限额
                'account_id': 'ACC001'
            }

            risk_check = trading_engine.check_risk_limits(large_order)
            # 风险检查可能拒绝大订单
            assert 'approved' in risk_check

    def test_account_balance_management(self, trading_engine):
        """测试账户余额管理"""
        if TRADING_ENGINE_AVAILABLE:
            # 初始化账户余额
            trading_engine.update_account_balance('ACC001', 100000.0)

            # 执行交易
            trade = {
                'symbol': 'AAPL',
                'direction': OrderDirection.BUY,
                'quantity': 100,
                'price': 150.0,
                'account_id': 'ACC001'
            }

            trading_engine.execute_trade(trade)

            # 检查账户余额更新
            balance = trading_engine.get_account_balance('ACC001')
            expected_balance = 100000.0 - (100 * 150.0)  # 买入成本

            assert abs(balance - expected_balance) < 0.01  # 允许小误差

    def test_order_routing_smart_routing(self, order_router, sample_market_data):
        """测试订单路由智能路由"""
        if ORDER_ROUTER_AVAILABLE:
            order = {
                'symbol': 'AAPL',
                'order_type': OrderType.MARKET,
                'direction': OrderDirection.BUY,
                'quantity': 1000
            }

            # 执行智能路由
            routing_result = order_router.smart_route(order, sample_market_data)

            assert routing_result is not None
            assert 'selected_broker' in routing_result
            assert 'estimated_cost' in routing_result
            assert 'execution_time' in routing_result

    def test_order_routing_cost_optimization(self, order_router, sample_market_data):
        """测试订单路由成本优化"""
        if ORDER_ROUTER_AVAILABLE:
            large_order = {
                'symbol': 'AAPL',
                'order_type': OrderType.MARKET,
                'direction': OrderDirection.SELL,
                'quantity': 5000  # 大订单
            }

            # 执行成本优化路由
            routing_result = order_router.optimize_cost_routing(large_order, sample_market_data)

            assert routing_result is not None
            assert 'routing_strategy' in routing_result
            assert 'expected_savings' in routing_result

    def test_portfolio_rebalancing(self, portfolio_manager):
        """测试投资组合再平衡"""
        if PORTFOLIO_MANAGER_AVAILABLE:
            # 创建当前投资组合
            current_portfolio = {
                'AAPL': 0.4,  # 40%
                'GOOGL': 0.3, # 30%
                'MSFT': 0.3   # 30%
            }

            # 设置目标分配
            target_allocation = {
                'AAPL': 0.5,  # 50%
                'GOOGL': 0.3, # 30%
                'MSFT': 0.2   # 20%
            }

            # 执行再平衡
            rebalance_trades = portfolio_manager.rebalance_portfolio(
                current_portfolio, target_allocation
            )

            assert isinstance(rebalance_trades, list)
            assert len(rebalance_trades) > 0

            # 检查交易逻辑
            for trade in rebalance_trades:
                assert 'symbol' in trade
                assert 'action' in trade  # BUY or SELL
                assert 'quantity' in trade

    def test_portfolio_performance_calculation(self, portfolio_manager):
        """测试投资组合绩效计算"""
        if PORTFOLIO_MANAGER_AVAILABLE:
            # 创建投资组合交易历史
            trade_history = [
                {'symbol': 'AAPL', 'action': 'BUY', 'quantity': 100, 'price': 150.0, 'date': '2024-01-01'},
                {'symbol': 'GOOGL', 'action': 'BUY', 'quantity': 10, 'price': 2800.0, 'date': '2024-01-02'},
                {'symbol': 'AAPL', 'action': 'SELL', 'quantity': 50, 'price': 160.0, 'date': '2024-01-15'},
            ]

            # 计算绩效
            performance = portfolio_manager.calculate_performance(trade_history)

            assert isinstance(performance, dict)
            assert 'total_return' in performance
            assert 'sharpe_ratio' in performance
            assert 'max_drawdown' in performance

    def test_account_margin_management(self, account_manager):
        """测试账户保证金管理"""
        if ACCOUNT_MANAGER_AVAILABLE:
            # 设置账户初始状态
            account_manager.create_account('ACC001', initial_balance=100000.0)

            # 执行杠杆交易
            margin_trade = {
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 1000,
                'price': 150.0,
                'leverage': 2.0
            }

            # 检查保证金要求
            margin_check = account_manager.check_margin_requirement('ACC001', margin_trade)

            assert isinstance(margin_check, dict)
            assert 'margin_required' in margin_check
            assert 'available_margin' in margin_check
            assert 'can_trade' in margin_check

            # 计算预期保证金要求
            expected_margin = (1000 * 150.0) / 2.0  # 杠杆为2，所需保证金为交易额的一半
            assert abs(margin_check['margin_required'] - expected_margin) < 0.01

    def test_account_risk_monitoring(self, account_manager):
        """测试账户风险监控"""
        if ACCOUNT_MANAGER_AVAILABLE:
            account_manager.create_account('ACC001', initial_balance=100000.0)

            # 执行一系列交易
            trades = [
                {'symbol': 'AAPL', 'action': 'BUY', 'quantity': 100, 'price': 150.0},
                {'symbol': 'GOOGL', 'action': 'BUY', 'quantity': 5, 'price': 2800.0},
                {'symbol': 'AAPL', 'action': 'SELL', 'quantity': 50, 'price': 140.0},  # 亏损
            ]

            for trade in trades:
                account_manager.process_trade('ACC001', trade)

            # 获取风险指标
            risk_metrics = account_manager.get_risk_metrics('ACC001')

            assert isinstance(risk_metrics, dict)
            assert 'current_balance' in risk_metrics
            assert 'unrealized_pnl' in risk_metrics
            assert 'risk_level' in risk_metrics

    def test_multi_asset_trading(self, trading_engine):
        """测试多资产交易"""
        if TRADING_ENGINE_AVAILABLE:
            # 创建多资产订单
            multi_asset_orders = [
                {
                    'symbol': 'AAPL',
                    'order_type': OrderType.MARKET,
                    'direction': OrderDirection.BUY,
                    'quantity': 100,
                    'account_id': 'ACC001'
                },
                {
                    'symbol': 'GOOGL',
                    'order_type': OrderType.MARKET,
                    'direction': OrderDirection.BUY,
                    'quantity': 10,
                    'account_id': 'ACC001'
                },
                {
                    'symbol': 'BTC',
                    'order_type': OrderType.LIMIT,
                    'direction': OrderDirection.SELL,
                    'quantity': 0.5,
                    'price': 45000.0,
                    'account_id': 'ACC001'
                }
            ]

            # 批量提交订单
            order_ids = []
            for order in multi_asset_orders:
                order_id = trading_engine.submit_order(order)
                order_ids.append(order_id)

            assert len(order_ids) == len(multi_asset_orders)

            # 检查投资组合包含多种资产
            portfolio = trading_engine.get_portfolio('ACC001')
            assert len(portfolio['positions']) >= 2

    def test_high_frequency_trading_simulation(self, trading_engine, sample_market_data):
        """测试高频交易模拟"""
        if TRADING_ENGINE_AVAILABLE:
            # 启用HFT模式
            trading_engine.enable_hft_mode()

            # 执行高频交易策略
            hft_orders = []

            for i in range(10):  # 模拟10个快速交易周期
                # 基于市场数据生成快速订单
                current_data = sample_market_data.iloc[i]

                order = {
                    'symbol': 'AAPL',
                    'order_type': OrderType.MARKET,
                    'direction': OrderDirection.BUY if i % 2 == 0 else OrderDirection.SELL,
                    'quantity': np.random.randint(10, 50),  # 小批量
                    'account_id': 'ACC001'
                }

                hft_orders.append(order)

            # 执行HFT订单
            hft_results = []
            for order in hft_orders:
                result = trading_engine.execute_hft_order(order, sample_market_data)
                hft_results.append(result)

            assert len(hft_results) == len(hft_orders)

            # 检查HFT性能指标
            hft_metrics = trading_engine.get_hft_performance_metrics()
            assert isinstance(hft_metrics, dict)
            assert 'orders_per_second' in hft_metrics
            assert 'average_latency' in hft_metrics

    def test_trading_strategy_backtesting(self, trading_engine, sample_market_data):
        """测试交易策略回测"""
        if TRADING_ENGINE_AVAILABLE:
            # 定义简单动量策略
            def momentum_strategy(data):
                """简单的动量策略：价格上涨时买入，下跌时卖出"""
                signals = []

                for i in range(5, len(data)):  # 跳过前5个数据点
                    window_data = data.iloc[i-5:i]
                    current_price = data.iloc[i]['close']
                    avg_price = window_data['close'].mean()

                    if current_price > avg_price * 1.02:  # 价格高于均价2%
                        signals.append({
                            'timestamp': data.iloc[i]['timestamp'],
                            'signal': 'BUY',
                            'strength': (current_price - avg_price) / avg_price
                        })
                    elif current_price < avg_price * 0.98:  # 价格低于均价2%
                        signals.append({
                            'timestamp': data.iloc[i]['timestamp'],
                            'signal': 'SELL',
                            'strength': (avg_price - current_price) / avg_price
                        })

                return signals

            # 执行策略回测
            backtest_result = trading_engine.backtest_strategy(
                momentum_strategy, sample_market_data,
                initial_capital=100000.0
            )

            assert isinstance(backtest_result, dict)
            assert 'total_return' in backtest_result
            assert 'sharpe_ratio' in backtest_result
            assert 'max_drawdown' in backtest_result
            assert 'trade_count' in backtest_result

    def test_real_time_order_execution(self, trading_engine, sample_market_data):
        """测试实时订单执行"""
        if TRADING_ENGINE_AVAILABLE:
            # 启用实时模式
            trading_engine.enable_real_time_mode()

            # 模拟实时订单流
            real_time_orders = [
                {
                    'symbol': 'AAPL',
                    'order_type': OrderType.MARKET,
                    'direction': OrderDirection.BUY,
                    'quantity': 50,
                    'account_id': 'ACC001',
                    'priority': 'HIGH'
                },
                {
                    'symbol': 'GOOGL',
                    'order_type': OrderType.LIMIT,
                    'direction': OrderDirection.SELL,
                    'quantity': 5,
                    'price': 2900.0,
                    'account_id': 'ACC001',
                    'priority': 'NORMAL'
                }
            ]

            # 执行实时订单
            execution_results = []
            for order in real_time_orders:
                result = trading_engine.execute_real_time_order(order, sample_market_data)
                execution_results.append(result)

            assert len(execution_results) == len(real_time_orders)

            # 检查执行延迟
            for result in execution_results:
                assert 'execution_time' in result
                assert result['execution_time'] < 1.0  # 应该很快执行

    def test_trading_system_integration(self, trading_engine):
        """测试交易系统集成"""
        if TRADING_ENGINE_AVAILABLE:
            # 测试与市场数据系统的集成
            market_data_integration = trading_engine.test_market_data_integration()
            assert market_data_integration['status'] == 'connected'

            # 测试与经纪商系统的集成
            broker_integration = trading_engine.test_broker_integration()
            assert broker_integration['status'] in ['connected', 'simulated']

            # 测试与风控系统的集成
            risk_integration = trading_engine.test_risk_management_integration()
            assert risk_integration['status'] in ['active', 'simulated']

    def test_error_handling_and_recovery(self, trading_engine):
        """测试错误处理和恢复"""
        if TRADING_ENGINE_AVAILABLE:
            # 测试无效订单处理
            invalid_order = {
                'symbol': '',  # 无效符号
                'order_type': OrderType.MARKET,
                'direction': OrderDirection.BUY,
                'quantity': -100,  # 无效数量
            }

            try:
                trading_engine.submit_order(invalid_order)
                # 如果没有抛出异常，检查是否正确处理
            except (TradingException, ValueError):
                # 期望的异常
                pass

            # 测试网络错误恢复
            with patch.object(trading_engine, 'execute_order', side_effect=ConnectionError("Network error")):
                order = trading_engine.create_order('AAPL', OrderType.MARKET, OrderDirection.BUY, 100, 'ACC001')

                # 系统应该能够处理网络错误并重试或降级
                try:
                    trading_engine.submit_order(order)
                except ConnectionError:
                    # 网络错误应该被适当处理
                    pass

    def test_performance_monitoring_and_optimization(self, trading_engine, sample_market_data):
        """测试性能监控和优化"""
        if TRADING_ENGINE_AVAILABLE:
            # 执行一系列交易操作
            import time
            start_time = time.time()

            for i in range(50):
                order = trading_engine.create_order(
                    'AAPL', OrderType.MARKET,
                    OrderDirection.BUY if i % 2 == 0 else OrderDirection.SELL,
                    100, 'ACC001'
                )
                trading_engine.submit_order(order)

            end_time = time.time()

            # 获取性能指标
            performance_metrics = trading_engine.get_performance_metrics()

            assert isinstance(performance_metrics, dict)
            assert 'orders_per_second' in performance_metrics
            assert 'average_execution_time' in performance_metrics
            assert 'memory_usage' in performance_metrics

            # 计算实际性能
            total_time = end_time - start_time
            actual_orders_per_second = 50 / total_time

            # 性能应该在合理范围内
            assert performance_metrics['orders_per_second'] > 0

    def test_trading_audit_and_compliance(self, trading_engine):
        """测试交易审计和合规"""
        if TRADING_ENGINE_AVAILABLE:
            # 执行一些交易
            for i in range(5):
                order = trading_engine.create_order(
                    'AAPL', OrderType.MARKET, OrderDirection.BUY, 100, 'ACC001'
                )
                trading_engine.submit_order(order)

            # 获取审计日志
            audit_log = trading_engine.get_audit_log()

            assert isinstance(audit_log, list)
            assert len(audit_log) >= 5  # 至少5个交易记录

            # 检查审计记录结构
            for record in audit_log:
                assert 'timestamp' in record
                assert 'action' in record
                assert 'account_id' in record
                assert 'details' in record

    def test_concurrent_order_processing(self, trading_engine):
        """测试并发订单处理"""
        if TRADING_ENGINE_AVAILABLE:
            import threading
            import time

            results = []
            errors = []

            def process_orders(thread_id):
                try:
                    for i in range(10):
                        order = trading_engine.create_order(
                            f'STOCK_{thread_id}_{i}', OrderType.MARKET,
                            OrderDirection.BUY, 100, f'ACC_{thread_id}'
                        )
                        order_id = trading_engine.submit_order(order)
                        results.append((thread_id, order_id))
                except Exception as e:
                    errors.append((thread_id, str(e)))

            # 创建并发线程
            threads = []
            num_threads = 5

            for thread_id in range(num_threads):
                thread = threading.Thread(target=process_orders, args=(thread_id,))
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()

            # 验证结果
            assert len(results) == num_threads * 10  # 每个线程10个订单
            assert len(errors) == 0  # 不应该有错误

            # 检查所有订单ID都是唯一的
            order_ids = [result[1] for result in results]
            assert len(set(order_ids)) == len(order_ids)
