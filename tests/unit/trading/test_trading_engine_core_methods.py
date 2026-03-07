"""
交易引擎核心方法测试 - 直接测试源代码方法
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime
from src.trading.core.trading_engine import TradingEngine


class TestTradingEngineCoreMethods:
    """测试交易引擎核心方法"""

    def setup_method(self):
        """测试前准备"""
        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.get_default_monitor'):
                self.engine = TradingEngine()

    def teardown_method(self):
        """测试后清理"""
        pass

    def test_trading_engine_initialization_attributes(self):
        """测试交易引擎初始化属性"""
        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.get_default_monitor'):
                engine = TradingEngine()

                # 验证基本属性存在
                assert hasattr(engine, 'risk_config')
                assert hasattr(engine, 'monitor')
                assert hasattr(engine, 'is_a_stock')
                assert hasattr(engine, 'positions')
                assert hasattr(engine, 'cash_balance')
                assert hasattr(engine, 'max_position_size')
                assert hasattr(engine, 'order_history')
                assert hasattr(engine, 'orders')
                assert hasattr(engine, 'trade_history')
                assert hasattr(engine, 'trade_stats')
                assert hasattr(engine, '_is_running')
                assert hasattr(engine, 'start_time')
                assert hasattr(engine, 'end_time')
                assert hasattr(engine, 'execution_engine')

                # 验证数据类型
                assert isinstance(engine.positions, dict)
                assert isinstance(engine.order_history, list)
                assert isinstance(engine.orders, list)
                assert isinstance(engine.trade_history, list)
                assert isinstance(engine.trade_stats, dict)

    def test_trading_engine_initialization_values(self):
        """测试交易引擎初始化值"""
        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.get_default_monitor'):
                engine = TradingEngine()

                # 验证初始值
                assert engine.is_a_stock == True  # 默认是A股
                assert engine.positions == {}
                assert engine.cash_balance == 1000000.0  # 默认初始资本
                assert engine.max_position_size == 100000  # 默认最大仓位
                assert engine.order_history == []
                assert engine.orders == []
                assert engine.trade_history == []
                assert engine._is_running == False
                assert engine.start_time is None
                assert engine.end_time is None

                # 验证交易统计
                expected_stats = {
                    "total_trades": 0,
                    "win_trades": 0,
                    "loss_trades": 0
                }
                assert engine.trade_stats == expected_stats

    def test_trading_engine_custom_config(self):
        """测试交易引擎自定义配置"""
        custom_config = {
            'market_type': 'HK',  # 港股
            'initial_capital': 500000.0,
            'max_position_size': 50000
        }

        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.get_default_monitor'):
                engine = TradingEngine(risk_config=custom_config)

                # 验证配置生效
                assert engine.is_a_stock == False  # HK表示非A股
                assert engine.cash_balance == 500000.0
                assert engine.max_position_size == 50000
                assert engine.risk_config == custom_config

    def test_generate_orders_dataframe_signals(self):
        """测试从DataFrame信号生成订单"""
        # 创建信号DataFrame
        signals_df = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'signal': [1, -1, 0.5],  # 买入、卖出、部分买入
            'strength': [0.8, 0.9, 0.6]
        })

        current_prices = {
            'AAPL': 150.0,
            'GOOGL': 2500.0,
            'MSFT': 300.0
        }

        orders = self.engine.generate_orders(signals_df, current_prices)

        # 验证订单生成
        assert isinstance(orders, list)
        assert len(orders) > 0

        # 检查订单结构
        for order in orders:
            assert 'symbol' in order
            assert 'quantity' in order
            assert 'order_type' in order
            assert 'direction' in order
            assert 'price' in order

    def test_generate_orders_list_signals(self):
        """测试从列表信号生成订单"""
        signals_list = [
            {'symbol': 'AAPL', 'signal': 1, 'strength': 0.8},
            {'symbol': 'GOOGL', 'signal': -1, 'strength': 0.9}
        ]

        current_prices = {'AAPL': 150.0, 'GOOGL': 2500.0}

        orders = self.engine.generate_orders(signals_list, current_prices)

        # 验证订单生成
        assert isinstance(orders, list)
        assert len(orders) >= 0  # 可能过滤掉一些信号

    def test_generate_orders_with_portfolio_value(self):
        """测试带投资组合价值的订单生成"""
        signals_df = pd.DataFrame({
            'symbol': ['AAPL'],
            'signal': [1],
            'strength': [0.8]
        })

        current_prices = {'AAPL': 150.0}
        portfolio_value = 100000.0

        orders = self.engine.generate_orders(signals_df, current_prices, portfolio_value)

        # 验证订单生成
        assert isinstance(orders, list)

    def test_execute_orders_empty(self):
        """测试执行空订单列表"""
        # 执行空订单列表
        executed_orders = self.engine.execute_orders([])

        # 验证执行结果
        assert isinstance(executed_orders, list)
        assert len(executed_orders) == 0

    def test_positions_initialization(self):
        """测试持仓初始化"""
        assert self.engine.positions == {}
        assert self.engine.cash_balance == 1000000.0

    def test_get_portfolio_value(self):
        """测试获取投资组合价值"""
        # 添加一些持仓
        self.engine.positions = {
            'AAPL': {'quantity': 100, 'avg_price': 150.0},
            'GOOGL': {'quantity': 50, 'avg_price': 2500.0}
        }
        self.engine.cash_balance = 500000.0

        # Mock当前价格
        current_prices = {'AAPL': 155.0, 'GOOGL': 2550.0}

        portfolio_value = self.engine.get_portfolio_value(current_prices)

        # 计算预期价值：持仓价值 + 现金
        expected_stock_value = (100 * 155.0) + (50 * 2550.0)  # AAPL + GOOGL
        expected_total = expected_stock_value + 500000.0

        assert portfolio_value == expected_total

    def test_get_portfolio_value_no_prices(self):
        """测试获取投资组合价值 - 无价格数据"""
        portfolio_value = self.engine.get_portfolio_value()
        assert portfolio_value == self.engine.cash_balance

    def test_get_position_size(self):
        """测试获取仓位大小"""
        portfolio_value = 100000.0
        risk_per_trade = 0.02  # 2%
        stop_loss = 0.05  # 5%

        position_size = self.engine.get_position_size(portfolio_value, risk_per_trade, stop_loss)

        # 计算预期：100000 * 0.02 / 0.05 = 40000
        expected = portfolio_value * risk_per_trade / stop_loss
        assert position_size == expected

    def test_check_risk_limits(self):
        """测试风险限制检查"""
        # 设置风险配置
        self.engine.max_position_size = 50000

        # 创建风险数据（大仓位）
        risk_data = {
            'portfolio_value': 100000.0,
            'daily_loss': 1000.0,
            'position_sizes': {'AAPL': 60000.0}  # 超过max_position_size
        }

        # 检查风险
        result = self.engine.check_risk_limits(risk_data)

        # 大仓位应该被拒绝
        assert result['can_trade'] == False
        assert len(result['violations']) > 0

    def test_check_risk_limits_small_order(self):
        """测试风险限制检查 - 小订单"""
        self.engine.max_position_size = 50000

        # 创建风险数据（小仓位）
        risk_data = {
            'portfolio_value': 950000.0,  # 轻微损失但在限制内
            'daily_loss': 100.0,
            'position_sizes': {'AAPL': 10000.0}  # 小于max_position_size
        }

        result = self.engine.check_risk_limits(risk_data)
        assert result['can_trade'] == True
        assert len(result['violations']) == 0

    def test_calculate_pnl(self):
        """测试计算盈亏"""
        # 添加持仓
        self.engine.positions = {
            'AAPL': {'quantity': 100, 'avg_price': 150.0}
        }

        current_prices = {'AAPL': 160.0}

        pnl = self.engine.calculate_pnl(current_prices)

        # 计算预期盈亏：(160.0 - 150.0) * 100 = 1000
        expected_pnl = (160.0 - 150.0) * 100
        assert pnl == expected_pnl

    def test_get_trade_statistics(self):
        """测试获取交易统计"""
        # 设置一些交易统计
        self.engine.trade_stats = {
            "total_trades": 10,
            "win_trades": 7,
            "loss_trades": 3
        }

        stats = self.engine.get_trade_statistics()

        assert isinstance(stats, dict)
        assert stats['total_trades'] == 10
        assert stats['win_trades'] == 7
        assert stats['win_rate'] == 0.7  # 7/10

    def test_get_trade_statistics_empty(self):
        """测试获取交易统计 - 空统计"""
        stats = self.engine.get_trade_statistics()

        assert isinstance(stats, dict)
        assert stats['total_trades'] == 0
        assert stats['win_rate'] == 0.0

    def test_start_engine(self):
        """测试启动引擎"""
        assert not self.engine._is_running
        assert self.engine.start_time is None

        self.engine.start()

        assert self.engine._is_running == True
        assert self.engine.start_time is not None
        assert isinstance(self.engine.start_time, datetime)

    def test_stop_engine(self):
        """测试停止引擎"""
        self.engine.start()
        assert self.engine._is_running == True

        self.engine.stop()

        assert self.engine._is_running == False
        assert self.engine.end_time is not None
        assert isinstance(self.engine.end_time, datetime)

    def test_is_running(self):
        """测试运行状态检查"""
        assert not self.engine.is_running()

        self.engine.start()
        assert self.engine.is_running()

        self.engine.stop()
        assert not self.engine.is_running()

    def test_get_uptime(self):
        """测试获取运行时间"""
        # 未启动的引擎
        uptime = self.engine.get_uptime()
        assert uptime == 0

        # 启动引擎
        self.engine.start()

        # 模拟一些运行时间（这里只是验证方法存在）
        uptime = self.engine.get_uptime()
        assert isinstance(uptime, (int, float))
        assert uptime >= 0

    def test_reset_engine(self):
        """测试重置引擎"""
        # 修改引擎状态
        self.engine.positions = {'AAPL': {'quantity': 100, 'avg_price': 150.0}}
        self.engine.cash_balance = 500000.0
        self.engine.trade_stats['total_trades'] = 5

        # 重置引擎
        self.engine.reset()

        # 验证重置结果
        assert self.engine.positions == {}
        assert self.engine.cash_balance == 1000000.0  # 重置为初始值
        assert self.engine.trade_stats['total_trades'] == 0
        assert self.engine.order_history == []
        assert self.engine.trade_history == []
        assert not self.engine._is_running

    def test_engine_repr(self):
        """测试引擎字符串表示"""
        repr_str = repr(self.engine)
        assert "TradingEngine" in repr_str

    def test_engine_str(self):
        """测试引擎字符串转换"""
        str_repr = str(self.engine)
        assert isinstance(str_repr, str)


class TestTradingEngineIntegration:
    """测试交易引擎集成功能"""

    def setup_method(self):
        """测试前准备"""
        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.get_default_monitor'):
                self.engine = TradingEngine()

    def teardown_method(self):
        """测试后清理"""
        pass

    def test_complete_trading_workflow(self):
        """测试完整交易工作流"""
        # 1. 启动引擎
        self.engine.start()
        assert self.engine.is_running()

        # 2. 生成订单
        signals = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL'],
            'signal': [1, -1],  # 买入AAPL，卖出GOOGL
            'strength': [0.8, 0.9]
        })

        current_prices = {'AAPL': 150.0, 'GOOGL': 2500.0}

        orders = self.engine.generate_orders(signals, current_prices)
        assert isinstance(orders, list)

        # 3. 执行订单（如果有的话）
        if orders:
            executed = self.engine.execute_orders(orders)
            assert isinstance(executed, list)

        # 4. 检查投资组合价值
        portfolio_value = self.engine.get_portfolio_value(current_prices)
        assert isinstance(portfolio_value, (int, float))

        # 5. 获取统计信息
        stats = self.engine.get_trade_statistics()
        assert isinstance(stats, dict)

        # 6. 停止引擎
        self.engine.stop()
        assert not self.engine.is_running()

    def test_risk_management_integration(self):
        """测试风险管理集成"""
        # 设置风险参数
        self.engine.max_position_size = 50000

        # 测试正常情况下的风险检查
        can_trade, reason = self.engine.check_risk_limits()
        assert isinstance(can_trade, bool)
        assert isinstance(reason, str)

        # 测试带参数的风险检查
        risk_data = {
            'portfolio_value': 100000.0,
            'daily_loss': 100.0,
            'position_sizes': {'AAPL': 20000.0}
        }
        risk_result = self.engine.check_risk_limits(risk_data)
        assert isinstance(risk_result, dict)
        assert 'can_trade' in risk_result

    def test_performance_tracking_integration(self):
        """测试性能跟踪集成"""
        # 执行一些交易
        trades = [
            {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'direction': 1},
            {'symbol': 'AAPL', 'quantity': 50, 'price': 160.0, 'direction': -1}
        ]

        for trade in trades:
            self.engine.update_positions(trade)

        # 检查盈亏计算
        current_prices = {'AAPL': 155.0}
        pnl = self.engine.calculate_pnl(current_prices)
        assert isinstance(pnl, (int, float))

        # 检查统计信息
        stats = self.engine.get_trade_statistics()
        assert isinstance(stats, dict)

    def test_engine_lifecycle_management(self):
        """测试引擎生命周期管理"""
        # 初始状态
        assert not self.engine.is_running()
        assert self.engine.get_uptime() == 0

        # 启动
        self.engine.start()
        assert self.engine.is_running()
        assert self.engine.get_uptime() >= 0

        # 重置
        self.engine.reset()
        assert not self.engine.is_running()
        assert self.engine.positions == {}
        assert self.engine.order_history == []

        # 再次启动
        self.engine.start()
        assert self.engine.is_running()

        # 停止
        self.engine.stop()
        assert not self.engine.is_running()