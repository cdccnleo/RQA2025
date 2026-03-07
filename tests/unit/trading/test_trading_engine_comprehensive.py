"""
交易引擎综合测试
测试TradingEngine核心功能，提升覆盖率
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.trading.core.trading_engine import TradingEngine
from src.trading.core.execution.execution_result import ExecutionResult
from src.trading.core.execution.execution_context import ExecutionContext


class TestTradingEngineComprehensive:
    """交易引擎综合测试"""

    @pytest.fixture
    def trading_engine(self):
        """创建交易引擎实例"""
        return TradingEngine()

    @pytest.fixture
    def trading_engine_with_config(self):
        """创建带配置的交易引擎实例"""
        config = {
            'max_position': 100000,
            'risk_limit': 0.02,
            'commission_rate': 0.0003
        }
        return TradingEngine(config)

    def test_trading_engine_initialization(self, trading_engine):
        """测试交易引擎初始化"""
        assert trading_engine is not None
        assert hasattr(trading_engine, 'risk_config')
        assert hasattr(trading_engine, 'monitor')

    def test_trading_engine_initialization_with_config(self, trading_engine_with_config):
        """测试交易引擎带配置初始化"""
        assert trading_engine_with_config.risk_config is not None

    def test_execute_market_order(self, trading_engine):
        """测试市价订单执行"""
        from src.trading.core.trading_engine import OrderDirection
        result = trading_engine.execute_market_order('000001', 100, OrderDirection.BUY)
        assert result is not None
        assert isinstance(result, dict)

    def test_create_order(self, trading_engine):
        """测试创建订单"""
        from src.trading.core.trading_engine import OrderType, OrderDirection
        order = trading_engine.create_order(
            symbol='000001',
            order_type=OrderType.MARKET,
            quantity=100,
            direction=OrderDirection.BUY
        )
        assert order is not None
        assert isinstance(order, dict)
        assert order['symbol'] == '000001'

    def test_validate_order(self, trading_engine):
        """测试订单验证"""
        valid_params = {
            'symbol': '000001',
            'quantity': 100,
            'price': 10.5
        }
        result = trading_engine.validate_order_params(valid_params)
        assert result is True

    def test_cancel_order(self, trading_engine):
        """测试取消订单"""
        # 先提交一个订单
        from src.trading.core.trading_engine import OrderType, OrderDirection
        order = trading_engine.create_order(
            symbol='000001',
            order_type=OrderType.MARKET,
            quantity=100,
            direction=OrderDirection.BUY
        )

        if order and 'order_id' in order:
            result = trading_engine.cancel_order(order['order_id'])
            assert isinstance(result, bool)

    def test_get_order_status(self, trading_engine):
        """测试获取订单状态"""
        # 创建并提交订单
        from src.trading.core.trading_engine import OrderType, OrderDirection
        order = trading_engine.create_order(
            symbol='000001',
            order_type=OrderType.MARKET,
            quantity=100,
            direction=OrderDirection.BUY
        )

        if order and 'order_id' in order:
            status = trading_engine.get_order_status(order['order_id'])
            assert status is not None

    def test_get_all_positions(self, trading_engine):
        """测试获取所有持仓"""
        positions = trading_engine.get_all_positions()
        assert isinstance(positions, dict)

    def test_get_account_balance(self, trading_engine):
        """测试获取账户余额"""
        balance = trading_engine.get_account_balance()
        assert isinstance(balance, (int, float))
        assert balance >= 0

    def test_calculate_portfolio_value(self, trading_engine):
        """测试计算投资组合价值"""
        value = trading_engine.calculate_portfolio_value()
        assert isinstance(value, (int, float))
        assert value >= 0

    def test_get_portfolio_pnl(self, trading_engine):
        """测试获取投资组合盈亏"""
        pnl = trading_engine.get_portfolio_pnl()
        assert isinstance(pnl, (int, float))

    def test_get_trading_statistics(self, trading_engine):
        """测试获取交易统计"""
        stats = trading_engine.get_trading_statistics()
        assert isinstance(stats, dict)

    def test_get_trade_statistics(self, trading_engine):
        """测试获取交易统计"""
        stats = trading_engine.get_trade_statistics()
        assert isinstance(stats, dict)

    def test_calculate_pnl(self, trading_engine):
        """测试计算盈亏"""
        pnl = trading_engine.calculate_pnl({})
        assert isinstance(pnl, (int, float))

    def test_get_market_data(self, trading_engine):
        """测试获取市场数据"""
        symbols = ['000001']
        market_data = trading_engine.get_market_data(symbols)
        # 市场数据应该返回DataFrame
        assert isinstance(market_data, pd.DataFrame)

    def test_validate_order_params(self, trading_engine):
        """测试订单参数验证"""
        valid_params = {
            'symbol': '000001',
                    'quantity': 100,
            'price': 10.5
        }

        result = trading_engine.validate_order_params(valid_params)
        assert isinstance(result, bool)

    def test_get_risk_metrics(self, trading_engine):
        """测试获取风险指标"""
        metrics = trading_engine.get_risk_metrics()
        assert isinstance(metrics, dict)

    def test_check_risk_limits(self, trading_engine):
        """测试检查风险限制"""
        result = trading_engine.check_risk_limits()
        # 方法返回元组格式 (can_trade, reason)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)  # can_trade
        assert isinstance(result[1], str)   # reason

    def test_get_performance_metrics(self, trading_engine):
        """测试获取性能指标"""
        metrics = trading_engine.get_performance_metrics()
        assert isinstance(metrics, dict)

    def test_calculate_position_size(self, trading_engine):
        """测试计算仓位大小"""
        size = trading_engine.calculate_position_size(100000, 0.01, 0.05)
        assert isinstance(size, (int, float))
        assert size >= 0

    def test_is_running(self, trading_engine):
        """测试运行状态检查"""
        running = trading_engine.is_running()
        assert isinstance(running, bool)

    def test_reset_engine(self, trading_engine):
        """测试重置引擎"""
        # 重置应该不抛出异常
        trading_engine.reset_engine()

    def test_generate_signal(self, trading_engine):
        """测试生成信号"""
        signal = trading_engine.generate_signal('000001')
        assert isinstance(signal, dict)

    def test_process_signal(self, trading_engine):
        """测试处理信号"""
        signal = {'symbol': '000001', 'action': 'BUY', 'strength': 0.8}
        result = trading_engine.process_signal(signal)
        assert isinstance(result, dict)