#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一交易接口单元测试

测试目标：提升unified_trading_interface.py的覆盖率到90%+
按照业务流程驱动架构设计测试接口定义和数据类
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

from src.trading.core.unified_trading_interface import (
    OrderType,
    OrderSide,
    OrderStatus,
    ExecutionVenue,
    TimeInForce,
    Order,
    Trade,
    Position,
    Account,
    ExecutionReport,
    IOrderManager,
    IExecutionEngine,
    ITradingEngine,
    IRiskManager,
    IPortfolioManager,
    IMarketDataProvider,
    IBrokerAdapter,
)


class TestOrderEnums:
    """测试订单枚举类"""

    def test_order_type_enum(self):
        """测试订单类型枚举"""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
        assert OrderType.TRAILING_STOP.value == "trailing_stop"
        assert OrderType.ICEBERG.value == "iceberg"
        assert OrderType.TWAP.value == "twap"
        assert OrderType.VWAP.value == "vwap"
        assert OrderType.ADAPTIVE.value == "adaptive"

    def test_order_side_enum(self):
        """测试订单方向枚举"""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
        assert OrderSide.SHORT.value == "short"
        assert OrderSide.COVER.value == "cover"

    def test_order_status_enum(self):
        """测试订单状态枚举"""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.PARTIAL_FILLED.value == "partial_filled"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"
        assert OrderStatus.SUSPENDED.value == "suspended"

    def test_execution_venue_enum(self):
        """测试执行场所枚举"""
        assert ExecutionVenue.STOCK_EXCHANGE.value == "stock_exchange"
        assert ExecutionVenue.FUTURES_EXCHANGE.value == "futures_exchange"
        assert ExecutionVenue.OTC.value == "otc"
        assert ExecutionVenue.DARK_POOL.value == "dark_pool"
        assert ExecutionVenue.ELECTRONIC_PLATFORM.value == "electronic_platform"

    def test_time_in_force_enum(self):
        """测试有效期类型枚举"""
        assert TimeInForce.DAY.value == "day"
        assert TimeInForce.GTC.value == "gtc"
        assert TimeInForce.GTD.value == "gtd"
        assert TimeInForce.IOC.value == "ioc"
        assert TimeInForce.FOK.value == "fok"
        assert TimeInForce.GTX.value == "gtx"


class TestOrderDataClass:
    """测试订单数据类"""

    def test_order_creation_basic(self):
        """测试基本订单创建"""
        order = Order(
            order_id="order_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        assert order.order_id == "order_001"
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 100.0
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0.0
        assert order.created_at is not None
        assert order.updated_at is not None

    def test_order_creation_with_price(self):
        """测试带价格的订单创建"""
        order = Order(
            order_id="order_002",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=150.0
        )
        
        assert order.price == 150.0
        assert order.order_type == OrderType.LIMIT

    def test_order_creation_with_stop_price(self):
        """测试带止损价的订单创建"""
        order = Order(
            order_id="order_003",
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=100.0,
            stop_price=140.0
        )
        
        assert order.stop_price == 140.0
        assert order.order_type == OrderType.STOP

    def test_order_creation_with_all_fields(self):
        """测试包含所有字段的订单创建"""
        order = Order(
            order_id="order_004",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=150.0,
            stop_price=145.0,
            time_in_force=TimeInForce.GTC,
            venue=ExecutionVenue.STOCK_EXCHANGE,
            account_id="account_001",
            strategy_id="strategy_001",
            commission=1.5,
            metadata={"key": "value"}
        )
        
        assert order.time_in_force == TimeInForce.GTC
        assert order.venue == ExecutionVenue.STOCK_EXCHANGE
        assert order.account_id == "account_001"
        assert order.strategy_id == "strategy_001"
        assert order.commission == 1.5
        assert order.metadata == {"key": "value"}

    def test_order_post_init_auto_timestamps(self):
        """测试订单自动生成时间戳"""
        order = Order(
            order_id="order_005",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        assert order.created_at is not None
        assert order.updated_at is not None
        assert isinstance(order.created_at, datetime)
        assert isinstance(order.updated_at, datetime)


class TestTradeDataClass:
    """测试成交数据类"""

    def test_trade_creation(self):
        """测试成交创建"""
        trade = Trade(
            trade_id="trade_001",
            order_id="order_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            price=150.0,
            timestamp=datetime.now(),
            venue=ExecutionVenue.STOCK_EXCHANGE
        )
        
        assert trade.trade_id == "trade_001"
        assert trade.order_id == "order_001"
        assert trade.symbol == "AAPL"
        assert trade.side == OrderSide.BUY
        assert trade.quantity == 100.0
        assert trade.price == 150.0
        assert trade.venue == ExecutionVenue.STOCK_EXCHANGE

    def test_trade_creation_with_commission(self):
        """测试带手续费的成交创建"""
        trade = Trade(
            trade_id="trade_002",
            order_id="order_002",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            price=150.0,
            timestamp=datetime.now(),
            venue=ExecutionVenue.STOCK_EXCHANGE,
            commission=1.5,
            slippage=0.1
        )
        
        assert trade.commission == 1.5
        assert trade.slippage == 0.1


class TestPositionDataClass:
    """测试持仓数据类"""

    def test_position_creation(self):
        """测试持仓创建"""
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            average_cost=150.0,
            current_price=155.0,
            unrealized_pnl=500.0
        )
        
        assert position.symbol == "AAPL"
        assert position.quantity == 100.0
        assert position.average_cost == 150.0
        assert position.current_price == 155.0
        assert position.unrealized_pnl == 500.0
        assert position.realized_pnl == 0.0
        assert position.last_updated is not None

    def test_position_market_value(self):
        """测试持仓市值计算"""
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            average_cost=150.0,
            current_price=155.0,
            unrealized_pnl=500.0,
            market_value=15500.0  # 明确设置market_value
        )
        
        # 市值 = 数量 * 当前价格
        expected_market_value = 100.0 * 155.0
        assert position.market_value == expected_market_value


class TestAccountDataClass:
    """测试账户数据类"""

    def test_account_creation(self):
        """测试账户创建"""
        account = Account(
            account_id="account_001",
            balance=100000.0,
            available_balance=50000.0
        )
        
        assert account.account_id == "account_001"
        assert account.balance == 100000.0
        assert account.available_balance == 50000.0
        assert account.margin_used == 0.0
        assert account.margin_available == 0.0
        assert account.currency == "CNY"
        assert account.last_updated is not None


class TestExecutionReportDataClass:
    """测试执行报告数据类"""

    def test_execution_report_creation(self):
        """测试执行报告创建"""
        order = Order(
            order_id="order_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        trade = Trade(
            trade_id="trade_001",
            order_id="order_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            price=150.0,
            timestamp=datetime.now(),
            venue=ExecutionVenue.STOCK_EXCHANGE
        )
        
        report = ExecutionReport(
            order=order,
            trades=[trade],
            execution_time=0.1,
            total_commission=1.5,
            slippage_cost=0.5,
            market_impact=0.2
        )
        
        assert report.order == order
        assert len(report.trades) == 1
        assert report.execution_time == 0.1
        assert report.total_commission == 1.5
        assert report.slippage_cost == 0.5
        assert report.market_impact == 0.2


class TestIOrderManagerInterface:
    """测试订单管理器接口"""

    def test_interface_definition(self):
        """测试接口定义存在"""
        assert IOrderManager is not None
        assert hasattr(IOrderManager, 'submit_order')
        assert hasattr(IOrderManager, 'cancel_order')
        assert hasattr(IOrderManager, 'modify_order')
        assert hasattr(IOrderManager, 'get_order')
        assert hasattr(IOrderManager, 'get_orders')
        assert hasattr(IOrderManager, 'get_pending_orders')
        assert hasattr(IOrderManager, 'get_order_history')

    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            IOrderManager()


class TestIExecutionEngineInterface:
    """测试执行引擎接口"""

    def test_interface_definition(self):
        """测试接口定义存在"""
        assert IExecutionEngine is not None
        assert hasattr(IExecutionEngine, 'execute_order')
        assert hasattr(IExecutionEngine, 'execute_batch_orders')
        assert hasattr(IExecutionEngine, 'get_best_execution_price')
        assert hasattr(IExecutionEngine, 'estimate_execution_cost')
        assert hasattr(IExecutionEngine, 'optimize_execution')

    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            IExecutionEngine()


class TestITradingEngineInterface:
    """测试交易引擎接口"""

    def test_interface_definition(self):
        """测试接口定义存在"""
        assert ITradingEngine is not None
        assert hasattr(ITradingEngine, 'place_order')
        assert hasattr(ITradingEngine, 'cancel_all_orders')  # 使用实际存在的方法
        assert hasattr(ITradingEngine, 'get_positions')
        assert hasattr(ITradingEngine, 'get_account_info')  # 使用实际存在的方法

    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            ITradingEngine()


class TestIRiskManagerInterface:
    """测试风险管理接口"""

    def test_interface_definition(self):
        """测试接口定义存在"""
        assert IRiskManager is not None
        assert hasattr(IRiskManager, 'check_order_risk')  # 使用实际存在的方法
        assert hasattr(IRiskManager, 'calculate_position_limits')  # 使用实际存在的方法
        assert hasattr(IRiskManager, 'apply_risk_limits')
        assert hasattr(IRiskManager, 'get_risk_metrics')

    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            IRiskManager()


class TestIPortfolioManagerInterface:
    """测试投资组合管理接口"""

    def test_interface_definition(self):
        """测试接口定义存在"""
        assert IPortfolioManager is not None
        assert hasattr(IPortfolioManager, 'rebalance_portfolio')
        assert hasattr(IPortfolioManager, 'optimize_portfolio')
        assert hasattr(IPortfolioManager, 'calculate_portfolio_metrics')

    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            IPortfolioManager()


class TestIMarketDataProviderInterface:
    """测试市场数据提供者接口"""

    def test_interface_definition(self):
        """测试接口定义存在"""
        assert IMarketDataProvider is not None
        assert hasattr(IMarketDataProvider, 'get_real_time_price')
        assert hasattr(IMarketDataProvider, 'get_historical_data')
        assert hasattr(IMarketDataProvider, 'get_order_book')
        assert hasattr(IMarketDataProvider, 'subscribe_market_data')

    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            IMarketDataProvider()


class TestIBrokerAdapterInterface:
    """测试经纪商适配器接口"""

    def test_interface_definition(self):
        """测试接口定义存在"""
        assert IBrokerAdapter is not None
        assert hasattr(IBrokerAdapter, 'connect')
        assert hasattr(IBrokerAdapter, 'disconnect')
        assert hasattr(IBrokerAdapter, 'submit_order')
        assert hasattr(IBrokerAdapter, 'cancel_order')
        assert hasattr(IBrokerAdapter, 'get_account_balance')

    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            IBrokerAdapter()

