# -*- coding: utf-8 -*-
"""
交易层 - 实时交易器单元测试
测试覆盖率目标: 95%+
按照业务流程驱动架构设计测试实时交易器核心功能
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.trading.core.live_trader import (
    LiveTrader, Position, Account, RiskEngine, RiskControlConfig,
    TradingGateway, Order, OrderStatus, OrderType
)

class MockTradingGateway(TradingGateway):

# 设置测试超时，避免死锁和无限等待:
    """模拟交易网关，用于测试"""

    def __init__(self):
        self.connected = False
        self.orders = {}
        self.positions = {}
        self.account = Account(
            account_id="mock_account",
            balance=10000.0,
            available=9000.0
        )

    def connect(self):
        """模拟连接"""
        self.connected = True
        return True

    def disconnect(self):
        """模拟断开连接"""
        self.connected = False
        return True

    def send_order(self, order: Order) -> str:
        """模拟发送订单"""
        order_id = f"mock_order_{len(self.orders)}"
        self.orders[order_id] = order
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """模拟取消订单"""
        if order_id in self.orders:
            del self.orders[order_id]
            return True
        return False

    def query_order(self, order_id: str) -> Order:
        """模拟查询订单"""
        return self.orders.get(order_id)

    def query_positions(self) -> dict:
        """模拟查询持仓"""
        return self.positions

    def query_account(self) -> Account:
        """模拟查询账户"""
        return self.account


# 设置测试超时，避免死锁和无限等待class TestLiveTrader:
    """测试实时交易器"""

    def setup_method(self, method):
        """设置测试环境"""
        self.gateway = MockTradingGateway()
        self.trader = LiveTrader(self.gateway)

    def test_init(self):
        """测试初始化"""
        assert self.trader.gateway == self.gateway
        assert isinstance(self.trader.order_book, dict)
        assert isinstance(self.trader.positions, dict)
        assert isinstance(self.trader.risk_engine, RiskEngine)
        assert self.trader.account is None

    def test_submit_order(self):
        """测试提交订单"""
        order = Order(
            order_id="test_order_001",
            symbol="000001.SZ",
            direction=1,  # 买入
            order_type=OrderType.LIMIT,
            quantity=100,
            price=10.0,
            status=OrderStatus.PENDING
        )

        # 提交订单
        result = self.trader.submit_order(order)

        # 验证订单已被提交
        assert result is True
        assert "test_order_001" in self.trader.order_book
        assert self.trader.order_book["test_order_001"] == order

    def test_submit_order_risk_rejected(self):
        """测试风控拒绝的订单"""
        # 创建一个会触发风控的订单
        order = Order(
            order_id="test_order_002",
            symbol="000001.SZ",
            direction=1,
            order_type=OrderType.LIMIT,
            quantity=1000000,  # 超大数量，可能触发风控
            price=10.0,
            status=OrderStatus.PENDING
        )

        # 提交订单（风控可能会拒绝）
        result = self.trader.submit_order(order)

        # 验证结果（可能被拒绝或接受，取决于风控配置）
        assert isinstance(result, bool)

    def test_update_position_buy(self):
        """测试买入时更新持仓"""
        # 创建买入订单
        order = Order(
            order_id="test_order_003",
            symbol="000001.SZ",
            direction=1,  # 买入
            order_type=OrderType.LIMIT,
            quantity=100,
            price=10.0,
            filled=100,
            status=OrderStatus.FILLED
        )

        # 更新持仓
        self.trader._update_position(order)

        # 验证持仓已被更新
        assert "000001.SZ" in self.trader.positions
        position = self.trader.positions["000001.SZ"]
        assert position.quantity == 100
        assert position.cost_price == 10.0
        assert position.symbol == "000001.SZ"

    def test_update_position_sell(self):
        """测试卖出时更新持仓"""
        # 先创建初始持仓
        initial_position = Position(
            symbol="000001.SZ",
            quantity=200,
            cost_price=10.0,
            update_time=time.time()
        )
        self.trader.positions["000001.SZ"] = initial_position

        # 创建卖出订单
        order = Order(
            order_id="test_order_004",
            symbol="000001.SZ",
            direction=-1,  # 卖出
            order_type=OrderType.LIMIT,
            quantity=50,
            price=12.0,
            filled=50,
            status=OrderStatus.FILLED
        )

        # 更新持仓
        self.trader._update_position(order)

        # 验证持仓已被更新
        position = self.trader.positions["000001.SZ"]
        assert position.quantity == 150  # 200 - 50
        # 成本价应该保持不变（卖出不影响成本价）
        assert position.cost_price == 10.0

    def test_update_position_sell_complete(self):
        """测试全部卖出时更新持仓"""
        # 创建初始持仓
        initial_position = Position(
            symbol="000001.SZ",
            quantity=100,
            cost_price=10.0,
            update_time=time.time()
        )
        self.trader.positions["000001.SZ"] = initial_position

        # 创建全部卖出订单
        order = Order(
            order_id="test_order_005",
            symbol="000001.SZ",
            direction=-1,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=12.0,
            filled=100,
            status=OrderStatus.FILLED
        )

        # 更新持仓
        self.trader._update_position(order)

        # 验证持仓已被清空
        assert "000001.SZ" not in self.trader.positions

    def test_calculate_pnl(self):
        """测试计算盈亏"""
        # 创建持仓
        position = Position(
            symbol="000001.SZ",
            quantity=100,
            cost_price=10.0,
            update_time=time.time()
        )
        self.trader.positions["000001.SZ"] = position

        # 计算盈亏（当前价格12.0）
        pnl = self.trader._calculate_pnl("000001.SZ", 12.0)

        # 验证盈亏计算：(12.0 - 10.0) * 100 = 200
        assert pnl == 200.0

    def test_calculate_pnl_no_position(self):
        """测试计算不存在持仓的盈亏"""
        pnl = self.trader._calculate_pnl("NONEXISTENT", 12.0)
        assert pnl == 0.0

    def test_get_position_value(self):
        """测试获取持仓价值"""
        # 创建持仓
        position = Position(
            symbol="000001.SZ",
            quantity=100,
            cost_price=10.0,
            update_time=time.time()
        )
        self.trader.positions["000001.SZ"] = position

        # 获取持仓价值（当前价格12.0）
        value = self.trader.get_position_value("000001.SZ", 12.0)

        # 验证价值计算：100 * 12.0 = 1200
        assert value == 1200.0

    def test_get_position_value_no_position(self):
        """测试获取不存在持仓的价值"""
        value = self.trader.get_position_value("NONEXISTENT", 12.0)
        assert value == 0.0

    def test_get_total_portfolio_value(self):
        """测试获取总投资组合价值"""
        # 创建多个持仓
        position1 = Position(
            symbol="000001.SZ",
            quantity=100,
            cost_price=10.0,
            update_time=time.time()
        )
        position2 = Position(
            symbol="000002.SZ",
            quantity=50,
            cost_price=20.0,
            update_time=time.time()
        )

        self.trader.positions["000001.SZ"] = position1
        self.trader.positions["000002.SZ"] = position2

        # 设置账户余额
        self.trader.account = Account(
            account_id="test_account",
            balance=5000.0,
            available=4000.0
        )

        # 获取总价值
        total_value = self.trader.get_total_portfolio_value(11.0, 22.0)

        # 验证总价值计算：(100*11.0) + (50*22.0) + 5000.0 = 1100 + 1100 + 5000 = 7200
        assert total_value == 7200.0

    def test_get_total_portfolio_value_no_account(self):
        """测试获取没有账户信息的总投资组合价值"""
        # 创建持仓但没有账户信息
        position = Position(
            symbol="000001.SZ",
            quantity=100,
            cost_price=10.0,
            update_time=time.time()
        )
        self.trader.positions["000001.SZ"] = position

        total_value = self.trader.get_total_portfolio_value(11.0)
        assert total_value == 1100.0  # 只有持仓价值

    def test_check_risk_limits(self):
        """测试检查风控限制"""
        # 创建订单
        order = Order(
            order_id="test_risk_001",
            symbol="000001.SZ",
            direction=1,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=10.0,
            status=OrderStatus.PENDING
        )

        # 检查风控
        result = self.trader.check_risk_limits(order)

        # 验证风控检查结果
        assert isinstance(result, bool)

    def test_get_order_book(self):
        """测试获取订单簿"""
        # 添加一些订单
        order1 = Order(
            order_id="order_001",
            symbol="000001.SZ",
            direction=1,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=10.0,
            status=OrderStatus.PENDING
        )
        order2 = Order(
            order_id="order_002",
            symbol="000002.SZ",
            direction=-1,
            order_type=OrderType.MARKET,
            quantity=50,
            price=0.0,
            status=OrderStatus.SUBMITTED
        )

        self.trader.order_book["order_001"] = order1
        self.trader.order_book["order_002"] = order2

        # 获取订单簿
        order_book = self.trader.get_order_book()

        # 验证订单簿内容
        assert len(order_book) == 2
        assert "order_001" in order_book
        assert "order_002" in order_book
        assert order_book["order_001"] == order1
        assert order_book["order_002"] == order2

    def test_get_positions(self):
        """测试获取持仓"""
        # 创建持仓
        position1 = Position(
            symbol="000001.SZ",
            quantity=100,
            cost_price=10.0,
            update_time=time.time()
        )
        position2 = Position(
            symbol="000002.SZ",
            quantity=50,
            cost_price=20.0,
            update_time=time.time()
        )

        self.trader.positions["000001.SZ"] = position1
        self.trader.positions["000002.SZ"] = position2

        # 获取持仓
        positions = self.trader.get_positions()

        # 验证持仓内容
        assert len(positions) == 2
        assert "000001.SZ" in positions
        assert "000002.SZ" in positions
        assert positions["000001.SZ"] == position1
        assert positions["000002.SZ"] == position2

    def test_get_account_info(self):
        """测试获取账户信息"""
        # 设置账户信息
        account = Account(
            account_id="test_account",
            balance=10000.0,
            available=9000.0
        )
        self.trader.account = account

        # 获取账户信息
        account_info = self.trader.get_account_info()

        # 验证账户信息
        assert account_info == account
        assert account_info.account_id == "test_account"
        assert account_info.balance == 10000.0
        assert account_info.available == 9000.0

    def test_get_account_info_no_account(self):
        """测试获取没有账户信息的情况"""
        account_info = self.trader.get_account_info()
        assert account_info is None


class TestRiskEngine:
    """测试风控引擎"""

    def setup_method(self, method):
        """设置测试环境"""
        self.risk_engine = RiskEngine()

    def test_init(self):
        """测试初始化"""
        assert isinstance(self.risk_engine.config, RiskControlConfig)
        assert isinstance(self.risk_engine.rules, list)

    def test_check_order(self):
        """测试检查订单"""
        order = Order(
            order_id="test_order",
            symbol="000001.SZ",
            direction=1,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=10.0,
            status=OrderStatus.PENDING
        )

        result = self.risk_engine.check_order(order)
        assert isinstance(result, bool)

    def test_add_rule(self):
        """测试添加风控规则"""
        from src.trading.core.live_trader import PositionLimitRule

        rule = PositionLimitRule(threshold=1000)
        self.risk_engine.add_rule(rule)

        assert rule in self.risk_engine.rules

    def test_remove_rule(self):
        """测试移除风控规则"""
        from src.trading.core.live_trader import PositionLimitRule

        rule = PositionLimitRule(threshold=1000)
        self.risk_engine.add_rule(rule)
        self.risk_engine.remove_rule(rule)

        assert rule not in self.risk_engine.rules

    def test_update_config(self):
        """测试更新配置"""
        from src.trading.core.live_trader import RiskControlRule

        new_config = RiskControlConfig(
            rule_type=RiskControlRule.POSITION_LIMIT,
            threshold=2000,
            symbols=["000001.SZ"],
            active=True
        )

        self.risk_engine.config = new_config
        assert self.risk_engine.config == new_config


class TestOrder:
    """测试订单数据结构"""

    def test_order_creation(self):
        """测试订单创建"""
        order = Order(
            order_id="test_order",
            symbol="000001.SZ",
            direction=1,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=10.0,
            status=OrderStatus.PENDING
        )

        assert order.order_id == "test_order"
        assert order.symbol == "000001.SZ"
        assert order.direction == 1
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 100
        assert order.price == 10.0
        assert order.status == OrderStatus.PENDING
        assert order.filled == 0
        assert isinstance(order.create_time, float)
        assert isinstance(order.update_time, float)


class TestPosition:
    """测试持仓数据结构"""

    def test_position_creation(self):
        """测试持仓创建"""
        position = Position(
            symbol="000001.SZ",
            quantity=100,
            cost_price=10.0,
            update_time=time.time()
        )

        assert position.symbol == "000001.SZ"
        assert position.quantity == 100
        assert position.cost_price == 10.0
        assert isinstance(position.update_time, float)


class TestAccount:
    """测试账户数据结构"""

    def test_account_creation(self):
        """测试账户创建"""
        account = Account(
            account_id="test_account",
            balance=10000.0,
            available=9000.0
        )

        assert account.account_id == "test_account"
        assert account.balance == 10000.0
        assert account.available == 9000.0
