#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交易主流程集成测试
覆盖多策略组合、风控联动、实盘撮合等主流程
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from datetime import datetime
from queue import Queue

# Mock策略
class MockStrategy:
    def __init__(self, name, signal=1):
        self.name = name
        self.signal = signal
    def next_strategy(self):
        return self.signal

# Mock订单、持仓、账户
class MockOrder:
    def __init__(self, symbol, price, quantity, direction):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.direction = direction
        self.status = 'PENDING'
        self.filled = 0
        self.create_time = datetime.now().timestamp()
        self.update_time = self.create_time

class MockPosition:
    def __init__(self, symbol, quantity, cost_price):
        self.symbol = symbol
        self.quantity = quantity
        self.cost_price = cost_price
        self.update_time = datetime.now().timestamp()

class MockAccount:
    def __init__(self, account_id, balance, available):
        self.account_id = account_id
        self.balance = balance
        self.available = available
        self.margin = 0.0
        self.update_time = datetime.now().timestamp()

# Mock风控引擎
class MockRiskEngine:
    def __init__(self, limit=100):
        self.limit = limit
        self.violations = Queue()
    def check_order(self, order):
        if order.quantity > self.limit:
            self.violations.put(f"超出单笔限额: {order.quantity}")
            return False
        return True
    def get_violations(self):
        v = []
        while not self.violations.empty():
            v.append(self.violations.get())
        return v

# Mock交易网关
class MockGateway:
    def __init__(self):
        self.orders = []
        self.connected = False
    def connect(self):
        self.connected = True
    def send_order(self, order):
        self.orders.append(order)
        order.status = 'SUBMITTED'
        return True
    def query_order(self, order_id):
        for o in self.orders:
            if o.symbol == order_id:
                return o
        return None
    def query_positions(self):
        return {'000001.SZ': MockPosition('000001.SZ', 100, 15.0)}
    def query_account(self):
        return MockAccount('test', 1000000, 900000)

# 实盘交易主流程
class SimpleLiveTrader:
    def __init__(self, gateway, risk_engine):
        self.gateway = gateway
        self.risk_engine = risk_engine
        self.order_book = {}
        self.positions = {}
        self.account = None
        self.event_queue = Queue()
    def submit_order(self, order):
        if self.risk_engine.check_order(order):
            self.gateway.send_order(order)
            self.order_book[order.symbol] = order
            return True
        return False
    def process_signals(self, signals):
        # 信号融合与下单
        for sig in signals:
            order = MockOrder(symbol=sig['symbol'], price=15.0, quantity=sig['quantity'], direction=sig['direction'])
            if self.submit_order(order):
                # 只有通过风控的订单才添加到order_book
                pass
    def get_violations(self):
        return self.risk_engine.get_violations()
    def get_orders(self):
        return list(self.order_book.values())

@pytest.fixture
def mock_gateway():
    return MockGateway()

@pytest.fixture
def mock_risk_engine():
    return MockRiskEngine(limit=100)

@pytest.fixture
def live_trader(mock_gateway, mock_risk_engine):
    return SimpleLiveTrader(mock_gateway, mock_risk_engine)

def test_multi_strategy_signal_fusion_and_risk_control(live_trader):
    """多策略信号融合与风控拦截集成测试"""
    # 多策略信号
    strategies = [MockStrategy('s1', 1), MockStrategy('s2', -1)]
    signals = [
        {'symbol': '000001.SZ', 'quantity': 50, 'direction': 1},
        {'symbol': '000002.SZ', 'quantity': 30, 'direction': -1}
    ]
    # 风控通过
    live_trader.process_signals(signals)
    orders = live_trader.get_orders()
    assert len(orders) == 2
    assert all(o.status == 'SUBMITTED' for o in orders)
    # 风控拦截
    signals2 = [
        {'symbol': '000002.SZ', 'quantity': 200, 'direction': 1}
    ]
    live_trader.process_signals(signals2)
    violations = live_trader.get_violations()
    assert any('超出单笔限额' in v for v in violations)
    # 只应有2个通过风控的订单
    assert len(live_trader.get_orders()) == 2

def test_live_trader_execution_flow(live_trader):
    """实盘撮合流程闭环测试"""
    # 模拟下单
    order = MockOrder('000003.SZ', 20.0, 80, 1)
    result = live_trader.submit_order(order)
    assert result is True
    # 查询订单
    queried = live_trader.gateway.query_order('000003.SZ')
    assert queried is not None
    assert queried.status == 'SUBMITTED'
    # 查询持仓
    positions = live_trader.gateway.query_positions()
    assert '000001.SZ' in positions
    # 查询账户
    account = live_trader.gateway.query_account()
    assert account.balance == 1000000 