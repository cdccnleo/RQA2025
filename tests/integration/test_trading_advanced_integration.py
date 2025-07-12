import pytest
from unittest.mock import MagicMock
from queue import Queue
from datetime import datetime
import threading
import time

# 复用基础Mock类
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

class MockAccount:
    def __init__(self, account_id, balance, available):
        self.account_id = account_id
        self.balance = balance
        self.available = available
        self.margin = 0.0
        self.update_time = datetime.now().timestamp()

# 多账户风控
class MultiAccountRiskEngine:
    def __init__(self, account_limits):
        self.account_limits = account_limits  # dict: account_id -> limit
        self.violations = Queue()
    def check_order(self, order, account_id):
        limit = self.account_limits.get(account_id, 100)
        if order.quantity > limit:
            self.violations.put(f"账户{account_id}超出限额: {order.quantity}")
            return False
        return True
    def get_violations(self):
        v = []
        while not self.violations.empty():
            v.append(self.violations.get())
        return v

# 支持异步撮合的MockGateway
class AsyncMockGateway:
    def __init__(self, fail_on=None):
        self.orders = []
        self.fail_on = fail_on or set()
        self.lock = threading.Lock()
        self.status_map = {}
    def send_order(self, order):
        def match():
            time.sleep(0.1)
            with self.lock:
                if order.symbol in self.fail_on:
                    order.status = 'FAILED'
                else:
                    order.status = 'FILLED'
                self.status_map[order.symbol] = order.status
        t = threading.Thread(target=match)
        t.start()
        self.orders.append(order)
        return True
    def query_order_status(self, symbol):
        return self.status_map.get(symbol, 'PENDING')

# 实盘交易主流程（多账户+异步撮合）
class AdvancedLiveTrader:
    def __init__(self, gateway, risk_engine):
        self.gateway = gateway
        self.risk_engine = risk_engine
        self.order_book = {}
        self.account_orders = {}
    def submit_order(self, order, account_id):
        if self.risk_engine.check_order(order, account_id):
            self.gateway.send_order(order)
            self.order_book[(order.symbol, account_id)] = order
            self.account_orders.setdefault(account_id, []).append(order)
            return True
        return False
    def get_violations(self):
        return self.risk_engine.get_violations()
    def get_orders(self, account_id=None):
        if account_id:
            return self.account_orders.get(account_id, [])
        return list(self.order_book.values())
    def query_order_status(self, symbol):
        return self.gateway.query_order_status(symbol)

@pytest.fixture
def async_gateway():
    return AsyncMockGateway(fail_on={'000005.SZ'})

@pytest.fixture
def multi_account_risk():
    return MultiAccountRiskEngine({'A1': 100, 'A2': 50})

@pytest.fixture
def adv_trader(async_gateway, multi_account_risk):
    return AdvancedLiveTrader(async_gateway, multi_account_risk)

# 新增：账户余额变动Mock
class AccountManager:
    def __init__(self):
        self.balances = {'A1': 100000, 'A2': 50000}
        self.lock = threading.Lock()
    def update_on_fill(self, account_id, order):
        with self.lock:
            self.balances[account_id] -= order.price * order.quantity
    def get_balance(self, account_id):
        return self.balances.get(account_id, 0)
    def adjust_limit(self, risk_engine, account_id, new_limit):
        risk_engine.account_limits[account_id] = new_limit

@pytest.fixture
def account_manager():
    return AccountManager()

def test_multi_account_risk_and_order_flow(adv_trader):
    """多账户风控与下单流程"""
    order1 = MockOrder('000001.SZ', 10.0, 80, 1)
    order2 = MockOrder('000002.SZ', 10.0, 60, 1)
    order3 = MockOrder('000003.SZ', 10.0, 120, 1)  # 超A1限额
    order4 = MockOrder('000004.SZ', 10.0, 40, 1)   # 超A2限额
    assert adv_trader.submit_order(order1, 'A1') is True
    assert adv_trader.submit_order(order2, 'A2') is False
    assert adv_trader.submit_order(order3, 'A1') is False
    assert adv_trader.submit_order(order4, 'A2') is True
    violations = adv_trader.get_violations()
    assert any('A1' in v for v in violations)
    assert any('A2' in v for v in violations)
    assert len(adv_trader.get_orders('A1')) == 1
    assert len(adv_trader.get_orders('A2')) == 1

def test_async_matching_and_status_query(adv_trader):
    """异步撮合与状态回查"""
    order1 = MockOrder('000005.SZ', 10.0, 30, 1)  # 设置为撮合失败
    order2 = MockOrder('000006.SZ', 10.0, 30, 1)
    adv_trader.submit_order(order1, 'A1')
    adv_trader.submit_order(order2, 'A1')
    # 初始状态应为PENDING
    assert adv_trader.query_order_status('000005.SZ') == 'PENDING'
    assert adv_trader.query_order_status('000006.SZ') == 'PENDING'
    # 等待异步撮合
    time.sleep(0.2)
    assert adv_trader.query_order_status('000005.SZ') == 'FAILED'
    assert adv_trader.query_order_status('000006.SZ') == 'FILLED' 

# 多线程高并发下单
def test_concurrent_order_submission_and_risk(adv_trader):
    """多线程高并发下单与风控"""
    results = []
    def submit():
        for i in range(20):
            order = MockOrder(f'0001{i:02d}.SZ', 10.0, 10, 1)
            results.append(adv_trader.submit_order(order, 'A1'))
    threads = [threading.Thread(target=submit) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 总共应有100个订单，全部通过风控
    assert sum(results) == 100
    assert len(adv_trader.get_orders('A1')) == 100

# 撮合超时与回退补偿
class TimeoutMockGateway(AsyncMockGateway):
    def send_order(self, order):
        def match():
            # 故意不设置status，模拟超时
            time.sleep(0.3)
        t = threading.Thread(target=match)
        t.start()
        self.orders.append(order)
        return True

def test_order_matching_timeout_and_compensation(multi_account_risk):
    """撮合超时与回退补偿"""
    gateway = TimeoutMockGateway()
    trader = AdvancedLiveTrader(gateway, multi_account_risk)
    order = MockOrder('000099.SZ', 10.0, 10, 1)
    trader.submit_order(order, 'A1')
    # 0.1秒后仍为PENDING
    time.sleep(0.1)
    assert trader.query_order_status('000099.SZ') == 'PENDING'
    # 0.4秒后仍为PENDING，模拟补偿逻辑
    time.sleep(0.4)
    # 假设补偿逻辑：超时未成交则回退为FAILED
    if trader.query_order_status('000099.SZ') == 'PENDING':
        gateway.status_map['000099.SZ'] = 'FAILED'
    assert trader.query_order_status('000099.SZ') == 'FAILED'

# 账户余额变动
def test_account_balance_update_on_fill(adv_trader, account_manager):
    """订单成交后账户余额变动"""
    order = MockOrder('000123.SZ', 100.0, 10, 1)
    adv_trader.submit_order(order, 'A1')
    time.sleep(0.2)
    if adv_trader.query_order_status('000123.SZ') == 'FILLED':
        account_manager.update_on_fill('A1', order)
    assert account_manager.get_balance('A1') == 100000 - 1000

# 风控动态调整
def test_dynamic_risk_limit_adjustment(adv_trader, multi_account_risk, account_manager):
    """风控限额动态调整"""
    # 初始限额A2=50
    order1 = MockOrder('000200.SZ', 10.0, 60, 1)
    assert adv_trader.submit_order(order1, 'A2') is False
    # 动态调整限额
    account_manager.adjust_limit(multi_account_risk, 'A2', 100)
    order2 = MockOrder('000201.SZ', 10.0, 60, 1)
    assert adv_trader.submit_order(order2, 'A2') is True 