import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import numpy as np
from src.trading.order_manager import (
    OrderManager,
    Order,
    OrderType,
    OrderStatus,
    TWAPExecution,
    ExecutionEngine
)

@pytest.fixture
def order_manager():
    """创建订单管理器实例"""
    return OrderManager()

@pytest.fixture
def sample_order():
    """创建测试订单"""
    return Order(
        order_id="ORD00000001",
        symbol="600519.SH",
        order_type=OrderType.LIMIT,
        quantity=100,
        price=1800.0,
        time_in_force="DAY"
    )

@pytest.fixture
def execution_engine(order_manager):
    """创建执行引擎实例"""
    return ExecutionEngine(order_manager)

def test_create_order(order_manager):
    """测试订单创建"""
    order = order_manager.create_order(
        symbol="600519.SH",
        quantity=100,
        order_type=OrderType.LIMIT,
        price=1800.0
    )
    assert order.order_id in order_manager.active_orders
    assert order.status == OrderStatus.NEW

def test_create_market_order(order_manager):
    """测试市价单创建"""
    order = order_manager.create_order(
        symbol="600519.SH",
        quantity=100,
        order_type=OrderType.MARKET
    )
    assert order.order_id in order_manager.active_orders
    assert order.price is None

def test_create_invalid_order(order_manager):
    """测试无效订单创建"""
    with pytest.raises(ValueError):
        order_manager.create_order(
            symbol="600519.SH",
            quantity=0,  # 无效数量
            order_type=OrderType.LIMIT,
            price=1800.0
        )

def test_submit_order(order_manager, sample_order):
    """测试订单提交"""
    assert order_manager.submit_order(sample_order)
    assert sample_order.status == OrderStatus.PENDING_NEW

def test_submit_invalid_order(order_manager):
    """测试无效订单提交"""
    invalid_order = Order(
        order_id="INVALID",
        symbol="600519.SH",
        order_type=OrderType.LIMIT,
        quantity=100,
        price=1800.0
    )
    assert not order_manager.submit_order(invalid_order)

def test_cancel_order(order_manager, sample_order):
    """测试订单取消"""
    order_manager.submit_order(sample_order)
    assert order_manager.cancel_order(sample_order.order_id)
    assert sample_order.status == OrderStatus.CANCELLED

def test_cancel_filled_order(order_manager, sample_order):
    """测试已成交订单取消"""
    order_manager.submit_order(sample_order)
    order_manager.update_order_status(
        sample_order.order_id,
        OrderStatus.FILLED,
        filled_qty=100,
        fill_price=1800.0
    )
    assert not order_manager.cancel_order(sample_order.order_id)

def test_update_order_status(order_manager, sample_order):
    """测试订单状态更新"""
    order_manager.submit_order(sample_order)
    assert order_manager.update_order_status(
        sample_order.order_id,
        OrderStatus.PARTIALLY_FILLED,
        filled_qty=50,
        fill_price=1800.0
    )
    assert sample_order.filled_quantity == 50
    assert sample_order.avg_fill_price == 1800.0

def test_twap_execution_init():
    """测试TWAP执行初始化"""
    parent_order = Order(
        order_id="TWAP001",
        symbol="600519.SH",
        order_type=OrderType.TWAP,
        quantity=1000
    )
    twap = TWAPExecution(parent_order, slices=5)
    assert twap.parent_order.order_id == "TWAP001"
    assert twap.slices == 5

def test_twap_generate_slices(order_manager):
    """测试TWAP切片生成"""
    parent_order = order_manager.create_order(
        symbol="600519.SH",
        quantity=1000,
        order_type=OrderType.TWAP
    )
    twap = TWAPExecution(parent_order, slices=5)
    slices = twap.generate_slice_orders(order_manager)
    assert len(slices) == 5
    assert all(s.parent_id == parent_order.order_id for s in slices)

def test_twap_get_next_slice():
    """测试TWAP获取下一个切片"""
    parent_order = Order(
        order_id="TWAP001",
        symbol="600519.SH",
        order_type=OrderType.TWAP,
        quantity=1000
    )
    twap = TWAPExecution(parent_order, slices=2)
    twap.slice_orders = [
        (datetime.now() - timedelta(minutes=1), MagicMock()),
        (datetime.now() + timedelta(minutes=1), MagicMock())
    ]
    assert twap.get_next_slice(datetime.now()) is not None
    assert twap.get_next_slice(datetime.now()) is None

def test_execution_engine_process(order_manager, execution_engine):
    """测试执行引擎处理"""
    # 添加测试订单到队列
    order = order_manager.create_order(
        symbol="600519.SH",
        quantity=100,
        order_type=OrderType.MARKET
    )
    order_manager.submit_order(order)

    # 处理队列
    execution_engine.process_order_queue()

    # 检查订单状态
    updated_order = order_manager.get_order(order.order_id)
    assert updated_order.status == OrderStatus.FILLED

def test_execution_engine_twap(order_manager, execution_engine):
    """测试执行引擎处理TWAP订单"""
    # 创建TWAP订单
    twap_order = order_manager.create_order(
        symbol="600519.SH",
        quantity=1000,
        order_type=OrderType.TWAP
    )
    order_manager.submit_order(twap_order)

    # 处理队列
    execution_engine.process_order_queue()

    # 检查切片订单是否生成
    assert twap_order.order_id in execution_engine.twap_executions

def test_get_active_orders(order_manager):
    """测试获取活动订单"""
    order1 = order_manager.create_order(
        symbol="600519.SH",
        quantity=100,
        order_type=OrderType.LIMIT,
        price=1800.0,
        strategy_id="strategy1"
    )
    order2 = order_manager.create_order(
        symbol="000858.SH",
        quantity=200,
        order_type=OrderType.LIMIT,
        price=150.0,
        strategy_id="strategy2"
    )

    # 测试按标的筛选
    orders = order_manager.get_active_orders(symbol="600519.SH")
    assert len(orders) == 1
    assert orders[0].order_id == order1.order_id

    # 测试按策略筛选
    orders = order_manager.get_active_orders(strategy_id="strategy2")
    assert len(orders) == 1
    assert orders[0].order_id == order2.order_id

def test_get_order_history(order_manager):
    """测试获取历史订单"""
    order = order_manager.create_order(
        symbol="600519.SH",
        quantity=100,
        order_type=OrderType.MARKET
    )
    order_manager.submit_order(order)
    order_manager.update_order_status(
        order.order_id,
        OrderStatus.FILLED,
        filled_qty=100,
        fill_price=1800.0
    )

    history = order_manager.get_order_history()
    assert len(history) == 1
    assert history[0].order_id == order.order_id
