import pytest
import time
import threading
from unittest.mock import MagicMock, patch
from src.trading.real_time_executor import (
    RealTimeExecutor,
    Order,
    OrderType,
    ExecutionStatus,
    ExecutionReport,
    RiskCheckResult
)

@pytest.fixture
def sample_orders():
    """生成测试订单"""
    return [
        Order(
            order_id="ORDER_1",
            symbol="600000",
            quantity=1000,
            order_type=OrderType.MARKET,
            price=None
        ),
        Order(
            order_id="ORDER_2",
            symbol="000001",
            quantity=-500,
            order_type=OrderType.LIMIT,
            price=50.0
        )
    ]

@pytest.fixture
def executor():
    """初始化执行器"""
    exe = RealTimeExecutor(max_order_rate=1000)  # 高频测试
    yield exe
    exe.stop()

def test_order_submission(executor, sample_orders):
    """测试订单提交"""
    order = sample_orders[0]
    order_id = executor.submit_order(order)

    assert order_id == order.order_id
    assert executor.order_queue.qsize() == 1

def test_order_execution(executor, sample_orders):
    """测试订单执行"""
    order = sample_orders[0]
    executor.submit_order(order)

    # 等待执行
    time.sleep(0.1)

    report = executor.get_execution_report(order.order_id)
    assert report is not None
    assert report.status == ExecutionStatus.FILLED
    assert report.filled_quantity == order.quantity

def test_position_update(executor, sample_orders):
    """测试持仓更新"""
    buy_order = sample_orders[0]
    sell_order = sample_orders[1]

    executor.submit_order(buy_order)
    executor.submit_order(sell_order)

    # 等待执行
    time.sleep(0.2)

    # 检查持仓
    assert executor.get_position(buy_order.symbol) == buy_order.quantity
    assert executor.get_position(sell_order.symbol) == sell_order.quantity

def test_risk_check_rejection(executor, sample_orders):
    """测试风控拒绝"""
    # 模拟风控失败
    with patch.object(executor, '_check_order_risk',
                    return_value=RiskCheckResult.REJECTED):
        order = sample_orders[0]
        executor.submit_order(order)

        # 等待处理
        time.sleep(0.1)

        report = executor.get_execution_report(order.order_id)
        assert report.status == ExecutionStatus.REJECTED

def test_order_rate_limit(executor):
    """测试订单速率限制"""
    start_time = time.time()
    count = 0

    # 提交大量订单
    for i in range(200):
        try:
            executor.submit_order(Order(
                order_id=f"STRESS_{i}",
                symbol="TEST",
                quantity=100,
                order_type=OrderType.MARKET
            ))
            count += 1
        except:
            break

    duration = time.time() - start_time
    assert duration >= 0.1  # 100订单按1000/s速率至少需要0.1秒
    assert count == 200  # 全部成功提交

def test_thread_safety(executor):
    """测试多线程安全"""
    results = []

    def worker(order_id):
        try:
            order = Order(
                order_id=order_id,
                symbol="THREAD_TEST",
                quantity=100,
                order_type=OrderType.MARKET
            )
            executor.submit_order(order)
            results.append(True)
        except:
            results.append(False)

    # 创建多个线程同时提交订单
    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(f"THREAD_{i}",))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert all(results)  # 所有线程都成功
    assert executor.order_queue.qsize() == 10

def test_execution_report_retrieval(executor, sample_orders):
    """测试执行报告获取"""
    order = sample_orders[0]
    executor.submit_order(order)

    # 等待执行
    time.sleep(0.1)

    # 测试获取不存在的订单
    assert executor.get_execution_report("NON_EXISTENT") is None

    # 测试获取存在的订单
    report = executor.get_execution_report(order.order_id)
    assert report is not None
    assert report.order_id == order.order_id

def test_stop_behavior(executor, sample_orders):
    """测试停止行为"""
    executor.submit_order(sample_orders[0])
    executor.stop()

    # 尝试提交新订单应该失败
    with pytest.raises(RuntimeError):
        executor.submit_order(sample_orders[1])

    # 检查线程是否停止
    assert not executor.active
    for thread in executor.threads:
        assert not thread.is_alive()

def test_market_data_integration():
    """测试行情数据集成"""
    # TODO: 实现行情数据集成测试
    pass

def test_risk_control_scenarios():
    """测试风控场景"""
    # TODO: 实现各种风控场景测试
    pass
