import pytest
import time
from unittest.mock import MagicMock, patch
from src.trading.execution_engine import (
    OrderType,
    OrderStatus,
    Order,
    ExecutionEngine,
    AccountManager,
    RealTimeRiskMonitor,
    ExecutionAnalyzer
)

@pytest.fixture
def sample_order():
    """创建测试订单"""
    return Order(
        order_id="ORD123",
        symbol="600519.SH",
        price=1800.0,
        quantity=100,
        order_type=OrderType.LIMIT,
        account="ACC001"
    )

@pytest.fixture
def execution_engine():
    """创建执行引擎实例"""
    return ExecutionEngine(max_retries=3, latency_threshold=0.1)

@pytest.fixture
def account_manager():
    """创建账户管理器实例"""
    return AccountManager()

@pytest.fixture
def risk_monitor():
    """创建风险监控实例"""
    return RealTimeRiskMonitor(max_position=0.2, max_loss=0.1)

@pytest.fixture
def execution_analyzer():
    """创建执行分析实例"""
    return ExecutionAnalyzer()

def test_order_management(execution_engine, sample_order):
    """测试订单管理"""
    # 测试订单提交
    order_id = execution_engine.place_order(sample_order)
    assert order_id == "ORD123"

    # 测试获取订单状态
    order = execution_engine.get_order_status("ORD123")
    assert order.status == OrderStatus.NEW

    # 测试订单取消
    assert execution_engine.cancel_order("ORD123")
    order = execution_engine.get_order_status("ORD123")
    assert order.status == OrderStatus.CANCELLED

    # 测试重复订单
    with pytest.raises(ValueError):
        execution_engine.place_order(sample_order)

def test_routing_strategy(execution_engine, sample_order):
    """测试路由策略"""
    # 添加模拟路由策略
    mock_strategy = MagicMock()
    mock_strategy.execute.return_value = (True, 0.05)
    execution_engine.add_routing_strategy("limit", mock_strategy)

    # 测试路由执行
    success, latency = execution_engine.route_order(sample_order)
    assert success
    assert latency == 0.05
    mock_strategy.execute.assert_called_once_with(sample_order)

def test_account_management(account_manager):
    """测试账户管理"""
    # 测试添加账户
    account_manager.add_account("ACC001", initial_balance=1000000)

    # 测试更新余额
    account_manager.update_balance("ACC001", 500000)
    summary = account_manager.get_account_summary("ACC001")
    assert summary['balance'] == 1500000

    # 测试更新持仓
    account_manager.update_position("ACC001", "600519.SH", 100, 1800.0)
    summary = account_manager.get_account_summary("ACC001")
    assert summary['positions']['600519.SH']['quantity'] == 100
    assert summary['positions']['600519.SH']['cost'] == 180000.0

    # 测试无效账户
    with pytest.raises(ValueError):
        account_manager.update_balance("INVALID", 1000)

def test_risk_monitoring(risk_monitor, sample_order):
    """测试风险监控"""
    # 准备账户数据
    account_data = {
        'positions': {
            '600519.SH': {
                'quantity': 500,
                'cost': 900000.0
            }
        },
        'risk_params': {
            'max_loss': 0.2
        },
        'performance': {
            'pnl': -0.05
        }
    }

    # 测试风险检查
    assert risk_monitor.check_order_risk(sample_order, account_data)

    # 测试仓位超限
    large_order = Order(
        order_id="ORD456",
        symbol="600519.SH",
        price=1800.0,
        quantity=10000,
        order_type=OrderType.LIMIT,
        account="ACC001"
    )
    assert not risk_monitor.check_order_risk(large_order, account_data)

    # 测试市场风险更新
    risk_monitor.update_market_risk("600519.SH", 0.25, 1000000)
    risk_data = risk_monitor.get_market_risk("600519.SH")
    assert risk_data['volatility'] == 0.25

def test_execution_analysis(execution_analyzer, sample_order):
    """测试执行分析"""
    # 测试执行分析
    stats = execution_analyzer.analyze_execution(
        sample_order,
        exec_price=1801.0,
        exec_quantity=100,
        latency=0.05
    )
    assert stats['slippage'] == pytest.approx(0.000555, rel=1e-3)
    assert stats['fill_rate'] == 1.0
    assert stats['latency'] == 0.05

    # 测试获取统计
    symbol_stats = execution_analyzer.get_symbol_stats("600519.SH")
    assert len(symbol_stats) == 1

@pytest.mark.parametrize("order_type,expected_strategy", [
    (OrderType.MARKET, 'market'),
    (OrderType.LIMIT, 'limit'),
    (OrderType.STOP, 'default')
])
def test_routing_strategy_selection(execution_engine, order_type, expected_strategy):
    """测试路由策略选择"""
    # 添加模拟策略
    for strategy in ['market', 'limit', 'default']:
        execution_engine.add_routing_strategy(strategy, MagicMock())

    # 创建测试订单
    order = Order(
        order_id="ORD789",
        symbol="600519.SH",
        price=1800.0,
        quantity=100,
        order_type=order_type,
        account="ACC001"
    )

    # 测试策略选择
    execution_engine.route_order(order)
    execution_engine.routing_strategies[expected_strategy].execute.assert_called_once()

def test_concurrent_order_handling(execution_engine):
    """测试并发订单处理"""
    from threading import Thread

    # 并发提交订单
    def place_order(order_id):
        order = Order(
            order_id=order_id,
            symbol="600519.SH",
            price=1800.0,
            quantity=100,
            order_type=OrderType.LIMIT,
            account="ACC001"
        )
        execution_engine.place_order(order)

    threads = []
    for i in range(10):
        t = Thread(target=place_order, args=(f"ORD{i}",))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # 验证所有订单都已处理
    for i in range(10):
        assert execution_engine.get_order_status(f"ORD{i}") is not None
