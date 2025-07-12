import pytest
import time
from unittest.mock import MagicMock, patch
from src.trading.live_trading import (
    BrokerGateway,
    RiskEngine,
    TradingMonitor,
    LiveTradingSystem,
    Order,
    OrderType,
    OrderStatus,
    Position,
    RiskRule,
    RiskRuleType
)

@pytest.fixture
def mock_gateway():
    """模拟交易网关"""
    gateway = MagicMock(spec=BrokerGateway)
    gateway.query_positions.return_value = {
        "600000.SH": Position(
            symbol="600000.SH",
            quantity=1000,
            cost_price=10.0,
            market_value=11000,
            pnl=1000,
            last_price=11.0
        )
    }
    gateway.query_account.return_value = {
        "total_assets": 100000,
        "available_cash": 50000
    }
    gateway.orders = {}
    return gateway

@pytest.fixture
def sample_risk_rules():
    """示例风控规则"""
    return [
        RiskRule(RiskRuleType.POSITION_LIMIT, 0.8, "alert"),
        RiskRule(RiskRuleType.LOSS_LIMIT, 0.1, "reduce"),
        RiskRule(RiskRuleType.VOLATILITY_LIMIT, 0.05, "stop")
    ]

def test_broker_gateway_send_order(mock_gateway):
    """测试发送订单"""
    order = Order(
        order_id="test123",
        symbol="600000.SH",
        price=10.5,
        quantity=100,
        order_type=OrderType.LIMIT
    )

    mock_gateway.send_order.return_value = True
    assert mock_gateway.send_order(order) is True
    mock_gateway.send_order.assert_called_once_with(order)

def test_broker_gateway_cancel_order(mock_gateway):
    """测试取消订单"""
    mock_gateway.cancel_order.return_value = True
    assert mock_gateway.cancel_order("test123") is True
    mock_gateway.cancel_order.assert_called_once_with("test123")

def test_risk_engine_check_position_limit(mock_gateway, sample_risk_rules):
    """测试仓位限制风控"""
    engine = RiskEngine(sample_risk_rules)
    positions = {
        "600000.SH": Position(
            symbol="600000.SH",
            quantity=10000,
            cost_price=10.0,
            market_value=110000,  # 超过总资产
            pnl=10000,
            last_price=11.0
        )
    }
    violations = engine.check_risk(
        positions,
        {"total_assets": 100000, "available_cash": 0},
        {"600000.SH": 11.0}
    )
    assert len(violations) == 1
    assert violations[0][0].rule_type == RiskRuleType.POSITION_LIMIT

def test_risk_engine_check_loss_limit(mock_gateway, sample_risk_rules):
    """测试亏损限制风控"""
    engine = RiskEngine(sample_risk_rules)
    positions = {
        "600000.SH": Position(
            symbol="600000.SH",
            quantity=1000,
            cost_price=10.0,
            market_value=9000,  # 亏损10%
            pnl=-1000,
            last_price=9.0
        )
    }
    violations = engine.check_risk(
        positions,
        {"total_assets": 10000, "available_cash": 1000},
        {"600000.SH": 9.0}
    )
    assert len(violations) == 1
    assert violations[0][0].rule_type == RiskRuleType.LOSS_LIMIT

def test_trading_monitor_handle_alert(mock_gateway, sample_risk_rules):
    """测试监控告警处理"""
    monitor = TradingMonitor(mock_gateway, RiskEngine(sample_risk_rules))
    rule = sample_risk_rules[0]  # alert规则
    message = "Position ratio exceeded"

    with patch.object(monitor, '_reduce_positions') as mock_reduce:
        monitor._handle_violation(rule, message)
        assert len(monitor.alert_history) == 1
        assert monitor.alert_history[0]["action"] == "alert"
        mock_reduce.assert_not_called()

def test_trading_monitor_handle_reduce(mock_gateway, sample_risk_rules):
    """测试监控减仓处理"""
    monitor = TradingMonitor(mock_gateway, RiskEngine(sample_risk_rules))
    rule = sample_risk_rules[1]  # reduce规则
    message = "Loss exceeded"

    with patch.object(monitor, '_reduce_positions') as mock_reduce:
        monitor._handle_violation(rule, message)
        mock_reduce.assert_called_once_with(0.5)

def test_trading_monitor_handle_stop(mock_gateway, sample_risk_rules):
    """测试监控停止处理"""
    monitor = TradingMonitor(mock_gateway, RiskEngine(sample_risk_rules))
    rule = sample_risk_rules[2]  # stop规则
    message = "Volatility exceeded"

    with patch.object(monitor, '_cancel_all_orders') as mock_cancel:
        monitor._handle_violation(rule, message)
        mock_cancel.assert_called_once()

def test_live_trading_execute_buy_signal(mock_gateway, sample_risk_rules):
    """测试执行买入信号"""
    strategy = MagicMock()
    strategy.generate_signals.return_value = {"600000.SH": 0.5}  # 50%仓位

    system = LiveTradingSystem(strategy, mock_gateway, sample_risk_rules)

    with patch.object(system, '_get_current_price', return_value=10.0):
        system._execute_signals({"600000.SH": 0.5})
        mock_gateway.send_order.assert_called_once()
        order = mock_gateway.send_order.call_args[0][0]
        assert order.symbol == "600000.SH"
        assert order.quantity > 0
        assert order.order_type == OrderType.LIMIT

def test_live_trading_execute_sell_signal(mock_gateway, sample_risk_rules):
    """测试执行卖出信号"""
    strategy = MagicMock()
    strategy.generate_signals.return_value = {"600000.SH": 0.1}  # 10%仓位

    # 设置当前持仓大于目标仓位
    mock_gateway.query_positions.return_value = {
        "600000.SH": Position(
            symbol="600000.SH",
            quantity=5000,  # 当前持仓
            cost_price=10.0,
            market_value=55000,
            pnl=5000,
            last_price=11.0
        )
    }
    mock_gateway.query_account.return_value = {
        "total_assets": 100000,
        "available_cash": 0
    }

    system = LiveTradingSystem(strategy, mock_gateway, sample_risk_rules)

    with patch.object(system, '_get_current_price', return_value=11.0):
        system._execute_signals({"600000.SH": 0.1})
        mock_gateway.send_order.assert_called_once()
        order = mock_gateway.send_order.call_args[0][0]
        assert order.symbol == "600000.SH"
        assert order.quantity > 0
        assert order.order_type == OrderType.LIMIT

def test_order_status_flow():
    """测试订单状态流转"""
    order = Order(
        order_id="test123",
        symbol="600000.SH",
        price=10.5,
        quantity=100,
        order_type=OrderType.LIMIT
    )
    assert order.status == OrderStatus.PENDING

    order.status = OrderStatus.PARTIAL
    order.filled_quantity = 50
    assert order.status == OrderStatus.PARTIAL

    order.status = OrderStatus.FILLED
    order.filled_quantity = 100
    assert order.status == OrderStatus.FILLED

    order.status = OrderStatus.CANCELLED
    assert order.status == OrderStatus.CANCELLED

    order.status = OrderStatus.REJECTED
    assert order.status == OrderStatus.REJECTED
