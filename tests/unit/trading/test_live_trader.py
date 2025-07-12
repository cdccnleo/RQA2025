import pytest
import asyncio
from unittest.mock import MagicMock, patch
from src.trading.live_trader import (
    LiveTrader,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Account,
    RiskControlRule,
    RiskControlConfig,
    TradingGateway,
    CTPGateway
)

@pytest.fixture
def mock_gateway():
    """模拟交易网关"""
    gateway = MagicMock(spec=CTPGateway)
    gateway.query_account.return_value = Account(
        account_id="test_account",
        balance=1_000_000,
        available=800_000
    )
    gateway.query_positions.return_value = {
        "600000.SH": Position(
            symbol="600000.SH",
            quantity=10000,
            cost_price=10.5
        )
    }
    return gateway

@pytest.fixture
def sample_order():
    """测试订单"""
    return Order(
        order_id="test_order",
        symbol="600000.SH",
        price=10.8,
        quantity=5000,
        direction=1,
        order_type=OrderType.LIMIT
    )

@pytest.fixture
def live_trader(mock_gateway):
    """初始化交易引擎"""
    trader = LiveTrader(gateway=mock_gateway)

    # 添加基本风控规则
    trader.risk_engine.add_rule(RiskControlConfig(
        rule_type=RiskControlRule.POSITION_LIMIT,
        threshold=0.5,  # 仓位不超过50%
        symbols=["600000.SH"]
    ))

    return trader

@pytest.mark.asyncio
async def test_trader_startup(live_trader, mock_gateway):
    """测试交易引擎启动"""
    with patch.object(live_trader, '_process_events') as mock_process:
        mock_process.side_effect = asyncio.CancelledError()

        try:
            await live_trader.run()
        except asyncio.CancelledError:
            pass

        assert mock_gateway.connect.called

def test_order_submission(live_trader, sample_order):
    """测试订单提交"""
    # 测试合规订单
    assert live_trader.submit_order(sample_order)
    assert sample_order.order_id in live_trader.order_book

    # 测试违规订单 (超过仓位限制)
    big_order = Order(
        order_id="big_order",
        symbol="600000.SH",
        price=10.8,
        quantity=50000,  # 过大数量
        direction=1,
        order_type=OrderType.LIMIT
    )
    assert not live_trader.submit_order(big_order)

def test_position_update(live_trader):
    """测试持仓更新逻辑"""
    # 初始持仓
    live_trader.positions = {
        "600000.SH": Position(
            symbol="600000.SH",
            quantity=10000,
            cost_price=10.5
        )
    }

    # 买入订单
    buy_order = Order(
        order_id="buy_order",
        symbol="600000.SH",
        price=10.8,
        quantity=5000,
        direction=1,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        filled=5000
    )

    live_trader._update_position(buy_order)
    assert live_trader.positions["600000.SH"].quantity == 15000
    assert live_trader.positions["600000.SH"].cost_price == pytest.approx(10.6)

    # 卖出订单
    sell_order = Order(
        order_id="sell_order",
        symbol="600000.SH",
        price=11.0,
        quantity=8000,
        direction=-1,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        filled=8000
    )

    live_trader._update_position(sell_order)
    assert live_trader.positions["600000.SH"].quantity == 7000

    # 全部卖出
    final_sell = Order(
        order_id="final_sell",
        symbol="600000.SH",
        price=11.2,
        quantity=7000,
        direction=-1,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        filled=7000
    )

    live_trader._update_position(final_sell)
    assert "600000.SH" not in live_trader.positions

def test_risk_control(live_trader, sample_order):
    """测试风控检查"""
    # 添加交易时间限制规则
    live_trader.risk_engine.add_rule(RiskControlConfig(
        rule_type=RiskControlRule.TRADING_HOURS,
        threshold=1,  # 假设1表示交易时间
        active=True
    ))

    # 模拟非交易时间
    with patch('time.localtime') as mock_time:
        mock_time.return_value.tm_hour = 3  # 凌晨3点
        assert not live_trader.risk_engine.check_order(sample_order)

    # 模拟交易时间
    with patch('time.localtime') as mock_time:
        mock_time.return_value.tm_hour = 10  # 上午10点
        assert live_trader.risk_engine.check_order(sample_order)

@pytest.mark.asyncio
async def test_event_handling(live_trader):
    """测试事件处理"""
    test_order = Order(
        order_id="test_event_order",
        symbol="600000.SH",
        price=10.5,
        quantity=1000,
        direction=1,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        filled=1000
    )

    # 放入订单事件
    live_trader.event_queue.put({
        'type': 'order',
        'data': test_order
    })

    # 处理事件
    await live_trader._process_events()

    # 检查持仓是否更新
    assert "600000.SH" in live_trader.positions
    assert live_trader.positions["600000.SH"].quantity == 1000

def test_gateway_interface(mock_gateway, sample_order):
    """测试交易网关接口"""
    # 测试连接
    mock_gateway.connect()
    assert mock_gateway.connect.called

    # 测试订单发送
    mock_gateway.send_order.return_value = "test_order_id"
    order_id = mock_gateway.send_order(sample_order)
    assert order_id == "test_order_id"

    # 测试订单查询
    sample_order.status = OrderStatus.FILLED
    mock_gateway.query_order.return_value = sample_order
    order = mock_gateway.query_order("test_order_id")
    assert order.status == OrderStatus.FILLED

@pytest.mark.asyncio
async def test_risk_monitoring(live_trader):
    """测试风险监控循环"""
    # 添加亏损限额规则
    live_trader.risk_engine.add_rule(RiskControlConfig(
        rule_type=RiskControlRule.LOSS_LIMIT,
        threshold=0.1  # 最大亏损10%
    ))

    # 模拟账户亏损
    live_trader.account = Account(
        account_id="test_account",
        balance=900_000,  # 初始1M, 亏损10%
        available=700_000
    )

    with patch.object(live_trader.risk_engine, 'get_violations') as mock_violations:
        mock_violations.return_value = ["Exceeded loss limit"]
        await live_trader._monitor_risk()
        assert mock_violations.called

@pytest.mark.asyncio
async def test_status_sync(live_trader, mock_gateway):
    """测试状态同步"""
    await live_trader._sync_status()
    assert mock_gateway.query_account.called
    assert mock_gateway.query_positions.called
    assert live_trader.account is not None
    assert "600000.SH" in live_trader.positions
